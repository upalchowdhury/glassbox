"""Learning from preferences: a reward model, and DPO.

These are two different answers to the same question - "we have pairs of
responses a person ranked, now what?" - and the course must not blur them:

* A REWARD MODEL turns a prompt/response pair into a scalar. It is a learned
  approximation of the ranking, trained with the Bradley-Terry pairwise loss.
  It is not a ground-truth judge, and this module measures exactly how often it
  disagrees with the rankings it was trained on.
* DPO optimises the policy directly against the pairs, using a frozen reference
  copy instead of a separately trained reward model. Standard offline DPO never
  samples new completions during the update.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data
from .model import TinyTransformer
from .tokenizer import CharTokenizer


# --------------------------------------------------------------------------
# Differentiable sequence log-probabilities
# --------------------------------------------------------------------------


def batch_sequence_logprobs(
    model: TinyTransformer,
    tok: CharTokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    include_eos: bool = True,
) -> torch.Tensor:
    """Summed log p(response | prompt) for each pair, keeping the graph.

    Only response tokens are scored. Padding is masked out, so a longer batch
    mate can never change another row's number.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must be the same length")

    rows: List[Tuple[List[int], List[int]]] = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = tok.encode(data.format_prompt(prompt), bos=True)
        response_ids = tok.encode(response) + ([tok.eos_id] if include_eos else [])
        ids = prompt_ids + response_ids
        targets = [-100] * len(prompt_ids) + response_ids
        rows.append((ids, targets))

    width = max(len(ids) for ids, _ in rows) - 1
    x = torch.full((len(rows), width), tok.pad_id, dtype=torch.long)
    y = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (ids, targets) in enumerate(rows):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1], dtype=torch.long)
        y[i, :n] = torch.tensor(targets[1:], dtype=torch.long)

    logits = model(x)["logits"]
    logprobs = F.log_softmax(logits, dim=-1)
    mask = y != -100
    gathered = logprobs.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


# --------------------------------------------------------------------------
# Reward model
# --------------------------------------------------------------------------


class RewardModel(nn.Module):
    """A transformer body plus a scalar head read at the final response token.

    The head is a single ``[width, 1]`` projection so a learner can look at all
    of it. The score is a learned number, not a probability and not a truth.
    """

    def __init__(self, body: TinyTransformer):
        super().__init__()
        self.body = body
        self.head = nn.Linear(body.cfg.width, 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def score(
        self, tok: CharTokenizer, prompts: Sequence[str], responses: Sequence[str]
    ) -> torch.Tensor:
        rows = [
            tok.encode(data.format_prompt(p), bos=True) + tok.encode(r) + [tok.eos_id]
            for p, r in zip(prompts, responses)
        ]
        width = max(len(r) for r in rows)
        x = torch.full((len(rows), width), tok.pad_id, dtype=torch.long)
        last = []
        for i, row in enumerate(rows):
            x[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            last.append(len(row) - 1)
        # The reward head reads REPRESENTATIONS, not vocabulary logits, so this
        # runs the body and stops at the final normalised hidden state.
        hidden_states = _final_hidden(self.body, x)
        picked = hidden_states[torch.arange(len(rows)), torch.tensor(last)]
        return self.head(picked).squeeze(-1)


def _final_hidden(model: TinyTransformer, ids: torch.Tensor) -> torch.Tensor:
    """The post-final-norm hidden state for every position."""
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device)
    x = model.token_embedding(ids) + model.position_embedding(pos).unsqueeze(0)
    for block in model.blocks:
        x, _ = block(x)
    return model.final_norm(x)


@dataclass
class RewardConfig:
    steps: int = 300
    learning_rate: float = 5e-3
    weight_decay: float = 0.0
    seed: int = 7

    def to_dict(self) -> Dict[str, object]:
        return {
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "loss": "-log sigmoid(score_chosen - score_rejected)  (Bradley-Terry)",
            "head": "single [width, 1] projection at the final response token",
        }


def train_reward_model(
    base: TinyTransformer,
    tok: CharTokenizer,
    pairs: Sequence[data.PreferencePair],
    cfg: RewardConfig,
) -> Dict[str, object]:
    """Train a reward model on ranked pairs, starting from a copy of the policy."""
    torch.manual_seed(cfg.seed)
    rm = RewardModel(copy.deepcopy(base))
    opt = torch.optim.AdamW(
        rm.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    prompts = [p.prompt for p in pairs]
    chosen = [p.chosen for p in pairs]
    rejected = [p.rejected for p in pairs]

    history: List[Dict[str, float]] = []
    rm.train()
    for step in range(cfg.steps + 1):
        sc = rm.score(tok, prompts, chosen)
        sr = rm.score(tok, prompts, rejected)
        loss = -F.logsigmoid(sc - sr).mean()
        accuracy = float((sc > sr).float().mean())
        if step % max(1, cfg.steps // 10) == 0 or step == cfg.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "ranking_accuracy": accuracy,
                    "mean_margin": float((sc - sr).mean()),
                }
            )
        if step == cfg.steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    rm.eval()
    with torch.no_grad():
        sc = rm.score(tok, prompts, chosen)
        sr = rm.score(tok, prompts, rejected)
    detail = [
        {
            "prompt": p.prompt,
            "chosen": p.chosen,
            "rejected": p.rejected,
            "reason": p.reason,
            "rejected_is_factual": p.rejected_is_factual,
            "score_chosen": float(a),
            "score_rejected": float(b),
            "margin": float(a - b),
            "ranked_correctly": bool(a > b),
        }
        for p, a, b in zip(pairs, sc, sr)
    ]
    return {
        "config": cfg.to_dict(),
        "history": history,
        "pairs": detail,
        "final_accuracy": float((sc > sr).float().mean()),
        "num_pairs": len(pairs),
        "head_weights": rm.head.weight.detach().tolist(),
        "model": rm,
    }


# --------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------


@dataclass
class DPOConfig:
    steps: int = 200
    learning_rate: float = 1e-3
    beta: float = 0.1
    seed: int = 5

    def to_dict(self) -> Dict[str, object]:
        return {
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "beta": self.beta,
            "seed": self.seed,
            "loss": (
                "-log sigmoid(beta * [ (logp_policy(chosen) - logp_ref(chosen)) "
                "- (logp_policy(rejected) - logp_ref(rejected)) ])"
            ),
            "reference": "frozen copy of the policy at DPO start",
            "samples_during_training": False,
        }


def dpo_terms(
    policy: TinyTransformer,
    reference: TinyTransformer,
    tok: CharTokenizer,
    pairs: Sequence[data.PreferencePair],
    beta: float,
) -> Dict[str, torch.Tensor]:
    """Every quantity in the DPO loss, kept separate so each can be displayed."""
    prompts = [p.prompt for p in pairs]
    chosen = [p.chosen for p in pairs]
    rejected = [p.rejected for p in pairs]

    pol_c = batch_sequence_logprobs(policy, tok, prompts, chosen)
    pol_r = batch_sequence_logprobs(policy, tok, prompts, rejected)
    with torch.no_grad():
        ref_c = batch_sequence_logprobs(reference, tok, prompts, chosen)
        ref_r = batch_sequence_logprobs(reference, tok, prompts, rejected)

    # The "implied reward" of DPO: beta times the log-ratio against the
    # reference. It is a by-product of the objective, not a trained reward model.
    reward_c = beta * (pol_c - ref_c)
    reward_r = beta * (pol_r - ref_r)
    margin = reward_c - reward_r
    loss = -F.logsigmoid(margin)
    return {
        "policy_chosen": pol_c,
        "policy_rejected": pol_r,
        "reference_chosen": ref_c,
        "reference_rejected": ref_r,
        "implied_reward_chosen": reward_c,
        "implied_reward_rejected": reward_r,
        "margin": margin,
        "loss_per_pair": loss,
        "loss": loss.mean(),
    }


def train_dpo(
    policy: TinyTransformer,
    tok: CharTokenizer,
    pairs: Sequence[data.PreferencePair],
    cfg: DPOConfig,
) -> Dict[str, object]:
    """Optimise the policy on preference pairs against a frozen reference."""
    torch.manual_seed(cfg.seed)
    reference = copy.deepcopy(policy)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    history: List[Dict[str, float]] = []

    policy.train()
    for step in range(cfg.steps + 1):
        terms = dpo_terms(policy, reference, tok, pairs, cfg.beta)
        if step % max(1, cfg.steps // 10) == 0 or step == cfg.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(terms["loss"]),
                    "margin": float(terms["margin"].mean()),
                    "accuracy": float((terms["margin"] > 0).float().mean()),
                    "policy_chosen_logprob": float(terms["policy_chosen"].mean()),
                    "policy_rejected_logprob": float(terms["policy_rejected"].mean()),
                }
            )
        if step == cfg.steps:
            break
        opt.zero_grad(set_to_none=True)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    policy.eval()
    with torch.no_grad():
        final = dpo_terms(policy, reference, tok, pairs, cfg.beta)
    detail = [
        {
            "prompt": p.prompt,
            "chosen": p.chosen,
            "rejected": p.rejected,
            "reason": p.reason,
            "rejected_is_factual": p.rejected_is_factual,
            "policy_chosen": float(final["policy_chosen"][i]),
            "policy_rejected": float(final["policy_rejected"][i]),
            "reference_chosen": float(final["reference_chosen"][i]),
            "reference_rejected": float(final["reference_rejected"][i]),
            "implied_reward_chosen": float(final["implied_reward_chosen"][i]),
            "implied_reward_rejected": float(final["implied_reward_rejected"][i]),
            "margin": float(final["margin"][i]),
            "loss": float(final["loss_per_pair"][i]),
        }
        for i, p in enumerate(pairs)
    ]
    return {
        "config": cfg.to_dict(),
        "history": history,
        "pairs": detail,
        "final_accuracy": float((final["margin"] > 0).float().mean()),
        "reference": reference,
    }
