"""Reinforcement learning from verifiable rewards.

Two things this module keeps strictly apart, because conflating them is the
most common misunderstanding in the whole area:

* RLVR says where the reward COMES FROM - a checkable result.
* GRPO is an OPTIMISATION method - it turns a group of rewards into
  advantages and takes a policy-gradient step.

They are not synonyms, and neither is a source of the other. The advantage
arithmetic alone is not GRPO either: a policy update needs sampled tokens,
completion masks, old and current log-probabilities, a ratio, an aggregation
rule, and whatever clipping or reference penalty the variant specifies. All of
those are computed here rather than described.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import data
from .generate import DecodeConfig, generate
from .model import TinyTransformer
from .tokenizer import CharTokenizer


# --------------------------------------------------------------------------
# Verifiers
# --------------------------------------------------------------------------

_STRICT = re.compile(r"^\s*answer:\s*(-?\d+)\s*$", re.IGNORECASE)


def strict_verifier(completion: str, answer: int) -> int:
    """Accept exactly one answer line, as the final line, equal to the answer.

    Two separate conditions, both necessary:

    * There must be exactly ONE line anywhere in the completion that parses as
      an answer. Reading only the last line looks equivalent and is not: it
      rewards "answer: 1 / answer: 2 / ... / answer: 5", so a policy can simply
      enumerate every candidate and collect the reward without ever solving
      anything. That is a reward hack the verifier must not permit.
    * That line must be the final non-empty line, so trailing commentary after
      the answer is rejected rather than silently ignored.

    Being generous anywhere here is how reward hacking starts.
    """
    lines = [l for l in completion.strip().splitlines() if l.strip()]
    if not lines:
        return 0
    matches = [m for m in (_STRICT.match(l) for l in lines) if m]
    if len(matches) != 1:
        return 0
    if not _STRICT.match(lines[-1]):
        return 0
    return int(int(matches[0].group(1)) == answer)


def substring_verifier(completion: str, answer: int) -> int:
    """Deliberately broken: rewards the digits appearing anywhere.

    Accepts "25" for the answer 5, and accepts a correct digit buried in
    working that reaches the wrong conclusion.
    """
    return int(str(answer) in completion)


def process_verifier(completion: str, answer: int) -> int:
    """Outcome AND stated working must both be right.

    Contrast with :func:`strict_verifier`, which cannot tell a correct answer
    reached by correct reasoning from one reached by wrong reasoning.
    """
    if not strict_verifier(completion, answer):
        return 0
    for a, b, c in re.findall(r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)", completion):
        if int(a) + int(b) != int(c):
            return 0
    return 1


VERIFIERS: Dict[str, Callable[[str, int], int]] = {
    "strict": strict_verifier,
    "substring_broken": substring_verifier,
    "process": process_verifier,
}

VERIFIER_NOTES = {
    "strict": "final line must parse as exactly one integer equal to the answer",
    "substring_broken": "rewards the answer's digits appearing anywhere; accepts 25 for 5",
    "process": "strict outcome AND every stated 'a + b = c' must be arithmetically true",
}


# --------------------------------------------------------------------------
# Group-relative advantages
# --------------------------------------------------------------------------


def group_advantages(
    rewards: Sequence[float], epsilon: float = 1e-8
) -> Dict[str, object]:
    """Normalise rewards within one prompt's group.

    Uses the POPULATION standard deviation. Other implementations divide by the
    sample standard deviation, skip the division entirely, or normalise across
    the whole batch; the choice is a variant decision, not a law, so it is
    reported alongside the numbers.
    """
    n = len(rewards)
    mean = sum(rewards) / max(1, n)
    variance = sum((r - mean) ** 2 for r in rewards) / max(1, n)
    std = math.sqrt(variance)
    if std == 0.0:
        advantages = [0.0] * n
    else:
        advantages = [(r - mean) / (std + epsilon) for r in rewards]
    return {
        "rewards": list(rewards),
        "mean": mean,
        "std": std,
        "std_kind": "population",
        "epsilon": epsilon,
        "advantages": advantages,
        "zero_variance": std == 0.0,
        "note": (
            "every completion scored the same, so there is no relative signal "
            "in this group" if std == 0.0 else
            "above-average completions get positive advantages"
        ),
    }


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------


@dataclass
class Rollout:
    task_index: int
    prompt: str
    answer: int
    text: str
    prompt_ids: List[int]
    completion_ids: List[int]
    reward: float = 0.0
    old_logprob: float = 0.0


def sample_rollouts(
    policy: TinyTransformer,
    tok: CharTokenizer,
    tasks: Sequence[data.VerifiableTask],
    group_size: int,
    decode: DecodeConfig,
    verifier: Callable[[str, int], int],
) -> List[List[Rollout]]:
    """Sample ``group_size`` completions per task and score each one.

    The hidden answer goes to the verifier only. It never enters the prompt the
    policy sees, which is what makes the reward a real check rather than a
    lookup.
    """
    groups: List[List[Rollout]] = []
    for ti, task in enumerate(tasks):
        group: List[Rollout] = []
        for k in range(group_size):
            cfg = DecodeConfig(
                max_new_tokens=decode.max_new_tokens,
                temperature=decode.temperature,
                top_k=decode.top_k,
                top_p=decode.top_p,
                # A distinct seed per sample, derived deterministically, so the
                # whole rollout set can be reproduced exactly.
                seed=None if decode.seed is None else decode.seed * 1000 + ti * 31 + k,
                use_cache=decode.use_cache,
            )
            g = generate(policy, tok, data.format_prompt(task.prompt), cfg)
            group.append(
                Rollout(
                    task_index=ti,
                    prompt=task.prompt,
                    answer=task.answer,
                    text=g.text,
                    prompt_ids=g.prompt_ids,
                    completion_ids=g.new_ids,
                    reward=float(verifier(g.text, task.answer)),
                )
            )
        groups.append(group)
    return groups


def _completion_logprobs(
    model: TinyTransformer, tok: CharTokenizer, rollouts: Sequence[Rollout]
) -> torch.Tensor:
    """Summed log-probability of each completion, differentiable.

    Only completion tokens are scored; the prompt is context, not a target.
    """
    rows = [(r.prompt_ids + r.completion_ids, len(r.prompt_ids)) for r in rollouts]
    width = max(len(ids) for ids, _ in rows) - 1
    x = torch.full((len(rows), width), tok.pad_id, dtype=torch.long)
    y = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (ids, plen) in enumerate(rows):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1], dtype=torch.long)
        targets = [-100] * plen + ids[plen:]
        y[i, :n] = torch.tensor(targets[1:], dtype=torch.long)
    logits = model(x)["logits"]
    logprobs = F.log_softmax(logits, dim=-1)
    mask = y != -100
    gathered = logprobs.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


# --------------------------------------------------------------------------
# GRPO
# --------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    iterations: int = 30
    group_size: int = 8
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.0
    temperature: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 12
    seed: int = 3
    verifier: str = "strict"
    #: Gradient steps taken per batch of rollouts. With one epoch the ratio is
    #: exactly 1 by construction and clipping can never engage; it only becomes
    #: a real constraint when the policy is updated more than once against the
    #: same sampled data.
    inner_epochs: int = 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "iterations": self.iterations,
            "group_size": self.group_size,
            "learning_rate": self.learning_rate,
            "clip_epsilon": self.clip_epsilon,
            "kl_coefficient": self.kl_coefficient,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "verifier": self.verifier,
            "inner_epochs": self.inner_epochs,
            "advantage": "(reward - group mean) / (population std + 1e-8)",
            "objective": "clipped ratio * advantage, averaged over completions",
            "variant_note": (
                "one concrete GRPO-style variant. Library defaults differ: "
                "normalisation, loss aggregation and the default KL coefficient "
                "have all changed between implementations."
            ),
        }


def grpo_train(
    policy: TinyTransformer,
    tok: CharTokenizer,
    tasks: Sequence[data.VerifiableTask],
    cfg: GRPOConfig,
    reference: Optional[TinyTransformer] = None,
) -> Dict[str, object]:
    """A real policy update driven by verifier rewards.

    Each iteration: sample a group per prompt, score it, convert rewards to
    group-relative advantages, recompute log-probabilities with gradients, form
    the clipped ratio objective, and step.
    """
    torch.manual_seed(cfg.seed)
    verifier = VERIFIERS[cfg.verifier]
    if reference is None:
        reference = copy.deepcopy(policy)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    history: List[Dict[str, object]] = []
    first_snapshot: Optional[Dict[str, object]] = None

    for it in range(cfg.iterations):
        decode = DecodeConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            seed=cfg.seed + it,
        )
        groups = sample_rollouts(policy, tok, tasks, cfg.group_size, decode, verifier)

        flat: List[Rollout] = []
        advantages: List[float] = []
        group_stats: List[Dict[str, object]] = []
        for group in groups:
            stats = group_advantages([r.reward for r in group])
            group_stats.append(stats)
            for r, a in zip(group, stats["advantages"]):
                flat.append(r)
                advantages.append(a)

        with torch.no_grad():
            old_logprobs = _completion_logprobs(policy, tok, flat)
        for r, lp in zip(flat, old_logprobs):
            r.old_logprob = float(lp)

        adv = torch.tensor(advantages, dtype=torch.float32)
        ref_logprobs = None
        if cfg.kl_coefficient > 0:
            with torch.no_grad():
                ref_logprobs = _completion_logprobs(reference, tok, flat)

        # `old_logprobs` is frozen before the first inner step, so later epochs
        # see a ratio that has genuinely moved away from 1.
        clipped_fraction = 0.0
        for _ in range(max(1, cfg.inner_epochs)):
            new_logprobs = _completion_logprobs(policy, tok, flat)
            ratio = torch.exp(new_logprobs - old_logprobs)
            unclipped = ratio * adv
            bounded = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon)
            clipped = bounded * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            clipped_fraction = float((ratio != bounded).float().mean())

            kl = torch.tensor(0.0)
            if ref_logprobs is not None:
                # k3 estimator: non-negative and lower variance than the plain
                # difference of log-probabilities.
                delta = ref_logprobs - new_logprobs
                kl = (torch.exp(delta) - delta - 1).mean()
            loss = policy_loss + cfg.kl_coefficient * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
            opt.step()

        rewards = [r.reward for r in flat]
        zero_var = sum(1 for s in group_stats if s["zero_variance"])
        record = {
            "iteration": it,
            "mean_reward": sum(rewards) / max(1, len(rewards)),
            "solved_fraction": sum(1 for r in rewards if r > 0) / max(1, len(rewards)),
            "zero_variance_groups": zero_var,
            "num_groups": len(group_stats),
            "policy_loss": float(policy_loss),
            "kl": float(kl),
            "grad_norm": grad_norm,
            "mean_ratio": float(ratio.mean()),
            "clipped_fraction": clipped_fraction,
        }
        history.append(record)

        if first_snapshot is None:
            first_snapshot = {
                "iteration": it,
                "groups": [
                    {
                        "prompt": tasks[gi].prompt,
                        "hidden_answer": tasks[gi].answer,
                        "family": tasks[gi].family,
                        "completions": [
                            {
                                "text": r.text,
                                "reward": r.reward,
                                "advantage": stats["advantages"][k],
                                "old_logprob": r.old_logprob,
                            }
                            for k, r in enumerate(groups[gi])
                        ],
                        **{
                            k: stats[k]
                            for k in ("mean", "std", "std_kind", "zero_variance", "note")
                        },
                    }
                    for gi, stats in enumerate(group_stats)
                ],
            }

    return {
        "config": cfg.to_dict(),
        "history": history,
        "first_iteration": first_snapshot,
        "verifier_note": VERIFIER_NOTES[cfg.verifier],
    }


def compare_verifiers(
    completions: Sequence[Tuple[str, int]]
) -> List[Dict[str, object]]:
    """Score the same completions under every verifier.

    This is the reward-hacking lesson in one table: a broken verifier can make
    the reward go up while task accuracy does not.
    """
    out = []
    for text, answer in completions:
        row: Dict[str, object] = {"completion": text, "answer": answer}
        for name, fn in VERIFIERS.items():
            row[name] = fn(text, answer)
        out.append(row)
    return out
