"""Inference: prefill, decode, and sampling controls that actually change the
output.

The previous engine accepted a ``temperature`` argument, divided the logits by
it, and then took ``argmax``. Argmax is invariant under positive scaling, so
temperature did nothing at all. Every control here is tested against a
behavioural assertion in ``checks.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .model import TinyTransformer
from .tokenizer import CharTokenizer


@dataclass
class DecodeConfig:
    """One decoding protocol. Evaluation must pin this, not leave it implicit."""

    max_new_tokens: int = 24
    #: 0.0 means greedy. Any positive value samples.
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 0.0
    seed: Optional[int] = None
    use_cache: bool = True
    stop_at_eos: bool = True

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0

    def describe(self) -> str:
        if self.is_greedy:
            return f"greedy, max {self.max_new_tokens} new tokens"
        bits = [f"temperature {self.temperature}"]
        if self.top_k:
            bits.append(f"top-k {self.top_k}")
        if self.top_p:
            bits.append(f"top-p {self.top_p}")
        bits.append(f"max {self.max_new_tokens} new tokens")
        if self.seed is not None:
            bits.append(f"seed {self.seed}")
        return ", ".join(bits)

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
            "use_cache": self.use_cache,
            "stop_at_eos": self.stop_at_eos,
            "protocol": self.describe(),
        }


GREEDY = DecodeConfig()


@dataclass
class Generation:
    """Result of one generation, with enough detail to explain every choice."""

    prompt: str
    text: str
    prompt_ids: List[int]
    new_ids: List[int]
    stopped_on_eos: bool
    hit_length_limit: bool
    steps: List[Dict[str, object]] = field(default_factory=list)
    protocol: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "prompt": self.prompt,
            "text": self.text,
            "prompt_ids": self.prompt_ids,
            "new_ids": self.new_ids,
            "stopped_on_eos": self.stopped_on_eos,
            "hit_length_limit": self.hit_length_limit,
            "protocol": self.protocol,
            "steps": self.steps,
        }


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Mask out tokens excluded by top-k / top-p, leaving the rest untouched."""
    out = logits.clone()
    if top_k and top_k < out.numel():
        kth = torch.topk(out, top_k).values[-1]
        out[out < kth] = float("-inf")
    if top_p and 0.0 < top_p < 1.0:
        ordered, index = torch.sort(out, descending=True)
        probs = F.softmax(ordered, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # Keep the smallest prefix whose mass reaches top_p. The shift keeps the
        # first token above the threshold, so the set is never empty.
        drop = cumulative - probs > top_p
        ordered[drop] = float("-inf")
        out = torch.full_like(out, float("-inf")).scatter_(0, index, ordered)
    return out


@torch.no_grad()
def generate(
    model: TinyTransformer,
    tok: CharTokenizer,
    prompt: str,
    cfg: DecodeConfig = GREEDY,
    record_steps: bool = False,
) -> Generation:
    """Prefill the prompt, then decode one token at a time.

    With ``use_cache`` the prompt is processed once and each new token attends
    to cached keys and values. Without it the whole prefix is recomputed every
    step. Both paths must produce identical tokens; that equality is asserted
    in ``checks.check_cache_equivalence``.
    """
    was_training = model.training
    model.eval()
    try:
        prompt_ids = tok.encode(prompt, bos=True)
        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

        ids = list(prompt_ids)
        new_ids: List[int] = []
        steps: List[Dict[str, object]] = []
        past = None
        stopped = False

        # Prefill.
        if cfg.use_cache:
            out = model(torch.tensor([ids]), use_cache=True)
            past = out["past"]
            next_logits = out["logits"][0, -1]
        else:
            next_logits = model(torch.tensor([ids]))["logits"][0, -1]

        for _ in range(cfg.max_new_tokens):
            if len(ids) >= model.cfg.context:
                break
            filtered = _filter_logits(next_logits, cfg.top_k, cfg.top_p)
            if cfg.is_greedy:
                chosen = int(torch.argmax(filtered))
                probs = F.softmax(filtered, dim=-1)
            else:
                probs = F.softmax(filtered / cfg.temperature, dim=-1)
                chosen = int(torch.multinomial(probs, 1, generator=generator))

            if record_steps:
                top = torch.topk(probs, min(5, probs.numel()))
                steps.append(
                    {
                        "position": len(ids),
                        "chosen_id": chosen,
                        "chosen_token": tok.display(chosen),
                        "chosen_probability": float(probs[chosen]),
                        "top": [
                            {
                                "id": int(i),
                                "token": tok.display(int(i)),
                                "probability": float(p),
                            }
                            for p, i in zip(top.values, top.indices)
                        ],
                        "candidates_kept": int(torch.isfinite(filtered).sum()),
                    }
                )

            ids.append(chosen)
            new_ids.append(chosen)
            if cfg.stop_at_eos and chosen == tok.eos_id:
                stopped = True
                break

            if cfg.use_cache:
                out = model(torch.tensor([[chosen]]), past=past, use_cache=True)
                past = out["past"]
                next_logits = out["logits"][0, -1]
            else:
                next_logits = model(torch.tensor([ids]))["logits"][0, -1]

        return Generation(
            prompt=prompt,
            text=tok.decode(new_ids),
            prompt_ids=prompt_ids,
            new_ids=new_ids,
            stopped_on_eos=stopped,
            hit_length_limit=not stopped and len(new_ids) >= cfg.max_new_tokens,
            steps=steps,
            protocol=cfg.describe(),
        )
    finally:
        model.train(was_training)


@torch.no_grad()
def sequence_logprob(
    model: TinyTransformer,
    tok: CharTokenizer,
    prompt: str,
    response: str,
    include_eos: bool = True,
) -> Dict[str, object]:
    """Total log-probability the model assigns to ``response`` after ``prompt``.

    Prompt tokens are never scored. This is the quantity DPO compares between
    the policy and its frozen reference, so it returns the per-token detail too
    rather than only the sum.
    """
    prompt_ids = tok.encode(prompt, bos=True)
    response_ids = tok.encode(response) + ([tok.eos_id] if include_eos else [])
    ids = torch.tensor([prompt_ids + response_ids])
    logits = model(ids)["logits"]
    logprobs = F.log_softmax(logits, dim=-1)

    # Position i predicts token i+1, so the first scored prediction comes from
    # the last prompt position.
    start = len(prompt_ids) - 1
    per_token: List[Dict[str, object]] = []
    total = 0.0
    for offset, target in enumerate(response_ids):
        lp = float(logprobs[0, start + offset, target])
        total += lp
        per_token.append(
            {"token": tok.display(target), "id": target, "logprob": lp}
        )
    return {
        "total_logprob": total,
        "mean_logprob": total / max(1, len(response_ids)),
        "num_scored_tokens": len(response_ids),
        "per_token": per_token,
    }
