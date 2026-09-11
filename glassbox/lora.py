"""LoRA: a frozen base matrix plus a trainable low-rank update.

The conventions here follow PyTorch's stored-weight layout, because silently
switching between ``A @ B`` and ``B @ A`` is the single most common way LoRA is
taught wrongly:

    W        [out, in]   frozen
    A        [r, in]     trainable, small random init
    B        [out, r]    trainable, ZERO init
    delta_W  = (alpha / r) * (B @ A)          [out, in]
    y        = x @ W.T + (alpha / r) * (x @ A.T) @ B.T

Rank is a property of the UPDATE to one matrix. It is not a number of attention
heads, and a low rank does not by itself guarantee a small behavioural change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .model import TinyTransformer


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        # B stays zero: the adapter starts as an exact no-op, so attaching it
        # cannot change the model's outputs before any training happens.

        self.merged = False

    @property
    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            out = out + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return out

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into the base weight.

        After merging the forward path is an ordinary Linear again: same
        arithmetic, no adapter branch, no extra latency.
        """
        if self.merged:
            return
        self.base.weight.add_(self.delta_weight)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight.sub_(self.delta_weight)
        self.merged = False


DEFAULT_TARGETS: Tuple[str, ...] = ("q", "v")


@dataclass
class LoRAConfig:
    rank: int = 2
    alpha: float = 4.0
    targets: Tuple[str, ...] = DEFAULT_TARGETS

    def to_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": self.alpha / self.rank,
            "targets": list(self.targets),
            "b_init": "zeros (adapter starts as a no-op)",
            "a_init": "normal(0, 0.02)",
            "convention": "W[out,in], A[r,in], B[out,r], delta_W=(alpha/r)*B@A",
        }


def apply_lora(model: TinyTransformer, cfg: LoRAConfig) -> Dict[str, object]:
    """Freeze every base parameter and attach adapters to the target matrices."""
    for p in model.parameters():
        p.requires_grad_(False)

    attached: List[str] = []
    for block in model.blocks:
        for name in cfg.targets:
            if not hasattr(block, name):
                raise KeyError(f"block has no submodule {name!r}")
            base = getattr(block, name)
            if isinstance(base, LoRALinear):
                continue
            if not isinstance(base, nn.Linear):
                raise TypeError(f"{name} is {type(base).__name__}, not nn.Linear")
            setattr(block, name, LoRALinear(base, cfg.rank, cfg.alpha))
            attached.append(f"blocks.{block.index}.{name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "config": cfg.to_dict(),
        "attached_to": attached,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_fraction": trainable / max(1, trainable + frozen),
    }


def lora_modules(model: TinyTransformer) -> List[Tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def merge_all(model: TinyTransformer) -> None:
    for _, m in lora_modules(model):
        m.merge()


def unmerge_all(model: TinyTransformer) -> None:
    for _, m in lora_modules(model):
        m.unmerge()


def adapter_report(model: TinyTransformer) -> List[Dict[str, object]]:
    """Per-adapter numbers for the interface: shapes, norms, and the rank of
    the update actually realised (which can be lower than ``r``)."""
    out: List[Dict[str, object]] = []
    for name, m in lora_modules(model):
        delta = m.delta_weight.detach()
        # Singular values say what "low rank" actually bought. Only `r` of them
        # can be non-zero; the rest are float32 round-off. matrix_rank's default
        # tolerance is tuned for the dtype it is GIVEN, so casting this float32
        # product to float64 first makes 1e-7 noise look like real structure and
        # reports full rank - the opposite of the lesson.
        singular = torch.linalg.svdvals(delta).tolist()
        largest = max(singular) if singular else 0.0
        tolerance = largest * 1e-5
        realised = int(sum(1 for s in singular if s > tolerance))
        out.append(
            {
                "module": name,
                "rank": m.rank,
                "alpha": m.alpha,
                "scaling": m.scaling,
                "base_shape": list(m.base.weight.shape),
                "A_shape": list(m.lora_A.shape),
                "B_shape": list(m.lora_B.shape),
                "delta_shape": list(delta.shape),
                "delta_frobenius_norm": float(delta.norm()),
                "base_frobenius_norm": float(m.base.weight.detach().norm()),
                "delta_matrix_rank": realised,
                "delta_singular_values": singular,
                "rank_tolerance": tolerance,
                "merged": m.merged,
                "A": m.lora_A.detach().tolist(),
                "B": m.lora_B.detach().tolist(),
                "delta_weight": delta.tolist(),
            }
        )
    return out


def first_step_gradients(
    model: TinyTransformer, loss: torch.Tensor
) -> List[Dict[str, object]]:
    """Gradient norms of A and B after one backward pass on a fresh adapter.

    With ``B = 0`` the gradient reaching ``A`` is multiplied by ``B``, so it is
    exactly zero on the first step while ``B`` already has a gradient. Teaching
    that both matrices move immediately is wrong, and this measures it instead
    of asserting it.
    """
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    out: List[Dict[str, object]] = []
    for name, m in lora_modules(model):
        out.append(
            {
                "module": name,
                "A_grad_norm": float(m.lora_A.grad.norm()) if m.lora_A.grad is not None else None,
                "B_grad_norm": float(m.lora_B.grad.norm()) if m.lora_B.grad is not None else None,
                "B_is_zero": bool(torch.all(m.lora_B.detach() == 0)),
            }
        )
    model.zero_grad(set_to_none=True)
    return out
