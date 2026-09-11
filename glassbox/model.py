"""The tiny inspectable transformer.

Design constraints that differ from an ordinary implementation, all in service
of being inspectable:

* Every intermediate tensor can be captured with a named-axis record, an
  operation id, and the ids of the tensors it was computed from. That is what
  lets the interface say "this number came from THESE numbers" instead of
  gesturing at a diagram.
* No fused kernels and no ``F.scaled_dot_product_attention``. A fused call
  would hide the masked score matrix, which is the thing a learner needs to
  see.
* Initialisation scale is explicit. With tied input/output embeddings,
  ``nn.Embedding``'s default ``N(0, 1)`` makes the untrained cross-entropy
  roughly twice ``ln(vocab)``, which destroys the single most useful sanity
  check in the whole course: an untrained model should be exactly as surprised
  as a uniform guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    width: int = 8
    heads: int = 2
    mlp_width: int = 16
    layers: int = 2
    #: Long enough for the longest document (159 tokens) plus generated text.
    #: The interface only ever inspects a short prompt, so this costs one
    #: position-embedding row per step of headroom and nothing else.
    context: int = 192
    #: Standard deviation for embedding and linear initialisation.
    init_std: float = 0.02
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.width % self.heads:
            raise ValueError(
                f"width {self.width} is not divisible by heads {self.heads}"
            )

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    def to_dict(self) -> Dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "width": self.width,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "mlp_width": self.mlp_width,
            "layers": self.layers,
            "context": self.context,
            "init_std": self.init_std,
            "dropout": self.dropout,
            "normalization": "pre-LayerNorm",
            "activation": "GELU",
            "position": "learned absolute",
            "output_embedding_tied": True,
        }


# --------------------------------------------------------------------------
# Trace capture
# --------------------------------------------------------------------------


def _to_nested(t: torch.Tensor) -> object:
    """Detached nested lists, with -inf replaced by None.

    JSON has no -inf. The masked score matrix is full of it, and it is exactly
    the tensor a learner must look at, so the null is meaningful here and the
    interface renders it as the masking symbol.
    """
    out = t.detach().to(torch.float64).tolist()

    def clean(o):
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
            return None
        return o

    return clean(out)


@dataclass
class Capture:
    """Collects named-axis tensor records during one forward pass."""

    records: Dict[str, Dict[str, object]] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    def add(
        self,
        operation_id: str,
        tensor: torch.Tensor,
        axes: Sequence[str],
        inputs: Sequence[str] = (),
        equation: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        if len(axes) != tensor.dim():
            raise ValueError(
                f"{operation_id}: {tensor.dim()} axes expected, got {list(axes)}"
            )
        finite = tensor.detach()
        finite = finite[torch.isfinite(finite)]
        self.records[operation_id] = {
            "operation_id": operation_id,
            "shape": list(tensor.shape),
            "axes": list(axes),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "input_operation_ids": list(inputs),
            "equation": equation,
            "note": note,
            "summary": {
                "min": float(finite.min()) if finite.numel() else None,
                "max": float(finite.max()) if finite.numel() else None,
                "mean": float(finite.mean()) if finite.numel() else None,
            },
            "values": _to_nested(tensor),
        }
        self.order.append(operation_id)


class Block(nn.Module):
    """One pre-LayerNorm transformer block."""

    def __init__(self, cfg: ModelConfig, index: int):
        super().__init__()
        self.cfg = cfg
        self.index = index
        d, f = cfg.width, cfg.mlp_width
        self.norm_attention = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.attention_out = nn.Linear(d, d, bias=False)
        self.norm_mlp = nn.LayerNorm(d)
        self.mlp_up = nn.Linear(d, f)
        self.mlp_down = nn.Linear(f, d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cap: Optional[Capture] = None,
        past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, D = x.shape
        H, hd = self.cfg.heads, self.cfg.head_dim
        p = f"block{self.index}."

        normed = self.norm_attention(x)
        q = self.q(normed)
        k = self.k(normed)
        v = self.v(normed)

        qh = q.view(B, T, H, hd).transpose(1, 2)
        kh = k.view(B, T, H, hd).transpose(1, 2)
        vh = v.view(B, T, H, hd).transpose(1, 2)

        # The KV cache concatenates PAST keys/values in front of the new ones.
        # Nothing about the learned weights changes; only how much work the
        # forward pass repeats.
        if past is not None:
            past_k, past_v = past
            kh = torch.cat([past_k, kh], dim=2)
            vh = torch.cat([past_v, vh], dim=2)
        present = (kh, vh) if use_cache else None

        S = kh.size(2)  # total source length, including cached positions
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(hd)

        # Query i is at absolute position (S - T + i); it may attend to any
        # source j <= that. With an empty cache this is the familiar lower
        # triangle; with a cache of length S-T it is "all of the past, plus the
        # causal part of the new block".
        qpos = torch.arange(S - T, S, device=x.device).view(T, 1)
        kpos = torch.arange(S, device=x.device).view(1, S)
        allowed = (kpos <= qpos).view(1, 1, T, S)
        masked = scores.masked_fill(~allowed, float("-inf"))
        weights = F.softmax(masked, dim=-1)

        mixed = weights @ vh
        merged = mixed.transpose(1, 2).contiguous().view(B, T, D)
        update = self.attention_out(merged)
        res_attention = x + self.drop(update)

        normed_mlp = self.norm_mlp(res_attention)
        up = self.mlp_up(normed_mlp)
        activated = F.gelu(up)
        down = self.mlp_down(activated)
        res_out = res_attention + self.drop(down)

        if cap is not None:
            cap.add(p + "norm_attention", normed, ["batch", "token", "channel"],
                    inputs=["residual_in" if self.index == 0 else f"block{self.index-1}.residual_out"],
                    equation="LayerNorm(residual) over the channel axis")
            cap.add(p + "q", q, ["batch", "token", "channel"], [p + "norm_attention"],
                    "q = normed @ W_q.T")
            cap.add(p + "k", k, ["batch", "token", "channel"], [p + "norm_attention"],
                    "k = normed @ W_k.T")
            cap.add(p + "v", v, ["batch", "token", "channel"], [p + "norm_attention"],
                    "v = normed @ W_v.T")
            cap.add(p + "q_heads", qh, ["batch", "head", "token", "head_channel"], [p + "q"],
                    "reshape channels into heads")
            cap.add(p + "scores", scores, ["batch", "head", "query_token", "source_token"],
                    [p + "q_heads", p + "k"], f"scores = q @ k.T / sqrt({hd})")
            cap.add(p + "masked_scores", masked,
                    ["batch", "head", "query_token", "source_token"], [p + "scores"],
                    "future source positions set to -inf",
                    note="null means -inf: masked before softmax")
            cap.add(p + "attention", weights,
                    ["batch", "head", "query_token", "source_token"], [p + "masked_scores"],
                    "softmax over the source axis")
            cap.add(p + "weighted_values", mixed,
                    ["batch", "head", "token", "head_channel"],
                    [p + "attention", p + "v"], "attention @ v")
            cap.add(p + "merged_heads", merged, ["batch", "token", "channel"],
                    [p + "weighted_values"], "concatenate heads")
            cap.add(p + "attention_update", update, ["batch", "token", "channel"],
                    [p + "merged_heads"], "merged @ W_o.T")
            cap.add(p + "residual_attention", res_attention, ["batch", "token", "channel"],
                    ["residual_in", p + "attention_update"],
                    "residual = residual + attention_update",
                    note="added to the UNNORMALISED residual, as pre-norm requires")
            cap.add(p + "norm_mlp", normed_mlp, ["batch", "token", "channel"],
                    [p + "residual_attention"], "LayerNorm(residual)")
            cap.add(p + "mlp_up", up, ["batch", "token", "mlp_channel"], [p + "norm_mlp"],
                    "up = normed @ W_up + b_up")
            cap.add(p + "mlp_activation", activated, ["batch", "token", "mlp_channel"],
                    [p + "mlp_up"], "GELU(up)")
            cap.add(p + "mlp_update", down, ["batch", "token", "channel"],
                    [p + "mlp_activation"], "update = activated @ W_down + b_down")
            cap.add(p + "residual_out", res_out, ["batch", "token", "channel"],
                    [p + "residual_attention", p + "mlp_update"],
                    "residual = residual + mlp_update")

        return res_out, present


class TinyTransformer(nn.Module):
    """Decoder-only causal transformer with tied input/output embeddings."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position_embedding = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.layers)])
        self.final_norm = nn.LayerNorm(cfg.width)
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        capture: bool = False,
        past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        offset = 0 if past is None else past[0][0].size(2)
        if offset + T > self.cfg.context:
            raise ValueError(
                f"sequence of {offset + T} exceeds context {self.cfg.context}"
            )

        pos = torch.arange(offset, offset + T, device=input_ids.device)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos).unsqueeze(0)
        x = self.drop(tok_emb + pos_emb)

        cap = Capture() if capture else None
        if cap is not None:
            cap.add("token_embedding", tok_emb, ["batch", "token", "channel"],
                    equation="one learned row per token id",
                    note="a looked-up weight row, not the id split into pieces")
            cap.add("position_embedding", pos_emb, ["batch", "token", "channel"],
                    equation="one learned row per absolute position")
            cap.add("residual_in", x, ["batch", "token", "channel"],
                    ["token_embedding", "position_embedding"],
                    "residual = token_embedding + position_embedding")

        presents: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            x, present = block(
                x, cap=cap, past=None if past is None else past[i], use_cache=use_cache
            )
            if present is not None:
                presents.append(present)

        final = self.final_norm(x)
        logits = F.linear(final, self.token_embedding.weight)

        if cap is not None:
            cap.add("final_norm", final, ["batch", "token", "channel"],
                    [f"block{self.cfg.layers - 1}.residual_out"], "LayerNorm(residual)")
            cap.add("logits", logits, ["batch", "token", "vocabulary"],
                    ["final_norm", "token_embedding"],
                    "logits = final_norm @ E.T  (E is the tied embedding table)")
            cap.add("probabilities", F.softmax(logits, dim=-1),
                    ["batch", "token", "vocabulary"], ["logits"],
                    "softmax over the vocabulary axis")

        return {
            "logits": logits,
            "capture": cap,
            "past": presents if use_cache else None,
        }

    # Convenience: the common case where only logits are wanted.
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self(input_ids)["logits"]


def build_model(cfg: ModelConfig, seed: int = 42) -> TinyTransformer:
    """Deterministic construction: same seed and config, same initial weights."""
    torch.manual_seed(seed)
    return TinyTransformer(cfg)
