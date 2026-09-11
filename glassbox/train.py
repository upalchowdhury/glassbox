"""Pretraining, instruction fine-tuning, and the one-step gradient demo.

Three bugs from the original engine are fixed here, each of which taught
something false:

1. Windows are built per document, so a training example never predicts the
   first character of one paragraph from the last character of another. The old
   loop joined all paragraphs with no separator and slid a stride-7 window
   across the seam, while the interface claimed "training windows stay inside
   paragraphs".
2. The gradient demo restores the model afterwards. The old one applied an SGD
   step to ``block0.mlp_up`` and never undid it, so base training began from
   weights the 'random initialisation' checkpoint had never recorded.
3. The demo is recomputed for each checkpoint. The old code aliased the same
   dict into all three, so the interface's checkpoint selector silently showed
   random-initialisation gradients while claiming to show the SFT model's.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import data
from .model import TinyTransformer
from .tokenizer import CharTokenizer

IGNORE = -100


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """One training window, tagged with the document it came from."""

    doc_id: str
    start: int
    ids: List[int]


def document_windows(
    tok: CharTokenizer, docs: Sequence[data.Document], context: int
) -> List[Window]:
    """Every window that fits inside a single document.

    Each document is wrapped as ``<bos> text <eos>`` so the model learns where a
    document starts and, crucially, where it ends - which is what makes EOS a
    real learned signal rather than decoration.
    """
    out: List[Window] = []
    for doc in docs:
        ids = tok.encode(doc.text, bos=True, eos=True)
        limit = context + 1
        for start in range(0, len(ids) - 1):
            chunk = ids[start : start + limit]
            if len(chunk) < 2:
                continue
            out.append(Window(doc.doc_id, start, chunk))
    return out


def pad_to_batch(
    rows: Sequence[Sequence[int]], pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shift-by-one inputs and targets, with padding excluded from the loss."""
    width = max(len(r) for r in rows) - 1
    x = torch.full((len(rows), width), pad_id, dtype=torch.long)
    y = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    for i, row in enumerate(rows):
        n = len(row) - 1
        x[i, :n] = torch.tensor(row[:-1], dtype=torch.long)
        y[i, :n] = torch.tensor(row[1:], dtype=torch.long)
    return x, y


def masked_rows_to_batch(
    rows: Sequence[Tuple[Sequence[int], Sequence[int]]], pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as :func:`pad_to_batch` but each row carries its own target mask."""
    width = max(len(ids) for ids, _ in rows) - 1
    x = torch.full((len(rows), width), pad_id, dtype=torch.long)
    y = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    for i, (ids, targets) in enumerate(rows):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1], dtype=torch.long)
        y[i, :n] = torch.tensor(targets[1:], dtype=torch.long)
    return x, y


@torch.no_grad()
def mean_token_loss(
    model: TinyTransformer,
    rows: Sequence[Sequence[int]],
    pad_id: int,
    batch_size: int = 64,
) -> float:
    """Cross-entropy per SCORED token over every row.

    Token-weighted, not batch-averaged: otherwise a short row would count as
    much as a long one and the number would depend on the batch size.
    """
    was_training = model.training
    model.eval()
    total, counted = 0.0, 0
    for i in range(0, len(rows), batch_size):
        x, y = pad_to_batch(rows[i : i + batch_size], pad_id)
        logits = model(x)["logits"]
        total += float(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE,
                reduction="sum",
            )
        )
        counted += int((y != IGNORE).sum())
    model.train(was_training)
    return total / max(1, counted)


# --------------------------------------------------------------------------
# Pretraining
# --------------------------------------------------------------------------


@dataclass
class PretrainConfig:
    steps: int = 2000
    batch_size: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    seed: int = 1234
    eval_every: int = 125

    def to_dict(self) -> Dict[str, object]:
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "optimizer": "AdamW",
            "schedule": "linear warmup then cosine decay",
            "loss": "next-token cross-entropy, padding ignored",
        }


def lr_at(step: int, cfg: PretrainConfig) -> float:
    """Linear warmup, then cosine decay. Written out so the interface can plot
    the schedule a learner is actually training under."""
    if step <= cfg.warmup_steps:
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def pretrain(
    model: TinyTransformer,
    tok: CharTokenizer,
    cfg: PretrainConfig,
    on_step=None,
) -> Dict[str, object]:
    train_windows = document_windows(tok, data.split_documents("train"), model.cfg.context)
    val_windows = document_windows(tok, data.split_documents("validation"), model.cfg.context)
    train_rows = [w.ids for w in train_windows]
    val_rows = [w.ids for w in val_windows]

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    history: List[Dict[str, float]] = []

    def record(step: int, lr: float, grad_norm: Optional[float]) -> None:
        history.append(
            {
                "step": step,
                "train": mean_token_loss(model, train_rows, tok.pad_id),
                "validation": mean_token_loss(model, val_rows, tok.pad_id),
                "learning_rate": lr,
                "grad_norm": grad_norm,
            }
        )

    record(0, 0.0, None)
    model.train()
    for step in range(1, cfg.steps + 1):
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        picks = torch.randint(0, len(train_rows), (cfg.batch_size,), generator=gen).tolist()
        x, y = pad_to_batch([train_rows[i] for i in picks], tok.pad_id)
        logits = model(x)["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        )
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            record(step, lr, grad_norm)
            if on_step:
                on_step(history[-1])

    best = min(history, key=lambda h: h["validation"])
    return {
        "history": history,
        "config": cfg.to_dict(),
        "num_train_windows": len(train_rows),
        "num_validation_windows": len(val_rows),
        "train_documents": [d.doc_id for d in data.split_documents("train")],
        "validation_documents": [d.doc_id for d in data.split_documents("validation")],
        "best_validation": {"step": best["step"], "validation": best["validation"]},
        "final": history[-1],
    }


# --------------------------------------------------------------------------
# Instruction fine-tuning
# --------------------------------------------------------------------------


@dataclass
class SFTConfig:
    steps: int = 400
    batch_size: int = 12
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0
    seed: int = 99
    eval_every: int = 25
    #: Supervise the end-of-sequence token. Without this the model never learns
    #: to stop, and generation only ever ends by hitting the length limit.
    supervise_eos: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "supervise_eos": self.supervise_eos,
            "optimizer": "AdamW",
            "loss": "next-token cross-entropy on assistant-answer targets only",
        }


def sft_row(
    tok: CharTokenizer, item: data.Instruction, supervise_eos: bool = True
) -> Tuple[List[int], List[int]]:
    """One SFT example as (input ids, target ids) with the prompt masked out.

    The returned target list is aligned to the INPUT positions, so
    ``targets[i]`` is what position ``i`` should predict. Callers shift by one;
    doing the shift in one place is what keeps the displayed mask and the
    trained mask identical.
    """
    prompt_ids = tok.encode(data.format_prompt(item.prompt), bos=True)
    answer_ids = tok.encode(item.answer)
    if supervise_eos:
        answer_ids = answer_ids + [tok.eos_id]
    ids = prompt_ids + answer_ids
    targets = [IGNORE] * len(prompt_ids) + answer_ids
    return ids, targets


def mask_explanation(
    tok: CharTokenizer, item: data.Instruction, supervise_eos: bool = True
) -> Dict[str, object]:
    """Exactly what the interface renders, derived from the training row itself.

    Built from :func:`sft_row` rather than reconstructed independently, so the
    chips a learner sees cannot drift from the mask the optimiser used.
    """
    ids, targets = sft_row(tok, item, supervise_eos)
    positions = []
    for i in range(len(ids) - 1):
        target = targets[i + 1]
        positions.append(
            {
                "position": i,
                "input_id": ids[i],
                "input_token": tok.display(ids[i]),
                "target_id": None if target == IGNORE else target,
                "target_token": None if target == IGNORE else tok.display(target),
                "scored": target != IGNORE,
            }
        )
    return {
        "prompt": item.prompt,
        "answer": item.answer,
        "formatted": data.format_prompt(item.prompt) + item.answer,
        "positions": positions,
        "num_scored": sum(1 for p in positions if p["scored"]),
        "num_ignored": sum(1 for p in positions if not p["scored"]),
        "supervise_eos": supervise_eos,
    }


def sft(
    model: TinyTransformer,
    tok: CharTokenizer,
    cfg: SFTConfig,
    retention_rows: Optional[Sequence[Sequence[int]]] = None,
    items: Optional[Sequence[data.Instruction]] = None,
) -> Dict[str, object]:
    """``items`` defaults to the instruction training split. Passing a different
    list is how the RL chapter warm-starts the policy on task formatting."""
    items = list(data.SFT_TRAIN if items is None else items)
    rows = [sft_row(tok, item, cfg.supervise_eos) for item in items]
    if retention_rows is None:
        retention_rows = [
            w.ids
            for w in document_windows(
                tok, data.split_documents("validation"), model.cfg.context
            )
        ]

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    history: List[Dict[str, float]] = []

    @torch.no_grad()
    def answer_loss() -> float:
        was = model.training
        model.eval()
        x, y = masked_rows_to_batch(rows, tok.pad_id)
        logits = model(x)["logits"]
        total = float(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE,
                reduction="sum",
            )
        )
        model.train(was)
        return total / max(1, int((y != IGNORE).sum()))

    def record(step: int, lr: float) -> None:
        history.append(
            {
                "step": step,
                "train": answer_loss(),
                "retention": mean_token_loss(model, retention_rows, tok.pad_id),
                "learning_rate": lr,
            }
        )

    record(0, 0.0)
    model.train()
    for step in range(1, cfg.steps + 1):
        if step <= cfg.warmup_steps:
            lr = cfg.learning_rate * step / max(1, cfg.warmup_steps)
        else:
            progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
            lr = cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        for group in opt.param_groups:
            group["lr"] = lr
        picks = torch.randint(
            0, len(rows), (min(cfg.batch_size, len(rows)),), generator=gen
        ).tolist()
        x, y = masked_rows_to_batch([rows[i] for i in picks], tok.pad_id)
        logits = model(x)["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            record(step, lr)

    return {
        "history": history,
        "config": cfg.to_dict(),
        "num_examples": len(rows),
        "mask_example": mask_explanation(tok, items[0], cfg.supervise_eos),
        "final": history[-1],
    }


# --------------------------------------------------------------------------
# One gradient step, inspected
# --------------------------------------------------------------------------


def sgd_step_demo(
    model: TinyTransformer,
    tok: CharTokenizer,
    prompt: str = "pip has ",
    learning_rate: float = 0.05,
    matrix_name: str = "blocks.0.mlp_up.weight",
    fd_epsilon: float = 1e-4,
) -> Dict[str, object]:
    """Take one real SGD step on every TRAINABLE parameter, then restore.

    Returns the before/gradient/delta/after view of one displayed matrix plus a
    float64 finite-difference check of the largest-magnitude gradient among the
    trainable parameters.

    If the displayed matrix is frozen - which is exactly the situation on a LoRA
    checkpoint - that is reported rather than worked around. A frozen matrix has
    no gradient and cannot move, and saying so is the lesson.

    The model is restored from a deep copy before returning, so inspecting a
    step never changes the checkpoint being inspected.
    """
    saved = copy.deepcopy(model.state_dict())
    was_training = model.training
    model.eval()
    try:
        params = dict(model.named_parameters())
        if matrix_name not in params:
            raise KeyError(f"{matrix_name} not in {sorted(params)}")
        target = params[matrix_name]
        target_is_trainable = bool(target.requires_grad)

        ids = torch.tensor([tok.encode(prompt, bos=True)])
        inputs, targets = ids[:, :-1], ids[:, 1:]

        def loss_now() -> torch.Tensor:
            logits = model(inputs)["logits"]
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        model.zero_grad(set_to_none=True)
        loss_before = loss_now()
        loss_before.backward()

        before = target.detach().clone()
        grad = target.grad.detach().clone() if target.grad is not None else None

        # The finite-difference check needs a parameter that actually HAS a
        # gradient. Prefer the displayed matrix; fall back to the largest
        # gradient among trainable 2-D parameters (the LoRA adapters).
        if grad is not None:
            fd_name, fd_grad = matrix_name, grad
        else:
            candidates = [
                (n, p.grad.detach())
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is not None and p.dim() == 2
            ]
            if not candidates:
                raise RuntimeError("no trainable 2-D parameter has a gradient")
            fd_name, fd_grad = max(candidates, key=lambda kv: float(kv[1].abs().max()))

        flat = int(torch.argmax(fd_grad.abs()))
        row, col = divmod(flat, fd_grad.size(1))

        delta = None if grad is None else -learning_rate * grad
        updated_tensors, updated_elements = 0, 0
        frozen_tensors, frozen_elements = 0, 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    p.add_(-learning_rate * p.grad)
                    updated_tensors += 1
                    updated_elements += p.numel()
                else:
                    frozen_tensors += 1
                    frozen_elements += p.numel()
            loss_after = float(loss_now())
        after = target.detach().clone()

        # Finite difference in float64. float32 with eps 1e-4 leaves a relative
        # error around 1e-2, which is not a gradient check - it is noise.
        model.load_state_dict(saved)
        fd_model = copy.deepcopy(model).double()
        fd_model.eval()
        fd_param = dict(fd_model.named_parameters())[fd_name]

        def fd_loss() -> float:
            logits = fd_model(inputs)["logits"]
            return float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
            )

        with torch.no_grad():
            original = fd_param[row, col].item()
            fd_param[row, col] = original + fd_epsilon
            plus = fd_loss()
            fd_param[row, col] = original - fd_epsilon
            minus = fd_loss()
            fd_param[row, col] = original
        finite_difference = (plus - minus) / (2 * fd_epsilon)
        analytic = float(fd_grad[row, col])
        denominator = max(abs(finite_difference), abs(analytic), 1e-12)

        return {
            "matrix": matrix_name,
            "matrix_is_trainable": target_is_trainable,
            "frozen_note": None if target_is_trainable else (
                f"{matrix_name} is frozen on this checkpoint: requires_grad is "
                "False, so it receives no gradient and cannot be updated. The "
                "trainable adapter matrices are what move here."
            ),
            "learning_rate": learning_rate,
            "prompt": prompt,
            "row": row,
            "column": col,
            "cell_selected_by": "largest absolute gradient in the checked matrix",
            "before": before.tolist(),
            "gradient": None if grad is None else grad.tolist(),
            "delta": None if delta is None else delta.tolist(),
            "after": after.tolist(),
            "display_matrix_changed": bool(not torch.equal(before, after)),
            "loss_before": float(loss_before),
            "loss_after": loss_after,
            "updated_parameter_tensors": updated_tensors,
            "updated_parameter_elements": updated_elements,
            "frozen_parameter_tensors": frozen_tensors,
            "frozen_parameter_elements": frozen_elements,
            "total_parameter_elements": model.num_parameters,
            "all_parameters_updated": frozen_elements == 0,
            "finite_difference": {
                "checked_matrix": fd_name,
                "epsilon": fd_epsilon,
                "dtype": "float64",
                "loss_plus": plus,
                "loss_minus": minus,
                "estimate": finite_difference,
                "analytic": analytic,
                "absolute_error": abs(finite_difference - analytic),
                "relative_error": abs(finite_difference - analytic) / denominator,
            },
        }
    finally:
        model.load_state_dict(saved)
        model.zero_grad(set_to_none=True)
        model.train(was_training)
