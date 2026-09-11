"""Evaluation: what counts as evidence that the model improved.

Every number here is produced under a PINNED decoding protocol that is reported
alongside it. "The model answered correctly" means nothing until you say how it
was decoded and how many attempts it had.

The instruction families are reported separately and never averaged together.
An average over trained and held-out prompts hides exactly the thing a learner
needs to see: that a 3,000-parameter model memorises twelve answers and
generalises to almost none of them.
"""

from __future__ import annotations

import math

from typing import Dict, Optional, Sequence

from . import data
from .generate import DecodeConfig, GREEDY, generate
from .model import TinyTransformer
from .rlvr import VERIFIERS
from .tokenizer import CharTokenizer
from .train import document_windows, mean_token_loss


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def evaluate_instructions(
    model: TinyTransformer,
    tok: CharTokenizer,
    families: Optional[Dict[str, Sequence[data.Instruction]]] = None,
    decode: DecodeConfig = GREEDY,
) -> Dict[str, object]:
    """Exact-match accuracy per family, plus formatting behaviour."""
    families = data.instruction_families() if families is None else families
    results: Dict[str, object] = {}
    for name, items in families.items():
        rows = []
        for item in items:
            g = generate(model, tok, data.format_prompt(item.prompt), decode)
            rows.append(
                {
                    "prompt": item.prompt,
                    "expected": item.answer,
                    "generated": g.text,
                    "exact_match": _normalise(g.text) == _normalise(item.answer),
                    "stopped_on_eos": g.stopped_on_eos,
                    "length": len(g.new_ids),
                }
            )
        n = max(1, len(rows))
        results[name] = {
            "items": rows,
            "count": len(rows),
            "exact_match": sum(r["exact_match"] for r in rows) / n,
            "eos_rate": sum(r["stopped_on_eos"] for r in rows) / n,
            "mean_length": sum(r["length"] for r in rows) / n,
        }
    return {
        "protocol": decode.describe(),
        "decode": decode.to_dict(),
        "families": results,
        "family_meaning": {
            "train": "prompts seen during SFT; measures memorisation only",
            "repeat": "identical wording to training; an upper bound, not a score",
            "paraphrase": "trained facts, untrained wording",
            "heldout": "facts in the paragraphs never asked during SFT",
        },
    }


def evaluate_tasks(
    model: TinyTransformer,
    tok: CharTokenizer,
    families: Optional[Dict[str, Sequence[data.VerifiableTask]]] = None,
    decode: DecodeConfig = GREEDY,
    verifier: str = "strict",
    attempts: int = 1,
    sample_temperature: float = 1.0,
    sample_top_k: int = 6,
) -> Dict[str, object]:
    """pass@1 under a fixed protocol, and optionally pass@k with its budget.

    pass@k is reported WITH the number of attempts and the token budget spent,
    because "best of sixteen" and "first try" are not the same claim.
    """
    families = data.task_families() if families is None else families
    check = VERIFIERS[verifier]
    out: Dict[str, object] = {}
    for name, tasks in families.items():
        rows = []
        for ti, task in enumerate(tasks):
            g1 = generate(model, tok, data.format_prompt(task.prompt), decode)
            first = check(g1.text, task.answer)
            tokens = len(g1.new_ids)
            any_ok = bool(first)
            samples = [g1.text]
            for k in range(1, attempts):
                gk = generate(
                    model,
                    tok,
                    data.format_prompt(task.prompt),
                    DecodeConfig(
                        max_new_tokens=decode.max_new_tokens,
                        temperature=sample_temperature,
                        top_k=sample_top_k,
                        seed=1000 + ti * 97 + k,
                    ),
                )
                samples.append(gk.text)
                tokens += len(gk.new_ids)
                any_ok = any_ok or bool(check(gk.text, task.answer))
            rows.append(
                {
                    "prompt": task.prompt,
                    "hidden_answer": task.answer,
                    "generated": g1.text,
                    "pass_1": bool(first),
                    "pass_k": any_ok,
                    "samples": samples,
                    "tokens_spent": tokens,
                }
            )
        n = max(1, len(rows))
        out[name] = {
            "items": rows,
            "count": len(rows),
            "pass_1": sum(r["pass_1"] for r in rows) / n,
            "pass_k": sum(r["pass_k"] for r in rows) / n,
            "attempts": attempts,
            "total_tokens_spent": sum(r["tokens_spent"] for r in rows),
        }
    return {
        "verifier": verifier,
        "protocol": decode.describe(),
        "attempts": attempts,
        "families": out,
        "family_meaning": {
            "add": "template and values a warm start may use",
            "template": "same template, values disjoint from 'add'",
            "compose": "two operations; a structurally held-out composition",
        },
    }


def retention(model: TinyTransformer, tok: CharTokenizer) -> Dict[str, float]:
    """Loss on the original prose task, to measure what adaptation cost."""
    train_rows = [w.ids for w in document_windows(tok, data.split_documents("train"), model.cfg.context)]
    val_rows = [w.ids for w in document_windows(tok, data.split_documents("validation"), model.cfg.context)]
    return {
        "train_prose_loss": mean_token_loss(model, train_rows, tok.pad_id),
        "validation_prose_loss": mean_token_loss(model, val_rows, tok.pad_id),
    }


def checkpoint_report(
    model: TinyTransformer,
    tok: CharTokenizer,
    label: str,
    decode: DecodeConfig = GREEDY,
) -> Dict[str, object]:
    """One comparable row per checkpoint, as the blueprint's section 10 requires."""
    instructions = evaluate_instructions(model, tok, decode=decode)
    tasks = evaluate_tasks(model, tok, decode=decode)
    ret = retention(model, tok)
    return {
        "checkpoint": label,
        "protocol": decode.describe(),
        "retention": ret,
        "instruction_exact_match": {
            k: v["exact_match"] for k, v in instructions["families"].items()
        },
        "instruction_eos_rate": {
            k: v["eos_rate"] for k, v in instructions["families"].items()
        },
        "task_pass_1": {k: v["pass_1"] for k, v in tasks["families"].items()},
        "instructions": instructions,
        "tasks": tasks,
    }


def compare_checkpoints(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Side-by-side table. Differences only mean something within a column."""
    return {
        "checkpoints": [r["checkpoint"] for r in reports],
        "rows": [
            {
                "metric": metric,
                "values": [r[group][metric] for r in reports],
                "group": group,
            }
            for group, metric in (
                [("retention", "train_prose_loss"), ("retention", "validation_prose_loss")]
                + [("instruction_exact_match", k) for k in ("train", "repeat", "paraphrase", "heldout")]
                + [("task_pass_1", k) for k in ("add", "template", "compose")]
            )
        ],
        "warning": (
            "Prose loss and instruction accuracy measure different "
            "distributions under different masks. Compare each metric with its "
            "own earlier value, never across rows."
        ),
    }


# --------------------------------------------------------------------------
# Trivial baselines
# --------------------------------------------------------------------------


def _count_baselines(tok: CharTokenizer) -> Dict[str, object]:
    """Unigram and bigram character models, fitted on the training split only."""
    from collections import Counter

    unigram: "Counter[int]" = Counter()
    bigram: Dict[int, "Counter[int]"] = {}
    for doc in data.split_documents("train"):
        ids = tok.encode(doc.text, bos=True, eos=True)
        for a, b in zip(ids, ids[1:]):
            unigram[b] += 1
            bigram.setdefault(a, Counter())[b] += 1
    return {"unigram": unigram, "bigram": bigram}


def baseline_losses(
    tok: CharTokenizer, context: int, alpha: float = 1.0
) -> Dict[str, Dict[str, float]]:
    """Loss of three trivial models on exactly the positions the model is scored on.

    This is the reference a "the loss went down" claim needs. Without it, a
    curve that falls from a broken initialisation to slightly-better-than-
    guessing reads as success. Laplace smoothing with alpha = 1 keeps every
    probability finite on unseen characters.
    """
    counts = _count_baselines(tok)
    unigram, bigram = counts["unigram"], counts["bigram"]
    V = len(tok)
    total_unigram = sum(unigram.values())

    out: Dict[str, Dict[str, float]] = {}
    for split in ("train", "validation"):
        windows = document_windows(tok, data.split_documents(split), context)
        n = 0
        uniform_sum = 0.0
        unigram_sum = 0.0
        bigram_sum = 0.0
        for w in windows:
            for a, b in zip(w.ids, w.ids[1:]):
                n += 1
                uniform_sum += math.log(V)
                p_uni = (unigram[b] + alpha) / (total_unigram + alpha * V)
                unigram_sum += -math.log(p_uni)
                row = bigram.get(a)
                denom = (sum(row.values()) if row else 0) + alpha * V
                p_bi = ((row[b] if row else 0) + alpha) / denom
                bigram_sum += -math.log(p_bi)
        out[split] = {
            "scored_positions": n,
            "uniform": uniform_sum / max(1, n),
            "unigram_laplace": unigram_sum / max(1, n),
            "bigram_laplace": bigram_sum / max(1, n),
        }
    out["note"] = (
        "Fitted on the training split only and scored on exactly the positions "
        "the model is scored on. A transformer that does not beat the bigram "
        "model has not earned its parameters."
    )
    out["alpha"] = alpha
    return out
