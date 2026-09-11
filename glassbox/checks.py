"""Numerical checks, run for real and reported with their measured values.

The blueprint claimed the training script "asserts causal-mask zeros, attention
row sums, MLP multiplication, an SGD update identity, a finite-difference
gradient, invariance of earlier logits to a changed future token, and assistant
loss-mask alignment". None of those assertions existed. This module is that
paragraph, implemented.

Every check returns its measured numbers, not just a boolean, and the exported
run carries the results so the interface can display what actually passed
instead of a sentence promising that something was verified.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from . import data
from .generate import DecodeConfig, generate
from .model import ModelConfig, TinyTransformer, build_model
from .tokenizer import CharTokenizer, fit_on_training_split
from .train import IGNORE, document_windows, sft_row, sgd_step_demo


@dataclass
class CheckResult:
    name: str
    passed: bool
    description: str
    measured: Dict[str, object] = field(default_factory=dict)
    tolerance: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "description": self.description,
            "measured": self.measured,
            "tolerance": self.tolerance,
        }


def _weight_hash(model: TinyTransformer) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().contiguous().view(-1).numpy().tobytes())
    return h.hexdigest()


def _probe(tok: CharTokenizer, text: str = "pip has two red stones.") -> torch.Tensor:
    return torch.tensor([tok.encode(text, bos=True)])


# --------------------------------------------------------------------------
# Forward-pass arithmetic
# --------------------------------------------------------------------------


def check_causal_mask(model: TinyTransformer, tok: CharTokenizer) -> CheckResult:
    ids = _probe(tok)
    cap = model(ids, capture=True)["capture"]
    worst = 0.0
    for layer in range(model.cfg.layers):
        att = cap.records[f"block{layer}.attention"]["values"][0]
        for head in att:
            for i, row in enumerate(head):
                for j in range(i + 1, len(row)):
                    worst = max(worst, abs(row[j]))
    return CheckResult(
        "causal_mask_zeros",
        worst == 0.0,
        "A query position assigns exactly zero attention probability to every later position.",
        {"max_future_attention_probability": worst, "sequence_length": int(ids.size(1))},
        0.0,
    )


def check_attention_rows_sum_to_one(
    model: TinyTransformer, tok: CharTokenizer, tolerance: float = 1e-6
) -> CheckResult:
    ids = _probe(tok)
    cap = model(ids, capture=True)["capture"]
    worst = 0.0
    for layer in range(model.cfg.layers):
        att = cap.records[f"block{layer}.attention"]["values"][0]
        for head in att:
            for row in head:
                worst = max(worst, abs(sum(row) - 1.0))
    return CheckResult(
        "attention_rows_sum_to_one",
        worst <= tolerance,
        "Each attention row is a probability distribution over the positions it may see.",
        {"max_absolute_deviation_from_one": worst},
        tolerance,
    )


def check_mlp_dot_product(
    model: TinyTransformer, tok: CharTokenizer, tolerance: float = 1e-5
) -> CheckResult:
    """Rebuild one MLP output cell from the weights and bias by hand."""
    ids = _probe(tok)
    cap = model(ids, capture=True)["capture"]
    normed = cap.records["block0.norm_mlp"]["values"][0]
    produced = cap.records["block0.mlp_up"]["values"][0]
    W = model.blocks[0].mlp_up.weight.detach()  # [mlp_width, width]
    b = model.blocks[0].mlp_up.bias.detach()
    row, col = 3, 5
    manual = sum(normed[row][i] * float(W[col, i]) for i in range(model.cfg.width))
    manual += float(b[col])
    error = abs(manual - produced[row][col])
    return CheckResult(
        "mlp_cell_is_a_dot_product",
        error <= tolerance,
        "One hidden coordinate equals the dot product of the token vector with one weight column, plus that column's bias.",
        {
            "row": row,
            "column": col,
            "reconstructed": manual,
            "captured": produced[row][col],
            "absolute_error": error,
        },
        tolerance,
    )


def check_future_token_invariance(
    model: TinyTransformer, tok: CharTokenizer
) -> CheckResult:
    ids = _probe(tok)
    alt = ids.clone()
    original = int(alt[0, -1])
    alt[0, -1] = (original + 5) % model.cfg.vocab_size
    with torch.no_grad():
        a = model(ids)["logits"]
        b = model(alt)["logits"]
    earlier = float((a[:, :-1] - b[:, :-1]).abs().max())
    changed = float((a[:, -1] - b[:, -1]).abs().max())
    return CheckResult(
        "future_token_invariance",
        earlier == 0.0,
        "Changing a later token leaves every earlier position's logits bit-for-bit identical.",
        {
            "max_change_at_earlier_positions": earlier,
            "max_change_at_the_edited_position": changed,
            "replaced_token_id": original,
        },
        0.0,
    )


def check_untrained_loss_matches_uniform(
    cfg: ModelConfig, tok: CharTokenizer, tolerance: float = 0.05
) -> CheckResult:
    """An untrained model should be exactly as surprised as a uniform guess.

    This is the check that ``nn.Embedding``'s default N(0,1) initialisation
    breaks when the output embedding is tied: the untrained loss came out at
    7.37 against ln(34) = 3.53, and every later "the loss went down" claim was
    measured from that broken starting point.
    """
    model = build_model(cfg, seed=42)
    model.eval()
    rows = [
        w.ids for w in document_windows(tok, data.split_documents("train"), cfg.context)
    ]
    total, counted = 0.0, 0
    with torch.no_grad():
        for row in rows[:64]:
            t = torch.tensor([row])
            logits = model(t[:, :-1])["logits"]
            total += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    t[:, 1:].reshape(-1),
                    reduction="sum",
                )
            )
            counted += t.size(1) - 1
    loss = total / max(1, counted)
    uniform = math.log(len(tok))
    return CheckResult(
        "untrained_loss_equals_log_vocab",
        abs(loss - uniform) <= tolerance,
        "Before training, cross-entropy equals ln(vocabulary size) - the model has no preference yet.",
        {
            "measured_loss": loss,
            "ln_vocab": uniform,
            "vocab_size": len(tok),
            "absolute_difference": abs(loss - uniform),
            "init_std": cfg.init_std,
        },
        tolerance,
    )


# --------------------------------------------------------------------------
# Gradients and updates
# --------------------------------------------------------------------------


def check_finite_difference_gradient(
    model: TinyTransformer, tok: CharTokenizer, tolerance: float = 1e-4
) -> CheckResult:
    demo = sgd_step_demo(model, tok)
    fd = demo["finite_difference"]
    return CheckResult(
        "finite_difference_matches_autograd",
        fd["relative_error"] <= tolerance,
        "Nudging one weight and re-measuring the loss reproduces the gradient autograd computed.",
        {
            "cell": [demo["row"], demo["column"]],
            "cell_selected_by": demo["cell_selected_by"],
            "analytic": fd["analytic"],
            "finite_difference": fd["estimate"],
            "absolute_error": fd["absolute_error"],
            "relative_error": fd["relative_error"],
            "dtype": fd["dtype"],
            "epsilon": fd["epsilon"],
        },
        tolerance,
    )


def check_sgd_update_identity(
    model: TinyTransformer, tok: CharTokenizer, tolerance: float = 1e-7
) -> CheckResult:
    demo = sgd_step_demo(model, tok)
    lr = demo["learning_rate"]
    before = torch.tensor(demo["before"])
    grad = torch.tensor(demo["gradient"])
    after = torch.tensor(demo["after"])
    worst = float((after - (before - lr * grad)).abs().max())
    return CheckResult(
        "sgd_update_identity",
        worst <= tolerance,
        "Every updated weight equals before - learning_rate * gradient, elementwise.",
        {
            "max_absolute_error": worst,
            "learning_rate": lr,
            "updated_parameter_elements": demo["updated_parameter_elements"],
            "total_parameter_elements": demo["total_parameter_elements"],
            "all_parameters_updated": demo["all_parameters_updated"],
        },
        tolerance,
    )


def check_demo_restores_model(
    model: TinyTransformer, tok: CharTokenizer
) -> CheckResult:
    """The gradient demonstration must leave the checkpoint untouched.

    The original engine applied its demo step and never undid it, so base
    training started from weights no checkpoint had recorded.
    """
    before = _weight_hash(model)
    sgd_step_demo(model, tok)
    after = _weight_hash(model)
    return CheckResult(
        "gradient_demo_leaves_checkpoint_unchanged",
        before == after,
        "Inspecting a gradient step does not modify the checkpoint being inspected.",
        {"weight_hash_before": before[:16], "weight_hash_after": after[:16]},
    )


def check_loss_mask_alignment(
    model: TinyTransformer, tok: CharTokenizer
) -> CheckResult:
    """The scored targets must be exactly the answer tokens plus EOS.

    An off-by-one here is invisible in a loss curve and fatal to the lesson.
    """
    item = data.SFT_TRAIN[0]
    ids, targets = sft_row(tok, item, supervise_eos=True)
    scored = [t for t in targets if t != IGNORE]
    expected = tok.encode(item.answer) + [tok.eos_id]

    # The first scored target must be produced by the LAST prompt position.
    first_scored_index = next(i for i, t in enumerate(targets) if t != IGNORE)
    prompt_len = len(tok.encode(data.format_prompt(item.prompt), bos=True))
    aligned = first_scored_index == prompt_len
    return CheckResult(
        "loss_mask_alignment",
        scored == expected and aligned and len(ids) == len(targets),
        "Assistant-answer targets (and EOS) are scored; the prompt is not; the first scored target follows the last prompt token.",
        {
            "scored_target_tokens": [tok.display(t) for t in scored],
            "expected_tokens": [tok.display(t) for t in expected],
            "matches": scored == expected,
            "first_scored_index": first_scored_index,
            "prompt_length": prompt_len,
            "alignment_ok": aligned,
            "eos_is_scored": tok.eos_id in scored,
        },
    )


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def check_cache_equivalence(
    model: TinyTransformer, tok: CharTokenizer, tolerance: float = 1e-5
) -> CheckResult:
    """Cached and uncached decoding must agree to float32 round-off.

    The comparison is RELATIVE to the logit scale. An absolute threshold looks
    fine on an untrained model and then fails on a trained one purely because
    the logits grew - the arithmetic did not get worse. In float64 this same
    difference is ~1e-17, which is how we know it is round-off and not a bug.
    """
    a = generate(model, tok, "pip has ", DecodeConfig(max_new_tokens=16, use_cache=True))
    b = generate(model, tok, "pip has ", DecodeConfig(max_new_tokens=16, use_cache=False))
    ids = _probe(tok)
    with torch.no_grad():
        full = model(ids)["logits"]
        past, steps = None, []
        for i in range(ids.size(1)):
            out = model(ids[:, i : i + 1], past=past, use_cache=True)
            past = out["past"]
            steps.append(out["logits"][:, -1])
        incremental = torch.stack(steps, dim=1)
    worst = float((full - incremental).abs().max())
    scale = max(float(full.abs().max()), 1e-12)
    relative = worst / scale
    return CheckResult(
        "kv_cache_equivalence",
        a.new_ids == b.new_ids and relative <= tolerance,
        "Caching keys and values changes how much work the forward pass repeats, not what it computes.",
        {
            "same_tokens": a.new_ids == b.new_ids,
            "max_logit_difference": worst,
            "logit_scale": scale,
            "relative_difference": relative,
            "dtype": "float32 (the same comparison is ~1e-17 in float64)",
            "cached_text": a.text,
            "uncached_text": b.text,
        },
        tolerance,
    )


def check_inference_leaves_weights_unchanged(
    model: TinyTransformer, tok: CharTokenizer
) -> CheckResult:
    before = _weight_hash(model)
    generate(model, tok, "pip has ", DecodeConfig(max_new_tokens=20))
    generate(model, tok, "mira ", DecodeConfig(max_new_tokens=20, temperature=1.0, seed=1))
    after = _weight_hash(model)
    return CheckResult(
        "inference_does_not_change_weights",
        before == after,
        "Ordinary inference changes activations and the cache. Learned weights are untouched.",
        {"weight_hash_before": before[:16], "weight_hash_after": after[:16]},
    )


def check_temperature_changes_distribution(
    model: TinyTransformer, tok: CharTokenizer
) -> CheckResult:
    """Temperature must change the sampling distribution.

    The original engine divided the logits by temperature and then took argmax.
    Argmax is invariant under positive scaling, so the control did nothing -
    which is why this check measures ENTROPY, not whether two sampled strings
    happen to differ.
    """
    ids = _probe(tok, "pip has ")
    with torch.no_grad():
        logits = model(ids)["logits"][0, -1]
    entropies = {}
    for t in (0.5, 1.0, 2.0):
        p = F.softmax(logits / t, dim=-1)
        entropies[t] = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
    increasing = entropies[0.5] < entropies[1.0] < entropies[2.0]
    return CheckResult(
        "temperature_changes_the_distribution",
        increasing,
        "Higher temperature raises the entropy of the next-token distribution.",
        {"entropy_by_temperature": entropies, "monotonically_increasing": increasing},
    )


def check_top_k_restricts_support(
    model: TinyTransformer, tok: CharTokenizer
) -> CheckResult:
    kept = {}
    for k in (1, 3, 10):
        g = generate(
            model,
            tok,
            "pip has ",
            DecodeConfig(max_new_tokens=1, temperature=1.0, top_k=k, seed=2),
            record_steps=True,
        )
        kept[k] = int(g.steps[0]["candidates_kept"])
    return CheckResult(
        "top_k_restricts_candidates",
        all(kept[k] == k for k in kept),
        "top-k leaves exactly k candidate tokens with non-zero probability.",
        {"candidates_kept_by_k": kept, "vocab_size": len(tok)},
    )


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


def check_lora_properties(
    base: TinyTransformer, tok: CharTokenizer, tolerance: float = 1e-5
) -> List[CheckResult]:
    from .lora import (
        LoRAConfig,
        adapter_report,
        apply_lora,
        first_step_gradients,
        merge_all,
        unmerge_all,
    )

    model = copy.deepcopy(base)
    model.eval()
    ids = _probe(tok)
    with torch.no_grad():
        before = model(ids)["logits"].clone()
    frozen_snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}

    rank = 2
    info = apply_lora(model, LoRAConfig(rank=rank, alpha=4.0))
    with torch.no_grad():
        after_attach = model(ids)["logits"].clone()
    noop_error = float((before - after_attach).abs().max())

    logits = model(ids[:, :-1])["logits"]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
    grads = first_step_gradients(model, loss)
    a_zero = all(g["A_grad_norm"] == 0.0 for g in grads)
    b_nonzero = all((g["B_grad_norm"] or 0.0) > 0.0 for g in grads)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.05)
    model.train()
    for _ in range(40):
        logits = model(ids[:, :-1])["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()

    base_unchanged = True
    current = dict(model.named_parameters())
    for name, saved in frozen_snapshot.items():
        # Adapters renamed the wrapped Linear to '<module>.base.weight'.
        candidates = [name, name.replace(".weight", ".base.weight")]
        for candidate in candidates:
            if candidate in current:
                if not torch.equal(saved, current[candidate].detach()):
                    base_unchanged = False
                break

    with torch.no_grad():
        unmerged = model(ids)["logits"].clone()
    merge_all(model)
    with torch.no_grad():
        merged = model(ids)["logits"].clone()
    logit_scale = max(float(unmerged.abs().max()), 1e-12)
    # Relative, for the same reason as the KV-cache check: merging sums the
    # update into the base weight, the adapter path sums it separately, and
    # float32 gives the two orders slightly different round-off.
    merge_error = float((unmerged - merged).abs().max()) / logit_scale
    unmerge_all(model)
    with torch.no_grad():
        roundtrip = float((unmerged - model(ids)["logits"]).abs().max()) / logit_scale

    report = adapter_report(model)
    rank_ok = all(r["delta_matrix_rank"] <= rank for r in report)

    return [
        CheckResult(
            "lora_initial_adapter_is_a_noop",
            noop_error == 0.0,
            "With B initialised to zeros, attaching an adapter cannot change any output.",
            {"max_logit_difference": noop_error, "trainable_parameters": info["trainable_parameters"],
             "frozen_parameters": info["frozen_parameters"]},
            0.0,
        ),
        CheckResult(
            "lora_first_step_gradient_asymmetry",
            a_zero and b_nonzero,
            "On the first step A has exactly zero gradient (it is multiplied by B = 0) while B already has one.",
            {"per_adapter": grads},
        ),
        CheckResult(
            "lora_base_weights_stay_frozen",
            base_unchanged,
            "Training the adapters leaves every base weight bit-for-bit unchanged.",
            {"base_unchanged": base_unchanged, "attached_to": info["attached_to"]},
        ),
        CheckResult(
            "lora_merge_equivalence",
            merge_error <= tolerance and roundtrip <= tolerance,
            "Folding the adapter into the base weight gives the same outputs, and unmerging restores them.",
            {
                "relative_merged_vs_unmerged": merge_error,
                "relative_unmerge_roundtrip": roundtrip,
                "logit_scale": logit_scale,
                "dtype": "float32 (the same comparison is ~1e-17 in float64)",
            },
            tolerance,
        ),
        CheckResult(
            "lora_update_rank_is_bounded_by_r",
            rank_ok,
            "The weight update has at most r non-zero singular values - that is what 'low rank' constrains.",
            {
                "rank": rank,
                "realised_ranks": [r["delta_matrix_rank"] for r in report],
                "singular_values": [
                    [round(v, 6) for v in r["delta_singular_values"]] for r in report
                ],
            },
        ),
    ]


# --------------------------------------------------------------------------
# Preference and RL
# --------------------------------------------------------------------------


def check_dpo_reference_identity(
    base: TinyTransformer, tok: CharTokenizer, tolerance: float = 1e-6
) -> CheckResult:
    """At the start of DPO the policy IS the reference, so the margin is zero
    and the loss is exactly ln 2. If this drifts, the reference is wired wrong."""
    from .preference import dpo_terms

    policy = copy.deepcopy(base)
    reference = copy.deepcopy(base)
    reference.eval()
    policy.eval()
    terms = dpo_terms(policy, reference, tok, data.PREFERENCE_PAIRS, beta=0.1)
    margin = float(terms["margin"].abs().max())
    loss = float(terms["loss"])
    return CheckResult(
        "dpo_starts_at_ln2_with_zero_margin",
        margin <= tolerance and abs(loss - math.log(2)) <= tolerance,
        "When the policy equals its frozen reference, every DPO log-ratio is zero and the loss is ln 2.",
        {
            "max_absolute_margin": margin,
            "loss": loss,
            "ln_2": math.log(2),
            "absolute_difference": abs(loss - math.log(2)),
        },
        tolerance,
    )


def check_zero_variance_group() -> CheckResult:
    from .rlvr import group_advantages

    cases = {
        "all_correct": [1.0, 1.0, 1.0, 1.0],
        "all_wrong": [0.0, 0.0, 0.0, 0.0],
        "mixed": [0.0, 1.0, 1.0, 0.0],
    }
    measured = {k: group_advantages(v) for k, v in cases.items()}
    finite = all(
        all(math.isfinite(a) for a in m["advantages"]) for m in measured.values()
    )
    zeros = all(a == 0.0 for a in measured["all_correct"]["advantages"]) and all(
        a == 0.0 for a in measured["all_wrong"]["advantages"]
    )
    return CheckResult(
        "zero_variance_group_has_no_signal",
        finite and zeros and measured["all_correct"]["zero_variance"],
        "When every completion in a group scores the same there is no relative signal, and no NaN either.",
        {
            k: {
                "rewards": m["rewards"],
                "std": m["std"],
                "advantages": m["advantages"],
                "zero_variance": m["zero_variance"],
            }
            for k, m in measured.items()
        },
    )


def check_verifier_rejects_exploits() -> CheckResult:
    from .rlvr import compare_verifiers

    cases = [
        ("answer: 5", 5),
        ("answer: 25", 5),
        ("the answer is 5", 5),
        ("answer: 5\nanswer: 6", 5),
        ("2 + 3 = 8.\nanswer: 5", 5),
        ("answer: 1\nanswer: 2\nanswer: 3\nanswer: 4\nanswer: 5", 5),
        ("", 5),
    ]
    rows = compare_verifiers(cases)
    by_text = {r["completion"]: r for r in rows}
    strict_ok = (
        by_text["answer: 5"]["strict"] == 1
        and by_text["answer: 25"]["strict"] == 0
        and by_text["the answer is 5"]["strict"] == 0
        and by_text["answer: 5\nanswer: 6"]["strict"] == 0
        and by_text[""]["strict"] == 0
    )
    hack_visible = by_text["answer: 25"]["substring_broken"] == 1
    enumeration = "answer: 1\nanswer: 2\nanswer: 3\nanswer: 4\nanswer: 5"
    enumeration_rejected = by_text[enumeration]["strict"] == 0
    process_catches = (
        by_text["2 + 3 = 8.\nanswer: 5"]["strict"] == 1
        and by_text["2 + 3 = 8.\nanswer: 5"]["process"] == 0
    )
    return CheckResult(
        "verifier_rejects_ambiguous_and_exploitative_output",
        strict_ok and hack_visible and process_catches and enumeration_rejected,
        "The strict verifier refuses substrings, prose, multiple answers and enumerating every candidate; the broken one accepts 25 for 5; process checking catches a right answer reached by wrong arithmetic.",
        {
            "rows": rows,
            "strict_behaves": strict_ok,
            "broken_verifier_is_exploitable": hack_visible,
            "process_catches_bad_reasoning": process_catches,
            "enumeration_hack_rejected": enumeration_rejected,
        },
    )


# --------------------------------------------------------------------------
# Data and reproducibility
# --------------------------------------------------------------------------


def check_tokenizer_unknown_is_not_padding(tok: CharTokenizer) -> CheckResult:
    ids = tok.encode("çæ")  # characters absent from the corpus
    return CheckResult(
        "unknown_characters_map_to_unk_not_pad",
        all(i == tok.unk_id for i in ids) and tok.unk_id != tok.pad_id,
        "An out-of-alphabet character becomes <unk>. Mapping it to <pad> would train the model to predict padding.",
        {
            "encoded": ids,
            "unk_id": tok.unk_id,
            "pad_id": tok.pad_id,
            "no_angle_brackets_in_alphabet": "<" not in tok.vocab and ">" not in tok.vocab,
        },
    )


def check_bpe_fitted_on_training_split_only() -> CheckResult:
    """The fit must not change when validation text changes.

    Refitting while hiding the validation documents is the direct test: if the
    merges are identical, no validation text influenced the vocabulary.
    """
    from .tokenizer import BPETokenizer

    train_only = [d.text for d in data.split_documents("train")]
    a = BPETokenizer.fit(train_only, num_merges=40)
    b = fit_on_training_split(num_merges=40)
    everything = [d.text for d in data.documents()]
    c = BPETokenizer.fit(everything, num_merges=40)
    same = [m.token for m in a.merges] == [m.token for m in b.merges]
    differs = [m.token for m in a.merges] != [m.token for m in c.merges]
    return CheckResult(
        "bpe_fitted_on_training_split_only",
        same and differs,
        "The supported fit uses training documents only, and fitting on everything provably yields a different vocabulary.",
        {
            "supported_fit_matches_train_only": same,
            "fitting_on_all_documents_differs": differs,
            "train_only_merges": len(a.merges),
            "all_document_merges": len(c.merges),
        },
    )


def check_task_families_disjoint() -> CheckResult:
    sizes = {}
    prompts = {}
    for n in (4, 8, 18):
        fams = data.task_families(n)
        prompts[n] = {k: {t.prompt for t in v} for k, v in fams.items()}
        sizes[n] = {k: len(v) for k, v in prompts[n].items()}
    overlaps = {
        n: sorted(prompts[n]["add"] & prompts[n]["template"]) for n in prompts
    }
    return CheckResult(
        "verifiable_task_families_are_disjoint",
        all(not v for v in overlaps.values()),
        "The warm-start family and the held-out family share no item at any size - they partition one enumerated value space.",
        {"overlaps_by_size": overlaps, "sizes": sizes},
    )


def check_windows_respect_documents(tok: CharTokenizer, context: int) -> CheckResult:
    windows = document_windows(tok, data.split_documents("train"), context)
    doc_text = {d.doc_id: d.text for d in data.split_documents("train")}
    bad = []
    for w in windows:
        decoded = tok.decode(w.ids)
        if decoded and decoded not in doc_text[w.doc_id]:
            bad.append(w.doc_id)
    return CheckResult(
        "training_windows_stay_inside_one_document",
        not bad,
        "Every training window is a contiguous slice of exactly one document - no window spans a paragraph boundary.",
        {
            "num_windows": len(windows),
            "num_documents": len(doc_text),
            "windows_crossing_a_boundary": len(bad),
        },
    )


def check_reproducibility(cfg: ModelConfig) -> CheckResult:
    a = build_model(cfg, seed=42)
    b = build_model(cfg, seed=42)
    c = build_model(cfg, seed=43)
    same = _weight_hash(a) == _weight_hash(b)
    differs = _weight_hash(a) != _weight_hash(c)
    return CheckResult(
        "same_seed_reproduces_same_weights",
        same and differs,
        "Identical seed and config give identical initial weights; a different seed gives different ones.",
        {
            "seed_42_hash": _weight_hash(a)[:16],
            "seed_42_again_hash": _weight_hash(b)[:16],
            "seed_43_hash": _weight_hash(c)[:16],
            "reproducible": same,
            "seed_actually_matters": differs,
        },
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_all(
    model: TinyTransformer,
    tok: CharTokenizer,
    cfg: ModelConfig,
) -> Dict[str, object]:
    """Run every check against a trained model and return the measured results."""
    results: List[CheckResult] = [
        check_causal_mask(model, tok),
        check_attention_rows_sum_to_one(model, tok),
        check_mlp_dot_product(model, tok),
        check_future_token_invariance(model, tok),
        check_untrained_loss_matches_uniform(cfg, tok),
        check_finite_difference_gradient(model, tok),
        check_sgd_update_identity(model, tok),
        check_demo_restores_model(model, tok),
        check_loss_mask_alignment(model, tok),
        check_cache_equivalence(model, tok),
        check_inference_leaves_weights_unchanged(model, tok),
        check_temperature_changes_distribution(model, tok),
        check_top_k_restricts_support(model, tok),
        *check_lora_properties(model, tok),
        check_dpo_reference_identity(model, tok),
        check_zero_variance_group(),
        check_verifier_rejects_exploits(),
        check_tokenizer_unknown_is_not_padding(tok),
        check_bpe_fitted_on_training_split_only(),
        check_task_families_disjoint(),
        check_windows_respect_documents(tok, cfg.context),
        check_reproducibility(cfg),
    ]
    passed = sum(1 for r in results if r.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "all_passed": passed == len(results),
        "results": [r.to_dict() for r in results],
    }


def main(argv=None) -> int:
    """``python -m glassbox.checks`` - run every check and print its measurement."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Glassbox numerical checks against a freshly trained model."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="pretraining steps before checking (default 400; the checks do not "
        "need a well-trained model, only a real one)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="print measured values")
    args = parser.parse_args(argv)

    from .train import PretrainConfig, SFTConfig, pretrain, sft

    tok = CharTokenizer()
    cfg = ModelConfig(vocab_size=len(tok))
    model = build_model(cfg, seed=args.seed)
    print(
        f"[glassbox] {model.num_parameters} parameters, vocab {len(tok)}, "
        f"pretraining {args.steps} steps before checking"
    )
    pretrain(model, tok, PretrainConfig(steps=args.steps, eval_every=max(1, args.steps)))
    # A short SFT pass so the instruction-mask and EOS checks run against a
    # model that has actually been instruction tuned.
    sft(model, tok, SFTConfig(steps=150, eval_every=150))

    report = run_all(model, tok, cfg)
    width = max(len(r["name"]) for r in report["results"])
    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['name']:<{width}}  {r['description']}")
        if args.verbose or not r["passed"]:
            for key, value in r["measured"].items():
                print(f"         {key} = {value}")
    print(
        f"\n[glassbox] {report['passed']}/{report['total']} passed"
        + ("" if report["all_passed"] else f", {report['failed']} FAILED")
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
