"""Run every stage once and write a single self-describing run artifact.

The artifact is the contract between the engine and the interface. The
interface never computes a model number and never invents one: if a field is
absent the page must say so rather than substitute an illustrative value.

Checkpoint lineage produced here:

    initial  random weights
      |
      +-- base            pretraining on the eight paragraphs
            |
            +-- sft       full-model instruction tuning
            |     |
            |     +-- dpo          preference optimisation
            |     +-- rl           warm start + GRPO on verifiable tasks
            |
            +-- lora      frozen base + rank-r adapters on the same SFT data

``sft`` and ``lora`` train on the SAME data with the SAME objective and differ
only in WHICH parameters may move. That is the distinction the course exists to
make, so they are produced as siblings rather than as a sequence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from . import checks, data, evaluate
from .generate import GREEDY, DecodeConfig, generate
from .lora import LoRAConfig, adapter_report, apply_lora
from .model import ModelConfig, TinyTransformer, build_model
from .preference import DPOConfig, RewardConfig, train_dpo, train_reward_model
from .rlvr import GRPOConfig, compare_verifiers, grpo_train
from .tokenizer import CharTokenizer, fit_on_training_split
from .train import (
    PretrainConfig,
    SFTConfig,
    document_windows,
    pretrain,
    sft,
    sgd_step_demo,
)

SCHEMA_VERSION = "1.0"

#: The prompt every checkpoint is inspected on, so tensors are comparable.
INSPECTION_PROMPT = "pip has "

#: Prompts shown as unedited sample output on every checkpoint.
SHOWCASE_PROMPTS = [
    "pip has ",
    data.format_prompt("how many red stones does pip have?"),
    data.format_prompt("how many stones are there in all?"),
]

DISPLAY_DECIMALS = 7


def _round(obj):
    """Round for storage, and say so. Display rounding is not the computation."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return round(obj, DISPLAY_DECIMALS)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    return obj


@dataclass
class RunConfig:
    seed: int = 42
    model: ModelConfig = None
    pretrain: PretrainConfig = field(default_factory=lambda: PretrainConfig(steps=2000))
    sft: SFTConfig = field(default_factory=lambda: SFTConfig(steps=400))
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    warm_start: SFTConfig = field(
        default_factory=lambda: SFTConfig(steps=600, learning_rate=3e-3, eval_every=300)
    )
    grpo: GRPOConfig = field(
        default_factory=lambda: GRPOConfig(
            iterations=60,
            group_size=8,
            learning_rate=3e-3,
            inner_epochs=4,
            kl_coefficient=0.02,
            temperature=1.0,
            top_k=6,
            max_new_tokens=12,
        )
    )


#: Field names the server reconstructs a ModelConfig from. Kept here so the
#: sidecar this module writes and the loader that reads it cannot drift.
MODEL_CONFIG_FIELDS = (
    "vocab_size", "width", "heads", "mlp_width", "layers", "context",
    "init_std", "dropout",
)


def _state_dict_copy(model: TinyTransformer) -> Dict[str, object]:
    """Detached CPU copy of every tensor, taken at this exact moment.

    Without the clone, later training would mutate tensors a previous
    checkpoint is still holding a reference to, and every saved checkpoint
    would end up identical to the last one.
    """
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _capture_checkpoint(
    model: TinyTransformer,
    tok: CharTokenizer,
    label: str,
    parent: Optional[str],
    stage: str,
) -> Dict[str, object]:
    """Tensors, a gradient step, and unedited samples for one checkpoint.

    The gradient step is recomputed HERE, for THIS checkpoint. The original
    engine aliased one recording into all three checkpoints while the interface
    let the learner switch between them.
    """
    model.eval()
    ids = torch.tensor([tok.encode(INSPECTION_PROMPT, bos=True)])
    out = model(ids, capture=True)
    cap = out["capture"]

    demo = sgd_step_demo(model, tok, prompt=INSPECTION_PROMPT)

    samples = {}
    for prompt in SHOWCASE_PROMPTS:
        g = generate(model, tok, prompt, DecodeConfig(max_new_tokens=24), record_steps=True)
        samples[prompt] = g.to_dict()

    tokens = tok.encode(INSPECTION_PROMPT, bos=True)
    return {
        "checkpoint_id": label,
        "parent": parent,
        "stage": stage,
        "trace_source": "live_pytorch_engine",
        "inspection": {
            "prompt": INSPECTION_PROMPT,
            "input_ids": tokens,
            "input_tokens": [tok.display(i) for i in tokens],
            "target_ids": tokens[1:],
            "target_tokens": [tok.display(i) for i in tokens[1:]],
        },
        "operations": cap.order,
        "tensors": _round(cap.records),
        "gradient_step": _round(demo),
        "weights": {
            "mlp_up": _round(model.blocks[0].mlp_up.weight.detach().tolist()),
            "mlp_up_bias": _round(model.blocks[0].mlp_up.bias.detach().tolist()),
            "mlp_down": _round(model.blocks[0].mlp_down.weight.detach().tolist()),
            "mlp_down_bias": _round(model.blocks[0].mlp_down.bias.detach().tolist()),
        },
        "samples": _round(samples),
    }


def build_run(
    cfg: Optional[RunConfig] = None,
    verbose: bool = True,
    weights_out: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Run every stage and return the JSON artifact.

    Trained tensors are NOT placed in the returned dict - it must stay
    JSON-serialisable, because callers hand it straight to ``json.dumps`` and to
    the browser. Pass a dict as ``weights_out`` to receive them separately.
    """
    started = time.time()
    cfg = cfg or RunConfig()
    tok = CharTokenizer()
    if cfg.model is None:
        cfg.model = ModelConfig(vocab_size=len(tok))

    def say(msg: str) -> None:
        if verbose:
            print(f"[glassbox] {msg}", flush=True)

    checkpoints: Dict[str, Dict[str, object]] = {}
    stages: Dict[str, object] = {}
    #: label -> state_dict, written to a sidecar so a server can actually
    #: decode from the checkpoint a caller asks for.
    weights: Dict[str, Dict[str, object]] = {}

    # -- initial ----------------------------------------------------------
    say("building model")
    model = build_model(cfg.model, seed=cfg.seed)
    checkpoints["initial"] = _capture_checkpoint(
        model, tok, "initial", None, "random initialisation"
    )
    weights["initial"] = _state_dict_copy(model)

    # -- pretraining ------------------------------------------------------
    say(f"pretraining for {cfg.pretrain.steps} steps")
    stages["pretrain"] = pretrain(model, tok, cfg.pretrain)
    num_train_docs = len(data.split_documents("train"))
    checkpoints["base"] = _capture_checkpoint(
        model,
        tok,
        "base",
        "initial",
        # Counted, not spelled out: the server lets a learner submit their own
        # corpus, and a label reading "eight paragraphs" beside a three-paragraph
        # run is exactly the kind of stale claim this rebuild exists to remove.
        f"pretrained on {num_train_docs} paragraph{'' if num_train_docs == 1 else 's'}",
    )
    weights["base"] = _state_dict_copy(model)
    base_state = copy.deepcopy(model.state_dict())

    # -- full SFT ---------------------------------------------------------
    say(f"instruction tuning for {cfg.sft.steps} steps")
    sft_model = build_model(cfg.model, seed=cfg.seed)
    sft_model.load_state_dict(base_state)
    stages["sft"] = sft(sft_model, tok, cfg.sft)
    checkpoints["sft"] = _capture_checkpoint(
        sft_model, tok, "sft", "base", "full-model instruction tuning"
    )
    weights["sft"] = _state_dict_copy(sft_model)

    # -- LoRA SFT (sibling of full SFT) -----------------------------------
    say("instruction tuning with LoRA adapters on a frozen base")
    lora_model = build_model(cfg.model, seed=cfg.seed)
    lora_model.load_state_dict(base_state)
    lora_info = apply_lora(lora_model, cfg.lora)
    lora_sft_cfg = SFTConfig(
        steps=cfg.sft.steps,
        batch_size=cfg.sft.batch_size,
        # Adapters start as a no-op, so they need a larger step than full
        # fine-tuning to move the same distance.
        learning_rate=cfg.sft.learning_rate * 5,
        warmup_steps=cfg.sft.warmup_steps,
        seed=cfg.sft.seed,
        eval_every=cfg.sft.eval_every,
    )
    stages["lora_sft"] = sft(lora_model, tok, lora_sft_cfg)
    stages["lora_sft"]["lora"] = lora_info
    stages["lora_sft"]["adapters"] = _round(adapter_report(lora_model))
    checkpoints["lora"] = _capture_checkpoint(
        lora_model, tok, "lora", "base", "frozen base + LoRA adapters, same SFT data"
    )
    weights["lora"] = _state_dict_copy(lora_model)

    # -- reward model + DPO ----------------------------------------------
    say("training a reward model on preference pairs")
    reward = train_reward_model(sft_model, tok, data.PREFERENCE_PAIRS, cfg.reward)
    reward.pop("model", None)
    stages["reward_model"] = _round(reward)

    say(f"DPO for {cfg.dpo.steps} steps")
    dpo_model = build_model(cfg.model, seed=cfg.seed)
    dpo_model.load_state_dict(sft_model.state_dict())
    dpo_out = train_dpo(dpo_model, tok, data.PREFERENCE_PAIRS, cfg.dpo)
    dpo_out.pop("reference", None)
    stages["dpo"] = _round(dpo_out)
    checkpoints["dpo"] = _capture_checkpoint(
        dpo_model, tok, "dpo", "sft", "preference optimisation (DPO)"
    )
    weights["dpo"] = _state_dict_copy(dpo_model)

    # -- warm start + GRPO ------------------------------------------------
    say("warm-starting the policy on task formatting")
    rl_model = build_model(cfg.model, seed=cfg.seed)
    rl_model.load_state_dict(sft_model.state_dict())
    warm_items = [
        data.Instruction(t.prompt, f"answer: {t.answer}", "warm_start")
        for t in data.task_families(18)["add"]
    ]
    stages["warm_start"] = sft(rl_model, tok, cfg.warm_start, items=warm_items)
    stages["warm_start"]["rationale"] = (
        "A policy that never produces a parseable answer gets reward 0 on every "
        "rollout, every group has zero variance, and there is nothing for a "
        "group-relative objective to learn from. That is a prerequisite failure "
        "to fix, not evidence that RL cannot work."
    )
    checkpoints["warm_start"] = _capture_checkpoint(
        rl_model, tok, "warm_start", "sft", "SFT on task answer formatting"
    )
    weights["warm_start"] = _state_dict_copy(rl_model)

    held_out_tasks = data.task_families(8)["template"]
    say(f"GRPO for {cfg.grpo.iterations} iterations on held-out task values")
    grpo_out = grpo_train(rl_model, tok, held_out_tasks, cfg.grpo)
    stages["grpo"] = _round(grpo_out)
    stages["grpo"]["tasks"] = [
        {"prompt": t.prompt, "hidden_answer": t.answer, "family": t.family, "recipe": t.recipe}
        for t in held_out_tasks
    ]
    checkpoints["rl"] = _capture_checkpoint(
        rl_model, tok, "rl", "warm_start", "GRPO against a strict verifier"
    )
    weights["rl"] = _state_dict_copy(rl_model)

    # -- evaluation -------------------------------------------------------
    say("evaluating every checkpoint under one pinned decoding protocol")
    models = {
        "initial": model if False else build_model(cfg.model, seed=cfg.seed),
        "base": None,
        "sft": sft_model,
        "lora": lora_model,
        "dpo": dpo_model,
        "warm_start": None,
        "rl": rl_model,
    }
    base_model = build_model(cfg.model, seed=cfg.seed)
    base_model.load_state_dict(base_state)
    models["base"] = base_model
    warm_model = build_model(cfg.model, seed=cfg.seed)
    # warm_start weights were overwritten by GRPO, so re-derive them for the
    # comparison table rather than reporting the RL model twice.
    warm_model.load_state_dict(sft_model.state_dict())
    sft(warm_model, tok, cfg.warm_start, items=warm_items)
    models["warm_start"] = warm_model

    reports = []
    for label in ("initial", "base", "sft", "lora", "dpo", "warm_start", "rl"):
        reports.append(evaluate.checkpoint_report(models[label], tok, label, GREEDY))
    stages["evaluation"] = {
        "reports": _round(reports),
        "comparison": _round(evaluate.compare_checkpoints(reports)),
        "baselines": _round(evaluate.baseline_losses(tok, cfg.model.context)),
    }

    # -- tokenizer lesson -------------------------------------------------
    bpe = fit_on_training_split(60)
    sample = "pip has two red stones."
    stages["tokenizers"] = {
        "character": {
            "vocab": tok.vocab,
            "size": len(tok),
            "example": sample,
            "ids": tok.encode(sample),
            "token_count": len(tok.encode(sample)),
        },
        "subword": {
            "size": len(bpe),
            "num_merges": len(bpe.merges),
            "fitted_on": "training split only",
            "merges": [
                {"rank": m.rank, "left": m.left, "right": m.right,
                 "token": m.token, "count": m.count}
                for m in bpe.merges
            ],
            "example": sample,
            "pieces": [p for w in sample.split(" ") for p in bpe.tokenize_word(w)],
            "token_count": len(bpe.encode(sample)),
        },
        "note": (
            "Ids are arbitrary categories. The subword vocabulary was fitted on "
            "training documents only; fitting on everything provably changes the "
            "merge list, which is checked in checks.py."
        ),
    }

    # -- verifier trap table ---------------------------------------------
    stages["verifier_table"] = compare_verifiers(
        [
            ("answer: 5", 5),
            ("answer: 25", 5),
            ("2 + 3 = 5.\nanswer: 5", 5),
            ("2 + 3 = 8.\nanswer: 5", 5),
            ("the answer is 5", 5),
            ("answer: 5\nanswer: 6", 5),
            ("answer: 1\nanswer: 2\nanswer: 3\nanswer: 4\nanswer: 5", 5),
            ("answer: 5\nthanks!", 5),
            ("", 5),
        ]
    )

    # -- checks -----------------------------------------------------------
    say("running numerical checks")
    check_results = checks.run_all(sft_model, tok, cfg.model)

    train_windows = document_windows(tok, data.split_documents("train"), cfg.model.context)
    elapsed = time.time() - started

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": cfg.seed,
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "display_decimals": DISPLAY_DECIMALS,
        "model": cfg.model.to_dict(),
        "parameters": build_model(cfg.model, seed=cfg.seed).num_parameters,
        "corpus_sha256": {
            "train": data.corpus_sha256("train"),
            "validation": data.corpus_sha256("validation"),
        },
        "dataset_sha256": data.dataset_sha256(),
        "tokenizer_sha256": hashlib.sha256(
            "\u0000".join(tok.vocab).encode("utf-8")
        ).hexdigest(),
        "documents": [
            {"doc_id": d.doc_id, "split": d.split, "sha256": d.sha256[:16], "length": len(d.text)}
            for d in data.documents()
        ],
        "num_train_windows": len(train_windows),
        "elapsed_seconds": round(elapsed, 2),
        "stage_configs": {
            "pretrain": cfg.pretrain.to_dict(),
            "sft": cfg.sft.to_dict(),
            "lora": cfg.lora.to_dict(),
            "reward_model": cfg.reward.to_dict(),
            "dpo": cfg.dpo.to_dict(),
            "warm_start": cfg.warm_start.to_dict(),
            "grpo": cfg.grpo.to_dict(),
        },
        "honest_limits": [
            "A two-paragraph validation set is a teaching diagnostic, not a benchmark.",
            "Twelve instruction examples measure memorisation. Held-out accuracy is reported separately and is low.",
            "Six preference pairs cannot establish that a reward model generalises.",
            "This model has a few thousand parameters. It demonstrates mechanisms, not capability.",
        ],
    }
    manifest["run_id"] = "run_" + hashlib.sha256(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "elapsed_seconds"},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:12]

    if weights_out is not None:
        weights_out.clear()
        weights_out.update(
            {
                "run_id": manifest["run_id"],
                "seed": cfg.seed,
                "vocab": list(tok.vocab),
                "model_config_kwargs": {
                    k: v
                    for k, v in cfg.model.to_dict().items()
                    if k in MODEL_CONFIG_FIELDS
                },
                "lora_config_kwargs": {
                    "rank": cfg.lora.rank,
                    "alpha": cfg.lora.alpha,
                    "targets": tuple(cfg.lora.targets),
                },
                "states": weights,
            }
        )

    return {
        "manifest": manifest,
        "data": {
            "paragraphs": [d.text for d in data.split_documents("train")],
            "validation_paragraphs": [d.text for d in data.split_documents("validation")],
            "instruction_families": {
                name: [{"prompt": i.prompt, "answer": i.answer} for i in items]
                for name, items in data.instruction_families().items()
            },
            "preference_pairs": [
                {
                    "prompt": p.prompt,
                    "chosen": p.chosen,
                    "rejected": p.rejected,
                    "reason": p.reason,
                    "rejected_is_factual": p.rejected_is_factual,
                }
                for p in data.PREFERENCE_PAIRS
            ],
            "task_families": {
                name: [
                    {"prompt": t.prompt, "hidden_answer": t.answer, "recipe": t.recipe}
                    for t in items
                ]
                for name, items in data.task_families(8).items()
            },
            "vocab": tok.vocab,
            "vocab_display": [tok.display(i) for i in range(len(tok))],
            "prompt_template": data.PROMPT_TEMPLATE,
        },
        "checkpoints": checkpoints,
        "checkpoint_order": ["initial", "base", "sft", "lora", "dpo", "warm_start", "rl"],
        "stages": stages,
        "checks": _round(check_results),
    }


def write_run(
    run: Dict[str, object],
    out_dir: Path,
    weights: Optional[Dict[str, object]] = None,
    set_latest: bool = True,
) -> Path:
    """Write the JSON artifact, plus a torch sidecar holding the trained tensors.

    The JSON carries recorded numbers for the interface. The sidecar carries the
    weights, so a server can decode from the checkpoint a caller actually asked
    for rather than answering from one checkpoint and relabelling the reply.
    The whole model is a few thousand parameters, so all seven fit in ~150 KB.

    ``set_latest`` guards the reference artifact. ``latest.json`` is what the app
    embeds and what ``glassbox.report`` regenerates the blueprint from, so a
    quick 300-step experiment that overwrote it would silently rewrite every
    documented number to the experiment's values. A named
    ``<run_id>.json`` is always written; the ``latest`` alias is not.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run["manifest"]["run_id"]
    # Defensive: a private key must never reach the JSON or the browser.
    payload = {k: v for k, v in run.items() if not k.startswith("_")}
    sidecar = weights if weights else None
    blob = json.dumps(payload, separators=(",", ":"))

    path = out_dir / f"{run_id}.json"
    path.write_text(blob)
    if set_latest:
        (out_dir / "latest.json").write_text(blob)

    if sidecar is not None:
        import torch

        torch.save(sidecar, out_dir / f"{run_id}.weights.pt")
        if set_latest:
            # latest.json is a copy, so its sidecar needs the same alias.
            torch.save(sidecar, out_dir / "latest.weights.pt")
    return path


#: The configuration the documented numbers were measured at. A run that differs
#: from this in any way is an experiment, not the reference.
def is_reference_run(cfg: "RunConfig", tok: CharTokenizer) -> bool:
    reference = RunConfig(model=ModelConfig(vocab_size=len(tok)))
    return (
        cfg.seed == reference.seed
        and cfg.model.to_dict() == reference.model.to_dict()
        and cfg.pretrain.to_dict() == reference.pretrain.to_dict()
        and cfg.sft.to_dict() == reference.sft.to_dict()
        and cfg.grpo.to_dict() == reference.grpo.to_dict()
    )


DATA_START = '<script id="run-data" type="application/json">'
DATA_END = "</script>"


def inject_into_html(run: Dict[str, object], html_path: Path) -> None:
    """Replace the run-data block in the app, escaping the one sequence that
    could break out of a script element.

    ``</script>`` inside JSON string data would terminate the element early.
    Training text is learner-supplied, so this is a real injection path and not
    a hypothetical one; escaping ``<`` as ``\\u003c`` is the standard fix and
    JSON.parse restores the original characters.
    """
    html = html_path.read_text()
    if DATA_START not in html:
        raise ValueError(f"{html_path} has no run-data block")
    serialisable = {k: v for k, v in run.items() if not k.startswith("_")}
    payload = json.dumps(serialisable, separators=(",", ":")).replace("<", "\\u003c")
    head, rest = html.split(DATA_START, 1)
    _, tail = rest.split(DATA_END, 1)
    html_path.write_text(head + DATA_START + payload + DATA_END + tail)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build a Glassbox run artifact.")
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--sft-steps", type=int, default=400)
    parser.add_argument("--grpo-iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "runs")
    parser.add_argument(
        "--html",
        type=Path,
        default=Path(__file__).parent.parent / "app" / "index.html",
        help="App file whose run-data block should be replaced.",
    )
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument(
        "--set-latest",
        choices=("auto", "always", "never"),
        default="auto",
        help="write runs/latest.json and update the app. 'auto' (the default) does "
        "so only when the configuration matches the documented reference run, so "
        "a reduced-step experiment cannot silently replace the numbers the "
        "README and blueprint quote.",
    )
    args = parser.parse_args(argv)

    tok = CharTokenizer()
    cfg = RunConfig(
        seed=args.seed,
        model=ModelConfig(
            vocab_size=len(tok), width=args.width, layers=args.layers, heads=args.heads
        ),
        pretrain=PretrainConfig(steps=args.pretrain_steps),
        sft=SFTConfig(steps=args.sft_steps),
    )
    cfg.grpo.iterations = args.grpo_iterations

    reference = is_reference_run(cfg, tok)
    if args.set_latest == "always":
        set_latest = True
    elif args.set_latest == "never":
        set_latest = False
    else:
        set_latest = reference

    weights: Dict[str, object] = {}
    run = build_run(cfg, weights_out=weights)
    path = write_run(run, args.out, weights=weights, set_latest=set_latest)
    size_mb = path.stat().st_size / 1e6
    print(f"[glassbox] wrote {path} ({size_mb:.2f} MB)")
    if set_latest:
        print(f"[glassbox] updated {args.out / 'latest.json'}")
    else:
        print(
            f"[glassbox] left {args.out / 'latest.json'} alone: this run is not the "
            "reference configuration, so it must not replace the documented numbers. "
            "Pass --set-latest always to override."
        )

    c = run["checks"]
    print(f"[glassbox] checks: {c['passed']}/{c['total']} passed")
    for r in c["results"]:
        if not r["passed"]:
            print(f"[glassbox]   FAILED: {r['name']}  {r['measured']}")

    inject = not args.no_html and set_latest
    if inject and args.html.exists():
        inject_into_html(run, args.html)
        print(f"[glassbox] injected run data into {args.html}")
    elif inject:
        print(f"[glassbox] note: {args.html} not found; skipped injection")
    elif not args.no_html:
        print(
            f"[glassbox] left {args.html} alone for the same reason; the named "
            "artifact above is complete and can be inspected directly."
        )

    return 0 if c["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
