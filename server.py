"""Glassbox HTTP server: serves a run artifact and answers live questions about it.

Design rules this file is built to satisfy, each one a correction of a defect in
the module it replaces:

* **Nothing trains at import time.** Importing this module touches only the
  standard library, FastAPI and pydantic. ``glassbox`` - and therefore torch -
  is imported lazily, so ``import server`` costs about 0.65s (measured)
  instead of the minutes the previous module spent training inside its own
  import. A run artifact is *loaded*, never recomputed, unless asked for.
* **Every accepted parameter is used or rejected.** ``checkpoint`` decodes from
  that checkpoint's weights. ``temperature``/``top_k``/``top_p``/``seed`` go
  into a real :class:`~glassbox.generate.DecodeConfig`. ``row``/``column``/
  ``epsilon`` drive an actual finite difference. A parameter that cannot be
  honoured produces an error, never a silently ignored field.
* **No raw JSON in a script element.** ``json.dumps`` does not escape ``<``, so
  a learner-supplied paragraph containing ``</script>`` closes the element and
  executes. :func:`render_app_html` applies the same escaping rule as
  :func:`glassbox.export.inject_into_html` and is covered by a regression test.
* **Mutation is serialised and state is swapped atomically.** A build holds a
  non-blocking lock; readers take one local reference to an immutable snapshot,
  so a concurrent build cannot tear what a reader is looking at.

Run it with the interpreter that actually has torch and fastapi::

    /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 server.py --port 8077

See ``API.md`` for a verified curl example per endpoint.
"""

from __future__ import annotations

import contextlib
import json
import platform
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

SERVER_VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
APP_HTML = ROOT / "app" / "index.html"

#: Checkpoint lineage produced by :func:`glassbox.export.build_run`, in order.
CHECKPOINT_NAMES: Tuple[str, ...] = (
    "initial",
    "base",
    "sft",
    "lora",
    "dpo",
    "warm_start",
    "rl",
)

#: Checkpoints whose model has LoRA adapters attached, so the module structure
#: must be rebuilt with :func:`glassbox.lora.apply_lora` before loading weights.
LORA_CHECKPOINTS: Tuple[str, ...] = ("lora",)

#: ``ModelConfig`` constructor arguments. ``ModelConfig.to_dict()`` also reports
#: derived and descriptive fields, which are not accepted by ``__init__``.
MODEL_CONFIG_FIELDS: Tuple[str, ...] = (
    "vocab_size",
    "width",
    "heads",
    "mlp_width",
    "layers",
    "context",
    "init_std",
    "dropout",
)

#: Input bounds. Every one of these is enforced, and the numbers appear in the
#: 422 body so a rejected request says what would have been accepted.
MAX_PARAGRAPHS = 24
MIN_PARAGRAPH_CHARS = 8
MAX_PARAGRAPH_CHARS = 1200
MAX_CORPUS_CHARS = 8000
MAX_INSTRUCTIONS = 24
MAX_PROMPT_CHARS = 400

#: ``run_<12 hex>`` is what :func:`glassbox.export.build_run` produces. Pinning
#: the shape is what keeps ``/api/run/{run_id}`` from reading ``../../secrets``.
RUN_ID_RE = re.compile(r"^run_[0-9a-f]{6,32}$")

MAX_MODEL_CACHE = 8


# --------------------------------------------------------------------------
# Loaded state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointWeights:
    """Every checkpoint's trained tensors, kept in memory.

    The whole model is 3048 parameters, so all seven checkpoints together are
    about 146 KB on disk (measured). Holding them is what lets ``/api/infer``
    honour ``checkpoint`` by actually decoding from those weights instead of
    answering from one checkpoint and relabelling the reply.
    """

    run_id: str
    seed: int
    #: Full vocabulary including the four special tokens, as the run used it.
    vocab: Tuple[str, ...]
    model_config_kwargs: Dict[str, Any]
    lora_config_kwargs: Dict[str, Any]
    states: Dict[str, Dict[str, Any]]
    source: str

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(n for n in CHECKPOINT_NAMES if n in self.states)


@dataclass(frozen=True)
class LoadedRun:
    """One immutable snapshot. Swapped in by a single attribute assignment."""

    run_id: str
    artifact: Dict[str, Any]
    source: str
    weights: Optional[CheckpointWeights]
    loaded_at: float

    @property
    def manifest(self) -> Dict[str, Any]:
        value = self.artifact.get("manifest")
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class RunSummary:
    """What a build actually did, for the POST /api/run response and /api/status."""

    run_id: str
    requested_at: float
    wall_seconds: float
    engine_seconds: Optional[float]
    corpus: str
    paragraph_count: int
    corpus_characters: int
    instructions: str
    instruction_count: int
    vocabulary_size: int
    parameters: int
    stage_configs: Dict[str, Any]
    checks: Dict[str, Any]
    checkpoints: List[str]
    saved_to: Optional[str]
    weights_saved_to: Optional[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requested_at": self.requested_at,
            "wall_seconds": round(self.wall_seconds, 2),
            "engine_seconds": self.engine_seconds,
            "corpus": self.corpus,
            "paragraph_count": self.paragraph_count,
            "corpus_characters": self.corpus_characters,
            "instructions": self.instructions,
            "instruction_count": self.instruction_count,
            "vocabulary_size": self.vocabulary_size,
            "parameters": self.parameters,
            "stage_configs": self.stage_configs,
            "checks": self.checks,
            "checkpoints": self.checkpoints,
            "saved_to": self.saved_to,
            "weights_saved_to": self.weights_saved_to,
            "notes": self.notes,
        }


#: The active snapshot. Replaced wholesale, never mutated in place.
_STATE: Optional[LoadedRun] = None

#: Held for the duration of a build. Non-blocking: a second build is refused
#: with 409 rather than queued, so the caller learns the truth immediately.
_RUN_LOCK = threading.Lock()
_RUN_ACTIVE: Optional[Dict[str, Any]] = None
_LAST_RUN: Optional[RunSummary] = None

#: Builds kept addressable by run_id even when they were not written to disk.
_RUN_CACHE: "Dict[str, LoadedRun]" = {}
_RUN_CACHE_ORDER: List[str] = []

#: (run_id, label) -> materialised model. Bounded; rebuilding one is cheap.
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_MODEL_CACHE_ORDER: List[Tuple[str, str]] = []


# --------------------------------------------------------------------------
# Artifact and weight loading (lazy: torch is not imported until needed)
# --------------------------------------------------------------------------


def _artifact_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _weights_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.weights.pt"


def _model_config_kwargs(model_section: Dict[str, Any]) -> Dict[str, Any]:
    return {k: model_section[k] for k in MODEL_CONFIG_FIELDS if k in model_section}


def _alphabet_from_vocab(vocab: Sequence[str]) -> List[str]:
    """Drop the special tokens; ``CharTokenizer`` re-adds them itself."""
    from glassbox.tokenizer import SPECIAL_TOKENS

    return [t for t in vocab if t not in SPECIAL_TOKENS]


def load_weights_file(path: Path) -> Optional[CheckpointWeights]:
    """Load checkpoint tensors written by a previous build, or return None.

    ``weights_only=True`` keeps this from being a pickle-execution path: the
    payload is tensors, strings and numbers, and nothing in it needs to be able
    to import or call anything.
    """
    if not path.exists():
        return None
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=True)
    return CheckpointWeights(
        run_id=str(blob["run_id"]),
        seed=int(blob["seed"]),
        vocab=tuple(blob["vocab"]),
        model_config_kwargs=dict(blob["model_config_kwargs"]),
        lora_config_kwargs=dict(blob["lora_config_kwargs"]),
        states={str(k): dict(v) for k, v in blob["states"].items()},
        source=str(path.relative_to(ROOT)),
    )


def save_weights_file(weights: CheckpointWeights, path: Path) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_id": weights.run_id,
            "seed": weights.seed,
            "vocab": list(weights.vocab),
            "model_config_kwargs": weights.model_config_kwargs,
            "lora_config_kwargs": weights.lora_config_kwargs,
            "states": weights.states,
        },
        path,
    )
    return path


def load_run_from_disk(run_id: str = "latest") -> Optional[LoadedRun]:
    """Read ``runs/<run_id>.json`` plus its sidecar weights if they match.

    Weights from a *different* run are not used. Decoding ``sft`` from another
    run's tensors and labelling the output ``sft`` is exactly the kind of quiet
    mislabelling this server exists to avoid, so a mismatch is reported instead.
    """
    path = _artifact_path(run_id)
    if not path.exists():
        return None
    artifact = json.loads(path.read_text())
    manifest = artifact.get("manifest") or {}
    actual_id = str(manifest.get("run_id") or run_id)

    weights = None
    for candidate in (_weights_path(run_id), _weights_path(actual_id)):
        if candidate.exists():
            found = load_weights_file(candidate)
            if found is not None and found.run_id == actual_id:
                weights = found
            break

    return LoadedRun(
        run_id=actual_id,
        artifact=artifact,
        source=str(path.relative_to(ROOT)),
        weights=weights,
        loaded_at=time.time(),
    )


def _remember(run: LoadedRun) -> None:
    _RUN_CACHE[run.run_id] = run
    if run.run_id in _RUN_CACHE_ORDER:
        _RUN_CACHE_ORDER.remove(run.run_id)
    _RUN_CACHE_ORDER.append(run.run_id)
    while len(_RUN_CACHE_ORDER) > 4:
        _RUN_CACHE.pop(_RUN_CACHE_ORDER.pop(0), None)


def ensure_loaded() -> Optional[LoadedRun]:
    """Load ``runs/latest.json`` once, on first need. Never trains."""
    global _STATE
    if _STATE is None:
        loaded = load_run_from_disk("latest")
        if loaded is not None:
            _STATE = loaded
            _remember(loaded)
    return _STATE


def require_loaded() -> LoadedRun:
    run = ensure_loaded()
    if run is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "no run artifact is loaded",
                "looked_for": str(_artifact_path("latest").relative_to(ROOT)),
                "remedy": [
                    "POST /api/run to build one in this process, or",
                    "run `python3 -m glassbox.export` to write runs/latest.json",
                ],
            },
        )
    return run


# --------------------------------------------------------------------------
# Materialising a checkpoint into a runnable model
# --------------------------------------------------------------------------


def _cache_model(key: Tuple[str, str], model: Any) -> None:
    _MODEL_CACHE[key] = model
    if key in _MODEL_CACHE_ORDER:
        _MODEL_CACHE_ORDER.remove(key)
    _MODEL_CACHE_ORDER.append(key)
    while len(_MODEL_CACHE_ORDER) > MAX_MODEL_CACHE:
        _MODEL_CACHE.pop(_MODEL_CACHE_ORDER.pop(0), None)


def materialise(run: LoadedRun, label: str) -> Tuple[Any, Any, str]:
    """Return ``(model, tokenizer, provenance)`` for one checkpoint.

    ``initial`` needs no training: ``build_model(cfg, seed)`` is deterministic,
    so random initialisation can always be reproduced from the manifest alone.
    Every other checkpoint needs the tensors a build produced. If they are not
    in memory this raises 409 and says how to get them, rather than quietly
    answering from whatever weights happen to be around.
    """
    if label not in CHECKPOINT_NAMES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"unknown checkpoint {label!r}",
                "valid_checkpoints": list(CHECKPOINT_NAMES),
            },
        )

    cached = _MODEL_CACHE.get((run.run_id, label))
    if cached is not None:
        return cached[0], cached[1], cached[2]

    from glassbox.lora import LoRAConfig, apply_lora
    from glassbox.model import ModelConfig, build_model
    from glassbox.tokenizer import CharTokenizer

    weights = run.weights
    if weights is not None and label in weights.states:
        tok = CharTokenizer(alphabet=_alphabet_from_vocab(weights.vocab))
        cfg = ModelConfig(**weights.model_config_kwargs)
        model = build_model(cfg, seed=weights.seed)
        if label in LORA_CHECKPOINTS and weights.lora_config_kwargs:
            apply_lora(model, LoRAConfig(**weights.lora_config_kwargs))
        model.load_state_dict(weights.states[label])
        provenance = f"trained tensors from {weights.source}"
    elif label == "initial":
        manifest = run.manifest
        vocab = (run.artifact.get("data") or {}).get("vocab") or []
        if not vocab:
            raise HTTPException(
                status_code=503,
                detail={"error": "loaded artifact has no vocabulary to rebuild from"},
            )
        tok = CharTokenizer(alphabet=_alphabet_from_vocab(vocab))
        cfg = ModelConfig(**_model_config_kwargs(manifest.get("model") or {}))
        model = build_model(cfg, seed=int(manifest.get("seed", 42)))
        provenance = (
            "rebuilt from the manifest config and seed; random initialisation is "
            "deterministic, so no trained tensors are needed"
        )
    else:
        available = list(weights.labels) if weights else ["initial"]
        raise HTTPException(
            status_code=409,
            detail={
                "error": f"no weights in memory for checkpoint {label!r}",
                "available_now": available,
                "why": (
                    "the loaded artifact records measurements, not tensors. "
                    "'initial' is reproducible from the seed; the trained "
                    "checkpoints need a build in this process."
                ),
                "remedy": (
                    "POST /api/run to train them (about two minutes at the "
                    "default step counts), or restore a runs/<run_id>.weights.pt "
                    "sidecar written by an earlier save=true run"
                ),
            },
        )

    model.eval()
    _cache_model((run.run_id, label), (model, tok, provenance))
    return model, tok, provenance


# --------------------------------------------------------------------------
# HTML rendering: the injection fix
# --------------------------------------------------------------------------


def escape_for_script(obj: Any) -> str:
    """Serialise ``obj`` for embedding inside a ``<script>`` element.

    ``json.dumps`` does not escape ``<``. A paragraph containing
    ``</script><script>alert(1)</script>`` therefore closes the data element and
    runs. Escaping ``<`` as ``\\u003c`` is the fix used by
    :func:`glassbox.export.inject_into_html`; ``JSON.parse`` restores the
    original characters, so the data is unchanged and only the transport is safe.
    """
    return json.dumps(obj, separators=(",", ":")).replace("<", "\\u003c")


def render_app_html(html: str, run: Optional[LoadedRun]) -> Tuple[str, str]:
    """Return ``(html, note)`` with the run artifact injected into the data block.

    The file on disk is never written. The app shell is authored separately and
    may be mid-edit, so a missing or half-written data block degrades to serving
    the page unchanged with a note, not to a traceback.
    """
    from glassbox.export import DATA_END, DATA_START

    if DATA_START not in html:
        return html, f"served unchanged: no {DATA_START!r} block in the app file"
    head, rest = html.split(DATA_START, 1)
    if DATA_END not in rest:
        return html, "served unchanged: data block is not closed (file may be mid-edit)"
    _, tail = rest.split(DATA_END, 1)
    if run is None:
        payload = escape_for_script(
            {"error": "no run artifact loaded", "server_version": SERVER_VERSION}
        )
        return head + DATA_START + payload + DATA_END + tail, "injected an empty-run marker"
    payload = escape_for_script(run.artifact)
    return head + DATA_START + payload + DATA_END + tail, f"injected run {run.run_id}"


# --------------------------------------------------------------------------
# Pipeline instrumentation
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _instrumented_pipeline(
    paragraphs: Optional[List[str]],
    instructions: Optional[List[Tuple[str, str]]],
    store: Dict[str, Dict[str, Any]],
) -> Iterator[None]:
    """Point the engine at learner text and tee each checkpoint's tensors out.

    Two narrow, reversible hooks, both restored in ``finally``:

    * ``data._TRAIN_PARAGRAPHS`` / ``data.SFT_TRAIN`` are the single source the
      rest of the package reads from - ``documents()``, ``all_text()``, the
      corpus hashes, ``pretrain()`` and ``sft()`` all go through them. Replacing
      their contents is therefore what makes ``CharTokenizer()`` rebuild its
      alphabet from the submitted text. Without that the learner's characters
      would encode to ``<unk>``, which is the defect being fixed.
    * ``export._capture_checkpoint`` is called once per checkpoint with the live
      model. Wrapping it copies out the state_dict at exactly the moment the
      artifact records that checkpoint, so the tensors and the reported numbers
      cannot disagree. The alternative - a second pipeline written here - would
      double the runtime and could drift from the engine it claims to mirror.

    The hooks are process-global, so this only ever runs while the build lock is
    held.
    """
    from glassbox import data as gdata
    from glassbox import export as gexport

    if not hasattr(gexport, "_capture_checkpoint"):  # pragma: no cover - guard
        raise RuntimeError(
            "glassbox.export._capture_checkpoint is gone; the weight-capture hook "
            "in server.py needs updating before checkpoint inference can work"
        )

    original = gexport._capture_checkpoint
    saved_paragraphs = list(gdata._TRAIN_PARAGRAPHS)
    saved_instructions = list(gdata.SFT_TRAIN)

    def capture(model: Any, tok: Any, label: str, parent: Any, stage: str) -> Any:
        result = original(model, tok, label, parent, stage)
        store[label] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return result

    try:
        if paragraphs is not None:
            gdata._TRAIN_PARAGRAPHS[:] = list(paragraphs)
        if instructions is not None:
            gdata.SFT_TRAIN[:] = [
                gdata.Instruction(prompt, answer, "train")
                for prompt, answer in instructions
            ]
        gexport._capture_checkpoint = capture
        yield
    finally:
        gexport._capture_checkpoint = original
        gdata._TRAIN_PARAGRAPHS[:] = saved_paragraphs
        gdata.SFT_TRAIN[:] = saved_instructions


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class InstructionIn(BaseModel):
    """One instruction pair. Both keys are required, so a dict missing one is a
    422 naming the missing field instead of a KeyError turning into a 500."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=200)
    answer: str = Field(min_length=1, max_length=200)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paragraphs: Optional[List[str]] = Field(
        default=None,
        description=(
            f"Replace the pretraining corpus. 1-{MAX_PARAGRAPHS} paragraphs, "
            f"{MIN_PARAGRAPH_CHARS}-{MAX_PARAGRAPH_CHARS} characters each, "
            f"{MAX_CORPUS_CHARS} characters total. The character tokenizer is "
            "rebuilt from the submitted text."
        ),
    )
    instructions: Optional[List[InstructionIn]] = Field(
        default=None,
        description="Replace the SFT training family. Held-out eval families are fixed.",
    )
    pretrain_steps: int = Field(default=2000, ge=1, le=4000)
    sft_steps: int = Field(default=400, ge=1, le=2000)
    warm_start_steps: int = Field(default=600, ge=1, le=2000)
    dpo_steps: int = Field(default=200, ge=1, le=1000)
    grpo_iterations: int = Field(default=60, ge=1, le=200)
    grpo_group_size: int = Field(default=8, ge=2, le=16)
    lora_rank: int = Field(default=2, ge=1, le=8)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    save: bool = Field(
        default=False,
        description=(
            "Write runs/<run_id>.json, runs/latest.json and the weight sidecar. "
            "Default false so a short experiment cannot overwrite the reference run."
        ),
    )

    @field_validator("paragraphs")
    @classmethod
    def _check_paragraphs(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        if not value:
            raise ValueError("paragraphs was empty; omit the field to use the default corpus")
        if len(value) > MAX_PARAGRAPHS:
            raise ValueError(f"at most {MAX_PARAGRAPHS} paragraphs, got {len(value)}")
        total = 0
        for i, text in enumerate(value):
            if not isinstance(text, str):
                raise ValueError(f"paragraph {i} is {type(text).__name__}, not a string")
            stripped = text.strip()
            if len(stripped) < MIN_PARAGRAPH_CHARS:
                raise ValueError(
                    f"paragraph {i} has {len(stripped)} characters; "
                    f"at least {MIN_PARAGRAPH_CHARS} are needed to form a training window"
                )
            if len(text) > MAX_PARAGRAPH_CHARS:
                raise ValueError(
                    f"paragraph {i} has {len(text)} characters; "
                    f"the limit is {MAX_PARAGRAPH_CHARS}"
                )
            total += len(text)
        if total > MAX_CORPUS_CHARS:
            raise ValueError(
                f"corpus is {total} characters; the limit is {MAX_CORPUS_CHARS}"
            )
        return value

    @field_validator("instructions")
    @classmethod
    def _check_instructions(
        cls, value: Optional[List[InstructionIn]]
    ) -> Optional[List[InstructionIn]]:
        if value is None:
            return None
        if not value:
            raise ValueError("instructions was empty; omit the field to use the default family")
        if len(value) > MAX_INSTRUCTIONS:
            raise ValueError(f"at most {MAX_INSTRUCTIONS} instructions, got {len(value)}")
        return value


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    checkpoint: str = Field(
        default="sft",
        description="Decoded from this checkpoint's weights. Unknown names are a 400.",
    )
    max_new_tokens: int = Field(default=24, ge=1, le=128)
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="0 means greedy. Any positive value samples.",
    )
    top_k: int = Field(default=0, ge=0, le=512)
    top_p: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)
    use_cache: bool = True
    stop_at_eos: bool = True
    apply_prompt_template: bool = Field(
        default=False,
        description="Wrap the prompt in the instruction template the SFT data used.",
    )
    record_steps: bool = Field(
        default=False, description="Return the top-5 distribution at each decoding step."
    )


class GradientCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: str = Field(default="sft")
    matrix: str = Field(
        default="blocks.0.mlp_up.weight",
        description="A named 2-D parameter. GET /api/parameters lists them.",
    )
    row: int = Field(default=0, ge=0, description="Negative indices are rejected, not wrapped.")
    column: int = Field(default=0, ge=0)
    epsilon: float = Field(
        default=1e-4,
        gt=0.0,
        le=0.1,
        description="Central-difference step. Must be positive; 0 would divide by zero.",
    )
    prompt: str = Field(default="pip has ", min_length=1, max_length=200)
    learning_rate: float = Field(default=0.05, gt=0.0, le=10.0)


# --------------------------------------------------------------------------
# The gradient check
# --------------------------------------------------------------------------


def gradient_check(
    model: Any,
    tok: Any,
    req: GradientCheckRequest,
) -> Dict[str, Any]:
    """Compare autograd against a central difference at the requested cell.

    Same shape of argument as :func:`glassbox.train.sgd_step_demo`, with one
    difference that is the whole point of the endpoint: that function picks the
    cell with the largest gradient, whereas this one checks the cell the caller
    asked about, after proving it exists.

    Both sides are computed in float64. A float32 forward pass with epsilon
    1e-4 leaves a relative error around 1e-2, which is noise rather than a
    check, so the float32 analytic gradient is reported alongside as a
    measurement of that gap rather than as the answer.
    """
    import copy

    import torch
    import torch.nn.functional as F

    from glassbox.train import sgd_step_demo

    params = dict(model.named_parameters())
    if req.matrix not in params:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"no parameter named {req.matrix!r}",
                "available_matrices": [
                    {"name": n, "shape": list(p.shape)}
                    for n, p in params.items()
                    if p.dim() == 2
                ],
            },
        )
    target = params[req.matrix]
    if target.dim() != 2:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"{req.matrix!r} has {target.dim()} dimensions, not 2",
                "shape": list(target.shape),
                "hint": "row/column address a matrix; pick a 2-D parameter",
            },
        )
    rows, columns = int(target.shape[0]), int(target.shape[1])
    if req.row >= rows or req.column >= columns:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "cell is outside the matrix",
                "matrix": req.matrix,
                "shape": [rows, columns],
                "requested": {"row": req.row, "column": req.column},
                "valid_row_range": [0, rows - 1],
                "valid_column_range": [0, columns - 1],
            },
        )

    ids = torch.tensor([tok.encode(req.prompt, bos=True)])
    if ids.shape[1] < 2:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "prompt is too short to form a next-token loss",
                "encoded_length": int(ids.shape[1]),
                "minimum": 2,
            },
        )
    inputs, targets = ids[:, :-1], ids[:, 1:]

    def loss_of(m: Any) -> Any:
        logits = m(inputs)["logits"]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )

    # -- analytic gradient, float64 --------------------------------------
    fd_model = copy.deepcopy(model).double()
    fd_model.eval()
    fd_param = dict(fd_model.named_parameters())[req.matrix]
    temporarily_enabled = not bool(fd_param.requires_grad)
    fd_param.requires_grad_(True)
    fd_model.zero_grad(set_to_none=True)
    loss_base_64 = loss_of(fd_model)
    loss_base_64.backward()
    analytic = float(fd_param.grad[req.row, req.column])
    grad_abs = fd_param.grad.abs()
    flat = int(torch.argmax(grad_abs))
    largest_row, largest_column = divmod(flat, columns)

    # -- central difference, float64 -------------------------------------
    with torch.no_grad():
        original_value = float(fd_param[req.row, req.column])
        fd_param[req.row, req.column] = original_value + req.epsilon
        loss_plus = float(loss_of(fd_model))
        fd_param[req.row, req.column] = original_value - req.epsilon
        loss_minus = float(loss_of(fd_model))
        fd_param[req.row, req.column] = original_value
    estimate = (loss_plus - loss_minus) / (2.0 * req.epsilon)
    denominator = max(abs(estimate), abs(analytic), 1e-12)
    relative_error = abs(estimate - analytic) / denominator

    # -- the same gradient in float32, for contrast ----------------------
    f32_model = copy.deepcopy(model)
    f32_model.eval()
    f32_param = dict(f32_model.named_parameters())[req.matrix]
    f32_param.requires_grad_(True)
    f32_model.zero_grad(set_to_none=True)
    loss_base_32 = loss_of(f32_model)
    loss_base_32.backward()
    analytic_float32 = float(f32_param.grad[req.row, req.column])

    # -- one real SGD step, via the engine's own recorder ----------------
    step = sgd_step_demo(
        model,
        tok,
        prompt=req.prompt,
        learning_rate=req.learning_rate,
        matrix_name=req.matrix,
        fd_epsilon=req.epsilon,
    )

    # A relative criterion alone is wrong for a near-zero gradient: the adapter
    # matrices of a fresh LoRA checkpoint have gradients around 1e-7, and the
    # float64 rounding of a loss around 3.8 is then a large FRACTION of a tiny
    # difference while being absolutely negligible. So either criterion passing
    # is enough, and both are reported rather than hidden behind one boolean.
    relative_tolerance = 1e-6
    absolute_tolerance = 1e-9
    absolute_error = abs(estimate - analytic)
    passes = relative_error < relative_tolerance or absolute_error < absolute_tolerance
    return {
        "matrix": req.matrix,
        "shape": [rows, columns],
        "row": req.row,
        "column": req.column,
        "prompt": req.prompt,
        "parameter_is_trainable": not temporarily_enabled,
        "frozen_note": step.get("frozen_note"),
        "gradient_requires_grad_temporarily_enabled": temporarily_enabled,
        "finite_difference": {
            "dtype": "float64",
            "method": "central difference, (L(w+eps) - L(w-eps)) / (2*eps)",
            "epsilon": req.epsilon,
            "loss_base": float(loss_base_64),
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "estimate": estimate,
            "analytic": analytic,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "criterion": (
                "relative_error < 1e-6 OR absolute_error < 1e-9; the absolute "
                "arm is what keeps a correct near-zero gradient from being "
                "reported as a failure"
            ),
            "passes_relative": relative_error < relative_tolerance,
            "passes_absolute": absolute_error < absolute_tolerance,
            "passes": passes,
        },
        "float32_comparison": {
            "dtype": "float32",
            "analytic": analytic_float32,
            "difference_from_float64_analytic": abs(analytic_float32 - analytic),
            "note": (
                "the check itself runs in float64; this is what the same gradient "
                "looks like at the precision the model trains in"
            ),
        },
        "largest_gradient_cell": {
            "row": largest_row,
            "column": largest_column,
            "analytic": float(fd_param.grad[largest_row, largest_column]),
            "note": "the cell sgd_step_demo would have checked on its own",
        },
        "sgd_step": {
            "learning_rate": req.learning_rate,
            "cell_before": float(target[req.row, req.column]),
            "cell_delta": -req.learning_rate * analytic,
            "cell_after": float(target[req.row, req.column]) - req.learning_rate * analytic,
            "loss_before": step["loss_before"],
            "loss_after_full_step": step["loss_after"],
            "updated_parameter_elements": step["updated_parameter_elements"],
            "frozen_parameter_elements": step["frozen_parameter_elements"],
            "note": (
                "loss_after_full_step moves every trainable parameter, not only "
                "this cell; the model is restored afterwards"
            ),
        },
    }


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the artifact and its weights when the app starts being served.

    This is deliberately not import-time work: a plain ``import server`` reaches
    none of it, which is what keeps the module cheap to import and reload.
    Nothing here trains.
    """
    run = ensure_loaded()
    if run is None:
        print("[glassbox] no runs/latest.json; POST /api/run to build one", flush=True)
    else:
        have = ", ".join(run.weights.labels) if run.weights else "initial (derived)"
        print(
            f"[glassbox] loaded {run.run_id} from {run.source}; weights: {have}",
            flush=True,
        )
    yield


app = FastAPI(
    title="Glassbox server",
    description=(
        "Serves a Glassbox run artifact and answers live questions about it. "
        "Nothing is trained unless POST /api/run is called."
    ),
    version=SERVER_VERSION,
    lifespan=_lifespan,
)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the app shell with the run artifact injected into its data block."""
    if not APP_HTML.exists():
        body = (
            "<!doctype html><title>Glassbox: app shell missing</title>"
            "<h1>The app shell is not built yet</h1>"
            f"<p>Expected a file at <code>{APP_HTML}</code>. "
            "It is authored separately from this server; the server does not "
            "create or edit it.</p>"
            "<p>The API is up. Try "
            "<a href='/api/status'>/api/status</a>, "
            "<a href='/api/checks'>/api/checks</a> or "
            "<a href='/docs'>/docs</a>.</p>"
        )
        return HTMLResponse(content=body, status_code=503)
    try:
        html = APP_HTML.read_text()
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail={"error": "could not read the app shell", "reason": str(exc)}
        )
    rendered, note = render_app_html(html, ensure_loaded())
    return HTMLResponse(content=rendered, headers={"X-Glassbox-Run-Data": note})


@app.get("/api/status")
def status() -> Dict[str, Any]:
    """What is loaded, what is running, and what versions are actually in use."""
    import torch

    run = ensure_loaded()
    manifest = run.manifest if run else {}
    checks = (run.artifact.get("checks") if run else None) or {}
    weights = run.weights if run else None

    app_html_note: Optional[str] = None
    has_block = False
    if APP_HTML.exists():
        from glassbox.export import DATA_START

        try:
            has_block = DATA_START in APP_HTML.read_text()
            if not has_block:
                app_html_note = "file exists but has no run-data block"
        except OSError as exc:
            app_html_note = f"unreadable: {exc}"

    return {
        "server_version": SERVER_VERSION,
        "loaded": run is not None,
        "run_id": run.run_id if run else None,
        "artifact_source": run.source if run else None,
        "artifact_seed": manifest.get("seed"),
        "run_in_progress": _RUN_ACTIVE is not None,
        "active_run": dict(_RUN_ACTIVE) if _RUN_ACTIVE else None,
        "last_run": _LAST_RUN.to_dict() if _LAST_RUN else None,
        "checks": {
            "total": checks.get("total"),
            "passed": checks.get("passed"),
            "failed": checks.get("failed"),
            "all_passed": checks.get("all_passed"),
        },
        "model": {
            "parameters": manifest.get("parameters"),
            "vocab_size": (manifest.get("model") or {}).get("vocab_size"),
            "context": (manifest.get("model") or {}).get("context"),
        },
        "checkpoints": {
            "names": list(CHECKPOINT_NAMES),
            "weights_in_memory": list(weights.labels) if weights else [],
            "weights_source": weights.source if weights else None,
            "derivable_without_weights": ["initial"],
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "run_built_with_python": manifest.get("python_version"),
            "run_built_with_torch": manifest.get("torch_version"),
        },
        "app_html": {
            "path": str(APP_HTML.relative_to(ROOT)),
            "exists": APP_HTML.exists(),
            "has_data_block": has_block,
            "note": app_html_note,
        },
        "stored_runs": sorted(
            p.stem for p in RUNS_DIR.glob("run_*.json")
        ) if RUNS_DIR.exists() else [],
        "runs_in_memory": list(_RUN_CACHE_ORDER),
    }


@app.get("/api/checks")
def get_checks() -> Dict[str, Any]:
    """The numerical checks recorded by the loaded run."""
    run = require_loaded()
    checks = run.artifact.get("checks")
    if not isinstance(checks, dict):
        raise HTTPException(
            status_code=503,
            detail={"error": "the loaded artifact has no checks section", "run_id": run.run_id},
        )
    return {"run_id": run.run_id, "source": run.source, **checks}


@app.get("/api/parameters")
def get_parameters(checkpoint: str = "initial") -> Dict[str, Any]:
    """Every named parameter of one checkpoint, with shapes.

    This is what makes ``/api/gradient-check`` usable without guessing: the
    matrix names and their real bounds come from the model itself.
    """
    run = require_loaded()
    model, _, provenance = materialise(run, checkpoint)
    params = list(model.named_parameters())
    return {
        "run_id": run.run_id,
        "checkpoint": checkpoint,
        "provenance": provenance,
        "total_parameters": model.num_parameters,
        "matrices": [
            {
                "name": name,
                "shape": list(p.shape),
                "trainable": bool(p.requires_grad),
                "rows": int(p.shape[0]),
                "columns": int(p.shape[1]),
            }
            for name, p in params
            if p.dim() == 2
        ],
        "vectors": [
            {"name": name, "shape": list(p.shape), "trainable": bool(p.requires_grad)}
            for name, p in params
            if p.dim() != 2
        ],
    }


@app.get("/api/run/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    """One stored artifact. ``latest`` means the run this server has loaded."""
    if run_id == "latest":
        return require_loaded().artifact
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"{run_id!r} is not a run id",
                "expected": "latest, or run_ followed by 6-32 hex characters",
            },
        )
    cached = _RUN_CACHE.get(run_id)
    if cached is not None:
        return cached.artifact
    path = _artifact_path(run_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"no stored run {run_id!r}",
                "in_memory": list(_RUN_CACHE_ORDER),
                "on_disk": sorted(p.stem for p in RUNS_DIR.glob("run_*.json")),
            },
        )
    return json.loads(path.read_text())


@app.post("/api/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    """Decode from one checkpoint's weights under an explicit decoding protocol."""
    run = require_loaded()
    model, tok, provenance = materialise(run, req.checkpoint)

    from glassbox.data import format_prompt
    from glassbox.generate import DecodeConfig, generate

    prompt = format_prompt(req.prompt) if req.apply_prompt_template else req.prompt
    cfg = DecodeConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        seed=req.seed,
        use_cache=req.use_cache,
        stop_at_eos=req.stop_at_eos,
    )
    try:
        result = generate(model, tok, prompt, cfg, record_steps=req.record_steps)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - surfaced, not swallowed
        raise HTTPException(
            status_code=500,
            detail={"error": "generation failed", "reason": f"{type(exc).__name__}: {exc}"},
        )

    unknown = sorted({c for c in prompt if c not in tok.stoi})
    return {
        "run_id": run.run_id,
        "checkpoint": req.checkpoint,
        "weights_provenance": provenance,
        "prompt_as_sent": req.prompt,
        "prompt_as_encoded": prompt,
        "prompt_template_applied": req.apply_prompt_template,
        "protocol": result.protocol,
        "decode_config": cfg.to_dict(),
        "deterministic": cfg.is_greedy or req.seed is not None,
        "text": result.text,
        "new_ids": result.new_ids,
        "prompt_ids": result.prompt_ids,
        "stopped_on_eos": result.stopped_on_eos,
        "hit_length_limit": result.hit_length_limit,
        "steps": result.steps,
        "unknown_characters": unknown,
        "unknown_character_note": (
            None
            if not unknown
            else (
                "these characters are not in this run's vocabulary and were "
                "encoded as <unk>; rebuild the tokenizer by POSTing paragraphs "
                "that contain them"
            )
        ),
        "vocabulary_size": len(tok),
    }


@app.post("/api/gradient-check")
def post_gradient_check(req: GradientCheckRequest) -> Dict[str, Any]:
    """Finite-difference the requested cell and compare it with autograd."""
    run = require_loaded()
    model, tok, provenance = materialise(run, req.checkpoint)
    try:
        result = gradient_check(model, tok, req)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - surfaced, not swallowed
        raise HTTPException(
            status_code=500,
            detail={"error": "gradient check failed", "reason": f"{type(exc).__name__}: {exc}"},
        )
    return {
        "run_id": run.run_id,
        "checkpoint": req.checkpoint,
        "weights_provenance": provenance,
        **result,
    }


@app.post("/api/run")
def post_run(req: RunRequest) -> Dict[str, Any]:
    """Build a fresh run in this process, then serve it.

    Synchronous and slow on purpose - at the default step counts this is about
    two minutes of real training on CPU, and the response reports the measured
    wall time rather than pretending a cached number is fresh. Declared ``def``
    rather than ``async def`` so FastAPI runs it in a worker thread and
    ``/api/status`` keeps answering while it works.
    """
    global _STATE, _RUN_ACTIVE, _LAST_RUN

    if not _RUN_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "a run is already in progress",
                "active_run": dict(_RUN_ACTIVE) if _RUN_ACTIVE else None,
                "why": (
                    "training mutates process-global engine state, so runs are "
                    "serialised rather than interleaved"
                ),
            },
        )

    started = time.time()
    paragraphs = list(req.paragraphs) if req.paragraphs else None
    instructions = (
        [(i.prompt, i.answer) for i in req.instructions] if req.instructions else None
    )
    _RUN_ACTIVE = {
        "started_at": started,
        "pretrain_steps": req.pretrain_steps,
        "sft_steps": req.sft_steps,
        "grpo_iterations": req.grpo_iterations,
        "custom_corpus": paragraphs is not None,
        "custom_instructions": instructions is not None,
        "seed": req.seed,
    }
    try:
        from glassbox.export import RunConfig, build_run, write_run
        from glassbox.lora import LoRAConfig
        from glassbox.preference import DPOConfig
        from glassbox.tokenizer import CharTokenizer
        from glassbox.train import PretrainConfig, SFTConfig

        states: Dict[str, Dict[str, Any]] = {}
        with _instrumented_pipeline(paragraphs, instructions, states):
            # Built after the hooks are active, so the alphabet comes from the
            # text this run will actually train on.
            tok = CharTokenizer()
            vocabulary_size = len(tok)

            cfg = RunConfig(seed=req.seed)
            cfg.model = None  # build_run sizes it from the rebuilt tokenizer
            cfg.pretrain = PretrainConfig(steps=req.pretrain_steps, seed=req.seed + 1192)
            cfg.sft = SFTConfig(steps=req.sft_steps, seed=req.seed + 57)
            cfg.warm_start = SFTConfig(
                steps=req.warm_start_steps,
                learning_rate=3e-3,
                eval_every=max(1, req.warm_start_steps // 2),
                seed=req.seed + 57,
            )
            cfg.dpo = DPOConfig(steps=req.dpo_steps, seed=req.seed + 5)
            cfg.lora = LoRAConfig(rank=req.lora_rank, alpha=float(2 * req.lora_rank))
            cfg.grpo.iterations = req.grpo_iterations
            cfg.grpo.group_size = req.grpo_group_size
            cfg.grpo.seed = req.seed + 3

            artifact = build_run(cfg, verbose=True)
            lora_kwargs = {
                "rank": cfg.lora.rank,
                "alpha": cfg.lora.alpha,
                "targets": tuple(cfg.lora.targets),
            }

        manifest = artifact["manifest"]
        run_id = str(manifest["run_id"])
        weights = CheckpointWeights(
            run_id=run_id,
            seed=int(manifest["seed"]),
            vocab=tuple(artifact["data"]["vocab"]),
            model_config_kwargs=_model_config_kwargs(manifest["model"]),
            lora_config_kwargs=lora_kwargs,
            states=states,
            source="trained in this process",
        )

        saved_to: Optional[str] = None
        weights_saved_to: Optional[str] = None
        if req.save:
            saved_to = str(write_run(artifact, RUNS_DIR).relative_to(ROOT))
            weights_saved_to = str(
                save_weights_file(weights, _weights_path(run_id)).relative_to(ROOT)
            )
            save_weights_file(weights, RUNS_DIR / "latest.weights.pt")

        loaded = LoadedRun(
            run_id=run_id,
            artifact=artifact,
            source="POST /api/run",
            weights=weights,
            loaded_at=time.time(),
        )
        # One assignment: a reader holding the previous snapshot keeps a
        # consistent artifact-plus-weights pair rather than a half-swapped one.
        _STATE = loaded
        _remember(loaded)

        notes: List[str] = []
        if paragraphs is not None:
            notes.append(
                f"tokenizer rebuilt from the submitted corpus: vocabulary is "
                f"{vocabulary_size} tokens"
            )
        if instructions is not None:
            notes.append(
                "the SFT training family was replaced; the repeat/paraphrase/"
                "heldout evaluation families are fixed and were not"
            )
        if not req.save:
            notes.append(
                "not written to disk (save=false); it is served from memory and "
                "addressable at /api/run/" + run_id
            )
        if req.pretrain_steps < 2000 or req.sft_steps < 400:
            notes.append(
                "fewer steps than the reference run, so the losses here are not "
                "comparable with the documented numbers"
            )
        checks = artifact.get("checks") or {}

        summary = RunSummary(
            run_id=run_id,
            requested_at=started,
            wall_seconds=time.time() - started,
            engine_seconds=manifest.get("elapsed_seconds"),
            corpus="submitted" if paragraphs is not None else "default",
            paragraph_count=len(artifact["data"]["paragraphs"]),
            corpus_characters=sum(len(p) for p in artifact["data"]["paragraphs"]),
            instructions="submitted" if instructions is not None else "default",
            instruction_count=len(artifact["data"]["instruction_families"]["train"]),
            vocabulary_size=vocabulary_size,
            parameters=int(manifest["parameters"]),
            stage_configs=manifest.get("stage_configs") or {},
            checks={
                "total": checks.get("total"),
                "passed": checks.get("passed"),
                "failed": checks.get("failed"),
                "all_passed": checks.get("all_passed"),
            },
            checkpoints=list(weights.labels),
            saved_to=saved_to,
            weights_saved_to=weights_saved_to,
            notes=notes,
        )
        _LAST_RUN = summary
        return summary.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "the run failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "state": "unchanged; the previously loaded run is still being served",
            },
        )
    finally:
        _RUN_ACTIVE = None
        _RUN_LOCK.release()


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the Glassbox app and API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
