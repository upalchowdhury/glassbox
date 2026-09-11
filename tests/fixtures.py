"""Shared fixtures. This module is the suite's fast mode.

Three decisions make the whole suite cheap:

1. The pretrained model is built ONCE per process and cached as a state dict.
   Every test that needs a trained model gets a fresh copy of those weights
   instead of training again. Training is the only expensive thing here.
2. ``PRETRAIN_STEPS`` is 400, not the 2000 a real run uses. 400 is not
   arbitrary: it is the smallest round step count whose validation loss
   (2.1794) still beats the bigram Laplace baseline (2.3025), which
   ``test_baselines`` asserts. At 300 steps the validation loss is 2.3269 and
   that assertion would be false - the model would not have earned its
   parameters yet, and a suite that passed anyway would be lying.
3. Tests that must mutate a model mutate a copy. ``trained_model()`` always
   returns a fresh object, so no test can leak state into another.

Environment knobs, all optional:

    GLASSBOX_TEST_PRETRAIN_STEPS   override the 400 (raising it is safe;
                                   lowering it below ~350 breaks the baseline
                                   assertion, which is the point of that test)
    GLASSBOX_SKIP_SLOW=1           skip the end-to-end export test, the only
                                   test that takes more than a second
"""

from __future__ import annotations

import copy
import hashlib
import os
import threading
from types import SimpleNamespace
from typing import Dict, Optional

import torch

from glassbox.model import ModelConfig, TinyTransformer, build_model
from glassbox.tokenizer import CharTokenizer
from glassbox.train import PretrainConfig, pretrain

#: Seed used for every model the suite builds, matching the exported run.
SEED = 42

PRETRAIN_STEPS = int(os.environ.get("GLASSBOX_TEST_PRETRAIN_STEPS", "400"))

_TRUTHY = {"1", "true", "yes", "on"}
SKIP_SLOW = os.environ.get("GLASSBOX_SKIP_SLOW", "").strip().lower() in _TRUTHY

SKIP_SLOW_REASON = "GLASSBOX_SKIP_SLOW is set (this test trains every stage once)"

#: Reentrant: _trained() holds the lock while _train_once() calls tokenizer(),
#: which takes it again. A plain Lock deadlocks there.
_lock = threading.RLock()
_cache: Dict[str, object] = {}


def tokenizer() -> CharTokenizer:
    """The character tokenizer. Construction is pure, but cache it anyway."""
    with _lock:
        if "tok" not in _cache:
            _cache["tok"] = CharTokenizer()
        return _cache["tok"]  # type: ignore[return-value]


def model_config(**overrides) -> ModelConfig:
    return ModelConfig(vocab_size=len(tokenizer()), **overrides)


def untrained_model(seed: int = SEED) -> TinyTransformer:
    return build_model(model_config(), seed=seed)


def _train_once() -> Dict[str, object]:
    tok = tokenizer()
    cfg = model_config()
    model = build_model(cfg, seed=SEED)
    history = pretrain(
        model,
        tok,
        PretrainConfig(steps=PRETRAIN_STEPS, eval_every=max(1, PRETRAIN_STEPS // 2)),
    )
    return {
        "state": copy.deepcopy(model.state_dict()),
        "pretrain": history,
        "steps": PRETRAIN_STEPS,
    }


def _trained() -> Dict[str, object]:
    with _lock:
        if "trained" not in _cache:
            _cache["trained"] = _train_once()
        return _cache["trained"]  # type: ignore[return-value]


def trained_model() -> TinyTransformer:
    """A freshly constructed model carrying the cached pretrained weights.

    Always a new object: a test may train, merge adapters into, or otherwise
    mutate what it gets back without affecting any other test.
    """
    model = build_model(model_config(), seed=SEED)
    model.load_state_dict(_trained()["state"])  # type: ignore[index]
    model.eval()
    return model


def pretrain_history() -> Dict[str, object]:
    return _trained()["pretrain"]  # type: ignore[return-value]


def weight_hash(model: torch.nn.Module) -> str:
    """Bit-exact hash of every stored tensor, for 'did this change at all'."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(p.detach().contiguous().view(-1).numpy().tobytes())
    return h.hexdigest()


class ForcedTokenModel(torch.nn.Module):
    """A stand-in model whose argmax is always one chosen token id.

    Used for the two generation properties that must hold regardless of what a
    real model has learned: that EOS ends decoding, and that the prompt handed
    to the prefill pass is the WHOLE prompt. Both are properties of the decode
    loop, and pinning them to a stub means they cannot start passing or failing
    because training changed.

    It records every input it was called with, so a test can assert on the
    sequence lengths the decoder actually fed forward.
    """

    def __init__(self, vocab_size: int, context: int, forced_id: int):
        super().__init__()
        self.cfg = SimpleNamespace(context=context, vocab_size=vocab_size)
        self.vocab_size = vocab_size
        self.forced_id = forced_id
        self.calls = []

    def forward(
        self,
        input_ids: torch.Tensor,
        capture: bool = False,
        past: Optional[object] = None,
        use_cache: bool = False,
    ):
        batch, length = input_ids.shape
        self.calls.append(input_ids[0].tolist())
        logits = torch.zeros(batch, length, self.vocab_size)
        logits[..., self.forced_id] = 10.0
        return {
            "logits": logits,
            "capture": None,
            # The real model returns one (k, v) pair per layer. The stub ignores
            # the cache contents; it only has to be a non-None value so the
            # decode loop takes its cached branch.
            "past": [] if use_cache else None,
        }
