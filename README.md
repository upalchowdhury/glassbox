# Glassbox

**Live:** https://upalchowdhury.github.io/glassbox/

A character-level transformer with **3048 parameters** that you can inspect at every
step: every attention row, every MLP dot product, every gradient. It is taken through
the whole modern post-training sequence — pretraining, supervised fine-tuning, LoRA,
reward modelling, DPO, and GRPO against a verifier — as one reproducible pipeline.
Every number the app displays is computed by that pipeline and written into a run
artifact, never hard-coded into the page.

---

## What is real and what is not

Read this before you read any number below.

**Real.** The model, the autograd, the optimiser, the KV cache, the LoRA decomposition,
the Bradley–Terry reward loss, the DPO objective, the clipped group-relative policy
objective, and the verifiers are all genuine implementations. 26 numerical checks
assert things that would break if they were not: attention rows sum to 1, a
finite-difference gradient matches autograd, cached and uncached decoding agree,
merging a LoRA adapter is equivalent to not merging it, a zero-variance reward group
carries no gradient signal. All 26 pass.

**Not real: the scale.** This is the entire dataset.

| Thing | Count |
|---|---|
| Parameters | 3048 |
| Vocabulary | 45 tokens (4 special + 41 characters) |
| Training paragraphs | 8 |
| Validation paragraphs | 2 |
| Instruction examples (SFT) | 12 |
| Preference pairs | 6 |
| Verifiable task families | 3 |

Eight paragraphs is not a corpus. Twelve instruction examples is not an instruction
dataset. Six preference pairs cannot establish that a reward model generalises.

**This demonstrates mechanisms, not capability.** The distinction matters because the
two are easy to confuse when a loss curve goes down.

**Held-out instruction accuracy is 0.000. That is the honest result, not a bug.** After
SFT the model scores 0.917 exact match on the 12 examples it trained on and 1.000 on a
re-ask of those same prompts — and **0.000** on paraphrases of them and **0.000** on
unseen prompts. It memorised. A model this size, with twelve examples, can do nothing
else. The pipeline reports the held-out number next to the training number on purpose,
because a project that showed you only the 0.917 would be lying to you.

The same honesty applies elsewhere: the `compose` task family scores **0.000 at every
single checkpoint**. Genuinely held-out composition is never solved here. The RL stage
does not fix it and is not reported as if it had.

---

## Setup

### The one thing that will bite you

`python3` on this machine is **not** the interpreter that can run this:

```
$ python3 -V
Python 3.12.1            # /opt/local/bin/python3
$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

There is exactly one interpreter here with torch installed:

```
$ /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -V
Python 3.9.1
```

So every command below uses that full path, or a virtualenv built from it. Plain
`python3` will fail at the first import. The `Makefile` defaults `PYTHON` to the
framework interpreter for this reason, so `make` targets work with no arguments.

### Option A — use the framework interpreter directly (nothing to install)

Verified: torch 2.2.2, numpy 1.26.4, fastapi 0.104.1, uvicorn 0.24.0, pydantic 2.12.5
and pytest 7.4.3 are already present there.

```sh
cd /Users/upalc/Documents/deeplearning-tuts/llmtuning
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m glassbox.export
```

### Option B — a virtualenv that borrows the installed torch (fast, no download)

This is the quickest way to get an isolated `python`/`pip` without re-downloading a
150 MB wheel. `--system-site-packages` lets the venv see the framework interpreter's
torch.

```sh
cd /Users/upalc/Documents/deeplearning-tuts/llmtuning
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
# 2.2.2 1.26.4
```

### Option C — a clean virtualenv that installs its own torch

Use this on a machine that is not this one. Note that a clean Python 3.9.1 venv ships
pip 20.2.3, which works but is old enough that some pip flags are missing; upgrading is
worthwhile.

```sh
cd /Users/upalc/Documents/deeplearning-tuts/llmtuning
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

This downloads about 150 MB. Dependency resolution is verified on this machine — a
fresh 3.9.1 venv resolves to `torch-2.2.2 numpy-1.26.4 Jinja2-3.1.6 MarkupSafe-3.0.3
filelock-3.19.1 fsspec-2025.10.0 mpmath-1.3.0 networkx-3.2.1 sympy-1.14.0
typing_extensions-4.16.0`.

For the optional server, add the extras:

```sh
.venv/bin/python -m pip install -r requirements-server.txt
```

### Requirements, and why numpy is pinned

`requirements.txt` is the engine: **torch 2.2.2** and **numpy 1.26.4**. numpy is listed
explicitly because torch 2.2.2 does **not** declare it as a dependency (it is absent
from torch's `Requires-Dist`), yet `glassbox/checks.py` hashes weights through
`tensor.numpy().tobytes()` in the reproducibility check. Install torch alone and that
check cannot run.

`requirements-server.txt` holds the extras that only `server.py` needs — fastapi,
uvicorn, pydantic. You never need them to learn anything; the app works offline from a
file.

**Python floor: 3.9**, verified on CPython 3.9.1 (macOS 26.5.1, x86_64, pip 24.1.1).
Every number in this README was measured on that interpreter. torch 2.2.2 publishes no
wheels for Python 3.13+, so a newer interpreter needs a newer torch and your numbers
will drift.

---

## Build a run

```sh
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m glassbox.export
```

That runs the entire pipeline. On this machine it takes **about 100 seconds** on CPU —
measured at 100.14 s and 118.56 s on two runs of the same config, the spread being
machine load. Every artifact records its own figure in `manifest.elapsed_seconds`, so
you never have to trust this paragraph. Output:

```
[glassbox] building model
[glassbox] pretraining for 2000 steps
[glassbox] instruction tuning for 400 steps
[glassbox] instruction tuning with LoRA adapters on a frozen base
[glassbox] training a reward model on preference pairs
[glassbox] DPO for 200 steps
[glassbox] warm-starting the policy on task formatting
[glassbox] GRPO for 60 iterations on held-out task values
[glassbox] evaluating every checkpoint under one pinned decoding protocol
[glassbox] running numerical checks
[glassbox] wrote .../runs/run_908d2b0535c2.json (0.71 MB)
[glassbox] checks: 26/26 passed
[glassbox] injected run data into .../app/index.html
```

It writes:

- `runs/run_<12 hex>.json` — the artifact for that run (0.71 MB). The id is a hash of
  the manifest, so the same config and seed produce the same filename.
- `runs/latest.json` — a byte-identical copy, so tooling has a stable path.
- `runs/run_<12 hex>.weights.pt` and `runs/latest.weights.pt` — a torch sidecar holding
  the trained tensors for all seven checkpoints (~146 KB). The JSON carries the recorded
  numbers; the sidecar carries the weights, so the server can decode from the checkpoint
  a caller actually asked for instead of answering from one and relabelling the reply.
- `app/index.html` — the run data is inlined into the page's `<script id="run-data">`
  block, which is what makes the app self-contained.

**Exit status is 0 only if all 26 checks pass**, so it works in a build script.

### Flags

| Flag | Default | Effect |
|---|---|---|
| `--pretrain-steps` | 2000 | Pretraining steps |
| `--sft-steps` | 400 | SFT steps (LoRA SFT uses the same count) |
| `--grpo-iterations` | 60 | GRPO iterations |
| `--seed` | 42 | Seeds model init; stages have their own sub-seeds |
| `--width` | 8 | Residual width |
| `--layers` | 2 | Transformer blocks |
| `--heads` | 2 | Attention heads (must divide `--width`) |
| `--out` | `runs/` | Artifact directory |
| `--html` | `app/index.html` | Page whose run-data block is replaced |
| `--no-html` | off | Skip the HTML injection entirely |

A fast smoke run, useful for checking a change end to end in well under a minute:

```sh
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m glassbox.export \
  --pretrain-steps 50 --sft-steps 20 --grpo-iterations 3 --no-html --out /tmp/glassbox-smoke
```

Its numbers are meaningless; it exists to prove the wiring works. It still reports
26/26 checks passing, because the checks do not depend on the model being good.

---

## Open the app

**In a browser, nothing installed:** **https://upalchowdhury.github.io/glassbox/**

That is the same `app/index.html` served by GitHub Pages. The page is fully functional
there: all fourteen lessons, all seven checkpoints, every recorded tensor, the
browser-side labs and the prediction probes. What is *not* available on Pages is
`server.py` and the `/api/*` routes, because Pages is static hosting — live inference from
a chosen checkpoint, the live gradient check, and rebuilding a run from your own text need
the local server below.

**Offline, no server.** `app/index.html` is a single self-contained file with the run
data already inlined by `glassbox.export`. Just open it:

```sh
open /Users/upalc/Documents/deeplearning-tuts/llmtuning/app/index.html
```

No build step, no web server, no network. If you move the file somewhere else it still
works, because the run data travels inside it.

**With the server, if you want the live API.** `server.py` serves the same page plus
JSON endpoints that can re-run stages on demand. It needs the extras from
`requirements-server.txt`.

```sh
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 server.py
# then open http://127.0.0.1:8077
```

It listens on `127.0.0.1:8077` by default; `--host`, `--port` and `--log-level` change
that. The server is strictly optional — it adds live interaction, not content.

---

## The path through it

The app's fourteen pages are an order, not a menu. Each one depends on the one before,
and every page ends with a **Predict before you look** card that commits you to an answer
before revealing the recorded number.

| # | Page | The question it answers |
|---|---|---|
| 1 | Start here | What is this, and did the model actually learn anything? |
| 2 | Text becomes ids | Why are ids categories rather than numbers, and what does a learned subword vocabulary buy? |
| 3 | Follow one token | Where does a single number in the middle of the model come from? |
| 4 | One gradient step | How does a mistake become a weight change, and can the gradient be trusted? |
| 5 | Channel mixing | What does an MLP do that a single projection cannot? |
| 6 | Pretrain and overfit | What is the difference between optimising, memorising and generalising? |
| 7 | Make text appear | How does text get produced, and what makes it stop? |
| 8 | Teach instructions | What changes when you mask the loss, and what does it cost? |
| 9 | Adapters and rank | What does freezing the base model actually constrain? |
| 10 | Preferences and DPO | What do you do with a ranking, and what does optimising it break? |
| 11 | Verifiers and GRPO | Where does a reward come from, and how is it turned into an update? |
| 12 | What counts as proof | Which observations would count as evidence of improvement? |
| 13 | Every check, measured | Which properties hold, and what are their measured values? |
| 14 | Course map | Which axes must not be conflated, and what is still absent? |

Pages 1-7 are the mechanism. Pages 8-11 are the four post-training stages, and pages
8 and 9 are deliberately siblings: same data, same objective, different trainable
parameters. Pages 12-13 are how you would know any of it worked.

The checkpoint selector in the top bar applies everywhere. Switching it re-renders every
tensor, gradient and sample on the current page from that checkpoint's own recorded run,
so the same lesson can be read at seven points in the model's life.

---

## Checks and tests

Two different things, and it is worth knowing which is which.

**The 26 numerical checks** are properties of the implementation. They ship inside the
package, run as the last stage of every `glassbox.export` build, and decide its exit code.
To run them on their own, in about ten seconds:

```sh
make checks          # 26/26 passed
make checks-verbose  # the same, printing every measured value
```

which is:

```sh
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m glassbox.checks
```

It pretrains for 400 steps and instruction-tunes for 150 before checking. That is
deliberate: several checks are vacuous on an untrained model — the loss-mask alignment,
EOS supervision and DPO-reference checks all need a model that has actually been trained —
so checking an untrained one would still print 26/26 while exercising far less. Use
`--steps N` to change how much training happens first.

**The test suite** lives in `tests/` as plain `unittest.TestCase` classes. It runs two
ways, with no third-party dependency required:

```sh
# standard library only — always works
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m tests.run_checks

# pytest collects the same TestCase classes (pytest 7.4.3 is installed here)
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest tests -q
```

236 tests are collected. All three targets pass, measured:

| Command | Collected | Passed | Failed | Skipped | Wall | Exit |
|---|---|---|---|---|---|---|
| `make test` (`--fast`) | 236 | 222 | 0 | 14 | 10.2 s | 0 |
| `make test-full` | 236 | 235 | 0 | 1 | 19.8 s | 0 |
| `make test-all` | 236 | **236** | 0 | **0** | 32.6 s | 0 |

`make test-all` is the only one that leaves nothing out — it sets
`GLASSBOX_SLOW_TESTS=1`. The single remaining skip in `make test-full` is a server test
that trains a whole run to prove the script-injection payload is escaped in the served
HTML.

`--fast` sets `GLASSBOX_SKIP_SLOW=1`, which skips the 13 end-to-end export tests that
each need a trained run. They are worth the extra ten seconds, so prefer `make test-full`
unless you are iterating tightly.

Other options are `-v` (repeat for more detail), `-k PATTERN` to pick files, and
`--failfast`. `tests/README.md` documents the expected-failure policy: a deliberately
recorded known defect keeps its full-strength assertion and is reported separately rather
than failing the run.

---

## Measured results

All from seed 42, torch 2.2.2, CPython 3.9.1, CPU. Losses are mean cross-entropy per
token in nats. `ln(45) = 3.8067` is the loss of a model that knows nothing.

### Pretraining beats a count-based model

This is the comparison that decides whether the transformer earned its parameters. The
baselines are fitted on the **training split only** and scored on **exactly** the
positions the model is scored on (12006 validation positions, 65334 train positions),
with Laplace smoothing at alpha = 1.

| Model, validation split | Loss |
|---|---|
| Uniform over 45 tokens | 3.8067 |
| Unigram + Laplace | 2.8704 |
| Bigram + Laplace | 2.3025 |
| **Glassbox transformer** | **2.1671** |

2.1671 < 2.3025, so it beats the bigram model. On the train split the gap is wider
(model 1.4840 vs bigram 2.2449). If a change you make pushes validation loss above
2.3025, the transformer has stopped being worth its parameters and the table will say
so.

### Stage by stage

| Stage | Config | Start | End |
|---|---|---|---|
| Pretrain | 2000 steps, AdamW, lr 3e-3, batch 16, 1013 train / 218 val windows | train 3.8071, val 3.8090 | train 1.4840, val **2.1671** |
| | best validation | — | **2.0431 at step 500** |
| SFT | 400 steps, lr 2e-3, 12 examples | answer loss 2.8470 | **0.1438** |
| | prose retention | 2.1671 | **4.2603** |
| LoRA SFT | rank 2, alpha 4, targets q and v, 128 of 3048 trainable (4.03%), lr 1e-2 | answer loss 2.8470 | **1.7672** |
| | prose retention | 2.1671 | **2.6110** |
| Reward model | 300 steps, lr 5e-3, 6 pairs, Bradley–Terry | loss 0.6931, accuracy 0.000 | loss **0.0003**, accuracy **1.000**, margin 8.1936 |
| DPO | 200 steps, lr 1e-3, beta 0.1 | loss 0.6931, margin 0.000 | loss **0.000111**, margin **10.8360**, accuracy **1.000** |
| Warm start | 600 SFT steps on task formatting, 18 examples | — | answer loss 0.1365, retention 7.2726 |
| GRPO | 60 iterations, group 8, lr 3e-3, clip 0.2, KL 0.02, top-k 6, 4 inner epochs | mean reward **0.2344** | mean reward **0.8125**, best 0.8594 |

Validation loss bottoms out at **2.0431 at step 500** and is **2.1671 by step 2000**.
Training for the full 2000 steps makes the model worse on held-out text. Early stopping
is not a slogan here; it is visible in the artifact.

### Every checkpoint, one decoding protocol

Greedy, max 24 new tokens, identical for every row.

| Checkpoint | train prose | val prose | instr. trained | repeat | paraphrase | held-out | add | template | compose |
|---|---|---|---|---|---|---|---|---|---|
| initial | 3.8071 | 3.8090 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| base | 1.4840 | **2.1671** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sft | 4.1389 | 4.2603 | **0.917** | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lora | 2.3327 | 2.6110 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dpo | 5.9886 | 6.4067 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| warm_start | 7.4372 | 7.2726 | 0.000 | 0.000 | 0.000 | 0.000 | **0.875** | 0.000 | 0.000 |
| rl | 9.4363 | **9.0946** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.875** | 0.000 |

Four findings worth sitting with:

1. **LoRA learns less and forgets less.** Full SFT drives answer loss to 0.1438 but
   pushes prose loss from 2.1671 to 4.2603. LoRA reaches only 1.7672 on the same data
   with the same step count, and prose loss lands at 2.6110. Neither is the winner; it
   is a trade-off, and it is measured rather than asserted.
2. **DPO made the chosen answer *less* likely too.** Accuracy 1.000 and margin 10.8360
   look like success. But on pair 0 the policy's log-probability of the *chosen* answer
   fell from the reference's −0.4262 to **−8.3794**, while the rejected answer fell from
   −135.2473 to −240.8289. The objective only asks for the *gap* to widen; it never asks
   for the good answer to stay probable. Instruction accuracy collapsed 0.917 → 0.000.
3. **Catastrophic forgetting, measured.** Validation prose loss climbs monotonically
   along the lineage: 2.1671 → 4.2603 (SFT) → 6.4067 (DPO) → 7.2726 (warm start) →
   **9.0946** (RL). The RL checkpoint is far worse at prose than the untrained model's
   3.8090.
4. **RL traded one task family for another.** GRPO took the held-out `template` family
   from 0.000 to 0.875 — and lost `add`, which the warm start had at 0.875, back down to
   0.000. The mean over the three families is 0.2917 before and 0.2917 after. `compose`
   stays 0.000 throughout.

During GRPO, the clipped fraction reached **0.8594** and **150 of 480 groups had zero
reward variance** — groups where every rollout scored the same, so the group-relative
advantage is zero and the group teaches nothing. Both are reported because both are how
this actually behaves.

### Checks

26 of 26 pass. Also verified directly: re-running the cached pipeline reproduces the
pretrain, SFT and warm-start losses to all 16 significant digits.

---

## How to experiment

Every number in this section was measured by running it, not predicted.

`--width`, `--layers`, `--heads`, `--seed` and the three step-count flags are on the CLI.
LoRA rank, DPO beta and GRPO `inner_epochs` are **not** — they live in `RunConfig`, so you
drive them from Python. The fast way is to train the shared prefix once, save it, and then
sweep from the saved state instead of re-pretraining every time:

```python
import copy, torch
from glassbox import data
from glassbox.model import ModelConfig, build_model
from glassbox.tokenizer import CharTokenizer
from glassbox.train import PretrainConfig, SFTConfig, pretrain, sft

tok = CharTokenizer()
cfg = ModelConfig(vocab_size=len(tok))
model = build_model(cfg, seed=42)
pretrain(model, tok, PretrainConfig(steps=2000))
base_state = copy.deepcopy(model.state_dict())
torch.save(base_state, "base.pt")          # reuse this for every variant

# e.g. one LoRA rank
from glassbox.lora import LoRAConfig, apply_lora
m = build_model(cfg, seed=42); m.load_state_dict(base_state)
apply_lora(m, LoRAConfig(rank=4, alpha=4.0))
out = sft(m, tok, SFTConfig(steps=400, learning_rate=1e-2, warmup_steps=50, seed=99))
print(out["final"]["train"], out["final"]["retention"])
```

The same pattern drives `train_dpo(..., DPOConfig(beta=...))` from the SFT state and
`grpo_train(..., GRPOConfig(inner_epochs=..., learning_rate=...))` from the warm-start
state. Every table below was produced this way, which is why the `0.2344` starting reward
is identical across all the GRPO rows — they all begin from one saved checkpoint.

### Width — the clearest overfitting demonstration in the project

```sh
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m glassbox.export --width 16 --no-html --out /tmp/w16
```

Parameter counts are exact: width 8 → **3048**, width 16 → **7088**, width 32 →
**18240**. `--layers 4` → 4184. `--heads` changes nothing about the count, only how the
width is split.

Doubling width to 16 gives this:

| | width 8 | width 16 |
|---|---|---|
| Parameters | 3048 | 7088 |
| Final train loss | 1.4840 | **0.5316** |
| Final validation loss | **2.1671** | **2.9387** |
| Best validation | 2.0431 at step 500 | 2.0345 at step 250 |
| SFT answer loss | 0.1438 | 0.0039 |
| SFT prose retention | 4.2603 | 4.5003 |
| SFT instr. trained / held-out | 0.917 / **0.000** | **1.000** / **0.000** |

Train loss falls by two thirds and validation loss gets **worse** — 2.9387 is above the
bigram baseline of 2.3025, so the bigger model stops being worth its parameters. Its
best validation point arrives twice as early (step 250), so it overfits sooner. And
held-out instruction accuracy is **still exactly 0.000**. More capacity buys
memorisation, not generalisation, and this is the cheapest place you will ever see that.

### Steps

You do not need a new run to see this: the artifact already records best validation
2.0431 at step 500 against 2.1671 at step 2000. Lowering `--pretrain-steps` toward 500
*improves* held-out loss. Raising it makes the train/validation gap wider.

### LoRA rank

Rank is not a free dial, because scaling is `alpha / rank` — changing rank with alpha
fixed at 4 also changes how hard the adapter pushes.

| rank | scaling | trainable | of 3048 | answer loss | prose retention |
|---|---|---|---|---|---|
| 1 | 4.0 | 64 | 2.06% | 2.8470 → 1.8742 | 2.1671 → **2.4110** |
| 2 | 2.0 | 128 | 4.03% | 2.8470 → **1.7672** | 2.1671 → 2.6110 |
| 4 | 1.0 | 256 | 7.75% | 2.8470 → 1.6639 | 2.1671 → 3.3165 |
| 8 | 0.5 | 512 | 14.38% | 2.8470 → **1.6428** | 2.1671 → 2.7822 |

Answer loss improves monotonically with rank. Forgetting does **not** — rank 1 forgets
least (2.4110) and rank 4 forgets most (3.3165), so the curve is not monotone and you
should not assume it is.

Targets are also editable. Valid names per block are `q`, `k`, `v`, `attention_out`,
`mlp_up`, `mlp_down`; the default is `('q', 'v')` at 128 trainable, and `('q','k','v')`
gives 192 (5.93%).

### DPO beta

Beta controls how far the policy may drift from the reference.

| beta | final loss | margin | accuracy | pair-0 chosen logprob (ref −0.4262) |
|---|---|---|---|---|
| 0.01 | 0.0517069 | 3.7213 | 1.000 | −14.7809 |
| 0.1 | 0.0001106 | 10.8360 | 1.000 | −8.3794 |
| 1.0 | 0.0000000 | 40.4758 | 1.000 | −12.4244 |

Accuracy is 1.000 at every beta, which is exactly why accuracy is a poor thing to watch.
The chosen answer's log-probability collapses from −0.4262 at **every** beta tried. No
value of beta rescued it here. Watch the chosen logprob, not the accuracy.

### GRPO inner_epochs — the clipping term is inert at 1

60 iterations from the same warm-start checkpoint:

| inner_epochs | lr | mean reward start → end | best | mean ratio range | clipped fraction | zero-variance groups |
|---|---|---|---|---|---|---|
| 4 | 3e-3 | 0.2344 → **0.8125** | 0.8594 | 0.7963 – 1.0838 | 0.0469 – 0.8594 | 150 |
| **1** | 3e-3 | 0.2344 → 0.4844 | 0.5156 | **1.0000000000 – 1.0000000000** | **0.000000 always** | 91 |

With one inner epoch the policy has not changed since the rollouts were sampled, so the
importance ratio is **exactly 1** at every iteration and the clipped objective never
engages — it is mathematically dead weight. It only becomes a real constraint when you
update more than once against the same sampled data. Note also that one epoch reaches
only 0.4844 reward instead of 0.8125, with a final gradient norm of 23.49 against 0.18.

### GRPO learning rate — how a policy collapses

| lr | mean reward start → end | best | max clipped fraction | zero-variance groups |
|---|---|---|---|---|
| 3e-3 | 0.2344 → **0.8125** | 0.8594 | 0.8594 | 150 |
| 1e-2 | 0.2344 → 0.3594 | 0.3750 | **1.0000** | 367 |
| 3e-2 | 0.2344 → **0.0000** | 0.2344 | **1.0000** | **473** |

At 1e-2 learning degrades badly — the ratio swings from 0.031 to 3.44 and at some
iterations *every* token is clipped. At 3e-2 the policy **collapses to 0.000 reward**
and never recovers: 473 zero-variance groups means it stopped producing anything the
verifier could distinguish. Once reward variance is gone, a group-relative objective has
no signal and cannot climb back out.

### Why the warm start exists

Skip it and every rollout scores 0, every group has zero variance, and GRPO has nothing
to learn from. That is a prerequisite failure, not evidence that RL does not work — the
pipeline records this rationale in `stages.warm_start.rationale`.

---

## File map

```
glassbox/              the package — the whole engine
  data.py              8 train + 2 validation paragraphs with provenance hashes;
                       4 instruction families (train/repeat/paraphrase/heldout);
                       6 preference pairs; 3 verifiable task families that partition
                       one enumerated value space, so they are provably disjoint.
                       format_prompt(), corpus_sha256(), dataset_sha256(), all_text()
  tokenizer.py         CharTokenizer (45 tokens) and BPETokenizer;
                       fit_on_training_split() — fitted on train only, and checked
  model.py             ModelConfig, TinyTransformer (3048 params) with named-axis
                       tensor capture and a real KV cache, build_model(cfg, seed)
  generate.py          DecodeConfig, GREEDY, generate(), sequence_logprob();
                       real temperature / top-k / top-p, greedy when temperature <= 0
  train.py             document_windows(), pad_to_batch(), mean_token_loss(),
                       pretrain(), sft(), sft_row(), mask_explanation(), sgd_step_demo()
  lora.py              LoRALinear, apply_lora(), merge_all(), unmerge_all(),
                       adapter_report(), first_step_gradients()
  preference.py        RewardModel, train_reward_model(), dpo_terms(), train_dpo(),
                       batch_sequence_logprobs()
  rlvr.py              strict / substring / process verifiers, group_advantages(),
                       sample_rollouts(), grpo_train(), compare_verifiers()
  evaluate.py          evaluate_instructions(), evaluate_tasks(), retention(),
                       checkpoint_report(), compare_checkpoints(), baseline_losses()
  checks.py            26 numerical checks and run_all(model, tok, cfg)
  export.py            build_run(), write_run(), inject_into_html(), main()
                       CLI: python -m glassbox.export

app/index.html         the app: one self-contained file, run data inlined, no server
server.py              optional FastAPI server and live API (port 8077)
tests/                 unittest suite; tests/run_checks.py runs it without pytest
                       tests/README.md explains the expected-failure policy
runs/                  generated artifacts: run_<hex>.json, latest.json, .weights.pt

requirements.txt           torch + numpy — the engine
requirements-server.txt    fastapi, uvicorn, pydantic — only for server.py
Makefile                   install, run, smoke, checks, checks-verbose, test,
                           test-full, test-all, serve, clean

API.md                              the server's endpoints, with observed responses
Glassbox_Course_Blueprint.md        the course blueprint; its factual sections are
                                    generated from runs/latest.json by
                                    python -m glassbox.report --update
legacy/                             the superseded implementation, kept for reference
  glassbox_engine.py                the old monolith; nothing imports it
  Glassbox_Interactive_Starter.html the old app, moved here from the project root
  README.md                         what was wrong with them, defect by defect
```

The numbers in this README all come from the `glassbox/` package, not from
`legacy/glassbox_engine.py`.

Run artifacts in `runs/` are generated and not meant to be committed — see
`.gitignore`. Regenerate any of them with `make run`.

---

## Licensing

**No licence has been chosen for this project yet.** Until one is added, no permission
to copy, modify or redistribute should be assumed.

The teaching content is original work written for this project: the eight training
paragraphs, the two validation paragraphs, all 12 instruction examples and their
paraphrase and held-out variants, the 6 preference pairs, and the 3 verifiable task
families. None of it is scraped, borrowed, or derived from another dataset.

Dependencies carry their own licences — torch, numpy, and for the optional server
fastapi, uvicorn and pydantic. **Review the licences of every dependency before
redistributing anything**, including before shipping the self-contained
`app/index.html`.
