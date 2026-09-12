# Glassbox

A reading-first guide to language-model systems, with an optional microscope into a
small recorded transformer run.

Open [app/index.html](app/index.html) in a modern browser. No code, account, Python
installation, quiz, or training run is required to read the course. Keep the entire
`app/` directory together: the HTML loads sibling JavaScript and CSS files.

## The reading journey

Eight connected chapters follow Pip from next-token prediction to agents and systems:

1. How models learn: tokens, embeddings, attention, loss, gradients, AdamW, generation.
2. Mixture of experts: routing, weighted merging, capacity, balancing, systems costs.
3. Teaching and adapting: SFT masks, LoRA, merging, forgetting, and preferences.
4. Define success: environments, reset, termination, task-aware verification, shortcuts.
5. Learning from rewards: group advantages, clipping, reference policies, credit assignment.
6. From answers to agents: harnesses, trajectories, permissions, retries, evaluation.
7. Scaling the system: parallelism, memory, queues, staleness, recovery.
8. Put it all together: diagnostic case studies and a conceptual review.

The chapters contain roughly 10,000 words, small worked calculations, failure cases,
fully explained mental-review questions, takeaways, and bridges. Readers can follow
the whole explanation without writing or executing code.

The learning target is conceptual understanding: explain mechanisms, predict
consequences, and diagnose misleading results. Reading does not certify professional
implementation or operational expertise. Reader testing is still needed to assess
actual comprehension and retention.

## Reading controls

- **Read** is the default: complete prose and worked examples.
- **Deep dive** opens additional notation and derivations.
- **Summary** is explicitly an overview, not a substitute for the chapter.
- **Complete book** displays every chapter, review explanation, derivation, and the
  glossary. Use its print button to print or save a PDF through the browser.
- Chapter contents link to individual sections. Routes such as
  `#grpo/section-3` can be bookmarked or shared.
- **Bookmark / Resume** stores a reading place, including the current section.
- **Mark chapter as read** is explicit and reversible. Visits do not count as
  completion. Read marks are not mastery scores.
- Reading depth, progress, and bookmarks use local browser storage when available.
  Reading remains usable when storage is blocked or corrupted. No tracking or uploads.

The original recorded-reference lessons remain available in the sidebar. They are
optional: opening them does not change the chapter sequence or gate later chapters.
Their checkpoint selector applies to the recorded inspectors, not the book examples.

## What the numbers mean

There are three separate kinds of material:

| Material | What it establishes |
|---|---|
| Chapter calculations | Arithmetic from explicitly stated teaching inputs |
| Hypothetical cases and trajectories | An explanation of a mechanism or failure, not a benchmark |
| Recorded reference traces | Measurements from the embedded Python run artifact |

The browser reader does not train TinyGPT or MoE, execute model-written repair code,
run agents, or benchmark hardware. A JSON verifier really parses and checks the
chapter's specified status task; it does not claim to check every aspect of helpfulness.

The former synthetic checkpoint controls, checkbox-based RL gate, source-text
“code test” matcher, fabricated agent scores, and invented utilization estimates
have been removed from the course path. The reading-first direction supersedes the
mandatory lab interactions in [the original brief](prompts/impl.md).

## Running and publishing

**Offline:** open `app/index.html`, with its sibling assets present. External paper
links are optional and require a connection. No network requests are needed to read
the local book.

**Static hosting:** publish the entire `app/` directory. The included Pages workflow
uploads that directory; there is no frontend build step. It does not publish the
Python API.

**Optional local API:** choose a compatible Python environment and install the
engine/server dependencies. The pinned requirements target Python 3.9-era tooling;
use an interpreter supported by those versions.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-server.txt
.venv/bin/python server.py --port 8077
```

Open `http://127.0.0.1:8077`. The server serves the reading assets from an explicit
allowlist as well as the recorded-run page. See [API.md](API.md) for the optional
inference, gradient-check, and run-building endpoints. This is not required for learning.

## Verification

Frontend arithmetic and content-contract tests need Node.js:

```sh
node --test app/course-engine.test.js
```

Browser smoke tests need Node 22+ and Chrome/Chromium; no npm packages:

```sh
node tests/reading-browser.mjs
# Nonstandard installation:
CHROME_BIN=/path/to/chromium node tests/reading-browser.mjs
```

The browser test starts a loopback-only fixture server and disposable Chrome profile.
It checks all chapters and recorded routes, reading depths, progress/undo, section
bookmarks and reload, the ungated GRPO route, glossary, PDF output, mobile overflow,
storage failures, missing content/run data, and offline file access.

Server assets and HTML-injection regressions:

```sh
.venv/bin/python -m unittest tests.test_reading_assets \
  tests.test_server.ScriptBreakoutTests tests.test_export.HtmlInjection
```

The model suite is separate and requires the Python dependencies:

```sh
.venv/bin/python -m tests.run_checks
.venv/bin/python -m glassbox.checks
```

Recorded check results belong to their artifact and environment. They are not a
claim that every test passes under every installed PyTorch version.

## Content architecture

- `app/reading-chapters.js`: original prose, review explanations, glossary, source links.
- `app/reading.js`: chapter, home, glossary, and full-book rendering.
- `app/reading.css`: readable typography, responsive tables, print layout.
- `app/course-engine.js`: pure worked-example calculations, with explicit assumptions.
- `app/course-content.js`: short navigation labels.
- `app/index.html`: existing shell, routing, persistence, recorded inspectors and run data.

To add a chapter, add its content and navigation entry. All essential explanation
belongs in the chapter text. Add any new calculation to the pure engine, render its
labelled example, and test the math and reading contract. Never replace an unavailable
measurement with an invented checkpoint or success badge.

## Optional: rebuild the recorded model

```sh
.venv/bin/python -m glassbox.export
```

The exporter writes run JSON, checkpoint-weight sidecars, and the embedded run-data
block in `app/index.html`. It does not rewrite the reading chapters or sibling assets.
Use `--no-html` to preserve the currently embedded artifact, and `--help` for model,
seed, and step-count options. Model size, data, losses, checks, and checkpoint lineage
are recorded in the artifact; the default tiny model demonstrates mechanisms, not
broad capability.

## Historical experiment notes

The following measurements and implementation notes were recorded by the original
project under its stated Python/PyTorch configuration. They are preserved as
reference, not newly measured results of the reading edition. Consult the artifact's
manifest and current test output before expecting exact reproduction.

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
