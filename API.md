# Glassbox server API

`server.py` serves a Glassbox run artifact and answers live questions about it.
It does not train at import time and it does not train at startup. Training only
happens when you ask for it, with `POST /api/run`.

Every curl command below was run against a live server and the HTTP status shown
is the status observed, not the status intended.

## Running it

The default `python3` on this machine has neither torch nor fastapi. Use the
framework interpreter:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 server.py --port 8077
```

Flags: `--host` (default `127.0.0.1`), `--port` (default `8077`), `--log-level`
(default `info`). Verified versions in this environment: Python 3.9.1,
torch 2.2.2, fastapi 0.104.1, pydantic 2.12.5, uvicorn 0.24.0.

Interactive docs are generated from the request models at `/docs` (200) and
`/openapi.json` (200).

### Import cost

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -c "
import time,sys
t=time.perf_counter(); import server
print('%.3fs' % (time.perf_counter()-t), 'torch loaded:', 'torch' in sys.modules)"
# 0.675s torch loaded: False
```

`glassbox` — and therefore torch — is imported lazily inside the functions that
need it, so `import server` pays only for fastapi and pydantic. The module this
replaced trained inside its own import, which is why `uvicorn --reload` used to
retrain on every keystroke. The first request that needs torch pays about 2.1s
once; after that it is warm.

## What is loaded, and what that means for `checkpoint`

A run artifact (`runs/latest.json`) records **measurements**, not tensors. So on
a cold start:

* `checkpoint=initial` works immediately. `build_model(cfg, seed)` is
  deterministic, so random initialisation is reproducible from the manifest
  alone. Verified: the text returned for `pip has ` is byte-identical to the
  `initial` sample the artifact itself recorded.
* `base`, `sft`, `lora`, `dpo`, `warm_start`, `rl` return **409** with the list
  of what is available and how to get the rest. They are not silently answered
  from some other checkpoint's weights — that mislabelling is the defect this
  endpoint was rewritten to remove.

Two ways to get the trained checkpoints:

1. `POST /api/run` trains them in this process and keeps the state_dicts in
   memory.
2. `POST /api/run` with `"save": true` also writes `runs/<run_id>.weights.pt`
   and `runs/latest.weights.pt` (about 146 KB for all seven). A later server
   start reloads them instantly — verified across a restart, with all seven
   checkpoints answering from `runs/latest.weights.pt` and no training.

A weights sidecar whose `run_id` does not match the loaded artifact is
**refused**, not used. Verified: with `latest.json` at `run_908d2b0535c2` and
`latest.weights.pt` at `run_c44095ee66e2`, `/api/status` reported
`weights_in_memory: []` and `sft` returned 409.

---

## `GET /`

Serves `app/index.html` with the run artifact injected into its
`<script id="run-data" type="application/json">` block.

```bash
curl -s -o /dev/null -D - http://127.0.0.1:8077/ | grep -iE 'HTTP/|x-glassbox'
# HTTP/1.1 200 OK
# x-glassbox-run-data: injected run run_908d2b0535c2
```

The app file is **never written**. Injection happens in memory on each request,
using the escaping rule from `glassbox.export.inject_into_html`: `<` becomes
the six characters `\u003c`. `json.dumps` does not escape `<`, so a paragraph containing
`</script><script>alert(1)</script>` would otherwise close the data element and
execute. Measured on a real run whose corpus contained that payload: the data
block held 779,189 bytes, **zero** `<` characters, 418 `\u003c` escapes, and the
paragraph still round-tripped byte-for-byte through `JSON.parse`.

Returns **503** with an HTML explanation when `app/index.html` is absent. If the
file exists but has no data block, or the block is unclosed (mid-edit), the page
is served unchanged and the `x-glassbox-run-data` header says so.

## `GET /api/status`

```bash
curl -s http://127.0.0.1:8077/api/status
# HTTP 200
```

Reports: `loaded`, `run_id`, `artifact_source`, `run_in_progress`, `active_run`,
`last_run`, the check pass counts, model size, which checkpoints have weights in
memory and where they came from, the live `python`/`torch` versions alongside the
versions the run was built with, whether `app/index.html` exists and has a data
block, and the run ids on disk and in memory.

Observed on the shipped artifact: `{"total": 26, "passed": 26, "failed": 0,
"all_passed": true}`, `parameters: 3048`, `vocab_size: 45`, python `3.9.1`,
torch `2.2.2`.

This endpoint keeps answering during a run, because `POST /api/run` is a
threadpool handler rather than an `async` one. Verified mid-run:
`run_in_progress: true` with the active step counts.

## `GET /api/checks`

```bash
curl -s http://127.0.0.1:8077/api/checks
# HTTP 200
# {"run_id":"run_908d2b0535c2","source":"runs/latest.json",
#  "total":26,"passed":26,"failed":0,"all_passed":true,"results":[...]}
```

Each entry in `results` carries its name, `passed`, a description, the measured
numbers and the tolerance. **503** if no run is loaded.

## `GET /api/parameters?checkpoint=<name>`

Lists every named parameter of one checkpoint with its real shape and
`trainable` flag. This is what makes `/api/gradient-check` usable without
guessing a matrix name or a bound.

```bash
curl -s 'http://127.0.0.1:8077/api/parameters?checkpoint=initial'
# HTTP 200 — 14 matrices, 14 vectors, total_parameters 3048
curl -s 'http://127.0.0.1:8077/api/parameters?checkpoint=bogus'
# HTTP 400
```

On a LoRA checkpoint the names and flags change, which is the lesson:
`blocks.0.q.base.weight` is `[8, 8]` **frozen** while `blocks.0.q.lora_A`
`[2, 8]` and `blocks.0.q.lora_B` `[8, 2]` are trainable — 128 trainable of 3216.

## `GET /api/run/{run_id}`

```bash
curl -s -o /dev/null -w '%{http_code} %{size_download}\n' http://127.0.0.1:8077/api/run/latest
# 200 709115   (the exact byte count moves with the run; ~0.7 MB)
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8077/api/run/run_908d2b0535c2
# 200
curl -s http://127.0.0.1:8077/api/run/run_aaaaaaaaaaaa
# HTTP 404 — lists what is in memory and on disk
curl -s http://127.0.0.1:8077/api/run/..long
# HTTP 422 — {"error":"'..long' is not a run id", ...}
```

`latest` means the run this server has loaded. Any other value must match
`run_` followed by 6–32 hex characters, which is what keeps a path out of the
`runs/` directory. In-memory runs are found even when they were never saved.

## `POST /api/infer`

Decodes from the named checkpoint's weights under an explicit protocol.

```bash
curl -s -X POST http://127.0.0.1:8077/api/infer \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"pip has ","checkpoint":"initial","max_new_tokens":12}'
# HTTP 200
# {"checkpoint":"initial","protocol":"greedy, max 12 new tokens",
#  "deterministic":true,"text":"::::00000000",
#  "stopped_on_eos":false,"hit_length_limit":true,"vocabulary_size":45}
```

| field | bounds | effect |
|---|---|---|
| `prompt` | 1–400 chars, required | encoded with `<bos>` |
| `checkpoint` | one of the seven names | selects the weights; 400 if unknown |
| `max_new_tokens` | 1–128 | decode budget |
| `temperature` | 0.0–5.0 | `0` is greedy; positive samples |
| `top_k` | 0–512 | `0` disables |
| `top_p` | 0.0–1.0 | `0` disables |
| `seed` | 0–2147483647 | makes sampling reproducible |
| `use_cache` | bool | KV cache on/off; must not change the tokens |
| `stop_at_eos` | bool | stop on the learned EOS |
| `apply_prompt_template` | bool | wrap in `user: …\nassistant: ` |
| `record_steps` | bool | return the top-5 distribution per step |

Every one of these is passed into `glassbox.generate.DecodeConfig` and the
resolved protocol string is echoed back. Measured, same prompt and checkpoint:

| request | `protocol` | `text` |
|---|---|---|
| greedy | `greedy, max 10 new tokens` | `::::000000` |
| `temperature 1.2, seed 7` | `temperature 1.2, max 10 new tokens, seed 7` | `nf81ffa.1` |
| `temperature 1.2, seed 7` again | same | `nf81ffa.1` |
| `temperature 1.2, seed 8` | `… seed 8` | `sr7t71` |
| `temperature 1.2, top_k 3, seed 7` | `… top-k 3 …` | `503300000` |
| `temperature 1.2, top_p 0.5, seed 7` | `… top-p 0.5 …` | `nf81` |
| `temperature 1.2`, no seed | no seed in protocol, `deterministic:false` | varies |
| `use_cache:false` | `greedy, max 10 new tokens` | `::::000000` |

Characters outside the run's vocabulary are reported rather than silently
becoming `<unk>`:

```bash
curl -s -X POST http://127.0.0.1:8077/api/infer -H 'Content-Type: application/json' \
  -d '{"prompt":"how many ZEBRAS?","checkpoint":"initial","apply_prompt_template":true}'
# HTTP 200 — "unknown_characters":["A","B","E","R","S","Z"] with a note
```

Failure cases, all observed:

```bash
# unknown checkpoint -> 400, listing the valid names
-d '{"prompt":"x","checkpoint":"nope"}'                     # 400
# a valid name whose weights are not loaded -> 409 with a remedy
-d '{"prompt":"x","checkpoint":"sft"}'                      # 409
-d '{"prompt":"x","checkpoint":"initial","max_new_tokens":9999}'  # 422
-d '{"prompt":"","checkpoint":"initial"}'                   # 422
-d '{"prompt":"x","checkpoint":"initial","temperature":99}' # 422
-d '{"prompt":"x","checkpoint":"initial","top_p":1.5}'      # 422
-d '{"checkpoint":"initial"}'                               # 422 (prompt required)
-d '{"prompt":"x","nucleus":0.9}'                           # 422 (unknown field)
```

Unknown fields are rejected rather than ignored. A parameter the server cannot
honour is an error, never a silent no-op.

## `POST /api/gradient-check`

Replaces the old `/api/sgd-check`, which computed nothing: it echoed the
requested row, column and learning rate back beside hardcoded numbers and
dropped `epsilon` entirely. This one computes a central difference in float64 at
the cell you asked about.

```bash
curl -s -X POST http://127.0.0.1:8077/api/gradient-check \
  -H 'Content-Type: application/json' \
  -d '{"checkpoint":"initial","matrix":"blocks.0.mlp_up.weight","row":3,"column":5,"epsilon":1e-4}'
# HTTP 200
# "shape":[16,8], "row":3, "column":5, "parameter_is_trainable":true,
# "finite_difference":{"dtype":"float64","epsilon":0.0001,
#   "estimate":-0.0029302741499925844,
#   "analytic":-0.0029302741428969487,
#   "absolute_error":7.095635710940851e-12,
#   "relative_error":2.4214921020133244e-09,
#   "passes":true}
```

| field | bounds | default |
|---|---|---|
| `checkpoint` | one of the seven names | `sft` |
| `matrix` | a named **2-D** parameter | `blocks.0.mlp_up.weight` |
| `row` | `0 <= row < shape[0]` | `0` |
| `column` | `0 <= column < shape[1]` | `0` |
| `epsilon` | `0 < epsilon <= 0.1` | `1e-4` |
| `prompt` | 1–200 chars | `pip has ` |
| `learning_rate` | `0 < lr <= 10` | `0.05` |

The response also carries `largest_gradient_cell` — the cell
`glassbox.train.sgd_step_demo` would have picked on its own — so you can see that
the cell you asked for is the cell that was checked, `float32_comparison` showing
the same gradient at training precision, and an `sgd_step` block reporting the
loss before and after one real SGD step on every trainable parameter (the model
is restored afterwards).

`epsilon` demonstrably drives the arithmetic. Measured on
`blocks.0.mlp_up.weight[4,3]`, `initial`:

| epsilon | estimate | analytic | relative error | passes |
|---|---|---|---|---|
| `1e-1` | +0.022541832445 | +0.022393091223 | 6.60e-03 | false |
| `1e-2` | +0.022394749208 | +0.022393091223 | 7.40e-05 | false |
| `1e-4` | +0.022393091383 | +0.022393091223 | 7.18e-09 | true |
| `1e-6` | +0.022393091159 | +0.022393091223 | 2.84e-09 | true |
| `1e-8` | +0.022393109589 | +0.022393091223 | 8.20e-07 | true |

That is the textbook U-curve: truncation error dominates at large epsilon,
rounding error creeps back at small epsilon. The analytic gradient does not move.

`passes` is `relative_error < 1e-6` **or** `absolute_error < 1e-9`, and both arms
are reported separately. The absolute arm matters: a freshly attached LoRA
adapter has gradients around 1e-7, where float64 rounding of a loss near 3.8 is a
large *fraction* of a tiny difference while being absolutely negligible. A purely
relative criterion reports a correct gradient as a failure.

On a frozen matrix the answer is a lesson rather than a workaround:

```bash
-d '{"checkpoint":"lora","matrix":"blocks.0.q.base.weight","row":2,"column":4,"epsilon":1e-5}'
# HTTP 200
# "parameter_is_trainable": false
# "gradient_requires_grad_temporarily_enabled": true
# "frozen_note": "blocks.0.q.base.weight is frozen on this checkpoint: requires_grad
#                 is False, so it receives no gradient and cannot be updated..."
# "sgd_step": {"updated_parameter_elements": 128, "frozen_parameter_elements": 3048}
```

Perturbing a frozen weight *does* change the loss, so the finite difference is
real; what the optimiser would not do is apply it. Both facts are reported.

Failure cases, all observed:

```bash
-d '{"checkpoint":"initial","matrix":"blocks.0.mlp_up.weight","row":9999}'
# 422 {"error":"cell is outside the matrix","shape":[16,8],
#      "valid_row_range":[0,15],"valid_column_range":[0,7]}
-d '{"checkpoint":"initial","column":-7}'      # 422 — rejected, NOT wrapped to 1
-d '{"checkpoint":"initial","epsilon":0}'      # 422 — no ZeroDivisionError
-d '{"checkpoint":"initial","epsilon":-0.0001}'# 422
-d '{"checkpoint":"initial","epsilon":0.5}'    # 422 — above the 0.1 cap
-d '{"checkpoint":"initial","matrix":"nope"}'  # 422 — lists every 2-D matrix and shape
-d '{"checkpoint":"initial","matrix":"blocks.0.mlp_up.bias"}'
# 422 {"error":"'blocks.0.mlp_up.bias' has 1 dimensions, not 2","shape":[16]}
-d '{"checkpoint":"bogus"}'                    # 400
-d '{"checkpoint":"initial","lr":0.1}'         # 422 — unknown field
```

## `POST /api/run`

Builds a fresh run through `glassbox.export.build_run`, then serves it.

```bash
curl -s -X POST http://127.0.0.1:8077/api/run \
  -H 'Content-Type: application/json' \
  -d '{"pretrain_steps":25,"sft_steps":15,"warm_start_steps":15,
       "dpo_steps":10,"grpo_iterations":2,"grpo_group_size":4,
       "seed":42,"save":true}'
# HTTP 200 in 20.4s
# run_id run_c44095ee66e2, wall_seconds 20.42, engine_seconds 20.26,
# vocabulary_size 45, parameters 3048,
# checkpoints ["initial","base","sft","lora","dpo","warm_start","rl"],
# checks {"total":26,"passed":26,"failed":0,"all_passed":true},
# saved_to runs/run_c44095ee66e2.json,
# weights_saved_to runs/run_c44095ee66e2.weights.pt
```

| field | bounds | default |
|---|---|---|
| `paragraphs` | 1–24 items, 8–1200 chars each, 8000 total | the course corpus |
| `instructions` | 1–24 items, each `{prompt, answer}` | the SFT train family |
| `pretrain_steps` | 1–4000 | 2000 |
| `sft_steps` | 1–2000 | 400 |
| `warm_start_steps` | 1–2000 | 600 |
| `dpo_steps` | 1–1000 | 200 |
| `grpo_iterations` | 1–200 | 60 |
| `grpo_group_size` | 2–16 | 8 |
| `lora_rank` | 1–8 | 2 |
| `seed` | 0–2147483647 | 42 |
| `save` | bool | **false** |

Defaults match the documented reference run, so the documented run is reachable
through the API. (The module this replaced clamped `sft_steps` to 200 while the
docs described 240, which made the documented run unreachable.) A default-step
run takes about two minutes on CPU; the artifact shipped in `runs/latest.json`
measured 118.56 s.

The response reports what was actually run: the measured `wall_seconds`, the
engine's own `elapsed_seconds`, the resolved `stage_configs` for every stage, the
vocabulary size, the parameter count, the check counts, which checkpoints now
have weights, where anything was written, and a `notes` list for things you should
know but did not ask about.

**`save` defaults to `false` on purpose.** `save: true` writes
`runs/<run_id>.json` *and overwrites `runs/latest.json` and
`runs/latest.weights.pt`*, so a ten-step experiment would otherwise replace the
reference artifact the app ships with. An unsaved run is served from memory and is
addressable at `/api/run/<run_id>`; `notes` says so.

### Submitting your own corpus rebuilds the tokenizer

```bash
curl -s -X POST http://127.0.0.1:8077/api/run -H 'Content-Type: application/json' -d '{
  "paragraphs": [
    "pip has two red stones. </script><script>alert(document.domain)</script> and three blue stones.",
    "mira visits the garden each morning and gives pip one green leaf.",
    "the round box holds the red stones and the square box holds the blue stones."
  ],
  "instructions": [{"prompt":"how many red stones does pip have?","answer":"two."}],
  "pretrain_steps": 30, "sft_steps": 20, "warm_start_steps": 20,
  "dpo_steps": 10, "grpo_iterations": 2, "grpo_group_size": 4, "seed": 42
}'
# HTTP 200 in 17.2s
# vocabulary_size 50  (was 45)
# parameters 3088     (was 3048)
# notes: "tokenizer rebuilt from the submitted corpus: vocabulary is 50 tokens"
#        "the SFT training family was replaced; the repeat/paraphrase/heldout
#         evaluation families are fixed and were not"
```

The character tokenizer is rebuilt from the text you actually submitted. The five
new characters the payload contributes (`<`, `>`, `/`, `(`, `)`) take the
vocabulary from 45 to 50, and the parameter count from 3048 to 3088 — five more
embedding rows at width 8 is exactly 40 more parameters. The module this replaced
never rebuilt the alphabet, so a learner's own characters were encoded through the
stale vocabulary and every unseen one became `<pad>`.

Only the SFT training family is replaceable. The held-out `repeat`, `paraphrase`
and `heldout` evaluation families stay fixed, because they are the measurement;
`notes` says this rather than leaving you to infer it.

### Concurrency

```bash
# while a run is in progress
curl -s -X POST http://127.0.0.1:8077/api/run -H 'Content-Type: application/json' -d '{"pretrain_steps":5}'
# HTTP 409 {"error":"a run is already in progress","active_run":{...},
#           "why":"training mutates process-global engine state, so runs are
#                  serialised rather than interleaved"}
```

A build holds a non-blocking lock, so the second caller is told immediately
instead of being queued behind two minutes of training. The finished run is
installed with a single assignment of an immutable snapshot, so a reader never
sees an artifact paired with another run's weights.

### Validation failures

Every one of these returned **422** with a message naming the field and the
bound, and none produced a 500:

```bash
-d '{"paragraphs":["'"$(printf 'y%.0s' {1..1300})"'"]}'  # paragraph over 1200 chars
-d '{"paragraphs":[]}'                                    # empty list
-d '{"paragraphs":["tiny"]}'                              # under 8 chars
-d '{"paragraphs":[12345]}'                               # not a string
-d '{"instructions":[{"prompt":"q"}]}'                     # missing "answer"
-d '{"instructions":[{"prompt":"q","response":"a"}]}'      # wrong key name
-d '{"pretrain_steps":999999}'                             # above 4000
-d '{"grpo_group_size":1}'                                 # below 2
-d '{"seed":-1}'                                           # negative
```

Ten paragraphs of 1000 characters returns
`"corpus is 10000 characters; the limit is 8000"`. A dict missing `answer`
returns a pydantic `missing` error located at
`["body","instructions",0,"answer"]` — the old server raised `KeyError` here and
turned it into a 500.

`HTTPException` is re-raised rather than caught by the surrounding
`except Exception`, so a deliberate 400/409/422 is never re-wrapped as a 500.

---

## Tests

`tests/test_server.py` holds 29 tests: the script-breakout regression, the
import-cost guard, the validation matrix, the gradient check and the decode
controls. All three of these commands were run and passed:

```bash
cd /Users/upalc/Documents/deeplearning-tuts/llmtuning
P=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3

$P tests/test_server.py                                  # Ran 29 tests — OK (skipped=1)
$P -m unittest discover -s tests -t . -p "test_server.py" # Ran 29 tests — OK (skipped=1)
$P -m pytest tests/test_server.py -q                      # 28 passed, 1 skipped
```

pytest 7.4.3 *is* installed under that interpreter, but none of these tests need
it — `$P tests/test_server.py` runs the whole file through `unittest` with no
third-party test runner, and `$P -m unittest tests.test_server` also works.

Note that `$P -m unittest discover -s tests` without a `-p` pattern collects the
whole `tests/` package, which is much larger than this file; the
`-p "test_server.py"` pattern above limits it to the server tests.

The one skipped test trains a real short run whose corpus contains the breakout
payload, serves it, and asserts the data block is clean and all seven checkpoints
decode. It is opt-in because it takes about 15 seconds:

```bash
GLASSBOX_SLOW_TESTS=1 $P -m unittest tests.test_server.EndToEndRunTests
# Ran 1 test in 14.346s — OK
```
