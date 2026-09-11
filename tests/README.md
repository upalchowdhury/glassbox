# glassbox tests

The original course material claimed that "the included training script asserts
causal-mask zeros, attention row sums, MLP multiplication, an SGD update
identity, a finite-difference gradient, invariance of earlier logits to a changed
future token, and assistant loss-mask alignment", and it also claimed browser
smoke tests. There were zero `assert` statements and no browser tests. This
directory makes the testing claim true, and cheap enough to re-run on every
change.

## Running it

Use the interpreter that has torch. The default `python3` on this machine has
neither torch nor fastapi.

**The verified command** — no third-party dependency, standard library only:

```
cd /Users/upalc/Documents/deeplearning-tuts/llmtuning
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m tests.run_checks
```

Add `-vv` for one line per test, `--failfast` to stop at the first failure, and
`--fast` to skip the slow end-to-end export class (see *Runtime* below).

Four other commands, every one of them actually run:

```
# plain unittest discovery, no custom runner  (-v optional)
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m unittest discover -s tests -t .

# one module
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m unittest tests.test_generate

# the glassbox modules only, without the server suite
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m unittest \
  tests.test_alignment tests.test_baselines tests.test_checks tests.test_data_tasks \
  tests.test_export tests.test_generate tests.test_lora tests.test_model \
  tests.test_preference tests.test_rlvr tests.test_tokenizer tests.test_windows

# pytest, which IS installed here (7.4.3) - the suite is plain unittest.TestCase
# classes, so pytest collects it unchanged and needs no conftest
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest tests -q
```

`tests/run_checks.py` exists so the suite cannot become un-runnable if pytest is
ever absent. Both runners execute the same tests.

## Measured results

Every row below was produced by running the command on this machine
(macOS, 16 cores, CPU only, torch 2.2.2, python 3.9.1). The counts include
`tests/test_server.py`, which belongs to the server work and is discovered
alongside these files; the glassbox modules alone are 203 tests, and `test_server.py` is 33.

Use the full interpreter path. Bare `python3` on this machine is 3.12.1 with no
torch, so every command below would fail at the first import.

| command | result | wall clock |
| --- | --- | --- |
| `/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m tests.run_checks` | 235 passed, 0 failed, 1 skipped, 0 expected failures (236 collected) | 19.8-32 s |
| `/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m tests.run_checks --fast` | PASS, 222 passed, 14 skipped | 10.2-16 s |
| `GLASSBOX_SLOW_TESTS=1 /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m tests.run_checks` | PASS, **236 passed, 0 skipped** | 32.6-48 s |
| `/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m unittest discover -s tests -t .` | OK, 236 run | 19.8 s |
| `/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m unittest tests.test_generate` | OK, 28 run | 6.3 s |
| the glassbox modules listed above | OK, 203 collected, 203 passed | 18.8 s |
| `/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest tests -q` | 235 passed, 1 skipped | 21.5 s |

The third row is the only one that leaves nothing out. `make test-all` runs it.

Three separate runs of the headline command took 19.8 s, 22.8 s and 26.1 s; the
spread is other work on the machine, not variance in the tests.

The one skip is the server suite's own opt-in end-to-end test
(`GLASSBOX_SLOW_TESTS=1`). The one expected failure is the recorded verifier gap
documented at the bottom of this file.

## Runtime and the fast mode

The fast mode is structural, not a reduced set of assertions:

* the pretrained model is trained **once per process** and cached as a state
  dict; every test that needs trained weights gets a fresh copy of it;
* pretraining runs for **400 steps**, not the 2000 a real run uses.

400 is not arbitrary. `test_baselines` asserts the model beats the bigram Laplace
baseline on validation, and that is the assertion that makes "the loss went down"
mean anything. Measured on this machine:

| pretrain steps | validation loss | bigram Laplace baseline | beats it? |
| --- | --- | --- | --- |
| 200 | 2.5867 | 2.3025 | no |
| 300 | 2.3269 | 2.3025 | no |
| **400** | **2.1794** | 2.3025 | **yes** |

So 300 steps would have produced a suite that passed while the model had not yet
earned its parameters. Override with `GLASSBOX_TEST_PRETRAIN_STEPS` if you want a
longer run; lowering it below ~350 will correctly fail `test_baselines`.

**The slow test** is `tests/test_export.py::ExportEndToEnd`. Its `setUpClass`
calls `build_run()` once at tiny step counts, which trains every stage
(pretrain, SFT, LoRA SFT, reward model, DPO, warm start, GRPO) and evaluates all
seven checkpoints, then 13 test methods share that one artifact. Skipping the
class takes the suite from 22.8-26.1 s to 15.0 s, so it is about 10 seconds. The next
most expensive thing is the shared 400-step pretraining run (~6 s, paid once for
the whole suite). No individual test outside those two costs more than about a
second.

```
GLASSBOX_SKIP_SLOW=1 /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m tests.run_checks
# or
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m tests.run_checks --fast
```

Environment knobs:

| variable | effect |
| --- | --- |
| `GLASSBOX_TEST_PRETRAIN_STEPS` | pretraining steps for the shared fixture (default 400) |
| `GLASSBOX_SKIP_SLOW=1` | skip `ExportEndToEnd` |

`tests/test_server.py` is a separate suite for `server.py` with its own
`GLASSBOX_SLOW_TESTS` switch; it is discovered alongside these files.

## What is covered

| file | what it pins |
| --- | --- |
| `test_checks.py` | every check in `checks.run_all` passes; the set of checks is pinned by name so one cannot be deleted silently; each clause of the blueprint's "asserts ..." sentence is mapped to the check that backs it |
| `test_tokenizer.py` | round-trip is exact; `<unk>` is not `<pad>`; `'<'`/`'>'` are not in the alphabet; every character of the chat template **including `':'`** is a real token and not silently `<pad>`; BPE fitted on train only differs from BPE fitted on everything |
| `test_model.py` | untrained cross-entropy within 0.05 of ln(vocab); KV cache matches the uncached forward pass relative to the logit scale; a changed future token leaves earlier logits bit-identical; the context guard raises (including when a cache offset pushes past it); same seed reproduces identical weights and a different seed does not |
| `test_windows.py` | every training window is the exact slice `ids[start : start+context+1]` of **one** document - stronger than the substring check, which can pass on a window that spans two documents |
| `test_alignment.py` | `pad_to_batch` shifts by exactly one; the first scored SFT target is produced by the **last prompt position**; the scored targets are exactly the answer tokens plus EOS; the displayed mask equals the optimised mask |
| `test_generate.py` | temperature changes the **entropy** of the next-token distribution, and argmax is invariant to it (which is how the original bug hid); top-k leaves exactly k candidates; top-p leaves a prefix of the sorted distribution; EOS stops decoding; a 74-token prompt is **not** truncated to 32 |
| `test_lora.py` | the adapter is an exact no-op at init; A has exactly zero gradient on the first step while B does not; base weights are bit-identical after adapter training; merge equals unmerge within float32 tolerance relative to the logit scale; the update has at most r non-zero singular values |
| `test_preference.py` | with policy == reference the DPO margin is exactly 0 and the loss is exactly ln 2 (in float32); a reward model starts at ln 2 and learns the ranking |
| `test_rlvr.py` | the strict verifier's refusals and the broken verifier's exploit; process checking catches a right answer from wrong arithmetic; zero-variance groups give all-zero finite advantages |
| `test_data_tasks.py` | `add` and `template` share no item at sizes 4, 8 and 18, and partition the whole 1-6 value space; provenance hashes |
| `test_baselines.py` | the trained model beats the bigram Laplace baseline on validation, and an untrained one loses to it |
| `test_export.py` | `build_run` at tiny step counts is JSON-serialisable with every stage key present; `inject_into_html` on a paragraph containing `</script>` yields a data block with no early terminator that still round-trips through a `JSON.parse` equivalent |

## Expected-failure policy

An `@unittest.expectedFailure` here means the assertion is at full strength and
the implementation does not satisfy it. The suite reports these separately and
does not count them as passes. **Do not weaken one to make it green** - either
fix the implementation or fix the claim it is testing.

**There are currently none.** The suite has 0 expected failures.

One was recorded and then fixed, which is the policy working as intended:

* `test_rlvr.py::StrictVerifier::test_it_rejects_two_answers_when_the_last_line_happens_to_be_right`
  held a failing full-strength assertion because `strict_verifier` parsed only
  the last non-empty line, so `"answer: 6\nanswer: 5"` scored 1 for the answer 5
  while `"answer: 5\nanswer: 6"` was correctly rejected. That made the verifier
  gameable: a policy unable to do arithmetic could enumerate every candidate and
  collect the reward. `strict_verifier` now requires exactly one answer line in
  the whole completion, and that line must be the last. The test is an ordinary
  passing test, and `test_it_rejects_enumerating_every_candidate_answer` and
  `test_it_rejects_commentary_after_the_answer` were added beside it.

## Notes

* These tests do not modify anything under `glassbox/`, and they never write to
  `runs/` or `app/index.html` - `test_export.py` writes into a temporary
  directory.
* Assertions about float equality are deliberate where they are exact
  (causal-mask zeros, the LoRA no-op, the DPO zero margin, bit-identical base
  weights) and relative to the logit scale where they are round-off
  (KV cache, LoRA merge). An absolute tolerance on the latter passes on an
  untrained model and then fails on a trained one purely because the logits
  grew, which is not a bug in the arithmetic.
