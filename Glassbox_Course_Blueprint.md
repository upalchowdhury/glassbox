# Glassbox: train a model you can see
## Product, curriculum, and implementation blueprint

**Prepared:** 10 September 2026  
**Status:** Product proposal plus a working first teaching slice; not a completed course.  
**Working name:** Glassbox. This is a placeholder, not a checked or cleared public brand.

## 1. The product promise

A learner starts with a small, visible dataset and a model whose internals can be inspected. They follow the same model through pretraining, instruction fine-tuning, alternative preference/reinforcement-learning branches, evaluation, and inference. At any point they can answer:

> What went in? What numbers were calculated? What is learned and what is temporary? What objective is being optimized? What changed? What evidence says the model improved?

The product is a guided experimentation course, not a collection of spectacular animations or a chat interface that explains concepts in long paragraphs. Its distinctive unit of learning is **one prediction, one controlled intervention, and one inspectable result**.

The intended first learner has engineering experience but wants intuitive command of the mathematics. The course must also work for people who have not previously trained a neural network. Existing strengths in Python, cloud infrastructure, and application development should accelerate implementation; they should not force beginners through infrastructure before they understand a token vector.

### Two connected tracks

| Track | Purpose | Starting point | Evidence of success |
|---|---|---|---|
| Microscope | Understand every calculation and training signal | Randomly initialized, very small causal transformer; 5–10 original paragraphs | Learner can trace tensors, predict update directions, and distinguish training from inference |
| Practice | Run meaningful small-model adaptation and reasoning experiments | An appropriately licensed small pretrained model, initially around 0.5–3B parameters | Reproducible improvements on separately held-out tasks, with compute and failure reporting |

Use the same interface vocabulary and trace format across both. Do not silently substitute a pretrained model while claiming it learned language from the eight paragraphs. Do not promise that a few paragraphs teach broad language competence or robust reasoning.

The microscope is technically a **tiny language model**, not a large language model. It teaches mechanisms used in LLMs. The practice track adds the scale and prior learning that the microscope deliberately removes.

## 2. What is already built in this package

This package is a Python engine, one run artifact, and a browser app that replays it.

| Path | What it is |
|---|---|
| `glassbox/` | The reference engine in plain PyTorch: `data.py`, `tokenizer.py`, `model.py`, `generate.py`, `train.py`, `lora.py`, `preference.py`, `rlvr.py`, `evaluate.py`, `checks.py`. |
| `glassbox/export.py` | Runs every stage once and writes one self-describing artifact. `python -m glassbox.export` writes `runs/<run_id>.json` and `runs/latest.json`, and injects the run data into `app/index.html` when that file is present. |
| `runs/latest.json` | The artifact. Every number in the generated blocks below is read from it. |
| `app/index.html` | The browser app. It replays the artifact and computes no model quantity of its own. |
| `glassbox/checks.py` | The numerical checks of section 10. `python -m glassbox.export` exits non-zero if any of them fails. |
| `glassbox/report.py` | Regenerates this document's factual sections from the artifact. |
| `server.py` | A local FastAPI application for live experiments. Its endpoint contract lives with the code, not in this document. |
| `tests/` | The automated test suite. See section 10. |

What the app *renders* is described in sections 4 and 8. This document does not certify the app's visual inventory; it certifies the numbers the app is handed.

### Why this section's numbers are generated, not written

Every figure below is rendered from `runs/latest.json` by:

```bash
python -m glassbox.report --update Glassbox_Course_Blueprint.md
```

That command replaces only the text inside the paired `BEGIN generated:<name>` / `END generated:<name>` HTML comments below, and leaves all surrounding prose untouched. Running it twice produces a byte-identical file, so it is safe in a pre-commit hook or a CI check.

It exists because the previous version of this section was fabricated end to end. It stated a parameter count and a vocabulary size that matched no version of the code, loss figures nobody could reproduce, reference library versions that were not the ones installed, two filenames that did not exist, and seven numerical assertions that existed nowhere in the repository. None of that was dishonesty. It was one person typing numbers into prose that then outlived the code. Generated facts cannot drift that way: rebuild the run, rerun the report, and the document either matches or the diff states exactly what moved. Do not hand-edit inside those markers; the next run silently overwrites whatever is there.

`glassbox/report.py` reads JSON and imports no PyTorch, so this document can be regenerated on a machine that cannot run the engine.

One failure mode survives this design: the report will faithfully render whichever artifact it is pointed at, including a short smoke run. The provenance block at the end of this section names the run that produced every figure above it - its id, seed, hashes and step counts. If that block describes a run you do not recognise, regenerate from the one you meant; do not correct the numbers by hand.

### Configuration as it actually ran

<!-- BEGIN generated:configuration -->
<!-- Generated by `python -m glassbox.report --update` from runs/latest.json. Do not edit by hand; edit the run, then regenerate. -->

#### Architecture as it actually ran

| Field | Value |
|---|---|
| architecture | decoder-only causal transformer |
| layers | 2 |
| residual width | 8 |
| attention heads | 2 |
| head width | 4 |
| MLP width | 16 |
| context limit | 192 |
| vocabulary size | 45 |
| total parameters | 3,048 |
| position encoding | learned absolute |
| normalization | pre-LayerNorm |
| activation | GELU |
| input/output embedding tied | yes |
| dropout | 0.00 |
| initialisation std | 0.020 |
| seed | 42 |
| device | cpu |

An untrained model over 45 tokens has no preference between them, so its cross-entropy starts at ln(45) = 3.8067 nats. Every loss below is in nats per token and is comparable to that figure.

#### Data as it actually ran

| Field | Count |
|---|---|
| training paragraphs | 8 |
| validation paragraphs | 2 |
| training windows | 1,013 |
| validation windows | 218 |
| SFT instruction examples | 12 |
| warm-start demonstrations | 18 |
| preference pairs | 6 |
| character tokenizer size | 45 |
| subword tokenizer size (train-only fit) | 91 |
| subword merges learned | 60 |
| instruction family `train` | 12 |
| instruction family `repeat` | 6 |
| instruction family `paraphrase` | 6 |
| instruction family `heldout` | 5 |
| task family `add` | 8 |
| task family `template` | 8 |
| task family `compose` | 8 |

#### Stage hyperparameters as they actually ran

| Stage | Field | Value |
|---|---|---|
| `pretrain` | steps | 2000 |
| `pretrain` | batch_size | 16 |
| `pretrain` | learning_rate | 0.003 |
| `pretrain` | weight_decay | 0.01 |
| `pretrain` | warmup_steps | 100 |
| `pretrain` | grad_clip | 1.0 |
| `pretrain` | seed | 1234 |
| `pretrain` | optimizer | AdamW |
| `pretrain` | schedule | linear warmup then cosine decay |
| `pretrain` | loss | next-token cross-entropy, padding ignored |
| `sft` | steps | 400 |
| `sft` | batch_size | 12 |
| `sft` | learning_rate | 0.002 |
| `sft` | weight_decay | 0.01 |
| `sft` | warmup_steps | 50 |
| `sft` | grad_clip | 1.0 |
| `sft` | seed | 99 |
| `sft` | supervise_eos | true |
| `sft` | optimizer | AdamW |
| `sft` | loss | next-token cross-entropy on assistant-answer targets only |
| `lora` | rank | 2 |
| `lora` | alpha | 4.0 |
| `lora` | scaling | 2.0 |
| `lora` | targets | [q, v] |
| `lora` | b_init | zeros (adapter starts as a no-op) |
| `lora` | a_init | normal(0, 0.02) |
| `lora` | convention | W[out,in], A[r,in], B[out,r], delta_W=(alpha/r)*B@A |
| `reward_model` | steps | 300 |
| `reward_model` | learning_rate | 0.005 |
| `reward_model` | weight_decay | 0.0 |
| `reward_model` | seed | 7 |
| `reward_model` | loss | -log sigmoid(score_chosen - score_rejected)  (Bradley-Terry) |
| `reward_model` | head | single [width, 1] projection at the final response token |
| `dpo` | steps | 200 |
| `dpo` | learning_rate | 0.001 |
| `dpo` | beta | 0.1 |
| `dpo` | seed | 5 |
| `dpo` | loss | -log sigmoid(beta * [ (logp_policy(chosen) - logp_ref(chosen)) - (logp_policy(rejected) - logp_ref(rejected)) ]) |
| `dpo` | reference | frozen copy of the policy at DPO start |
| `dpo` | samples_during_training | false |
| `warm_start` | steps | 600 |
| `warm_start` | batch_size | 12 |
| `warm_start` | learning_rate | 0.003 |
| `warm_start` | weight_decay | 0.01 |
| `warm_start` | warmup_steps | 50 |
| `warm_start` | grad_clip | 1.0 |
| `warm_start` | seed | 99 |
| `warm_start` | supervise_eos | true |
| `warm_start` | optimizer | AdamW |
| `warm_start` | loss | next-token cross-entropy on assistant-answer targets only |
| `grpo` | iterations | 60 |
| `grpo` | group_size | 8 |
| `grpo` | learning_rate | 0.003 |
| `grpo` | clip_epsilon | 0.2 |
| `grpo` | kl_coefficient | 0.02 |
| `grpo` | temperature | 1.0 |
| `grpo` | top_k | 6 |
| `grpo` | max_new_tokens | 12 |
| `grpo` | seed | 3 |
| `grpo` | verifier | strict |
| `grpo` | inner_epochs | 4 |
| `grpo` | advantage | (reward - group mean) / (population std + 1e-8) |
| `grpo` | objective | clipped ratio * advantage, averaged over completions |
| `grpo` | variant_note | one concrete GRPO-style variant. Library defaults differ: normalisation, loss aggregation and the default KL coefficient have all changed between implementations. |
<!-- END generated:configuration -->

### Measured results, not promised outcomes

These are observations from one seeded run of a model with a few thousand parameters. They are evidence about mechanisms, not about capability. Two cautions apply to every table below.

- Answer-token loss and prose validation loss are computed over different distributions under different masks. Compare each against its own earlier value; never read one against the other as a task score.
- A two-paragraph validation set is a teaching diagnostic, not a statistically persuasive benchmark.

<!-- BEGIN generated:results -->
<!-- Generated by `python -m glassbox.report --update` from runs/latest.json. Do not edit by hand; edit the run, then regenerate. -->

#### Measured results, one row per stage metric

| Stage | Length | Metric | Start | End |
|---|---|---|---|---|
| Pretraining | 2,000 steps | train loss (next token, 8 paragraphs) | 3.8071 | 1.4840 |
|  |  | validation loss (2 held-out paragraphs) | 3.8090 | 2.1671 |
| Full SFT | 400 steps | answer-token loss | 2.8470 | 0.1438 |
|  |  | prose validation loss (retention) | 2.1671 | 4.2603 |
| LoRA SFT | 400 steps | answer-token loss | 2.8470 | 1.7672 |
|  |  | prose validation loss (retention) | 2.1671 | 2.6110 |
| Reward model | 300 steps | Bradley-Terry loss | 0.6931 | 0.0003 |
|  |  | pair ranking accuracy | 0.000 | 1.000 |
|  |  | mean score margin | 0.0000 | 8.1936 |
| DPO | 200 steps | DPO loss | 0.693147 | 0.000111 |
|  |  | implied-reward margin | 0.0000 | 10.8360 |
|  |  | pair accuracy | 0.000 | 1.000 |
|  |  | policy log-prob of chosen | -1.3015 | -31.0372 |
|  |  | policy log-prob of rejected | -158.6360 | -296.7314 |
| Warm start (SFT on task demonstrations) | 600 steps | answer-token loss | 5.5584 | 0.1365 |
|  |  | prose validation loss (retention) | 4.2603 | 7.2726 |
| GRPO on verifiable tasks | 60 iterations | mean verifier reward | 0.234 | 0.812 |
|  |  | clipped fraction | 0.859 | 0.281 |
|  |  | KL to reference | 1.7725 | 0.9715 |
|  |  | policy loss | 0.1435 | -0.0365 |

Best validation loss during pretraining was 2.0431 at step 500. The final step's validation loss is 2.1671, which is higher than the best. The last checkpoint is therefore not the best checkpoint, which is what makes early stopping a demonstrable lesson here rather than an assertion.

#### Did the transformer earn its parameters?

| Model | Train loss | Validation loss |
|---|---|---|
| uniform over the vocabulary | 3.8067 | 3.8067 |
| unigram + Laplace (train-split counts) | 2.8633 | 2.8704 |
| bigram + Laplace (train-split counts) | 2.2449 | 2.3025 |
| **the `base` checkpoint** | **1.4840** | **2.1671** |

All four are scored on exactly the same 12,006 validation positions, and the two count models were fitted on the training split only.

The model beats all three (uniform 3.8067, unigram 2.8704, bigram 2.3025), so the falling curve is evidence of learning and not just evidence of leaving a broken initialisation.

#### Preference optimisation, pair by pair

| Pair | policy log p(chosen) | reference log p(chosen) | policy log p(rejected) | reference log p(rejected) | margin |
|---|---|---|---|---|---|
| 0 | -8.3794 | -0.4262 | -240.8289 | -135.2473 | 9.7628 |
| 1 | -24.7707 | -0.5160 | -357.0306 | -226.0511 | 10.6725 |
| 2 | -35.8732 | -1.0681 | -188.8516 | -50.3134 | 10.3733 |
| 3 | -38.0847 | -1.2113 | -631.1926 | -425.4499 | 16.8869 |
| 4 | -20.9726 | -3.0703 | -194.0939 | -79.1892 | 9.7002 |
| 5 | -58.1426 | -1.5172 | -168.3908 | -35.5652 | 7.6200 |

In 6 of 6 pairs the trained policy assigns the CHOSEN answer a LOWER log-probability than the frozen reference did. DPO's margin is a difference of differences: pushing the rejected answer down far enough raises the margin even while the preferred answer also becomes less likely. The margin improved; the chosen answer did not.

#### GRPO, including the parts that did not work

| Quantity | Value |
|---|---|
| iterations | 60 |
| mean reward, first -> last iteration | 0.234 -> 0.812 |
| best mean reward | 0.859 at iteration 51 |
| zero-variance groups | 150 of 480 groups (31.2%) |
| maximum clipped fraction in any iteration | 0.859 |
| KL to reference, first -> last | 1.7725 -> 0.9715 |
| verifier | strict |
| reward rule | final line must parse as exactly one integer equal to the answer |

150 of 480 sampled groups had zero reward variance. Every completion in such a group scores the same, the group-relative advantage is exactly zero, and that iteration's policy gradient carries no information from those prompts. That is roughly 31.2% of the sampling budget spent for no learning signal - the single most common way a small-model RLVR run quietly fails.

#### Three verifiers on the same completions

| Completion | Answer | strict | substring (broken) | process |
|---|---|---|---|---|
| `answer: 5` | 5 | 1 | 1 | 1 |
| `answer: 25` | 5 | 0 | 1 | 0 |
| `2 + 3 = 5.\nanswer: 5` | 5 | 1 | 1 | 1 |
| `2 + 3 = 8.\nanswer: 5` | 5 | 1 | 1 | 0 |
| `the answer is 5` | 5 | 0 | 1 | 0 |
| `answer: 5\nanswer: 6` | 5 | 0 | 1 | 0 |
| `answer: 1\nanswer: 2\nanswer: 3\nanswer: 4\nanswer: 5` | 5 | 0 | 1 | 0 |
| `answer: 5\nthanks!` | 5 | 0 | 1 | 0 |
| _(empty completion)_ | 5 | 0 | 0 | 0 |

The broken substring verifier awards reward for `answer: 25` when the answer is 5, and the process verifier refuses a correct final answer reached through wrong intermediate arithmetic. A reward number is only as meaningful as the function that produced it.
<!-- END generated:results -->

### What each checkpoint can actually do

<!-- BEGIN generated:evaluation -->
<!-- Generated by `python -m glassbox.report --update` from runs/latest.json. Do not edit by hand; edit the run, then regenerate. -->

Every checkpoint is evaluated under one fixed decoding protocol (greedy, max 24 new tokens), so the columns are comparable to each other.

#### Retention: next-token loss on the pretraining prose

| Metric | `initial` | `base` | `sft` | `lora` | `dpo` | `warm_start` | `rl` |
|---|---|---|---|---|---|---|---|
| `train_prose_loss` | 3.8071 | 1.4840 | 4.1389 | 2.3327 | 5.9886 | 7.4372 | 9.4363 |
| `validation_prose_loss` | 3.8090 | 2.1671 | 4.2603 | 2.6110 | 6.4067 | 7.2726 | 9.0946 |

#### Instruction following: exact match by family

| Metric | `initial` | `base` | `sft` | `lora` | `dpo` | `warm_start` | `rl` |
|---|---|---|---|---|---|---|---|
| `train` | 0.000 | 0.000 | 0.917 | 0.000 | 0.000 | 0.000 | 0.000 |
| `repeat` | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `paraphrase` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `heldout` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

#### Verifiable tasks: pass@1 by family

| Metric | `initial` | `base` | `sft` | `lora` | `dpo` | `warm_start` | `rl` |
|---|---|---|---|---|---|---|---|
| `add` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 |
| `template` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 |
| `compose` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

| Task family | What it holds out |
|---|---|
| `add` | template and values a warm start may use |
| `template` | same template, values disjoint from 'add' |
| `compose` | two operations; a structurally held-out composition |

pass@1 here means 1 attempt per prompt under the protocol above, with the tokens spent recorded per item in the artifact. It is not best-of-N.

#### One trained question at every checkpoint

Prompt `how many red stones does pip have?`, expected answer `two.`, same greedy protocol throughout.

| Checkpoint | Greedy output | Exact match | Stopped on EOS |
|---|---|---|---|
| `initial` | `b.00000nnnf\n\n\ntttttttttt` | no | no (hit the limit) |
| `base` | `s stoures stoures s stou` | no | no (hit the limit) |
| `sft` | `two.` | yes | yes |
| `lora` | `the the the the the the ` | no | no (hit the limit) |
| `dpo` | `.` | no | yes |
| `warm_start` | `6` | no | yes |
| `rl` | `6` | no | yes |

At `sft`, the best instruction checkpoint (0.917 exact match on the trained family), 1 item still fails:

| Prompt | Expected | Produced |
|---|---|---|
| `where are the blue stones?` | `in the square box.` | `an the square box.` |

_Prose loss and instruction accuracy measure different distributions under different masks. Compare each metric with its own earlier value, never across rows._
<!-- END generated:evaluation -->

### Full SFT against LoRA: the trade-off this run measured

This is the comparison the Microscope track exists to make, and the one result in this section that should change how a reader thinks rather than merely confirm that the arithmetic runs.

<!-- BEGIN generated:lora -->
<!-- Generated by `python -m glassbox.report --update` from runs/latest.json. Do not edit by hand; edit the run, then regenerate. -->

The two runs see the SAME data with the SAME objective and the SAME number of steps. They differ only in which parameters are allowed to move.

| Quantity | Full SFT | LoRA SFT |
|---|---|---|
| trainable parameters | 3,048 | 128 (4.0% of 3,048) |
| answer-token loss, start -> end | 2.8470 -> 0.1438 | 2.8470 -> 1.7672 |
| prose validation loss, start -> end | 2.1671 -> 4.2603 | 2.1671 -> 2.6110 |
| instruction exact match, `train` family | 0.917 | 0.000 |
| instruction exact match, `repeat` family | 1.000 | 0.000 |
| instruction exact match, `paraphrase` family | 0.000 | 0.000 |
| instruction exact match, `heldout` family | 0.000 | 0.000 |

LoRA fit the task less well (1.7672 against 0.1438 on answer tokens) and damaged the pretrained behaviour less (2.6110 against 4.2603 on held-out prose). Both halves of that sentence are measurements from this run. Neither is a general law about LoRA: at 4% of a 3,048-parameter model the adapter simply does not have the capacity to overwrite much, which is exactly why it overwrites less.

The cost of forgetting less, stated plainly: the LoRA checkpoint answers 0.000 of the trained instructions exactly, against 0.917 for full SFT. Halving the answer loss was not enough to produce one correct answer. 'Forgot less' here partly means 'changed less', and changing less includes not learning the task.

The update each adapter applies is rank-constrained as advertised: realised matrix ranks [2, 2, 2, 2] against a configured rank of 2, attached to [blocks.0.q, blocks.0.v, blocks.1.q, blocks.1.v].
<!-- END generated:lora -->

Read that as a result about this run, not about LoRA. A rank-2 adapter on a 3,048-parameter model is a severe capacity constraint, and on a real pretrained model the same experiment can show full fine-tuning winning on both axes or LoRA matching it on both. What transfers is the shape of the question - which parameters were allowed to move, and what did that cost on each measured axis - not a ranking of the two methods.

### What this run does not show

<!-- BEGIN generated:limits -->
<!-- Generated by `python -m glassbox.report --update` from runs/latest.json. Do not edit by hand; edit the run, then regenerate. -->

The artifact records these limits with the run:

- A two-paragraph validation set is a teaching diagnostic, not a benchmark.
- Twelve instruction examples measure memorisation. Held-out accuracy is reported separately and is low.
- Six preference pairs cannot establish that a reward model generalises.
- This model has a few thousand parameters. It demonstrates mechanisms, not capability.

Measured in this run, and unflattering:

- Held-out instruction accuracy is 0.000 at every one of the 7 checkpoints. Nothing in this pipeline has demonstrated instruction following that generalises beyond the wording it was trained on.
- Paraphrased versions of TRAINED questions also score 0.000 everywhere. The SFT checkpoint reproduces memorised strings; it does not answer the same question asked differently.
- The `compose` task family - two chained operations, structurally held out - is never solved: pass@1 is 0.000 at every checkpoint, including after GRPO. Genuinely held-out composition is an open failure in this run, not a gap left for brevity.
- DPO reached pair accuracy 1.000 and a margin of 10.8360 while instruction exact match on the trained family fell from 0.917 to 0.000. The preference objective was optimised. The behaviour anyone cared about got worse.
- GRPO raised pass@1 on the `template` values it was optimised against to 0.875, and in the same run pass@1 on the warm-start `add` values fell from 0.875 to 0.000. Reward went up; the previously solved family was lost.
- Prose validation loss at the `rl` checkpoint is 9.0946 against 2.1671 at `base`. Catastrophic forgetting here is a measurement, not a warning in a footnote.
- The reward model reaches ranking accuracy 1.000 on the same 6 pairs it was trained on, and is never evaluated on a held-out pair. That number measures fitting, not generalisation, and no claim about reward-model quality should be drawn from it.
<!-- END generated:limits -->

### Deliberately not implemented yet

Implemented and measured since this document's first draft, which listed all four as absent: LoRA training, reward-model training, DPO, and GRPO policy updates.

Still genuinely absent:

- Training or arbitrary-prompt inference inside the browser itself. `app/index.html` replays `runs/latest.json`; any live computation runs in the Python process behind `server.py`.
- QLoRA, and any quantized base representation.
- PPO with a learned critic. The only policy-gradient implementation here is one GRPO-style variant; there is no value head and no critic anywhere in `glassbox/`.
- Reasoning distillation, and reasoning-trace SFT.
- Process-reward-model *training*. `rlvr.py` contains a process *verifier*, which is a hand-written function, not a learned model.
- Distributed training, mixed precision, and quantization. The engine is CPU float32 throughout.
- A hosted job service, accounts, and per-user run authorization.
- A complete graded curriculum. Section 6 is a proposal; no chapter has authored lessons.
- Modern architecture variants. RMSNorm, RoPE, SwiGLU, GQA/MQA and MoE are all absent from `model.py`, which is pre-LayerNorm with learned absolute positions and GELU.

The Python engine trains the model. The browser replays its results. **Do not rename a replay control “Train now.”**

### Provenance

<!-- BEGIN generated:provenance -->
<!-- Generated by `python -m glassbox.report --update` from runs/latest.json. Do not edit by hand; edit the run, then regenerate. -->

| Field | Value |
|---|---|
| run id | run_908d2b0535c2 |
| artifact schema version | 1.0 |
| seed | 42 |
| dataset sha256 (all splits, instructions, preferences, tasks) | `72acfc603b2e8d196d9da228c41ac0910b301d02a3c1f03b8db3384e843e2f72` |
| tokenizer sha256 | `aa0bdeb88f34e8de2a7e9126cd708e2cb29969222505ddc968e87f3bfda37781` |
| train corpus sha256 | `e2f7525715c8a8d7863ed21e52b6ce96b5b4206c9a5d444e33f9d60212be540a` |
| validation corpus sha256 | `4b20f6341cd55287628aa5c1f25c4829bf9facb3f880c7470ed5111f24fc22cf` |
| PyTorch | 2.2.2 |
| Python | 3.9.1 |
| platform | macOS-26.5.1-x86_64-i386-64bit |
| device | cpu |
| total parameters | 3,048 |
| approximate build time | about 90 s on one CPU |
| numerical checks | 26/26 passed |
| display rounding in this document | 7 decimals stored in the artifact; tables above round further for reading |

Reproduce it with `python -m glassbox.export`, which writes `runs/<run_id>.json` and `runs/latest.json`. These are the versions this artifact was produced with, not a claim that any other version is bit-for-bit identical.
<!-- END generated:provenance -->

## 3. Learning experience

### One lesson loop

**Predict → run one step → inspect the difference → explain it → change one variable.**

A proposed lesson should take roughly 3–7 minutes on its main path, but this is a design target, not a deadline or a measured learning claim. No required timer. Stop points should be explicit. Longer derivations and experiments belong behind optional controls.

Example: “What changes when an MLP expands a token vector?”

1. Show one three-coordinate token vector and five empty output boxes.
2. Ask whether five coordinates require five copies of the token or five weighted combinations.
3. Reveal one column of the weight matrix and calculate one output value.
4. Reveal the other four columns only after the learner is ready.
5. Apply an activation, then project back to three coordinates.
6. Ask whether the returned values must equal the original values.
7. Add the update to the original vector and let the learner change one input coordinate.

The first wrong answer should trigger a smaller numerical example, not a longer lecture.

### Three depth settings, one lesson

| Setting | What the learner sees |
|---|---|
| See it | Small vectors, highlighted cells, one sentence explaining each movement |
| Explain it | Exact shapes, operation names, input/output roles, and controlled experiments |
| Derive it | Equations, numerical checks, gradients, implementation details, and corresponding source code |

Progressively reveal depth. Do not build three disconnected curricula. Let learners move between settings without losing the current token, weight, checkpoint, or experiment.

### Low-fatigue interaction rules

Use a calm default layout, one central visual, one main question, and one primary action. Keep a persistent context breadcrumb: checkpoint → layer → operation → batch → token → channel/head. Every “reset” should state what it resets. Animation is optional and paused by default. No auto-spinning 3D objects. No forced streaks, countdowns, or punitive lives. Keep an undo/fork path so experimentation feels safe.

Use earned milestones tied to actual skills: “I can trace a token,” “I can predict a gradient sign,” “I can catch a broken verifier.” Do not award mastery for merely clicking Next.

Optional narrated explanations need transcripts and speed controls. An AI tutor is not necessary for the first release. If added, it must be grounded in the selected tensors, objective, and lesson content, and must not invent semantic interpretations of channels.

## 4. Visual grammar: make shapes understandable

### Separate five kinds of objects

| Object | Meaning | Lifetime |
|---|---|---|
| Input/token IDs | Discrete symbols selected by tokenization | A sequence or batch |
| Parameters | Learned embeddings, projection matrices, normalization parameters | Persistent across ordinary inference calls |
| Activations | Input-dependent intermediate numerical values | A forward computation; some may be cached |
| Gradients | Local derivatives of an objective with respect to differentiable values | Backpropagation/update computation |
| Optimizer state | Momentum/moment estimates and other update bookkeeping | Across optimizer steps; not the same as model parameters |

A separate KV-cache object belongs to inference state. Do not visually merge it with learned weights or a long-term memory system.

Show numeric values and labels as well as color. Never rely on hue alone for sign, masking, frozen/trainable state, or success. Round only for display. Report the precision used in calculations.

### Three views, each with a different job

**Matrix view:** default. It should be easy to read, select, and compare numbers. Every axis has a name, index range, and meaning.

**Tensor-stack view:** a spatial layout of slices. For an activation tensor `[B,T,D]`, select or stack batches of `T×D` matrices. For attention `[B,H,T,T]`, choose a batch and stack head matrices. Select layers and training steps using controls rather than pretending every extra axis is a spatial dimension.

**Vector-geometry view:** plot a genuine 2D/3D toy vector, or a clearly labeled projection of a higher-dimensional vector. A PCA view must state that information is omitted, identify the fitting dataset and basis, and keep the basis fixed when comparing checkpoints. Do not refit a projection separately per checkpoint and call the resulting apparent movement a learned direction.

A 3D arrangement of matrix sheets is not the same as representing the model's embedding space in three dimensions.

### Reusable visual components

`TensorGrid`, `TensorSliceSelector`, `DotProductInspector`, `VectorSliders`, `ResidualAddView`, `AttentionHeadStack`, `ActivationCurve`, `ParameterDiff`, `GradientPath`, `LossMaskStrip`, `ProbabilityBars`, `CheckpointCompare`, `RolloutTable`, `VerifierInspector`, and `TrainingTimeline`.

The value inspector should show: exact index; named axes; value and dtype; producing operation; operand references; relevant equation; checkpoint/run; and whether the value is computed live, replayed, or illustrative.

## 5. Data and model journey

Keep one small fictional world across the beginner modules. Pip's stones, boxes, lamp, and garden are sufficient. Use original or permissively licensed text; the eight paragraphs in this package are original teaching content.

### Pretraining data

Start with the complete readable paragraphs. Show character tokenization first, then contrast it with a separately implemented train-only subword tokenizer. The starter uses a fixed alphabet chosen in advance. Token IDs are arbitrary categories, not scalar magnitudes; ID 40 is not “twice” ID 20.

Teach next-token shift explicitly:

```text
input:   p i p ␠ h a s ␠
target:  i p ␠ h a s ␠ t
```

Show all supervised positions in a training batch, not only the last prompt token. Causal masking prevents an earlier position from seeing later targets, even when positions are computed in parallel.

Split data before creating overlapping windows. Track document/provenance IDs. Fit learned tokenizers on training data only. Freeze vocabulary IDs across continuation-training stages unless deliberately running a vocabulary-expansion lesson with embedding resizing and initialization.

### Instruction data

Convert some world facts into prompt–answer examples. Show formatting, role boundaries, end-of-sequence tokens, padding, truncation, and the shifted loss mask. A masked prompt remains available as context. Ignoring direct prompt-token loss does not remove gradients flowing through prompt representations.

The full product needs held-out instructions and held-out paraphrase families. One memorized answer must never become a progress badge labeled “instruction following mastered.”

### Preference data

Represent each example as a prompt plus chosen and rejected responses. A beginner can supply a few rankings to make the source of the signal concrete. Track annotator disagreement and distinguish human rankings from synthetic or AI-generated preferences.

```json
{"prompt":"How many red stones?","chosen":"Two.","rejected":"I will not answer this ordinary counting question."}
```

Preference for clarity and correctness are different properties. Teach that a preference label is not an objective proof that every factual statement in the chosen answer is correct.

### Verifiable task data

Use simple bounded counting/arithmetic tasks before general natural-language reasoning. Later add composition: moves between boxes, small expressions, short paths in a graph, or controlled symbolic transformations. The hidden answer is available to the verifier, not leaked into the policy prompt.

Provide deterministic generators with separate seeds and split families. Separate exact repeats, familiar templates with new values, and genuinely held-out compositions. Do not equate random paraphrase variation with a strong out-of-distribution test.

The tiny model may need task examples and a warm start before it generates any successful rollouts. When every sampled completion is wrong and receives the same reward, a group-relative objective has no relative reward advantage. Treat this as a prerequisite/curriculum failure to diagnose, not as evidence that reinforcement learning cannot work.

## 6. Curriculum: full lifecycle without one giant lesson

The proposed structure is about 50–60 short labs, with a core path and optional deeper labs. The counts are a scope target. Content should be cut or expanded based on learner tests, not to hit a marketing number.

| Chapter | Lessons and concepts | Signature interaction | Observable mastery | Engine support today |
|---|---|---|---|---|
| 1. Numbers with shape | Scalar, vector, matrix, tensor, axes, transpose, broadcasting, dot product | Fill one matrix-product cell | Explain `[B,T,D]` without calling a token a channel | `model.py` captures named-axis tensors; in the artifact |
| 2. Words become vectors | Vocabulary, character/byte/subword tokenization, IDs, embeddings, positions | Click text → ID → embedding row | Explain why embeddings are learned and IDs are not continuous features | `tokenizer.py`: 45-token character set and a train-only BPE fit |
| 3. One token looks around | Q/K/V, scaling, causal mask, softmax, weighted values | Change one visible source token; inspect a head | Distinguish attention weights from values and from final-output explanations | `model.py` per-head attention capture; checks 1, 2, 4 |
| 4. Transformer building blocks | Heads, concatenation/output projection, MLP expansion/down-projection, ReLU/GELU/SiLU, residuals, LayerNorm | Follow one token through one complete block | Explain token mixing versus channel mixing and why shape restoration is not inversion | `model.py` block capture; the MLP dot-product check (3) |
| 5. How learning happens | Targets, logits, cross-entropy, chain rule, autograd, gradients, SGD, AdamW, learning rate | Follow loss back to one weight, then take one update | Identify which quantity is a derivative and predict the sign of an SGD change | `train.py` one-step SGD demo; checks 5, 6, 7 |
| 6. Pretrain a tiny model | Batch/context/epoch/step, shuffling, splits, schedules, clipping, dropout, checkpointing, overfitting | Train, rewind, and compare held-out loss | Distinguish memorization, optimization progress, and generalization | `train.py` pretraining, count-model baselines, a visible best-step |
| 7. Make text appear | Prefill, autoregressive decoding, greedy/temperature/top-k/top-p, EOS, KV caching | Generate one token, show reused K/V, compare cache on/off | Explain why inference normally changes activations/cache rather than weights | `generate.py`: greedy, temperature, top-k, top-p, KV cache; checks 10-13 |
| 8. Adapt the model | Continued pretraining versus instruction SFT, chat templates, loss masks, full FT, LoRA, QLoRA, forgetting | Freeze base weights, train adapter matrices, merge and compare | Distinguish the training objective from the parameter-update method | `train.py` + `lora.py`: full SFT and LoRA, merge/unmerge. QLoRA absent |
| 9. Preferences before RL | Human pair rankings, reward scores, reward-model training, DPO | Rank a pair and inspect both the RM loss and the DPO loss | Explain why DPO does not require a separately trained reward model in its standard form | `preference.py`: reward model and DPO, both measured in section 2 |
| 10. RLHF from the inside | Policy, state/action, rollout, reward, baseline, advantage, reference policy, critic/value estimate, PPO ratio/clipping | Track the same response through score → advantage → policy loss → update | Distinguish policy, old policy, reference model, reward model, and critic roles | **Not implemented.** No PPO, no critic, no value head anywhere |
| 11. RLVR and reasoning | Checkable reward, sparse reward, GRPO groups, normalization, zero-variance groups, KL choices, reward hacking | Repair a verifier and observe a reward/accuracy mismatch | Explain why a better reward score may not mean better task performance | `rlvr.py`: GRPO, three verifiers, zero-variance groups measured |
| 12. Reasoning training strategies | Reasoning SFT, outcome/process supervision, rejection sampling, distillation, inference budget, self-consistency, best-of-N | Compare training-time changes against extra inference attempts | Distinguish longer text, successful sampled answers, and demonstrated generalization | **Not implemented.** A process *verifier* exists; no learned process RM |
| 13. Modern architectures | RMSNorm, RoPE, SwiGLU, GQA/MQA, dense versus MoE, tied embeddings, attention-kernel optimization | Toggle one architectural choice; compare shapes and compute | Identify what changed mathematically versus what is only a faster implementation | **Not implemented.** `model.py` is pre-LN, learned positions, GELU only |
| 14. Scale, evaluate, deploy | Mixed precision, quantization, accumulation, activation checkpointing, data/tensor/pipeline parallelism, sharding, evaluation leakage, safety, serving latency/cost | Partition a training step; run a final checkpoint comparison | Give a reproducible report with tradeoffs, limits, and resource measurements | `evaluate.py`: pass@1 under one pinned protocol. Scale/parallelism absent |

Architectural and post-training concepts above are grounded in the primary sources in Section 15. The sequencing and interactions are product-design proposals.

The last column describes the reference engine in `glassbox/`, not lesson content. A chapter that names a module has the real numbers a lesson would need and nothing more: **no chapter has authored lessons, and the interaction column remains a proposal everywhere.** Chapters 10, 12 and 13 have no engine behind them at all.

### Final projects

The first project is an inspectable model trained from scratch, with a short explanation of why its training loss and quality differ. The second compares full SFT with LoRA on a held-out instruction task. The third implements a verifiable task experiment with warm-start and no-RL controls, then documents at least one verifier failure. Advanced projects compare a learned preference reward against objective task correctness, or transfer one verified training recipe to a small pretrained model.

Do not require a GPU cluster to complete the core course. The distributed-training chapter can use exact numerical partitions and replay traces before offering an optional real multi-device lab.

## 7. Mathematical and conceptual correctness requirements

### MLP and channel mixing

Use one explicit convention at a time. In a row-vector presentation:

```text
X:       [B,T,D]
W_up:    [D,F]
Z:       X @ W_up + b_up             [B,T,F]
A:       activation(Z)               [B,T,F]
W_down:  [F,D]
U:       A @ W_down + b_down         [B,T,D]
H_out:   H_in + U                    [B,T,D]
```

In a pre-normalized transformer, `X` here is a normalized version of the incoming residual, while the residual addition uses the unnormalized incoming residual at that sublayer. Do not accidentally add the update to the normalized vector while narrating the pre-norm architecture.

Each hidden output coordinate uses a different column of `W_up`. No random new weights are generated for each token. The same weights are applied across positions. Nonlinear activations operate on current values; they are not all-or-nothing “neuron switches.” Two affine projections with no nonlinearity between them collapse to one affine map.

Down-projection does not reverse up-projection. Residual addition provides a direct path but does not guarantee perfect preservation of every input feature. Distributed channels are not automatically clean, human-named semantic features.

### LoRA: rank is not head count

Preserve the common stored-weight convention used in PyTorch:

```text
W:       [D_out,D_in]              frozen
A:       [r,D_in]                  trainable
B:       [D_out,r]                 trainable
Delta_W = (alpha/r) * (B @ A)
y = x @ W.T + (alpha/r) * (x @ A.T) @ B.T
```

If the display transposes `W` to show row-vector multiplication, transpose the displayed update consistently and clearly label it. Never switch silently between `AB` and `BA`.

Low rank constrains the form of each adapted matrix update, not the number of attention heads changed. It does not guarantee small behavioral changes or prevent forgetting. With typical zero-initialized `B`, the initial adapter is a no-op; `A` can initially have zero gradient while `B` gets a gradient. Test this rather than teaching that both matrices must change on the first step.

QLoRA combines a quantized base representation with adapter training; it does not mean all parameters and all arithmetic are trainable four-bit values. Kernel/backend support must be verified.

### Three axes that must not be conflated

| Axis | Examples |
|---|---|
| Training signal/objective | Next-token likelihood, demonstrated answers, pairwise preferences, reward maximization |
| Parameter-update method | Full-model training, frozen backbone plus LoRA/adapters |
| Optimization algorithm | AdamW for a differentiable loss, PPO/GRPO policy objectives with a chosen parameter optimizer |

RLHF and RLVR primarily distinguish the origin of reward. PPO and GRPO are not reward sources. LoRA is not an alternative source of supervision. DPO is a direct preference-training objective; standard DPO does not require the classic reward-model-plus-online-RL pipeline.

### Reward models and RLHF

The classic demonstration is: SFT policy → collect human comparisons → train a reward model → optimize the policy against that reward with controls on deviation. A reward model maps a prompt/response to a score; it is not automatically a ground-truth judge.

Show the roles separately. The current policy is optimized. An old policy supplies rollout probabilities for an importance ratio. A reference policy anchors a chosen regularizer. A critic estimates a value/baseline in a PPO-style setup. A reward model supplies a learned score. Implementations may share components or store cached outputs, so do not claim these roles always require five independent full model copies.

Start with a one-action contextual bandit and a score-function gradient before full autoregressive PPO. During a language-model rollout, the next-token distribution is the policy and the prompt plus generated prefix is the state. Standard sampling/verifier code is not differentiated through; gradients flow through policy log-probabilities. The reward is treated as a scalar training signal.

### GRPO and RLVR

Use a plainly labeled teaching variant first:

```text
For one prompt, generate G candidate completions.
r_i = verifier(completion_i, hidden_answer)
A_i = (r_i - group_mean_reward) / (group_std + epsilon)
```

The starter uses population standard deviation. Other implementations/configurations make different normalization choices; expose them explicitly. This arithmetic alone is not full GRPO.

In a policy-update lab, also show sampled tokens, completion masks, old/current log-probabilities, ratios, objective aggregation, clipping when used, and optional reference regularization. Pin the exact algorithm variant. Current TRL documentation describes implementation choices that differ from original GRPO, including normalization/loss changes and a default zero KL coefficient; do not present one library default as the timeless definition of the algorithm.

An all-equal reward group has zero relative advantage. A separate regularizer can still create an update. A positive advantage encourages a response under the chosen local objective; it does not guarantee a monotonic probability or accuracy increase after every finite update of a shared neural network.

Strictly separate final-answer checking from process checking. A response with wrong intermediate arithmetic but the correct final answer can receive an outcome reward. Never call that proof of faithful reasoning.

### DPO

Show chosen/rejected sequence log-probabilities and a frozen reference comparison. The original DPO loss can be written as:

```text
L = -log sigmoid(beta * [
    log pi_theta(chosen|prompt) - log pi_ref(chosen|prompt)
  - log pi_theta(rejected|prompt) + log pi_ref(rejected|prompt)
])
```

Use sequence log-probabilities with explicit prompt/padding masks and EOS policy. Avoid replacing this with “add 1 to the probability of the good answer.” Standard offline DPO does not sample new completions during each optimization update; generating new preference data is a separate stage.

### Reasoning training and inference

A “reasoning model” does not necessarily introduce a new transformer layer type. Teach concrete recipes rather than a mandatory ladder of base → SFT → RLHF → DPO → RLVR. These can be alternative or interleaved stages. DeepSeek-R1 and R1-Zero are useful case studies precisely because their post-training paths differ; R1-Zero did not start from random weights.

Separate reasoning-trace SFT, verifier-driven policy optimization, reward-model/process-supervision approaches, rejection-sampled SFT, and distillation. Also separate changes in weights from test-time methods such as extra sampled attempts or a larger generation budget. Longer answers are not evidence of better reasoning by themselves.

## 8. Application architecture

### Recommended starting stack

**Frontend:** React + TypeScript for the main application, with a componentized SVG/Canvas 2D inspector. Add Three.js/React Three Fiber only for optional tensor layouts that demonstrably help. The included starter intentionally uses plain HTML/CSS/JavaScript to be easy to open and inspect; it is not a requirement to rewrite the training logic in JavaScript.

**Reference computation engine:** Python + PyTorch, with an intentionally readable tiny transformer and explicit instrumented operations. Use straightforward operations initially; fused kernels often hide the intermediate values the course is trying to expose.

**Local API:** FastAPI with a job abstraction and a trace event stream. A single local worker and filesystem-backed runs are enough initially. Stream compact progress events via server-sent events; fetch tensor chunks on demand. WebSockets are optional for genuinely bidirectional controls, not mandatory for every progress update.

**Mac development:** CPU is the deterministic teaching baseline. Test MPS for suitable local training; PyTorch documents the MPS device path. MLX LM is a separate Apple-silicon option for realistic-model inference and fine-tuning. Do not assume CUDA-specific trainers, quantization libraries, or inference kernels work unchanged on MPS/MLX. Keep backend adapters explicit.

**Public service later:** authenticated API, per-user run authorization, bounded workers, object storage for datasets/checkpoints/traces, and a database for runs/progress. A queue, PostgreSQL, and Redis may become useful at this stage, but are unnecessary prerequisites for the first local course.

### Logical flow

```text
Learner interface
    ├─ lesson state and mastery checks
    ├─ matrix/tensor/gradient visual components
    ├─ checkpoint comparisons
    └─ experiment controls
                  |
                  v
Experiment API / run controller
    ├─ validate configuration and limits
    ├─ immutable run/checkpoint IDs
    ├─ pause/cancel/fork semantics
    └─ choose execution backend
                  |
       +----------+-----------+
       |                      |
Tiny PyTorch worker       Practice-model adapter
    |                      |
Trace writer + metrics + artifacts
                  |
Named-axis trace store → selected slices → same visual components
```

The model engine produces numbers. The renderer displays them. **No component invents a loss curve because a trainer is not ready.**

### Three execution modes

| Mode | Use | Interface rule |
|---|---|---|
| Live computation | Small matrix operations or a real local training/inference job | Label backend; cancellation and bounds must work |
| Recorded replay | Instant lessons and expensive jobs | Show source run/checkpoint; input edits create a new run rather than changing a replay's purported result |
| Illustrative example | Hand-chosen numbers for intuition | Label it explicitly; never report its outputs as model performance |

A hosted model API that returns only text cannot supply authentic internal tensors. Use an instrumentable local/open-weight model for internal inspection. Do not fabricate inaccessible hidden activations.

## 9. Trace contract and data flow

Version the schema from the start. The included trace is a small prototype schema; the application version should support chunking, named axes, and exact provenance.

```json
{
  "schema_version": "1.0",
  "run_id": "run_042",
  "checkpoint_id": "base_step_400",
  "source": "live_computation",
  "phase": "forward",
  "step": 401,
  "operation_id": "blocks.0.mlp.up",
  "tensor_id": "run_042/401/blocks.0.mlp.up/output",
  "shape": [1, 8, 16],
  "axes": ["batch", "token", "mlp_channel"],
  "dtype": "float32",
  "input_tensor_ids": [".../normalized_residual", ".../up_weight"],
  "display_convention": "row_vector",
  "slice": {"batch": 0},
  "storage_key": ".../tensor_chunk.bin",
  "summary": {"min": -1.4, "max": 2.1},
  "provenance": {"seed": 42, "code_revision": "recorded_commit", "device": "cpu"}
}
```

Values in this schema example are illustrative, not measurements. Include source code revision, dataset/tokenizer hashes, hardware/backend, framework version, optimizer configuration, precision, loss mask, and parent checkpoint in actual run manifests.

Store only selected tensors at selected steps. Summary statistics can be frequent; full activation and gradient captures should be sparse and configurable. Detached CPU copies prevent accidental retention of full autograd graphs in a web session. Do not stream every element of a billion-parameter model every step. Separate full-precision storage from display rounding.

Before/after experiments need an immutable checkpoint, fixed input, explicit random seed, and a clear intervention. Editing a weight should fork a run and recompute downstream quantities, not mutate a recorded tensor without updating its dependents.

## 10. Test strategy and release acceptance

### Numerical correctness

| Test | Required behavior |
|---|---|
| Scalar versus matrix multiplication | Selected output equals its explicit multiply-and-sum reconstruction |
| Causal masking | Future positions have zero attention probability; changing a future input does not change earlier logits |
| Attention normalization | Per-head rows sum to one over allowed sources within a documented tolerance |
| Loss alignment | Each input position predicts the next token; padding and role masks have no off-by-one errors |
| Finite-difference gradient | Selected tiny-model derivatives agree with autograd within precision-appropriate tolerances |
| SGD update | `after = before - lr * gradient` for the actual updated parameters |
| AdamW | Tests use the implemented optimizer's moments, bias correction, and weight decay rather than claiming an SGD identity |
| LoRA | Frozen weights remain unchanged; adapters update; initial no-op and merged/unmerged outputs agree where expected |
| DPO | Synthetic chosen/rejected pairs produce the expected loss/log-ratio gradients |
| PPO/GRPO | Masks, detached old log-probabilities, reward/advantage construction, clipping, and chosen KL variant match a reference calculation |
| Zero reward variance | Finite values; no accidental NaNs; relative advantage behaves as documented |
| Verifier | Exact parsing rejects ambiguous/multiple answers, malformed output, and naive substring exploits |
| Inference | Repeated ordinary inference does not alter weight hashes; cache/no-cache outputs agree within tolerance with correct positions |
| Reproducibility | Same code/config/data/seed on the reference backend reproduces stored results within declared limits |

### Current checks

`glassbox/checks.py` is the executable half of the table above. Each check recomputes the quantity it asserts about against the live model and returns a record carrying the measured value, not only a boolean - so a check that passes still shows its arithmetic, and a reader can disagree with the tolerance rather than take the word "pass" on faith. `python -m glassbox.export` runs all of them on every build and exits non-zero if any fails.

<!-- BEGIN generated:checks -->
<!-- Generated by `python -m glassbox.report --update` from runs/latest.json. Do not edit by hand; edit the run, then regenerate. -->

`glassbox/checks.py` runs 26 numerical checks against the live model on every run. This table is the result of the run recorded in `runs/latest.json`: 26 passed, 0 failed. Each check recomputes the quantity it is asserting about - none of them compares a stored number to itself.

| # | Check | Result | Tolerance | Measured |
|---|---|---|---|---|
| 1 | `causal_mask_zeros` | pass | 0.0 | max_future_attention_probability=0.0; sequence_length=24 |
| 2 | `attention_rows_sum_to_one` | pass | 1e-06 | max_absolute_deviation_from_one=2e-07 |
| 3 | `mlp_cell_is_a_dot_product` | pass | 1e-05 | row=3; column=5; reconstructed=0.4096497; captured=0.4096497; absolute_error=0.0 |
| 4 | `future_token_invariance` | pass | 0.0 | max_change_at_earlier_positions=0.0; max_change_at_the_edited_position=6.887528; replaced_token_id=8 |
| 5 | `untrained_loss_equals_log_vocab` | pass | 0.05 | measured_loss=3.807371; ln_vocab=3.806662; vocab_size=45; absolute_difference=0.0007085; init_std=0.02 |
| 6 | `finite_difference_matches_autograd` | pass | 0.0001 | cell=[2, 5]; cell_selected_by=largest absolute gradient in the checke…; analytic=2.216676; finite_difference=2.216676; absolute_error=2e-07; relative_error=1e-07; dtype=float64; e… |
| 7 | `sgd_update_identity` | pass | 1e-07 | max_absolute_error=0.0; learning_rate=0.05; updated_parameter_elements=3048; total_parameter_elements=3048; all_parameters_updated=true |
| 8 | `gradient_demo_leaves_checkpoint_unchanged` | pass | - | weight_hash_before=9278a55809057171; weight_hash_after=9278a55809057171 |
| 9 | `loss_mask_alignment` | pass | - | scored_target_tokens=[t, w, o, ., … (+1)]; expected_tokens=[t, w, o, ., … (+1)]; matches=true; first_scored_index=53; prompt_length=53; alignment_ok=true; eos_is_scored=true |
| 10 | `kv_cache_equivalence` | pass | 1e-05 | same_tokens=true; max_logit_difference=1.14e-05; logit_scale=14.4625; relative_difference=8e-07; dtype=float32 (the same comparison is ~1e-17 …; cached_text=s.; uncached_text=s. |
| 11 | `inference_does_not_change_weights` | pass | - | weight_hash_before=9278a55809057171; weight_hash_after=9278a55809057171 |
| 12 | `temperature_changes_the_distribution` | pass | - | entropy_by_temperature={0.5: 1.568917, 1.0: 2.104102, 2.0: 2.671269}; monotonically_increasing=true |
| 13 | `top_k_restricts_candidates` | pass | - | candidates_kept_by_k={1: 1, 3: 3, 10: 10}; vocab_size=45 |
| 14 | `lora_initial_adapter_is_a_noop` | pass | 0.0 | max_logit_difference=0.0; trainable_parameters=128; frozen_parameters=3048 |
| 15 | `lora_first_step_gradient_asymmetry` | pass | - | per_adapter=[4 entries] |
| 16 | `lora_base_weights_stay_frozen` | pass | - | base_unchanged=true; attached_to=[blocks.0.q, blocks.0.v, blocks.1.q, blocks.1.v] |
| 17 | `lora_merge_equivalence` | pass | 1e-05 | relative_merged_vs_unmerged=1.4e-06; relative_unmerge_roundtrip=6e-07; logit_scale=12.20669; dtype=float32 (the same comparison is ~1e-17 … |
| 18 | `lora_update_rank_is_bounded_by_r` | pass | - | rank=2; realised_ranks=[2, 2, 2, 2]; singular_values=[4 entries] |
| 19 | `dpo_starts_at_ln2_with_zero_margin` | pass | 1e-06 | max_absolute_margin=0.0; loss=0.6931472; ln_2=0.6931472; absolute_difference=0.0 |
| 20 | `zero_variance_group_has_no_signal` | pass | - | all_correct={rewards: [1.0, 1.0, 1.0, 1.0], std: 0.0, advantages: [0.0, 0.0, 0.0, 0.0], …}; all_wrong={rewards: [0.0, 0.0, 0.0, 0.0], std: 0.0, advantages: [0.0, 0.0, 0.0, 0.0], …… |
| 21 | `verifier_rejects_ambiguous_and_exploitative_output` | pass | - | rows=[7 entries]; strict_behaves=true; broken_verifier_is_exploitable=true; process_catches_bad_reasoning=true; enumeration_hack_rejected=true |
| 22 | `unknown_characters_map_to_unk_not_pad` | pass | - | encoded=[3, 3]; unk_id=3; pad_id=0; no_angle_brackets_in_alphabet=true |
| 23 | `bpe_fitted_on_training_split_only` | pass | - | supported_fit_matches_train_only=true; fitting_on_all_documents_differs=true; train_only_merges=40; all_document_merges=40 |
| 24 | `verifiable_task_families_are_disjoint` | pass | - | overlaps_by_size={4: [], 8: [], 18: []}; sizes={4: {add: 4, template: 4, compose: 4}, 8: {add: 8, template: 8, compose: 8}, 18: {add: 18, template: 18, compose: 18}} |
| 25 | `training_windows_stay_inside_one_document` | pass | - | num_windows=1013; num_documents=8; windows_crossing_a_boundary=0 |
| 26 | `same_seed_reproduces_same_weights` | pass | - | seed_42_hash=1f4481a07bec2afa; seed_42_again_hash=1f4481a07bec2afa; seed_43_hash=a1fe1d85b4676546; reproducible=true; seed_actually_matters=true |

Descriptions of what each check asserts live beside the code in `glassbox/checks.py`; the artifact carries them too, under `checks.results[].description`.
<!-- END generated:checks -->

Three gaps in that coverage, so the table above is not read as fully satisfied:

- **AdamW is not checked.** The optimizer identity that is asserted is the SGD one (`sgd_update_identity`), against the deliberate one-step demo in `train.py`. AdamW drives every actual training stage here, and its moments, bias correction and decoupled weight decay are asserted nowhere.
- **The PPO/GRPO reference calculation is partial.** `zero_variance_group_has_no_signal` and the three verifier checks cover advantage construction and reward parsing. The clipped ratio, the completion mask and the detached old log-probabilities run on every GRPO iteration but are not compared against an independent reference calculation.
- **The DPO check covers the initial condition only.** It asserts that a policy equal to its frozen reference gives exactly zero margin and a loss of ln 2. It asserts nothing about the gradient direction afterwards.

### Automated tests

The automated suite lives in `tests/`, separate from the in-run checks above. Its cases are plain `unittest.TestCase` classes, so `python -m unittest discover -s tests -t .` runs them with no third-party dependency, and pytest collects them unchanged where it is installed. The suite trains real models, so it takes minutes rather than seconds; budget for that before assuming it has hung.

Anything that imports the engine needs an interpreter with PyTorch installed, which is not necessarily the default `python3` on a given machine. `glassbox/report.py` is the one exception: it reads JSON and imports no PyTorch, so this document can be regenerated anywhere.

Read the suite for what it covers, and run it for whether it passes. **This document deliberately quotes no test count and no pass rate** - a number typed here is exactly the failure mode section 2 exists to prevent, and unlike the figures in section 2 a test count has no artifact to generate it from.

**No browser testing has been performed.** An earlier version of this section claimed that browser smoke tests exercised all seven pages, checkpoint changes, 3D head layout, verifier behaviour and narrow mobile layouts. No such tests existed, in any form. There is at present no cross-browser check, no accessibility audit, no mobile-layout verification and no end-to-end test that drives `app/index.html`. The accessibility requirements in section 11 are therefore entirely unverified, and should be treated as requirements rather than as a description.

### Learning correctness

A novice should be able to explain the next step on a new example, not merely reproduce a memorized diagram. Test the following misconceptions directly: token versus channel; weight versus activation; token position versus last generated token; head versus KV head; loss versus reward; LoRA rank versus head count; reward model versus reference policy; and inference context versus permanent training.

Use a brief pre-lesson and delayed transfer question. Test the proposed pacing and interaction pattern with actual learners before making learning-effectiveness claims. Keep the course usable when a learner chooses not to share analytics.

### Model evaluation

For each checkpoint compare held-out next-token loss, task accuracy, formatting compliance, prior-task retention, response length, compute, and failure examples. For RL additionally report actual verifier-independent task correctness where possible, average reward, per-prompt reward variance, successful-group fraction, and KL/entropy when relevant.

Compare pass@1 under a fixed decoding protocol, not only the best of many attempts. For best-of-N or pass@k experiments, report the number of attempts and token budget. Preserve a test set not used to tune the lesson's winning configuration.

Section 2's per-checkpoint tables are the current implementation of this requirement. `glassbox/evaluate.py` reports held-out next-token loss, per-family exact match, pass@1 under one pinned greedy protocol, retention at every checkpoint, EOS rate as a formatting-compliance proxy, and mean response length; the artifact also records the attempt count and the tokens spent on each task item, so no pass@1 figure here is a disguised best-of-N. `glassbox/rlvr.py` records average reward, the zero-variance group count and KL per iteration. Not yet reported: per-checkpoint entropy, and any per-stage compute measurement beyond one elapsed time for the whole pipeline.

## 11. Public release requirements

Ship a static/replay version first to make access easy and costs predictable. Add hosted computation only after the training engine and cancellation/limits are tested. Do not require a learner to submit a credit card just to inspect a tensor.

Track licenses and attribution for papers, model weights, datasets, fonts, code, and any reused visuals. Link to primary sources and write original explanations. The example text and diagrams in this starter are not evidence that every external asset is cleared for a future commercial release. Review the final distribution's dependency licenses and choose a project license before publication.

Use trusted built-in verifier functions initially. Arbitrary submitted Python, shell commands, or generated code must not execute in the web/API process. A future coding-verifier lab needs a real sandbox with CPU/memory/time/network limits, ephemeral storage, restricted privileges, and an explicit threat model. Separate untrusted code execution from training workers and secrets.

For hosted jobs enforce per-user authorization, bounded dataset sizes, token/run budgets, cancellation, expiration, and isolation. A cancelled job must actually stop worker computation. Treat training text as user data; local-first processing, explicit upload consent, retention controls, and deletion paths should precede analytics. Do not silently log prompts or datasets into product telemetry.

Support keyboard access, readable high-contrast numbers, reduced motion, a non-3D alternative, touch-friendly targets, and meaningful screen-reader labels. Large matrices need slice navigation instead of making the whole page thousands of pixels wide.

## 12. Build order and go/no-go gates

| Milestone | Scope | Go/no-go gate |
|---|---|---|
| M0: concept slice | This starter, real traces, prototype lesson experience | Learner can trace a token and explain one weight update |
| M1: live core | Local tiny-model run controller, editable data, bounded training, inference, checkpoint fork/compare | A learner can perform one genuine end-to-end train → inspect → generate experiment without leaving the app |
| M2: adaptation | SFT mask editor, retention evaluation, LoRA, checkpoint lineage, DPO | Numerical tests pass and learner can distinguish objective from parameter-update method |
| M3: reinforcement learning | Bounded rollout generation, learned reward model, simple bandit primer, PPO lesson, GRPO/RLVR lesson | End-to-end reward → gradient → update is real, with working zero-variance and verifier-failure lessons |
| M4: reasoning/practice | Small pretrained-model adapter, warm-start tasks, distillation and test-time comparisons | Held-out results are reproducible and not driven by leakage or an unreported sampling budget |
| M5: public beta | Content editing/versioning, accessibility, licensed assets, onboarding, privacy, optional hosted jobs | External learners finish the core tasks; no hidden mock trainers; operational limits hold |

### Where M0-M5 stand today

| Milestone | Passes? | Why |
|---|---|---|
| M0: concept slice | **Partly** | The engine half is real: seven checkpoints with distinct recorded gradient steps, real traces, all numerical checks passing. The gate itself is a learner claim, and no learner has been tested. |
| M1: live core | **No** | The engine can train, generate, fork and compare checkpoints, and a local API exists. Whether a learner can complete one end-to-end experiment without leaving the app is not demonstrated anywhere in this document, and `app/index.html` is a replay of `runs/latest.json`. |
| M2: adaptation | **Engine half yes, learner half untested** | Assistant-only loss masking, retention evaluation at every checkpoint, LoRA with merge and unmerge, checkpoint lineage with parents, and DPO are all implemented, and the numerical checks pass. An interactive mask *editor* is not implemented. The gate's second clause is a learner claim and is untested. |
| M3: reinforcement learning | **Mostly, two scope items missing** | Reward to advantage to gradient to update is real end to end. The zero-variance lesson is not hypothetical: a measured share of sampled groups carried no signal. The verifier-failure lesson is a measured table, not an anecdote. Missing from the scope: the contextual-bandit primer, and the PPO lesson, since no critic exists. |
| M4: reasoning/practice | **No** | Warm-start tasks exist, the task families are provably disjoint, and pass@1 is reported under one pinned decoding protocol with the token budget recorded - so the anti-leakage half of the gate is genuinely met. But there is no pretrained-model adapter, no distillation, no test-time comparison, and the structurally held-out `compose` family is never solved at any checkpoint. |
| M5: public beta | **No** | None of content editing/versioning, accessibility work, licensed-asset review, onboarding, privacy controls, or hosted jobs exists. |

Three of these six gates are written as learner-outcome claims, and no quantity of passing numerics can satisfy them. They need learner testing, which has not happened. That is the same highest-risk unknown named below, and it is still unknown.

Do not build a multi-tenant GPU platform, payments, a general-purpose tutor, and dozens of 3D scenes before M1. The highest-risk unknown is whether the interaction genuinely makes hard concepts understandable. Test that first.

## 13. Lesson authoring schema

Keep content separate from rendering code. A lesson definition should refer to real engine operation IDs and validated trace data.

```yaml
id: mlp.expansion.one_token
version: 1
prerequisites: [shapes.dot_product, embeddings.token_vector]
objective: Explain how one token with 3 coordinates produces 5 hidden coordinates.
execution_mode: illustrative_example
example_id: mlp_3_5_3_v1
main_question: Where does the fourth output number come from?
steps:
  - select_input_token
  - reveal_weight_column
  - predict_dot_product
  - calculate_dot_product
  - reveal_remaining_columns
  - apply_activation
  - down_project
  - add_residual
misconceptions:
  - expansion_copies_the_token
  - down_projection_recovers_original_values
  - weights_are_randomized_for_each_token
transfer_check:
  task: Compute one output coordinate with a new input vector.
  tolerance: 0.0001
stop_point: You can now explain channel mixing for one token.
sources: [attention_2017, pytorch_autograd]
```

For live or replay lessons, include required tensor IDs and failure behavior when a tensor is unavailable. A lesson must fail visibly rather than substitute illustrative numbers without changing the source badge.

## 14. Product success definition

The first meaningful success is not “we rendered an entire transformer in 3D.” It is:

> A learner can point at a number, explain how it was calculated, predict how a controlled change will affect the next operation, and understand which observation would count as evidence of learning.

The full course should then extend that understanding from next-token training to post-training and inference without changing the meaning of its visual objects or concealing its simplifications.

## 15. Primary sources and implementation references

These are reference anchors, not a claim that the proposed curriculum or UI has been scientifically validated. Library documentation is changeable; pin and record the versions used by the implementation.

1. Vaswani et al. (2017). **Attention Is All You Need.** https://arxiv.org/abs/1706.03762 — transformer architecture, attention and feed-forward foundations.
2. PyTorch. **Automatic Differentiation with torch.autograd.** https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html — computational graphs and gradient mechanics.
3. Ouyang et al. (2022). **Training language models to follow instructions with human feedback.** https://arxiv.org/abs/2203.02155 — demonstration data, comparisons, reward modeling, RLHF pipeline.
4. Hu et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models.** https://arxiv.org/abs/2106.09685 — frozen pretrained weights and trainable low-rank updates.
5. Rafailov et al. (2023). **Direct Preference Optimization: Your Language Model is Secretly a Reward Model.** https://arxiv.org/abs/2305.18290 — direct preference training without an explicit intermediate reward model.
6. Schulman et al. (2017). **Proximal Policy Optimization Algorithms.** https://arxiv.org/abs/1707.06347 — PPO objectives and clipped policy updates.
7. Shao et al. (2024). **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.** https://arxiv.org/abs/2402.03300 — GRPO and mathematical post-training.
8. DeepSeek-AI et al. (2025). **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** https://arxiv.org/abs/2501.12948 — alternative reasoning post-training paths, RL and distillation.
9. Hugging Face TRL. **SFT Trainer.** https://huggingface.co/docs/trl/main/en/sft_trainer — conversational data, completion/assistant masks, implementation details.
10. Hugging Face TRL. **GRPO Trainer.** https://huggingface.co/docs/trl/main/en/grpo_trainer — algorithm details and evolving implementation choices.
11. Hugging Face PEFT. **LoRA.** https://huggingface.co/docs/peft/main/package_reference/lora — adapter conventions, initialization, targeting and backend-sensitive implementation details.
12. Ainslie et al. (2023). **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.** https://arxiv.org/abs/2305.13245 — query heads versus shared key/value heads.
13. PyTorch. **MPS backend.** https://docs.pytorch.org/docs/stable/notes/mps.html — Apple-silicon device integration; verify actual operator support.
14. MLX contributors. **MLX LM.** https://github.com/ml-explore/mlx-lm — Apple-silicon inference and model fine-tuning.
15. Georgia Tech Polo Club. **Transformer Explainer.** https://poloclub.github.io/transformer-explainer/ — reference for browser-based interactive tensor explanations, not a source of copied course content.
16. Brendan Bycroft. **LLM Visualization.** https://github.com/bbycroft/llm-viz — reference for spatial model visualization. Inspect licenses before any reuse.
