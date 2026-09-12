## Current direction: reading-first (supersedes interaction requirements below)

The reader must be able to learn the conceptual journey without writing code, running
experiments, or passing quizzes. The original implementation brief below remains as
topic coverage and historical context; its mandatory interaction loops, experiment
forks, and GRPO unlock gate are no longer requirements for the reading path.

Use eight connected chapters with complete prose, worked numerical examples, failure
cases, explained mental-review questions, takeaways, and bridges. Default to the full
explanation. Offer summaries, optional derivations, a glossary, section bookmarks,
explicit reading progress, and a complete-book/print view. Essential explanations
must not depend on clicking a control. Preserve the recorded tensor reference lessons
as optional material. Label hypothetical calculations and measured traces separately.

Aim for deep conceptual understanding: explaining, predicting, and diagnosing. Do not
promise professional expertise or treat visits/completion as evidence of mastery.
Do not invent checkpoints, benchmark improvements, test execution, or GPU utilization.

## Original interactive implementation brief

You are implementing the next major version of this repository: a visual, browser-based course that teaches one continuous LLM journey:

**Tiny GPT pretraining → MoE → SFT/LoRA → RL environment design → GRPO/RLVR → agent evaluation → distributed scaling.**

First inspect the existing codebase and preserve its framework, styling, routing, reusable components, and current prototype functionality. Do not replace the project with a new scaffold. Implement this as a polished extension of the existing app.

## Product and teaching requirements

This is not a slide deck or a static explainer. Every module must follow this interaction loop:

> predict → run one controlled step → inspect exact values → explain what changed → modify one variable → compare the result

The learner should be able to see the same model evolve through the full course. Use fixed seeds, deterministic fixtures, immutable saved checkpoints, and visible numeric labels. Clearly distinguish:

* tokens
* activations / residual vectors
* model parameters
* gradients
* optimizer state
* rewards
* rollout trajectories
* KV cache

Do not use color alone to convey information. No timers, streaks, punitive lives, autoplay, distracting animations, or fake “mastery” indicators. Animations must be pauseable and optional. Preserve undo, reset, and fork/compare experimentation.

Provide three depth modes everywhere:

* **See it** — visual intuition
* **Explain it** — clear technical explanation
* **Derive it** — equations, tensor shapes, and exact calculations

Use the existing visual primitives where applicable, including tensor grids, vector sliders, probability bars, parameter diffs, gradient paths, rollout tables, verifier inspectors, checkpoint comparisons, and training timelines.

## Architecture

Create a maintainable course architecture:

* `CourseShell` with persistent progress, breadcrumb, depth selector, reset/fork controls
* a course map and module routes
* shared serializable lab state
* deterministic simulation engine using fixed small numerical fixtures
* separate lesson content/configuration from visual components
* local persistence for learner progress and saved experiments
* unit tests for core simulations and end-to-end tests for main flows

Do not pretend this is live GPU training. Label simulations honestly as “interactive deterministic training trace” or “recorded training trace.” Design interfaces so a future backend/GPU training service can be added without rewriting the UI.

## Build these modules

### 1. TinyGPT: pretraining from scratch

Create an end-to-end small-token training lab:

* tiny corpus and tokenizer view
* token IDs → embeddings + positions → attention → MLP → logits
* select one prediction target
* show cross-entropy loss numerically
* trace gradient flow backward
* show AdamW-style parameter update
* compare checkpoint 0, checkpoint 10, and checkpoint 100
* allow changing learning rate, batch, and one token in the prompt
* explain why lower loss is not automatically good generalization

### 2. Dense MLP to Mixture of Experts

Add an MoE lab that swaps a dense MLP for multiple MLP experts.

Required interaction:

* show a token residual vector entering a router
* router produces expert logits and probabilities
* user chooses top-k, initially 2 of 4 experts
* show masked router probabilities
* show selected expert outputs and weighted merge
* show capacity limits, dropped/rerouted tokens, and load balancing
* show the auxiliary load-balancing loss separately from language-model loss
* allow user to deliberately cause expert collapse, then repair it with balancing
* explicitly explain that experts are not clean human domains such as “coding” or “history”

### 3. SFT and LoRA

Create a post-training lab:

* compare base checkpoint and SFT checkpoint on the same task
* show successful teacher trajectory data
* show loss masking: prompt tokens versus assistant-answer tokens
* implement visual LoRA matrices A and B with `ΔW = B @ A`
* allow rank changes and show parameter-count / capacity tradeoff
* show base weights frozen while LoRA update changes behavior
* let learner compare base, LoRA adapter, and merged checkpoint

### 4. RL environment designer — must come before GRPO

Create the central RL-environment lab. The learner must define an environment before optimization is allowed.

The environment specification must visibly include:

* task and train/validation/held-out split
* initial state and reset behavior
* available typed tools/actions
* observations returned after actions
* episode state/session
* termination criteria and max-turn cap
* deterministic verifier
* reward timing: step-level versus final-episode
* failure states
* reward-hacking tests

Implement two environments:

1. **Structured-output environment:** JSON schema + correct-field verifier
2. **Mini code-repair environment:** pinned toy repository, file/shell-like tools, `submit_solution()`, and deterministic unit tests

The UI must prevent GRPO until the environment has a valid reset, termination rule, verifier, held-out tasks, and at least one anti-reward-hacking test.

### 5. GRPO / RLVR playground

Build an honest small-scale GRPO visual simulator over the environment above:

* generate a group of 8 rollouts for one prompt
* show each trajectory, reward, verifier evidence, token count, termination reason, and failure reason
* compute group-relative advantage visually
* identify which trajectories increase or decrease probability
* show a simplified clipped policy-update calculation
* compare pre-RL, post-SFT, and post-GRPO checkpoints
* include controls for group size, reward weights, reward sparsity, and rollout budget
* demonstrate reward hacking with a bad verifier, then show why a repaired verifier changes learning
* do not claim GRPO creates broad reasoning from tiny data

### 6. Agent evaluation and harness lab

Add a multi-turn agent lab where the model works through the mini code-repair environment:

* task → tool call → observation → next action → final submission
* show the full trajectory in a readable timeline
* include per-task pass rate, cost/tokens, number of tool calls, and completion time
* compare a naïve agent, SFT agent, and GRPO agent
* show train versus held-out results distinctly
* add a “coverage” metric: what portion of relevant files/tests the agent inspected
* include an explicit example where a short low-token trajectory fails because it gives up early

### 7. Scaling and systems lab

Create a conceptual but numerically grounded systems module:

* data parallelism
* tensor/model parallelism
* expert parallelism for MoE
* rollout workers versus training workers
* queueing and asynchronous rollouts
* checkpointing and recoverability
* environment servers versus in-process environments

Use adjustable worker counts and throughput/cost estimates. Show the tradeoff between more concurrent environments, slower long-horizon episodes, model updates, and GPU utilization. Never imply that more parallelism automatically improves quality.

## UX and quality bar

* Calm, clean, high-quality learning UI.
* One main question and one primary action per screen.
* Keep complex details progressively disclosed.
* Use diagrams plus exact small numeric examples.
* Make every visual keyboard accessible and responsive.
* Add empty/loading/error states.
* Add concise misconceptions/caveats in each lab.
* Include a “What changed?” panel after every interaction.
* Include a final course dashboard showing the model journey across all checkpoints.

## Deliverables

1. Implement the complete feature set in the existing codebase.
2. Add sensible routing/navigation and course progress.
3. Add deterministic fixtures and simulation logic.
4. Add tests for:

   * MoE top-k routing and load-balance calculations
   * LoRA update dimensions
   * JSON/code-environment verifier behavior
   * GRPO group-relative advantage calculation
   * train/held-out split isolation
5. Update the README with how to run, test, and extend labs.
6. Before finishing, run the app, run tests, fix errors, and provide a concise implementation summary listing changed files and any intentionally simulated components.

Prioritize a complete, coherent vertical slice over superficial placeholders. Implement the TinyGPT → MoE → environment → GRPO flow first, then complete the remaining modules using the same reusable architecture.
