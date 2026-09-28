---
name: model-a-mvp
description: Investigate, implement, test, and verify the fixed five-point SPY/IEX microstructure MVP according to the Model A PRD.
argument-hint: <investigate|implement|verify>
triggers: ["user"]
---

# Model A MVP Procedure

Read these files first:

1. `AGENTS.md`
2. `docs/MODEL_A_MVP_PRD.md`

Treat the PRD as the product contract.

$ARGUMENTS determines the requested phase.

## Phase: investigate

Do not modify application code.

Inspect the repository for:

- existing Model A architecture
- event-data schema
- Alpaca/IEX provider code
- event canonicalization
- quote handling
- trade signing
- microstructure feature construction
- sequence generation
- target generation
- scaling
- training code
- prediction code
- CLI commands
- tests
- existing artifacts

Locate the existing canonical August data or determine whether it is unavailable
inside the current Devin environment.

Compare current behavior against the PRD.

Produce:

1. relevant file map;
2. current data-flow summary;
3. reusable components;
4. suspected problems or inconsistencies;
5. files that minimally need modification/addition;
6. exact implementation plan;
7. exact tests that will prove correctness.

Do not change code.

Stop for human approval.

## Phase: implement

Implementation is allowed only after the investigation plan has been approved.

Implement the smallest historical MVP satisfying the PRD.

Do not implement live trading or options logic.

Prefer reuse of validated existing ingestion/canonicalization logic.

Keep the new microstructure experiment isolated from unrelated legacy
model-selection code.

Requirements:

- five consecutive one-minute observations
- four fixed features each
- 20 flattened model inputs
- causal construction only
- five-minute target
- chronological split
- training-only normalization
- fixed 20 -> 8 tanh -> 1 MLP
- persisted model and preprocessing state
- historical prediction without retraining
- deterministic seeds
- JSON or similarly machine-readable prediction artifact

Add/update focused automated tests.

Run them before stopping.

Then report:

- files changed
- commands used
- tests
- sample counts
- remaining limitations

Stop for human review.

## Phase: verify

Do not redesign the model.

Audit the implementation against every relevant PRD requirement.

Specifically test for:

- future leakage
- timestamp alignment
- sequence ordering
- 20-input dimensionality
- target alignment
- train/validation contamination
- scaler contamination
- accidental use of Aug 25 during fitting
- prediction-time retraining
- nondeterminism
- fabricated handling of missing values

Run the historical 2026-08-25 09:55 demonstration if the required real cached
data is available.

Report model and majority-baseline results without hiding poor performance.

Do not change the model merely because the result is disappointing.