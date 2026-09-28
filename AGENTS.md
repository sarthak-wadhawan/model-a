# AGENTS.md

## Product

This repository contains Model A, a small numeric ML research system for
short-horizon SPY prediction.

The current milestone is the five-point SPY/IEX microstructure MVP described in:

docs/MODEL_A_MVP_PRD.md

Read that document before modifying Model A microstructure code.

## Non-Negotiable Rules

1. Do not use future information in model features.
2. Do not redownload historical August 2026 data if the canonical cached data exists.
3. Do not rerun or replace the old Elastic-Net/model-selection pipeline.
4. Do not add indicators/features outside the MVP specification.
5. Do not perform architecture search or hyperparameter sweeps.
6. Do not submit brokerage orders.
7. Do not add options-trading logic during the MVP.
8. Do not treat 2026-08-25 as fresh out-of-sample data.
9. Fit normalization using training data only.
10. Preserve chronological splits.
11. Prefer minimal modifications over broad rewrites.
12. Existing code is not automatically correct. Validate it against the PRD and tests.

## Fixed MVP

Symbol/feed:
SPY / Alpaca IEX

Observation window:
09:35-12:00 America/New_York

Five-point sequence:
five consecutive one-minute summaries ending at decision time.

Features per minute:
- queue imbalance
- signed trade-volume imbalance
- relative spread
- midpoint return

Input dimension:
5 × 4 = 20

Model:
20 -> 8 tanh -> 1 sigmoid-equivalent probability

Prediction horizon:
5 minutes

Target:
UP iff future midpoint is strictly greater than decision-time midpoint.
Flat = DOWN.
Missing valid future midpoint = exclude.

Development sessions:
2026-08-18
2026-08-19
2026-08-20
2026-08-21
2026-08-24

Initial chronological validation:
2026-08-24

Historical demonstration:
2026-08-25

## Existing Data

Prefer existing canonical files:

artifacts/spy_iex_august/market_data/events.csv
artifacts/spy_iex_august/market_data/provenance.json

Do not automatically redownload these data.

## Engineering Style

Keep the implementation small and readable.

Prefer:
- explicit code
- small pure functions
- deterministic behavior
- typed/documented interfaces where useful
- focused tests
- machine-readable artifacts

Avoid:
- unnecessary frameworks
- elaborate abstractions
- duplicated feature logic
- hidden fallbacks
- silent data repair

## Verification

Before claiming completion:

1. run focused Model A microstructure tests;
2. run relevant existing regression tests;
3. verify no future leakage;
4. verify chronological splitting;
5. verify scaler fit uses training only;
6. verify historical prediction does not retrain;
7. report commands and actual results.

Never say "works" without running the relevant verification.

## Human Checkpoints

For substantial Model A changes:

First investigate and present a minimal plan.

Do not begin broad implementation until the product owner approves the plan.

When blocked, state the exact missing file, credential, dependency, or decision.
Do not replace missing real data with synthetic data unless explicitly instructed.