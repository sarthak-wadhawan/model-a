# Model A — Five-Point SPY Microstructure MVP

## 1. Product Vision

Model A is a small, deterministic, non-generative machine-learning system intended to investigate whether very recent SPY market microstructure contains enough information to predict the direction of SPY approximately five minutes into the future.

The system should be intentionally small, understandable, reproducible, fast, and resistant to data leakage.

This is a research prototype first.

The initial MVP does not execute trades and does not interact with brokerage trading endpoints.

The long-term product may eventually provide a directional signal that another system could use when evaluating SPY shares or options, but trading strategy logic is explicitly outside the scope of this MVP.

---

## 2. Business Problem

The user wants to answer a narrow question:

> Given the most recent five consecutive one-minute summaries of SPY microstructure, can a tiny neural network estimate whether SPY's midpoint will be higher or lower five minutes later?

We are intentionally testing a small hypothesis before adding complexity.

The objective is not to maximize backtest performance through feature engineering or architecture search.

The objective is to build a trustworthy experimental foundation.

---

## 3. MVP Output

For a prediction made at time `t`, return:

- symbol
- prediction timestamp
- direction: `UP` or `DOWN`
- `P(UP)`
- `P(DOWN)`
- the five input timestamps
- the 20 model inputs
- model identifier/version
- whether the result is historical or live/replay
- when historical, the actual five-minute outcome when available

For MVP classification:

- `P(UP) >= 0.50` → `UP`
- `P(UP) < 0.50` → `DOWN`

Do not manufacture confidence categories such as HIGH/MEDIUM/LOW.

Return the probability itself.

---

## 4. Market/Data Scope

Instrument:

`SPY`

Initial market-data provider/feed:

Alpaca IEX.

Important limitation:

IEX represents one exchange/feed and is not the entire consolidated U.S. equity market.

Do not claim that successful results establish profitability or a trading edge.

---

## 5. Existing Historical Data

Existing canonical data should be reused.

Do not automatically redownload the August dataset.

Known existing data locations are:

`artifacts/spy_iex_august/market_data/events.csv`

`artifacts/spy_iex_august/market_data/provenance.json`

Known development sessions:

- 2026-08-18
- 2026-08-19
- 2026-08-20
- 2026-08-21
- 2026-08-24

Historical demonstration session:

- 2026-08-25

August 25 has already been examined previously and must therefore be described as a historical demonstration/replay, NOT as a fresh untouched out-of-sample test.

---

## 6. Observation Window

Use regular-market data from:

09:35 ET through 12:00 ET.

A small amount of preceding data may be loaded solely when causally necessary to calculate the first valid feature at 09:35.

No future information may enter any feature.

---

## 7. Five-Point Input

Each model observation contains exactly five consecutive one-minute summaries.

For example, a prediction at 09:55 uses:

- 09:51
- 09:52
- 09:53
- 09:54
- 09:55

The sequence must contain five consecutive valid one-minute observations.

Do not replace missing minutes with future information.

The chronological ordering must always be preserved.

---

## 8. Features

Each one-minute observation contains exactly four microstructure features.

Therefore:

`5 minutes × 4 features = 20 numerical model inputs`

The four features are:

1. Displayed bid/ask queue-size imbalance
2. Signed trade-volume imbalance
3. Relative bid/ask spread
4. Midpoint return

The feature order must remain fixed and explicitly documented.

Do not add RSI, MACD, Bollinger Bands, VWAP, news, sentiment, options data, market indices, embeddings, or other indicators to this MVP.

### Queue Imbalance

Use displayed top-of-book bid and ask sizes according to the existing canonical provider representation.

Do not silently assume Alpaca IEX quote-size fields have different units than the provider supplies.

### Signed Trade-Volume Imbalance

Preserve causal trade-signing logic.

Preferred signing order:

1. use provider aggressor side when a valid side is supplied;
2. otherwise compare trade price with the most recent preceding available quote midpoint;
3. above midpoint → buy;
4. below midpoint → sell;
5. exact midpoint → causal tick rule;
6. unresolved trade signs remain unresolved/missing rather than using future information.

### Relative Spread

Calculate from the available causal bid and ask around the observation.

The implementation must document the exact mathematical definition and units.

### Midpoint Return

Calculate from causal midpoint information.

The implementation must document the exact mathematical definition and units.

If suitable implementations already exist in the repository, inspect and reuse them rather than creating subtly different definitions.

---

## 9. Prediction Target

Prediction horizon:

exactly five minutes.

At prediction time `t`:

`UP = midpoint(t + 5 minutes) > midpoint(t)`

Otherwise:

`DOWN`.

A flat midpoint is classified as DOWN for this fixed experiment.

If a valid future midpoint cannot be determined according to the target-alignment policy, the example must be marked unavailable and excluded from supervised training/evaluation.

Never use future events to construct the input features.

---

## 10. Model

The MVP model is fixed.

Architecture:

`20 inputs → 8 hidden units → 1 binary output`

Hidden activation:

`tanh`

Output:

sigmoid-equivalent probability of UP.

No architecture search.

No hyperparameter sweep.

No AutoML.

No ensemble.

No LSTM.

No attention.

No transformer.

No feature-selection search.

The purpose of the MVP is to determine whether this tiny model can learn useful information from the fixed microstructure representation.

---

## 11. Normalization

Numerical normalization/scaling must be fit using training data only.

Validation and historical demonstration data must never influence the scaler.

The fitted scaler must be persisted with, or otherwise reproducibly associated with, the trained model.

---

## 12. Chronological Training Design

Five development sessions:

2026-08-18  
2026-08-19  
2026-08-20  
2026-08-21  
2026-08-24

Initial chronological split:

Training:

2026-08-18 through 2026-08-21

Validation / early stopping:

2026-08-24

Historical demonstration:

2026-08-25

No random train/test split.

No shuffling dates across chronological boundaries.

Do not tune the model using August 25.

---

## 13. Baseline

At minimum compare the classifier against a simple development-set majority-class baseline.

The purpose is to determine whether the MLP is learning anything beyond blindly choosing the most common direction.

Do not hide baseline results when the model performs poorly.

---

## 14. Determinism

Use explicit random seeds wherever practical.

Repeated training using the same environment, dataset, and seed should produce the same or materially equivalent output.

Record important training configuration alongside the model.

---

## 15. Historical MVP Workflow

The first completed product milestone is historical-only.

The workflow should be:

raw cached events  
→ causal one-minute feature summaries  
→ five-point sequences  
→ chronological development split  
→ training-only normalization  
→ fixed tiny MLP  
→ saved model  
→ one historical prediction/replay  
→ machine-readable result

The canonical historical demonstration should be:

2026-08-25 at 09:55 ET.

The corresponding input sequence should be:

09:51–09:55 ET.

Again: this is a demonstration because that date has been seen in previous work.

---

## 16. Expected CLI Experience

Prefer a simple interface compatible with the existing project.

Desired conceptual operations are:

`micro-cache`

Create/reuse the one-minute feature cache.

`micro-train`

Train the single fixed tiny model.

`micro-predict`

Load the saved model and perform a historical prediction without retraining.

A later milestone may add:

`micro-live`

But live mode is NOT required for the first MVP milestone.

---

## 17. Required Tests

Automated tests must verify at minimum:

1. five input timestamps are consecutive and ordered;
2. exactly four features occur at each timestamp;
3. exactly 20 model inputs are produced;
4. feature construction uses no events occurring after its decision timestamp;
5. the prediction label is aligned five minutes forward;
6. missing future outcomes are excluded;
7. flat future midpoint is handled according to the fixed DOWN rule;
8. chronological development split is correct;
9. scaler is fit from training observations only;
10. August 25 does not influence training or early stopping;
11. historical prediction loads the saved model rather than retraining;
12. repeated prediction from identical inputs produces identical results;
13. malformed/missing data fails clearly rather than silently fabricating values.

---

## 18. Non-Functional Requirements

The implementation should be:

- small;
- readable;
- modular;
- deterministic;
- testable;
- CPU-friendly;
- runnable on an ordinary personal computer;
- free of unnecessary infrastructure;
- free of unnecessary dependencies.

Favor clear Python over abstractions that make the experiment harder to inspect.

---

## 19. Existing Code Policy

The repository already contains previous Model A code and experimental microstructure code.

Existing code is evidence and potentially reusable infrastructure, not unquestionable truth.

Before editing:

1. inspect the repository;
2. identify existing data-loading, Alpaca canonicalization, causal signing, feature construction, CLI, and tests;
3. determine which parts are trustworthy and reusable;
4. identify duplicated or contradictory implementations.

Reuse correct infrastructure where possible.

Do not rewrite unrelated parts of the project.

Do not rerun the old Elastic-Net/model-selection pipeline.

Do not intentionally reproduce an existing bug simply for backwards compatibility.

---

## 20. Explicit MVP Exclusions

Do NOT implement during this milestone:

- brokerage order submission;
- options selection;
- call/put purchasing;
- position sizing;
- automated trading;
- stop losses;
- profit targets;
- portfolio management;
- news;
- sentiment;
- LLM inference;
- reinforcement learning;
- transformer models;
- LSTMs;
- architecture searches;
- extensive feature engineering;
- multi-symbol support;
- premarket/postmarket support;
- full production deployment.

Those are later product decisions.

---

## 21. Human-in-the-Loop Development

The product owner remains the human decision maker.

Devin should work in explicit checkpoints.

### Checkpoint A — Investigation

Before modifying code, inspect the repository and report:

- relevant files;
- current architecture;
- current data flow;
- whether the historical data is accessible;
- existing reusable code;
- existing tests;
- contradictions between current implementation and this PRD;
- minimal implementation plan.

STOP and wait for product-owner approval.

### Checkpoint B — Historical MVP

After approval:

implement only the historical MVP.

Run all focused tests.

Run the historical demonstration.

Report:

- files changed;
- commands executed;
- tests passed/failed;
- training sample counts;
- validation sample counts;
- prediction;
- probability;
- actual historical direction;
- majority baseline;
- known limitations.

STOP for product-owner review.

### Checkpoint C — Later Work

Live prediction and further product development occur only after explicit product-owner approval.

---

## 22. Definition of Done

Milestone 1 is complete only when:

- the fixed five-point data representation is correct;
- no future leakage exists;
- the fixed 20→8→1 MLP trains;
- model/scaler artifacts are saved;
- historical prediction works without retraining;
- relevant tests pass;
- results are reproducible;
- August 25 is honestly labeled as a historical demonstration;
- no trading orders are submitted;
- implementation and commands are documented.

Model accuracy does NOT determine whether the engineering milestone is complete.

A correctly implemented model that performs poorly is still a valid experimental result.