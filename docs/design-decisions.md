# Design decisions

For each decision: what was chosen, why, what the alternative was, and where the evidence is. Items 5 and 6 are incidents that changed the design. The last section lists what was tried and rejected (details in [evaluation.md](evaluation.md)) and what was deliberately not done.

1. [Log-return target, one direct model per horizon](#1-log-return-target-one-direct-model-per-horizon)
2. [Purge equal to the longest horizon](#2-purge-equal-to-the-longest-horizon)
3. [Expanding window, equal weights, yearly folds](#3-expanding-window-equal-weights-yearly-folds)
4. [Naive baselines first, and statistics sized to the data](#4-naive-baselines-first-and-statistics-sized-to-the-data)
5. [The obv train-serve skew](#5-the-obv-train-serve-skew)
6. [Point model and corridor are independent decisions](#6-point-model-and-corridor-are-independent-decisions)
7. [Two gates: offline holdout, then live shadow traffic](#7-two-gates-offline-holdout-then-live-shadow-traffic)
8. [Fixed slots in the object store](#8-fixed-slots-in-the-object-store)
9. [Schema changes get their own path](#9-schema-changes-get-their-own-path)
10. [One pure feature function](#10-one-pure-feature-function)
11. [Side channels never fail a run](#11-side-channels-never-fail-a-run)
12. [News: publication time, resumable passes](#12-news-publication-time-resumable-passes)
13. [Rejected and not done](#rejected-and-not-done)

## 1. Log-return target, one direct model per horizon

**Chosen.** `r_h = log(close[t + h] / close[t])`, 12 independent LightGBM models (`h = 1..12`), each fitted to its own target. The corridor is separate quantile models.

**Why.** A price target is dominated by the current price: whatever the model does, it will look excellent on error and tell nothing about the move. Log-returns are roughly stationary and additive, so horizons are comparable. A direct model per horizon means a forecast for 60 minutes does not run through the 55-minute output, so errors do not accumulate, and each horizon can be judged alone.

**Alternative not taken.** A recursive model (predict 5 minutes, feed it back) or one multi-output network. The repository does not test them. The cost of the chosen design is 12 times the training work per symbol and no sharing between horizons.

**Evidence.** [`lightgbm_model.py`](../src/pinance_ml/models/lightgbm_model.py); per-horizon results in [evaluation.md, section 2](evaluation.md#2-lightgbm-12-horizons).

## 2. Purge equal to the longest horizon

**Chosen.** Before every train/test boundary, remove `max(HORIZONS) = 12` training rows.

**Why.** The target of a row at time `t` looks up to 12 candles ahead. The last 12 training rows therefore have targets that reach into the test window; training on them leaks the test period into the model. Twelve is exactly the reach, no more (extra purge only discards data).

**Consequence worth knowing.** The 24 h experiment uses its own `LONG_PURGE_ROWS = 288`; folding 288 into the shared horizon list would have made the 5-to-60-minute models discard 288 rows per boundary for no benefit.

**Evidence.** `tests/test_splits.py::test_purge_removes_rows_immediately_before_test_boundary`; [`config.py`](../src/pinance_ml/config.py).

## 3. Expanding window, equal weights, yearly folds

**Chosen.** Train on all history before the test window. LightGBM folds are 365 days; the naive baseline uses 7-day folds.

**Why.** Both alternatives were measured on BTC and lost: a sliding window of 90 to 730 days is worse on MAE and DA at every length, and the damage grows monotonically as the window shrinks (90 days: MAE +4.5 %, DA -1.0 pp, p = 0.002); exponential recency weights (half-lives 30, 180, 730 days) are also worse. A model that has seen every regime beats one that has seen only the latest. Yearly folds exist for cost: 12 models on a 26,000-row fold took about 52 s, scaling roughly linearly, and 7-day folds would mean about 400 retrains per symbol on near-identical expanding sets.

**The cost of the choice.** Eight folds per symbol give coarse statistics (see 4).

**Evidence.** [evaluation.md, sections 3 and 4](evaluation.md#3-training-window-length). The retraining cadence in production ([deployment.md](deployment.md)) is a different question: one more day moves the expanding set by about 0.03 %, so daily retraining mostly refreshes the most recent regime rather than changing the model.

## 4. Naive baselines first, and statistics sized to the data

**Chosen.** Every model is reported against `r = 0` (error size) and the majority-sign constant (direction), on the same windows. Per-fold values are pooled with `n_test` weights and compared with paired t-tests; directional accuracy additionally gets a day-block bootstrap.

**Why.** On a 5-minute horizon a model cannot be trusted without a floor to beat; the point models turned out to sit on that floor for MAE, and the only measurable edge is in direction (2 to 3 pp for BTC and ETH). With eight folds the smallest resolvable DA change is about 1.5 pp, so smaller effects are invisible to the t-test; the block bootstrap trades that for an assumption (seven-day blocks hold the autocorrelation) and gives about 400 units. For experiments that record one, the success bar is written into the script before the first run, and falling short closes the hypothesis ("methodology rule" in the docstrings) instead of prompting variations.

**What went wrong once.** The cross-article novelty test met its bar (a simple majority of seven folds) while its paired test was nowhere near significance (p = 0.77). The lesson recorded in the summary: a criterion must also demand significance. Another time an unweighted mean over folds flipped the sign of a result (6 h horizon: +0.28 pp unweighted, -0.91 pp weighted) because a 221-row fold counted like a 105,000-row one. Hence the rule: always weight by `n_test`.

**Evidence.** [`evaluation.py`](../src/pinance_ml/evaluation.py); [evaluation.md, Method](evaluation.md#method).

## 5. The obv train-serve skew

**What happened.** On-balance volume (`obv`) is a cumulative sum from the first candle. Training computed it over years of history; the inference service sees only the last 150 candles, so it computed it from a different starting point. In the repository's synthetic regression test the value on the serving window differed from the full-history value by about 73 %; the exact size depends on history length and was not measured on real candles. The model was fed a feature distribution it had never been trained on.

**Why offline metrics did not show it.** Training and test both used the full-history `obv`, which was internally consistent. Re-running the walk-forward with the fix moved MAE by -0.13 % (p = 0.098) and DA by +0.05 pp (p = 0.52): no visible change. The damage existed only in serving, and offline evaluation is structurally unable to see it.

**Fix.** Replace `obv` with `obv_roc_36`: signed volume over 36 candles divided by volume over the same candles. It is a windowed sum, so it needs no cumulative state, is bounded in [-1, 1] and is comparable across symbols.

**What now catches it.** `test_serving_window_features_match_full_history` computes the last feature row on 5,000 candles and on the last 150 and requires them to agree. With the old formula restored in a scratch copy the test fails (`-4880.56` against `-1296.17`); with the fix all nine tests in the file pass ([transcript and figure](evaluation.md#15-the-obv-fix)).

**Side effect.** The feature list changed, so `schema_version` changed, which sent every symbol through the schema-change path (decision 9). The live model on 2 October 2026 was the first trained on the new schema.

**Not done.** ETH, BNB and SOL were not re-evaluated offline after the change.

## 6. Point model and corridor are independent decisions

**What happened (2026-08-03, BTCUSDT and BNBUSDT).** One combined script retrained the point model and the corridor and pushed them as one unit. The point model was rejected on its own offline holdout (directional accuracy 52.6 % and 52.1 % against production's 58.8 % and 59.5 % on that holdout), yet it was pushed to the candidate slot moments later because it travelled with an approved corridor. The promotion step copied whole slots, so the same coupling reappeared one level up: a promotion that was "about the corridor" copied whatever point model sat next to it.

**Patches that did not fix it.** A cap on how much live accuracy a corridor promotion could regress, and a flag marking a point model as offline-rejected. Both treated symptoms of a structurally coupled action.

**Fix.** Two retrain scripts on separate schedules (point daily, corridor weekly), disjoint metadata keys, and two promotion functions that each copy only their own files and merge into production's metadata. A corridor promotion now has no code path that can move a point model, so the regression cap was removed. The flag stays because it guards the point decision on its own merits.

**Side effect.** A refreshed corridor (a new quantile version on top of an already promoted one) previously had no promotion path at all, because "gained a corridor" could only be true once. Comparing `quantile_model_version` directly covers both cases.

**Evidence.** The module docstring of [`promote_if_better.py`](../scripts/promote_if_better.py) and of [`model_storage.py`](../src/pinance_ml/model_storage.py); `tests/test_promote_if_better.py` (27 tests).

## 7. Two gates: offline holdout, then live shadow traffic

**Chosen.** `auto_retrain*.py` asks "would the candidate have done better on the last 30 days if it had been serving?" and pushes only to the candidate slot. `promote_if_better.py` then asks "did it also do better on real live traffic while shadow-serving?", using the backend's own record of live forecasts against realised candles, and it needs at least 100 resolved samples and a 1 pp directional-accuracy edge over seven days.

**Why.** An offline holdout is necessary but not sufficient: it is one window, subject to noise and to walk-forward artefacts. The second gate answers a different question on different data. The two are kept as separate programs on separate timers so the daily retrain never blocks on network calls to the backend and never promotes from a holdout comparison alone.

**Known weaknesses.** The 1 pp bar on seven days of heavily overlapping rows is itself noisy: the backend repository puts the approximate 95 % interval of a 7-day directional accuracy at about +/-2.2 pp. Live statistics are also only as good as the backend's scoring of forecasts.

## 8. Fixed slots in the object store

**Chosen.** `production`, `candidate` and `previous` at fixed paths per symbol; the real version identity lives in `metadata.json`.

**Why.** The inference service re-polls the same keys and reloads when `model_version` changes. It never lists a bucket or discovers "the latest". Promotion is a same-bucket copy, so it is cheap and does not create a new upload that could be half-written. `previous` is a one-level undo; the design explicitly does not keep a version history.

**Caveat.** Rolling back across a feature-schema change only works together with rolling back the feature code, otherwise the inference service rejects the older model for the same schema mismatch that motivated the promotion.

## 9. Schema changes get their own path

**Chosen.** `schema_version` is a short hash of the ordered feature list. If production's schema differs from the current feature code, the retrain does not compare against production; the candidate must pass the naive-baseline sanity gate, is pushed with `new_scheme: true`, and `promote_if_better.py` refuses to promote it. `promote_new_scheme.py` is the explicit manual step, dry-run by default.

**Why.** Comparing live metrics across different inputs is meaningless, and the inference service rejects models whose feature names differ from what it computes (LightGBM does not check names at prediction time, so the service enforces it). Before this path existed, a schema change left no quality bar on the candidate at all. The naive baseline is the only reference that is always available.

**Cost.** Promotion across a schema change is a human decision. A real run is in [`docs/examples/retrain-log-BTCUSDT.jsonl`](examples/retrain-log-BTCUSDT.jsonl): three dry runs and one apply on 2 October 2026.

## 10. One pure feature function

**Chosen.** `compute_features` is a plain `DataFrame -> DataFrame` with no state. The inference repository carries a pinned copy.

**Why.** Training and serving cannot drift if they run the same code, and a pure function is easy to test: no look-ahead, determinism, serving-window parity. The drift-monitored feature list (`DRIFT_FEATURE_COLUMNS`) is the one thing the two repositories still keep in sync by hand.

**Cost.** Two copies must be updated together; a change in one without the other is the same class of problem as decision 5. The schema hash is what makes the mismatch visible at load time.

## 11. Side channels never fail a run

**Chosen.** MLflow tracking and the backend event journal swallow their own errors and are silent no-ops when not configured.

**Why.** They are monitoring, not part of the pipeline. A tracking-server outage must not fail a retrain, and scripts that never needed the backend must not start to.

**Cost.** A misconfigured tracker produces one warning and no runs, so absence of data in MLflow is not evidence that nothing ran.

## 12. News: publication time, resumable passes

**Chosen.** News features use the article's publication time; fetching and FinBERT scoring are separate passes with nullable result columns.

**Why.** The parse time is later than the publication time, so using it would put information into the past (look-ahead). Separating fetch from scoring lets a fast network-bound backfill land first and a slow CPU-bound pass fill in later without redoing work (`WHERE ... IS NULL`).

## Rejected and not done

Rejected with evidence ([evaluation.md](evaluation.md)):
- **Sliding or recency-weighted training windows**: worse at every setting tried (sections 3 and 4).
- **FinBERT sentiment, time-decayed, as features**: no gain on four symbols (section 5).
- **LoRA fine-tuning of FinBERT on market-derived labels**: five configurations, loss at chance (section 6).
- **LLM-extracted event type**: related to the size of the move, not the direction; no gain for the point forecast, and no gain for the corridor against a 1 % bar (sections 7, 12).
- **24-hour horizon and 2 to 6 hours**: no out-of-sample predictability (sections 8, 9).
- **Cross-article novelty**: null (section 10).
- **Order-flow imbalance, funding rate, volatility-scaled target, microstructure proxies**: screened; none carried over, and several result files were lost (section 14).
- **A heavier hyperparameter search**: a 144-cell grid won on selection folds and lost on held-out folds (section 13).

Deliberately not done (no claim about what would happen):
- No deep sequence models, no recursive multi-step forecasting, no multi-output networks. Not tried in this repository.
- No transaction-cost model and no trading backtest. The directional edge is small enough that costs would exceed it; this repository measures forecast quality only.
- No cross-symbol pooled model. Each symbol has its own models, with BTC's return as a feature for the others.
- No automated promotion across a schema change.
