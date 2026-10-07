# Features of the training repository

Each section says what the part does, why it is built that way, and where the proof is (code, test or committed result). Experiment outcomes are in [evaluation.md](evaluation.md); the retrain loop in production is in [deployment.md](deployment.md); the reasoning behind choices is in [design-decisions.md](design-decisions.md).

## Contents

[Data and targets](#data-and-targets) · [Feature pipeline](#feature-pipeline) · [Validation harness](#validation-harness) · [Models](#models) · [Baselines and sanity gate](#baselines-and-the-sanity-gate) · [Retrain, gating and promotion](#retrain-gating-and-promotion) · [Artifact storage](#artifact-storage) · [News pipeline](#news-pipeline) · [Tracking and journaling](#tracking-and-journaling) · [Script catalogue](#script-catalogue) · [Tests](#tests)

## Data and targets

- **Input.** 5-minute OHLCV candles for BTCUSDT, ETHUSDT, BNBUSDT and SOLUSDT, read from a TimescaleDB `candles` table (`symbol, ts, open, high, low, close, volume`) by [`data/db.py`](../src/pinance_ml/data/db.py). The connection is read-only by design. History starts in 2018 (BTC, ETH, BNB: about 884,000 rows each by October 2026; SOL about 646,000).
- **Targets.** `r_h = log(close[t + 5h minutes] / close[t])` for `h = 1..12`, plus `r_288` (24 hours) for one experiment. Targets are looked up by exact timestamp, so a gap yields NaN, never a wrong pairing ([`features/targets.py`](../src/pinance_ml/features/targets.py), test `tests/test_targets.py`).
- **One place joins features and targets**: [`dataset.build_dataset`](../src/pinance_ml/dataset.py). Raw OHLCV columns are carried but are never model inputs, because price and volume levels are not stationary over years; `feature_columns()` returns exactly the model inputs and excludes every target.

## Feature pipeline

[`features/pipeline.py`](../src/pinance_ml/features/pipeline.py) is a pure function, `compute_features(candles, btc_candles=None) -> DataFrame`, shared by training and by the inference service (which carries a pinned copy; see [`Pinance_ml_inference`](https://github.com/GKatzer/Pinance_ml_inference)). **93 features** for BTC and **94** for the other symbols (one extra, the cross-asset BTC return):

| Group | Columns | Count |
|---|---|---|
| Lags, 2 hours of history | `ret_lag_1..24`, `vol_lag_1..24` (return and volume) | 48 |
| Rolling statistics, windows 6, 12, 36, 144 candles (30 min, 1 h, 3 h, 12 h) | mean, std, min, max of returns and of volume | 32 |
| Indicators | `rsi` (14), `macd`, `macd_signal`, `macd_diff` (12/26/9), `bb_width` (20, 2 sd), `atr` (14), `obv_roc_36` | 7 |
| Derived | `ret_sq`, `ret_atr_norm` (return divided by ATR) | 2 |
| Time of day and week | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` (cyclical) | 4 |
| Cross-asset | `btc_ret` (BTC's log-return at the same timestamp; absent for BTC itself) | 0 or 1 |

Checked in two ways, both unit tests:
- **No look-ahead.** Every feature at row `t` uses candles at or before `t`: `test_no_lookahead_prefix_features_match_when_future_rows_removed` requires the features of a prefix of the series to equal the same rows computed on the full series.
- **Training and serving agree.** Training builds features over years of history; serving gets only the last 150 candles. `test_serving_window_features_match_full_history` requires the last row to agree (exactly for `obv_roc_36`, within `rtol = 1e-4` for the exponential-smoothing indicators `rsi`, `macd`, `atr`, which forget their start geometrically). This test is the guard that came out of the [`obv` incident](design-decisions.md#5-the-obv-train-serve-skew).

NaNs in the warm-up rows are left in place; LightGBM routes missing values natively, and only rows with a NaN **target** are dropped for that horizon.

## Validation harness

- **Walk-forward with purging** ([`splits.py`](../src/pinance_ml/splits.py), 13 tests): first test window 90 days after the start, expanding training window, `PURGE_ROWS = 12` rows removed before every boundary. Boundaries use `searchsorted` and positional slices, because boolean masks over a million rows and hundreds of folds were measured at about 30 s of pure copying. A sliding-window option (`max_train_days`) and `recency_sample_weight` exist for the window and recency experiments.
- **Fold pooling** ([`evaluation.pool_fold_metrics`](../src/pinance_ml/evaluation.py)): per-fold means are combined with `n_test` weights, algebraically identical to recomputing the metric on all test rows together. The same function reports naive, point and quantile results, so all use one convention.
- **Day-block bootstrap** (`daily_prediction_stats`, `directional_block_bootstrap`): resamples 7-day blocks of calendar days, about 400 near-independent units instead of nine folds, giving a paired interval for a change in directional accuracy. Both variants are scored on the same resampled blocks.
- **Metrics** ([`metrics.py`](../src/pinance_ml/metrics.py)): MAE, directional accuracy, pinball loss, coverage.

## Models

[`models/lightgbm_model.py`](../src/pinance_ml/models/lightgbm_model.py)

- **Point forecast: 12 direct models**, one `LGBMRegressor` per horizon, all on the same features and each fitted to its own target, so there is no error accumulation from feeding predictions back. `DEFAULT_PARAMS`: `objective="regression_l1"` (its minimiser is the median, which also serves as the corridor's centre), 300 trees, learning rate 0.05, 31 leaves, `min_child_samples` 100, `random_state` 42, `deterministic`. MAE is the comparison metric, and the median-minimising L1 objective lets the same models serve as the corridor's centre line. No other objective was compared in the repository.
- **Corridor: 24 quantile models**, one per horizon and quantile (0.1 and 0.9), `objective="quantile"` with the same capacity as the point models, so a quantile-against-point comparison is not confounded by model size. The 0.5 quantile is not a separate model: the L1 point model is reused as the centre line. Independently fitted quantile models can cross; this is checked at evaluation time through coverage, not prevented in training.
- **No Python pickles.** Models are saved in LightGBM's text format (`Booster.save_model`), stable across library versions.

## Baselines and the sanity gate

- **Naive baselines** ([`baselines/naive.py`](../src/pinance_ml/baselines/naive.py)): `r = 0` for error size; the majority sign of the training window for direction.
- **Sanity gate** ([`sanity.py`](../src/pinance_ml/sanity.py)): when the feature schema changes, a candidate cannot be compared with production (different inputs), which used to leave no quality bar at all. The gate scores the candidate and the naive baseline on the same held-out window: average MAE within `SANITY_MAX_MAE_RATIO = 1.02` of the naive MAE, directional accuracy not below the naive constant plus `SANITY_MIN_DIR_ACC_EDGE = 0.0`; for the corridor, pinball loss within 1.02 of the constant empirical-quantile forecast and calibrated coverage. A broken pipeline (NaN or constant columns, a column silently zeroed) typically lands at or below the naive floor. The ratio is above 1 deliberately, because pooled MAE already sits at the naive floor and a strict bar would reject on noise. The verdict is stored in the candidate's metadata. 8 tests.

## Retrain, gating and promotion

Four scripts, each deliberately separate. Timers and units are in [deployment.md](deployment.md).

| Script | Decides | Gate | Result |
|---|---|---|---|
| [`auto_retrain.py`](../scripts/auto_retrain.py) | point models | trained on all but the last 30 days; must beat production's average MAE across the 12 horizons by at least 0.5 % on that held-out window | pushes to the `candidate` slot (never production, except the first bootstrap) |
| [`auto_retrain_quantiles.py`](../scripts/auto_retrain_quantiles.py) | corridor | pooled pinball loss at least 0.5 % better **and** coverage within 0.03 of nominal | pushes to `candidate` |
| [`promote_if_better.py`](../scripts/promote_if_better.py) | promotion, point and corridor as **two independent decisions** | live shadow metrics from the backend's `/admin/metrics/{symbol}/compare`: at least 100 resolved samples, directional accuracy at least 1 pp above production over 7 days; corridor must have a new version and honest live coverage | copies only its own files and metadata keys candidate to production |
| [`promote_new_scheme.py`](../scripts/promote_new_scheme.py) | candidates on a **changed feature schema** | explicit `new_scheme` label, schema really differs, recorded sanity pass, files match what the metadata promises | dry run by default, `--apply` to move, `--rollback` to restore the `previous` slot |

Behaviours worth knowing:
- If the schema changed, `auto_retrain.py` does not crash and does not compare apples with oranges: it applies the sanity gate and pushes with `new_scheme: true`, which `promote_if_better.py` will not auto-promote ("manual promotion required").
- `promote_if_better.py` appends an audit line to `<staging-dir>/promote_if_better_log.jsonl` on every run, whatever the outcome; the other scripts keep similar JSONL logs. Real samples are in [`docs/examples/retrain-log-BTCUSDT.jsonl`](examples/retrain-log-BTCUSDT.jsonl).
- `--dry-run` exists on `promote_if_better.py`; `promote_new_scheme.py` is dry-run unless `--apply` is given.
- Thresholds are environment-configurable (see the README configuration table).

Tests: 9 + 11 (retrain), 27 (`promote_if_better`, pure decision functions), 12 (`promote_new_scheme`).

## Artifact storage

[`model_storage.py`](../src/pinance_ml/model_storage.py) (17 tests) is the contract with the inference service, over an S3-compatible store (MinIO). **Fixed per-slot paths**, not a version-numbered path per run:

```
{bucket}/{symbol}/production/metadata.json
{bucket}/{symbol}/production/h{1..12}.txt                 point models (also the corridor's centre line)
{bucket}/{symbol}/production/h{1..12}_q{0.1,0.9}.txt      corridor tails (optional)
{bucket}/{symbol}/candidate/...                            same layout
{bucket}/{symbol}/previous/...                             byte-for-byte snapshot of production before the last promotion
```

The inference service always re-polls the same keys and reloads when `metadata.json`'s `model_version` changes, so it never lists a bucket. Properties:
- **Point and corridor do not clobber each other.** Their metadata keys are disjoint (`quantile_`-prefixed vs unprefixed) and every write goes through `merge_metadata`; promotion moves only its own half (`promote_candidate_point`, `promote_candidate_quantiles`).
- **One-level rollback.** `snapshot_production` and `rollback_production`; an undo button, not a version history.
- **Consistency check** (`slot_consistency_problems`, pure): every file the metadata promises exists, `schema_version` really is the hash of `feature_columns`, and point and corridor agree on the schema.
- **`metadata.json`** carries the feature list in order, `schema_version` (a hash of that list), `model_version`, source commit, training time, row count, library versions, evaluation metrics and the gate decision, and a **feature baseline**: mean, standard deviation and ten quantile bins for six drift-monitored features (`btc_ret`, `atr`, `bb_width`, `rsi`, `ret_std_144`, `macd_diff`). A sample is in [`docs/examples/slot-metadata-BTCUSDT.json`](examples/slot-metadata-BTCUSDT.json). The baseline is written for a population-stability check, but nothing in the backend consumes it today.
- The artifacts themselves live in the object store, not in git. Scripts that stage artifacts locally take `--staging-dir` (default `.auto_retrain_staging`, git-ignored).

## News pipeline

Research code for the text-signal experiments (all of them ended without a gain; see [evaluation.md](evaluation.md)). It is kept because it is real working infrastructure and because the negative result is only credible next to it.

| Stage | Code | Notes |
|---|---|---|
| Sources | [`news/parser.py`](../src/pinance_ml/news/parser.py) (RSS: 7 feeds in `NEWS_FEEDS`), [`gdelt.py`](../src/pinance_ml/news/gdelt.py) (GDELT DOC 2.0 history back to 2017), [`sitemap.py`](../src/pinance_ml/news/sitemap.py) (outlets that block APIs but serve a sitemap; titles come from URL slugs), [`wordpress_api.py`](../src/pinance_ml/news/wordpress_api.py) | every source yields the same `NewsItem`, so everything downstream is shared |
| De-duplication | by URL, and near-duplicate by embedding cosine similarity above 0.85 | a wire story picked up by two outlets in one poll |
| Level 1 | [`sentiment.py`](../src/pinance_ml/news/sentiment.py): FinBERT (`ProsusAI/finbert`) positive/neutral/negative with confidence; mentioned assets by word-boundary aliases; event flags for hack and regulation by keywords | word-boundary matching so "eth" does not fire inside "together" and "hack" not on "hackathon" |
| Level 2 | [`extraction.py`](../src/pinance_ml/news/extraction.py): Qwen2.5-3B-Instruct (GGUF Q4, llama.cpp, CPU) classifies each item into one of ten event types | a pilot with six more fields found them degenerate or uninformative, and they were dropped |
| Level 3 | [`labels.py`](../src/pinance_ml/news/labels.py), `build_finbert_labels.py`, `finetune_finbert_lora.py` | label = sign of the realised return 60 minutes after publication; market-derived, no human labelling |
| Features | [`decay.py`](../src/pinance_ml/news/decay.py): exponential decay `w = exp(-lambda * dt)`, half-life 90 minutes, truncated after 16 half-lives; recent-window flags | uses `published_at`, never `parsed_at`, so features cannot look ahead (16 tests) |
| Storage | `sql/news_schema.sql`, [`data/news_db.py`](../src/pinance_ml/data/news_db.py) | columns are filled in later passes (`NULL` = not attempted; `''` for a body = attempted, nothing extractable), which makes each pass resumable |

Fetching and scoring are separate passes on purpose: a historical backfill is network-bound and fast, FinBERT scoring is CPU-bound and slow, so rows land first and `score_pending_news.py` fills sentiment later (`WHERE sentiment_pos IS NULL`). The live parser runs as a systemd timer every five minutes.

## Tracking and journaling

- **MLflow** ([`tracking.py`](../src/pinance_ml/tracking.py), 5 tests): a best-effort wrapper. With `PINANCE_MLFLOW_TRACKING_URI` unset, or with the server down, or `mlflow` not installed, every helper is a silent no-op and logs a single warning, so tracking can never fail or slow a retrain. Retrain scripts use `with mlflow_run(...)` and `log_params / log_metrics / set_tags / log_dict / log_artifact`; research scripts call `log_research_run(...)` once at the end. Experiments: `retrain-point`, `retrain-quantile`, `research-feature-gain`, `research-screening`. Models are not pushed to MLflow; the object store remains the single source of truth for what serves.
- **Backend event journal** ([`backend_client.py`](../src/pinance_ml/backend_client.py), 4 tests): each retrain and promotion decision is posted to the backend's `POST /admin/retrain-events` for the retrain timeline in the web UI. Same contract: best effort, silent no-op without `PREDICTOR_BACKEND_URL`.

## Script catalogue

Production loop, one line each; everything else is an experiment that produced a result file in [`reports/`](../reports/) (see the [file index](evaluation.md#file-index)).

| Group | Scripts |
|---|---|
| Evaluation | `run_naive_baseline.py`, `train_lightgbm.py` (walk-forward report per symbol) |
| Production loop | `auto_retrain.py`, `auto_retrain_quantiles.py`, `promote_if_better.py`, `promote_new_scheme.py`, `export_models.py` (train final models on full history, save the slot layout locally) |
| News operations | `fetch_news.py` (one poll cycle), `score_pending_news.py`, `score_pending_news_llm.py`, `backfill_gdelt_news.py`, `backfill_outlet_history.py`, `backfill_article_bodies.py`, `retag_assets.py` |
| Level 3 | `build_finbert_labels.py`, `finetune_finbert_lora.py`, `rescore_and_measure_gain.py` |
| Experiments (measure) | `measure_news_feature_gain.py`, `measure_llm_event_gain.py`, `measure_24h_horizon_baseline.py`, `measure_quantile_gain.py`, `measure_funding_feature_gain.py`, `measure_vol_scaled_target_gain.py`, `measure_directional_gain.py` |
| Experiments (screen) | `screen_intermediate_horizons.py`, `screen_cross_article_novelty.py`, `screen_quantile_magnitude_signal.py`, `screen_funding_signal.py`, `screen_ofi_signal.py`, `screen_regime_error.py` |
| Window and tuning | `compare_train_windows.py`, `compare_recency_weighting.py`, `compare_daily.py`, `tune_lightgbm_directional.py`, `gridsearch_lightgbm.py`, `validate_reg_strong.py` |
| Data fetch for experiments | `fetch_funding_rates.py`, `fetch_agg_trades_ofi.py`, `_feature_sets.py` (named extra-feature builders) |

## Tests

235 tests in 24 files, all offline (no database, network or model weights); see the README for the command and the table of what each file covers.
