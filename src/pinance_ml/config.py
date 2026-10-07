import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

# Target definition per README: r_h = log(P[t + 5h min] / P[t]), h = 1..12 (1 hour horizon).
CANDLE_INTERVAL_MINUTES = 5
HORIZONS = list(range(1, 13))

# Rows purged immediately before every train/test boundary, since their
# target windows (up to `max(HORIZONS)` candles ahead) reach into the test
# period otherwise.
PURGE_ROWS = max(HORIZONS)

# Walk-forward validation per README: "обучение на [0, T], тест на
# [T, T+7d], сдвиг" — expanding train window, fixed-width sliding test
# window. MIN_TRAIN_DAYS (how much history the first fold requires before
# it can test at all) isn't specified there; 90 days is a judgment call —
# enough rows for stable rolling_144 features/model fitting later, small
# enough to leave many folds even for the shorter-history symbols.
WALK_FORWARD_MIN_TRAIN_DAYS = 90
WALK_FORWARD_TEST_DAYS = 7

# LightGBM re-trains 12 models per fold (vs. the naive baseline's ~free
# constant prediction) — measured ~52s to train 12 models on a 26k-row
# fold, scaling roughly linearly with row count. At the naive baseline's
# 7-day folds that's ~400 folds over BTC's 8-year history: hours of
# redundant retraining on near-identical expanding windows. A 1-year test
# window over the same harness gives ~8 folds instead — still genuine
# expanding-window walk-forward with purging, just coarse enough to run in
# a reasonable amount of wall-clock time.
LIGHTGBM_TEST_DAYS = 365

# 24h horizon, kept deliberately separate from HORIZONS/PURGE_ROWS above
# (2026-08-01, project notes): PURGE_ROWS is
# derived as max(HORIZONS) and shared by every model trained on one
# walk_forward_folds() call, so folding a long horizon into HORIZONS
# directly would force the already-validated 5-60min models to discard
# 288 rows per fold boundary instead of 12, for no benefit to them.
# Starting with a single point (24h) rather than full 5-min resolution
# out to 24h (which would mean 288 separate models, one per candle step,
# mirroring how HORIZONS=1..12 covers every step up to 1h) -- the horizon
# sweep behind this only tested 6 discrete points and found signal
# specifically at 1440min, not evidence that every 5-min step in between
# carries independent information worth a dedicated model.
LONG_HORIZONS = [288]
LONG_PURGE_ROWS = max(LONG_HORIZONS)

# --- News / text layer (README "Текстовый слой") ---

# The connection above is read-only (pinance_ro, see .env.example) by
# design — it's meant for training reads, not writes. The news ingestion
# job is this repo's one writer, so it gets its own credential rather than
# silently requiring DATABASE_URL to become read-write everywhere it's
# used. Defaults to DATABASE_URL so a single-role dev/test setup still works.
NEWS_DATABASE_URL = os.environ.get("NEWS_DATABASE_URL", DATABASE_URL)

# README's original 3 ("RSS: CoinDesk, CoinTelegraph, The Block") plus 4
# more added 2026-07-22 to widen live-collection volume -- GDELT's
# historical backfill is blocked on API rate limiting, so real-time
# accumulation via more sources is currently the only reliable path to
# building up news history, not just a nice-to-have. Each URL verified
# live (fetch_feed + parse_feed against the real feed) before adding.
# Poll cadence (every 5 minutes) lives in the scheduler that invokes
# scripts/fetch_news.py (systemd timer / APScheduler on the training host), not here.
NEWS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "theblock": "https://www.theblock.co/rss.xml",
    "decrypt": "https://decrypt.co/feed",
    "cryptopotato": "https://cryptopotato.com/feed",
    "beincrypto": "https://beincrypto.com/feed",
    "bitcoinmagazine": "https://bitcoinmagazine.com/feed",
}

# README level 1: "ProsusAI/finbert ... sentiment (pos/neu/neg + confidence)".
# Env override so a Level-3 fine-tuned candidate (e.g. models/finbert-lora)
# can be A/B'd via measure_news_feature_gain.py without editing this file
# per comparison run.
NEWS_FINBERT_MODEL = os.environ.get("NEWS_FINBERT_MODEL", "ProsusAI/finbert")

# README level 2: small instruct model for structured extraction into
# strict JSON, on top of level 1's FinBERT sentiment. Env override for the
# same reason NEWS_FINBERT_MODEL is: A/B a different GGUF/quant without
# editing this file per comparison run.
NEWS_LLM_MODEL_PATH = os.environ.get("NEWS_LLM_MODEL_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf")

# README's production target (training host) is CPU-only ("llama.cpp на CPU"), so
# this defaults to 0 (no GPU offload). A one-off historical backfill on a
# dev machine with a spare CUDA GPU can override this (e.g. -1 = offload
# every layer) to run in hours instead of days -- doesn't change what
# actually ships to the training host, which never sets this env var.
NEWS_LLM_GPU_LAYERS = int(os.environ.get("NEWS_LLM_GPU_LAYERS", "0"))

# Fixed vocabulary for the extraction JSON's "event_type" field -- unlike
# NEWS_EVENT_KEYWORDS above (keyword-matched, precision-biased, only
# "hack"/"regulation" so a miss just falls back to plain sentiment), the
# LLM classifies every item into exactly one of these, so the set needs to
# actually cover the news this project's feeds produce. "other" is the
# explicit catch-all -- it's what makes a real extraction distinguishable
# from "not yet attempted" (NULL) in news_items.llm_event_type.
NEWS_LLM_EVENT_TYPES = [
    "hack",
    "regulation",
    "partnership",
    "listing",
    "product_launch",
    "funding",
    "macro",
    "adoption",
    "market_move",
    "other",
]

# Title+summary cosine-similarity cutoff above which a new item is treated
# as a near-duplicate of an already-known one (README: "дедупликация ...
# near-duplicate по эмбеддингам") — e.g. the same wire story picked up by
# two outlets within the same poll.
NEWS_NEAR_DUP_SIMILARITY_THRESHOLD = 0.85

# README: "экспоненциальным затуханием: вес w = exp(-λ·Δt) ... период
# полураспада ~90 минут (гиперпараметр, тюнится)".
NEWS_DECAY_HALF_LIFE_MINUTES = 90.0

# Contributions past this many half-lives are ~1.5e-5 of their initial
# weight — indistinguishable from zero — so the decay sum is truncated
# here instead of scanning an item's entire, ever-growing tail.
NEWS_DECAY_HORIZON_HALF_LIVES = 16

# Window for max_magnitude_60m / event_type_flags (README), independent of
# the decay half-life above.
NEWS_RECENT_WINDOW_MINUTES = 60.0

# 24h-horizon counterpart (the project notes) -- the existing
# 60-minute window above was sized for the 60-min-horizon regressor;
# "did this event type occur in the last 60 minutes" is too narrow a
# question for a model looking 24h ahead, so event_llm_{type}_24h
# (news/decay.py) uses this instead.
NEWS_RECENT_WINDOW_24H_MINUTES = 1440.0

# "Упомянутые тикеры" heuristic (README level 1): lowercase, word-boundary
# matched against title+summary text (see news/sentiment.py) -- plain
# substring search would false-positive short aliases like "eth" inside
# ordinary words ("together", "weather").
NEWS_ASSET_ALIASES = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum", "ether"],
    "BNB": ["bnb", "binance coin"],
    "SOL": ["sol", "solana"],
}

# "Категория события" heuristic (README level 1) / event_type_flags
# (README's time-decay aggregation): hack and regulation, matched by
# keyword (word-boundary matched, see news/sentiment.py -- plain substring
# search would false-positive "hack" on "hackathon"). Word forms are
# listed out explicitly (hacked/hacking/hackers, not just "hack") since
# word-boundary matching means the bare stem no longer catches them.
# Kept short and precision-biased — a missed event just falls back to the
# plain sentiment features, a false positive pollutes them.
NEWS_EVENT_KEYWORDS = {
    "hack": ["hack", "hacked", "hacking", "hacker", "hackers", "exploit", "exploited", "breach", "breached", "stolen", "drained"],
    "regulation": [
        "sec",
        "regulation",
        "regulations",
        "regulatory",
        "lawsuit",
        "lawsuits",
        "bans",
        "banned",
        "banning",
        "compliance",
    ],
}

# Quote-currency suffixes stripped to go from a trading symbol to the base
# asset news is written about ("BTCUSDT" -> "BTC"). Every pair in the
# README today is USDT-quoted; unrecognized suffixes pass through
# unchanged rather than guessing.
QUOTE_ASSET_SUFFIXES = ("USDT", "USDC", "BUSD", "USD")

# --- Auto-retrain / model storage (README "Ретрейн регрессора") ---
# scripts/auto_retrain.py's contract with predictor-ml-inference (inference host):
# fixed per-symbol "slots" in MinIO, not a version-numbered path per run --
# the inference host's /predict and /predict/{symbol}/shadow always read the same two
# keys, so it never needs to discover "what's the latest version" via a
# bucket listing. The real version identity lives inside each slot's own
# metadata.json (model_version field, already written by export_models.py),
# not in the object path.
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "true").lower() not in ("false", "0", "")
MINIO_MODELS_BUCKET = os.environ.get("MINIO_MODELS_BUCKET", "pinance-models")

# README: "сравнение с production по среднему MAE через горизонты ...
# при улучшении сверх порога". A candidate must beat current production's
# average MAE (across HORIZONS) by at least this fraction to be pushed --
# guards against promoting a candidate that's "better" only by walk-forward
# noise (retraining on one more day of an 8-year history barely moves
# anything, so even a tiny relative bar filters out most no-op retrains).
AUTO_RETRAIN_MIN_IMPROVEMENT = float(os.environ.get("AUTO_RETRAIN_MIN_IMPROVEMENT", "0.005"))

# Width of the held-out comparison window candidate-vs-production is judged
# on -- must be recent (the question is "is the candidate better *now*",
# not historically) and long enough for the comparison to not be one bad
# day's noise. Excluded from the candidate's own training data (see
# scripts/auto_retrain.py) so the comparison is genuinely out-of-sample for
# the candidate, not just for production.
AUTO_RETRAIN_HOLDOUT_DAYS = float(os.environ.get("AUTO_RETRAIN_HOLDOUT_DAYS", "30"))

# README's "доверительный коридор" (confidence corridor): tails only, not
# alpha=0.5 -- the existing 12 point models above already train with
# objective="regression_l1", whose population minimizer *is* the median,
# so a separate alpha=0.5 quantile model would just duplicate that at
# extra training cost for no new information. h{h}.txt (the point model)
# is reused as the corridor's center line; these two quantiles are the
# only genuinely new per-horizon artifacts. Trained/gated/pushed by their
# own script, scripts/auto_retrain_quantiles.py -- deliberately separate
# from scripts/auto_retrain.py (not a flag on it), since the two are
# different models with different promotion criteria and refresh cadence;
# see that script's own docstring and project memory:
# the project notes.
CORRIDOR_QUANTILES = (0.1, 0.9)

# Coverage-calibration tolerance for the corridor's own push gate in
# auto_retrain_quantiles.py -- same +/-0.03 convention
# scripts/measure_quantile_gain.py used to judge fold-level calibration
# (project notes). A candidate
# that "improves" pooled pinball loss while drifting outside this band on
# most (horizon, quantile) pairs is rejected regardless of the
# pinball-loss number -- lying about its own uncertainty makes a corridor
# worse, not better.
CORRIDOR_COVERAGE_TOLERANCE = 0.03

# --- New-scheme sanity gate (auto_retrain.py / auto_retrain_quantiles.py) ---
# When the feature schema changed, production can't be compared like for
# like (different inputs), so the usual "beat production by X%" gate is
# skipped -- see decide_target_slot's schema_drift path. That left NO
# quality bar at all on that path. These thresholds replace it with the
# only reference that's always available: the naive baseline (see
# pinance_ml.baselines.naive / pinance_ml.sanity), scored on the same
# held-out window. A broken feature pipeline (NaN/constant columns, leakage
# removed, a column silently zeroed) typically lands at or worse than the
# naive floor; this gate exists to catch exactly that before a candidate
# with a new schema is ever pushed.
#
# Point model: avg MAE across horizons must be <= naive MAE * this ratio.
# >1.0 on purpose -- pooled MAE already sits at the naive floor (see the
# note above LGBM_* in this file), so a strict <= 1.0 would reject on noise.
SANITY_MAX_MAE_RATIO = float(os.environ.get("SANITY_MAX_MAE_RATIO", "1.02"))
# Point model: avg directional accuracy must be >= naive majority-class
# accuracy + this edge (0.0 = merely not worse than the base rate).
SANITY_MIN_DIR_ACC_EDGE = float(os.environ.get("SANITY_MIN_DIR_ACC_EDGE", "0.0"))
# Corridor: avg pinball loss must be <= the constant-empirical-quantile
# baseline's * this ratio (and coverage must still be calibrated).
SANITY_MAX_PINBALL_RATIO = float(os.environ.get("SANITY_MAX_PINBALL_RATIO", "1.02"))

# --- Promote-if-better (scripts/promote_if_better.py) ---
# auto_retrain*.py's own gates are decided against an OFFLINE holdout --
# "would this candidate have done better over the last N days if it had
# been serving". That's necessary but not sufficient: predictor-backend
# (backend host) is the only place that knows how the candidate actually performed
# on REAL live traffic once predictor-ml-inference started shadow-serving
# it (README: "predictor-backend's live metrics ... confirm it's actually
# better on live traffic"). This is that second, independent gate.

# predictor-backend's admin API -- deliberately not proxied publicly (see
# Pinance_backend/app/api/admin.py), only reachable over the private network from
# the backend host itself or another machine on the same private network (this one). Empty
# default like MINIO_*, not a hard os.environ[...] like DATABASE_URL --
# this module is imported by every script here, not just
# promote_if_better.py, and that one's the only thing that actually needs
# this; it checks for empty itself at startup instead of breaking imports
# repo-wide for scripts that never touch promotion.
PREDICTOR_BACKEND_URL = os.environ.get("PREDICTOR_BACKEND_URL", "")

# /compare computes all 5 SUMMARY_WINDOWS (1h/24h/7d/30d/all) concurrently
# per request regardless of which one PROMOTION_WINDOW actually asks for —
# "all" in particular has no time filter, so on a symbol with a long
# prediction history (pre-versioning rows still carry the literal
# model_version='production', see admin.py) it can scan a lot more than
# this script needs. 15s (a reasonable default for a single indexed
# query) isn't enough for that -- observed timing out in practice against
# real production history. Raise via env if it's still not enough; the
# real fix (only computing the window actually requested) belongs in
# predictor-backend, not here.
PREDICTOR_BACKEND_TIMEOUT_S = float(os.environ.get("PREDICTOR_BACKEND_TIMEOUT_S", "60"))

# Which /admin/metrics/{symbol}/compare window to judge live traffic on —
# must be wide enough that a bad hour doesn't decide it, but short enough
# that it's asking "how's the CURRENT candidate doing", not folding in
# whatever a previous candidate version did before today's point retrain
# replaced it (see admin.py's own docstring on why compare is now grouped
# by (slot, model_version), not just slot).
PROMOTION_WINDOW = os.environ.get("PROMOTION_WINDOW", "7d")

# Floor on `n` (matured, resolved predictions) for the SPECIFIC candidate
# model_version currently in MinIO before its accuracy number is trusted
# at all -- guards the case where today's point retrain landed minutes
# ago and there's barely any live traffic yet to judge it on. Low by
# design, not a "long observation period" gate: 12 horizons close every
# 5 minutes, so even a few hours of shadow serving clears this easily —
# it's a sanity floor against near-zero data, not a waiting period.
PROMOTION_MIN_SAMPLES = int(os.environ.get("PROMOTION_MIN_SAMPLES", "100"))

# Required live directional_accuracy edge (percentage points, not
# relative -- accuracy is already a 0-100 number, comparing two figures
# near 50% relatively is a less legible bar than "beat it by >= 1pp") for
# decide_point_promotion.
PROMOTION_MIN_ACCURACY_IMPROVEMENT_PP = float(os.environ.get("PROMOTION_MIN_ACCURACY_IMPROVEMENT_PP", "1.0"))

# There used to be a PROMOTION_MAX_ACCURACY_REGRESSION_PP here: how much
# live accuracy regression a "candidate gained a corridor" promotion would
# still tolerate. It existed only because model_storage.promote_candidate
# copied the candidate slot wholesale (point + corridor together), so a
# quantile-driven promotion could otherwise smuggle in a real point-model
# regression riding along. scripts/promote_if_better.py now promotes the
# point model and the corridor as two fully independent decisions, each
# through its own model_storage.promote_candidate_point /
# promote_candidate_quantiles (see that script's module docstring, "Why
# two independent decisions") -- a corridor promotion can no longer move
# the point model at all, so there's nothing left for this cap to guard.

# --- Feature-drift baseline (scripts/export_models.py) ---
# Curated drift-monitoring subset of the model's ~93 input columns -- must
# stay byte-for-byte identical to Pinance_ml_inference's own
# DRIFT_FEATURE_COLUMNS (src/pinance_ml_inference/predict.py), which reports
# live values for this same set on every /predict call as
# response["feature_snapshot"]. The two repos deploy independently (no
# shared import, unlike vendor/features.py's pinned copy of this repo's own
# feature pipeline) -- keep both lists in sync by hand if either changes.
DRIFT_FEATURE_COLUMNS = ("btc_ret", "atr", "bb_width", "rsi", "ret_std_144", "macd_diff")

# Quantile-bin count the training-sample baseline is computed over for each
# DRIFT_FEATURE_COLUMNS entry. predictor-backend's PSI drift check needs
# baseline bin edges/fractions to compare live feature_snapshot values
# against, not just mean/std (which alone only supports a z-score-style
# check, not true PSI) -- 10 (deciles) is the conventional PSI bucket count.
DRIFT_BASELINE_N_BINS = 10

# --- Retrain-event journaling (predictor-backend's POST /admin/retrain-events) ---
# scripts/auto_retrain.py, scripts/auto_retrain_quantiles.py and
# scripts/promote_if_better.py each already compute a retrain/promotion
# decision in memory (kind/decision/metric_value/threshold/n_samples) and
# used to just log it locally and move on. pinance_ml.backend_client.
# post_retrain_event ships that same, already-computed decision to
# predictor-backend for the MLOps dashboard's event timeline -- best-effort
# only (see that module), reusing PREDICTOR_BACKEND_URL/_TIMEOUT_S above
# rather than a separate config surface. Scripts that never had a
# PREDICTOR_BACKEND_URL requirement before (auto_retrain*.py) still don't:
# an empty URL makes post_retrain_event a silent no-op.

# --- Experiment tracking (pinance_ml.tracking -> MLflow) ---
# Same empty-default / silent-no-op contract as PREDICTOR_BACKEND_URL
# above, and for the same reason: pinance_ml.tracking is imported by every
# retrain and research script here, but MLflow is an optional monitoring
# side-channel, not a pipeline dependency -- an unset URI (or an
# unreachable server, or mlflow not installed) degrades to a no-op instead
# of breaking a run. Points at the MLflow tracking server on the training host
# (deploy/mlflow/), co-located with MinIO. The mlflow client separately
# reads MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
# straight from the environment (see .env.example) to reach the
# `pinance-mlflow` artifact bucket -- deliberately not re-exported through
# this module, that's mlflow's own config surface.
MLFLOW_TRACKING_URI = os.environ.get("PINANCE_MLFLOW_TRACKING_URI", "")
