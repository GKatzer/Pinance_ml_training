"""One-off: marginal-test-first screen for a cross-article novelty
feature at the 24h horizon -- the last branch of the news-direction
investigation before the project's pre-committed stop-rule kicks in
(the project notes, 2026-08-01).

Unlike every prior feature tried this session (event_type, sentiment,
magnitude, ...), novelty is structurally different: "is this substantively
new information relative to recent coverage of the same asset" requires
memory of the article stream, which none of Level 1/2/3's single-item
extraction methods had. Built here as:

    novelty = 1 - max(cosine_similarity(article_i, article_j))
              for all same-asset articles j published in the 24h before i

using a dedicated sentence-embedding model (all-MiniLM-L6-v2 -- not
FinBERT/Qwen, which are classification/generation models, not trained for
semantic similarity).

Pre-registered pass criterion (agreed before running, not after seeing
results): n_test-weighted mean marginal-test delta > 0 in a MAJORITY of
folds, AND the effect survives residualizing against same-asset news
volume in the same window (the obvious confound: quiet periods make
everything look "novel" just because there's less to compare against).
Anything short of that closes the whole news-direction investigation per
the stop-rule -- no chasing a masked effect, no seventh branch.

Test-window boundaries are pulled from the same walk_forward_folds() call
used for event_type_24h, so results are directly comparable to everything
already done at this horizon. Per-article (not per-candle) framing,
matching the horizon-sweep's original Kruskal-Wallis analysis, and
avoiding the boolean-OR aggregation trap that made event_type_24h
degenerate at the candle level.

Usage: python scripts/screen_cross_article_novelty.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from pinance_ml.config import LIGHTGBM_TEST_DAYS, LONG_PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.data.news_db import load_news
from pinance_ml.news.decay import base_asset
from pinance_ml.splits import walk_forward_folds
from pinance_ml.tracking import log_research_run

NOVELTY_WINDOW = np.timedelta64(24 * 60 * 60, "s")
MAX_TEXT_CHARS = 1500


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_novelty(news: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """One row per news item: novelty (1 - max cosine sim to same-asset
    articles in the preceding 24h) and news_count_24h (the confound
    control). `news` must be sorted by published_at ascending; `embeddings`
    L2-normalized so dot product == cosine similarity."""
    published = pd.to_datetime(news["published_at"]).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(
        dtype="datetime64[ns]"
    )
    n = len(news)
    novelty = np.empty(n)
    news_count = np.empty(n, dtype=int)
    lo = 0
    for i in range(n):
        window_start = published[i] - NOVELTY_WINDOW
        while lo < i and published[lo] < window_start:
            lo += 1
        if i > lo:
            sims = embeddings[lo:i] @ embeddings[i]
            novelty[i] = 1.0 - float(sims.max())
            news_count[i] = i - lo
        else:
            novelty[i] = 1.0
            news_count[i] = 0
    return pd.DataFrame({"novelty": novelty, "news_count_24h": news_count}, index=news.index)


def label_at_horizon(news: pd.DataFrame, candles: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    """Same idea as news/labels.py's label_news, but with an explicit
    horizon instead of that module's hardcoded LABEL_HORIZON_MINUTES=60 --
    this screen needs 24h (1440min), not 60min."""
    steps = horizon_minutes // 5
    ts = candles["ts"]
    close = candles["close"].to_numpy()
    n = len(candles)
    entry_pos = ts.searchsorted(news["published_at"].to_numpy(), side="left")
    label_pos = entry_pos + steps
    valid = (entry_pos < n) & (label_pos < n)
    log_return = np.full(len(news), np.nan)
    log_return[valid] = np.log(close[label_pos[valid]] / close[entry_pos[valid]])
    result = news.copy()
    result["log_return"] = log_return
    return result[valid].reset_index(drop=True)


def marginal_test_by_window(
    labeled: pd.DataFrame, feature_col: str, windows: list[tuple], n_buckets: int = 4
) -> pd.DataFrame:
    """Same marginal-test protocol as screen_intermediate_horizons.py
    (P(up | bucket) fit on train, applied out-of-sample), but with a
    time-based purge suited to irregularly-spaced articles instead of a
    row-count purge (which assumes a regular candle grid) -- train is
    every article whose own 24h-forward target resolves before the test
    window starts; test is every article published inside the window with
    a resolvable target."""
    rows = []
    for i, (test_start, test_end) in enumerate(windows):
        train_mask = labeled["published_at"] + pd.Timedelta(hours=24) <= test_start
        test_mask = (labeled["published_at"] >= test_start) & (labeled["published_at"] < test_end)
        train = labeled[train_mask]
        test = labeled[test_mask]
        if len(train) < 50 or len(test) == 0:
            continue

        bucket_edges = np.quantile(train[feature_col], np.linspace(0, 1, n_buckets + 1))
        bucket_edges[0], bucket_edges[-1] = -np.inf, np.inf
        # Integer bucket indices (not pd.cut's Interval categories) -- a
        # Categorical Series here makes .map() silently keep categorical
        # dtype and comparisons blow up downstream; plain ints sidestep it.
        train_bucket = np.digitize(train[feature_col].to_numpy(), bucket_edges[1:-1])
        test_bucket = np.digitize(test[feature_col].to_numpy(), bucket_edges[1:-1])

        train_sign = np.sign(train["log_return"]).to_numpy()
        base_rate = max((train_sign > 0).mean(), (train_sign < 0).mean())
        overall_up_rate = (train_sign > 0).mean()
        cond_prob_up = pd.Series(train_sign > 0).groupby(train_bucket).mean()

        test_pred_up_prob = pd.Series(test_bucket).map(cond_prob_up).fillna(overall_up_rate).to_numpy()
        test_pred_sign = np.where(test_pred_up_prob > 0.5, 1, -1)
        test_actual_sign = np.sign(test["log_return"]).to_numpy()
        marginal_acc = float((test_pred_sign == test_actual_sign).mean())

        rows.append(
            {"fold": i, "n_train": len(train), "n_test": len(test), "base_rate": base_rate, "marginal_acc": marginal_acc}
        )
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> None:
    delta = df["marginal_acc"] - df["base_rate"]
    w_mean = np.average(delta, weights=df["n_test"])
    log(f"[{label}] " + df.to_string(index=False).replace("\n", "\n         "))
    log(f"[{label}] n_test-weighted mean delta: {w_mean * 100:+.3f}pp, folds>0: {(delta > 0).sum()}/{len(df)}")


def main():
    symbol = "BTCUSDT"
    log("Loading candles + news")
    candles = load_candles(symbol)
    asset = base_asset(symbol)
    news = load_news(asset=asset, include_body=True).sort_values("published_at").reset_index(drop=True)

    # Only test_start/test_end are used below (purge only affects which
    # TRAIN rows a fold gets, not window boundaries) -- candles[["ts"]] is
    # enough, no need for build_dataset()'s expensive technical features.
    folds_ref = list(walk_forward_folds(candles[["ts"]], WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, LONG_PURGE_ROWS))
    windows = [(f.test_start, f.test_end) for f in folds_ref]
    log(f"{len(windows)} reference test windows (same as event_type_24h)")

    log("Encoding articles with all-MiniLM-L6-v2")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [
        (row.body if isinstance(row.body, str) and len(row.body) > 20 else row.title)[:MAX_TEXT_CHARS]
        for row in news.itertuples()
    ]
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    log(f"Encoded {len(texts)} articles in {time.time() - t0:.1f}s")

    log("Computing novelty + news_count_24h")
    novelty_df = compute_novelty(news, embeddings)
    news = pd.concat([news.reset_index(drop=True), novelty_df.reset_index(drop=True)], axis=1)
    log(f"novelty: mean={news['novelty'].mean():.3f} std={news['novelty'].std():.3f}")
    log(f"news_count_24h: mean={news['news_count_24h'].mean():.1f}")
    log(f"corr(novelty, news_count_24h) = {news['novelty'].corr(news['news_count_24h']):.4f}")

    labeled = label_at_horizon(news, candles, horizon_minutes=1440)
    log(f"{len(labeled)}/{len(news)} articles have a resolvable 24h return")

    log("\n=== Raw novelty marginal test ===")
    raw_result = marginal_test_by_window(labeled, "novelty", windows)
    report(raw_result, "raw novelty")

    log("\n=== Confound check: residualize novelty against news_count_24h ===")
    coef = np.polyfit(labeled["news_count_24h"], labeled["novelty"], deg=1)
    labeled["novelty_resid"] = labeled["novelty"] - np.polyval(coef, labeled["news_count_24h"])
    resid_result = marginal_test_by_window(labeled, "novelty_resid", windows)
    report(resid_result, "residualized novelty")

    def _weighted_mean_delta_pp(df: pd.DataFrame) -> float | None:
        if df.empty:
            return None
        return float(np.average(df["marginal_acc"] - df["base_rate"], weights=df["n_test"]) * 100)

    log_research_run(
        __file__,
        run_name=f"novelty-{symbol}",
        params={
            "symbol": symbol,
            "n_windows": len(windows),
            "n_articles": len(news),
            "n_labeled": len(labeled),
            "max_text_chars": MAX_TEXT_CHARS,
        },
        metrics={
            "raw_novelty_delta_pp": _weighted_mean_delta_pp(raw_result),
            "resid_novelty_delta_pp": _weighted_mean_delta_pp(resid_result),
        },
    )


if __name__ == "__main__":
    main()
