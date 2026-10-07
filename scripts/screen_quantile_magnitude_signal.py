"""One-off: cheap marginal check for the quantile-modeling task
(quantile_modeling_task.md, methodology rule #1) -- confirms event_type /
max_magnitude_60m still show a real, non-degenerate relationship with
|realized return| at the 60-minute horizon, before committing to the full
36-model (12 horizons x 3 quantiles x 2 variants x N folds) walk-forward
stack.

This is NOT a new discovery -- the project notes'
2026-08-01 addendum already established event_type correlates with
60-minute reaction MAGNITUDE (permutation-tested MI, survives a
year/volatility-regime confound check). The question here is narrower:
does that relationship show up cleanly in the exact form this task will
actually use (max_magnitude_60m / event_llm_type_60m against |r_12|, BTC,
full already-backfilled history, no walk-forward needed yet), as a sanity
check before spending the walk-forward compute budget.

No LightGBM, no walk-forward folds here -- full-history conditional
means/std by category/bucket. Kruskal-Wallis for the categorical
(event_type) test, matching the horizon-sweep's original nonparametric
test choice for this same signal (see screen_cross_article_novelty.py's
docstring).

Usage: python scripts/screen_quantile_magnitude_signal.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from pinance_ml.config import CANDLE_INTERVAL_MINUTES
from pinance_ml.data.db import load_candles
from pinance_ml.data.news_db import load_news
from pinance_ml.features.targets import compute_log_return_targets
from pinance_ml.news.decay import base_asset, compute_news_features
from pinance_ml.tracking import log_research_run

HORIZON_MINUTES = 60
HORIZON_CANDLES = HORIZON_MINUTES // CANDLE_INTERVAL_MINUTES  # r_12


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def most_recent_category(news: pd.DataFrame, timestamps: pd.Series, window_minutes: float) -> pd.Series:
    """Same protocol as screen_intermediate_horizons.py's helper of the
    same name: single-label most recent llm_event_type within
    window_minutes of each timestamp, 'none' if nothing in that window."""
    ts = pd.to_datetime(timestamps).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    news_sorted = news.dropna(subset=["llm_event_type"]).sort_values("published_at").reset_index(drop=True)
    published = pd.to_datetime(news_sorted["published_at"]).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(
        dtype="datetime64[ns]"
    )
    categories = news_sorted["llm_event_type"].to_numpy()

    window = np.timedelta64(round(window_minutes * 60), "s")
    n_news = len(news_sorted)
    lo = 0
    result = []
    for t in ts:
        window_start = t - window
        while lo < n_news and published[lo] < window_start:
            lo += 1
        hi = int(np.searchsorted(published, t, side="right"))
        result.append(categories[hi - 1] if hi > lo else "none")
    return pd.Series(result, index=range(len(ts)))


def report_by_category(abs_r: pd.Series, category: pd.Series, min_count: int = 30) -> pd.DataFrame:
    grouped = pd.DataFrame({"abs_r": abs_r, "category": category}).groupby("category")["abs_r"]
    summary = grouped.agg(["count", "mean", "std"]).sort_values("mean", ascending=False)
    log("\n=== |r_12| by most-recent event_type (60min window) ===")
    log(summary.to_string())

    testable = [cat for cat, n in summary["count"].items() if n >= min_count]
    samples = [abs_r[category == cat].to_numpy() for cat in testable]
    if len(samples) >= 2:
        h_stat, p_value = stats.kruskal(*samples)
        log(f"Kruskal-Wallis across {len(testable)} categories (n>={min_count}): H={h_stat:.2f}, p={p_value:.2e}")
    else:
        log(f"Not enough categories with n>={min_count} to run Kruskal-Wallis ({len(testable)} qualify)")
    return summary.reset_index()


def report_by_magnitude(abs_r: pd.Series, magnitude: pd.Series) -> pd.DataFrame:
    log("\n=== |r_12| by max_magnitude_60m bucket ===")
    has_news = magnitude > 0
    log(f"Rows with no news in the last 60m: {(~has_news).mean():.1%}")

    bucket = pd.Series("no_recent_news", index=magnitude.index)
    if has_news.sum() >= 4:
        bucket.loc[has_news] = pd.qcut(magnitude[has_news], q=4, labels=["q1_low", "q2", "q3", "q4_high"])
    summary = pd.DataFrame({"abs_r": abs_r, "bucket": bucket}).groupby("bucket", observed=True)["abs_r"].agg(
        ["count", "mean", "std"]
    )
    order = ["no_recent_news", "q1_low", "q2", "q3", "q4_high"]
    summary = summary.reindex([b for b in order if b in summary.index])
    log(summary.to_string())

    slope, intercept, r_value, p_value, std_err = stats.linregress(magnitude, abs_r)
    log(
        f"linregress(|r_12| ~ max_magnitude_60m): slope={slope:.6f}, r={r_value:.4f}, "
        f"r^2={r_value**2:.4f}, p={p_value:.2e}"
    )
    return summary.reset_index()


def main():
    symbol = "BTCUSDT"
    log(f"{symbol}: loading candles")
    candles = load_candles(symbol)

    targets = compute_log_return_targets(candles, [HORIZON_CANDLES], CANDLE_INTERVAL_MINUTES)
    dataset = targets[["ts", f"r_{HORIZON_CANDLES}"]].dropna().reset_index(drop=True)
    log(f"{len(dataset)} rows with a resolvable {HORIZON_MINUTES}min return")

    asset = base_asset(symbol)
    news = load_news(asset=asset, include_llm=True)
    n_classified = int(news["llm_event_type"].notna().sum())
    log(f"{asset}: {len(news)} news items, {n_classified} with Level 2 event_type ({n_classified / max(len(news), 1):.1%})")

    # max_magnitude_60m's window comes from config.NEWS_RECENT_WINDOW_MINUTES
    # (60min, matching HORIZON_MINUTES here) regardless of half_life_minutes
    # -- that param only affects the decay-weighted columns this script
    # doesn't use, so the default is fine.
    news_features = compute_news_features(news, dataset["ts"])
    dataset["max_magnitude_60m"] = news_features["max_magnitude_60m"].to_numpy()
    dataset["category"] = most_recent_category(news, dataset["ts"], window_minutes=HORIZON_MINUTES).to_numpy()

    abs_r = dataset[f"r_{HORIZON_CANDLES}"].abs()

    category_summary = report_by_category(abs_r, dataset["category"])
    magnitude_summary = report_by_magnitude(abs_r, dataset["max_magnitude_60m"])

    Path("reports").mkdir(exist_ok=True)
    category_summary.to_csv("reports/quantile_screen_by_category.csv", index=False)
    magnitude_summary.to_csv("reports/quantile_screen_by_magnitude_bucket.csv", index=False)
    log("\nWrote reports/quantile_screen_by_category.csv and reports/quantile_screen_by_magnitude_bucket.csv")

    log_research_run(
        __file__,
        run_name=f"quantile-magnitude-{symbol}",
        params={
            "symbol": symbol,
            "horizon_minutes": HORIZON_MINUTES,
            "n_rows": len(dataset),
            "n_news_items": len(news),
            "n_llm_classified": n_classified,
        },
        report_paths=sorted(Path("reports").glob("quantile_screen_by_*")),
    )


if __name__ == "__main__":
    main()
