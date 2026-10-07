"""One-off: marginal-test-first screen for event_type direction signal at
intermediate horizons (2h/3h/4h/6h), interpolating between the two
calibrated points from the project notes -- 60min (no raw
correlation at all) and 24h (real raw correlation, but the marginal test
came back negative out-of-sample: -1.19pp vs base rate, 2/8 folds).

Per the project's stop-rule (2026-08-01): this + cross-article novelty
are the last two branches before closing the news-direction-prediction
investigation as a completed negative result. Runs ONLY the cheap,
high-power marginal test (P(up | most-recent-category-in-window), fit per
fold, applied out-of-sample) -- not the full LightGBM walk-forward stack.
The 24h diagnostic showed the 8-fold LightGBM comparison lacks power to
resolve effects under ~1.5pp, while this marginal test (100k+ rows/fold,
not 8 fold-level points) doesn't have that limitation and already gave a
clean answer once. Only a horizon that clears this filter is worth
escalating to the full stack.

Feature window matches the horizon being tested (2h feature for the 2h
target, etc.) -- the same convention used for 60min/24h.

Usage: python scripts/screen_intermediate_horizons.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from pinance_ml.config import CANDLE_INTERVAL_MINUTES, LIGHTGBM_TEST_DAYS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.data.news_db import load_news
from pinance_ml.features.targets import compute_log_return_targets
from pinance_ml.news.decay import base_asset
from pinance_ml.splits import walk_forward_folds
from pinance_ml.tracking import log_research_run


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def most_recent_category(news: pd.DataFrame, timestamps: pd.Series, window_minutes: float) -> pd.Series:
    """Single-label: llm_event_type of the most recent news item published
    within `window_minutes` of each timestamp, 'none' if nothing in that
    window. Same two-pointer scan as news/decay.py's helpers."""
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


def marginal_test(dataset: pd.DataFrame, target_col: str, purge_rows: int) -> pd.DataFrame:
    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, purge_rows))
    rows = []
    for fold in folds:
        valid_train = fold.train[target_col].notna()
        valid_test = fold.test[target_col].notna()
        if valid_train.sum() == 0 or valid_test.sum() == 0:
            continue
        train = fold.train.loc[valid_train]
        test = fold.test.loc[valid_test]

        train_sign = np.sign(train[target_col])
        base_rate = max((train_sign > 0).mean(), (train_sign < 0).mean())
        overall_up_rate = (train_sign > 0).mean()
        cond_prob_up = train_sign.gt(0).groupby(train["recent_category"]).mean()

        test_pred_up_prob = test["recent_category"].map(cond_prob_up).fillna(overall_up_rate)
        test_pred_sign = np.where(test_pred_up_prob > 0.5, 1, -1)
        test_actual_sign = np.sign(test[target_col]).to_numpy()
        marginal_acc = float((test_pred_sign == test_actual_sign).mean())

        rows.append(
            {
                "fold": fold.index,
                "n_train": len(train),
                "n_test": len(test),
                "base_rate": base_rate,
                "marginal_acc": marginal_acc,
                "none_frac": float((test["recent_category"] == "none").mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    symbol = "BTCUSDT"
    log("Loading candles + news")
    candles = load_candles(symbol)
    asset = base_asset(symbol)
    news = load_news(asset=asset, include_llm=True)

    horizons_minutes = [120, 180, 240, 360]  # 2h, 3h, 4h, 6h

    summary_rows = []
    for h_min in horizons_minutes:
        h_candles = h_min // CANDLE_INTERVAL_MINUTES
        target_col = f"r_{h_candles}"
        log(f"\n=== Horizon {h_min}min ({h_candles} candles) ===")

        targets = compute_log_return_targets(candles, [h_candles], CANDLE_INTERVAL_MINUTES)
        dataset = targets[["ts", target_col]].copy()
        dataset["recent_category"] = most_recent_category(news, dataset["ts"], window_minutes=h_min).to_numpy()

        result = marginal_test(dataset, target_col, purge_rows=h_candles)
        log(result.to_string(index=False))

        delta = result["marginal_acc"] - result["base_rate"]
        mean_delta = delta.mean()
        n_beats = int((delta > 0).sum())
        log(f"Mean delta (marginal - base_rate): {mean_delta:.5f} ({mean_delta * 100:.2f}pp)")
        log(f"Folds beating base rate: {n_beats}/{len(result)}, mean none_frac={result['none_frac'].mean():.3f}")

        summary_rows.append(
            {
                "horizon_min": h_min,
                "mean_delta_pp": mean_delta * 100,
                "n_folds": len(result),
                "folds_beating_base_rate": n_beats,
                "mean_none_frac": result["none_frac"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    Path("reports").mkdir(exist_ok=True)
    summary.to_csv("reports/horizon_screen_summary.csv", index=False)
    log("\n=== Summary across all tested horizons ===")
    log(summary.to_string(index=False))

    log_research_run(
        __file__,
        run_name=f"intermediate-horizons-{symbol}",
        params={
            "symbol": symbol,
            "horizons_min": ",".join(map(str, horizons_minutes)),
        },
        metrics={
            "max_mean_delta_pp": float(summary["mean_delta_pp"].max()),
            "min_mean_delta_pp": float(summary["mean_delta_pp"].min()),
        },
        report_paths=sorted(Path("reports").glob("horizon_screen_*")),
    )


if __name__ == "__main__":
    main()
