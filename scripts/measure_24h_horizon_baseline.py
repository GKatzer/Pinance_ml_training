"""One-off: 3-way walk-forward comparison at the 24h prediction horizon
(project notes) -- technical_only vs
+Level 1 news vs +Level 2's event_type classified over a 24h recency
window (news/decay.py's `compute_llm_event_features(..., suffix="24h")`,
motivated by the horizon-sweep finding in the project notes
that event_type correlates with *direction* at 24h, unlike at 60min).

The technical_only vs with_level1_news baseline (run first, 2026-08-01)
came back weak: 50.72% directional accuracy on technical indicators alone
(barely above a coin flip), and Level 1 news made it *worse*
(50.60%, mae +0.00027) -- context for interpreting whatever the 24h
event_type variant shows here, since it's being added on top of an
already near-random baseline, not a strong one.

Uses LONG_HORIZONS/LONG_PURGE_ROWS (config.py), never HORIZONS/PURGE_ROWS
-- folding r_288 into the existing 5-60min models' fold generation would
force their purge width from 12 to 288 rows for no benefit to them (see
the project notes, point 1). ROLLING_WINDOWS (feature lookback,
max 12h) is deliberately left untouched for this first pass too.

Usage: python scripts/measure_24h_horizon_baseline.py [BTCUSDT]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.config import (
    LIGHTGBM_TEST_DAYS,
    LONG_HORIZONS,
    LONG_PURGE_ROWS,
    NEWS_RECENT_WINDOW_24H_MINUTES,
    WALK_FORWARD_MIN_TRAIN_DAYS,
)
from pinance_ml.data.db import load_candles
from pinance_ml.data.news_db import load_news
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.news.decay import (
    LLM_EVENT_FEATURE_COLUMNS_24H,
    NEWS_FEATURE_COLUMNS,
    base_asset,
    compute_llm_event_features,
    compute_news_features,
)
from pinance_ml.splits import walk_forward_folds
from pinance_ml.tracking import log_research_run


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"

    log(f"{symbol}: loading candles")
    candles = load_candles(symbol)
    dataset = build_dataset(candles)
    base_feat_cols = feature_columns(dataset)

    asset = base_asset(symbol)
    news = load_news(asset=asset, include_llm=True)
    n_classified = int(news["llm_event_type"].notna().sum())
    log(f"{asset}: {len(news)} news items, {n_classified} with Level 2 event_type ({n_classified / max(len(news), 1):.1%})")

    news_features = compute_news_features(news, dataset["ts"])
    llm_features_24h = compute_llm_event_features(
        news, dataset["ts"], window_minutes=NEWS_RECENT_WINDOW_24H_MINUTES, suffix="24h"
    )
    dataset = pd.concat([dataset.reset_index(drop=True), news_features, llm_features_24h], axis=1)

    variants = {
        "technical_only": base_feat_cols,
        "with_level1_news": base_feat_cols + NEWS_FEATURE_COLUMNS,
        "with_level1_and_llm_event_24h": base_feat_cols + NEWS_FEATURE_COLUMNS + LLM_EVENT_FEATURE_COLUMNS_24H,
    }

    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, LONG_PURGE_ROWS))
    log(f"{symbol}: {len(folds)} walk-forward folds (purge={LONG_PURGE_ROWS} rows for horizon(s) {LONG_HORIZONS})")

    rows = []
    t0 = time.time()
    for fold in folds:
        for variant, feat_cols in variants.items():
            vt0 = time.time()
            models = train_horizon_models(fold.train, feat_cols, horizons=LONG_HORIZONS)
            preds = predict_horizons(models, fold.test, feat_cols)
            for h in LONG_HORIZONS:
                actual = fold.test[f"r_{h}"].to_numpy()
                predicted = preds[f"r_{h}_pred"].to_numpy()
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": variant,
                        "fold": fold.index,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "horizon": h,
                        "mae": mae(actual, predicted),
                        "directional_accuracy": directional_accuracy(actual, predicted),
                        "n_train": len(fold.train),
                        "n_test": int(pd.notna(actual).sum()),
                    }
                )
            log(f"  [fold {fold.index}/{len(folds) - 1}] {variant}: {time.time() - vt0:.1f}s")

    fold_results = pd.DataFrame(rows)
    out_path = Path("reports/horizon_24h_baseline_folds.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(out_path, index=False)

    pooled = {
        variant: pool_fold_metrics(fold_results[fold_results["variant"] == variant]).set_index(["symbol", "horizon"])
        for variant in variants
    }
    comparison = pooled["technical_only"][["mae", "directional_accuracy", "n_folds"]].rename(
        columns=lambda c: f"{c}_technical_only" if c != "n_folds" else c
    )
    for variant in ["with_level1_news", "with_level1_and_llm_event_24h"]:
        comparison = comparison.join(pooled[variant][["mae", "directional_accuracy"]].rename(columns=lambda c: f"{c}_{variant}"))

    comparison["mae_delta_level1_vs_technical"] = comparison["mae_with_level1_news"] - comparison["mae_technical_only"]
    comparison["mae_delta_llm24h_vs_level1"] = (
        comparison["mae_with_level1_and_llm_event_24h"] - comparison["mae_with_level1_news"]
    )
    comparison["dir_acc_delta_llm24h_vs_level1"] = (
        comparison["directional_accuracy_with_level1_and_llm_event_24h"]
        - comparison["directional_accuracy_with_level1_news"]
    )
    comparison = comparison.reset_index()
    comparison.to_csv("reports/horizon_24h_baseline_summary.csv", index=False)

    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 180):
        log("\n" + comparison.to_string(index=False))
    log(f"Total time: {(time.time() - t0) / 60:.1f}min")

    log_research_run(
        __file__,
        run_name=f"horizon24h-{symbol}",
        params={
            "symbol": symbol,
            "n_folds": len(folds),
            "n_news_items": len(news),
            "n_llm_classified": n_classified,
            "variants": ",".join(variants),
            "long_horizons": ",".join(map(str, LONG_HORIZONS)),
        },
        metrics={
            "mae_delta_level1_vs_technical_mean": float(comparison["mae_delta_level1_vs_technical"].mean()),
            "mae_delta_llm24h_vs_level1_mean": float(comparison["mae_delta_llm24h_vs_level1"].mean()),
            "dir_acc_delta_llm24h_vs_level1_mean": float(comparison["dir_acc_delta_llm24h_vs_level1"].mean()),
        },
        report_paths=sorted(Path("reports").glob("horizon_24h_baseline_*")),
    )


if __name__ == "__main__":
    main()
