"""One-off: does Level 2's LLM-classified event_type (README "Уровень 2")
add downstream signal beyond Level 1's FinBERT-based features? Same
walk-forward gain-measurement structure as scripts/rescore_and_measure_gain.py
used for Level 3, but additive (a new orthogonal feature set) rather than
a model swap -- three variants per fold instead of two:

  - technical_only: price/indicator features alone
  - with_level1_news: + compute_news_features (FinBERT sentiment decay +
    keyword-matched event flags) -- today's production feature set
  - with_level1_and_llm_events: + compute_llm_event_features on top --
    the candidate (news/decay.py's LLM_EVENT_FEATURE_COLUMNS)

The technical_only vs with_level1_news delta is a sanity check (should
roughly reproduce whatever gain Level 1 already showed); the question
this script actually answers is with_level1_news vs
with_level1_and_llm_events.

Needs scripts/score_pending_news_llm.py already run for `symbol`'s asset
(news_items.llm_event_type populated) -- rows still NULL there just fall
through compute_llm_event_features as "no recent event of that type",
same as if they'd never been fetched, so a partial backfill doesn't
crash this, it just weakens the candidate's chance of showing gain.

Usage: python scripts/measure_llm_event_gain.py [BTCUSDT]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.data.news_db import load_news
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.news.decay import (
    LLM_EVENT_FEATURE_COLUMNS,
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

    level1_features = compute_news_features(news, dataset["ts"])
    llm_features = compute_llm_event_features(news, dataset["ts"])
    dataset = pd.concat([dataset.reset_index(drop=True), level1_features, llm_features], axis=1)

    variants = {
        "technical_only": base_feat_cols,
        "with_level1_news": base_feat_cols + NEWS_FEATURE_COLUMNS,
        "with_level1_and_llm_events": base_feat_cols + NEWS_FEATURE_COLUMNS + LLM_EVENT_FEATURE_COLUMNS,
    }

    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    log(f"{symbol}: {len(folds)} walk-forward folds")

    rows = []
    t0 = time.time()
    for fold in folds:
        for variant, feat_cols in variants.items():
            vt0 = time.time()
            models = train_horizon_models(fold.train, feat_cols)
            preds = predict_horizons(models, fold.test, feat_cols)
            for h in HORIZONS:
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
    out_path = Path("reports/llm_event_gain_folds.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(out_path, index=False)

    pooled = {
        variant: pool_fold_metrics(fold_results[fold_results["variant"] == variant]).set_index(["symbol", "horizon"])
        for variant in variants
    }
    comparison = pooled["technical_only"][["mae", "directional_accuracy"]].rename(
        columns=lambda c: f"{c}_technical_only"
    )
    for variant in ["with_level1_news", "with_level1_and_llm_events"]:
        comparison = comparison.join(pooled[variant][["mae", "directional_accuracy"]].rename(columns=lambda c: f"{c}_{variant}"))

    comparison["mae_delta_level1_vs_technical"] = (
        comparison["mae_with_level1_news"] - comparison["mae_technical_only"]
    )
    comparison["mae_delta_llm_vs_level1"] = (
        comparison["mae_with_level1_and_llm_events"] - comparison["mae_with_level1_news"]
    )
    comparison["dir_acc_delta_llm_vs_level1"] = (
        comparison["directional_accuracy_with_level1_and_llm_events"]
        - comparison["directional_accuracy_with_level1_news"]
    )
    comparison = comparison.reset_index()
    comparison.to_csv("reports/llm_event_gain_summary.csv", index=False)

    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 200, "display.max_columns", 20):
        log("\n" + comparison.to_string(index=False))
    log(f"Total time: {(time.time() - t0) / 60:.1f}min")

    log_research_run(
        __file__,
        run_name=f"llm-event-{symbol}",
        params={
            "symbol": symbol,
            "n_folds": len(folds),
            "n_news_items": len(news),
            "n_llm_classified": n_classified,
            "variants": ",".join(variants),
        },
        metrics={
            "mae_delta_level1_vs_technical_mean": float(comparison["mae_delta_level1_vs_technical"].mean()),
            "mae_delta_llm_vs_level1_mean": float(comparison["mae_delta_llm_vs_level1"].mean()),
            "dir_acc_delta_llm_vs_level1_mean": float(comparison["dir_acc_delta_llm_vs_level1"].mean()),
        },
        report_paths=sorted(Path("reports").glob("llm_event_gain_*")),
    )


if __name__ == "__main__":
    main()
