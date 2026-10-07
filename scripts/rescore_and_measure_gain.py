"""One-off: measure_news_feature_gain.py's with_news variant reads
sentiment_pos/neu/neg straight from news_items (decay.py's
compute_news_features never calls score_sentiment) -- those columns hold
whatever NEWS_FINBERT_MODEL scored them with at fetch/backfill time
(ProsusAI/finbert, always, so far). Pointing the NEWS_FINBERT_MODEL env
var at a Level-3 candidate and rerunning measure_news_feature_gain.py
verbatim therefore compares nothing -- it reads the same stored
ProsusAI/finbert scores either way, regardless of the env var.

This script re-scores the symbol's news in memory (title only, matching
build_finbert_labels.py) with whatever NEWS_FINBERT_MODEL currently
points at, then runs the identical walk-forward gain-measurement loop
against those fresh in-memory scores -- the DB itself is never written,
this is read-only against news_items and candles.

Usage: NEWS_FINBERT_MODEL=/path/to/candidate python scripts/rescore_and_measure_gain.py BTCUSDT
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, NEWS_FINBERT_MODEL, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.data.news_db import load_news
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.news.decay import NEWS_FEATURE_COLUMNS, base_asset, compute_news_features
from pinance_ml.news.sentiment import score_sentiment
from pinance_ml.splits import walk_forward_folds


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    log(f"NEWS_FINBERT_MODEL = {NEWS_FINBERT_MODEL}")

    log(f"{symbol}: loading candles")
    candles = load_candles(symbol)
    dataset = build_dataset(candles)
    base_feat_cols = feature_columns(dataset)

    asset = base_asset(symbol)
    news = load_news(asset=asset)
    log(f"{asset}: {len(news)} news items -- re-scoring with {NEWS_FINBERT_MODEL} (title only)")
    t0 = time.time()
    fresh = score_sentiment(news["title"].tolist(), batch_size=64)
    news = news.reset_index(drop=True)
    news[["sentiment_pos", "sentiment_neu", "sentiment_neg", "sentiment_confidence"]] = fresh
    log(f"Re-scored {len(news)} items in {time.time() - t0:.1f}s")

    news_features = compute_news_features(news, dataset["ts"])
    dataset = pd.concat([dataset.reset_index(drop=True), news_features], axis=1)
    all_feat_cols = base_feat_cols + NEWS_FEATURE_COLUMNS

    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    log(f"{symbol}: {len(folds)} walk-forward folds")

    rows = []
    symbol_t0 = time.time()
    for fold in folds:
        for variant, feat_cols in (("technical_only", base_feat_cols), ("with_news", all_feat_cols)):
            t0 = time.time()
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
            log(f"  [fold {fold.index}/{len(folds) - 1}] {variant}: {time.time() - t0:.1f}s")

    fold_results = pd.DataFrame(rows)
    out_path = Path("reports/finbert_lora_gain_folds.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(out_path, index=False)

    technical = pool_fold_metrics(fold_results[fold_results["variant"] == "technical_only"])
    with_news = pool_fold_metrics(fold_results[fold_results["variant"] == "with_news"])
    comparison = technical.set_index(["symbol", "horizon"])[["mae", "directional_accuracy"]].join(
        with_news.set_index(["symbol", "horizon"])[["mae", "directional_accuracy"]],
        lsuffix="_technical_only", rsuffix="_with_news",
    )
    comparison["mae_delta"] = comparison["mae_with_news"] - comparison["mae_technical_only"]
    comparison["directional_accuracy_delta"] = (
        comparison["directional_accuracy_with_news"] - comparison["directional_accuracy_technical_only"]
    )
    comparison = comparison.reset_index()
    comparison.to_csv("reports/finbert_lora_gain_summary.csv", index=False)

    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 160):
        log("\n" + comparison.to_string(index=False))
    log(f"Total time: {(time.time() - symbol_t0) / 60:.1f}min")


if __name__ == "__main__":
    main()
