"""One-off: which max_train_days actually gives the best pooled MAE?

Reuses the same walk-forward-with-purging harness as
scripts/measure_news_feature_gain.py, technical-only features, comparing
a few candidate window lengths against the current expanding-window
default (max_train_days=None) -- see splits.py::walk_forward_folds for
why "expanding forever" stops meaningfully refreshing anything once
history spans years. Not meant to be a permanent script: a quick,
disposable check to pick a number instead of guessing one, per this
repo's own rule of measuring before concluding.

Usage: python scripts/compare_train_windows.py [SYMBOL] [--windows 90,180,365,730,]
(empty entry in --windows = expanding/None)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.splits import walk_forward_folds


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbol", nargs="?", default="BTCUSDT")
    parser.add_argument("--windows", default="90,180,365,730,")
    parser.add_argument("--out", default="reports/train_window_comparison.csv")
    args = parser.parse_args()

    windows = [float(w) if w.strip() else None for w in args.windows.split(",")]

    symbol = args.symbol
    log(f"{symbol}: loading candles")
    candles = load_candles(symbol)
    dataset = build_dataset(candles)
    feat_cols = feature_columns(dataset)
    log(f"{symbol}: {len(dataset)} rows, {len(feat_cols)} features")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for window in windows:
        label = f"{window:.0f}d" if window is not None else "expanding"
        folds = list(
            walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, max_train_days=window)
        )
        log(f"{label}: {len(folds)} folds")

        for fold in folds:
            t0 = time.time()
            models = train_horizon_models(fold.train, feat_cols)
            preds = predict_horizons(models, fold.test, feat_cols)
            for h in HORIZONS:
                actual = fold.test[f"r_{h}"].to_numpy()
                predicted = preds[f"r_{h}_pred"].to_numpy()
                rows.append(
                    {
                        "symbol": label,  # pool_fold_metrics groups by "symbol"; window label stands in for it here
                        "fold": fold.index,
                        "test_start": fold.test_start,
                        "n_train": len(fold.train),
                        "horizon": h,
                        "mae": mae(actual, predicted),
                        "directional_accuracy": directional_accuracy(actual, predicted),
                        "n_test": int(pd.notna(actual).sum()),
                    }
                )
            log(f"  [{label}] fold {fold.index}/{len(folds) - 1}: n_train={len(fold.train)} ({time.time() - t0:.1f}s)")
        pd.DataFrame(rows).to_csv(out_path, index=False)

    results = pd.DataFrame(rows)
    per_horizon = pool_fold_metrics(results)  # one row per (window, horizon)
    per_horizon.to_csv(out_path.parent / "train_window_comparison_per_horizon.csv", index=False)

    # Pool once more across horizons, weighting by each horizon's own
    # n_test_total, for a single overall-MAE-per-window number.
    per_horizon["mae_weighted"] = per_horizon["mae"] * per_horizon["n_test_total"]
    per_horizon["da_weighted"] = per_horizon["directional_accuracy"] * per_horizon["n_test_total"]
    overall = per_horizon.groupby("symbol").apply(
        lambda g: pd.Series(
            {
                "mean_mae": g["mae_weighted"].sum() / g["n_test_total"].sum(),
                "mean_directional_accuracy": g["da_weighted"].sum() / g["n_test_total"].sum(),
                "n_folds": results[results["symbol"] == g.name]["fold"].nunique(),
            }
        )
    ).rename_axis("window")

    log("\n" + overall.to_string())
    overall.to_csv(out_path.parent / "train_window_comparison_summary.csv")
    log(f"Saved -> {out_path}, per-horizon breakdown, and summary in {out_path.parent}")


if __name__ == "__main__":
    main()
