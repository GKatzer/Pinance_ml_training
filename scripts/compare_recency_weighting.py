"""One-off: does recency-weighted sample_weight beat plain uniform weight
on the expanding window that scripts/compare_train_windows.py already
found to win over any sliding window?

Same harness as compare_train_windows.py, but instead of varying
max_train_days, keeps the window expanding (max_train_days=None, the
empirical winner) and varies the recency half-life used for
train_horizon_models' sample_weight -- see splits.recency_sample_weight.
Uniform-weight baseline is NOT recomputed here -- with deterministic=True
and the same folds/params, it would just reproduce
compare_train_windows.py's "expanding" row (mean_mae=0.002796,
mean_directional_accuracy=0.531371 for BTCUSDT) at the cost of another
~28 minutes. Pass "none" in --half-lives if you want it recomputed anyway
(e.g. checking a different symbol that hasn't been run yet).

Each half-life costs about as long as one full expanding-window pass
(~25-30 min for BTCUSDT, 8 folds) since weighting doesn't reduce the
number of rows LightGBM processes -- default below is 3 values, not
6+, to keep total runtime to roughly 1-1.5 hours instead of 3.

Usage: python scripts/compare_recency_weighting.py [SYMBOL] [--half-lives 30,180,730]
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
from pinance_ml.splits import recency_sample_weight, walk_forward_folds


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbol", nargs="?", default="BTCUSDT")
    parser.add_argument("--half-lives", default="30,180,730")
    parser.add_argument("--out", default="reports/recency_weighting_comparison.csv")
    args = parser.parse_args()

    half_lives = [None if h.strip().lower() == "none" else float(h) for h in args.half_lives.split(",")]

    symbol = args.symbol
    log(f"{symbol}: loading candles")
    candles = load_candles(symbol)
    dataset = build_dataset(candles)
    feat_cols = feature_columns(dataset)
    log(f"{symbol}: {len(dataset)} rows, {len(feat_cols)} features")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # Expanding window throughout -- compare_train_windows.py already
    # found it beats every bounded window tested, so that's the baseline
    # recency-weighting needs to improve on, not re-litigate.
    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, max_train_days=None))
    log(f"{len(folds)} folds (expanding window)")

    for half_life in half_lives:
        label = f"hl_{half_life:.0f}d" if half_life is not None else "uniform"

        for fold in folds:
            t0 = time.time()
            weights = (
                recency_sample_weight(fold.train["ts"], fold.test_start, half_life)
                if half_life is not None
                else None
            )
            models = train_horizon_models(fold.train, feat_cols, sample_weight=weights)
            preds = predict_horizons(models, fold.test, feat_cols)
            for h in HORIZONS:
                actual = fold.test[f"r_{h}"].to_numpy()
                predicted = preds[f"r_{h}_pred"].to_numpy()
                rows.append(
                    {
                        "symbol": label,  # pool_fold_metrics groups by "symbol"; half-life label stands in
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
    per_horizon = pool_fold_metrics(results)
    per_horizon.to_csv(out_path.parent / "recency_weighting_comparison_per_horizon.csv", index=False)

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
    ).rename_axis("half_life")

    log("\n" + overall.to_string())
    overall.to_csv(out_path.parent / "recency_weighting_comparison_summary.csv")
    log(f"Saved -> {out_path}, per-horizon breakdown, and summary in {out_path.parent}")


if __name__ == "__main__":
    main()
