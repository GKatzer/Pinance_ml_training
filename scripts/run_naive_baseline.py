"""Naive baseline report: MAE and directional accuracy per horizon, per symbol.

Naive forecast is r_h = 0 (next price = current price), the efficient-market
lower bound described in README.md. Evaluated via walk-forward validation
with purging (expanding train window, sliding weekly test window) rather
than a single train/test split, so this doubles as the reusable evaluation
harness later models will be scored with. Usage:

    python scripts/run_naive_baseline.py [SYMBOL ...]

With no arguments, runs against every symbol found in the `candles` table.
Writes per-fold detail to --out and prints a pooled summary per horizon.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.baselines.naive import naive_directional_accuracy, naive_mae
from pinance_ml.config import (
    CANDLE_INTERVAL_MINUTES,
    HORIZONS,
    PURGE_ROWS,
    WALK_FORWARD_MIN_TRAIN_DAYS,
    WALK_FORWARD_TEST_DAYS,
)
from pinance_ml.data.db import list_symbols, load_candles
from pinance_ml.evaluation import pool_fold_metrics
from pinance_ml.features.targets import compute_log_return_targets
from pinance_ml.splits import walk_forward_folds


def run_for_symbol(symbol: str) -> pd.DataFrame:
    candles = load_candles(symbol)
    targets = compute_log_return_targets(candles, HORIZONS, CANDLE_INTERVAL_MINUTES)

    rows = []
    for fold in walk_forward_folds(targets, WALK_FORWARD_MIN_TRAIN_DAYS, WALK_FORWARD_TEST_DAYS, PURGE_ROWS):
        for h in HORIZONS:
            col = f"r_{h}"
            train_vals = fold.train[col].to_numpy()
            test_vals = fold.test[col].to_numpy()
            rows.append(
                {
                    "symbol": symbol,
                    "fold": fold.index,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "horizon": h,
                    "mae": naive_mae(test_vals),
                    "directional_accuracy": naive_directional_accuracy(train_vals, test_vals),
                    "n_train": fold.train[col].notna().sum(),
                    "n_test": fold.test[col].notna().sum(),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", help="Symbols to evaluate (default: all in DB)")
    parser.add_argument("--out", default="reports/naive_baseline_folds.csv")
    args = parser.parse_args()

    symbols = args.symbols or list_symbols()
    print(f"Symbols: {symbols}")

    fold_results = pd.concat([run_for_symbol(s) for s in symbols], ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(out_path, index=False)
    print(f"\nSaved {len(fold_results)} fold/horizon rows to {out_path}\n")

    summary = pool_fold_metrics(fold_results)
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 120):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
