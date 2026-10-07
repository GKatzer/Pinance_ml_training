"""LightGBM report: MAE and directional accuracy per horizon, per symbol.

Direct multi-horizon (per README): 12 independent LGBMRegressor models, one
per r_h, sharing the same engineered feature matrix. Evaluated with the same
walk-forward-with-purging harness as the naive baseline, but at a wider test
window (LIGHTGBM_TEST_DAYS) — retraining 12 models per fold is far more
expensive than the naive baseline's constant prediction, so fewer/bigger
folds keep this tractable while staying genuinely walk-forward. Usage:

    python scripts/train_lightgbm.py [SYMBOL ...]

With no arguments, runs against every symbol found in the `candles` table.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import list_symbols, load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.splits import walk_forward_folds


def run_for_symbol(symbol: str, btc_candles: pd.DataFrame | None) -> pd.DataFrame:
    candles = load_candles(symbol)
    dataset = build_dataset(candles, btc_candles=btc_candles)
    feat_cols = feature_columns(dataset)

    rows = []
    for fold in walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS):
        models = train_horizon_models(fold.train, feat_cols)
        preds = predict_horizons(models, fold.test, feat_cols)
        for h in HORIZONS:
            actual = fold.test[f"r_{h}"].to_numpy()
            predicted = preds[f"r_{h}_pred"].to_numpy()
            rows.append(
                {
                    "symbol": symbol,
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
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", help="Symbols to evaluate (default: all in DB)")
    parser.add_argument("--out", default="reports/lightgbm_folds.csv")
    args = parser.parse_args()

    symbols = args.symbols or list_symbols()
    print(f"Symbols: {symbols}")

    btc_candles = load_candles("BTCUSDT") if any(s != "BTCUSDT" for s in symbols) else None

    fold_results = pd.concat(
        [run_for_symbol(s, None if s == "BTCUSDT" else btc_candles) for s in symbols],
        ignore_index=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(out_path, index=False)
    print(f"\nSaved {len(fold_results)} fold/horizon rows to {out_path}\n")

    summary = pool_fold_metrics(fold_results)
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 120):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
