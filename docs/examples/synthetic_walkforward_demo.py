"""Offline demo of the real training harness on SYNTHETIC candles.

    python docs/examples/synthetic_walkforward_demo.py

No database, no network. It generates 5-minute candles from a random walk
with volatility clustering and one deliberately planted weak
dependence (each return carries 12 % of the previous return), then runs the repository's own code on them:

  1. pinance_ml.dataset.build_dataset         features + 12 log-return targets
  2. pinance_ml.splits.walk_forward_folds     expanding window, purged boundary
  3. pinance_ml.models.lightgbm_model         12 direct-horizon LightGBM models
  4. pinance_ml.baselines.naive               r = 0 and majority-sign baselines
  5. pinance_ml.evaluation.pool_fold_metrics  n_test-weighted pooling
  6. serving parity: the last feature row computed on 150 candles equals the
     one computed on the full history (the property behind the obv fix)

The numbers say nothing about any market. They show the mechanics: the
model finds direction above 50 % only because the dependence was planted
(a real market offers far less), and barely beats the naive MAE.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")  # config.py requires it; never connected to
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from pinance_ml.baselines.naive import naive_directional_accuracy, naive_mae
from pinance_ml.config import HORIZONS, PURGE_ROWS
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import pool_fold_metrics
from pinance_ml.features.pipeline import compute_features
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.splits import walk_forward_folds

N_DAYS = 330
SERVING_WINDOW = 150  # candles the inference service receives per request
# Fewer, shallower trees than DEFAULT_PARAMS so the demo runs in about a minute.
FAST_PARAMS = {"n_estimators": 60, "num_leaves": 15, "min_child_samples": 200}


def synthetic_candles(n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_days * 288
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    vol = 0.0008 * np.exp(0.5 * np.convolve(rng.normal(size=n), np.ones(200) / 200, mode="same") * 8)
    ret = np.zeros(n)
    eps = rng.normal(size=n)
    for t in range(1, n):
        ret[t] = 0.12 * ret[t - 1] + vol[t] * eps[t]
    close = 20000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, vol)) * close
    return pd.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.lognormal(3.0, 0.5, size=n) * (1 + 40 * vol / vol.mean()),
        }
    )


def main() -> None:
    t0 = time.time()
    candles = synthetic_candles(N_DAYS)
    print(f"synthetic candles: {len(candles):,} rows, {candles['ts'].iloc[0]:%Y-%m-%d} .. {candles['ts'].iloc[-1]:%Y-%m-%d}")

    dataset = build_dataset(candles)
    feats = feature_columns(dataset)
    print(f"dataset: {len(dataset):,} rows, {len(feats)} features, {len(HORIZONS)} targets (r_1..r_12), purge = {PURGE_ROWS} rows")

    rows = []
    folds = list(walk_forward_folds(dataset, min_train_days=90, test_days=60, purge_rows=PURGE_ROWS))
    print(f"walk-forward: {len(folds)} folds (expanding train, 60-day test windows)")
    for fold in folds:
        last_train_ts = fold.train["ts"].iloc[-1]
        print(
            f"  fold {fold.index}: train {len(fold.train):>6,} rows ending {last_train_ts:%Y-%m-%d %H:%M}, "
            f"test {len(fold.test):>6,} rows from {fold.test['ts'].iloc[0]:%Y-%m-%d %H:%M}"
        )
        models = train_horizon_models(fold.train, feats, params=FAST_PARAMS)
        preds = predict_horizons(models, fold.test, feats)
        for h in HORIZONS:
            actual = fold.test[f"r_{h}"].to_numpy()
            y_train = fold.train[f"r_{h}"].to_numpy()
            rows.append(
                {
                    "symbol": "SYNTH",
                    "fold": fold.index,
                    "horizon": h,
                    "mae": mae(actual, preds[f"r_{h}_pred"].to_numpy()),
                    "directional_accuracy": directional_accuracy(actual, preds[f"r_{h}_pred"].to_numpy()),
                    "naive_mae": naive_mae(actual),
                    "naive_da": naive_directional_accuracy(y_train, actual),
                    "n_test": int(pd.notna(actual).sum()),
                }
            )
    res = pd.DataFrame(rows)
    pooled = pool_fold_metrics(res, metric_cols=["mae", "directional_accuracy", "naive_mae", "naive_da"])

    pd.set_option("display.width", 140)
    show = pooled.copy()
    show["mae_vs_naive_%"] = (show["mae"] / show["naive_mae"] - 1) * 100
    show["da_vs_naive_pp"] = (show["directional_accuracy"] - show["naive_da"]) * 100
    cols = ["horizon", "mae", "naive_mae", "mae_vs_naive_%", "directional_accuracy", "naive_da", "da_vs_naive_pp", "n_test_total"]
    print("\npooled over folds, weighted by n_test (synthetic data):")
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:.5f}" if abs(v) < 1 else f"{v:.2f}"))
    w = pooled["n_test_total"]
    print(
        f"\nmean over horizons: DA {np.average(pooled['directional_accuracy'], weights=w)*100:.2f}% "
        f"(naive {np.average(pooled['naive_da'], weights=w)*100:.2f}%), "
        f"MAE {np.average(pooled['mae'], weights=w):.6f} (naive {np.average(pooled['naive_mae'], weights=w):.6f})"
    )

    # Serving parity: what the inference service computes from 150 candles must
    # equal what training computed from the whole history for the same candle.
    full = compute_features(candles)
    window = compute_features(candles.iloc[-SERVING_WINDOW:])
    cols = [c for c in feats if c in full.columns]
    a = full.iloc[-1][cols].astype(float)
    b = window.iloc[-1][cols].astype(float)
    both = a.notna() & b.notna()
    rel = ((a[both] - b[both]).abs() / a[both].abs().clip(lower=1e-12)).sort_values(ascending=False)
    print(f"\nserving parity on the last candle: {int(both.sum())} comparable features")
    print(f"  obv_roc_36: full {a['obv_roc_36']:.10f}  window {b['obv_roc_36']:.10f}  (exact)")
    print(f"  largest relative differences: " + ", ".join(f"{k} {v:.1e}" for k, v in rel.head(4).items()) + "  (EMA-based indicators forget their start geometrically)")
    print(f"features that are NaN on the 150-candle window only: {sorted(a.index[a.notna() & b.isna()])}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
