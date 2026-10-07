"""Accuracy-program check #3: regime-stratified error diagnostic (NOT a feature).

One straight walk-forward pass of the current production point models
(train_horizon_models, unchanged), dumping per-(fold, horizon, regime
bucket) error aggregates. Regimes are terciles cut on each fold's OWN
training rows -- causal, and regime-relative, since 2018's "low vol" is
not 2024's. Two stratifiers:

  A. vol LEVEL         ret_std_144            (12h realized vol)
  B. vol ACCELERATION  ret_std_6 / ret_std_144  (30min vs 12h; >1 = vol rising)

Reported per bucket: n, mae_model, mae_naive (= mean|r|, the r=0 baseline),
skill (= 1 - mae_model/mae_naive, the fraction of naive error removed),
and directional accuracy.

Question: are skill and directional accuracy roughly flat across regimes?
Flat => the single all-history model isn't leaving regime-specific accuracy
on the table, and the regime-conditioning branch (separate models / an
explicit regime feature) closes cheap. A collapse (or spike) in one bucket
=> there's headroom worth a follow-up.

Motivated by check #1's nuance: vol-scaling the target helped ~1.5% MAE
specifically in fold 4 (2022, fast vol regime shifts) and hurt in the 2018
low-liquidity era -- hence axis B, not just axis A.

Usage: python scripts/screen_regime_error.py [SYMBOL ...]   (default BTCUSDT)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.splits import walk_forward_folds
from pinance_ml.tracking import log_research_run

BUCKET_LABELS = ["lo", "mid", "hi"]
OUT_DIR = Path("reports")
FOLDS_PATH = OUT_DIR / "regime_error_folds.csv"
SUMMARY_PATH = OUT_DIR / "regime_error_summary.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def regime_series(df: pd.DataFrame, axis: str) -> pd.Series:
    if axis == "vol_level":
        return df["ret_std_144"]
    if axis == "vol_accel":
        return df["ret_std_6"] / df["ret_std_144"].replace(0.0, np.nan)
    raise ValueError(axis)


def bucketize(train_vals: pd.Series, test_vals: pd.Series) -> tuple[np.ndarray, tuple[float, float]]:
    e1, e2 = train_vals.quantile([1 / 3, 2 / 3]).to_numpy()
    b = np.digitize(test_vals.to_numpy(), [e1, e2])  # 0,1,2 ; NaN -> 2, masked out below via notna
    b = np.where(test_vals.isna().to_numpy(), -1, b)
    return b, (float(e1), float(e2))


def run_symbol(symbol: str, btc_candles: pd.DataFrame | None) -> list[dict]:
    candles = load_candles(symbol)
    dataset = build_dataset(candles, btc_candles=btc_candles)
    feat_cols = feature_columns(dataset)
    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    log(f"{symbol}: {len(dataset)} rows, {len(feat_cols)} features, {len(folds)} folds")

    rows: list[dict] = []
    run_t0 = time.time()
    for fold in folds:
        t0 = time.time()
        models = train_horizon_models(fold.train, feat_cols)
        preds = predict_horizons(models, fold.test, feat_cols)

        for axis in ("vol_level", "vol_accel"):
            tr = regime_series(fold.train, axis)
            te = regime_series(fold.test, axis)
            bucket, (e1, e2) = bucketize(tr, te)
            for h in HORIZONS:
                actual = fold.test[f"r_{h}"].to_numpy()
                pred = preds[f"r_{h}_pred"].to_numpy()
                for bi, blabel in enumerate(BUCKET_LABELS):
                    m = (bucket == bi) & ~np.isnan(actual) & ~np.isnan(pred)
                    n = int(m.sum())
                    if n == 0:
                        continue
                    a, p = actual[m], pred[m]
                    rows.append(
                        {
                            "symbol": symbol,
                            "axis": axis,
                            "fold": fold.index,
                            "test_start": fold.test_start,
                            "horizon": h,
                            "bucket": blabel,
                            "n": n,
                            "edge_lo": e1,
                            "edge_hi": e2,
                            "sum_abs_err": float(np.abs(a - p).sum()),
                            "sum_abs_actual": float(np.abs(a).sum()),
                            "n_dir_correct": int((np.sign(a) == np.sign(p)).sum()),
                        }
                    )
        eta = (time.time() - run_t0) / (fold.index + 1) * (len(folds) - fold.index - 1) / 60
        log(f"  fold {fold.index}/{len(folds) - 1}: {time.time() - t0:.1f}s (ETA {eta:.1f}min)")
    return rows


def pooled(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols).agg(
        n=("n", "sum"),
        sum_abs_err=("sum_abs_err", "sum"),
        sum_abs_actual=("sum_abs_actual", "sum"),
        n_dir_correct=("n_dir_correct", "sum"),
        n_folds=("fold", "nunique"),
    )
    g["mae_model"] = g["sum_abs_err"] / g["n"]
    g["mae_naive"] = g["sum_abs_actual"] / g["n"]
    g["skill"] = 1 - g["mae_model"] / g["mae_naive"]
    g["dir_acc"] = g["n_dir_correct"] / g["n"]
    return g[["n", "n_folds", "mae_model", "mae_naive", "skill", "dir_acc"]].reset_index()


def main() -> None:
    symbols = sys.argv[1:] or ["BTCUSDT"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    btc_candles = load_candles("BTCUSDT") if any(s != "BTCUSDT" for s in symbols) else None

    all_rows: list[dict] = []
    for s in symbols:
        all_rows.extend(run_symbol(s, None if s == "BTCUSDT" else btc_candles))
        pd.DataFrame(all_rows).to_csv(FOLDS_PATH, index=False)

    df = pd.DataFrame(all_rows)

    by_axis_bucket = pooled(df, ["symbol", "axis", "bucket"])
    by_axis_h_bucket = pooled(df, ["symbol", "axis", "horizon", "bucket"])
    by_axis_h_bucket.to_csv(SUMMARY_PATH, index=False)

    order = {"lo": 0, "mid": 1, "hi": 2}
    with pd.option_context("display.width", 200, "display.float_format", "{:.6f}".format):
        for sym in symbols:
            for axis in ("vol_level", "vol_accel"):
                sub = by_axis_bucket[(by_axis_bucket.symbol == sym) & (by_axis_bucket.axis == axis)].copy()
                sub = sub.sort_values("bucket", key=lambda s: s.map(order))
                log(f"\n=== {sym} / {axis} : pooled across all horizons & folds ===")
                log("\n" + sub.to_string(index=False))
                sh = by_axis_h_bucket[(by_axis_h_bucket.symbol == sym) & (by_axis_h_bucket.axis == axis)]
                piv_skill = sh.pivot(index="horizon", columns="bucket", values="skill")[["lo", "mid", "hi"]]
                piv_da = sh.pivot(index="horizon", columns="bucket", values="dir_acc")[["lo", "mid", "hi"]]
                piv = piv_skill.join(piv_da, lsuffix="_skill", rsuffix="_da")
                piv["skill_hi_minus_lo"] = piv_skill["hi"] - piv_skill["lo"]
                piv["da_hi_minus_lo"] = piv_da["hi"] - piv_da["lo"]
                log(f"\n{sym} / {axis} : skill & dir_acc by horizon x bucket")
                log("\n" + piv.to_string())

    log(f"\nSaved fold rows -> {FOLDS_PATH}, per-horizon summary -> {SUMMARY_PATH}")
    log("\nRead: if skill and dir_acc are ~flat lo->hi (|hi - lo| small vs the numbers themselves), "
        "the single model isn't leaving regime accuracy on the table -> regime-conditioning branch closes. "
        "A consistent collapse in one bucket across horizons -> headroom.")

    log_research_run(
        __file__,
        run_name=f"regime-error-{'+'.join(symbols)}" if len(symbols) <= 4 else f"regime-error-{len(symbols)}sym",
        params={
            "symbols": ",".join(symbols),
            "n_fold_rows": len(df),
        },
        report_paths=sorted(OUT_DIR.glob("regime_error_*")),
    )


if __name__ == "__main__":
    main()
