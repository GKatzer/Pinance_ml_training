"""Accuracy-program "point 0" follow-up: was the point model ever tuned?

DEFAULT_PARAMS (lightgbm_model.py) has been fixed at n_estimators=300,
learning_rate=0.05, num_leaves=31, min_child_samples=100 since the start.
Checks #4/#5/#6 each showed the model spends 6-9% of its split-gain budget
on features that add nothing out-of-sample -- i.e. it overfits weak
inputs. Hypothesis: a more regularised config resists that AND squeezes a
little directional accuracy out of the real features. No new data.

Pre-registered grid (written before the first run) -- five regularisation
variants plus one deliberately higher-capacity config as a direction check:

  default        current production params (baseline)
  reg_mild       num_leaves 15, min_child 300, feature/bagging subsample
  reg_strong     num_leaves 15, min_child 500, heavier subsample + L1/L2
  reg_shallow    num_leaves 7,  min_child 500, lr 0.03 / 600 trees
  fewer_trees    default but n_estimators 150 -- is 300 already past the OOS peak?
  more_capacity  num_leaves 63, min_child 50, 400 trees -- opposite direction

Pre-registered read: a config only counts as a real improvement if its
pooled Delta-DA vs default has a 95% CI (day-block bootstrap) excluding 0
on the positive side on the HELD-OUT folds 5-8, having also been >= default
on the selection folds 0-4. Folds 0-4 pick, folds 5-8 judge; all-9 is
context only. Anything short of that closes the "just tune it" branch.

Usage: python scripts/tune_lightgbm_directional.py [SYMBOL]   (default BTCUSDT)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import daily_prediction_stats, directional_block_bootstrap, pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.splits import walk_forward_folds

CONFIGS: dict[str, dict] = {
    "default": {},
    "reg_mild": dict(num_leaves=15, min_child_samples=300, feature_fraction=0.8,
                     bagging_fraction=0.8, bagging_freq=1),
    "reg_strong": dict(num_leaves=15, min_child_samples=500, feature_fraction=0.6,
                       bagging_fraction=0.7, bagging_freq=1, lambda_l1=1.0, lambda_l2=1.0, n_estimators=400,
                       learning_rate=0.04),
    "reg_shallow": dict(num_leaves=7, min_child_samples=500, feature_fraction=0.7,
                        bagging_fraction=0.8, bagging_freq=1, n_estimators=600, learning_rate=0.03),
    "fewer_trees": dict(n_estimators=150),
    "more_capacity": dict(num_leaves=63, min_child_samples=50, feature_fraction=0.9, n_estimators=400),
}

SELECT_FOLDS = range(0, 5)
JUDGE_FOLDS = range(5, 9)

FOLDS_PATH = Path("reports/tune_directional_folds.csv")
DAILY_PATH = Path("reports/tune_directional_daily.csv")
LOG_PATH = Path("reports/tune_directional.log")


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, "a", encoding="utf-8")

    def log(m: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True)
        print(line, file=fh, flush=True)

    run_t0 = time.time()
    log(f"{symbol}: building dataset")
    dataset = build_dataset(load_candles(symbol))
    feat_cols = feature_columns(dataset)
    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    total = len(folds) * len(CONFIGS)
    log(f"{symbol}: {len(dataset)} rows, {len(folds)} folds x {len(CONFIGS)} configs = {total} fits-of-12")

    fold_rows, daily_frames = [], []
    done = 0
    for fold in folds:
        for name, params in CONFIGS.items():
            t0 = time.time()
            models = train_horizon_models(fold.train, feat_cols, params=params or None)
            preds = predict_horizons(models, fold.test, feat_cols)
            for h in HORIZONS:
                a = fold.test[f"r_{h}"].to_numpy()
                p = preds[f"r_{h}_pred"].to_numpy()
                fold_rows.append({
                    "symbol": symbol, "variant": name, "fold": fold.index, "horizon": h,
                    "test_start": fold.test_start, "mae": mae(a, p),
                    "directional_accuracy": directional_accuracy(a, p), "n_test": int(pd.notna(a).sum()),
                })
            daily_frames.append(daily_prediction_stats(fold.test, preds, HORIZONS, name, symbol).assign(fold=fold.index))
            done += 1
            eta = (time.time() - run_t0) / done * (total - done) / 60
            log(f"  [fold {fold.index}/{len(folds)-1}] {name}: {time.time()-t0:.1f}s ({done}/{total}, ETA {eta:.1f}min)")
        pd.DataFrame(fold_rows).to_csv(FOLDS_PATH, index=False)
        pd.concat(daily_frames, ignore_index=True).to_csv(DAILY_PATH, index=False)

    daily = pd.concat(daily_frames, ignore_index=True)
    fold_results = pd.DataFrame(fold_rows)

    log("\n=== Pooled DA per config (fold-level, n-weighted across horizons) ===")
    for tag, fset in (("select folds 0-4", SELECT_FOLDS), ("judge folds 5-8", JUDGE_FOLDS), ("all 9", range(9))):
        sub = fold_results[fold_results["fold"].isin(list(fset))]
        pooled = pool_fold_metrics(sub, group_cols=["symbol", "variant"])
        pooled = pooled.sort_values("directional_accuracy", ascending=False)
        log(f"\n[{tag}]\n" + pooled[["variant", "directional_accuracy", "mae"]].to_string(index=False))

    log("\n=== Day-block bootstrap vs default (pooled Delta-DA) ===")
    for tag, fset in (("SELECT 0-4", SELECT_FOLDS), ("JUDGE 5-8", JUDGE_FOLDS), ("all 9", range(9))):
        d = daily[daily["fold"].isin(list(fset))]
        rows = []
        for name in CONFIGS:
            if name == "default":
                continue
            b = directional_block_bootstrap(d, "default", name, horizons=HORIZONS, n_boot=10000)
            p = b[b["horizon"] == "pooled"].iloc[0]
            rows.append({"config": name, "delta_da_pp": p["delta_da_pp"], "ci_lo_pp": p["ci_lo_pp"],
                         "ci_hi_pp": p["ci_hi_pp"], "p_gt_0": p["p_delta_gt_0"], "delta_mae_pct": p["delta_mae_pct"]})
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
            log(f"\n[{tag}]\n" + pd.DataFrame(rows).to_string(index=False))

    log(f"\nSaved -> {FOLDS_PATH}, {DAILY_PATH}. Total {(time.time()-run_t0)/60:.1f}min")
    fh.close()


if __name__ == "__main__":
    main()
