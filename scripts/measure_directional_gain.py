"""Accuracy-program harness, DA-focused: technical_only vs technical_only +
<feature set>, point models, walk-forward on BTCUSDT, judged on directional
accuracy with a calendar-day-block bootstrap (evaluation.directional_block_bootstrap)
rather than the ~9-fold paired t-test that can't resolve DA under ~1.5pp.

Why this exists (accuracy-program "point 0"): #1-#3 established the point
models sit at the naive MAE floor in every regime, so pooled-MAE screens
were measuring near-zero headroom. The signal the models do carry is
directional (~53%), and #3 showed it has stable regime structure. This
harness makes DA the primary axis and gives every future candidate (#4
microstructure proxies, ...) a ~40x more sensitive read on it. MAE deltas
are still reported, just not the target.

Emits reports/directional_gain_<set>_folds.csv (per fold/horizon, both
metrics) and reports/directional_gain_<set>_daily.csv (per variant/horizon/
day sums, the bootstrap input), then prints the bootstrap table.

Usage: python scripts/measure_directional_gain.py <feature-set> [SYMBOL]
       feature sets: funding    (default SYMBOL: BTCUSDT)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _feature_sets import FEATURE_SETS
from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import daily_prediction_stats, directional_block_bootstrap, pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.splits import walk_forward_folds
from pinance_ml.tracking import log_research_run

OUT_DIR = Path("reports")


def log(msg: str, fh=None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        print(line, file=fh, flush=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in FEATURE_SETS:
        sys.exit(f"usage: measure_directional_gain.py <{'|'.join(FEATURE_SETS)}> [SYMBOL]")
    set_name = sys.argv[1]
    symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
    builder = FEATURE_SETS[set_name]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds_path = OUT_DIR / f"directional_gain_{set_name}_folds.csv"
    daily_path = OUT_DIR / f"directional_gain_{set_name}_daily.csv"
    fh = open(OUT_DIR / f"directional_gain_{set_name}.log", "a", encoding="utf-8")

    run_t0 = time.time()
    log(f"{symbol}: loading candles / building dataset [{set_name}]", fh)
    candles = load_candles(symbol)
    btc = load_candles("BTCUSDT") if symbol != "BTCUSDT" else None
    dataset = build_dataset(candles, btc_candles=btc)
    base_cols = feature_columns(dataset)
    dataset, extra_cols = builder(dataset, symbol)
    cov = dataset[extra_cols].notna().all(axis=1).mean()
    log(f"{symbol}: {len(dataset)} rows, +{extra_cols} (all-present coverage {cov:.1%})", fh)

    variants = {"technical_only": base_cols, f"with_{set_name}": base_cols + extra_cols}
    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    total = len(folds) * len(variants)
    log(f"{symbol}: {len(folds)} folds x {len(variants)} variants = {total} fold-variants", fh)

    fold_rows, daily_frames, imp_rows = [], [], []
    done = 0
    for fold in folds:
        for variant, cols in variants.items():
            t0 = time.time()
            models = train_horizon_models(fold.train, cols)
            preds = predict_horizons(models, fold.test, cols)
            for h in HORIZONS:
                actual = fold.test[f"r_{h}"].to_numpy()
                predicted = preds[f"r_{h}_pred"].to_numpy()
                fold_rows.append(
                    {
                        "symbol": symbol, "variant": variant, "fold": fold.index,
                        "test_start": fold.test_start, "horizon": h,
                        "mae": mae(actual, predicted),
                        "directional_accuracy": directional_accuracy(actual, predicted),
                        "n_test": int(pd.notna(actual).sum()),
                    }
                )
                if variant != "technical_only":
                    b = models[h].booster_
                    gains = dict(zip(b.feature_name(), b.feature_importance("gain")))
                    gain_total = sum(gains.values()) or 1.0
                    for col in extra_cols:
                        imp_rows.append(
                            {"fold": fold.index, "horizon": h, "feature": col,
                             "gain_pct_of_total": gains.get(col, 0.0) / gain_total * 100}
                        )
            daily_frames.append(daily_prediction_stats(fold.test, preds, HORIZONS, variant, symbol))
            done += 1
            eta = (time.time() - run_t0) / done * (total - done) / 60
            log(f"  [fold {fold.index}/{len(folds) - 1}] {variant}: 12 models in {time.time() - t0:.1f}s "
                f"({done}/{total}, ETA {eta:.1f}min)", fh)
        pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
        pd.concat(daily_frames, ignore_index=True).to_csv(daily_path, index=False)

    fold_results = pd.DataFrame(fold_rows)
    daily = pd.concat(daily_frames, ignore_index=True)
    daily.to_csv(daily_path, index=False)

    imp = pd.DataFrame(imp_rows)
    imp_summary = imp.groupby("feature")["gain_pct_of_total"].mean().sort_values(ascending=False)
    log(f"\n=== {set_name} feature gain (% of total split gain, mean over folds x horizons) ===", fh)
    log("\n" + imp_summary.to_string(), fh)
    log(f"total candidate-feature gain share: {imp_summary.sum():.2f}%", fh)

    log("\n=== Fold-level pooled (the old low-power view, for cross-check) ===", fh)
    pooled = pool_fold_metrics(fold_results)
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 160):
        log("\n" + pooled.to_string(index=False), fh)

    log(f"\n=== Day-block bootstrap: with_{set_name} vs technical_only (10000 draws, 7-day blocks) ===", fh)
    boot = directional_block_bootstrap(daily, "technical_only", f"with_{set_name}", horizons=HORIZONS)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        log("\n" + boot.to_string(index=False), fh)

    pooled_row = boot[boot["horizon"] == "pooled"].iloc[0]
    verdict = (
        "DA IMPROVES (CI excludes 0)" if pooled_row["ci_lo_pp"] > 0
        else "DA REGRESSES (CI excludes 0)" if pooled_row["ci_hi_pp"] < 0
        else "no resolvable DA effect (CI spans 0)"
    )
    log(f"\n>>> Pooled: dDA = {pooled_row['delta_da_pp']:+.3f}pp "
        f"[{pooled_row['ci_lo_pp']:+.3f}, {pooled_row['ci_hi_pp']:+.3f}] "
        f"P(dDA>0)={pooled_row['p_delta_gt_0']:.3f}  MAE {pooled_row['delta_mae_pct']:+.2f}% -> {verdict}", fh)
    log(f"\nSaved -> {folds_path}, {daily_path}. Total {(time.time() - run_t0) / 60:.1f}min", fh)

    log_research_run(
        __file__,
        run_name=f"{set_name}-{symbol}",
        params={
            "feature_set": set_name,
            "symbol": symbol,
            "n_folds": len(folds),
            "extra_cols": ",".join(extra_cols),
        },
        metrics={
            "delta_da_pp": float(pooled_row["delta_da_pp"]),
            "ci_lo_pp": float(pooled_row["ci_lo_pp"]),
            "ci_hi_pp": float(pooled_row["ci_hi_pp"]),
            "p_delta_gt_0": float(pooled_row["p_delta_gt_0"]),
            "delta_mae_pct": float(pooled_row["delta_mae_pct"]),
        },
        tags={"verdict": verdict},
        report_paths=sorted(OUT_DIR.glob(f"directional_gain_{set_name}_*")),
    )

    fh.close()


if __name__ == "__main__":
    main()
