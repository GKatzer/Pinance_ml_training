"""Cross-symbol validation of the `reg_strong` LightGBM config that passed
the pre-registered directional bar on BTCUSDT (tune_lightgbm_directional.py).

default vs reg_strong only, on ETH/BNB/SOL, same walk-forward + day-block
bootstrap + select(0-4)/judge(5-8) split. reg_strong is a real find only if
its held-out pooled Delta-DA CI excludes 0 on the positive side on at least
2 of the 3 symbols (and the 3-symbol pooled CI too). Falling short means the
BTCUSDT gain doesn't generalise and DEFAULT_PARAMS stays as is.

Usage: python scripts/validate_reg_strong.py [SYMBOL ...]
Default symbols: ETHUSDT BNBUSDT SOLUSDT
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import daily_prediction_stats, directional_block_bootstrap, pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.splits import walk_forward_folds
from tune_lightgbm_directional import CONFIGS

VARIANTS = {"default": {}, "reg_strong": CONFIGS["reg_strong"]}
DAILY_PATH = Path("reports/validate_reg_strong_daily.csv")
SUMMARY_PATH = Path("reports/validate_reg_strong_summary.csv")
LOG_PATH = Path("reports/validate_reg_strong.log")


def main() -> None:
    symbols = sys.argv[1:] or ["ETHUSDT", "BNBUSDT", "SOLUSDT"]
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, "a", encoding="utf-8")

    def log(m: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True)
        print(line, file=fh, flush=True)

    run_t0 = time.time()
    btc = load_candles("BTCUSDT")
    all_daily, all_fold = [], []

    for sym in symbols:
        dataset = build_dataset(load_candles(sym), btc_candles=None if sym == "BTCUSDT" else btc)
        feat_cols = feature_columns(dataset)
        folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
        log(f"{sym}: {len(dataset)} rows, {len(folds)} folds")
        for fold in folds:
            for name, params in VARIANTS.items():
                t0 = time.time()
                models = train_horizon_models(fold.train, feat_cols, params=params or None)
                preds = predict_horizons(models, fold.test, feat_cols)
                for h in HORIZONS:
                    a = fold.test[f"r_{h}"].to_numpy()
                    p = preds[f"r_{h}_pred"].to_numpy()
                    all_fold.append({"symbol": sym, "variant": name, "fold": fold.index, "horizon": h,
                                     "mae": mae(a, p), "directional_accuracy": directional_accuracy(a, p),
                                     "n_test": int(pd.notna(a).sum())})
                all_daily.append(
                    daily_prediction_stats(fold.test, preds, HORIZONS, name, sym).assign(fold=fold.index)
                )
                log(f"  {sym} fold {fold.index}/{len(folds)-1} {name}: {time.time()-t0:.1f}s")
            pd.concat(all_daily, ignore_index=True).to_csv(DAILY_PATH, index=False)

    daily = pd.concat(all_daily, ignore_index=True)
    fold_results = pd.DataFrame(all_fold)

    log("\n=== Fold-level pooled DA (context) ===")
    for tag, fset in (("select 0-4", range(5)), ("judge 5-8", range(5, 9)), ("all", range(20))):
        sub = fold_results[fold_results["fold"].isin(list(fset))]
        if len(sub):
            p = pool_fold_metrics(sub, group_cols=["symbol", "variant"]).sort_values(["symbol", "variant"])
            log(f"\n[{tag}]\n" + p[["symbol", "variant", "directional_accuracy", "mae"]].to_string(index=False))

    log("\n=== Day-block bootstrap: reg_strong vs default ===")
    summary = []
    scopes = [(s, [s]) for s in symbols] + [("POOLED", symbols)]
    for scope_name, syms in scopes:
        for tag, fset in (("select 0-4", range(5)), ("judge 5-8", range(5, 9)), ("all", range(20))):
            d = daily[daily["symbol"].isin(syms) & daily["fold"].isin(list(fset))]
            if d.empty:
                continue
            b = directional_block_bootstrap(d, "default", "reg_strong", horizons=HORIZONS, n_boot=10000)
            row = b[b["horizon"] == "pooled"].iloc[0]
            summary.append({"scope": scope_name, "folds": tag, "delta_da_pp": row["delta_da_pp"],
                            "ci_lo_pp": row["ci_lo_pp"], "ci_hi_pp": row["ci_hi_pp"],
                            "p_gt_0": row["p_delta_gt_0"], "delta_mae_pct": row["delta_mae_pct"]})
    sdf = pd.DataFrame(summary)
    sdf.to_csv(SUMMARY_PATH, index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        log("\n" + sdf.to_string(index=False))

    judge = sdf[sdf["folds"] == "judge 5-8"]
    passed = judge[(judge["scope"].isin(symbols)) & (judge["ci_lo_pp"] > 0)]
    pooled_ok = judge[(judge["scope"] == "POOLED") & (judge["ci_lo_pp"] > 0)].shape[0] == 1
    log(f"\n>>> held-out symbols with CI>0: {list(passed['scope'])} ({len(passed)}/{len(symbols)}); "
        f"pooled held-out CI>0: {pooled_ok}. "
        f"Verdict: {'reg_strong GENERALISES' if (len(passed) >= 2 and pooled_ok) else 'does NOT generalise cleanly'}")
    log(f"\nTotal {(time.time()-run_t0)/60:.1f}min")
    fh.close()


if __name__ == "__main__":
    main()
