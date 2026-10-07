"""GridSearch over the LightGBM point model's hyperparameters.

Follow-up to tune_lightgbm_directional.py, which tried six hand-picked
configs. Same pre-registered protocol, wider net: folds 0-4 SELECT, folds
5-8 JUDGE, and a config only counts as a real improvement if its pooled
Delta-DA vs default has a 95% CI (day-block bootstrap) excluding 0 on the
positive side on the held-out folds, having also been >= default on the
selection folds.

Cost is the constraint, not the grid: one 12-horizon fit on the largest BTC
fold takes ~8min, so a full 9-fold pass is ~35-40min PER config. The search
is therefore two stages:

  Stage 1 (screen)   full grid x folds 0-4 x horizons {1, 6, 12}. Every
                     structural cell is trained once at MAX_TREES and scored
                     at each of TREE_STEPS via predict(num_iteration=k) --
                     the first k trees of a fixed-learning-rate fit are
                     exactly what an n_estimators=k fit would produce, so
                     the tree-count axis is free. Ranked by pooled DA.
  Stage 2 (confirm)  top-K distinct cells from stage 1, plus `default` and
                     `reg_strong` (the tune script's reference points), on
                     ALL 12 horizons x ALL 9 folds; select/judge/bootstrap
                     exactly as in tune_lightgbm_directional.py.

Stage 1 picks the max over ~140 candidates on 5 folds, so its winner is
biased upward (winner's curse) -- that is what the held-out folds in stage 2
are for. Stage 2 looks at K+1 configs on the judge folds; treat a single
barely-positive CI among them with suspicion.

Both stages checkpoint to --out-dir after every fit and resume on restart.

Usage: python scripts/gridsearch_lightgbm.py [SYMBOL] [--screen-only] [--top-k 3]
"""

import argparse
import itertools
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
from tune_lightgbm_directional import CONFIGS, JUDGE_FOLDS, SELECT_FOLDS

NUM_LEAVES = [7, 15, 31, 63]
MIN_CHILD_SAMPLES = [100, 300, 500]
# Feature and row subsampling move together on purpose: three legible levels
# (none / mid / strong, mirroring reg_mild and reg_strong) instead of a 3x3
# sub-grid whose corners nobody would ship. "none" is DEFAULT_PARAMS' own.
SUBSAMPLE = {
    "none": {},
    "mid": dict(feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1),
    "strong": dict(feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1),
}
# learning_rate stays at DEFAULT_PARAMS' 0.05: with the tree-count axis free,
# lr x n_estimators is already swept along its main diagonal.
MAX_TREES = 400
TREE_STEPS = [100, 200, 300, 400]

SCREEN_HORIZONS = [1, 6, 12]  # short / mid / long end of the 1..12 range
SCREEN_FOLDS = list(SELECT_FOLDS)


def cell_name(nl: int, mcs: int, sub: str) -> str:
    return f"nl{nl}_mcs{mcs}_{sub}"


def cell_params(nl: int, mcs: int, sub: str) -> dict:
    return {"num_leaves": nl, "min_child_samples": mcs, **SUBSAMPLE[sub]}


def make_logger(path: Path):
    fh = open(path, "a", encoding="utf-8")

    def log(m: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True)
        print(line, file=fh, flush=True)

    return log


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_screen(folds, feat_cols, symbol: str, out_dir: Path, log) -> pd.DataFrame:
    path = out_dir / "screen_folds.csv"
    df = read_csv_or_empty(path)
    rows = df.to_dict("records")
    done = set(zip(df["fold"], df["cell"])) if len(df) else set()

    cells = list(itertools.product(NUM_LEAVES, MIN_CHILD_SAMPLES, SUBSAMPLE))
    total = len(SCREEN_FOLDS) * len(cells)
    log(f"screen: {len(cells)} cells x {len(SCREEN_FOLDS)} folds = {total} fits "
        f"({len(done)} already done), {len(cells) * len(TREE_STEPS)} candidates, "
        f"horizons {SCREEN_HORIZONS}, trees {TREE_STEPS}")

    t_start, fitted = time.time(), 0
    for fold in (folds[i] for i in SCREEN_FOLDS):
        x_test = fold.test[feat_cols]
        for nl, mcs, sub in cells:
            name = cell_name(nl, mcs, sub)
            if (fold.index, name) in done:
                continue
            t0 = time.time()
            params = {**cell_params(nl, mcs, sub), "n_estimators": MAX_TREES}
            models = train_horizon_models(fold.train, feat_cols, horizons=SCREEN_HORIZONS, params=params)
            for k in TREE_STEPS:
                for h in SCREEN_HORIZONS:
                    a = fold.test[f"r_{h}"].to_numpy()
                    p = models[h].predict(x_test, num_iteration=k)
                    rows.append({
                        "symbol": symbol, "cell": name, "num_leaves": nl, "min_child_samples": mcs,
                        "subsample": sub, "n_estimators": k, "fold": fold.index, "horizon": h,
                        "mae": mae(a, p), "directional_accuracy": directional_accuracy(a, p),
                        "n_test": int(pd.notna(a).sum()),
                    })
            pd.DataFrame(rows).to_csv(path, index=False)
            fitted += 1
            remaining = total - len(done) - fitted
            eta = (time.time() - t_start) / fitted * remaining / 60
            log(f"  [screen fold {fold.index}] {name}: {time.time()-t0:.0f}s "
                f"({len(done) + fitted}/{total}, ETA {eta:.0f}min)")
    return pd.DataFrame(rows)


def report_screen(df: pd.DataFrame, log) -> pd.DataFrame:
    """Rank candidates by pooled DA on the selection folds; return the ranked table."""
    pooled = pool_fold_metrics(df, group_cols=["cell", "num_leaves", "min_child_samples", "subsample", "n_estimators"])
    pooled = pooled.sort_values("directional_accuracy", ascending=False).reset_index(drop=True)
    pooled["da_pct"] = pooled["directional_accuracy"] * 100

    base = pooled[(pooled["cell"] == cell_name(31, 100, "none")) & (pooled["n_estimators"] == 300)].iloc[0]
    pooled["delta_vs_default_pp"] = pooled["da_pct"] - base["da_pct"]
    pooled["mae_vs_default_pct"] = (pooled["mae"] / base["mae"] - 1) * 100

    log(f"\n=== SCREEN: pooled DA on select folds {list(SCREEN_FOLDS)}, horizons {SCREEN_HORIZONS} ===")
    log(f"in-grid default (nl31_mcs100_none @300 trees): DA {base['da_pct']:.3f}%, MAE {base['mae']:.6f}, "
        f"rank {int(base.name) + 1}/{len(pooled)}")
    cols = ["cell", "n_estimators", "da_pct", "delta_vs_default_pp", "mae_vs_default_pct"]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        log("\nTop 15:\n" + pooled[cols].head(15).to_string(index=False))
        log("\nBottom 5:\n" + pooled[cols].tail(5).to_string(index=False))
        for axis in ("num_leaves", "min_child_samples", "subsample", "n_estimators"):
            marg = pooled.groupby(axis)["da_pct"].agg(["mean", "max", "min"])
            log(f"\nMarginal DA by {axis} (mean/max/min over all candidates):\n" + marg.to_string())
    spread = pooled["da_pct"].max() - pooled["da_pct"].min()
    log(f"\nDA spread across the whole grid: {spread:.3f}pp")
    return pooled


def pick_finalists(pooled: pd.DataFrame, top_k: int) -> dict[str, dict]:
    """Top-K DISTINCT cells (each at its best tree count): the ranked table
    lists adjacent tree counts of the same fit back to back, which would
    otherwise fill the shortlist with near-duplicates."""
    best_per_cell = pooled.drop_duplicates("cell").head(top_k)
    finalists = {}
    for _, r in best_per_cell.iterrows():
        params = cell_params(int(r["num_leaves"]), int(r["min_child_samples"]), r["subsample"])
        finalists[f"grid_{r['cell']}_k{int(r['n_estimators'])}"] = {**params, "n_estimators": int(r["n_estimators"])}
    return finalists


def run_confirm(folds, feat_cols, symbol: str, variants: dict[str, dict], out_dir: Path, log):
    folds_path, daily_path = out_dir / "confirm_folds.csv", out_dir / "confirm_daily.csv"
    fold_df, daily_df = read_csv_or_empty(folds_path), read_csv_or_empty(daily_path)
    fold_rows = fold_df.to_dict("records")
    daily_frames = [daily_df] if len(daily_df) else []
    done = set(zip(fold_df["variant"], fold_df["fold"])) if len(fold_df) else set()

    total = len(folds) * len(variants)
    log(f"\nconfirm: {len(variants)} variants x {len(folds)} folds x {len(HORIZONS)} horizons "
        f"({len(done)} fits already done): {list(variants)}")
    t_start, fitted = time.time(), 0
    for fold in folds:
        for name, params in variants.items():
            if (name, fold.index) in done:
                continue
            t0 = time.time()
            models = train_horizon_models(fold.train, feat_cols, params=params or None)
            preds = predict_horizons(models, fold.test, feat_cols)
            for h in HORIZONS:
                a, p = fold.test[f"r_{h}"].to_numpy(), preds[f"r_{h}_pred"].to_numpy()
                fold_rows.append({"symbol": symbol, "variant": name, "fold": fold.index, "horizon": h,
                                  "mae": mae(a, p), "directional_accuracy": directional_accuracy(a, p),
                                  "n_test": int(pd.notna(a).sum())})
            daily_frames.append(daily_prediction_stats(fold.test, preds, HORIZONS, name, symbol).assign(fold=fold.index))
            pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
            pd.concat(daily_frames, ignore_index=True).to_csv(daily_path, index=False)
            fitted += 1
            remaining = total - len(done) - fitted
            eta = (time.time() - t_start) / fitted * remaining / 60
            log(f"  [confirm fold {fold.index}/{len(folds)-1}] {name}: {time.time()-t0:.0f}s "
                f"({len(done) + fitted}/{total}, ETA {eta:.0f}min)")
    return pd.DataFrame(fold_rows), pd.concat(daily_frames, ignore_index=True)


def report_confirm(fold_results: pd.DataFrame, daily: pd.DataFrame, variants: dict[str, dict], log) -> None:
    fold_sets = (("select folds 0-4", SELECT_FOLDS), ("judge folds 5-8", JUDGE_FOLDS), ("all 9", range(9)))
    log("\n=== CONFIRM: pooled DA per variant (all 12 horizons) ===")
    for tag, fset in fold_sets:
        sub = fold_results[fold_results["fold"].isin(list(fset))]
        pooled = pool_fold_metrics(sub, group_cols=["symbol", "variant"]).sort_values("directional_accuracy", ascending=False)
        log(f"\n[{tag}]\n" + pooled[["variant", "directional_accuracy", "mae"]].to_string(index=False))

    challengers = [v for v in variants if v != "default"]
    results: dict[tuple[str, str], pd.Series] = {}
    log("\n=== Day-block bootstrap vs default (pooled Delta-DA) ===")
    for tag, fset in fold_sets:
        d = daily[daily["fold"].isin(list(fset))]
        rows = []
        for name in challengers:
            b = directional_block_bootstrap(d, "default", name, horizons=HORIZONS, n_boot=10000)
            p = b[b["horizon"] == "pooled"].iloc[0]
            results[(tag, name)] = p
            rows.append({"variant": name, "delta_da_pp": p["delta_da_pp"], "ci_lo_pp": p["ci_lo_pp"],
                         "ci_hi_pp": p["ci_hi_pp"], "p_gt_0": p["p_delta_gt_0"], "delta_mae_pct": p["delta_mae_pct"]})
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
            log(f"\n[{tag}]\n" + pd.DataFrame(rows).to_string(index=False))

    grid_variants = [v for v in challengers if v.startswith("grid_")]
    if grid_variants:
        log("\n=== Day-block bootstrap: grid finalists vs reg_strong (judge folds 5-8) ===")
        d = daily[daily["fold"].isin(list(JUDGE_FOLDS))]
        rows = []
        for name in grid_variants:
            p = directional_block_bootstrap(d, "reg_strong", name, horizons=HORIZONS, n_boot=10000)
            p = p[p["horizon"] == "pooled"].iloc[0]
            rows.append({"variant": name, "delta_da_pp": p["delta_da_pp"], "ci_lo_pp": p["ci_lo_pp"],
                         "ci_hi_pp": p["ci_hi_pp"], "delta_mae_pct": p["delta_mae_pct"]})
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
            log("\n" + pd.DataFrame(rows).to_string(index=False))

    log("\n=== Pre-registered verdict (judge CI lo > 0 AND select delta >= 0) ===")
    for name in challengers:
        judge, select = results[("judge folds 5-8", name)], results[("select folds 0-4", name)]
        ok = judge["ci_lo_pp"] > 0 and select["delta_da_pp"] >= 0
        log(f"  {name}: judge dDA {judge['delta_da_pp']:+.3f}pp CI [{judge['ci_lo_pp']:+.3f}, {judge['ci_hi_pp']:+.3f}], "
            f"select dDA {select['delta_da_pp']:+.3f}pp -> {'PASSES' if ok else 'does not pass'}")
    log(f"  ({len(challengers)} challengers judged on the same folds: one marginal pass among them is weak evidence)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbol", nargs="?", default="BTCUSDT")
    parser.add_argument("--top-k", type=int, default=3, help="distinct stage-1 cells promoted to stage 2")
    parser.add_argument("--screen-only", action="store_true", help="stop after stage 1")
    parser.add_argument("--out-dir", default="reports/gridsearch")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(out_dir / "gridsearch.log")
    run_t0 = time.time()

    log(f"{args.symbol}: building dataset")
    dataset = build_dataset(load_candles(args.symbol))
    feat_cols = feature_columns(dataset)
    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    log(f"{args.symbol}: {len(dataset)} rows, {len(feat_cols)} features, {len(folds)} folds")

    screen = run_screen(folds, feat_cols, args.symbol, out_dir, log)
    pooled = report_screen(screen, log)
    pooled.to_csv(out_dir / "screen_ranked.csv", index=False)
    if args.screen_only:
        log(f"\nscreen-only: done in {(time.time()-run_t0)/60:.1f}min")
        return

    variants = {"default": {}, "reg_strong": CONFIGS["reg_strong"], **pick_finalists(pooled, args.top_k)}
    fold_results, daily = run_confirm(folds, feat_cols, args.symbol, variants, out_dir, log)
    report_confirm(fold_results, daily, variants, log)
    log(f"\nSaved -> {out_dir}/. Total {(time.time()-run_t0)/60:.1f}min")


if __name__ == "__main__":
    main()
