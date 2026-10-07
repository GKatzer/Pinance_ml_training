import numpy as np
import pandas as pd

METRIC_COLS = ["mae", "directional_accuracy"]

# Quantile-model fold results (pinball loss + calibration coverage per
# (symbol, horizon, quantile, fold)) need their own group key alongside
# metric_cols below -- one pooled number per quantile level, not per
# horizon alone, since alpha=0.1/0.5/0.9 are three independently-fit
# models with their own pinball loss and coverage.
QUANTILE_METRIC_COLS = ["pinball_loss", "coverage"]
QUANTILE_GROUP_COLS = ["symbol", "horizon", "quantile"]


def pool_fold_metrics(
    fold_results: pd.DataFrame,
    metric_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pool per-(group, fold) metrics into one number per group, weighted by
    each fold's own n_test.

    Every metric this pools (mae/directional_accuracy, or pinball_loss/
    coverage for quantile models) is itself a mean over a fold's test rows,
    so weighting each fold's value by its n_test and averaging is
    algebraically identical to recomputing the metric on every fold's test
    rows concatenated — without needing to re-run anything over pooled data.
    Shared by the naive, LightGBM point, and LightGBM quantile walk-forward
    evaluations so all three report numbers the same way.

    metric_cols/group_cols default to the original point-regression shape
    (METRIC_COLS grouped by symbol+horizon); pass metric_cols=
    QUANTILE_METRIC_COLS, group_cols=QUANTILE_GROUP_COLS for quantile-model
    fold results instead.
    """
    metric_cols = list(metric_cols) if metric_cols is not None else list(METRIC_COLS)
    group_cols = list(group_cols) if group_cols is not None else ["symbol", "horizon"]

    scored = fold_results[fold_results["n_test"] > 0].copy()
    for col in metric_cols:
        scored[f"{col}_weighted"] = scored[col] * scored["n_test"]

    grouped = scored.groupby(group_cols).agg(
        **{f"{col}_weighted": (f"{col}_weighted", "sum") for col in metric_cols},
        n_test_total=("n_test", "sum"),
        n_folds=("fold", "nunique"),
    )
    for col in metric_cols:
        grouped[col] = grouped[f"{col}_weighted"] / grouped["n_test_total"]
    return grouped[metric_cols + ["n_folds", "n_test_total"]].reset_index()


# --- Directional-accuracy day-block bootstrap (accuracy-program "point 0") ---
# The fold-level paired t-test over ~9 walk-forward folds cannot resolve a
# directional-accuracy difference under ~1.5pp (screen_intermediate_horizons.py).
# DA is where the point models actually carry signal (~53% vs a ~50% coin),
# so a candidate that moves DA by 0.3-0.5pp is worth catching. These two
# helpers keep the SAME walk-forward predictions but resample calendar-day
# blocks instead of whole folds: ~2900 days / 7-day blocks ~= 400 near-
# independent units instead of 9, at the cost of assuming a 7-day block is
# long enough to carry the return autocorrelation (overlapping r_h targets
# + slow features). MAE deltas ride along as a secondary readout -- reported,
# not optimized, since #1-#3 showed pooled MAE sits at the naive floor.

DAILY_STAT_COLS = ["n", "n_dir_correct", "sum_abs_err", "sum_abs_actual"]


def daily_prediction_stats(
    test: pd.DataFrame, preds: pd.DataFrame, horizons: list[int], variant: str, symbol: str
) -> pd.DataFrame:
    """Per-(symbol, variant, horizon, day) directional + error sums for one fold.

    Lossless for pooled DA and MAE (both are ratios of these sums). The
    calendar day is the aggregation atom because it already holds ~288
    heavily autocorrelated 5-minute predictions -- the day-block bootstrap
    below resamples days, not rows. `test` needs `ts` + `r_{h}`; `preds`
    needs `r_{h}_pred` positionally aligned to `test` (predict_horizons output).
    """
    day = test["ts"].dt.floor("D").to_numpy()
    frames = []
    for h in horizons:
        a = test[f"r_{h}"].to_numpy(dtype=float)
        p = preds[f"r_{h}_pred"].to_numpy(dtype=float)
        m = ~np.isnan(a) & ~np.isnan(p)
        g = (
            pd.DataFrame(
                {
                    "day": day[m],
                    "sum_abs_err": np.abs(a[m] - p[m]),
                    "sum_abs_actual": np.abs(a[m]),
                    "n_dir_correct": (np.sign(a[m]) == np.sign(p[m])).astype(int),
                }
            )
            .groupby("day", as_index=False)
            .agg(
                n=("n_dir_correct", "size"),
                n_dir_correct=("n_dir_correct", "sum"),
                sum_abs_err=("sum_abs_err", "sum"),
                sum_abs_actual=("sum_abs_actual", "sum"),
            )
        )
        g.insert(0, "symbol", symbol)
        g.insert(1, "variant", variant)
        g.insert(2, "horizon", h)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def directional_block_bootstrap(
    daily: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    horizons: list[int] | None = None,
    block_days: int = 7,
    n_boot: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Moving-block bootstrap over calendar days: pooled DA per variant and
    the paired delta DA(b) - DA(a), per horizon and pooled across horizons.

    Both variants are gathered on the SAME resampled day blocks each draw,
    so the delta CI reflects only the between-variant difference, not the
    day-to-day DA level both share. Point estimates are the full-sample
    pooled numbers; CI/p come from the bootstrap distribution of the delta.
    Returns one row per horizon plus a horizon="pooled" row.
    """
    d = daily[daily["variant"].isin([variant_a, variant_b])].copy()
    horizons = horizons if horizons is not None else sorted(d["horizon"].unique())
    days = np.sort(d["day"].unique())
    D = len(days)
    blk = min(block_days, D)
    day_ix = {v: i for i, v in enumerate(days)}
    rng = np.random.default_rng(seed)

    def arrays(variant: str, hs: list[int]) -> tuple[np.ndarray, ...]:
        sub = d[(d["variant"] == variant) & (d["horizon"].isin(hs))]
        agg = sub.groupby("day")[DAILY_STAT_COLS].sum()
        n = np.zeros(D); c = np.zeros(D); e = np.zeros(D); s = np.zeros(D)
        idx = [day_ix[v] for v in agg.index]
        n[idx] = agg["n"].to_numpy()
        c[idx] = agg["n_dir_correct"].to_numpy()
        e[idx] = agg["sum_abs_err"].to_numpy()
        s[idx] = agg["sum_abs_actual"].to_numpy()
        return n, c, e, s

    n_blocks = int(np.ceil(D / blk))
    starts_max = D - blk + 1
    offsets = np.arange(blk)

    rows = []
    for label, hs in [(h, [h]) for h in horizons] + [("pooled", list(horizons))]:
        na, ca, ea, sa = arrays(variant_a, hs)
        nb, cb, eb, sb = arrays(variant_b, hs)
        da_a = ca.sum() / na.sum()
        da_b = cb.sum() / nb.sum()
        mae_a = ea.sum() / na.sum()
        mae_b = eb.sum() / nb.sum()

        deltas = np.empty(n_boot)
        for k in range(n_boot):
            starts = rng.integers(0, starts_max, size=n_blocks)
            ix = (starts[:, None] + offsets).ravel()[:D]
            boot_a = ca[ix].sum() / na[ix].sum()
            boot_b = cb[ix].sum() / nb[ix].sum()
            deltas[k] = boot_b - boot_a

        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append(
            {
                "horizon": label,
                "n_days": D,
                "da_a": da_a,
                "da_b": da_b,
                "delta_da_pp": (da_b - da_a) * 100,
                "ci_lo_pp": lo * 100,
                "ci_hi_pp": hi * 100,
                "p_delta_gt_0": float((deltas > 0).mean()),
                "mae_a": mae_a,
                "mae_b": mae_b,
                "delta_mae_pct": (mae_b - mae_a) / mae_a * 100,
            }
        )
    return pd.DataFrame(rows)
