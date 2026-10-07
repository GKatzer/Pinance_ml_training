"""Accuracy-program check #2, escalation step: does adding funding-rate
features to the confidence-corridor quantile models improve pooled pinball
loss / calibration over technical features alone?

Why quantile and not the point regressor: the full-history marginal screen
(scripts/screen_funding_signal.py) found NO consistent correctly-signed
directional relationship between funding-z and 30-60min *signed* return on
any of the 4 symbols -- only an 8h, BTC-only whiff, below this harness's
resolution. It DID find a strong, consistent |funding-z| -> |return|
relationship on every symbol and horizon (panel Newey-West t +5..+7,
funding-period-level p 1e-4..1e-12, top vs bottom |z| quartile ~+8-13%
larger |return|). That's a magnitude signal, so the live hypothesis is a
corridor-width one, tested exactly the way measure_quantile_gain.py tests
the Level-1/2 magnitude features -- same walk-forward harness, same
pre-registered bar.

Open question the screen can't answer and this run can: is |funding-z|
just a proxy for realized volatility the model already has (ret_std_*,
atr, bb_width)? Both variants carry those, so a null here means "no
*incremental* corridor value", not "no relationship".

Candidate feature set (as-of merged onto each candle ts, all causal --
funding_time is the settlement instant, rate known from then on):
  f_rate  last realized funding rate
  f_z     (f_rate - trailing-30d mean) / trailing-30d std
  f_absz  |f_z|

Pre-registered pass bar (identical to measure_quantile_gain.py, written
before this run): candidate must beat technical_only on POOLED
n_test-weighted mean pinball loss across 12 horizons x 3 quantiles by
>= PINBALL_IMPROVEMENT_BAR relative, AND the per-fold paired delta must be
significant and directionally consistent (ttest_1samp p < SIGNIFICANCE_ALPHA,
mean delta < 0), AND alpha=0.1/0.9 coverage must stay within
+/-COVERAGE_TOLERANCE of nominal in a majority of folds. Falling short
closes the funding-corridor hypothesis -- not a cue to try more funding
transforms or a bigger model.

Usage: python scripts/measure_funding_feature_gain.py [SYMBOL]   (default BTCUSDT)
Requires reports/funding_cache/{symbol}.parquet (scripts/fetch_funding_rates.py).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from scipy import stats

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import QUANTILE_GROUP_COLS, QUANTILE_METRIC_COLS, pool_fold_metrics
from pinance_ml.metrics import coverage, pinball_loss
from pinance_ml.models.lightgbm_model import DEFAULT_QUANTILES, predict_quantile_horizons, train_quantile_models
from pinance_ml.splits import walk_forward_folds
from pinance_ml.tracking import log_research_run

PINBALL_IMPROVEMENT_BAR = 0.01  # 1% pooled reduction, relative
COVERAGE_TOLERANCE = 0.03
SIGNIFICANCE_ALPHA = 0.05

CANDIDATE = "with_funding"
FUNDING_FEATURE_COLUMNS = ["f_rate", "f_z", "f_absz"]
Z_WINDOW = "30D"
Z_MIN_PERIODS = 30
CACHE_DIR = Path("reports/funding_cache")

FOLDS_PATH = Path("reports/funding_gain_folds.csv")
SUMMARY_PATH = Path("reports/funding_gain_summary.csv")
LOG_PATH = Path("reports/funding_gain.log")


def make_logger(log_file):
    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        print(line, file=log_file, flush=True)

    return log


def attach_funding_features(dataset: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """dataset + [f_rate, f_z, f_absz], as-of merged (backward) on ts.

    f_z is computed on the funding series itself (trailing 30d, current
    settlement included since its rate is known) then carried onto candles;
    rows before the symbol's first settlement get NaN, which LightGBM
    routes natively like any other missing feature.
    """
    f = pd.read_parquet(CACHE_DIR / f"{symbol}.parquet").sort_values("funding_time").reset_index(drop=True)
    s = f.set_index("funding_time")["funding_rate"]
    roll = s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS)
    f["f_rate"] = f["funding_rate"].to_numpy()
    f["f_z"] = ((s - roll.mean()) / roll.std()).to_numpy()

    merged = pd.merge_asof(
        dataset.sort_values("ts"),
        f[["funding_time", "f_rate", "f_z"]].sort_values("funding_time"),
        left_on="ts", right_on="funding_time", direction="backward",
    )
    merged["f_absz"] = merged["f_z"].abs()
    return merged.drop(columns=["funding_time"])


def fold_variant_rows(symbol: str, variant: str, fold, feat_cols: list[str]) -> list[dict]:
    models = train_quantile_models(fold.train, feat_cols, horizons=HORIZONS, quantiles=DEFAULT_QUANTILES)
    preds = predict_quantile_horizons(models, fold.test, feat_cols)

    rows = []
    for h in HORIZONS:
        actual = fold.test[f"r_{h}"].to_numpy()
        for q in DEFAULT_QUANTILES:
            predicted = preds[f"r_{h}_q{q}_pred"].to_numpy()
            rows.append(
                {
                    "symbol": symbol,
                    "variant": variant,
                    "fold": fold.index,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "horizon": h,
                    "quantile": q,
                    "pinball_loss": pinball_loss(actual, predicted, q),
                    "coverage": coverage(actual, predicted),
                    "n_train": len(fold.train),
                    "n_test": int(pd.notna(actual).sum()),
                }
            )
    return rows


def fold_level_overall_pinball(fold_results: pd.DataFrame) -> pd.DataFrame:
    scored = fold_results.copy()
    scored["weighted"] = scored["pinball_loss"] * scored["n_test"]
    grouped = scored.groupby(["variant", "fold"]).agg(
        weighted_sum=("weighted", "sum"), n_test_total=("n_test", "sum")
    )
    grouped["overall_pinball_loss"] = grouped["weighted_sum"] / grouped["n_test_total"]
    return grouped.reset_index()[["variant", "fold", "overall_pinball_loss"]]


def paired_significance(overall: pd.DataFrame) -> dict:
    wide = overall.pivot(index="fold", columns="variant", values="overall_pinball_loss")
    deltas = (wide[CANDIDATE] - wide["technical_only"]).to_numpy()
    n = len(deltas)
    t_stat, p_value = stats.ttest_1samp(deltas, popmean=0.0)

    sd = deltas.std(ddof=1)
    t_alpha = stats.t.ppf(1 - SIGNIFICANCE_ALPHA / 2, df=n - 1)
    t_power = stats.t.ppf(0.8, df=n - 1)
    mde = (t_alpha + t_power) * sd / (n**0.5)

    return {
        "n_folds": n,
        "mean_delta": float(deltas.mean()),
        "std_delta": float(sd),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "folds_favoring_candidate": int((deltas < 0).sum()),
        "minimum_detectable_effect_80pct_power": float(mde),
    }


def coverage_calibration_check(fold_results: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for q in (0.1, 0.9):
        sub = fold_results[(fold_results["variant"] == variant) & (fold_results["quantile"] == q)]
        per_fold = sub.groupby("fold").apply(
            lambda g: pd.Series(
                {"coverage": (g["coverage"] * g["n_test"]).sum() / g["n_test"].sum(), "n_test": g["n_test"].sum()}
            ),
            include_groups=False,
        )
        within_tol = (per_fold["coverage"] - q).abs() <= COVERAGE_TOLERANCE
        rows.append(
            {
                "variant": variant,
                "quantile": q,
                "n_folds": len(per_fold),
                "folds_within_tolerance": int(within_tol.sum()),
                "mean_coverage": float(per_fold["coverage"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_PATH, "a", encoding="utf-8")
    log = make_logger(log_file)

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    run_t0 = time.time()

    log(f"{symbol}: loading candles")
    candles = load_candles(symbol)
    btc_candles = load_candles("BTCUSDT") if symbol != "BTCUSDT" else None
    dataset = build_dataset(candles, btc_candles=btc_candles)
    base_feat_cols = feature_columns(dataset)

    dataset = attach_funding_features(dataset, symbol)
    cov = dataset["f_z"].notna().mean()
    log(f"{symbol}: {len(dataset)} rows, funding-feature coverage {cov:.1%} "
        f"(f_z range [{dataset['f_z'].min():.2f}, {dataset['f_z'].max():.2f}])")

    variants = {
        "technical_only": base_feat_cols,
        CANDIDATE: base_feat_cols + FUNDING_FEATURE_COLUMNS,
    }

    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    total_fold_variants = len(folds) * len(variants)
    log(f"{symbol}: {len(folds)} folds x {len(variants)} variants x {len(HORIZONS)} horizons x "
        f"{len(DEFAULT_QUANTILES)} quantiles = {total_fold_variants} fold-variants to train+score")

    all_rows = []
    done = 0
    for fold in folds:
        for variant, feat_cols in variants.items():
            t0 = time.time()
            all_rows.extend(fold_variant_rows(symbol, variant, fold, feat_cols))
            done += 1
            eta_min = (time.time() - run_t0) / done * (total_fold_variants - done) / 60
            log(f"  [fold {fold.index}/{len(folds) - 1}] {variant}: "
                f"{len(HORIZONS) * len(DEFAULT_QUANTILES)} models in {time.time() - t0:.1f}s "
                f"({done}/{total_fold_variants}, ETA {eta_min:.1f}min)")
            pd.DataFrame(all_rows).to_csv(FOLDS_PATH, index=False)

    fold_results = pd.DataFrame(all_rows)

    log("\n=== BASELINE: technical_only, pooled across folds ===")
    baseline_pooled = pool_fold_metrics(
        fold_results[fold_results["variant"] == "technical_only"],
        metric_cols=QUANTILE_METRIC_COLS, group_cols=QUANTILE_GROUP_COLS,
    )
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 160):
        log("\n" + baseline_pooled.to_string(index=False))

    log(f"\n=== HYPOTHESIS: {CANDIDATE} vs technical_only, pooled per (horizon, quantile) ===")
    candidate_pooled = pool_fold_metrics(
        fold_results[fold_results["variant"] == CANDIDATE],
        metric_cols=QUANTILE_METRIC_COLS, group_cols=QUANTILE_GROUP_COLS,
    )
    comparison = baseline_pooled.set_index(["symbol", "horizon", "quantile"])[["pinball_loss", "coverage"]].join(
        candidate_pooled.set_index(["symbol", "horizon", "quantile"])[["pinball_loss", "coverage"]],
        lsuffix="_technical_only", rsuffix=f"_{CANDIDATE}",
    )
    comparison["pinball_loss_delta"] = (
        comparison[f"pinball_loss_{CANDIDATE}"] - comparison["pinball_loss_technical_only"]
    )
    comparison = comparison.reset_index()
    comparison.to_csv(SUMMARY_PATH, index=False)
    with pd.option_context("display.float_format", "{:.6f}".format, "display.width", 200, "display.max_rows", 40):
        log("\n" + comparison.to_string(index=False))

    log("\n=== Pre-registered pass-bar check ===")
    overall = fold_level_overall_pinball(fold_results)
    sig = paired_significance(overall)
    overall_technical = overall[overall["variant"] == "technical_only"]["overall_pinball_loss"].mean()
    relative_improvement = -sig["mean_delta"] / overall_technical
    log(f"Overall pooled pinball loss -- technical_only: {overall_technical:.6f}")
    log(f"Mean per-fold delta (candidate - technical_only): {sig['mean_delta']:+.6f} "
        f"({relative_improvement * 100:+.2f}% relative improvement if positive)")
    log(f"Paired t-test across {sig['n_folds']} folds: t={sig['t_stat']:.3f}, p={sig['p_value']:.4f}, "
        f"folds favoring candidate: {sig['folds_favoring_candidate']}/{sig['n_folds']}")
    log(f"Minimum detectable effect at 80% power: {sig['minimum_detectable_effect_80pct_power']:.6f} absolute pinball loss")

    coverage_technical = coverage_calibration_check(fold_results, "technical_only")
    coverage_candidate = coverage_calibration_check(fold_results, CANDIDATE)
    log(f"\nCoverage calibration (folds within +/-{COVERAGE_TOLERANCE:.2f} of nominal alpha):")
    log("\n" + pd.concat([coverage_technical, coverage_candidate]).to_string(index=False))

    passes_pinball = relative_improvement >= PINBALL_IMPROVEMENT_BAR
    passes_significance = sig["p_value"] < SIGNIFICANCE_ALPHA and sig["mean_delta"] < 0
    passes_coverage = (coverage_candidate["folds_within_tolerance"] > coverage_candidate["n_folds"] / 2).all()
    verdict = "PASSES" if (passes_pinball and passes_significance and passes_coverage) else "DOES NOT PASS"
    log(f"\n>>> Pre-registered bar: pinball improvement>={PINBALL_IMPROVEMENT_BAR:.1%} [{passes_pinball}], "
        f"significant+directionally consistent [{passes_significance}], "
        f"majority-fold coverage within tolerance [{passes_coverage}] "
        f"=> candidate {verdict} the pre-registered bar.")

    log(f"\nTotal run time: {(time.time() - run_t0) / 60:.1f}min")

    log_research_run(
        __file__,
        run_name=f"{CANDIDATE}-{symbol}",
        params={
            "symbol": symbol,
            "candidate": CANDIDATE,
            "n_folds": sig["n_folds"],
            "pinball_improvement_bar": PINBALL_IMPROVEMENT_BAR,
            "significance_alpha": SIGNIFICANCE_ALPHA,
            "coverage_tolerance": COVERAGE_TOLERANCE,
        },
        metrics={
            "relative_improvement": float(relative_improvement),
            "mean_fold_delta": float(sig["mean_delta"]),
            "t_stat": float(sig["t_stat"]),
            "p_value": float(sig["p_value"]),
            "overall_pinball_technical": float(overall_technical),
            "mde_80pct_power": float(sig["minimum_detectable_effect_80pct_power"]),
            "folds_favoring_candidate": float(sig["folds_favoring_candidate"]),
        },
        tags={
            "verdict": verdict,
            "passes_pinball": str(passes_pinball),
            "passes_significance": str(passes_significance),
            "passes_coverage": str(passes_coverage),
        },
        report_paths=sorted(Path("reports").glob("funding_gain*")),
    )

    log_file.close()


if __name__ == "__main__":
    main()
