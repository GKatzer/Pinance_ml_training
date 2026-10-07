"""Confidence-corridor quantile modeling (quantile_modeling_task.md,
README: "Доверительный коридор: квантильная регрессия, objective='quantile'
при α = 0.1 / 0.5 / 0.9 -- 36 моделей"). Two things in one script, since
they share the identical walk-forward harness and the second only makes
sense read against the first:

  1. BASELINE (task breakdown step 4): technical_only -- is the alpha=0.1/
     0.9 corridor even reasonably calibrated on technical features alone,
     across all 12 horizons x 3 quantiles, walk-forward validated on
     BTCUSDT? Read this number first, on its own, before looking at the
     delta below -- a candidate "win" over a poorly-calibrated baseline
     means something different than one over a well-calibrated baseline.

  2. HYPOTHESIS (task breakdown step 5): with_event_and_magnitude -- does
     adding event_llm_type_60m (Level 2) and max_magnitude_60m (Level 1,
     already the production FinBERT-based magnitude proxy) improve pinball
     loss / calibration over the baseline? These two features specifically
     (not the full Level 1 NEWS_FEATURE_COLUMNS set -- sentiment_weighted/
     news_intensity/sentiment_dispersion/keyword event flags are a
     *direction*-oriented feature set already measured and found not to
     help at this horizon, see the project notes)
     are the ones scripts/screen_quantile_magnitude_signal.py confirmed
     show a real, non-degenerate relationship with |r_12| on this exact
     BTC history.

Pre-registered pass bar (written before this script's first run, per the
task brief's methodology rule #3 -- decide the bar before seeing the
number, not after):

  The candidate must beat technical_only on POOLED (n_test-weighted) mean
  pinball loss across all 12 horizons x 3 quantiles by >= PINBALL_IMPROVEMENT_BAR
  (relative), AND the per-fold paired delta must be directionally
  consistent (scipy.stats.ttest_1samp p < SIGNIFICANCE_ALPHA), AND
  calibration coverage for the alpha=0.1/0.9 models must stay within
  +/-COVERAGE_TOLERANCE of nominal in a majority of folds. A model that
  "wins" on pinball loss only by becoming overconfident (narrower
  corridor, worse coverage) is not a real win for a confidence corridor's
  actual job -- pinball loss alone doesn't catch that, same reason
  metrics.coverage exists alongside metrics.pinball_loss.

  Falling short of this bar closes the hypothesis with the same rigor as
  the news-direction investigation (the project notes) -- not a
  cue to try more quantile levels, richer event_type features, or a bigger
  model (methodology rule #5).

Same HORIZONS (1..12, 5-60min)/PURGE_ROWS/walk-forward harness as the
point regressor (train_lightgbm.py) -- this is about the regressor's
*output shape*, not a new horizon or a new asset (scope boundary in the
task brief).

Usage: python scripts/measure_quantile_gain.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from scipy import stats

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import load_candles
from pinance_ml.data.news_db import load_news
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import QUANTILE_GROUP_COLS, QUANTILE_METRIC_COLS, pool_fold_metrics
from pinance_ml.metrics import coverage, pinball_loss
from pinance_ml.models.lightgbm_model import DEFAULT_QUANTILES, predict_quantile_horizons, train_quantile_models
from pinance_ml.news.decay import LLM_EVENT_FEATURE_COLUMNS, base_asset, compute_llm_event_features, compute_news_features
from pinance_ml.splits import walk_forward_folds
from pinance_ml.tracking import log_research_run

PINBALL_IMPROVEMENT_BAR = 0.01  # 1% pooled reduction, relative
COVERAGE_TOLERANCE = 0.03
SIGNIFICANCE_ALPHA = 0.05

FOLDS_PATH = Path("reports/quantile_gain_folds.csv")
SUMMARY_PATH = Path("reports/quantile_gain_summary.csv")
LOG_PATH = Path("reports/quantile_gain.log")


def make_logger(log_file):
    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        print(line, file=log_file, flush=True)

    return log


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
    """One n_test-weighted mean pinball loss per (variant, fold), collapsed
    across every horizon x quantile cell -- the single number the
    pre-registered pass bar's paired ttest runs on. Weighting matters here
    exactly as it does in pool_fold_metrics (n_test differs by horizon
    within a fold), just collapsed to fold-level instead of pooled across
    folds."""
    scored = fold_results.copy()
    scored["weighted"] = scored["pinball_loss"] * scored["n_test"]
    grouped = scored.groupby(["variant", "fold"]).agg(
        weighted_sum=("weighted", "sum"), n_test_total=("n_test", "sum")
    )
    grouped["overall_pinball_loss"] = grouped["weighted_sum"] / grouped["n_test_total"]
    return grouped.reset_index()[["variant", "fold", "overall_pinball_loss"]]


def paired_significance(overall: pd.DataFrame) -> dict:
    wide = overall.pivot(index="fold", columns="variant", values="overall_pinball_loss")
    deltas = (wide["with_event_and_magnitude"] - wide["technical_only"]).to_numpy()
    n = len(deltas)
    t_stat, p_value = stats.ttest_1samp(deltas, popmean=0.0)

    # Minimum detectable effect at 80% power for this fold count/variance,
    # so a null result here is interpreted as "underpowered" vs "no
    # effect" correctly (methodology rule #4) -- MDE = (t_crit(alpha/2) +
    # t_crit(power)) * sd / sqrt(n) for a one-sample paired t-test.
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
    """For alpha in {0.1, 0.9}, what fraction of folds land within
    +/-COVERAGE_TOLERANCE of nominal, weighting each fold's coverage by
    n_test the same way everything else here is weighted (a fold's
    coverage number is itself a mean over its test rows)."""
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

    symbol = "BTCUSDT"
    run_t0 = time.time()

    log(f"{symbol}: loading candles")
    candles = load_candles(symbol)
    dataset = build_dataset(candles)
    base_feat_cols = feature_columns(dataset)

    asset = base_asset(symbol)
    news = load_news(asset=asset, include_llm=True)
    n_classified = int(news["llm_event_type"].notna().sum())
    log(f"{asset}: {len(news)} news items, {n_classified} with Level 2 event_type ({n_classified / max(len(news), 1):.1%})")

    news_features = compute_news_features(news, dataset["ts"])
    llm_features = compute_llm_event_features(news, dataset["ts"])
    dataset = pd.concat([dataset.reset_index(drop=True), news_features, llm_features], axis=1)

    variants = {
        "technical_only": base_feat_cols,
        "with_event_and_magnitude": base_feat_cols + ["max_magnitude_60m"] + LLM_EVENT_FEATURE_COLUMNS,
    }

    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    total_fold_variants = len(folds) * len(variants)
    log(f"{symbol}: {len(folds)} walk-forward folds x {len(variants)} variants x "
        f"{len(HORIZONS)} horizons x {len(DEFAULT_QUANTILES)} quantiles = {total_fold_variants} fold-variants to train+score")

    all_rows = []
    done = 0
    for fold in folds:
        for variant, feat_cols in variants.items():
            t0 = time.time()
            rows = fold_variant_rows(symbol, variant, fold, feat_cols)
            all_rows.extend(rows)
            done += 1
            elapsed = time.time() - run_t0
            eta_min = elapsed / done * (total_fold_variants - done) / 60
            log(f"  [fold {fold.index}/{len(folds) - 1}] {variant}: "
                f"{len(HORIZONS) * len(DEFAULT_QUANTILES)} models trained+scored in {time.time() - t0:.1f}s "
                f"({done}/{total_fold_variants}, ETA {eta_min:.1f}min)")
            pd.DataFrame(all_rows).to_csv(FOLDS_PATH, index=False)

    fold_results = pd.DataFrame(all_rows)

    log("\n=== BASELINE: technical_only, pooled across folds ===")
    baseline_pooled = pool_fold_metrics(
        fold_results[fold_results["variant"] == "technical_only"],
        metric_cols=QUANTILE_METRIC_COLS,
        group_cols=QUANTILE_GROUP_COLS,
    )
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 160):
        log("\n" + baseline_pooled.to_string(index=False))

    log("\n=== HYPOTHESIS: with_event_and_magnitude vs technical_only, pooled per (horizon, quantile) ===")
    candidate_pooled = pool_fold_metrics(
        fold_results[fold_results["variant"] == "with_event_and_magnitude"],
        metric_cols=QUANTILE_METRIC_COLS,
        group_cols=QUANTILE_GROUP_COLS,
    )
    comparison = baseline_pooled.set_index(["symbol", "horizon", "quantile"])[["pinball_loss", "coverage"]].join(
        candidate_pooled.set_index(["symbol", "horizon", "quantile"])[["pinball_loss", "coverage"]],
        lsuffix="_technical_only",
        rsuffix="_with_event_and_magnitude",
    )
    comparison["pinball_loss_delta"] = (
        comparison["pinball_loss_with_event_and_magnitude"] - comparison["pinball_loss_technical_only"]
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
        f"({relative_improvement * 100:+.2f}% relative improvement if negative)")
    log(f"Paired t-test across {sig['n_folds']} folds: t={sig['t_stat']:.3f}, p={sig['p_value']:.4f}, "
        f"folds favoring candidate: {sig['folds_favoring_candidate']}/{sig['n_folds']}")
    log(f"Minimum detectable effect at 80% power given this fold count/variance: "
        f"{sig['minimum_detectable_effect_80pct_power']:.6f} absolute pinball loss")

    coverage_technical = coverage_calibration_check(fold_results, "technical_only")
    coverage_candidate = coverage_calibration_check(fold_results, "with_event_and_magnitude")
    log("\nCoverage calibration (folds within +/-{:.2f} of nominal alpha):".format(COVERAGE_TOLERANCE))
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
        run_name=f"quantile-gain-{symbol}",
        params={
            "symbol": symbol,
            "n_folds": sig["n_folds"],
            "pinball_improvement_bar": PINBALL_IMPROVEMENT_BAR,
            "significance_alpha": SIGNIFICANCE_ALPHA,
            "coverage_tolerance": COVERAGE_TOLERANCE,
            "variants": ",".join(variants),
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
        report_paths=sorted(Path("reports").glob("quantile_gain*")),
    )

    log_file.close()


if __name__ == "__main__":
    main()
