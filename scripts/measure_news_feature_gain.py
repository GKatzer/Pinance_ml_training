"""Measures the LightGBM MAE/directional-accuracy delta from adding
time-decay news features (README status checklist: "замер прироста").

Runs the same walk-forward-with-purging harness as
scripts/train_lightgbm.py twice per fold -- once on technical features
only ("technical_only"), once with news/decay.py's time-decay features
added ("with_news") -- from the *same* train/test split each time, so the
only thing differing between the two runs is the feature set. Usage:

    python scripts/measure_news_feature_gain.py [SYMBOL ...]

With no arguments, runs against every symbol found in the `candles` table.
News features come from whatever `news_items` holds for that symbol's base
asset (news/decay.py::base_asset) -- an empty/thin table just means the
"with_news" run sees all-zero news columns, which LightGBM should learn to
ignore rather than crash on. Per README: this is a measured result to be
reported either way, not an assumption that news features help.

Progress is logged with flush=True to both stdout and reports/news_feature_gain.log
(appended across runs, since this process is long enough to outlive whatever
terminal launched it), and fold_results.csv is rewritten after every fold
(not just every symbol, and not just at the end) -- an earlier version of
this script only saved once per symbol and had no log file, so a stall or
crash mid-symbol was indistinguishable from ~45 minutes of normal silent
progress. Re-running is still not resumable mid-symbol: a symbol that didn't
finish gets recomputed from its first fold.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import list_symbols, load_candles
from pinance_ml.data.news_db import load_news
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.news.decay import NEWS_FEATURE_COLUMNS, base_asset, compute_news_features
from pinance_ml.splits import walk_forward_folds
from pinance_ml.tracking import log_research_run


def _news_importance_rows(symbol: str, fold_index: int, horizon: int, model) -> list[dict]:
    """Gain-based feature importance for the with_news variant's model at
    this horizon, restricted to the news columns -- answers "does the model
    use news features at all" independently of whether MAE improved, since
    a feature can get real split gain and still not move pooled MAE if its
    signal is inconsistent in direction."""
    booster = model.booster_
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    total_gain = float(gains.sum())
    return [
        {
            "symbol": symbol,
            "fold": fold_index,
            "horizon": horizon,
            "feature": name,
            "gain": float(gain),
            "total_gain": total_gain,
            "gain_pct_of_total": (float(gain) / total_gain * 100) if total_gain > 0 else 0.0,
        }
        for name, gain in zip(names, gains)
        if name in NEWS_FEATURE_COLUMNS
    ]


def run_for_symbol(symbol, btc_candles, log, save_progress) -> tuple[pd.DataFrame, pd.DataFrame]:
    """log(msg) writes a timestamped progress line. save_progress(symbol, rows_df,
    importance_df) persists whatever's been computed for this symbol so far --
    called after every fold, not just once the whole symbol is done."""
    log(f"{symbol}: loading candles + news")
    candles = load_candles(symbol)
    dataset = build_dataset(candles, btc_candles=btc_candles)
    base_feat_cols = feature_columns(dataset)

    news = load_news(asset=base_asset(symbol))
    log(f"{symbol}: {len(news)} news items for asset {base_asset(symbol)}")
    news_features = compute_news_features(news, dataset["ts"])
    dataset = pd.concat([dataset.reset_index(drop=True), news_features], axis=1)
    all_feat_cols = base_feat_cols + NEWS_FEATURE_COLUMNS

    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    total_fold_variants = len(folds) * 2
    log(f"{symbol}: {len(folds)} walk-forward folds ({total_fold_variants} fold-variants to train+score)")

    rows = []
    importance_rows = []
    symbol_t0 = time.time()
    for fold in folds:
        for variant, feat_cols in (("technical_only", base_feat_cols), ("with_news", all_feat_cols)):
            t0 = time.time()
            models = train_horizon_models(fold.train, feat_cols)
            preds = predict_horizons(models, fold.test, feat_cols)
            for h in HORIZONS:
                actual = fold.test[f"r_{h}"].to_numpy()
                predicted = preds[f"r_{h}_pred"].to_numpy()
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": variant,
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
                if variant == "with_news":
                    importance_rows.extend(_news_importance_rows(symbol, fold.index, h, models[h]))
            done = len(rows) // len(HORIZONS)
            avg_per_fold_variant = (time.time() - symbol_t0) / done
            eta_min = avg_per_fold_variant * (total_fold_variants - done) / 60
            extra = ""
            if variant == "with_news":
                fold_imp = pd.DataFrame(importance_rows)
                fold_imp = fold_imp[fold_imp["fold"] == fold.index]
                by_feat = fold_imp.groupby("feature")["gain_pct_of_total"].mean().sort_values(ascending=False)
                news_share = by_feat.sum()
                top = f"{by_feat.index[0]}={by_feat.iloc[0]:.2f}%" if len(by_feat) else "n/a"
                extra = f" | news gain share {news_share:.2f}% of total (top: {top})"
            log(
                f"  [fold {fold.index}/{len(folds) - 1}] {variant}: 12 models trained+scored in "
                f"{time.time() - t0:.1f}s ({done}/{total_fold_variants} fold-variants, "
                f"ETA {eta_min:.1f}min for {symbol}){extra}"
            )
        save_progress(symbol, pd.DataFrame(rows), pd.DataFrame(importance_rows))
    return pd.DataFrame(rows), pd.DataFrame(importance_rows)


def compare_variants(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Pooled MAE/directional accuracy per (symbol, horizon) for each
    variant, joined side by side with the with_news-minus-technical_only
    delta -- negative mae_delta means the news features reduced error."""
    technical = pool_fold_metrics(fold_results[fold_results["variant"] == "technical_only"])
    with_news = pool_fold_metrics(fold_results[fold_results["variant"] == "with_news"])

    comparison = technical.set_index(["symbol", "horizon"])[["mae", "directional_accuracy"]].join(
        with_news.set_index(["symbol", "horizon"])[["mae", "directional_accuracy"]],
        lsuffix="_technical_only",
        rsuffix="_with_news",
    )
    comparison["mae_delta"] = comparison["mae_with_news"] - comparison["mae_technical_only"]
    comparison["directional_accuracy_delta"] = (
        comparison["directional_accuracy_with_news"] - comparison["directional_accuracy_technical_only"]
    )
    return comparison.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", help="Symbols to evaluate (default: all in DB)")
    parser.add_argument("--out", default="reports/news_feature_gain_folds.csv")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Derived from out_path's own stem (with a trailing "_folds" stripped,
    # so the default --out's sibling files keep their original names), not
    # a hardcoded name -- so a custom --out (e.g. a single-symbol pilot
    # run) gets its own sibling files instead of clobbering the default
    # multi-symbol run's summary/importance CSVs.
    stem = out_path.stem[: -len("_folds")] if out_path.stem.endswith("_folds") else out_path.stem
    importance_path = out_path.with_name(f"{stem}_importance.csv")
    log_path = out_path.parent / "news_feature_gain.log"
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        print(line, file=log_file, flush=True)

    run_t0 = time.time()
    symbols = args.symbols or list_symbols()
    log(f"Symbols ({len(symbols)}): {symbols}")
    log(f"Logging to {log_path}, incremental fold results to {out_path}, "
        f"news feature importance to {importance_path}")

    btc_candles = load_candles("BTCUSDT") if any(s != "BTCUSDT" for s in symbols) else None

    all_rows = []
    all_importance_rows = []

    def save_progress(symbol: str, symbol_rows_so_far: pd.DataFrame, symbol_importance_so_far: pd.DataFrame) -> None:
        fold_results = pd.concat(all_rows + [symbol_rows_so_far], ignore_index=True)
        fold_results.to_csv(out_path, index=False)
        importance = pd.concat(all_importance_rows + [symbol_importance_so_far], ignore_index=True)
        importance.to_csv(importance_path, index=False)

    for i, s in enumerate(symbols, start=1):
        log(f"[{i}/{len(symbols)}] starting {s}")
        symbol_rows, symbol_importance = run_for_symbol(
            s, None if s == "BTCUSDT" else btc_candles, log, save_progress
        )
        all_rows.append(symbol_rows)
        all_importance_rows.append(symbol_importance)
        fold_results = pd.concat(all_rows, ignore_index=True)
        fold_results.to_csv(out_path, index=False)
        importance = pd.concat(all_importance_rows, ignore_index=True)
        importance.to_csv(importance_path, index=False)
        elapsed_min = (time.time() - run_t0) / 60
        log(f"[{i}/{len(symbols)}] {s}: done, {len(fold_results)} total rows saved to {out_path} "
            f"(run elapsed {elapsed_min:.1f}min)")

    comparison = compare_variants(fold_results)
    comparison_path = out_path.with_name(f"{stem}_summary.csv")
    comparison.to_csv(comparison_path, index=False)
    log(f"Saved per-horizon comparison to {comparison_path}")

    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 160):
        log("\n" + comparison.to_string(index=False))

    importance_summary = (
        importance.groupby(["symbol", "feature"])["gain_pct_of_total"].mean().unstack("feature").round(3)
    )
    log(f"Saved news feature importance to {importance_path}")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        log("Mean gain % of total per news feature, by symbol:\n" + importance_summary.to_string())

    log(f"Total run time: {(time.time() - run_t0) / 60:.1f}min")

    log_research_run(
        __file__,
        run_name=f"news-gain-{'+'.join(symbols)}" if len(symbols) <= 3 else f"news-gain-{len(symbols)}sym",
        params={
            "symbols": ",".join(symbols),
            "n_symbols": len(symbols),
            "out": str(out_path),
            "n_fold_rows": len(fold_results),
        },
        report_paths=[*sorted(out_path.parent.glob(f"{stem}_*")), log_path],
    )

    log_file.close()


if __name__ == "__main__":
    main()
