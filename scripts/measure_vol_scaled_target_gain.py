"""One-off: does dividing the regression target by a causal volatility
estimate (a "vol-scaled target") beat the raw log-return target on pooled
MAE?

Motivation: r_h at a 5-minute cadence is strongly heteroskedastic -- a
handful of high-volatility windows dominate the L1 loss, so the model
spends capacity fitting scale rather than direction/shape. Training each
horizon on

    z_h[t] = r_h[t] / sigma[t]        (sigma[t] known at t, no lookahead)

lets the model allocate capacity by information content instead of by
return magnitude. Predictions are rescaled back by the *same* sigma[t]
and every metric is computed in raw log-return units on identical folds,
so the only thing differing between conditions is the target
parametrization -- the same harness discipline as
scripts/measure_news_feature_gain.py (there the feature set differs, here
the target does).

Closest public analog: G-Research Crypto Forecasting -- vol-scaling /
winsorizing the target is among the most common target transforms in top
solutions.

sigma candidates are causal rolling stats already in the feature matrix
(ret_std_144 = 12h, ret_std_36 = 3h, atr) -- no new inputs and no
lookahead (tests/test_pipeline.py::test_no_lookahead already covers these
columns). A separate model is fit per horizon, so the constant sqrt(h)
factor between a 1-candle sigma and an h-candle target is absorbed on
rescale and doesn't need modeling here -- only sigma's time variation
matters. sigma is also already a model *feature* in every condition, so
the scaled variant can partly learn to undo the scaling; that's fine, the
target is still an invertible reparametrization of a known quantity.

Directional accuracy is NOT trivially unchanged by the transform: a
single prediction keeps its sign under positive scaling, but a model fit
to z_h learns different splits than one fit to r_h, so its sign
predictions can differ. Pooled MAE in raw units is the headline number;
DA is reported alongside, with the caveat that this 8-fold harness can't
resolve DA effects under ~1.5pp (see scripts/screen_intermediate_horizons.py).

Usage:
    python scripts/measure_vol_scaled_target_gain.py [SYMBOL ...] \
        [--sigma-cols ret_std_144,ret_std_36] \
        [--out reports/vol_scaled_target_folds.csv] [--limit-days N]

With no SYMBOL, runs BTCUSDT only -- the first-screen convention the
other compare_*/screen_* scripts use. --limit-days loads only the most
recent N days of candles (smoke-testing the wiring in a couple of
minutes, not a real measurement).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from pinance_ml.config import HORIZONS, LIGHTGBM_TEST_DAYS, PURGE_ROWS, WALK_FORWARD_MIN_TRAIN_DAYS
from pinance_ml.data.db import list_symbols, load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.evaluation import pool_fold_metrics
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.splits import walk_forward_folds
from pinance_ml.tracking import log_research_run

# Pre-registered relative-MAE bar reused from the auto-retrain gate
# (config.AUTO_RETRAIN_MIN_IMPROVEMENT): a scaled condition has to beat
# raw by at least this fraction to count as a real improvement rather
# than walk-forward noise.
IMPROVEMENT_BAR = 0.005


def scale_targets(df: pd.DataFrame, sigma_col: str, horizons: list[int]) -> pd.DataFrame:
    """Copy of `df` with each r_h replaced by r_h / sigma.

    sigma == 0 is mapped to NaN (dead-market rows), so those rows drop out
    via train_horizon_models' per-horizon `notna()` target mask instead of
    turning into +/-inf. sigma NaN (warm-up rows before the rolling window
    fills) propagates to a NaN target the same way -- a ~0.06% train-row
    difference vs the raw condition on BTC's history, noted so it isn't a
    hidden confound.
    """
    out = df.copy()
    sigma = out[sigma_col].replace(0.0, np.nan)
    for h in horizons:
        out[f"r_{h}"] = out[f"r_{h}"] / sigma
    return out


def predict_for_condition(
    fold, feat_cols: list[str], sigma_col: str | None
) -> pd.DataFrame:
    """r_h_pred for one (fold, condition), always in raw log-return units.

    sigma_col=None is the raw target (current production behaviour).
    Otherwise the models are fit to the vol-scaled target and their
    predictions are multiplied back by the test rows' own sigma.
    """
    train = fold.train if sigma_col is None else scale_targets(fold.train, sigma_col, HORIZONS)
    models = train_horizon_models(train, feat_cols)
    preds = predict_horizons(models, fold.test, feat_cols)
    if sigma_col is not None:
        rescale = pd.Series(
            fold.test[sigma_col].replace(0.0, np.nan).to_numpy(), index=preds.index
        )
        preds = preds.multiply(rescale, axis=0)
    return preds


def run_for_symbol(symbol, btc_candles, sigma_cols, limit_days, log, save):
    candles = load_candles(symbol)
    if limit_days is not None:
        cutoff = candles["ts"].max() - pd.Timedelta(days=limit_days)
        candles = candles[candles["ts"] >= cutoff]
        log(f"{symbol}: --limit-days {limit_days} -> {len(candles)} candles from {candles['ts'].min()}")

    dataset = build_dataset(candles, btc_candles=btc_candles)
    feat_cols = feature_columns(dataset)
    missing = [c for c in sigma_cols if c not in dataset.columns]
    if missing:
        raise SystemExit(f"--sigma-cols entries not in the feature matrix: {missing}")

    folds = list(walk_forward_folds(dataset, WALK_FORWARD_MIN_TRAIN_DAYS, LIGHTGBM_TEST_DAYS, PURGE_ROWS))
    conditions = [("raw", None)] + [(f"scaled_{c}", c) for c in sigma_cols]
    total = len(folds) * len(conditions)
    log(
        f"{symbol}: {len(dataset)} rows, {len(feat_cols)} features, {len(folds)} folds, "
        f"{len(conditions)} conditions -> {total} fold-variants to train+score"
    )

    rows = []
    t0 = time.time()
    for fold in folds:
        for name, sigma_col in conditions:
            ct0 = time.time()
            preds = predict_for_condition(fold, feat_cols, sigma_col)
            for h in HORIZONS:
                actual = fold.test[f"r_{h}"].to_numpy()
                predicted = preds[f"r_{h}_pred"].to_numpy()
                rows.append(
                    {
                        "symbol": symbol,
                        "condition": name,
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
            done = len(rows) // len(HORIZONS)
            eta_min = (time.time() - t0) / done * (total - done) / 60
            log(
                f"  [fold {fold.index}/{len(folds) - 1}] {name}: 12 models trained+scored in "
                f"{time.time() - ct0:.1f}s ({done}/{total} fold-variants, ETA {eta_min:.1f}min for {symbol})"
            )
        save(symbol, pd.DataFrame(rows))
    return pd.DataFrame(rows)


def summarize(fold_results: pd.DataFrame, out_path: Path, log) -> pd.DataFrame:
    stem = out_path.stem[: -len("_folds")] if out_path.stem.endswith("_folds") else out_path.stem

    per_horizon = pool_fold_metrics(fold_results, group_cols=["symbol", "condition", "horizon"])
    per_horizon_path = out_path.with_name(f"{stem}_per_horizon.csv")
    per_horizon.to_csv(per_horizon_path, index=False)

    ph = per_horizon.copy()
    ph["mae_w"] = ph["mae"] * ph["n_test_total"]
    ph["da_w"] = ph["directional_accuracy"] * ph["n_test_total"]
    overall = (
        ph.groupby(["symbol", "condition"])
        .apply(
            lambda g: pd.Series(
                {
                    "mean_mae": g["mae_w"].sum() / g["n_test_total"].sum(),
                    "mean_directional_accuracy": g["da_w"].sum() / g["n_test_total"].sum(),
                    "n_folds": int(g["n_folds"].max()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    raw_by_symbol = overall[overall["condition"] == "raw"].set_index("symbol")
    overall["mae_delta_vs_raw"] = overall["mean_mae"] - overall["symbol"].map(raw_by_symbol["mean_mae"])
    overall["mae_pct_vs_raw"] = overall["mae_delta_vs_raw"] / overall["symbol"].map(raw_by_symbol["mean_mae"])
    overall["da_delta_vs_raw_pp"] = (
        overall["mean_directional_accuracy"] - overall["symbol"].map(raw_by_symbol["mean_directional_accuracy"])
    ) * 100

    summary_path = out_path.with_name(f"{stem}_summary.csv")
    overall.to_csv(summary_path, index=False)

    with pd.option_context("display.float_format", "{:.6f}".format, "display.width", 160):
        log("\n=== Pooled across horizons (weighted by n_test) ===\n" + overall.to_string(index=False))
    log(f"\nSaved per-horizon -> {per_horizon_path}, summary -> {summary_path}")

    for _, r in overall[overall["condition"] != "raw"].iterrows():
        verdict = (
            "CLEARS bar" if r["mae_pct_vs_raw"] <= -IMPROVEMENT_BAR
            else "worse than raw" if r["mae_pct_vs_raw"] > 0
            else "below bar"
        )
        log(
            f"  {r['symbol']}/{r['condition']}: MAE {r['mae_pct_vs_raw'] * 100:+.2f}% vs raw "
            f"(bar -{IMPROVEMENT_BAR * 100:.1f}%) -> {verdict}; "
            f"DA {r['da_delta_vs_raw_pp']:+.2f}pp (harness noise floor ~1.5pp)"
        )
    return overall


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="*", help="Symbols to evaluate (default: BTCUSDT only)")
    parser.add_argument("--sigma-cols", default="ret_std_144,ret_std_36")
    parser.add_argument("--out", default="reports/vol_scaled_target_folds.csv")
    parser.add_argument("--limit-days", type=float, default=None, help="Smoke test: only the most recent N days")
    args = parser.parse_args()

    sigma_cols = [c.strip() for c in args.sigma_cols.split(",") if c.strip()]
    symbols = args.symbols or ["BTCUSDT"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "vol_scaled_target_gain.log"
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        print(line, file=log_file, flush=True)

    run_t0 = time.time()
    log(f"symbols={symbols} sigma_cols={sigma_cols} out={out_path} limit_days={args.limit_days}")

    btc_candles = load_candles("BTCUSDT") if any(s != "BTCUSDT" for s in symbols) else None

    all_rows = []

    def save(symbol: str, symbol_rows_so_far: pd.DataFrame) -> None:
        pd.concat(all_rows + [symbol_rows_so_far], ignore_index=True).to_csv(out_path, index=False)

    for i, s in enumerate(symbols, start=1):
        log(f"[{i}/{len(symbols)}] starting {s}")
        symbol_rows = run_for_symbol(
            s, None if s == "BTCUSDT" else btc_candles, sigma_cols, args.limit_days, log, save
        )
        all_rows.append(symbol_rows)
        pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)
        log(f"[{i}/{len(symbols)}] {s} done (run elapsed {(time.time() - run_t0) / 60:.1f}min)")

    fold_results = pd.concat(all_rows, ignore_index=True)
    summarize(fold_results, out_path, log)
    log(f"Total run time: {(time.time() - run_t0) / 60:.1f}min")

    log_research_run(
        __file__,
        run_name=f"vol-scaled-{'+'.join(symbols)}" if len(symbols) <= 3 else f"vol-scaled-{len(symbols)}sym",
        params={
            "symbols": ",".join(symbols),
            "sigma_cols": ",".join(sigma_cols),
            "improvement_bar": IMPROVEMENT_BAR,
            "limit_days": args.limit_days,
            "out": str(out_path),
            "n_fold_rows": len(fold_results),
        },
        report_paths=sorted(out_path.parent.glob("vol_scaled_target_*")),
    )

    log_file.close()


if __name__ == "__main__":
    main()
