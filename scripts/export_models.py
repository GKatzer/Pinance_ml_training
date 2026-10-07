"""Export final LightGBM models for serving (predictor-ml-inference).

Unlike scripts/train_lightgbm.py (walk-forward eval, models discarded after
scoring), this trains one final model per symbol/horizon on the full
available history and persists it via LightGBM's native text format
(`Booster.save_model`) rather than pickling the sklearn wrapper -- stable
across library versions, unlike pickle.

Point models (h{horizon}.txt, also the confidence corridor's median) and
the corridor's tail models (h{horizon}_q{quantile}.txt) are independently
trainable/exportable -- export_symbol_point() / export_symbol_quantiles()
below, used separately by scripts/auto_retrain.py and
scripts/auto_retrain_quantiles.py respectively, each merging its own
metadata fields onto whatever's already in a MinIO slot rather than
overwriting it wholesale (see model_storage.merge_metadata). export_symbol()
combines both into one fresh, self-contained local export -- the
manual/CLI/from-scratch case (bootstrap a brand new symbol, or a one-off
full re-export), not tied to any existing MinIO state.

Per symbol, writes:

    models/{symbol}/h{horizon}.txt             -- point estimate / median
    models/{symbol}/h{horizon}_q{quantile}.txt -- confidence-corridor tails
                                                   (only with --include-quantiles)
    models/{symbol}/metadata.json              -- feature schema + provenance,
                                                   checked by the inference
                                                   service before it trusts a
                                                   model directory

Usage:

    python scripts/export_models.py [SYMBOL ...] [--start 2024-01-01 | --window-days 730]
        [--out models] [--include-quantiles]

With no symbols, exports every symbol found in the `candles` table.
--start pins an exact history start date (also handy for a fast local
smoke run on a short slice). --window-days is the sliding-window
alternative for real/production exports: trains on the most recent N
days as of *this run*, recomputed fresh each time rather than a fixed
date that goes stale -- see splits.py's walk_forward_folds docstring for
why an ever-expanding full-history window stops meaningfully refreshing
anything once history spans years. Omit both for the old expanding-window
behavior (full history, every run).

--include-quantiles additionally trains+saves the CORRIDOR_QUANTILES tail
models (README's "доверительный коридор").
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.config import CORRIDOR_QUANTILES, DRIFT_BASELINE_N_BINS, DRIFT_FEATURE_COLUMNS
from pinance_ml.data.db import list_symbols, load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.models.lightgbm_model import train_horizon_models, train_quantile_models


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent).decode().strip()


def _schema_version(feat_cols: list[str]) -> str:
    """Short hash of the feature name list, in stored order.

    Not a manually-bumped counter: any change to compute_features that adds,
    removes, or reorders columns changes this automatically, so the
    inference service's staleness check can't be forgotten to update.
    """
    digest = hashlib.sha256(",".join(feat_cols).encode()).hexdigest()
    return digest[:12]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _library_versions() -> dict:
    return {
        "pandas": version("pandas"),
        "numpy": version("numpy"),
        "lightgbm": version("lightgbm"),
        "scikit-learn": version("scikit-learn"),
        "ta": version("ta"),
    }


def _feature_baseline(dataset: pd.DataFrame) -> dict:
    """Population baseline for DRIFT_FEATURE_COLUMNS, computed on this
    export's training sample -- the reference distribution predictor-
    backend's PSI drift check (see config.py's DRIFT_FEATURE_COLUMNS
    docstring) compares live feature_snapshot values against. Stored in
    metadata.json alongside model_version/feature_columns, so it's
    versioned and refreshed with the model itself (recomputed on every
    export_symbol_point call) rather than living as a separate static file.

    `duplicates="drop"` -- a feature with a long flat/repeated stretch
    (e.g. many identical values) can collapse decile edges into fewer than
    DRIFT_BASELINE_N_BINS bins; still a valid, if coarser, baseline rather
    than pd.qcut raising on non-unique bin edges.
    """
    baseline = {}
    for col in DRIFT_FEATURE_COLUMNS:
        if col not in dataset.columns:
            # e.g. btc_ret when exporting BTCUSDT itself -- compute_features
            # never adds that column for the base asset (see dataset.py),
            # same convention predictor-ml-inference's _feature_snapshot uses.
            baseline[col] = None
            continue
        values = dataset[col].dropna()
        if values.empty:
            baseline[col] = None
            continue
        binned, edges = pd.qcut(values, DRIFT_BASELINE_N_BINS, duplicates="drop", retbins=True)
        fractions = binned.value_counts(sort=False, normalize=True)
        baseline[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "bin_edges": edges.tolist(),
            "bin_fractions": fractions.tolist(),
        }
    return baseline


def export_symbol_point(
    symbol: str,
    dataset: pd.DataFrame,
    feat_cols: list[str],
    out_dir: Path,
    start: str | None,
) -> dict:
    """Train+save the 12 point/horizon models and return their metadata
    fields -- does NOT write metadata.json itself (the caller may need to
    merge these onto an existing slot's quantile_* fields first, see
    model_storage.merge_metadata). Doesn't touch or need anything
    quantile-related."""
    t0 = time.time()
    _log(f"{symbol}: {len(dataset)} rows, {len(feat_cols)} features -- training 12 horizon models")
    models = train_horizon_models(dataset, feat_cols)

    symbol_dir = out_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    for h, model in models.items():
        model.booster_.save_model(str(symbol_dir / f"h{h}.txt"))

    trained_at = datetime.now(timezone.utc)
    source_commit = _source_commit()
    metadata = {
        "symbol": symbol,
        "schema_version": _schema_version(feat_cols),
        # predictor-ml-inference's shadow-deployment gate keys off this to
        # tell production and candidate models apart in the predictions
        # table (model_version column) -- not schema_version, which only
        # changes when the *feature set* changes, not on every retrain.
        "model_version": f"{trained_at:%Y%m%d%H%M}-{source_commit[:8]}",
        "feature_columns": feat_cols,
        "horizons": sorted(models.keys()),
        "source_commit": source_commit,
        "trained_at": trained_at.isoformat(),
        "n_rows": len(dataset),
        "history_start": start,
        "library_versions": _library_versions(),
        "feature_baseline": _feature_baseline(dataset),
    }
    _log(f"{symbol}: wrote {len(models)} point models -> {symbol_dir} ({time.time() - t0:.1f}s)")
    return metadata


def export_symbol_quantiles(
    symbol: str,
    dataset: pd.DataFrame,
    feat_cols: list[str],
    out_dir: Path,
    start: str | None,
    quantiles: tuple[float, ...] = CORRIDOR_QUANTILES,
) -> dict:
    """Train+save the confidence-corridor tail models and return their
    metadata fields (all `quantile_`-prefixed, disjoint from
    export_symbol_point's keys -- see model_storage.merge_metadata).
    Independent of the point models: doesn't train or need them, only
    `feat_cols` for schema compatibility. The median (alpha=0.5) is
    deliberately not trained here -- whatever h{h}.txt currently lives in
    the target slot (independently pushed/promoted, possibly by a
    different retrain run) already estimates it, since the point model's
    objective="regression_l1" already minimizes L1 loss (median-optimal).
    """
    t0 = time.time()
    quantile_models = train_quantile_models(dataset, feat_cols, quantiles=quantiles)
    _log(f"{symbol}: training {len(quantile_models)} quantile-corridor models {quantiles}")

    symbol_dir = out_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    for (h, q), model in quantile_models.items():
        model.booster_.save_model(str(symbol_dir / f"h{h}_q{q}.txt"))

    trained_at = datetime.now(timezone.utc)
    source_commit = _source_commit()
    metadata = {
        "symbol": symbol,
        "quantile_schema_version": _schema_version(feat_cols),
        "quantile_model_version": f"{trained_at:%Y%m%d%H%M}-{source_commit[:8]}",
        "quantile_levels": sorted({q for _, q in quantile_models}),
        "quantile_horizons": sorted({h for h, _ in quantile_models}),
        # Not a separate file -- see docstring above.
        "quantile_median_source": "point_model",
        "quantile_source_commit": source_commit,
        "quantile_trained_at": trained_at.isoformat(),
        "quantile_n_rows": len(dataset),
        "quantile_history_start": start,
        "library_versions": _library_versions(),
    }
    _log(f"{symbol}: wrote {len(quantile_models)} quantile models -> {symbol_dir} ({time.time() - t0:.1f}s)")
    return metadata


def export_symbol(
    symbol: str,
    btc_candles: pd.DataFrame | None,
    out_dir: Path,
    start: str | None,
    quantiles: tuple[float, ...] | None = None,
) -> None:
    """Combined point + (optionally) corridor export into one fresh,
    self-contained metadata.json -- the manual/CLI/from-scratch case. The
    automated retrain scripts don't call this: they call
    export_symbol_point/export_symbol_quantiles directly so they can
    merge onto whatever's already in the target MinIO slot instead of
    overwriting it wholesale."""
    candles = load_candles(symbol, start=start)
    dataset = build_dataset(candles, btc_candles=btc_candles)
    feat_cols = feature_columns(dataset)

    metadata = export_symbol_point(symbol, dataset, feat_cols, out_dir, start)
    if quantiles:
        metadata.update(export_symbol_quantiles(symbol, dataset, feat_cols, out_dir, start, quantiles))
    else:
        metadata["quantile_levels"] = []
        metadata["quantile_median_source"] = None

    symbol_dir = out_dir / symbol
    (symbol_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    _log(f"{symbol}: wrote metadata.json -> {symbol_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="*", help="Symbols to export (default: all in DB)")
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--start", default=None, help="ISO date; pins an exact training history start")
    window_group.add_argument(
        "--window-days", type=float, default=None,
        help="Sliding window: train on the most recent N days as of now (recomputed each run)",
    )
    parser.add_argument("--out", default="models", help="Output directory (default: models/)")
    parser.add_argument(
        "--include-quantiles", action="store_true",
        help="Also train+export the confidence-corridor tail models (CORRIDOR_QUANTILES)",
    )
    args = parser.parse_args()
    quantiles = CORRIDOR_QUANTILES if args.include_quantiles else None

    if args.window_days is not None:
        start = (datetime.now(timezone.utc) - timedelta(days=args.window_days)).strftime("%Y-%m-%d")
    else:
        start = args.start

    symbols = args.symbols or list_symbols()
    out_dir = Path(args.out)
    run_t0 = time.time()
    _log(f"Symbols ({len(symbols)}): {symbols}")
    _log(f"History start: {start or '(full)'}" + (f" (sliding {args.window_days:.0f}d window)" if args.window_days else ""))

    btc_candles = load_candles("BTCUSDT", start=start) if any(s != "BTCUSDT" for s in symbols) else None

    for i, symbol in enumerate(symbols, start=1):
        _log(f"[{i}/{len(symbols)}] starting {symbol}")
        export_symbol(symbol, None if symbol == "BTCUSDT" else btc_candles, out_dir, start, quantiles=quantiles)

    _log(f"All done. Total run time: {(time.time() - run_t0) / 60:.1f}min")


if __name__ == "__main__":
    main()
