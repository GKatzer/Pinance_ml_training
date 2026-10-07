"""Auto-retrain the confidence corridor (README "Доверительный коридор"):
scheduled retrain of the CORRIDOR_QUANTILES tail models, gated against
current production, entirely independent of scripts/auto_retrain.py's
point regressor.

A separate script/schedule, not a flag on auto_retrain.py, by design (see
project notes): the corridor is a
different model with a different promotion criterion (pinball loss +
coverage, not MAE) and a different natural refresh cadence (weekly is
plenty -- the corridor's calibration doesn't drift day to day the way
point predictions can). Bundling the two was tried first and reverted --
a candidate corridor pushed alongside a point model that happened to be
marginally worse than production (pure retrain noise, unrelated to
corridor quality) would have blocked an improving corridor from ever
shipping, since both travelled as one atomic push.

Per symbol:
  1. Build features on the full available history, split off a recent
     AUTO_RETRAIN_HOLDOUT_DAYS-wide window the candidate's own training
     never sees.
  2. Train a quantile "eval candidate" (CORRIDOR_QUANTILES) on everything
     before that window.
  3. Download current production's model set from MinIO (if any) --
     whatever point models happen to be there are irrelevant here, only
     the quantile_* metadata fields and h{h}_q{q}.txt files matter.
  4. Evaluate both on the same held-out window -- pooled pinball loss +
     a coverage-calibration sanity check (a candidate that fails
     calibration is rejected outright, regardless of its pinball loss).
  5. Decide (decide_quantile_target_slot):
       - no metadata.json in production at all -> bootstrap straight to
         production (mirrors auto_retrain.py's own bootstrap case).
       - production has a point model but no matching corridor yet ->
         nothing to compare against, push to candidate unconditionally.
       - production's corridor feature schema doesn't match this repo's
         current pipeline -> same treatment, push to candidate -- but first
         the corridor must clear a naive-baseline sanity gate
         (pinance_ml.sanity: pinball loss vs the constant empirical-quantile
         forecast + calibrated coverage); fail it and nothing is pushed
         ("schema_drift_rejected_sanity"). The same gate applies when the
         corridor is new and production's POINT model is on an older schema.
         A pushed corridor on a changed schema carries metadata
         `quantile_new_scheme: true` (written explicitly, false otherwise),
         promoted by scripts/promote_new_scheme.py.
       - candidate clears AUTO_RETRAIN_MIN_IMPROVEMENT on pinball loss
         AND passes the coverage check -> push to candidate.
       - otherwise -> log and stop, nothing pushed.
  6. On push: only the quantile files (h{h}_q{q}.txt) + `quantile_`-
     prefixed metadata fields are written, merged onto whatever's already
     in the target slot (model_storage.merge_metadata) -- this never
     touches or clobbers the point model's own fields/files, whichever
     retrain run they last came from. See auto_retrain.py, which does the
     same merge in the other direction.

Usage:
    python scripts/auto_retrain_quantiles.py [SYMBOL ...]
        [--holdout-days 30] [--min-improvement 0.005] [--staging-dir .auto_retrain_staging]
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_models import _schema_version, _source_commit, export_symbol_quantiles  # noqa: E402

from pinance_ml.backend_client import post_retrain_event
from pinance_ml.config import (
    AUTO_RETRAIN_HOLDOUT_DAYS,
    AUTO_RETRAIN_MIN_IMPROVEMENT,
    CORRIDOR_COVERAGE_TOLERANCE,
    CORRIDOR_QUANTILES,
    SANITY_MAX_PINBALL_RATIO,
)
from pinance_ml.data.db import list_symbols, load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.metrics import coverage, pinball_loss
from pinance_ml.model_storage import (
    CANDIDATE_SLOT,
    PRODUCTION_SLOT,
    download_metadata,
    download_model_set,
    merge_metadata,
    upload_model_set,
)
from pinance_ml.models.lightgbm_model import predict_quantile_horizons, train_quantile_models
from pinance_ml.sanity import naive_quantile_pinball, quantile_sanity_check
from pinance_ml.tracking import log_artifact, log_dict, log_metrics, log_params, mlflow_run, set_tags

# See auto_retrain.py's own _ADVANCING_DECISIONS -- same binary-mapping
# rationale, for decide_quantile_target_slot's labels instead.
_ADVANCING_DECISIONS = {"bootstrap", "pushed_to_candidate", "schema_drift_push_as_candidate", "no_production_quantiles_push_as_candidate"}

# Pushes that skip the candidate-vs-production comparison (nothing
# comparable to score against) -- the only ones the naive-baseline sanity
# gate applies to, and the only ones that can carry quantile_new_scheme.
_UNCOMPARED_PUSH_DECISIONS = {"schema_drift_push_as_candidate", "no_production_quantiles_push_as_candidate"}
SCHEMA_DRIFT_REJECTED = "schema_drift_rejected_sanity"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _quantile_eval_frame(
    models: dict[tuple[int, float], lgb.Booster],
    feat_cols: list[str],
    eval_data: pd.DataFrame,
    coverage_tolerance: float,
) -> pd.DataFrame:
    """Per-(horizon, quantile) pinball loss, empirical coverage, and
    whether that coverage lands within `coverage_tolerance` of nominal
    alpha. _avg_pinball_and_coverage_ok reduces this to (mean pinball,
    majority-calibrated bool); the retrain run also logs the whole frame
    as an artifact."""
    preds = predict_quantile_horizons(models, eval_data, feat_cols)
    rows = []
    for h, q in models:
        actual = eval_data[f"r_{h}"].to_numpy()
        predicted = preds[f"r_{h}_q{q}_pred"].to_numpy()
        valid = pd.notna(actual)
        if valid.sum() == 0:
            continue
        cov = coverage(actual[valid], predicted[valid])
        rows.append(
            {
                "horizon": h,
                "quantile": q,
                "pinball": pinball_loss(actual[valid], predicted[valid], quantile=q),
                "coverage": cov,
                "within_tolerance": bool(abs(cov - q) <= coverage_tolerance),
                "n": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def _avg_pinball_and_coverage_ok(
    models: dict[tuple[int, float], lgb.Booster],
    feat_cols: list[str],
    eval_data: pd.DataFrame,
    coverage_tolerance: float,
) -> tuple[float, bool]:
    """Mean pinball loss across every (horizon, quantile) pair on
    `eval_data`, plus a coverage-calibration sanity check -- the quantile
    corridor's counterpart of auto_retrain.py's _avg_mae_and_dir_acc.
    `coverage_ok` requires a *majority* of (horizon, quantile) pairs to
    land within `coverage_tolerance` of their own nominal alpha (same
    convention scripts/measure_quantile_gain.py used for fold-level
    calibration) -- a corridor that improves its pooled pinball loss
    while drifting miscalibrated on most horizons isn't actually a better
    corridor."""
    frame = _quantile_eval_frame(models, feat_cols, eval_data, coverage_tolerance)
    if frame.empty:
        return float("nan"), False
    return float(frame["pinball"].mean()), bool(frame["within_tolerance"].mean() > 0.5)


def _load_production_quantile_boosters(
    metadata: dict, model_dir: Path, quantiles: tuple[float, ...]
) -> dict[tuple[int, float], lgb.Booster]:
    return {
        (h, q): lgb.Booster(model_file=str(model_dir / f"h{h}_q{q}.txt"))
        for h in metadata["quantile_horizons"]
        for q in quantiles
    }


def decide_quantile_target_slot(
    has_production: bool,
    quantile_schema_matches: bool,
    has_production_quantiles: bool,
    candidate_pinball: float,
    production_pinball: float | None,
    candidate_coverage_ok: bool,
    min_improvement: float,
    sanity_passed: bool | None = None,
) -> tuple[str, str | None]:
    """Pure push-gate decision for the confidence corridor, factored out
    of retrain_symbol_quantiles so it's testable without a DB/MinIO/any
    actual training -- fully self-contained (no borrowed point-model
    state), the corridor's counterpart of auto_retrain.py's
    decide_target_slot. Scored on pinball loss (the quantile equivalent
    of MAE), with an extra required condition: `candidate_coverage_ok`. A
    corridor that "improves" pooled pinball loss while its coverage has
    drifted away from nominal on most (horizon, quantile) pairs is
    rejected outright, regardless of the pinball-loss number (see
    _avg_pinball_and_coverage_ok).

    `has_production` here means "does PRODUCTION_SLOT have ANY
    metadata.json at all" (a point-only slot counts). `has_production_
    quantiles` is the corridor-specific check on top of that: a symbol
    can already have a production point regressor with no corridor in it
    yet (this script never ran for it before, or CORRIDOR_QUANTILES
    changed) -- that's not schema drift, just nothing to compare the
    corridor against yet, so it pushes as a candidate unconditionally,
    same as bootstrap/schema-drift do.

    `sanity_passed` (pinance_ml.sanity.quantile_sanity_check vs the
    constant-quantile naive baseline) gates the two pushes that skip the
    comparison against production, when the caller computed it (it does so
    only when the feature schema actually changed). False -> nothing is
    pushed ("schema_drift_rejected_sanity"); None (default) keeps the old
    unconditional push.

    Returns (decision_label, target_slot) -- target_slot is None for
    "rejected"/"rejected_miscalibrated"/"schema_drift_rejected_sanity"
    (nothing gets pushed at all).
    """
    if not has_production:
        return "bootstrap", PRODUCTION_SLOT
    if not has_production_quantiles:
        if sanity_passed is False:
            return SCHEMA_DRIFT_REJECTED, None
        return "no_production_quantiles_push_as_candidate", CANDIDATE_SLOT
    if not quantile_schema_matches:
        if sanity_passed is False:
            return SCHEMA_DRIFT_REJECTED, None
        return "schema_drift_push_as_candidate", CANDIDATE_SLOT
    if not candidate_coverage_ok:
        return "rejected_miscalibrated", None

    improvement = (production_pinball - candidate_pinball) / production_pinball
    if improvement >= min_improvement:
        return "pushed_to_candidate", CANDIDATE_SLOT
    return "rejected", None


def retrain_symbol_quantiles(
    symbol: str,
    btc_candles: pd.DataFrame | None,
    staging_dir: Path,
    holdout_days: float,
    min_improvement: float,
) -> None:
    # One MLflow run per symbol per corridor retrain, experiment
    # "retrain-quantile". Silent no-op unless PINANCE_MLFLOW_TRACKING_URI
    # is set (pinance_ml.tracking) -- everything below is unchanged either
    # way. Mirrors auto_retrain.py's own "retrain-point" wrapper.
    with mlflow_run("retrain-quantile", run_name=symbol, tags={"kind": "quantile", "symbol": symbol}):
        log(f"{symbol}: loading candles + building features")
        candles = load_candles(symbol)
        dataset = build_dataset(candles, btc_candles=btc_candles)
        feat_cols = feature_columns(dataset)

        holdout_start = dataset["ts"].iloc[-1] - pd.Timedelta(days=holdout_days)
        train_data = dataset[dataset["ts"] < holdout_start]
        eval_data = dataset[dataset["ts"] >= holdout_start]
        log(f"{symbol}: {len(train_data)} train rows, {len(eval_data)} held-out eval rows ({holdout_days:.0f}d)")

        log(f"{symbol}: training quantile-corridor eval candidate {CORRIDOR_QUANTILES} (held-out data excluded)")
        train_t0 = time.time()
        eval_candidate = train_quantile_models(train_data, feat_cols, quantiles=CORRIDOR_QUANTILES)
        train_wall_seconds = time.time() - train_t0
        cand_frame = _quantile_eval_frame(
            {k: m.booster_ for k, m in eval_candidate.items()}, feat_cols, eval_data, CORRIDOR_COVERAGE_TOLERANCE
        )
        candidate_pinball = float(cand_frame["pinball"].mean()) if not cand_frame.empty else float("nan")
        candidate_coverage_ok = bool(not cand_frame.empty and cand_frame["within_tolerance"].mean() > 0.5)
        log(f"{symbol}: candidate  avg_pinball_loss={candidate_pinball:.6f} coverage_ok={candidate_coverage_ok}")

        prod_dir = staging_dir / symbol / "production_download_quantiles"
        if prod_dir.exists():
            shutil.rmtree(prod_dir)
        prod_metadata = download_model_set(symbol, PRODUCTION_SLOT, prod_dir)

        has_production = prod_metadata is not None
        has_production_quantiles = has_production and sorted(prod_metadata.get("quantile_levels") or []) == sorted(
            CORRIDOR_QUANTILES
        )
        quantile_schema_matches = has_production_quantiles and prod_metadata.get(
            "quantile_schema_version"
        ) == _schema_version(feat_cols)
        production_pinball = None
        prod_frame = None
        eval_summary = {"candidate_avg_pinball_loss": candidate_pinball, "candidate_coverage_ok": candidate_coverage_ok}

        if has_production_quantiles and not quantile_schema_matches:
            log(
                f"{symbol}: production's corridor feature schema doesn't match this repo's current pipeline "
                f"(quantile_schema_version {prod_metadata.get('quantile_schema_version')} vs {_schema_version(feat_cols)}) -- "
                "can't do a like-for-like comparison; treating as if no comparable production corridor exists"
            )
            eval_summary["production_quantile_schema_version"] = prod_metadata.get("quantile_schema_version")
        elif has_production_quantiles:
            production_quantile_boosters = _load_production_quantile_boosters(prod_metadata, prod_dir, CORRIDOR_QUANTILES)
            prod_frame = _quantile_eval_frame(
                production_quantile_boosters, feat_cols, eval_data, CORRIDOR_COVERAGE_TOLERANCE
            )
            production_pinball = float(prod_frame["pinball"].mean()) if not prod_frame.empty else float("nan")
            log(
                f"{symbol}: production avg_pinball_loss={production_pinball:.6f} "
                f"(quantile_model_version={prod_metadata.get('quantile_model_version')})"
            )
            eval_summary.update(
                production_avg_pinball_loss=production_pinball,
                production_quantile_model_version=prod_metadata.get("quantile_model_version"),
                min_improvement_required=min_improvement,
            )

        # The schema the corridor would REPLACE: production's own corridor
        # schema if it has one, else production's point-model schema (a
        # corridor on new features can't sit on top of an old-feature point
        # model any more than the point model can). Differs from this
        # repo's current schema -> this push is a "new scheme".
        current_schema = _schema_version(feat_cols)
        reference_schema = (
            prod_metadata.get("quantile_schema_version") if has_production_quantiles
            else (prod_metadata.get("schema_version") if has_production else None)
        )
        new_scheme = has_production and reference_schema != current_schema
        sanity_passed = None
        if new_scheme:
            naive_pinball = naive_quantile_pinball(
                train_data, eval_data, sorted({h for h, _ in eval_candidate}), CORRIDOR_QUANTILES
            )
            sanity_passed, sanity_reason = quantile_sanity_check(
                candidate_pinball, naive_pinball, candidate_coverage_ok, SANITY_MAX_PINBALL_RATIO
            )
            eval_summary["sanity"] = {
                "passed": sanity_passed,
                "reason": sanity_reason,
                "naive_avg_pinball_loss": naive_pinball,
                "max_pinball_ratio": SANITY_MAX_PINBALL_RATIO,
            }
            log(
                f"{symbol}: sanity vs naive: naive_pinball={naive_pinball:.6f} "
                f"-> {'PASS' if sanity_passed else 'FAIL'} ({sanity_reason})"
            )

        decision, target_slot = decide_quantile_target_slot(
            has_production,
            quantile_schema_matches,
            has_production_quantiles,
            candidate_pinball,
            production_pinball,
            candidate_coverage_ok,
            min_improvement,
            sanity_passed,
        )
        eval_summary["decision"] = decision
        log(f"{symbol}: decision={decision}" + (f" -> slot '{target_slot}'" if target_slot else " -- nothing pushed"))

        improvement = (production_pinball - candidate_pinball) / production_pinball if production_pinball else None
        log_params(
            {
                "symbol": symbol,
                "holdout_days": holdout_days,
                "min_improvement": min_improvement,
                "quantile_schema_version": _schema_version(feat_cols),
                "quantile_levels": ",".join(str(q) for q in sorted(CORRIDOR_QUANTILES)),
                "n_train_rows": len(train_data),
                "n_eval_rows": len(eval_data),
                "n_features": len(feat_cols),
                "production_quantile_model_version": prod_metadata.get("quantile_model_version")
                if has_production_quantiles
                else None,
            }
        )
        run_metrics = {
            "candidate_avg_pinball_loss": candidate_pinball,
            "production_avg_pinball_loss": production_pinball,
            "candidate_within_tol_rate": float(cand_frame["within_tolerance"].mean()) if not cand_frame.empty else None,
            "production_within_tol_rate": float(prod_frame["within_tolerance"].mean())
            if prod_frame is not None and not prod_frame.empty
            else None,
            "improvement": improvement,
            "train_wall_seconds": train_wall_seconds,
        }
        for q in sorted(CORRIDOR_QUANTILES):
            qi = int(round(q * 100))
            sub = cand_frame[cand_frame["quantile"] == q] if not cand_frame.empty else cand_frame
            if not sub.empty:
                run_metrics[f"candidate_coverage_q{qi}"] = float(sub["coverage"].mean())
            if prod_frame is not None and not prod_frame.empty:
                psub = prod_frame[prod_frame["quantile"] == q]
                if not psub.empty:
                    run_metrics[f"production_coverage_q{qi}"] = float(psub["coverage"].mean())
        log_metrics(run_metrics)
        set_tags({"decision": decision, "target_slot": target_slot or "none"})

        (staging_dir / symbol).mkdir(parents=True, exist_ok=True)
        per_hq = cand_frame.rename(columns={"pinball": "candidate_pinball", "coverage": "candidate_coverage"})
        if prod_frame is not None and not prod_frame.empty:
            per_hq = per_hq.merge(
                prod_frame.rename(
                    columns={"pinball": "production_pinball", "coverage": "production_coverage"}
                ).drop(columns=["n", "within_tolerance"]),
                on=["horizon", "quantile"],
                how="left",
            )
        per_hq_path = staging_dir / symbol / "per_horizon_quantile_metrics.csv"
        per_hq.to_csv(per_hq_path, index=False)
        log_artifact(per_hq_path)

        if target_slot is None:
            _write_eval_log(staging_dir, symbol, eval_summary)
            log_dict(eval_summary, "quantile_eval_metrics.json")
            post_retrain_event(
                symbol, "quantile", "rejected",
                production_version=prod_metadata.get("quantile_model_version") if has_production_quantiles else None,
                metric_name=f"avg_pinball_loss ({decision})",
                candidate_value=candidate_pinball, production_value=production_pinball,
                threshold=min_improvement, n_samples=len(eval_data),
                train_wall_seconds=train_wall_seconds,
            )
            return

        log(f"{symbol}: training final quantile-corridor models on the full history (holdout window included)")
        final_dir = staging_dir / symbol / "final_quantiles"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        # `dataset` (built above, before the train/eval split) already covers
        # the full history including the holdout window -- no need to reload.
        quantile_metadata = export_symbol_quantiles(
            symbol, dataset, feat_cols, final_dir, start=None, quantiles=CORRIDOR_QUANTILES
        )

        # Merge onto whatever's already in the target slot -- re-fetch rather
        # than reuse prod_metadata: the target slot might be `candidate`,
        # whose current contents (if any) weren't downloaded above, only
        # production's were (for the gate comparison). See module docstring.
        existing_metadata = download_metadata(symbol, target_slot)
        # Explicit, always-written (True AND False, never absent) -- see
        # auto_retrain.py's `new_scheme`. quantile_-prefixed so it travels
        # with the corridor half of the metadata (disjoint-keys contract).
        is_new_scheme = bool(new_scheme and decision in _UNCOMPARED_PUSH_DECISIONS)
        quantile_metadata["quantile_new_scheme"] = is_new_scheme
        quantile_metadata["quantile_replaces_schema_version"] = reference_schema if is_new_scheme else None
        merged_metadata = merge_metadata(existing_metadata, quantile_metadata)
        merged_metadata["quantile_eval_metrics"] = eval_summary
        (final_dir / symbol / "metadata.json").write_text(json.dumps(merged_metadata, indent=2))

        upload_model_set(symbol, target_slot, final_dir / symbol)
        log(
            f"{symbol}: uploaded -> MinIO slot '{target_slot}' "
            f"(quantile_model_version={merged_metadata['quantile_model_version']})"
        )
        _write_eval_log(staging_dir, symbol, eval_summary)

        set_tags({"quantile_model_version": merged_metadata["quantile_model_version"]})
        log_dict(eval_summary, "quantile_eval_metrics.json")
        log_artifact(final_dir / symbol / "metadata.json")

        post_retrain_event(
            symbol, "quantile", "promoted" if decision in _ADVANCING_DECISIONS else "rejected",
            candidate_version=merged_metadata["quantile_model_version"],
            production_version=prod_metadata.get("quantile_model_version") if has_production_quantiles else None,
            metric_name=f"avg_pinball_loss ({decision})",
            candidate_value=candidate_pinball, production_value=production_pinball,
            threshold=min_improvement, n_samples=len(eval_data),
            train_wall_seconds=train_wall_seconds,
        )


def _write_eval_log(staging_dir: Path, symbol: str, eval_summary: dict) -> None:
    log_path = staging_dir / "auto_retrain_quantiles_log.jsonl"
    entry = {"symbol": symbol, "retrained_at": pd.Timestamp.now("UTC").isoformat(), **eval_summary}
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="*", help="Symbols to retrain (default: all in DB)")
    parser.add_argument("--holdout-days", type=float, default=AUTO_RETRAIN_HOLDOUT_DAYS)
    parser.add_argument("--min-improvement", type=float, default=AUTO_RETRAIN_MIN_IMPROVEMENT)
    parser.add_argument("--staging-dir", default=".auto_retrain_staging")
    args = parser.parse_args()

    symbols = args.symbols or list_symbols()
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    log(f"Symbols ({len(symbols)}): {symbols}")
    log(f"holdout_days={args.holdout_days}, min_improvement={args.min_improvement:.1%}")
    _ = _source_commit()  # fail fast if this isn't a git checkout, same as export_models.py would at push time

    btc_candles = load_candles("BTCUSDT") if any(s != "BTCUSDT" for s in symbols) else None

    run_t0 = time.time()
    for i, symbol in enumerate(symbols, start=1):
        log(f"[{i}/{len(symbols)}] {symbol}")
        try:
            retrain_symbol_quantiles(
                symbol,
                None if symbol == "BTCUSDT" else btc_candles,
                staging_dir,
                args.holdout_days,
                args.min_improvement,
            )
        except Exception:
            log(f"{symbol}: FAILED -- see traceback below, continuing with remaining symbols")
            import traceback

            traceback.print_exc()

    log(f"All done. Total run time: {(time.time() - run_t0) / 60:.1f}min")


if __name__ == "__main__":
    main()
