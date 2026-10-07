"""Auto-retrain (README "Ретрейн регрессора"): daily/scheduled retrain of
the LightGBM POINT regressor (12 horizons), gated against current
production before ever touching MinIO.

Per symbol:
  1. Build features on the full available history, split off a recent
     AUTO_RETRAIN_HOLDOUT_DAYS-wide window the candidate's own training
     never sees.
  2. Train an "eval candidate" on everything before that window.
  3. Download current production's model set from MinIO (if any).
  4. Evaluate both on the same held-out window -- average MAE across
     HORIZONS, per README's own comparison metric.
  5. Decide:
       - no production yet             -> bootstrap: train on the FULL
                                           history, push straight to the
                                           production slot.
       - candidate beats production by
         >= AUTO_RETRAIN_MIN_IMPROVEMENT -> train a final model on the
                                            FULL history (the holdout
                                            split above was only for the
                                            gate, not for what actually
                                            ships) and push to the
                                            candidate slot -- NOT
                                            production. A separate,
                                            not-yet-built promotion step
                                            (predictor-backend's live
                                            shadow metrics, see project
                                            memory) decides if/when a
                                            candidate that's been running
                                            in shadow actually replaces
                                            production.
       - otherwise                     -> log and stop, nothing pushed.
  6. Schema drift (production's recorded feature_columns no longer a
     match for what this repo's feature pipeline computes today) is
     treated like "no comparable production exists" rather than crashing
     or silently comparing incompatible feature sets. The candidate must
     first clear a naive-baseline sanity gate (pinance_ml.sanity: avg MAE
     within SANITY_MAX_MAE_RATIO of the naive r_h=0 forecast and
     directional accuracy not below the majority-class base rate) -- fail
     it and nothing is pushed ("schema_drift_rejected_sanity"). Pass it and
     the candidate is pushed to the candidate slot (not production) with
     metadata `new_scheme: true`; promotion is then an explicit step,
     scripts/promote_new_scheme.py (promote_if_better.py can't compare
     across schemas and flags it as needing manual promotion). Every push
     writes `new_scheme` explicitly (false outside this path).

This script only ever touches the point models (h{h}.txt) and their own
metadata fields. The confidence corridor (CORRIDOR_QUANTILES tail models)
is a separate model with its own promotion criteria (pinball loss +
coverage, not MAE) and its own natural refresh cadence -- retrained,
gated, and pushed independently by scripts/auto_retrain_quantiles.py, on
its own schedule. Bundling them was tried first and reverted: a candidate
whose point model happened to be marginally worse than production (pure
retrain noise, unrelated to corridor quality) would block an improving
corridor from ever shipping, since both were pushed as one atomic unit --
project notes. Every push here
merges its point-only fields onto whatever's already in the target slot
(model_storage.merge_metadata) rather than overwriting metadata.json
wholesale, so this never clobbers an independently-promoted corridor's
quantile_* fields -- auto_retrain_quantiles.py does the same in reverse.

Usage:
    python scripts/auto_retrain.py [SYMBOL ...]
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
from export_models import _schema_version, _source_commit, export_symbol_point  # noqa: E402

from pinance_ml.backend_client import post_retrain_event
from pinance_ml.config import (
    AUTO_RETRAIN_HOLDOUT_DAYS,
    AUTO_RETRAIN_MIN_IMPROVEMENT,
    SANITY_MAX_MAE_RATIO,
    SANITY_MIN_DIR_ACC_EDGE,
)
from pinance_ml.data.db import list_symbols, load_candles
from pinance_ml.dataset import build_dataset, feature_columns
from pinance_ml.metrics import directional_accuracy, mae
from pinance_ml.model_storage import (
    CANDIDATE_SLOT,
    PRODUCTION_SLOT,
    download_metadata,
    download_model_set,
    merge_metadata,
    upload_model_set,
)
from pinance_ml.models.lightgbm_model import predict_horizons, train_horizon_models
from pinance_ml.sanity import naive_point_baseline, point_sanity_check
from pinance_ml.tracking import log_artifact, log_dict, log_metrics, log_params, mlflow_run, set_tags

# decide_target_slot's non-"rejected" labels (bootstrap/pushed_to_candidate/
# schema_drift_push_as_candidate) all mean a point model actually landed in
# MinIO -- predictor-backend's RetrainEventIn.decision only has room for
# "promoted"/"rejected" (see backend_client.post_retrain_event's own
# docstring), so every advancing label maps to "promoted" there; the
# original label is still preserved via metric_name below.
_ADVANCING_DECISIONS = {"bootstrap", "pushed_to_candidate", "schema_drift_push_as_candidate"}

# Decision label for a schema change whose candidate failed the naive-
# baseline sanity gate -- not in _ADVANCING_DECISIONS, nothing is pushed.
SCHEMA_DRIFT_REJECTED = "schema_drift_rejected_sanity"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _eval_frame(models: dict[int, lgb.Booster], feat_cols: list[str], eval_data: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon MAE + directional accuracy on `eval_data`. Rows with a
    NaN target for a given horizon are excluded from that horizon's score,
    same convention as train_horizon_models/predict_horizons.
    _avg_mae_and_dir_acc reduces this to its two means; the retrain run
    also logs the whole frame as an artifact."""
    preds = predict_horizons(models, eval_data, feat_cols)
    rows = []
    for h in models:
        actual = eval_data[f"r_{h}"].to_numpy()
        predicted = preds[f"r_{h}_pred"].to_numpy()
        valid = pd.notna(actual)
        if valid.sum() == 0:
            continue
        rows.append(
            {
                "horizon": h,
                "mae": mae(actual[valid], predicted[valid]),
                "directional_accuracy": directional_accuracy(actual[valid], predicted[valid]),
                "n": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def _avg_mae_and_dir_acc(models: dict[int, lgb.Booster], feat_cols: list[str], eval_data: pd.DataFrame) -> tuple[float, float]:
    """Mean MAE and directional accuracy across HORIZONS on `eval_data` --
    README's own comparison metric ("сравнение с production по среднему
    MAE через горизонты")."""
    frame = _eval_frame(models, feat_cols, eval_data)
    return float(frame["mae"].mean()), float(frame["directional_accuracy"].mean())


def _feature_importance_frame(model_dir: Path, horizons: list[int]) -> pd.DataFrame:
    """Per-(horizon, feature) split gain of the just-exported final point
    models, read back from their saved .txt files -- attached to the
    retrain run so a promoted candidate's importances are inspectable
    without re-running anything."""
    rows = []
    for h in horizons:
        booster = lgb.Booster(model_file=str(model_dir / f"h{h}.txt"))
        for name, gain in zip(booster.feature_name(), booster.feature_importance("gain")):
            rows.append({"horizon": h, "feature": name, "gain": float(gain)})
    return pd.DataFrame(rows)


def _load_production_boosters(metadata: dict, model_dir: Path) -> dict[int, lgb.Booster]:
    return {h: lgb.Booster(model_file=str(model_dir / f"h{h}.txt")) for h in metadata["horizons"]}


def decide_target_slot(
    has_production: bool,
    schema_matches: bool,
    candidate_mae: float,
    production_mae: float | None,
    min_improvement: float,
    sanity_passed: bool | None = None,
) -> tuple[str, str | None]:
    """Pure push-gate decision, factored out of retrain_symbol so it's
    testable without a DB/MinIO/any actual model training.

    `sanity_passed` only matters on the schema-drift path (see the
    docstring's step 6): there's no production to compare against, so the
    candidate must at least clear the naive-baseline sanity gate
    (pinance_ml.sanity.point_sanity_check) or nothing is pushed. None
    (default) means "no sanity verdict supplied" and keeps the old
    behavior, so existing callers/tests are unaffected.

    Returns (decision_label, target_slot) -- target_slot is None for
    "rejected" (nothing gets pushed at all).
    """
    if not has_production:
        return "bootstrap", PRODUCTION_SLOT
    if not schema_matches:
        if sanity_passed is False:
            return SCHEMA_DRIFT_REJECTED, None
        return "schema_drift_push_as_candidate", CANDIDATE_SLOT

    improvement = (production_mae - candidate_mae) / production_mae
    if improvement >= min_improvement:
        return "pushed_to_candidate", CANDIDATE_SLOT
    return "rejected", None


def retrain_symbol(
    symbol: str,
    btc_candles: pd.DataFrame | None,
    staging_dir: Path,
    holdout_days: float,
    min_improvement: float,
) -> None:
    # One MLflow run per symbol per retrain, experiment "retrain-point".
    # A silent no-op unless PINANCE_MLFLOW_TRACKING_URI is set (see
    # pinance_ml.tracking) -- everything below runs identically with or
    # without it.
    with mlflow_run("retrain-point", run_name=symbol, tags={"kind": "point", "symbol": symbol}):
        log(f"{symbol}: loading candles + building features")
        candles = load_candles(symbol)
        dataset = build_dataset(candles, btc_candles=btc_candles)
        feat_cols = feature_columns(dataset)

        holdout_start = dataset["ts"].iloc[-1] - pd.Timedelta(days=holdout_days)
        train_data = dataset[dataset["ts"] < holdout_start]
        eval_data = dataset[dataset["ts"] >= holdout_start]
        log(f"{symbol}: {len(train_data)} train rows, {len(eval_data)} held-out eval rows ({holdout_days:.0f}d)")

        log(f"{symbol}: training eval candidate (held-out data excluded)")
        train_t0 = time.time()
        eval_candidate = train_horizon_models(train_data, feat_cols)
        train_wall_seconds = time.time() - train_t0
        cand_frame = _eval_frame({h: m.booster_ for h, m in eval_candidate.items()}, feat_cols, eval_data)
        candidate_mae = float(cand_frame["mae"].mean())
        candidate_acc = float(cand_frame["directional_accuracy"].mean())
        log(f"{symbol}: candidate  avg_mae={candidate_mae:.6f} avg_dir_acc={candidate_acc:.4f}")

        prod_dir = staging_dir / symbol / "production_download"
        if prod_dir.exists():
            shutil.rmtree(prod_dir)
        prod_metadata = download_model_set(symbol, PRODUCTION_SLOT, prod_dir)

        has_production = prod_metadata is not None
        schema_matches = has_production and prod_metadata["feature_columns"] == feat_cols
        production_mae = production_acc = None
        prod_frame = None
        eval_summary = {"candidate_avg_mae": candidate_mae, "candidate_avg_dir_acc": candidate_acc}

        if has_production and not schema_matches:
            log(
                f"{symbol}: production's feature schema doesn't match this repo's current pipeline "
                f"(schema_version {prod_metadata.get('schema_version')} vs {_schema_version(feat_cols)}) -- "
                "can't do a like-for-like comparison; treating as if no comparable production exists"
            )
            eval_summary["production_schema_version"] = prod_metadata.get("schema_version")
        elif has_production:
            production_boosters = _load_production_boosters(prod_metadata, prod_dir)
            prod_frame = _eval_frame(production_boosters, feat_cols, eval_data)
            production_mae = float(prod_frame["mae"].mean())
            production_acc = float(prod_frame["directional_accuracy"].mean())
            log(
                f"{symbol}: production avg_mae={production_mae:.6f} avg_dir_acc={production_acc:.4f} "
                f"(model_version={prod_metadata.get('model_version')})"
            )
            eval_summary.update(
                production_avg_mae=production_mae,
                production_avg_dir_acc=production_acc,
                production_model_version=prod_metadata.get("model_version"),
                min_improvement_required=min_improvement,
            )

        schema_drift = has_production and not schema_matches
        sanity_passed = None
        if schema_drift:
            horizons = sorted(eval_candidate.keys())
            naive_avg_mae, naive_avg_acc = naive_point_baseline(train_data, eval_data, horizons)
            sanity_passed, sanity_reason = point_sanity_check(
                candidate_mae, candidate_acc, naive_avg_mae, naive_avg_acc, SANITY_MAX_MAE_RATIO, SANITY_MIN_DIR_ACC_EDGE
            )
            eval_summary["sanity"] = {
                "passed": sanity_passed,
                "reason": sanity_reason,
                "naive_avg_mae": naive_avg_mae,
                "naive_avg_dir_acc": naive_avg_acc,
                "max_mae_ratio": SANITY_MAX_MAE_RATIO,
                "min_dir_acc_edge": SANITY_MIN_DIR_ACC_EDGE,
            }
            log(
                f"{symbol}: sanity vs naive: naive_mae={naive_avg_mae:.6f} naive_dir_acc={naive_avg_acc:.4f} "
                f"-> {'PASS' if sanity_passed else 'FAIL'} ({sanity_reason})"
            )

        decision, target_slot = decide_target_slot(
            has_production, schema_matches, candidate_mae, production_mae, min_improvement, sanity_passed
        )
        eval_summary["decision"] = decision
        log(f"{symbol}: decision={decision}" + (f" -> slot '{target_slot}'" if target_slot else " -- nothing pushed"))

        improvement = (production_mae - candidate_mae) / production_mae if production_mae else None
        log_params(
            {
                "symbol": symbol,
                "holdout_days": holdout_days,
                "min_improvement": min_improvement,
                "schema_version": _schema_version(feat_cols),
                "n_train_rows": len(train_data),
                "n_eval_rows": len(eval_data),
                "n_features": len(feat_cols),
                "production_model_version": prod_metadata.get("model_version") if has_production else None,
            }
        )
        log_metrics(
            {
                "candidate_avg_mae": candidate_mae,
                "candidate_avg_dir_acc": candidate_acc,
                "production_avg_mae": production_mae,
                "production_avg_dir_acc": production_acc,
                "improvement": improvement,
                "train_wall_seconds": train_wall_seconds,
            }
        )
        set_tags({"decision": decision, "target_slot": target_slot or "none"})

        (staging_dir / symbol).mkdir(parents=True, exist_ok=True)
        per_horizon = cand_frame.rename(columns={"mae": "candidate_mae", "directional_accuracy": "candidate_dir_acc"})
        if prod_frame is not None:
            per_horizon = per_horizon.merge(
                prod_frame.rename(columns={"mae": "production_mae", "directional_accuracy": "production_dir_acc"}).drop(
                    columns="n"
                ),
                on="horizon",
                how="left",
            )
        per_horizon_path = staging_dir / symbol / "per_horizon_metrics.csv"
        per_horizon.to_csv(per_horizon_path, index=False)
        log_artifact(per_horizon_path)

        if target_slot is None:
            _write_eval_log(staging_dir, symbol, eval_summary)
            log_dict(eval_summary, "eval_metrics.json")
            post_retrain_event(
                symbol, "point", "rejected",
                production_version=prod_metadata.get("model_version") if has_production else None,
                metric_name=f"avg_mae ({decision})",
                candidate_value=candidate_mae, production_value=production_mae,
                threshold=min_improvement, n_samples=len(eval_data),
                train_wall_seconds=train_wall_seconds,
            )
            return

        log(f"{symbol}: training final point model on the full history (holdout window included)")
        final_dir = staging_dir / symbol / "final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        # `dataset` (built above, before the train/eval split) already covers
        # the full history including the holdout window -- no need to reload.
        point_metadata = export_symbol_point(symbol, dataset, feat_cols, final_dir, start=None)

        # Merge onto whatever's already in the target slot -- re-fetch rather
        # than reuse prod_metadata: the target slot might be `candidate`,
        # whose current contents (if any) weren't downloaded above, only
        # production's were (for the gate comparison). See module docstring.
        existing_metadata = download_metadata(symbol, target_slot)
        # Explicit, always-written label (True AND False -- never absent) so
        # scripts/promote_new_scheme.py / promote_if_better.py don't have to
        # infer "did the schema change" from missing keys. Only the
        # schema-drift path marks a candidate as a new scheme.
        point_metadata["new_scheme"] = decision == "schema_drift_push_as_candidate"
        point_metadata["replaces_schema_version"] = prod_metadata.get("schema_version") if schema_drift else None
        merged_metadata = merge_metadata(existing_metadata, point_metadata)
        merged_metadata["eval_metrics"] = eval_summary
        (final_dir / symbol / "metadata.json").write_text(json.dumps(merged_metadata, indent=2))

        upload_model_set(symbol, target_slot, final_dir / symbol)
        log(f"{symbol}: uploaded -> MinIO slot '{target_slot}' (model_version={merged_metadata['model_version']})")
        _write_eval_log(staging_dir, symbol, eval_summary)

        set_tags({"candidate_model_version": merged_metadata["model_version"]})
        log_dict(eval_summary, "eval_metrics.json")
        log_artifact(final_dir / symbol / "metadata.json")
        try:
            fi_path = staging_dir / symbol / "feature_importance.csv"
            _feature_importance_frame(final_dir / symbol, point_metadata["horizons"]).to_csv(fi_path, index=False)
            log_artifact(fi_path)
        except Exception:
            log(f"{symbol}: WARNING -- feature-importance artifact failed, continuing")

        post_retrain_event(
            symbol, "point", "promoted" if decision in _ADVANCING_DECISIONS else "rejected",
            candidate_version=merged_metadata["model_version"],
            production_version=prod_metadata.get("model_version") if has_production else None,
            metric_name=f"avg_mae ({decision})",
            candidate_value=candidate_mae, production_value=production_mae,
            threshold=min_improvement, n_samples=len(eval_data),
            train_wall_seconds=train_wall_seconds,
        )


def _write_eval_log(staging_dir: Path, symbol: str, eval_summary: dict) -> None:
    log_path = staging_dir / "auto_retrain_log.jsonl"
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
            retrain_symbol(
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
