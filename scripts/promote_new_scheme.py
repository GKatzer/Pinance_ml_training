"""Explicit promotion of candidates built on a CHANGED feature schema.

Why this exists: scripts/promote_if_better.py decides by comparing live
metrics of candidate vs production, which is meaningless across a feature
schema change (different inputs; predictor-ml-inference also rejects the
old production model once the new schema is live). So for such candidates
promote_if_better.py only reports "new_scheme_manual_promotion_required".
This script is that manual step -- DRY-RUN BY DEFAULT, nothing is moved
without --apply.

A point and/or corridor candidate is promoted only if ALL of these hold
(decide_new_scheme_promotion, pure and unit-tested):
  - the candidate slot has it, and so does production (no production ->
    auto_retrain.py's bootstrap handles that, out of scope);
  - it carries the explicit `new_scheme` / `quantile_new_scheme: true`
    label written by auto_retrain*.py (never inferred from a missing key);
  - its schema_version really differs from production's;
  - it recorded a PASSED naive-baseline sanity verdict
    (eval_metrics["sanity"] / quantile_eval_metrics["sanity"]);
  - the candidate's files match what its metadata.json promises
    (model_storage.slot_consistency_problems).

Point and corridor are independent decisions, like in promote_if_better.py,
but a combined run takes ONE snapshot of production into `previous` first,
so `--rollback` restores the true pre-promotion state. After applying, the
whole production slot is re-checked for consistency (including "point and
corridor disagree about the schema" -- promote both halves together, or
retrain the missing one first).

Known gap, on purpose not hidden: nothing here recomputes the CURRENT
pipeline's schema (that needs the DB and a full feature build). Run this
from a checkout whose feature code matches the one that trained the
candidate; the runbook (deploy/promote_new_scheme/README.md) says so.

Legacy candidates (pushed before the labels / sanity verdict existed) have
neither. Explicit, audited escape hatches, both off by default:
  --label-legacy        write `new_scheme` / `quantile_new_scheme: true`
                        onto an UNLABELED candidate whose schema differs
                        from production's (then re-run to promote);
  --skip-sanity-check   promote without a recorded sanity verdict (the
                        override is written to the audit log).

Usage:
    python scripts/promote_new_scheme.py [SYMBOL ...]                 # dry-run (default)
    python scripts/promote_new_scheme.py [SYMBOL ...] --apply
    python scripts/promote_new_scheme.py SYMBOL [...] --rollback --apply
    python scripts/promote_new_scheme.py [SYMBOL ...] --label-legacy [--apply]
        [--skip-sanity-check] [--staging-dir .auto_retrain_staging]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.backend_client import post_retrain_event
from pinance_ml.data.db import list_symbols
from pinance_ml.model_storage import (
    CANDIDATE_SLOT,
    PREVIOUS_SLOT,
    PRODUCTION_SLOT,
    _put_metadata,
    check_slot_consistency,
    download_metadata,
    promote_candidate_point,
    promote_candidate_quantiles,
    rollback_production,
    snapshot_production,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def decide_new_scheme_promotion(
    has_candidate: bool,
    has_production: bool,
    label: bool | None,
    candidate_schema: str | None,
    production_schema: str | None,
    sanity_passed: bool | None,
    candidate_problems: list[str],
    skip_sanity_check: bool = False,
) -> tuple[str, bool]:
    """Pure decision for ONE pipeline half (point or corridor) -- the
    caller passes that half's own label/schemas/sanity/problems.
    Returns (decision_label, should_promote). Order matters only for which
    reason gets reported first; every guard must hold to promote."""
    if not has_candidate:
        return "no_candidate", False
    if not has_production:
        return "no_production", False
    if label is None:
        # Unlabeled AND the schema didn't change: nothing to promote here
        # (e.g. an ordinary corridor refresh), not a labeling problem.
        if candidate_schema is not None and candidate_schema == production_schema:
            return "schema_unchanged", False
        return "label_missing", False
    if label is not True:
        return "not_new_scheme", False
    if candidate_schema is None or candidate_schema == production_schema:
        return "schema_unchanged", False
    if candidate_problems:
        return "candidate_inconsistent", False
    if sanity_passed is None and not skip_sanity_check:
        return "sanity_missing", False
    if sanity_passed is False:
        # An override can't paper over a recorded FAIL -- only a missing verdict.
        return "sanity_failed", False
    return "promote_new_scheme", True


def _sanity_verdict(candidate_meta: dict, eval_key: str) -> bool | None:
    sanity = (candidate_meta.get(eval_key) or {}).get("sanity")
    if not isinstance(sanity, dict) or "passed" not in sanity:
        return None
    return bool(sanity["passed"])


def _half_inputs(candidate_meta: dict | None, production_meta: dict | None, quantile: bool) -> dict:
    """Pick one pipeline half's label/schemas/sanity out of the two
    slots' metadata. The corridor is compared against the schema it would
    replace: production's corridor schema if it has one, else production's
    point-model schema (same reference auto_retrain_quantiles.py uses)."""
    cand, prod = candidate_meta or {}, production_meta or {}
    if quantile:
        has_half = bool(cand.get("quantile_levels"))
        return dict(
            has_candidate=has_half,
            label=cand.get("quantile_new_scheme"),
            candidate_schema=cand.get("quantile_schema_version"),
            production_schema=prod.get("quantile_schema_version") or prod.get("schema_version"),
            sanity_passed=_sanity_verdict(cand, "quantile_eval_metrics"),
        )
    return dict(
        has_candidate=candidate_meta is not None and bool(cand.get("model_version")),
        label=cand.get("new_scheme"),
        candidate_schema=cand.get("schema_version"),
        production_schema=prod.get("schema_version"),
        sanity_passed=_sanity_verdict(cand, "eval_metrics"),
    )


def _write_log(staging_dir: Path, symbol: str, entry: dict) -> None:
    log_path = staging_dir / "promote_new_scheme_log.jsonl"
    row = {"symbol": symbol, "checked_at": pd.Timestamp.now("UTC").isoformat(), **entry}
    with log_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def label_legacy_symbol(symbol: str, apply: bool) -> dict:
    """Write the explicit new_scheme labels onto an unlabeled candidate
    whose schema differs from production's. Returns what was (or would be)
    labeled, for the audit log."""
    candidate_meta = download_metadata(symbol, CANDIDATE_SLOT)
    production_meta = download_metadata(symbol, PRODUCTION_SLOT)
    labeled = {}
    if candidate_meta is None or production_meta is None:
        log(f"{symbol}: --label-legacy needs both candidate and production, skipping")
        return labeled
    for quantile, key in ((False, "new_scheme"), (True, "quantile_new_scheme")):
        inputs = _half_inputs(candidate_meta, production_meta, quantile)
        if not inputs["has_candidate"] or inputs["label"] is not None:
            continue
        if inputs["candidate_schema"] and inputs["candidate_schema"] != inputs["production_schema"]:
            candidate_meta[key] = True
            candidate_meta[("quantile_" if quantile else "") + "replaces_schema_version"] = inputs["production_schema"]
            labeled[key] = True
            log(f"{symbol}: {'LABEL' if apply else 'would label'} {key}=true (schema {inputs['production_schema']} -> {inputs['candidate_schema']})")
    if labeled and apply:
        _put_metadata(symbol, CANDIDATE_SLOT, candidate_meta)
    return labeled


def promote_symbol(symbol: str, apply: bool, skip_sanity_check: bool, staging_dir: Path) -> None:
    candidate_meta = download_metadata(symbol, CANDIDATE_SLOT)
    production_meta = download_metadata(symbol, PRODUCTION_SLOT)
    has_production = production_meta is not None

    point_in = _half_inputs(candidate_meta, production_meta, quantile=False)
    quant_in = _half_inputs(candidate_meta, production_meta, quantile=True)
    point_problems = (
        check_slot_consistency(symbol, CANDIDATE_SLOT, point=True, quantiles=False) if point_in["has_candidate"] else []
    )
    quant_problems = (
        check_slot_consistency(symbol, CANDIDATE_SLOT, point=False, quantiles=True) if quant_in["has_candidate"] else []
    )

    point_decision, promote_point = decide_new_scheme_promotion(
        has_production=has_production, candidate_problems=point_problems, skip_sanity_check=skip_sanity_check, **point_in
    )
    quant_decision, promote_quant = decide_new_scheme_promotion(
        has_production=has_production, candidate_problems=quant_problems, skip_sanity_check=skip_sanity_check, **quant_in
    )
    log(
        f"{symbol}: point={point_decision}{' -> PROMOTE' if promote_point else ''} | "
        f"quantile={quant_decision}{' -> PROMOTE' if promote_quant else ''}"
        + ("" if apply else "   [dry-run]")
    )
    for problem in point_problems + quant_problems:
        log(f"{symbol}:   candidate problem: {problem}")

    applied = {"point": False, "quantile": False}
    post_problems: list[str] = []
    if apply and (promote_point or promote_quant):
        if snapshot_production(symbol):
            log(f"{symbol}: production snapshotted -> '{PREVIOUS_SLOT}'")
        # snapshot=False: ONE snapshot above covers the whole combined run.
        if promote_point:
            promote_candidate_point(symbol, snapshot=False)
            applied["point"] = True
            log(f"{symbol}: promoted point model -> production (model_version={candidate_meta['model_version']})")
        if promote_quant:
            promote_candidate_quantiles(symbol, snapshot=False)
            applied["quantile"] = True
            log(
                f"{symbol}: promoted corridor -> production "
                f"(quantile_model_version={candidate_meta.get('quantile_model_version')})"
            )
        post_problems = check_slot_consistency(symbol, PRODUCTION_SLOT)
        if post_problems:
            log(f"{symbol}: *** WARNING *** production is inconsistent after promotion: {post_problems}")
            log(f"{symbol}: roll back with: python scripts/promote_new_scheme.py {symbol} --rollback --apply")
        for half, promoted, decision, inputs, ver_key in (
            ("point", applied["point"], point_decision, point_in, "model_version"),
            ("quantile", applied["quantile"], quant_decision, quant_in, "quantile_model_version"),
        ):
            if promoted:
                post_retrain_event(
                    symbol, half, "promoted",
                    candidate_version=(candidate_meta or {}).get(ver_key),
                    production_version=(production_meta or {}).get(ver_key),
                    metric_name=f"new_scheme ({decision})",
                )

    _write_log(
        staging_dir,
        symbol,
        {
            "point_decision": point_decision,
            "point_promoted": applied["point"],
            "quantile_decision": quant_decision,
            "quantile_promoted": applied["quantile"],
            "dry_run": not apply,
            "skip_sanity_check": skip_sanity_check,
            "candidate_model_version": (candidate_meta or {}).get("model_version"),
            "production_model_version": (production_meta or {}).get("model_version"),
            "candidate_quantile_model_version": (candidate_meta or {}).get("quantile_model_version"),
            "production_quantile_model_version": (production_meta or {}).get("quantile_model_version"),
            "candidate_schema_version": point_in["candidate_schema"],
            "production_schema_version": point_in["production_schema"],
            "candidate_problems": point_problems + quant_problems,
            "post_promotion_problems": post_problems,
        },
    )


def rollback_symbol(symbol: str, apply: bool, staging_dir: Path) -> None:
    previous = download_metadata(symbol, PREVIOUS_SLOT)
    if previous is None:
        log(f"{symbol}: no '{PREVIOUS_SLOT}' snapshot, nothing to roll back to")
        return
    problems = check_slot_consistency(symbol, PREVIOUS_SLOT)
    log(
        f"{symbol}: rollback would restore production to model_version={previous.get('model_version')} "
        f"(schema {previous.get('schema_version')}, quantiles={previous.get('quantile_model_version')})"
        + ("" if apply else "   [dry-run]")
    )
    if problems:
        log(f"{symbol}: '{PREVIOUS_SLOT}' is inconsistent, rollback refused: {problems}")
    elif apply:
        rollback_production(symbol)
        log(f"{symbol}: production restored from '{PREVIOUS_SLOT}'")
        log(
            f"{symbol}: NOTE a different feature schema than the current code "
            f"({previous.get('schema_version')}) is only usable together with the matching feature code"
        )
    _write_log(
        staging_dir,
        symbol,
        {"rollback": True, "dry_run": not apply, "rolled_back": bool(apply and not problems),
         "restored_model_version": previous.get("model_version"), "problems": problems},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="*", help="Symbols to process (default: all in DB)")
    parser.add_argument("--apply", action="store_true", help="Actually modify MinIO (default is a dry-run)")
    parser.add_argument("--rollback", action="store_true", help="Restore production from the 'previous' snapshot")
    parser.add_argument("--label-legacy", action="store_true", help="Label unlabeled candidates with a differing schema")
    parser.add_argument("--skip-sanity-check", action="store_true", help="Allow a candidate with NO recorded sanity verdict")
    parser.add_argument("--staging-dir", default=".auto_retrain_staging")
    args = parser.parse_args()

    symbols = args.symbols or list_symbols()
    if args.rollback and not args.symbols:
        parser.error("--rollback needs explicit SYMBOL(s); refusing to roll back everything by default")
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    log(f"Symbols ({len(symbols)}): {symbols}")
    log(f"mode={'APPLY' if args.apply else 'DRY-RUN'} rollback={args.rollback} label_legacy={args.label_legacy} "
        f"skip_sanity_check={args.skip_sanity_check}")

    for i, symbol in enumerate(symbols, start=1):
        log(f"[{i}/{len(symbols)}] {symbol}")
        try:
            if args.rollback:
                rollback_symbol(symbol, args.apply, staging_dir)
                continue
            if args.label_legacy:
                labeled = label_legacy_symbol(symbol, args.apply)
                _write_log(staging_dir, symbol, {"label_legacy": labeled, "dry_run": not args.apply})
            promote_symbol(symbol, args.apply, args.skip_sanity_check, staging_dir)
        except Exception:
            log(f"{symbol}: FAILED -- see traceback below, continuing with remaining symbols")
            import traceback

            traceback.print_exc()

    if not args.apply:
        log("Dry-run only -- re-run with --apply to change anything.")
    log("All done.")


if __name__ == "__main__":
    main()
