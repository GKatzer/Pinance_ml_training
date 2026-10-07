"""Promote-if-better (README "Ретрейн регрессора" / "Доверительный
коридор"): the live-metrics-driven promotion step that auto_retrain.py's
own docstring explicitly defers -- "A separate, not-yet-built promotion
step (predictor-backend's live shadow metrics) decides if/when a candidate
that's been running in shadow actually replaces production."

auto_retrain.py / auto_retrain_quantiles.py already gate their own pushes
against an OFFLINE holdout window -- "would this have done better over the
last N days if it had been serving". That's necessary but provably not
sufficient on its own (holdout noise, walk-forward artifacts) -- this
script is the SECOND, independent gate: does whatever's currently sitting
in the candidate slot also hold up on REAL live traffic, once
predictor-ml-inference has actually been shadow-serving it? The live
numbers themselves live in predictor-backend's Postgres (real predictions
joined against real realized candles) -- this script is a thin client over
its admin API, not a second source of truth.

Point model and confidence corridor are evaluated and promoted as TWO
INDEPENDENT decisions (decide_point_promotion / decide_quantile_promotion
below), each calling its own model_storage.promote_candidate_point /
promote_candidate_quantiles -- one clearing its gate never moves the
other's artifact as a side effect. This wasn't always true: see "Why two
independent decisions" below for the incident that made it so.

Per symbol:
  1. Read both slots' metadata.json from MinIO (download_metadata --
     cheap, no booster files). No candidate at all, or no production at
     all (auto_retrain.py hasn't bootstrapped it yet) -> nothing to do
     for either decision, that's out of scope here.
  2. Fetch predictor-backend's GET /admin/metrics/{symbol}/compare, and
     pick out PROMOTION_WINDOW's numbers for exactly (slot='candidate',
     model_version=<candidate's current point model_version>) and
     (slot='production', model_version=<production's current one>) --
     not just "the candidate slot" or "the production slot", since a
     daily point retrain means a wide window almost certainly straddles
     more than one past candidate version (see Pinance_backend's
     admin.py docstring on why /compare groups by the pair now).
     q10_coverage/q90_coverage now live under that response's own
     "quantiles" key, grouped by (slot, quantile_model_version)
     INDEPENDENTLY of the point grouping above (predictor-backend's
     `predictions` table tags rows with model_version and
     quantile_model_version separately -- see Pinance_backend's
     app/db/models.py Prediction docstring) -- looked up by
     candidate_quantile_version, not candidate_version. Its own row's
     `n` (candidate_quantile_n below) is a genuinely independent
     quantile-only sample count, used as decide_quantile_promotion's
     min_samples floor instead of the point row's `n` -- this used to
     be a known gap ("needs schema changes on predictor-backend's
     side"), closed once predictor-backend started grouping this way.
  3. decide_point_promotion (pure, tested): promote the point model if
     candidate clears PROMOTION_MIN_SAMPLES and its live
     directional_accuracy beats production's by
     >= PROMOTION_MIN_ACCURACY_IMPROVEMENT_PP -- candidate_point_rejected
     (was this artifact itself rejected by auto_retrain.py's own offline
     gate?) doesn't gate this at all, it only relabels an
     already-failing comparison for a clearer audit trail (see
     decide_point_promotion's own docstring for why).
     decide_quantile_promotion (pure, tested): promote the corridor if
     candidate has a quantile_model_version production doesn't already
     have (covers both "gained a corridor for the first time" and "the
     corridor refreshed to a newer version" -- see "Why two independent
     decisions"), candidate clears PROMOTION_MIN_SAMPLES, and the
     corridor's own live calibration is confirmed honest
     (corridor_verified, see below). Independent outcomes: any of
     {neither, point only, quantiles only, both} can happen in the same
     run.
  4. Whichever decision(s) promote: model_storage.promote_candidate_point
     and/or model_storage.promote_candidate_quantiles(symbol) -- each a
     same-bucket MinIO copy of only its own files + metadata keys,
     merged onto production rather than overwriting it (see
     model_storage.py's module docstring), picked up by
     predictor-ml-inference on its next poll (MODEL_POLL_INTERVAL_SECONDS).
  5. Always appends a JSON-line audit entry to
     .auto_retrain_staging/promote_if_better_log.jsonl, same convention as
     auto_retrain.py's own log, regardless of outcome.

Why two independent decisions (2026-08-03 incident, BTCUSDT/BNBUSDT):
before the auto_retrain.py / auto_retrain_quantiles.py pipeline split
(see model_storage.py's module docstring), a single combined script
could reject a point candidate on its own offline holdout ("decision":
"rejected" in eval_metrics: candidate dir_acc 52.6%/52.1% vs production's
58.8%/59.5%) and *still* push that same rejected point model to the
candidate slot moments later, bundled as part of an approved quantile
push. This script's ORIGINAL design (a single decide_promotion, and
model_storage.promote_candidate copying the whole slot) reproduced that
same coupling one level up: a "candidate gained a corridor" promotion
copied whatever point model happened to be sitting in candidate right
along with it, whether or not that point model's own live accuracy
justified moving it. A regression cap (promote only if the point side
wasn't more than N points worse) patched the *live-accuracy* half of
that risk, and candidate_point_rejected (added the same week) patched
the *offline-rejected* half -- but both were band-aids on a promotion
ACTION that was structurally coupled at the MinIO-copy level. Splitting
into decide_point_promotion / decide_quantile_promotion, each calling
its own promote_candidate_point / promote_candidate_quantiles, removes
the coupling itself: a quantile promotion literally cannot move the
point model anymore (there's no code path left that would), so the
regression cap that existed only to police that coupling is gone too --
see PROMOTION_MAX_ACCURACY_REGRESSION_PP's removal from config.py.
candidate_point_rejected stays (see decide_point_promotion) since it
guards the point decision on its own merits now, not as a coupling
patch.

One more thing this split fixes as a side effect: candidate_gained_
quantiles (the old signal) could only ever be True the FIRST time a
corridor was added -- once production had ANY quantile_levels, "gained"
could never be true again, so a corridor refresh (auto_retrain_quantiles.py
pushing a new quantile_model_version on top of an already-promoted one,
e.g. its weekly Sunday run) had no promotion path at all. Comparing
candidate_meta["quantile_model_version"] against production's directly
(candidate_has_new_quantiles below) covers both cases.

Live corridor verification: a candidate "gaining a corridor" was,
originally, an unconditional bypass of the accuracy-improvement bar --
accuracy-not-much-worse was the only check, nothing confirmed the
corridor itself was honest on live traffic (only ever offline-validated
by auto_retrain_quantiles.py's holdout gate). `_corridor_verified` closes
that: it reads /admin/metrics/{symbol}/compare's `q10_coverage`/
`q90_coverage` -- the fraction of resolved outcomes at/below price_q10 /
price_q90, pooled across horizons the same way directional_accuracy
already is (see predictions' r_q10/r_q90/price_q10/price_q90 columns) --
grouped by (slot, quantile_model_version) under the response's own
"quantiles" key, independently of the point grouping (see step 2 above).
Mirrors auto_retrain_quantiles.py's offline _avg_pinball_and_coverage_ok
but on real traffic instead of a holdout split. Both tails must
independently land within
CORRIDOR_COVERAGE_TOLERANCE of their nominal alpha (CORRIDOR_QUANTILES)
for the corridor to count as "verified" -- an unverified corridor (either
coverage field still missing, e.g. pointed at an older predictor-backend
build, or verified-but-miscalibrated) doesn't get promoted merely for
existing (see the "quantile_unverified" decision label).

Point-model-rejection guard (`candidate_point_rejected`, in
decide_point_promotion): reads `candidate_meta["eval_metrics"]
["decision"]` (already written by auto_retrain.py, no new data needed).
Only relabels an already-failing comparison (delta below
PROMOTION_MIN_ACCURACY_IMPROVEMENT_PP) from generic "not_better" to the
more specific "point_model_rejected" -- does NOT block a delta that
clears the bar on its own. A single offline holdout snapshot is real
signal but not infallible (see this script's own opening paragraphs on
why a second, independent live gate exists at all); a full pp of live
outperformance over min_samples real resolved predictions is
independent, arguably stronger evidence, and refusing to ever act on it
would leave a symbol permanently stuck once candidate and production
converge to the same once-rejected artifact -- there'd be no path back
to a genuinely better model without the rejection eventually clearing
some future retrain's own offline gate first. Does not affect
decide_quantile_promotion at all (no coupling left to guard on that
side).

New feature schema (`new_scheme` / `quantile_new_scheme` metadata labels,
or simply differing schema_versions): live metrics of the two slots aren't
comparable, so neither decision auto-promotes -- both report
"new_scheme_manual_promotion_required" with an ACTION REQUIRED log line,
and the audit entry records it. Promotion of such a candidate is the
explicit, dry-run-by-default scripts/promote_new_scheme.py.

Usage:
    python scripts/promote_if_better.py [SYMBOL ...]
        [--window 7d] [--min-samples 100]
        [--min-accuracy-improvement-pp 1.0]
        [--staging-dir .auto_retrain_staging] [--dry-run]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import requests

from pinance_ml.backend_client import post_retrain_event
from pinance_ml.config import (
    CORRIDOR_COVERAGE_TOLERANCE,
    CORRIDOR_QUANTILES,
    PREDICTOR_BACKEND_TIMEOUT_S,
    PREDICTOR_BACKEND_URL,
    PROMOTION_MIN_ACCURACY_IMPROVEMENT_PP,
    PROMOTION_MIN_SAMPLES,
    PROMOTION_WINDOW,
)
from pinance_ml.data.db import list_symbols
from pinance_ml.model_storage import (
    CANDIDATE_SLOT,
    PRODUCTION_SLOT,
    download_metadata,
    promote_candidate_point,
    promote_candidate_quantiles,
    snapshot_production,
)

# Decision label for a candidate on a changed feature schema: there's
# nothing to compare against (production's live numbers are on different
# features, and predictor-ml-inference rejects the old model once the new
# schema is live), so this script deliberately does NOT promote it -- it
# surfaces it as needing the explicit scripts/promote_new_scheme.py step.
NEW_SCHEME_MANUAL = "new_scheme_manual_promotion_required"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _corridor_verified(
    candidate_q_low_coverage: float | None,
    candidate_q_high_coverage: float | None,
    quantiles: tuple[float, float],
    tolerance: float,
) -> bool:
    """Is the candidate's confidence corridor confirmed honest on LIVE
    traffic, not just "it exists"? Mirrors auto_retrain_quantiles.py's
    offline _avg_pinball_and_coverage_ok, on real resolved predictions
    instead of a holdout split: both tails must independently land within
    `tolerance` of their own nominal alpha (CORRIDOR_QUANTILES).

    Either coverage arg is None if predictor-backend hasn't computed it
    for this row (e.g. an older build) -- this always returns False in
    that case. That's the point: an unverified corridor should never look
    "verified" by accident just because the data isn't there.
    """
    if candidate_q_low_coverage is None or candidate_q_high_coverage is None:
        return False
    q_low, q_high = quantiles
    return abs(candidate_q_low_coverage - q_low) <= tolerance and abs(candidate_q_high_coverage - q_high) <= tolerance


def decide_point_promotion(
    has_candidate: bool,
    has_production: bool,
    candidate_n: int,
    min_samples: int,
    candidate_accuracy: float | None,
    production_accuracy: float | None,
    candidate_point_rejected: bool,
    min_accuracy_improvement_pp: float,
    needs_manual_new_scheme: bool = False,
) -> tuple[str, bool]:
    """Pure point-model promotion decision, factored out of promote_symbol
    so it's testable without any network/MinIO calls -- this script's
    counterpart of auto_retrain.py's decide_target_slot. Entirely
    independent of the corridor's own state (see decide_quantile_promotion)
    -- see module docstring's "Why two independent decisions".

    `candidate_point_rejected` (see module docstring) does NOT short-
    circuit ahead of the accuracy comparison -- checked only AFTER delta
    fails to clear min_accuracy_improvement_pp on its own. A full pp of
    genuine *live* outperformance, measured over min_samples real
    resolved predictions, is independent evidence strong enough to
    override a single stale/noisy offline holdout snapshot; refusing to
    ever promote a model this decisively better on live traffic, no
    matter how much live data piles up, would leave a symbol stuck
    reporting "point_model_rejected" forever once its candidate and
    production converge to the same (rejected) artifact -- exactly the
    coupling-era bug this guard exists to prevent doesn't require that
    much caution. When live delta does clear the bar, "accuracy_improved"
    wins outright; candidate_point_rejected only changes the label (not
    the outcome) of an already-failing comparison, from the generic
    "not_better" to the more specific "point_model_rejected", for a
    clearer audit trail.

    Returns (decision_label, should_promote).
    """
    if not has_candidate:
        return "no_candidate", False
    if not has_production:
        # Nothing to compare against -- auto_retrain.py's own bootstrap
        # case pushes straight to PRODUCTION_SLOT itself when this is
        # true, bypassing candidate entirely, so this state means that
        # hasn't happened yet. Out of scope here.
        return "no_production", False
    if needs_manual_new_scheme:
        # Checked before any live-metrics gate: the candidate's feature
        # schema differs from production's (`new_scheme` label, or the
        # schema_versions simply differ), so live accuracy on the two isn't
        # comparable. Never auto-promoted here -- see NEW_SCHEME_MANUAL.
        return NEW_SCHEME_MANUAL, False
    if candidate_n < min_samples:
        return "insufficient_samples", False
    if candidate_accuracy is None or production_accuracy is None:
        # Production's specific current model_version has no matured
        # rows yet in this window (e.g. it was just promoted itself) --
        # can't judge a delta against nothing.
        return "no_comparable_metrics", False

    delta = candidate_accuracy - production_accuracy
    if delta >= min_accuracy_improvement_pp:
        return "accuracy_improved", True
    if candidate_point_rejected:
        return "point_model_rejected", False
    return "not_better", False


def decide_quantile_promotion(
    has_candidate: bool,
    has_production: bool,
    candidate_n: int,
    min_samples: int,
    candidate_has_new_quantiles: bool,
    corridor_verified: bool,
    needs_manual_new_scheme: bool = False,
) -> tuple[str, bool]:
    """Pure confidence-corridor promotion decision -- this script's other
    half of the split described in the module docstring. No accuracy
    comparison and no candidate_point_rejected check here at all: since
    promote_candidate_quantiles only ever moves quantile-tail files and
    quantile_-prefixed metadata keys, the point model's own state
    (rejected or not, accurate or not) simply can't ride along anymore,
    so there's nothing left on that side to guard against.

    `candidate_has_new_quantiles` -- candidate's quantile_model_version
    differs from production's (and isn't empty) -- covers both "gained a
    corridor for the first time" and "the corridor refreshed to a newer
    version" (see module docstring). Checked before the sample-count
    floor: a symbol whose corridor genuinely hasn't changed shouldn't
    report "insufficient_samples" every run just because nothing new is
    even being evaluated.

    Returns (decision_label, should_promote).
    """
    if not has_candidate:
        return "no_candidate", False
    if not has_production:
        return "no_production", False
    if not candidate_has_new_quantiles:
        return "no_new_quantiles", False
    if needs_manual_new_scheme:
        # Corridor trained on a changed feature schema (`quantile_new_scheme`
        # label, or differing schema versions) -- same rule as the point
        # decision: surfaced, never auto-promoted here.
        return NEW_SCHEME_MANUAL, False
    if candidate_n < min_samples:
        return "insufficient_samples", False
    if corridor_verified:
        return "quantile_gain", True
    return "quantile_unverified", False


def _fetch_compare(symbol: str, window: str) -> dict:
    # ?window=<label> -- predictor-backend computes only this one window,
    # not all five (SUMMARY_WINDOWS['all'] has no real time bound, only
    # predictions' retention does; this script only ever needs `window`,
    # so there's no reason to pay for that one too -- see admin.py's
    # module docstring on the parameter).
    url = f"{PREDICTOR_BACKEND_URL.rstrip('/')}/admin/metrics/{symbol}/compare"
    resp = requests.get(url, params={"window": window}, timeout=PREDICTOR_BACKEND_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def promote_symbol(
    symbol: str,
    window: str,
    min_samples: int,
    min_accuracy_improvement_pp: float,
    staging_dir: Path,
    dry_run: bool,
) -> None:
    candidate_meta = download_metadata(symbol, CANDIDATE_SLOT)
    production_meta = download_metadata(symbol, PRODUCTION_SLOT)

    has_candidate = candidate_meta is not None
    has_production = production_meta is not None
    candidate_version = (candidate_meta or {}).get("model_version")
    production_version = (production_meta or {}).get("model_version")
    candidate_quantile_version = (candidate_meta or {}).get("quantile_model_version")
    production_quantile_version = (production_meta or {}).get("quantile_model_version")
    candidate_has_new_quantiles = bool(candidate_quantile_version) and candidate_quantile_version != production_quantile_version
    # See module docstring's "Point-model-rejection guard" -- catches a
    # candidate slot whose point model auto_retrain.py's own offline gate
    # rejected, independent of whatever's true of the corridor.
    candidate_point_rejected = (candidate_meta or {}).get("eval_metrics", {}).get("decision") == "rejected"
    # New-scheme detection: the explicit labels auto_retrain*.py writes, OR
    # a plain schema_version mismatch (covers candidates pushed before the
    # labels existed, which would otherwise sit at "insufficient_samples"
    # forever -- the old schema's production can't be compared to them).
    point_needs_manual = bool(
        has_candidate and has_production and (
            (candidate_meta or {}).get("new_scheme") is True
            or (candidate_meta or {}).get("schema_version") != (production_meta or {}).get("schema_version")
        )
    )
    quantile_needs_manual = bool(
        has_candidate and has_production and candidate_has_new_quantiles and (
            (candidate_meta or {}).get("quantile_new_scheme") is True
            or (candidate_meta or {}).get("quantile_schema_version")
            != ((production_meta or {}).get("quantile_schema_version") or (production_meta or {}).get("schema_version"))
        )
    )

    candidate_n = 0
    candidate_quantile_n = 0
    candidate_accuracy = production_accuracy = None
    candidate_q_low_coverage = candidate_q_high_coverage = None
    if has_candidate and has_production:
        compare = _fetch_compare(symbol, window)
        window_data = compare.get("windows", {}).get(window, {})
        candidate_stats = window_data.get("candidate", {}).get(candidate_version)
        production_stats = window_data.get("production", {}).get(production_version)
        if candidate_stats:
            candidate_n = candidate_stats["n"]
            candidate_accuracy = candidate_stats["directional_accuracy"]
        if production_stats:
            production_accuracy = production_stats["directional_accuracy"]
        # predictor-backend now groups q10_coverage/q90_coverage by (slot,
        # quantile_model_version) INDEPENDENTLY of the point grouping above
        # (see Pinance_backend/app/api/admin.py's _compare_window) -- this
        # closes the gap this module docstring used to flag as "needs
        # schema changes on predictor-backend's side, not attempted here".
        # .get() chain throughout: older predictor-backend builds, or a
        # window with no matured quantile rows yet, simply omit "quantiles"
        # (or the specific slot/version under it) entirely -- that must
        # degrade to "unverified", never KeyError.
        candidate_quantile_stats = (
            window_data.get("quantiles", {}).get("candidate", {}).get(candidate_quantile_version)
        )
        if candidate_quantile_stats:
            candidate_quantile_n = candidate_quantile_stats["n"]
            candidate_q_low_coverage = candidate_quantile_stats.get("q10_coverage")
            candidate_q_high_coverage = candidate_quantile_stats.get("q90_coverage")
        log(
            f"{symbol}: candidate {candidate_version} (quantiles={candidate_quantile_version}) n={candidate_n} "
            f"accuracy={candidate_accuracy} vs production {production_version} "
            f"(quantiles={production_quantile_version}) accuracy={production_accuracy} "
            f"(window={window}, new_quantiles={candidate_has_new_quantiles}, "
            f"quantile_n={candidate_quantile_n}, "
            f"q10_coverage={candidate_q_low_coverage}, q90_coverage={candidate_q_high_coverage}, "
            f"point_rejected={candidate_point_rejected})"
        )
    else:
        log(f"{symbol}: has_candidate={has_candidate} has_production={has_production} -- skipping compare fetch")

    corridor_verified = _corridor_verified(
        candidate_q_low_coverage, candidate_q_high_coverage, CORRIDOR_QUANTILES, CORRIDOR_COVERAGE_TOLERANCE
    )

    point_decision, should_promote_point = decide_point_promotion(
        has_candidate,
        has_production,
        candidate_n,
        min_samples,
        candidate_accuracy,
        production_accuracy,
        candidate_point_rejected,
        min_accuracy_improvement_pp,
        point_needs_manual,
    )
    quantile_decision, should_promote_quantiles = decide_quantile_promotion(
        has_candidate,
        has_production,
        candidate_quantile_n,
        min_samples,
        candidate_has_new_quantiles,
        corridor_verified,
        quantile_needs_manual,
    )
    if NEW_SCHEME_MANUAL in (point_decision, quantile_decision):
        log(
            f"{symbol}: *** ACTION REQUIRED *** candidate is on a changed feature schema "
            f"(point={point_decision == NEW_SCHEME_MANUAL}, quantile={quantile_decision == NEW_SCHEME_MANUAL}) "
            f"-- not auto-promoted. Review and run: python scripts/promote_new_scheme.py {symbol} --apply"
        )
    log(
        f"{symbol}: point_decision={point_decision}" + (" -> PROMOTE" if should_promote_point else " -- not promoting")
        + f" | quantile_decision={quantile_decision}"
        + (" -> PROMOTE" if should_promote_quantiles else " -- not promoting")
    )
    if not dry_run:
        # dry-run never actually calls promote_candidate_point/_quantiles
        # below, so posting here too would put events on predictor-
        # backend's timeline for promotions that never happened in MinIO.
        # kind must be exactly "point"/"quantile" and decision exactly
        # "promoted"/"rejected" -- predictor-backend's RetrainEventIn
        # Literal-validates both (see backend_client.post_retrain_event's
        # docstring for the 422 this used to cause with the richer
        # point_decision/quantile_decision labels passed straight
        # through); those labels are preserved via metric_name instead.
        post_retrain_event(
            symbol, "point", "promoted" if should_promote_point else "rejected",
            candidate_version=candidate_version, production_version=production_version,
            metric_name=f"directional_accuracy ({point_decision})",
            candidate_value=candidate_accuracy, production_value=production_accuracy,
            threshold=min_accuracy_improvement_pp, n_samples=candidate_n or None,
        )
        post_retrain_event(
            symbol, "quantile", "promoted" if should_promote_quantiles else "rejected",
            candidate_version=candidate_quantile_version, production_version=production_quantile_version,
            metric_name=f"corridor ({quantile_decision})",
            n_samples=candidate_quantile_n or None,
        )

    if (should_promote_point or should_promote_quantiles) and not dry_run:
        # ONE snapshot of the pre-promotion state for the whole run -- each
        # promote_candidate_* would otherwise snapshot on its own and the
        # second would overwrite `previous` with the half-promoted state.
        snapshot_production(symbol)

    if should_promote_point and not dry_run:
        promote_candidate_point(symbol, snapshot=False)
        log(f"{symbol}: promoted point model -> production (model_version={candidate_version})")
    elif should_promote_point and dry_run:
        log(f"{symbol}: --dry-run, not actually calling promote_candidate_point")

    if should_promote_quantiles and not dry_run:
        promote_candidate_quantiles(symbol, snapshot=False)
        log(f"{symbol}: promoted quantile corridor -> production (quantile_model_version={candidate_quantile_version})")
    elif should_promote_quantiles and dry_run:
        log(f"{symbol}: --dry-run, not actually calling promote_candidate_quantiles")

    _write_promotion_log(
        staging_dir,
        symbol,
        {
            "point_decision":                    point_decision,
            "point_promoted":                    bool(should_promote_point and not dry_run),
            "quantile_decision":                 quantile_decision,
            "quantile_promoted":                 bool(should_promote_quantiles and not dry_run),
            "dry_run":                           dry_run,
            "candidate_model_version":           candidate_version,
            "production_model_version":          production_version,
            "candidate_quantile_model_version":  candidate_quantile_version,
            "production_quantile_model_version": production_quantile_version,
            "candidate_n":                       candidate_n,
            "candidate_quantile_n":              candidate_quantile_n,
            "candidate_accuracy":                candidate_accuracy,
            "production_accuracy":               production_accuracy,
            "candidate_has_new_quantiles":        candidate_has_new_quantiles,
            "candidate_q10_coverage":            candidate_q_low_coverage,
            "candidate_q90_coverage":            candidate_q_high_coverage,
            "corridor_verified":                 corridor_verified,
            "candidate_point_rejected":           candidate_point_rejected,
            "point_needs_manual_new_scheme":     point_needs_manual,
            "quantile_needs_manual_new_scheme":  quantile_needs_manual,
            "window":                            window,
        },
    )


def _write_promotion_log(staging_dir: Path, symbol: str, entry: dict) -> None:
    log_path = staging_dir / "promote_if_better_log.jsonl"
    row = {"symbol": symbol, "checked_at": pd.Timestamp.now("UTC").isoformat(), **entry}
    with log_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="*", help="Symbols to evaluate (default: all in DB)")
    parser.add_argument("--window", default=PROMOTION_WINDOW)
    parser.add_argument("--min-samples", type=int, default=PROMOTION_MIN_SAMPLES)
    parser.add_argument("--min-accuracy-improvement-pp", type=float, default=PROMOTION_MIN_ACCURACY_IMPROVEMENT_PP)
    parser.add_argument("--staging-dir", default=".auto_retrain_staging")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Decide and log, but never actually call promote_candidate_point/promote_candidate_quantiles",
    )
    args = parser.parse_args()

    if not PREDICTOR_BACKEND_URL:
        log("PREDICTOR_BACKEND_URL is not set -- nothing to do, refusing to start")
        sys.exit(1)

    symbols = args.symbols or list_symbols()
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    log(f"Symbols ({len(symbols)}): {symbols}")
    log(
        f"window={args.window}, min_samples={args.min_samples}, "
        f"min_accuracy_improvement_pp={args.min_accuracy_improvement_pp}, dry_run={args.dry_run}"
    )

    for i, symbol in enumerate(symbols, start=1):
        log(f"[{i}/{len(symbols)}] {symbol}")
        try:
            promote_symbol(
                symbol,
                args.window,
                args.min_samples,
                args.min_accuracy_improvement_pp,
                staging_dir,
                args.dry_run,
            )
        except Exception:
            log(f"{symbol}: FAILED -- see traceback below, continuing with remaining symbols")
            import traceback

            traceback.print_exc()

    log("All done.")


if __name__ == "__main__":
    main()
