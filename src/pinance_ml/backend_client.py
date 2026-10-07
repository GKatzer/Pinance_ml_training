"""Thin best-effort client for predictor-backend's admin API (backend host) --
shared by scripts/auto_retrain.py, scripts/auto_retrain_quantiles.py and
scripts/promote_if_better.py to journal a retrain/promotion decision they've
already computed in memory, via POST /admin/retrain-events, for the MLOps
dashboard's event timeline.

Best-effort by design: this is a monitoring side-channel, not a dependency
of the retrain/promotion pipeline itself -- a predictor-backend hiccup must
never fail an actual retrain or promotion run, so failures are logged and
swallowed here rather than raised. Also a silent no-op when
PREDICTOR_BACKEND_URL isn't set, so auto_retrain.py / auto_retrain_
quantiles.py (which never required predictor-backend before) still don't.
"""

import time
from datetime import UTC, datetime

import requests

from pinance_ml.config import PREDICTOR_BACKEND_TIMEOUT_S, PREDICTOR_BACKEND_URL


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def post_retrain_event(
    symbol: str,
    kind: str,
    decision: str,
    *,
    candidate_version: str | None = None,
    production_version: str | None = None,
    metric_name: str | None = None,
    candidate_value: float | None = None,
    production_value: float | None = None,
    threshold: float | None = None,
    n_samples: int | None = None,
    train_wall_seconds: float | None = None,
) -> None:
    """POST one already-decided retrain/promotion event. `kind` must be
    exactly "point" or "quantile" and `decision` exactly "promoted" or
    "rejected" -- predictor-backend's RetrainEventIn validates both as a
    Literal, 422-ing (silently swallowed below, see module docstring) on
    anything else. This caught out an earlier version of this function
    that passed kind="point_promotion"/"quantile_promotion" and raw
    decision labels ("not_better", "pushed_to_candidate", ...) straight
    through from the caller's own richer decide_*() labels -- every
    promote_if_better.py call 422'd outright (invalid kind), and only
    auto_retrain*.py's "rejected" case happened to pass by coincidence
    (with metric_value silently dropped -- that field doesn't exist on
    RetrainEventIn either, superseded by metric_name/candidate_value/
    production_value below). Callers now do that promoted/rejected
    mapping themselves and pass the original richer label via
    `metric_name` instead, so it isn't lost.
    """
    if not PREDICTOR_BACKEND_URL:
        return
    url = f"{PREDICTOR_BACKEND_URL.rstrip('/')}/admin/retrain-events"
    payload = {
        "symbol": symbol,
        "kind": kind,
        "decision": decision,
        "candidate_version": candidate_version,
        "production_version": production_version,
        "metric_name": metric_name,
        "candidate_value": candidate_value,
        "production_value": production_value,
        "threshold": threshold,
        "n_samples": n_samples,
        "train_wall_seconds": train_wall_seconds,
        # Required by POST /admin/retrain-events -- missing this made every
        # call 422 silently (caught by the except below, only visible as a
        # WARNING line in cron output), so retrain-timeline stayed empty
        # end-to-end despite the backend side working. Same Z-suffix ISO
        # format predictor-backend's other admin endpoints already emit
        # (see app.api.admin's own datetime.now(UTC).isoformat() calls).
        "decided_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    try:
        resp = requests.post(url, json=payload, timeout=PREDICTOR_BACKEND_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as exc:
        _log(f"{symbol}: WARNING -- failed to POST retrain event (kind={kind}, decision={decision}): {exc}")
