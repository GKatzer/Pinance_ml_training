# Promote-if-better deploy (<training-host>)

Runs `scripts/promote_if_better.py` every 6 hours via a systemd timer:
the live-traffic promotion gate that `deploy/auto_retrain/README.md`
explicitly defers to this. `auto-retrain*.timer` retrain and gate against
an *offline* holdout, then push a candidate to MinIO -- this timer asks
the second, independent question: has that candidate, once
predictor-ml-inference started shadow-serving it, actually held up on
*real* live traffic? The live numbers come from predictor-backend's
`GET /admin/metrics/{symbol}/compare` (real predictions joined against
real realized candles in Postgres) -- this script never touches Postgres
itself, only MinIO (read both slots' `metadata.json`, and
`promote_candidate_point` / `promote_candidate_quantiles` for whichever
gate(s) clear) and that one HTTP endpoint.

See the script's own module docstring for the full decision logic
(`decide_point_promotion` / `decide_quantile_promotion`, unit-tested in
`tests/test_promote_if_better.py`). Short version -- two fully
independent decisions per symbol per run, either, both, or neither can
fire:
- Point model: promote if the candidate has enough live samples to
  trust, its point model wasn't itself rejected by auto_retrain.py's own
  offline gate, and its live directional accuracy beats production's by
  a margin.
- Confidence corridor: promote if the candidate has a quantile_model_version
  production doesn't already have (first corridor, or a refreshed one),
  it has enough live samples of ITS OWN to trust (predictor-backend now
  tags predictions with model_version and quantile_model_version
  separately, so this is a genuinely independent count from the point
  decision's -- a thin point sample no longer blocks a well-sampled
  corridor refresh, or vice versa), and its live calibration
  (`q10_coverage`/`q90_coverage` against nominal) checks out.

Each decision calls its own `model_storage.promote_candidate_point` /
`promote_candidate_quantiles` -- moving one never moves the other's
files or metadata keys as a side effect (see model_storage.py's module
docstring for the 2026-08-03 incident this replaced: a point model
rejected by auto_retrain.py's offline gate rode into production bundled
with an approved quantile push, because the only tool at the time,
`promote_candidate`, copied the whole slot).

New feature schema: a candidate labeled `new_scheme` / `quantile_new_scheme`
(or whose `schema_version` simply differs from production's) is **not**
auto-promoted -- live numbers on two different feature sets aren't
comparable. The decision is `new_scheme_manual_promotion_required` and the
log carries an `ACTION REQUIRED` line; promote it with
`scripts/promote_new_scheme.py` (runbook: `deploy/promote_new_scheme/README.md`).
Every promotion here also snapshots production into `previous/` first.

## Setup (once, on <training-host>)

Assumes `deploy/auto_retrain/README.md`'s setup is already done (same
venv, same `.env`, same `pinance` service user, same MinIO settings).
Additionally set in `.env`:

```
PREDICTOR_BACKEND_URL=http://<backend-host-address>:8002
```

`/admin/*` on predictor-backend is deliberately not proxied publicly (see
`Pinance_backend/app/api/admin.py`) -- this only works if <training-host> is on the
same tailnet as <backend-host> and can reach it directly on that port. The rest of
the policy has sane defaults (see `pinance_ml/config.py`'s
`PROMOTION_*` constants) and doesn't need overriding to start:

```
PROMOTION_WINDOW=7d
PROMOTION_MIN_SAMPLES=100
PROMOTION_MIN_ACCURACY_IMPROVEMENT_PP=1.0
```

```bash
sudo cp deploy/promote_if_better/promote-if-better.service deploy/promote_if_better/promote-if-better.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now promote-if-better.timer
```

Same `/opt/Pinance_ML` placeholder convention as the auto-retrain units --
if you edit one unit's paths to match this repo's actual clone path, edit
all of them (`news-parser.service`, `auto-retrain*.service`, this one) the
same way.

## Operating

```bash
systemctl status promote-if-better.timer     # next/last scheduled run
journalctl -u promote-if-better.service -f   # tail a run
sudo systemctl start promote-if-better.service   # run once, right now
python scripts/promote_if_better.py --dry-run   # decide and log, never actually promote
```

Every run's per-symbol decision is logged to stdout and appended as a
JSON line to `.auto_retrain_staging/promote_if_better_log.jsonl` in the
working directory -- same audit-trail convention as
`auto_retrain_log.jsonl`, readable without MinIO or predictor-backend
access to see what a past run decided and why. Two independent labels per
entry now (`point_decision` and `quantile_decision`):
- point: `no_candidate` / `no_production` / `insufficient_samples` /
  `no_comparable_metrics` / `point_model_rejected` / `accuracy_improved` /
  `not_better`
- quantile: `no_candidate` / `no_production` / `no_new_quantiles` /
  `insufficient_samples` / `quantile_unverified` / `quantile_gain`

Running this every 6 hours is not a carefully hand-picked offset from the
retrain timers (daily 03:00 point, weekly Sunday 04:30 corridor) -- it
doesn't need to be. A candidate that just landed in MinIO minutes ago
naturally has too few live samples yet (`PROMOTION_MIN_SAMPLES`) and gets
rejected as `insufficient_samples`, not promoted prematurely; the next
run a few hours later re-evaluates it with more live data. The sample
floor is what paces this, not the schedule.
