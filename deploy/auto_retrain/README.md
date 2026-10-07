# Auto-retrain deploy (<training-host>)

Runs `scripts/auto_retrain.py` once a day via a systemd timer (README
"Ретрейн регрессора"): retrain the 12-horizon regressor per symbol, gate
the result against current production on a held-out window, push a
candidate to MinIO if it clears the bar. Doesn't touch production
directly except on first bootstrap (no production model yet) -- see the
script's own docstring for the full decision logic.

## What this does NOT do

**Doesn't promote a candidate to production itself.** A candidate pushed
here sits in MinIO's `candidate` slot until predictor-ml-inference (<inference-host>)
picks it up for shadow evaluation. The actual promotion decision -- does
this candidate also hold up on real live traffic, not just this script's
offline holdout -- is a separate script/timer on its own schedule, see
`deploy/promote_if_better/README.md`. Deliberately kept out of this
service: the offline gate above and the live-traffic gate are answering
different questions on different data, and coupling them would mean this
daily run either blocks on network calls to predictor-backend or silently
promotes off a holdout comparison alone.

## Feature-schema change

If the feature schema changed, the candidate can't be compared with
production. It must instead clear a naive-baseline sanity gate, and is
pushed with an explicit `new_scheme` label that `promote_if_better.py`
will not auto-promote -- see `deploy/promote_new_scheme/README.md`.

## Manual runs (`--staging-dir`)

The systemd service runs as `pinance` and owns `.auto_retrain_staging/`.
Running `auto_retrain.py` by hand as another user fails with
`PermissionError` when it clears `production_download/`. Pass your own
directory: `python scripts/auto_retrain.py --staging-dir .manual_staging`.

## Setup (once, on <training-host>)

Assumes the repo is already deployed per `deploy/news_parser/README.md`
(same venv, same `.env`, same `pinance` service user). Additionally set
in `.env` (see `.env.example`):

```
MINIO_ENDPOINT=<host:port>
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
MINIO_SECURE=false   # true if MinIO is behind TLS
MINIO_MODELS_BUCKET=pinance-models   # default, override only if it needs to differ
```

```bash
sudo cp deploy/auto_retrain/auto-retrain.service deploy/auto_retrain/auto-retrain.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now auto-retrain.timer
```

### Confidence corridor, separate script + weekly timer

`auto-retrain.service` above only ever trains/gates/pushes the point
models -- the corridor (`CORRIDOR_QUANTILES` tail models) is trained by a
genuinely separate script, `scripts/auto_retrain_quantiles.py`, not a
flag on the same one. They were bundled behind a `--include-quantiles`
flag at first; that was reverted after finding a real coupling bug (a
candidate point model that happened to be marginally worse than
production -- pure retrain noise, unrelated to the corridor -- could
block an improving corridor from ever shipping, since both travelled as
one atomic push). See `scripts/auto_retrain_quantiles.py`'s own module
docstring and project memory: `project_quantile_modeling_task.md`. Each
script's push now merges only its own metadata fields onto whatever's
already in the target slot (`pinance_ml.model_storage.merge_metadata`),
so running them independently, on independent schedules, is safe by
construction -- neither can clobber the other's fields or files. Enable
the second timer once predictor-ml-inference is actually consuming the
corridor:

```bash
sudo cp deploy/auto_retrain/auto-retrain-quantiles.service deploy/auto_retrain/auto-retrain-quantiles.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now auto-retrain-quantiles.timer
```

Both units hardcode the same `/opt/Pinance_ML` placeholders as
`auto-retrain.service` -- if you edit one to match this repo's actual
clone path, edit the other (and `news-parser.service`) the same way. All
paths across all three units must agree, or you'll hit the exact
WorkingDirectory/EnvironmentFile/ExecStart mismatch this project's own
deploy history already ran into once.

## Operating

```bash
systemctl status auto-retrain.timer              # next/last scheduled point-only run
systemctl status auto-retrain-quantiles.timer     # next/last scheduled corridor run
journalctl -u auto-retrain.service -f             # tail point-model progress
journalctl -u auto-retrain-quantiles.service -f   # tail corridor progress
sudo systemctl start auto-retrain.service             # run point-model retrain once, right now
sudo systemctl start auto-retrain-quantiles.service   # run corridor retrain once, right now
```

A run's per-symbol decisions (bootstrap / pushed_to_candidate / rejected /
schema_drift, or their `_quantiles`-suffixed corridor equivalents) are
logged to stdout (journalctl) and appended as JSON lines to
`.auto_retrain_staging/auto_retrain_log.jsonl` (point) or
`.auto_retrain_staging/auto_retrain_quantiles_log.jsonl` (corridor) in
the working directory -- separate files since the two scripts run as
independent processes on independent schedules -- a lightweight audit
trail without needing MinIO access to see what a past run decided and
why.

Expect a real run to take longer than the news-parser's few minutes: it
trains 12 horizon models *twice* per symbol (once on a held-out split for
the gate decision, once on the full history for whatever actually gets
pushed) -- budget accordingly when picking the daily time slot (`03:00`
UTC by default in `auto-retrain.timer`, adjust if it collides with
anything else scheduled on <training-host>).
