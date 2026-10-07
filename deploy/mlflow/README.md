# MLflow tracking server deploy (<training-host>)

Runs `mlflow server` as a systemd unit on <training-host>, co-located with MinIO.
It's the experiment tracker for this repo and the instance the frontend's
tab 03 ("EXPERIMENT TRACKING -> Open MLflow UI") links out to.

Two kinds of run land here:

| Experiment | Written by | One run per |
|---|---|---|
| `retrain-point` | `scripts/auto_retrain.py` | symbol, per scheduled point retrain |
| `retrain-quantile` | `scripts/auto_retrain_quantiles.py` | symbol, per scheduled corridor retrain |
| `research-feature-gain` | `scripts/measure_*.py` | invocation |
| `research-screening` | `scripts/screen_*.py` | invocation |

Models are **not** pushed here -- MinIO's `pinance-models` bucket (see
`deploy/auto_retrain/`) stays the single source of truth for what
serves. MLflow only gets metrics, params, tags, and the same
`reports/*` / eval-summary files those scripts already write locally.

## Optional by construction

`pinance_ml.tracking` is a silent no-op unless `PINANCE_MLFLOW_TRACKING_URI`
is set in the training box's `.env` (same contract as
`PREDICTOR_BACKEND_URL`). With it unset, every `auto_retrain*.py` /
`measure_*` / `screen_*` run behaves exactly as before -- nothing logged,
nothing slower, nothing able to fail because this server is down. So this
unit can be stood up (or torn down) independently of the training
schedule.

## Setup (once, on <training-host>)

Assumes the repo is already deployed per `deploy/news_parser/README.md`
(same venv, same `pinance` user) and MinIO is up per `deploy/minio/`.

0. **Dependencies** -- after pulling the change that added this directory,
   re-run `pip install -r requirements.txt` in the training venv. It now
   pins `mlflow` **and `boto3`** (core mlflow does not pull boto3, and
   without it every artifact upload silently no-ops -- see the comment in
   `requirements.txt`). The `mlflow` CLI this unit's `ExecStart` calls is
   that same venv's entrypoint.
1. **Backend DB** -- create the `mlflow` role + database on <backend-host>'s
   Postgres (commands in `mlflow.env.example`). Hand the resulting
   `postgresql://mlflow:...@<db-host-ip>:5432/mlflow` string to whoever
   fills in `/opt/mlflow/mlflow.env`.
2. **Artifact bucket** -- create the `pinance-mlflow` MinIO bucket and a
   scoped access key, distinct from the `pinance-models` credential
   (commands in `mlflow.env.example`). The desktop training box needs
   this same key too, not just <training-host> -- see "Where runs come from" below.
3. **Server env**

   ```bash
   sudo mkdir -p /opt/mlflow
   sudo cp deploy/mlflow/mlflow.env.example /opt/mlflow/mlflow.env
   sudo $EDITOR /opt/mlflow/mlflow.env      # fill in the CHANGE_ME values
   sudo chmod 600 /opt/mlflow/mlflow.env && sudo chown pinance /opt/mlflow/mlflow.env
   ```

4. **Unit**

   ```bash
   sudo cp deploy/mlflow/mlflow.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now mlflow.service
   ```

5. **`.env` on every box that runs training code** -- add the client
   block from `Pinance_ML/.env.example` (`PINANCE_MLFLOW_TRACKING_URI` +
   `MLFLOW_S3_ENDPOINT_URL` + `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).
   That means **<training-host> and the desktop** -- the last three let the mlflow
   client PUT artifacts straight into MinIO.

6. **Frontend** -- point tab 03's link at `http://<training-host-ip>:5000` (or
   whatever `MLFLOW_BIND_ADDRESS:MLFLOW_PORT` resolves to on the tailnet)
   once this is up.

## Where runs come from (desktop needs the artifact key)

`auto_retrain*.py` runs on <training-host>, but the `measure_*` / `screen_*`
experiments run **on the desktop** (CPU LightGBM, heavy) -- so if only
<training-host> has the `pinance-mlflow` key, every research run still logs its
params/metrics but loses its `reports/*` artifacts (`log_artifact` swallows
the auth failure). Give the desktop the **same bucket-scoped** access key.
Blast radius is one non-critical bucket; the alternative ("artifacts only
from <training-host>") throws away exactly the runs that produce the most files.

## Bind address

`mlflow.service` binds `${MLFLOW_BIND_ADDRESS}` = the Tailscale IP, the
same way `deploy/minio/minio.service` binds its own tailnet address rather
than loopback. <backend-host> (for the tab-03 link) and the desktop training box
both have to reach it, and this project has no public ports. If you'd
rather keep mlflow on `127.0.0.1` and reverse-proxy it through Caddy
(like the rest of the stack), set `MLFLOW_BIND_ADDRESS=127.0.0.1` and add
the vhost -- the client URI then becomes the Caddy hostname.

## No authentication

`mlflow server` ships no auth. On the tailnet IP that means **anyone on
the tailnet can read, write, or delete any experiment** -- same trust
model as MinIO and Postgres on this project (Tailscale is the perimeter,
no box has a public port). Stated here so it's a deliberate choice, not an
oversight. If the UI ever needs to be reachable outside the tailnet
(e.g. the frontend is served publicly via Caddy on `<public-domain>`,
in which case a public visitor clicking "Open MLflow UI" currently hits a
dead `<training-host-ip>` link), put it behind a Caddy vhost with `basicauth`
and set `MLFLOW_BIND_ADDRESS=127.0.0.1`.

## Operating

```bash
systemctl status mlflow.service
journalctl -u mlflow.service -f
curl -sf http://<training-host-ip>:5000/health && echo OK    # readiness
```

A run that shows up as `FAILED` in the UI means the training script
raised inside the `with mlflow_run(...)` block (e.g. one symbol's retrain
threw) -- the run is still closed cleanly and the other symbols continue;
check `journalctl -u auto-retrain.service` for that symbol's traceback.
