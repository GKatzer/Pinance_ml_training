# MinIO deploy (<training-host>)

Object storage for model artifacts (README architecture diagram already
lists MinIO as living on <training-host>): `predictor-ml-training`
(`scripts/auto_retrain.py` / `export_models.py`, this repo) pushes
`production`/`candidate` model sets here; `predictor-ml-inference` (<inference-host>)
polls them. See `Pinance_ML/src/pinance_ml/model_storage.py`'s docstring
for the exact bucket/object layout.

Native binary + systemd, not Docker -- same call `deploy/news_parser/`
already made for this project's <training-host> infra (systemd is the established
pattern here, and MinIO is a single static Go binary with zero runtime
dependencies, so a container buys nothing).

## Setup (once, on <training-host>)

```bash
# 1. Dedicated system user -- MinIO's own data directory shouldn't be
#    owned by the `pinance` user that runs auto_retrain.py/news-parser,
#    same least-privilege separation as any other service account.
sudo useradd --system --home /opt/minio --shell /usr/sbin/nologin minio || true
sudo mkdir -p /opt/minio/data
sudo chown -R minio:minio /opt/minio

# 2. dl.min.io/server/... now 302-redirects to a GitHub release asset --
#    -L is required or curl saves the tiny redirect-notice HTML page
#    instead of the binary (systemd then fails with "Exec format error",
#    since the kernel can't execute an HTML file). Sanity-check with
#    `file` before installing -- catches this class of mistake before it
#    reaches systemd.
curl -L -o /tmp/minio https://dl.min.io/server/minio/release/linux-amd64/minio
file /tmp/minio   # expect "ELF 64-bit LSB executable, x86-64" -- if it says
                  # "HTML document" instead, the download didn't go through
sudo install -m 755 /tmp/minio /usr/local/bin/minio
minio --version

# Alternative: build from source instead of trusting a prebuilt binary --
#   sudo apt install -y golang-go && go install github.com/minio/minio@latest
#   sudo install -m 755 "$HOME/go/bin/minio" /usr/local/bin/minio
# (same `file`/`minio --version` sanity check applies either way)

# 3. Credentials + config -- copy the template, then edit in real values.
#    NEVER commit the filled-in version; it only ever lives on <training-host>.
sudo cp deploy/minio/minio.env.example /opt/minio/minio.env
sudo nano /opt/minio/minio.env   # fill MINIO_ROOT_USER/PASSWORD via `openssl rand -base64 24`,
                                  # confirm MINIO_BIND_ADDRESS matches `tailscale ip -4`
sudo chown minio:minio /opt/minio/minio.env
sudo chmod 600 /opt/minio/minio.env

# 4. systemd unit
sudo cp deploy/minio/minio.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now minio.service
```

`minio.service` hardcodes `/opt/minio` and a `minio` service user as
placeholders -- edit `WorkingDirectory`/`EnvironmentFile`/`User`/`Group`
in the `.service` file if this actually lives somewhere else.

The bucket itself (`pinance-models` by default) does **not** need a
manual `mc mb` step -- `model_storage.py`'s `_client()` already calls
`make_bucket` if it doesn't exist, so the first real `auto_retrain.py`
push creates it automatically.

## Verify

```bash
systemctl status minio.service                          # active (running)?
curl http://<training-host-ip>:9000/minio/health/live          # MinIO's own health check, expect 200
                                                          # (use this host's own MINIO_BIND_ADDRESS if it differs)
journalctl -u minio.service -f                           # tail startup/access logs
```

Then confirm this repo can actually reach it: fill in the same
`MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` as `MINIO_ACCESS_KEY`/
`MINIO_SECRET_KEY` in `Pinance_ML/.env` (`MINIO_ENDPOINT` = the same
`MINIO_BIND_ADDRESS` from step 3, `MINIO_SECURE=false` -- no TLS, this
never leaves the Tailscale-private network) and run
`python scripts/auto_retrain.py BTCUSDT` once by hand -- see
`deploy/auto_retrain/README.md`'s own setup section for that half.

## Operating

```bash
systemctl status minio.service       # is it up
sudo systemctl restart minio.service # after editing minio.env
journalctl -u minio.service -f       # tail logs
```

## Disk / backup

Model sets are tiny (a full point+corridor bundle for one symbol is well
under 1MB * ~36-48 files, see project memory: quantile-modeling task) --
<training-host>'s 1TB is not a real constraint here. There's no backup step beyond
whatever <training-host>'s own backup story already is: every object in
`production`/`candidate` is fully regenerable by re-running
`export_models.py` against TimescaleDB, so losing this bucket is an
inconvenience (retrain once), not data loss.
