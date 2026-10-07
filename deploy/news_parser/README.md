# News parser deploy (<training-host>)

Runs `scripts/fetch_news.py` + `scripts/score_pending_news.py` every 5
minutes via a systemd timer, so `news_items` stays live instead of only
updating when someone runs the scripts by hand. (Considered Docker first;
dropped it -- this project's own stated infra for this kind of job is
already systemd, and a container adds real overhead here: FinBERT's
~1.5GB of torch/transformers weights, a Tailscale-reachability workaround
for the DB connection, and slower edit-deploy cycles while this code is
still actively changing. Two short-lived periodic scripts is the
textbook systemd-timer case, not a long-running containerized service.)

## Setup (once, on <training-host>)

```bash
# adjust the path to wherever this repo is actually cloned on <training-host>
sudo useradd --system --home /opt/Pinance_ML --shell /usr/sbin/nologin pinance || true
cd /opt/Pinance_ML
cp .env.example .env   # fill in DATABASE_URL / NEWS_DATABASE_URL for real
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
sudo chown -R pinance:pinance /opt/Pinance_ML

sudo cp deploy/news_parser/news-parser.service deploy/news_parser/news-parser.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now news-parser.timer
```

`news-parser.service`/`.timer` hardcode `/opt/Pinance_ML` and a `pinance`
service user as placeholders -- edit `WorkingDirectory`, `EnvironmentFile`,
and `User` in the `.service` file to match wherever this actually lives.

## Operating

```bash
systemctl status news-parser.timer          # next/last scheduled run
journalctl -u news-parser.service -f         # tail progress (both scripts log with flush=True)
sudo systemctl start news-parser.service     # run once, right now, outside the schedule
sudo systemctl restart news-parser.timer     # after editing the timer interval
```

After a code change: no rebuild step, just `git pull` in
`/opt/Pinance_ML` -- the next timer tick picks it up. If `requirements.txt`
changed, also rerun `./.venv/bin/pip install -r requirements.txt`.

Both scripts are safe to fail mid-cycle: `fetch_news.py` dedups by URL
(DB unique constraint), `score_pending_news.py` only touches rows where
`sentiment_pos IS NULL`. The `-` prefix on both `ExecStart=` lines in the
service file means a failed step doesn't block the other or fail the
unit -- a crashed cycle just gets retried cleanly 5 minutes later.

`Persistent=true` in the timer means a tick missed while <training-host> was off
(the one non-24/7 node in this project) fires once promptly on the next
boot rather than waiting for the next natural 5-minute mark.
