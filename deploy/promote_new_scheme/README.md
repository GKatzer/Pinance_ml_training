# Promoting a candidate after a feature-schema change (runbook)

When `compute_features` adds/removes/reorders a column, `schema_version`
(hash of `feature_columns`) changes. Production's models then can't be
compared to a new candidate, and predictor-ml-inference rejects the old
models once the new schema is live. This is the path for that situation.

No timer, no automation: **a human runs this** (automatic promotion behind
a flag is a separate, later phase).

## What happens automatically

1. `auto_retrain.py` / `auto_retrain_quantiles.py` detect the mismatch,
   train, and run a **naive-baseline sanity gate** on the held-out window
   (point: avg MAE within `SANITY_MAX_MAE_RATIO` of the naive `r_h=0`
   forecast and directional accuracy >= majority-class base rate;
   corridor: pinball loss vs constant empirical quantile, plus calibrated
   coverage). Fail -> decision `schema_drift_rejected_sanity`, nothing is
   pushed. Pass -> pushed to the `candidate` slot.
2. The pushed metadata carries explicit labels, always written (true or
   false, never absent): `new_scheme` / `replaces_schema_version` (point),
   `quantile_new_scheme` / `quantile_replaces_schema_version` (corridor),
   and the sanity verdict under `eval_metrics.sanity` /
   `quantile_eval_metrics.sanity`.
3. `promote_if_better.py` sees the schema change and reports
   `new_scheme_manual_promotion_required` plus an `ACTION REQUIRED` log
   line. It never promotes such a candidate.

## Manual promotion

Run from a checkout whose feature code matches the commit that trained the
candidate (`source_commit` in the candidate's `metadata.json`) -- this
script does not recompute the current schema itself.

```bash
# 1. Dry-run (default) -- prints the decision per symbol, changes nothing
python scripts/promote_new_scheme.py BTCUSDT ETHUSDT

# 2. Apply -- snapshots production to `previous/`, promotes, re-checks consistency
python scripts/promote_new_scheme.py BTCUSDT ETHUSDT --apply
```

A half (point or corridor) is promoted only if: candidate and production
both exist; the explicit label is `true`; the schema really differs; the
sanity verdict is a recorded PASS; and the candidate's files match its
`metadata.json`. Promote the point model and the corridor **together** (or
retrain the missing one first) -- otherwise production ends up with a point
model and a corridor on different schemas; the post-promotion check logs
that as `mixed schema`.

Legacy candidates (pushed before labels/sanity existed) have neither:

```bash
python scripts/promote_new_scheme.py SYMBOL --label-legacy --apply   # writes the label
python scripts/promote_new_scheme.py SYMBOL --skip-sanity-check --apply  # explicit, audited override
```

A recorded sanity **failure** cannot be overridden.

## Rollback

`previous/` is the state of production just before the last promotion
(one level deep, not a history).

```bash
python scripts/promote_new_scheme.py BTCUSDT --rollback          # dry-run
python scripts/promote_new_scheme.py BTCUSDT --rollback --apply
```

Rolling back to a model on a different schema only helps together with
rolling back the feature code -- otherwise inference rejects it again.

## Audit

Every run appends to `.auto_retrain_staging/promote_new_scheme_log.jsonl`
(`dry_run`, decisions, versions, schema versions, consistency problems,
`skip_sanity_check`). Applied promotions also post a retrain event to
predictor-backend (best-effort: a failed POST is only a warning).

## Running as your own user

The timers run as `pinance` and own `.auto_retrain_staging/`. Manual runs
as another user hit `PermissionError` when the script cleans its download
directory -- pass your own directory:

```bash
python scripts/auto_retrain.py --staging-dir .manual_staging
python scripts/promote_new_scheme.py SYMBOL --staging-dir .manual_staging
```
