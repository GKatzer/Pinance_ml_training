"""Best-effort MLflow experiment-tracking wrapper -- the training side's
counterpart of backend_client.py, built to the same contract: a monitoring
side-channel that must never fail, slow down, or change the outcome of an
actual retrain or research run.

Silent no-op unless PINANCE_MLFLOW_TRACKING_URI is set (surfaced as
config.MLFLOW_TRACKING_URI). Even when it is, everything here still
swallows its own errors -- `import mlflow` missing, the tracking server on
the training host being down, an artifact-store credential typo -- degrading to the
same no-op rather than raising into the caller. Each failure is logged
once as a WARNING and then ignored, exactly like backend_client.

Retrain scripts (scripts/auto_retrain.py, scripts/auto_retrain_quantiles.py):

    with mlflow_run("retrain-point", run_name=symbol, tags={"kind": "point"}):
        log_params({...})
        log_metrics({...})
        set_tags({"decision": decision})       # tags known only mid-run
        log_dict(eval_summary, "eval_metrics.json")
        log_artifact(some_csv_path)

Outside a `with mlflow_run(...)` block, or when it no-op'd, the log_* /
set_tags helpers are themselves no-ops -- callers never branch on whether
tracking is configured.

Research scripts (scripts/measure_*.py, scripts/screen_*.py): one
log_research_run(...) call at the end of main().
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pinance_ml.config import MLFLOW_TRACKING_URI

# True only between a successful mlflow.start_run() and its end_run(). The
# log_* helpers check this so a no-op'd (or never-opened) run makes them
# no-ops too, instead of mlflow raising "no active run". Single-threaded,
# one run at a time by construction (auto_retrain*.py loop over symbols is
# sequential) -- a module global is enough, no need for contextvars.
_RUN_ACTIVE = False


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _source_commit() -> str:
    """Git HEAD SHA, best-effort ('' if this isn't a git checkout).

    Prefers scripts/export_models.py's helper -- every script that uses
    this module already puts scripts/ on sys.path -- and falls back to its
    own git call for contexts where it doesn't (pytest, whose path is
    src/ only). Not imported at module top level: that would make
    `import pinance_ml.tracking` depend on scripts/ being importable.
    """
    try:
        from export_models import _source_commit as _sc

        return _sc()
    except Exception:
        pass
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


@contextmanager
def mlflow_run(
    experiment: str,
    run_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Context manager around one MLflow run.

    Yields the mlflow Run object on success, or None when tracking is a
    no-op (URI unset, mlflow not importable, or the tracking server can't
    be reached / errored on start). The `with` body always executes either
    way. Never raises out of setup or teardown. On success the run is
    closed FINISHED, or FAILED if the body raised (the exception still
    propagates -- retrain scripts catch it per-symbol themselves).
    """
    global _RUN_ACTIVE

    if not MLFLOW_TRACKING_URI:
        yield None
        return

    try:
        import mlflow
    except Exception as exc:  # ImportError, or a broken partial install
        _log(f"WARNING -- mlflow not importable, experiment tracking off for this run: {exc}")
        yield None
        return

    started = False
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name)
        started = True
        merged = {"source_commit": _source_commit(), **dict(tags or {})}
        mlflow.set_tags({k: v for k, v in merged.items() if v is not None})
        _RUN_ACTIVE = True
    except Exception as exc:
        # mlflow wraps a wide range of failures (network, auth, backend
        # store) in its own exception types -- catch broadly, same
        # best-effort stance as backend_client's `except RequestException`.
        _log(f"WARNING -- failed to start MLflow run (experiment={experiment!r}): {exc}")
        if started:
            try:
                mlflow.end_run("FAILED")
            except Exception:
                pass
        _RUN_ACTIVE = False
        yield None
        return

    ok = False
    try:
        yield run
        ok = True
    finally:
        _RUN_ACTIVE = False
        try:
            mlflow.end_run("FINISHED" if ok else "FAILED")
        except Exception as exc:
            _log(f"WARNING -- failed to close MLflow run: {exc}")


def log_params(params: Mapping[str, Any]) -> None:
    """Log run params. None values are kept (logged as the string 'None' --
    "there was no production model" is itself worth seeing in the UI).
    No-op outside an active run."""
    if not _RUN_ACTIVE:
        return
    try:
        import mlflow

        mlflow.log_params(dict(params))
    except Exception as exc:
        _log(f"WARNING -- MLflow log_params failed: {exc}")


def log_metrics(metrics: Mapping[str, Any], step: int | None = None) -> None:
    """Log run metrics. None values are dropped (mlflow requires floats);
    NaN/inf are passed through (informative, and the backend tolerates
    them). No-op outside an active run."""
    if not _RUN_ACTIVE:
        return
    clean = {k: float(v) for k, v in metrics.items() if v is not None}
    if not clean:
        return
    try:
        import mlflow

        mlflow.log_metrics(clean, step=step)
    except Exception as exc:
        _log(f"WARNING -- MLflow log_metrics failed: {exc}")


def set_tags(tags: Mapping[str, Any]) -> None:
    """Set run tags mid-run (decision label, target slot, model version --
    known only after the gate has run). None values are dropped. No-op
    outside an active run."""
    if not _RUN_ACTIVE:
        return
    clean = {k: v for k, v in tags.items() if v is not None}
    if not clean:
        return
    try:
        import mlflow

        mlflow.set_tags(clean)
    except Exception as exc:
        _log(f"WARNING -- MLflow set_tags failed: {exc}")


def log_dict(payload: Mapping[str, Any], artifact_file: str) -> None:
    """Write a dict as a JSON/YAML artifact (extension decides). No-op
    outside an active run."""
    if not _RUN_ACTIVE:
        return
    try:
        import mlflow

        mlflow.log_dict(dict(payload), artifact_file)
    except Exception as exc:
        _log(f"WARNING -- MLflow log_dict failed ({artifact_file}): {exc}")


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    """Attach a local file to the run. Silently skips a path that doesn't
    exist (a research script may not write every sibling report on every
    run). No-op outside an active run."""
    if not _RUN_ACTIVE:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        import mlflow

        mlflow.log_artifact(str(p), artifact_path)
    except Exception as exc:
        _log(f"WARNING -- MLflow log_artifact failed ({p}): {exc}")


def log_research_run(
    script_name: str,
    *,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    report_paths: Sequence[str | Path] | None = None,
    experiment: str | None = None,
    run_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
) -> None:
    """One-call MLflow logging for a measure_*/screen_* research run: open
    a run in the right experiment, log params + metrics, attach every
    report file that exists, close. A no-op when tracking is
    unconfigured/unreachable, like everything else here.

    `experiment` defaults by script-name prefix -- `screen_*` ->
    'research-screening', anything else (the `measure_*` family) ->
    'research-feature-gain' -- matching the experiment names agreed for
    this project. Pass it to override.
    """
    stem = Path(script_name).stem
    if experiment is None:
        experiment = "research-screening" if stem.startswith("screen_") else "research-feature-gain"
    run_tags = {"script": Path(script_name).name, **dict(tags or {})}
    with mlflow_run(experiment, run_name=run_name or stem, tags=run_tags):
        if params:
            log_params(params)
        if metrics:
            log_metrics(metrics)
        for rp in report_paths or []:
            log_artifact(rp)
