import builtins

import pinance_ml.tracking as tracking
from pinance_ml.tracking import (
    log_artifact,
    log_dict,
    log_metrics,
    log_params,
    log_research_run,
    mlflow_run,
    set_tags,
)


def test_mlflow_run_is_noop_context_when_uri_unset(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    seen = []
    with mlflow_run("retrain-point", run_name="BTCUSDT", tags={"kind": "point"}) as run:
        seen.append(run)
        # log_* inside a no-op context must also be no-ops, never raise
        log_params({"symbol": "BTCUSDT", "production_model_version": None})
        log_metrics({"avg_mae": 0.01, "improvement": None})
        set_tags({"decision": "rejected"})
        log_dict({"a": 1}, "eval_metrics.json")
        log_artifact("does/not/exist.csv")
    assert seen == [None]
    assert tracking._RUN_ACTIVE is False


def test_log_helpers_are_noop_outside_any_run(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    # no `with mlflow_run(...)` at all
    log_params({"x": 1})
    log_metrics({"y": 2.0})
    set_tags({"z": "w"})
    log_dict({"z": 3}, "f.json")
    log_artifact(__file__)  # a real file, but still nothing should happen


def test_log_research_run_is_noop_when_uri_unset(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    log_research_run(
        "measure_funding_feature_gain.py",
        params={"symbols": "BTCUSDT", "pinball_improvement_bar": 0.01},
        metrics={"pooled_pinball_delta_pct": -0.3, "p_value": 0.2},
        report_paths=["reports/funding_gain_folds.csv", "reports/does_not_exist.log"],
    )


def test_mlflow_run_swallows_missing_mlflow(monkeypatch):
    # URI configured, but `import mlflow` blows up -> still a no-op context,
    # body still runs, nothing raises.
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "http://vds3.local:5000")
    real_import = builtins.__import__

    def _no_mlflow(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("simulated: mlflow not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_mlflow)

    ran = False
    with mlflow_run("retrain-point", run_name="BTCUSDT") as run:
        assert run is None
        ran = True
        log_metrics({"avg_mae": 0.01})
    assert ran is True
    assert tracking._RUN_ACTIVE is False


def test_source_commit_never_raises():
    # Whatever the environment, this degrades to a string (possibly empty).
    assert isinstance(tracking._source_commit(), str)
