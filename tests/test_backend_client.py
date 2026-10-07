import requests

import pinance_ml.backend_client as backend_client
from pinance_ml.backend_client import post_retrain_event


def test_post_retrain_event_noop_when_url_not_configured(monkeypatch):
    monkeypatch.setattr(backend_client, "PREDICTOR_BACKEND_URL", "")
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **kw: calls.append((a, kw)))

    post_retrain_event("BTCUSDT", "point", "rejected", threshold=0.005, n_samples=100)

    assert calls == []


def test_post_retrain_event_posts_expected_payload(monkeypatch):
    monkeypatch.setattr(backend_client, "PREDICTOR_BACKEND_URL", "http://predictor-backend.local")
    captured = {}

    class _FakeResp:
        def raise_for_status(self):
            pass

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setattr(requests, "post", _fake_post)

    post_retrain_event(
        "BTCUSDT", "point", "promoted",
        candidate_version="v2", production_version="v1",
        metric_name="avg_mae (pushed_to_candidate)",
        candidate_value=0.0095, production_value=0.01,
        threshold=0.005, n_samples=42,
    )

    assert captured["url"] == "http://predictor-backend.local/admin/retrain-events"
    decided_at = captured["json"].pop("decided_at")
    assert decided_at.endswith("Z")  # required by POST /admin/retrain-events -- see module docstring
    assert captured["json"] == {
        "symbol": "BTCUSDT",
        "kind": "point",
        "decision": "promoted",
        "candidate_version": "v2",
        "production_version": "v1",
        "metric_name": "avg_mae (pushed_to_candidate)",
        "candidate_value": 0.0095,
        "production_value": 0.01,
        "threshold": 0.005,
        "n_samples": 42,
        "train_wall_seconds": None,
    }


def test_post_retrain_event_swallows_request_failures(monkeypatch):
    # A predictor-backend hiccup must never blow up the caller's actual
    # retrain/promotion run -- this is a best-effort monitoring side-channel.
    monkeypatch.setattr(backend_client, "PREDICTOR_BACKEND_URL", "http://predictor-backend.local")

    def _raise(*a, **kw):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "post", _raise)

    post_retrain_event("BTCUSDT", "point", "rejected", threshold=0.005, n_samples=100)


def test_post_retrain_event_strips_trailing_slash_from_base_url(monkeypatch):
    monkeypatch.setattr(backend_client, "PREDICTOR_BACKEND_URL", "http://predictor-backend.local/")
    captured = {}

    class _FakeResp:
        def raise_for_status(self):
            pass

    def _fake_post(url, json, timeout):
        captured["url"] = url
        return _FakeResp()

    monkeypatch.setattr(requests, "post", _fake_post)

    post_retrain_event("ETHUSDT", "quantile", "rejected", n_samples=0)

    assert captured["url"] == "http://predictor-backend.local/admin/retrain-events"
