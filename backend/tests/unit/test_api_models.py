"""Per-model ctx-check endpoint (14c §1.12), against a stub upstream.

These assert the endpoint's honesty contract end to end: attribute /slots to a
model by slot count, return match/drift/unknown, and NEVER 502-into-silence or
collapse a live-read failure into "no drift".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kblocallm.upstream.client import UpstreamError, UpstreamErrorKind

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


class StubClient:
    def __init__(self):
        self.health = _fixture("health")
        #: What /slots returns. Default: the 26B's 8 slots @ 65536 (the real box
        #: shape — /slots resolves to one instance and carries no model label).
        self.slots = [{"id": i, "n_ctx": 65536, "is_processing": False} for i in range(8)]
        self.slots_fail: UpstreamError | None = None

    @property
    def has_admin_key(self) -> bool:
        return True

    async def request(self, method, path, *, admin=False, timeout=None, json_body=None, params=None):
        if path.endswith("/health"):
            return self.health
        if path.endswith("/slots"):
            if self.slots_fail is not None:
                raise self.slots_fail
            return self.slots
        return {"ok": True}

    async def aclose(self):
        pass


@pytest.fixture
def client(data_dir):
    from kblocallm.main import create_app
    app = create_app()
    with TestClient(app) as c:
        app.state.lemonade = StubClient()
        c.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
        c.stub = app.state.lemonade
        yield c


B26 = "Gemma-4-26B-A4B-it-MTP-GGUF"
AEVUM = "Aevum-0.6B-Finetuned.Q4_K_M"


def test_26b_matches_when_slots_are_attributable(client):
    r = client.get(f"/api/v1/models/{B26}/ctx-check")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "match"
    assert body["computed_per_slot"] == 65536
    assert body["running_n_ctx"] == 65536


def test_26b_reports_drift_when_running_ctx_differs(client):
    client.stub.slots = [{"id": i, "n_ctx": 40960} for i in range(8)]
    body = client.get(f"/api/v1/models/{B26}/ctx-check").json()
    assert body["state"] == "drift"
    assert body["computed_per_slot"] == 65536
    assert body["running_n_ctx"] == 40960


def test_aevum_is_unknown_because_slots_are_not_its_own(client):
    """The real-box case: /slots resolves to the 26B (8 slots); Aevum has 1.
    The endpoint must NOT report Aevum's context from the 26B's slots."""
    body = client.get(f"/api/v1/models/{AEVUM}/ctx-check").json()
    assert body["state"] == "unknown"
    assert body["running_n_ctx"] is None


def test_unloaded_model_is_404(client):
    r = client.get("/api/v1/models/Not-Loaded-Model/ctx-check")
    assert r.status_code == 404


def test_slots_failure_degrades_to_unknown_not_error(client):
    client.stub.slots_fail = UpstreamError(UpstreamErrorKind.TIMEOUT, "slow")
    r = client.get(f"/api/v1/models/{B26}/ctx-check")
    assert r.status_code == 200
    assert r.json()["state"] == "unknown"


def test_health_failure_is_502_not_a_silent_no_drift(client):
    async def boom(*a, **k):
        raise UpstreamError(UpstreamErrorKind.UNREACHABLE, "box down")
    client.stub.request = boom
    r = client.get(f"/api/v1/models/{B26}/ctx-check")
    assert r.status_code == 502


def test_batch_returns_one_row_per_loaded_llamacpp_model(client):
    """The dashboard shape: every loaded llamacpp model, one /slots read.
    On this fixture: 26B (unique 8 slots) matches; the two 1-slot models are
    unknown because the live slots aren't theirs."""
    r = client.get("/api/v1/models/ctx-drift")
    assert r.status_code == 200
    rows = {row["model_name"]: row for row in r.json()["results"]}
    assert set(rows) == {B26, AEVUM, "Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M"}
    assert rows[B26]["state"] == "match"
    assert rows[AEVUM]["state"] == "unknown"


def test_batch_degrades_to_unknown_when_slots_fail(client):
    client.stub.slots_fail = UpstreamError(UpstreamErrorKind.TIMEOUT, "slow")
    rows = client.get("/api/v1/models/ctx-drift").json()["results"]
    assert rows and all(row["state"] == "unknown" for row in rows)


def test_batch_502_when_health_unreachable(client):
    async def boom(*a, **k):
        raise UpstreamError(UpstreamErrorKind.UNREACHABLE, "box down")
    client.stub.request = boom
    assert client.get("/api/v1/models/ctx-drift").status_code == 502


def test_requires_auth(data_dir):
    from kblocallm.main import create_app
    app = create_app()
    with TestClient(app) as c:
        app.state.lemonade = StubClient()
        c.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
        c.cookies.clear()
        r = c.get(f"/api/v1/models/{B26}/ctx-check")
        assert r.status_code in (401, 403)
