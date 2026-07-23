"""Config-surface endpoint tests (14c §2.3), against the stub upstream.

The pipeline's guarantees are what these check: read-first, refuse-if-blocked,
confirm-before-reload, back-up-before-write, and Level B apply staying honestly
unwired.
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
    def __init__(self, *, admin: bool = True):
        self._admin = admin
        self.health = _fixture("health")
        self.sent: list[tuple[str, str, dict | None]] = []
        self.fail_with: UpstreamError | None = None

    @property
    def has_admin_key(self) -> bool:
        return self._admin

    async def request(self, method, path, *, admin=False, timeout=None, json_body=None, params=None):
        self.sent.append((method, path, json_body))
        if self.fail_with is not None and method == "POST" and "/load" in path:
            raise self.fail_with
        if path == "/api/v1/load":
            # Reflect the applied options into /health so the read-back sees them.
            opts = (json_body or {}).get("recipe_options", {})
            for m in self.health.get("all_models_loaded") or []:
                if m.get("model_name") == json_body.get("model_name"):
                    m["recipe_options"] = opts
            return {"ok": True}
        if path.endswith("/health"):
            return self.health
        if "/models" in path:
            return _fixture("models")
        if path.endswith("/params"):
            return _fixture("params")
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


@pytest.fixture
def readonly(data_dir):
    from kblocallm.main import create_app
    app = create_app()
    with TestClient(app) as c:
        app.state.lemonade = StubClient(admin=False)
        c.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
        yield c


MODEL = "Gemma-4-26B-A4B-it-MTP-GGUF"


# -- read-first ------------------------------------------------------------


def test_read_never_hands_back_a_blank_editor(client):
    """14c §2.3 step 1: load the live config first, always."""
    r = client.get(f"/api/v1/config/model/{MODEL}")
    assert r.status_code == 200
    body = r.json()
    assert "524288" in body["running_text"]     # the real running options
    assert body["max_context_window"] == 262144  # the ceiling, for validation


# -- validate (no write) ---------------------------------------------------


def test_validate_is_a_dry_run(client):
    r = client.post(f"/api/v1/config/model/{MODEL}/validate",
                    json={"text": '{"llamacpp_args": "--ctx-size 4096"}'})
    assert r.status_code == 200 and r.json()["blocked"] is True
    assert not [s for s in client.stub.sent if s[0] == "POST" and "/load" in s[1]]


# -- apply pipeline --------------------------------------------------------


def test_apply_needs_confirmation_because_it_reloads(client):
    r = client.post(f"/api/v1/config/model/{MODEL}/apply",
                    json={"text": '{"ctx_size": 262144, "llamacpp_args": "--parallel 8"}'})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "needs_confirmation"
    assert not [s for s in client.stub.sent if s[1] == "/api/v1/load"]


def test_apply_backs_up_before_writing(client):
    """14c §2.3 step 5: the prior version is snapshotted before the write, so
    a bad edit is always one click from being walked back."""
    r = client.post(f"/api/v1/config/model/{MODEL}/apply?confirm=true",
                    json={"text": '{"ctx_size": 262144, "llamacpp_args": "--parallel 8"}'})
    assert r.status_code == 200 and r.json()["applied"] is True
    assert r.json()["backup_id"] >= 1
    # The backup holds the PRIOR options (the 524288 that was there before).
    backups = client.get("/api/v1/config/backups", params={"scope": f"model:{MODEL}"}).json()
    assert "524288" in backups["backups"][0]["preview"]


def test_apply_verifies_by_read_back(client):
    r = client.post(f"/api/v1/config/model/{MODEL}/apply?confirm=true",
                    json={"text": '{"ctx_size": 262144, "llamacpp_args": "--parallel 8"}'})
    assert r.json()["verified"] is True
    sent = next(s for s in client.stub.sent if s[1] == "/api/v1/load")
    assert sent[2]["save_options"] is True
    assert sent[2]["recipe_options"]["ctx_size"] == 262144


def test_a_blocked_edit_never_reaches_apply(client):
    r = client.post(f"/api/v1/config/model/{MODEL}/apply?confirm=true",
                    json={"text": '{"ctx_size": "not a number"}'})
    assert r.status_code == 422
    assert not [s for s in client.stub.sent if s[1] == "/api/v1/load"]


def test_apply_keeps_the_upstream_failure_kind(client):
    client.stub.fail_with = UpstreamError(kind=UpstreamErrorKind.REJECTED, message="no")
    r = client.post(f"/api/v1/config/model/{MODEL}/apply?confirm=true",
                    json={"text": '{"ctx_size": 262144, "llamacpp_args": "--parallel 8"}'})
    assert r.json()["applied"] is False and r.json()["error_kind"] == "rejected"
    # Backup still taken before the attempt, so nothing is lost.
    assert r.json()["backup_id"] >= 1


# -- restore ---------------------------------------------------------------


def test_restore_replays_a_backup(client):
    apply = client.post(f"/api/v1/config/model/{MODEL}/apply?confirm=true",
                        json={"text": '{"ctx_size": 262144, "llamacpp_args": "--parallel 8"}'})
    backup_id = apply.json()["backup_id"]
    r = client.post(f"/api/v1/config/backups/{backup_id}/restore?confirm=true")
    assert r.status_code == 200 and r.json()["restored"] is True
    # It reloaded with the backed-up (original 524288) options.
    last_load = [s for s in client.stub.sent if s[1] == "/api/v1/load"][-1]
    assert last_load[2]["recipe_options"]["ctx_size"] == 524288


def test_restore_needs_confirmation(client):
    apply = client.post(f"/api/v1/config/model/{MODEL}/apply?confirm=true",
                        json={"text": '{"ctx_size": 262144}'})
    backup_id = apply.json()["backup_id"]
    assert client.post(f"/api/v1/config/backups/{backup_id}/restore").status_code == 409


# -- Level B: read + validate wired, apply honestly not ---------------------


def test_global_read_returns_the_live_config(client):
    r = client.get("/api/v1/config/global")
    assert r.status_code == 200
    assert r.json()["current"]["enable_dgpu_gtt"] is True
    assert "_generated" not in r.json()["current"]  # banner stripped


def test_disabling_gtt_globally_is_flagged_as_danger(client):
    r = client.post("/api/v1/config/global/validate",
                    json={"text": json.dumps({"enable_dgpu_gtt": False}),
                          "current": {"enable_dgpu_gtt": True}})
    assert r.json()["needs_danger_ack"] is True
    assert "enable_dgpu_gtt" in r.json()["dangerous_keys"]


def test_global_apply_refuses_honestly_until_the_mechanism_is_traced(client):
    r = client.post("/api/v1/config/global/apply",
                    json={"text": json.dumps({"log_level": "debug"})})
    assert r.status_code == 501
    assert r.json()["detail"]["code"] == "apply_not_wired"
    assert "lemonade config set" in r.json()["detail"]["message"]


# -- read-only mode --------------------------------------------------------


def test_config_writes_refuse_without_an_admin_key(readonly):
    r = readonly.post(f"/api/v1/config/model/{MODEL}/apply?confirm=true",
                      json={"text": "{}"})
    assert r.status_code == 503


def test_reading_config_works_without_an_admin_key(readonly):
    """Reading and validating are safe; only applying needs the key."""
    assert readonly.get(f"/api/v1/config/model/{MODEL}").status_code == 200
