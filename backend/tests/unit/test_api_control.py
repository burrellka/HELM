"""Endpoint tests for Phase 2/3/4, against a stub upstream.

Nothing here touches the live box. The stub answers with the real captured
fixtures, so these exercise the actual payload shapes from the Lemonade host.

The cases that matter: the confirm gates (an NPU eviction must not happen on
one click) and the no-admin-key refusals (a read-only HELM must say so, not
401 mysteriously).
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
    """Stands in for LemonadeClient. Records what would have been sent."""

    def __init__(self, *, admin: bool = True, health: dict | None = None):
        self._admin = admin
        self.health = health if health is not None else _fixture("health")
        self.sent: list[tuple[str, str, dict | None]] = []
        self.fail_with: UpstreamError | None = None
        #: When True with fail_with set, a failed /api/v1/load first evicts every
        #: OTHER model from health — Lemonade evicts to make room before the load
        #: lands, so a load that then fails takes them with it. Models the exact
        #: fleet-wipe the snapshot/restore protection exists for.
        self.evict_on_failed_load: bool = False
        #: What /api/v1/downloads returns (the download-jobs endpoint).
        self.downloads: list[dict] = []

    @property
    def has_admin_key(self) -> bool:
        return self._admin

    #: When True, /internal/pin returns success but does NOT mutate health, so
    #: the read-back fails. Simulates a pin that the server accepted but that
    #: did not take effect — the case the read-back guard exists to catch.
    pin_silently_noops: bool = False

    async def request(self, method, path, *, admin=False, timeout=None,
                      json_body=None, params=None):
        self.sent.append((method, path, json_body))
        if self.fail_with is not None and method == "POST":
            if path == "/api/v1/load" and self.evict_on_failed_load and json_body:
                target = json_body.get("model_name")
                self.health["all_models_loaded"] = [
                    m for m in self.health.get("all_models_loaded") or []
                    if m.get("model_name") == target
                ]
            raise self.fail_with
        if path == "/internal/pin":
            # Realistic: the flag flips in /health, so the read-back sees it.
            if not self.pin_silently_noops:
                for m in self.health.get("all_models_loaded") or []:
                    if m.get("model_name") == json_body.get("model_name"):
                        m["pinned"] = json_body.get("pinned")
            return {"ok": True}
        if path.endswith("/health"):
            return self.health
        if "/models" in path:
            return _fixture("models")
        if path.endswith("/system-stats"):
            return _fixture("system-stats")
        if path.endswith("/stats"):
            return _fixture("stats")
        if path.endswith("/system-info"):
            return _fixture("system-info")
        if path.endswith("/params"):
            return _fixture("params")
        if path.endswith("/downloads"):
            return self.downloads
        if path == "/live":
            return {}
        return {"ok": True}

    async def aclose(self):
        pass


@pytest.fixture
def app_client(data_dir, monkeypatch):
    """An authenticated client with a stub upstream."""
    from kblocallm.main import create_app

    app = create_app()
    with TestClient(app) as c:
        stub = StubClient()
        app.state.lemonade = stub
        c.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
        c.stub = stub
        yield c


@pytest.fixture
def readonly_client(data_dir):
    """A HELM with no admin key — a legitimate, supported mode."""
    from kblocallm.main import create_app

    app = create_app()
    with TestClient(app) as c:
        app.state.lemonade = StubClient(admin=False)
        c.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
        yield c


# -- read-only mode --------------------------------------------------------


@pytest.mark.parametrize("path,body", [
    ("/api/v1/control/load", {"model_name": "x"}),
    ("/api/v1/control/unload", {"model_name": "x"}),
    ("/api/v1/library/pull", {"model_name": "x"}),
    ("/api/v1/library/delete", {"model_name": "x", "confirm": True}),
    ("/api/v1/operations/warm", None),
])
def test_writes_refuse_clearly_without_an_admin_key(readonly_client, path, body):
    """URSA 1.1.8's lesson: never a bare 401 the operator has to interpret.
    503 plus the actual reason and the actual fix."""
    r = readonly_client.post(path, json=body)
    assert r.status_code == 503
    assert "admin key" in json.dumps(r.json()).lower()
    assert "HELM_LEMONADE_ADMIN_API_KEY" in json.dumps(r.json())


def test_reads_still_work_without_an_admin_key(readonly_client):
    """Read-only is a supported mode, not a broken one."""
    assert readonly_client.get("/api/v1/dashboard").status_code == 200


def test_operator_metadata_works_without_an_admin_key(readonly_client):
    """It writes to HELM's own DB, not the fleet. Gating it on the Lemonade
    key would be gatekeeping for no reason."""
    r = readonly_client.patch("/api/v1/library/qwen3.5-2b-FLM/metadata",
                              json={"favorite": True})
    assert r.status_code == 200 and r.json()["favorite"] is True


# -- the confirm gates -----------------------------------------------------


def test_an_npu_load_that_evicts_requires_confirmation(app_client):
    """Rule 7. One click must not silently take an agent's brain away."""
    app_client.stub.health = {"all_models_loaded": [
        {"model_name": "qwen3.5-2b-FLM", "recipe": "flm", "device": "npu"},
    ]}
    r = app_client.post("/api/v1/control/load", json={"model_name": "gemma4-it-e2b-FLM",
                                                      "recipe": "flm"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["evicts"] == ["qwen3.5-2b-FLM"]
    assert detail["needs_confirmation"] is True
    # And nothing was sent.
    assert not [s for s in app_client.stub.sent if s[1] == "/api/v1/load"]


def test_the_same_load_proceeds_once_confirmed(app_client):
    app_client.stub.health = {"all_models_loaded": [
        {"model_name": "qwen3.5-2b-FLM", "recipe": "flm", "device": "npu"},
    ]}
    r = app_client.post("/api/v1/control/load?confirm=true",
                        json={"model_name": "gemma4-it-e2b-FLM", "recipe": "flm"})
    assert r.status_code == 200 and r.json()["performed"] is True
    assert any(s[1] == "/api/v1/load" for s in app_client.stub.sent)


def test_unloading_a_pinned_model_requires_confirmation(app_client):
    r = app_client.post("/api/v1/control/unload",
                        json={"model_name": "Gemma-4-26B-A4B-it-MTP-GGUF"})
    assert r.status_code == 409
    codes = {i["code"] for i in r.json()["detail"]["issues"]}
    assert "unloading_pinned" in codes


def test_a_plan_preview_never_performs_anything(app_client):
    r = app_client.post("/api/v1/control/plan/load",
                        json={"model_name": "x", "recipe": "llamacpp", "slots": 8,
                              "ctx_per_slot": 65536})
    assert r.status_code == 200
    assert r.json()["computed_ctx_size"] == 524288
    assert not [s for s in app_client.stub.sent if s[0] == "POST"]


def test_a_blocked_load_is_422_and_says_why(app_client):
    r = app_client.post("/api/v1/control/load",
                        json={"model_name": "x", "recipe": "llamacpp", "extra_args": "-c 4096"})
    assert r.status_code == 422
    assert "reserved_args" in {i["code"] for i in r.json()["detail"]["issues"]}


def test_an_upstream_refusal_keeps_its_kind(app_client):
    """House rule #5: a rejected key is not a dead backend is not a timeout."""
    app_client.stub.fail_with = UpstreamError(
        kind=UpstreamErrorKind.REJECTED, message="NPU busy", status_code=409,
    )
    r = app_client.post("/api/v1/control/load?confirm=true", json={"model_name": "x"})
    assert r.status_code == 200
    assert r.json()["performed"] is False
    assert r.json()["error_kind"] == "rejected"


# -- fleet snapshot + restore-on-failure (P1) ------------------------------


def test_failed_load_reports_what_it_evicted(app_client):
    """The confirmed fleet-wipe: a load that fails still evicts resident models
    (pinned included). HELM must name them and offer them for restore."""
    app_client.stub.health = {"all_models_loaded": [
        {"model_name": "target", "recipe": "llamacpp", "loaded": True},
        {"model_name": "gemma-26b", "recipe": "llamacpp", "loaded": True, "pinned": True},
        {"model_name": "hermes-8b", "recipe": "llamacpp", "loaded": True, "pinned": False},
    ]}
    app_client.stub.fail_with = UpstreamError(
        kind=UpstreamErrorKind.UNREACHABLE, message="llama-server failed to start", status_code=500,
    )
    app_client.stub.evict_on_failed_load = True

    r = app_client.post("/api/v1/control/load?confirm=true", json={"model_name": "target"})
    body = r.json()
    assert body["performed"] is False
    assert set(body["evicted"]) == {"gemma-26b", "hermes-8b"}
    restorable = {m["model_name"]: m["pinned"] for m in body["restorable"]}
    assert restorable == {"gemma-26b": True, "hermes-8b": False}


def test_successful_load_reports_no_eviction(app_client):
    """No false alarm: a load that succeeds reports an empty evicted list even
    though other models are resident."""
    app_client.stub.health = {"all_models_loaded": [
        {"model_name": "other", "recipe": "llamacpp", "loaded": True, "pinned": True},
    ]}
    r = app_client.post("/api/v1/control/load?confirm=true", json={"model_name": "target"})
    body = r.json()
    assert body["performed"] is True
    assert body["evicted"] == []
    assert body["restorable"] == []


def test_restore_reloads_by_name_and_repins(app_client):
    """Restore reloads each evicted model BY NAME (Lemonade re-applies the saved
    recipe — HELM re-sends no options) and re-pins the ones that were pinned."""
    r = app_client.post("/api/v1/control/restore", json={"models": [
        {"model_name": "gemma-26b", "pinned": True},
        {"model_name": "hermes-8b", "pinned": False},
    ]})
    assert r.status_code == 200
    assert all(m["ok"] for m in r.json()["restored"])

    loads = [s for s in app_client.stub.sent if s[1] == "/api/v1/load"]
    # Each restored by name only — no ctx_size/llamacpp_args re-sent.
    assert {s[2]["model_name"] for s in loads} == {"gemma-26b", "hermes-8b"}
    assert all(set(s[2].keys()) == {"model_name"} for s in loads)
    # Only the pinned one got re-pinned.
    pins = [s for s in app_client.stub.sent if s[1] == "/internal/pin"]
    assert [p[2]["model_name"] for p in pins] == ["gemma-26b"]


# -- pin via the dedicated /internal/pin route (14c §2.1, trace-confirmed) --


def test_pin_uses_the_dedicated_route_not_a_reload(app_client):
    """Trace settled it: pin is POST /internal/pin {model_name, pinned}, no
    reload. Aevum is unpinned in the fixture, so pinning it is a real change."""
    r = app_client.post("/api/v1/control/pin",
                        json={"model_name": "Aevum-0.6B-Finetuned.Q4_K_M", "pinned": True})
    assert r.status_code == 200 and r.json()["performed"] is True
    # It hit /internal/pin, and it did NOT reload (no /api/v1/load).
    sent = app_client.stub.sent
    pin_calls = [s for s in sent if s[1] == "/internal/pin"]
    assert pin_calls, "pin must call /internal/pin"
    assert pin_calls[0][2] == {"model_name": "Aevum-0.6B-Finetuned.Q4_K_M", "pinned": True}
    assert not [s for s in sent if s[1] == "/api/v1/load"], "pin must NOT reload the model"


def test_unpin_sends_pinned_false(app_client):
    """The 26B is pinned in the fixture, so unpinning it is the real change."""
    r = app_client.post("/api/v1/control/pin",
                        json={"model_name": "Gemma-4-26B-A4B-it-MTP-GGUF", "pinned": False})
    assert r.status_code == 200
    pin_call = next(s for s in app_client.stub.sent if s[1] == "/internal/pin")
    assert pin_call[2]["pinned"] is False


def test_pin_is_verified_by_read_back(app_client):
    """House rule: never accept 'it worked'. When the server accepts the call
    but the flag does not flip, HELM reports unconfirmed, not success."""
    app_client.stub.pin_silently_noops = True
    r = app_client.post("/api/v1/control/pin",
                        json={"model_name": "Aevum-0.6B-Finetuned.Q4_K_M", "pinned": True})
    assert r.status_code == 200            # the HTTP call to HELM succeeded
    body = r.json()
    assert body["performed"] is False      # but HELM does not claim success
    assert body["error_kind"] == "unconfirmed"
    assert "did not read back" in body["error_detail"]


def test_pinning_an_already_pinned_model_is_a_no_op_not_a_reload(app_client):
    """The 26B is already pinned. Asking to pin it again should report done
    without making a needless call."""
    r = app_client.post("/api/v1/control/pin",
                        json={"model_name": "Gemma-4-26B-A4B-it-MTP-GGUF", "pinned": True})
    assert r.status_code == 200 and r.json()["performed"] is True
    assert not [s for s in app_client.stub.sent if s[1] == "/internal/pin"]


def test_pin_keeps_the_upstream_failure_kind(app_client):
    app_client.stub.fail_with = UpstreamError(
        kind=UpstreamErrorKind.UNAUTHORIZED, message="bad key", status_code=401,
    )
    r = app_client.post("/api/v1/control/pin",
                        json={"model_name": "Aevum-0.6B-Finetuned.Q4_K_M", "pinned": True})
    assert r.json()["performed"] is False
    assert r.json()["error_kind"] == "unauthorized"


def test_pinning_a_model_that_is_not_loaded_is_a_clear_404(app_client):
    r = app_client.post("/api/v1/control/pin", json={"model_name": "nope", "pinned": True})
    assert r.status_code == 404
    assert "nothing to pin" in r.json()["detail"]


def test_pin_needs_an_admin_key(readonly_client):
    r = readonly_client.post("/api/v1/control/pin",
                             json={"model_name": "x", "pinned": True})
    assert r.status_code == 503


# -- library ---------------------------------------------------------------


def test_the_library_lists_the_live_catalog(app_client):
    r = app_client.get("/api/v1/library")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 162
    assert body["unmatched_loaded"] == []


def test_the_library_flags_the_silent_tool_strippers(app_client):
    r = app_client.get("/api/v1/library", params={"recipes": "flm", "downloaded": "true"})
    entries = {e["model_id"]: e for e in r.json()["entries"]}
    assert entries["llama3.2-1b-FLM"]["strips_tools_silently"] is True
    assert entries["llama3.2-1b-FLM"]["label_join"] == "no_labels"
    assert entries["qwen3.5-2b-FLM"]["strips_tools_silently"] is False


def test_the_library_surfaces_live_drift(app_client):
    r = app_client.get("/api/v1/library", params={"text": "Gemma-4-26B-A4B-it-MTP"})
    entry = r.json()["entries"][0]
    assert entry["drift"]["has_drift"] is True


def test_metadata_round_trips_and_filters(app_client):
    app_client.patch("/api/v1/library/hermes-3-llama-3.1-8b-lorablated.Q4_K_M/metadata",
                     json={"tags": ["uncensored-do-not-deploy"], "description": "do not deploy"})
    r = app_client.get("/api/v1/library", params={"tags": "uncensored-do-not-deploy"})
    assert [e["model_id"] for e in r.json()["entries"]] == [
        "hermes-3-llama-3.1-8b-lorablated.Q4_K_M"]


def test_metadata_patch_does_not_clear_omitted_fields(app_client):
    path = "/api/v1/library/qwen3.5-2b-FLM/metadata"
    app_client.patch(path, json={"description": "Karl's travel brain"})
    app_client.patch(path, json={"favorite": True})
    body = app_client.patch(path, json={}).json()
    assert body["description"] == "Karl's travel brain"
    assert body["favorite"] is True


def test_metadata_export_import_round_trip(app_client):
    app_client.patch("/api/v1/library/qwen3.5-2b-FLM/metadata",
                     json={"tags": ["a", "b"], "favorite": True})
    exported = app_client.get("/api/v1/library/export").json()
    assert exported["metadata"]["qwen3.5-2b-FLM"]["tags"] == ["a", "b"]
    assert app_client.post("/api/v1/library/import", json=exported).json()["imported"] >= 1


def test_delete_requires_confirmation(app_client):
    r = app_client.post("/api/v1/library/delete", json={"model_name": "gemma4-it-e2b-FLM"})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "needs_confirmation"


def test_delete_refuses_a_loaded_model(app_client):
    r = app_client.post("/api/v1/library/delete",
                        json={"model_name": "Gemma-4-26B-A4B-it-MTP-GGUF", "confirm": True})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "loaded"


def test_pull_is_server_owned_so_it_survives_a_refresh(app_client):
    """§5.5's entire point: stream:true + subscribe:false hands the job to
    Lemonade, so closing the tab does not kill a 6GB download."""
    app_client.post("/api/v1/library/pull",
                    json={"model_name": "user.Thing", "checkpoint": "org/repo:Q4"})
    sent = next(s for s in app_client.stub.sent if s[1] == "/api/v1/pull")
    assert sent[2]["stream"] is True and sent[2]["subscribe"] is False


def test_a_user_prefixed_pull_keeps_its_checkpoint_and_recipe(app_client):
    """Custom pulls are the ONLY case Lemonade accepts a checkpoint/recipe."""
    app_client.post("/api/v1/library/pull",
                    json={"model_name": "user.Thing", "checkpoint": "org/repo:Q4",
                          "recipe": "llamacpp"})
    sent = next(s for s in app_client.stub.sent if s[1] == "/api/v1/pull")
    assert sent[2]["checkpoint"] == "org/repo:Q4" and sent[2]["recipe"] == "llamacpp"


def test_a_builtin_pull_drops_the_checkpoint_so_lemonade_does_not_400(app_client):
    """The Download-button bug: Lemonade hard-400s a pull that sends a checkpoint
    for a non-user. model. HELM pulls the built-in BY NAME (Lemonade resolves the
    checkpoint from its own catalog) and warns instead of forwarding the 400 as a
    502. The frontend passes the catalog checkpoint for convenience; the backend
    strips it here."""
    r = app_client.post("/api/v1/library/pull",
                        json={"model_name": "Bonsai-4B-gguf", "checkpoint": "org/repo:Q1_0"})
    assert r.status_code == 200
    sent = next(s for s in app_client.stub.sent if s[1] == "/api/v1/pull")
    assert sent[2]["model_name"] == "Bonsai-4B-gguf"     # not rewritten
    assert "checkpoint" not in sent[2]                    # dropped, not forwarded
    assert "recipe" not in sent[2]
    assert r.json()["warnings"]                            # said what it did


def test_a_plain_builtin_pull_sends_only_the_name(app_client):
    app_client.post("/api/v1/library/pull", json={"model_name": "Bonsai-1.7B-gguf"})
    sent = next(s for s in app_client.stub.sent if s[1] == "/api/v1/pull")
    assert sent[2] == {"model_name": "Bonsai-1.7B-gguf", "stream": True, "subscribe": False}


def test_the_library_never_leaks_host_paths_by_default(app_client):
    """F11: ?include_paths=true publishes the host filesystem layout. It must
    never ride along on a routine list."""
    app_client.get("/api/v1/library")
    assert not any("include_paths" in str(s) for s in app_client.stub.sent)


# -- operations ------------------------------------------------------------


def test_the_warm_set_starts_empty_and_says_why(app_client):
    body = app_client.get("/api/v1/operations/warm-set").json()
    assert body["models"] == []
    assert "operator-triggered only" in body["note"]
    assert "no restart signal" in body["note"]


def test_warming_never_auto_warms_an_npu_model(app_client):
    """Rule 7 unattended: warming an NPU model evicts the incumbent with
    nobody watching. Skipped and reported, never silently dropped."""
    app_client.post("/api/v1/operations/warm-set", json={"model_id": "qwen3.5-2b-FLM"})
    r = app_client.post("/api/v1/operations/warm")
    assert r.status_code == 409
    skipped = r.json()["detail"]["skipped"]
    assert skipped[0]["model_id"] == "qwen3.5-2b-FLM"
    assert "NPU model" in skipped[0]["reason"]


def test_warming_requires_confirmation_because_the_box_may_be_mid_incident(app_client):
    app_client.post("/api/v1/operations/warm-set", json={"model_id": "gemma-3-4b-it-Q4_K_M"})
    r = app_client.post("/api/v1/operations/warm")
    assert r.status_code == 409
    assert "mid-incident" in r.json()["detail"]["message"]


def test_warming_skips_already_loaded_models(app_client):
    app_client.post("/api/v1/operations/warm-set",
                    json={"model_id": "Gemma-4-26B-A4B-it-MTP-GGUF"})
    r = app_client.post("/api/v1/operations/warm?confirm=true")
    assert r.json()["skipped"][0]["reason"] == "already loaded"


def test_log_classification_is_served_from_the_tested_rules(app_client):
    r = app_client.post("/api/v1/operations/logs/classify",
                        json={"lines": ["Unloading models to clear hardware state",
                                        "Largest memory pool: 0.50 GB"]})
    events = [line["event"] for line in r.json()["lines"]]
    assert events == ["mass_evict", "pool"]


def test_benchmark_rejects_an_unknown_scenario(app_client):
    r = app_client.post("/api/v1/operations/benchmark",
                        json={"model_name": "x", "scenario": "nonsense"})
    assert r.status_code == 422


def test_benchmark_stores_its_config_note(app_client):
    """A tok/s number without its slot/context config is not comparable to the
    next one, and comparing them anyway is how folklore forms."""
    app_client.post("/api/v1/operations/benchmark",
                    json={"model_name": "Gemma-4-26B-A4B-it-MTP-GGUF", "scenario": "chat"})
    runs = app_client.get("/api/v1/operations/benchmarks").json()["runs"]
    assert runs and "ctx_size=524288" in runs[0]["config_note"]


# -- download jobs: the header indicator's data source ---------------------


def test_jobs_endpoint_normalizes_the_active_downloads(app_client):
    """The header indicator polls this. It must return ONLY the pulls running
    right now — a mixed list that also holds finished/failed jobs would light
    the indicator for a download that is already over."""
    app_client.stub.downloads = [
        {"id": "model:user.A", "model_name": "user.A", "running": True, "complete": False,
         "error": None, "percent": 42, "overall_bytes_downloaded": 100, "total_download_size": 240,
         "status": "downloading"},
        {"id": "model:user.B", "model_name": "user.B", "running": False, "complete": True,
         "error": None, "percent": 100},                                    # finished
        {"id": "model:user.C", "model_name": "user.C", "running": False, "complete": False,
         "error": "429 from HuggingFace", "percent": 0, "status": "error"},  # failed
    ]
    r = app_client.get("/api/v1/library/jobs").json()
    assert r["active_count"] == 1
    active = r["active"]
    assert [j["model_name"] for j in active] == ["user.A"]
    assert active[0]["percent"] == 42
    assert active[0]["bytes_downloaded"] == 100 and active[0]["bytes_total"] == 240
    # The full list is still returned for anyone who wants it.
    assert len(r["jobs"]) == 3


def test_jobs_endpoint_is_empty_when_nothing_is_downloading(app_client):
    app_client.stub.downloads = []
    r = app_client.get("/api/v1/library/jobs").json()
    assert r["active_count"] == 0 and r["active"] == []


# ── read-back verification of load options (the --parallel-drop safety net) ──

def test_load_flags_when_options_do_not_take(app_client):
    """The bug that ran silent for days: Lemonade accepts the reload but comes
    back running different options. `performed` is True, but `verified` must be
    False with a note — never a silent success. The stub's /health keeps the
    26B at --parallel 8, so asking for 2 slots is a mismatch it must catch."""
    r = app_client.post("/api/v1/control/load?confirm=true", json={
        "model_name": "Gemma-4-26B-A4B-it-MTP-GGUF", "recipe": "llamacpp",
        "slots": 2, "ctx_per_slot": 65536, "save_options": True,
    })
    body = r.json()
    assert body["performed"] is True
    assert body["verified"] is False
    assert "slots" in (body["verify_note"] or "")


def test_load_verified_true_when_options_match(app_client):
    """Requesting the 26B's actual running config (8 slots x 64K) reads back
    clean, so verified is True and there is no warning."""
    r = app_client.post("/api/v1/control/load?confirm=true", json={
        "model_name": "Gemma-4-26B-A4B-it-MTP-GGUF", "recipe": "llamacpp",
        "slots": 8, "ctx_per_slot": 65536, "save_options": True,
    })
    body = r.json()
    assert body["performed"] is True
    assert body["verified"] is True
    assert body["verify_note"] is None
