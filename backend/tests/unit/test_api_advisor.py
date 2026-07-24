"""Model Advisor API tests. The provenance boundaries are what these guard:
Layer 1 always present, Layer 2 honest about absence, Layer 3 clearly labeled
and never fabricated on failure."""
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
        self._admin = True
        self.chat_reply = "This model has 26B parameters and fits the pool."
        self.chat_fails = False
        #: When set, simulate a reasoning model that ate its whole budget: empty
        #: content, reasoning present, finish_reason=length.
        self.chat_starved = False
        #: The last chat/completions body, so a test can assert WHICH model was
        #: called and WHAT context it was fed (the summary path is verified this
        #: way: right model, starved prompt).
        self.last_chat_body = None

    @property
    def has_admin_key(self) -> bool:
        return self._admin

    async def request(self, method, path, *, admin=False, timeout=None, json_body=None, params=None):
        if "chat/completions" in path:
            self.last_chat_body = json_body
            if self.chat_fails:
                raise UpstreamError(kind=UpstreamErrorKind.UNREACHABLE, message="no model")
            if self.chat_starved:
                return {"choices": [{"finish_reason": "length", "message": {
                    "content": "", "reasoning_content": "thinking hard about this...", "role": "assistant"}}]}
            return {"choices": [{"finish_reason": "stop", "message": {"content": self.chat_reply}}]}
        if "/models" in path:
            return _fixture("models")
        if path.endswith("/system-info"):
            return _fixture("system-info")
        if path.endswith("/system-stats"):
            return _fixture("system-stats")
        return {"ok": True}

    async def stream_lines(self, method, path, *, admin=False, json_body=None):
        """Mimic Lemonade's streamed chat: reasoning deltas first, then content,
        then a usage-only trailing chunk, then [DONE]. Honours chat_fails /
        chat_starved so the streaming path is tested against the same modes."""
        self.last_chat_body = json_body
        if self.chat_fails:
            raise UpstreamError(kind=UpstreamErrorKind.UNREACHABLE, message="no model")

        def frame(obj):
            return "data: " + json.dumps(obj)

        # Reasoning channel first (the 26B thinks before it answers).
        yield frame({"choices": [{"delta": {"reasoning_content": "let me think..."}}]})
        if not self.chat_starved:
            for piece in ["Hello", " there", "."]:
                yield frame({"choices": [{"delta": {"content": piece}}]})
        # Usage-only trailing chunk, then DONE.
        yield frame({"choices": [], "usage": {
            "prompt_tokens": 8030, "completion_tokens": 3, "total_tokens": 8033,
            "prompt_tokens_details": {"cached_tokens": 8000}}})
        yield "data: [DONE]"

    async def aclose(self):
        pass


@pytest.fixture
def client(data_dir, monkeypatch):
    # Block real HF egress in tests: force the metadata service to "not found"
    # unless a test overrides it, so nothing here touches the network.
    from kblocallm import advisor_service

    async def _no_network(self, repo_id, *, force=False):
        return advisor_service.HFResult("not_found", {}, cached=False,
                                        reason="network disabled in tests")
    monkeypatch.setattr(advisor_service.HFMetadataService, "fetch", _no_network)

    from kblocallm.main import create_app
    app = create_app()
    with TestClient(app) as c:
        app.state.lemonade = StubClient()
        c.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
        c.stub = app.state.lemonade
        yield c


GEMMA = "Gemma-4-26B-A4B-it-MTP-GGUF"
AEVUM = "Aevum-0.6B-Finetuned.Q4_K_M"


# -- Layer 1: always there, deterministic ----------------------------------


def test_analyze_returns_local_facts_without_running_a_model(client):
    r = client.post(f"/api/v1/advisor/analyze/{GEMMA}")
    assert r.status_code == 200
    facts = r.json()["local_facts"]
    assert facts["parameters"]["total_b"] == 26.0
    assert facts["parameters"]["moe"] is True and facts["parameters"]["active_b"] == 4.0
    assert facts["quant"]["label"] == "Q4_K_M"
    assert facts["tool_calling"] is True
    assert facts["fit"]["verdict"] == "fits_resident"
    # No model was asked to synthesize.
    assert r.json()["synthesis"] is None


def test_the_quant_explainer_is_in_the_facts(client):
    facts = client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()["local_facts"]
    assert "sweet spot" in facts["quant"]["explanation"]
    assert "size_impact" in facts["quant"]


# -- Layer 2: honest about absence -----------------------------------------


def test_a_custom_finetune_has_no_external_record_not_an_invented_one(client):
    """Aevum is a local-path finetune. The external layer must say 'no record',
    never guess a repo (§1.8)."""
    external = client.post(f"/api/v1/advisor/analyze/{AEVUM}").json()["external"]
    assert external["repo_id"] is None
    assert external["status"] == "no_external_record"
    assert external["metadata"] is None


def test_an_explicit_repo_is_resolved_and_fetched(client):
    """Gemma's checkpoint carries unsloth/... so the repo resolves. The stubbed
    fetch returns not_found, and that is reported honestly, not as an error or
    an invention."""
    external = client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()["external"]
    assert external["repo_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert external["status"] == "not_found"
    assert external["metadata"] is None


def test_a_remembered_pull_checkpoint_restores_the_repo(client):
    """When Lemonade only has a local path (as it does for Aevum), but HELM
    remembered the HF checkpoint from the pull, the advisor resolves from THAT —
    still an explicit operator-provided repo, never a guess. This is the fix for
    'everything I download shows no external record'."""
    from kblocallm.advisor_service import ModelSourceStore
    ModelSourceStore(client.app.state.db).set(AEVUM, "someorg/aevum-gguf:Q4_K_M")
    external = client.post(f"/api/v1/advisor/analyze/{AEVUM}").json()["external"]
    # It now resolves the repo (and honestly reports the stubbed not_found),
    # instead of "no external record".
    assert external["repo_id"] == "someorg/aevum-gguf"
    assert external["status"] == "not_found"


def test_without_a_remembered_checkpoint_a_local_path_still_says_no_record(client):
    """No invention: a local-path model with nothing remembered stays honest."""
    external = client.post(f"/api/v1/advisor/analyze/{AEVUM}").json()["external"]
    assert external["repo_id"] is None and external["status"] == "no_external_record"


# -- Layer 3: labeled, never fabricated on failure -------------------------


def test_synthesis_is_labeled_generated(client):
    r = client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true")
    syn = r.json()["synthesis"]
    assert syn["generated"] is True and syn["available"] is True
    assert syn["model"] == GEMMA  # the default analysis model (26B)
    assert "Not an independent source" in syn["label"]


def test_synthesis_failure_is_honest_not_faked(client):
    """If the analysis model cannot answer, the synthesis says so. It never
    invents a summary — that would be the exact §6 failure moved into prose."""
    client.stub.chat_fails = True
    syn = client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true").json()["synthesis"]
    assert syn["available"] is False
    assert "unavailable" in syn["note"]


def test_reasoning_starvation_is_named_not_a_blank(client):
    """The token-budget gotcha: a reasoning model spends its whole budget on
    the hidden reasoning_content channel and emits empty content at
    finish_reason=length. HELM must recognise that and point at the fix, not
    report a bare 'returned nothing' — and it must NOT surface the reasoning
    scratchpad as if it were the answer."""
    client.stub.chat_starved = True
    syn = client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true").json()["synthesis"]
    assert syn["available"] is False
    assert syn["finish_reason"] == "length"
    assert "reasoning" in syn["note"] and "non-reasoning analyst" in syn["note"]
    assert "thinking hard" not in str(syn)  # the scratchpad is never the answer


# -- settings: default 26B, operator-selectable, sub-2B flagged -------------


def test_default_analysis_model_is_the_26b_but_not_hardcoded(client):
    s = client.get("/api/v1/advisor/settings").json()
    assert s["settings"]["analysis_model"] == GEMMA
    assert s["settings"]["analysis_model_is_default"] is True
    assert s["operational_prompt_locked"] is True


def test_the_picker_flags_sub_2b_models_as_fabricating(client):
    picker = client.get("/api/v1/advisor/settings").json()["analysis_models"]
    by_id = {o["id"]: o for o in picker}
    assert by_id[AEVUM]["not_recommended"] is True   # 0.6B
    assert "fabricates" in by_id[AEVUM]["warning"]
    assert by_id[GEMMA]["not_recommended"] is False  # 26B


def test_operator_can_choose_a_different_analysis_model(client):
    client.put("/api/v1/advisor/settings",
               json={"analysis_model": "Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M"})
    s = client.get("/api/v1/advisor/settings").json()["settings"]
    assert s["analysis_model"] == "Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M"
    assert s["analysis_model_is_default"] is False


def test_persona_is_editable_but_operational_prompt_is_not_a_field(client):
    """§1.7: the operational (no-fabrication) layer is locked in code. The API
    exposes no way to edit it."""
    r = client.put("/api/v1/advisor/settings", json={"persona_prompt": "Be terse."})
    assert r.json()["persona_prompt"] == "Be terse."
    # There is no operational_prompt field to set.
    assert "operational_prompt" not in r.json()


# -- chat: grounded on a model ---------------------------------------------


def test_the_advisor_grounds_in_detected_hardware_not_baked_specs(client, monkeypatch):
    """Portability: the advisor's hardware context comes from the live box
    (system-info), so HELM deployed elsewhere reasons about that box. The
    detected pool (120) must be in the prompt, and the kernel must instruct
    grounding in it."""
    captured = {}
    orig = client.stub.request

    async def spy(method, path, **kw):
        if "chat/completions" in path:
            captured["system"] = kw["json_body"]["messages"][0]["content"]
        return await orig(method, path, **kw)
    monkeypatch.setattr(client.stub, "request", spy)

    client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true")
    sysmsg = captured["system"]
    assert "this_machine_detected_hardware" in sysmsg
    assert "DETECTED from this box" in sysmsg
    assert "120" in sysmsg                              # the live pool, not a baked 128
    assert "DETECTED hardware provided in the context" in sysmsg  # the kernel rule


def test_the_default_analysis_model_falls_back_when_absent_on_this_box(client, monkeypatch):
    """Portability: the shipped default assumes the 26B. On a box that does not
    have it, the DEFAULT resolves to the largest suitable owned model rather
    than failing. An explicit operator choice is still respected as-is."""
    from kblocallm.api import advisor as adv_api

    async def only_small(client_):
        return [
            {"id": "tiny-1b", "params_b": 1.0, "not_recommended": True, "warning": "x"},
            {"id": "hermes-8b", "params_b": 8.0, "not_recommended": False, "warning": None},
        ]
    monkeypatch.setattr(adv_api, "_analysis_model_options", only_small)

    # Default (26B) is absent -> falls back to the 8B (largest suitable).
    m = client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true").json()["synthesis"]["model"]
    assert m == "hermes-8b"


def test_an_explicit_choice_is_not_auto_overridden(client, monkeypatch):
    from kblocallm.api import advisor as adv_api

    async def only_small(client_):
        return [{"id": "hermes-8b", "params_b": 8.0, "not_recommended": False, "warning": None}]
    monkeypatch.setattr(adv_api, "_analysis_model_options", only_small)

    client.put("/api/v1/advisor/settings", json={"analysis_model": "my-explicit-pick"})
    m = client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true").json()["synthesis"]["model"]
    assert m == "my-explicit-pick"   # respected even though absent (fails honestly downstream)


def test_the_profile_endpoint_reports_the_live_box(client):
    p = client.get("/api/v1/system/profile").json()
    assert p["detected"] is True
    assert p["gpu_pool_gb"] == 120.0
    assert p["gpu_family"] == "gfx1151"
    assert p["npu_family"] == "XDNA2"


def test_the_locked_kernel_holds_the_safety_floor():
    """The kernel is minimal and un-editable, and it carries the two absolute
    rules. Guard the wording so no refactor of the editable instructions can
    quietly drop the floor a novice depends on."""
    from kblocallm.advisor_service import OPERATIONAL_PROMPT
    low = OPERATIONAL_PROMPT.lower()
    assert "absolute" in low and "cannot be overridden" in low
    assert "model-specific fact" in low            # rule 1
    assert "unknown" in low
    assert "fabricate a source" in low             # rule 2
    assert "concepts" in low                        # concepts are still teachable


def test_synthesis_context_carries_owned_inventory_for_alternatives(client, monkeypatch):
    """§1.5: the advisor recommends from real inventory. Capture the prompt and
    confirm the operator's OTHER models are in it, so 'better matches' cannot be
    a model the operator does not own."""
    captured = {}
    orig = client.stub.request

    async def spy(method, path, **kw):
        if "chat/completions" in path:
            captured["system"] = kw["json_body"]["messages"][0]["content"]
        return await orig(method, path, **kw)
    monkeypatch.setattr(client.stub, "request", spy)

    client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true")
    system = captured["system"]
    assert "operators_other_owned_models" in system
    # A real other model from the fixture is offered as a candidate.
    assert "Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M" in system
    # The target is excluded from its own alternatives list (it appears in the
    # inventory of no other model's analysis, only as the subject here).
    assert '"id": "' + GEMMA + '"' not in system.split("operators_other_owned_models")[1]
    # Web search seam present, honestly marked not-run until wired.
    assert "not_run" in system


def test_advisor_chat_is_grounded_on_a_model(client):
    r = client.post("/api/v1/advisor/chat",
                    json={"model_id": GEMMA, "message": "is this good for tool use?"})
    assert r.status_code == 200
    assert r.json()["grounded_on"] == GEMMA


# -- streaming chat (the perf fix) ------------------------------------------


def _sse_events(raw: str) -> list[dict]:
    """Parse the SSE body into the JSON events, skipping keepalive comments."""
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        out.append(json.loads(line[len("data:"):].strip()))
    return out


def test_streaming_chat_relays_text_then_completes(client):
    """The fix: tokens stream as they arrive. The reasoning channel comes first
    (motion), then answer text deltas, then a terminal complete."""
    r = client.post("/api/v1/advisor/chat/stream",
                    json={"model_id": GEMMA, "message": "what is this good for?"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert r.headers.get("x-accel-buffering") == "no"   # nginx must not buffer it
    events = _sse_events(r.text)
    kinds = [e["event"] for e in events]
    assert "reasoning" in kinds and "text" in kinds
    assert kinds[-1] == "complete"
    # The answer text is the concatenation of the text deltas, never the reasoning.
    answer = "".join(e["text"] for e in events if e["event"] == "text")
    assert answer == "Hello there."
    assert "let me think" not in answer                 # reasoning is never the answer


def test_streaming_complete_carries_the_per_turn_numbers(client):
    """The complete event feeds the per-turn line: tokens, wall-clock, model,
    and the cached-prompt count when the server reports it."""
    r = client.post("/api/v1/advisor/chat/stream",
                    json={"model_id": GEMMA, "message": "hi"})
    done = _sse_events(r.text)[-1]
    assert done["event"] == "complete"
    assert done["model"] == GEMMA and done["grounded_on"] == GEMMA
    assert done["elapsed_s"] is not None
    assert done["usage"]["prompt_tokens"] == 8030
    assert done["usage"]["prompt_tokens_details"]["cached_tokens"] == 8000


def test_streaming_reasoning_drain_is_named_not_blank(client):
    """If the model spends its whole budget reasoning and emits no answer, the
    complete event says so and points at a non-reasoning analyst — never a blank
    bubble."""
    client.stub.chat_starved = True
    done = _sse_events(client.post("/api/v1/advisor/chat/stream",
                                   json={"model_id": GEMMA, "message": "hi"}).text)[-1]
    assert done["event"] == "complete"
    assert "reasoning" in done["note"] and "non-reasoning" in done["note"]


def test_streaming_upstream_failure_is_delivered_in_the_stream(client):
    """A model failure must arrive as an error EVENT on the open stream, not a
    dropped connection that reads as 'stream ended'."""
    client.stub.chat_fails = True
    events = _sse_events(client.post("/api/v1/advisor/chat/stream",
                                     json={"model_id": GEMMA, "message": "hi"}).text)
    assert events[-1]["event"] == "error"
    assert events[-1]["kind"] and events[-1]["detail"]


def test_streaming_chat_about_an_unknown_model_is_a_404(client):
    r = client.post("/api/v1/advisor/chat/stream", json={"model_id": "nope", "message": "hi"})
    assert r.status_code == 404


# -- web search (ported from KAIROS) ---------------------------------------


def test_search_is_off_and_honest_without_a_key(client, monkeypatch):
    """No key = search not_run, and the prompt tells the model to say a live
    lookup was not performed rather than recall."""
    captured = {}
    orig = client.stub.request

    async def spy(method, path, **kw):
        if "chat/completions" in path:
            captured["system"] = kw["json_body"]["messages"][0]["content"]
        return await orig(method, path, **kw)
    monkeypatch.setattr(client.stub, "request", spy)

    client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true")
    assert '"status": "not_run"' in captured["system"]


def test_search_result_is_grounded_into_the_prompt_when_a_key_is_set(client, monkeypatch):
    """With a key, a grounded search runs and its answer + sources land in the
    context as an external source the model can cite."""
    from kblocallm import config as cfg
    cfg.get_settings.cache_clear()
    monkeypatch.setenv("HELM_WEB_SEARCH_KEY", "test-key")
    cfg.get_settings.cache_clear()

    from kblocallm.api import advisor as adv_api
    from kblocallm.web_search import SearchResult

    async def fake_search(query, **kw):
        return SearchResult(ok=True, answer="Qwen3 is a strong open MoE family.",
                            sources=[{"title": "HF", "url": "https://hf.co/x"}])
    monkeypatch.setattr(adv_api, "web_search", fake_search)

    captured = {}
    orig = client.stub.request

    async def spy(method, path, **kw):
        if "chat/completions" in path:
            captured["system"] = kw["json_body"]["messages"][0]["content"]
        return await orig(method, path, **kw)
    monkeypatch.setattr(client.stub, "request", spy)

    client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true")
    assert "Qwen3 is a strong open MoE family" in captured["system"]
    assert "hf.co/x" in captured["system"]
    cfg.get_settings.cache_clear()


def test_the_search_model_is_editable_not_hardcoded(client):
    """Kevin's ask: swap the grounding model from the UI, no redeploy. Defaults
    to gemini-2.5-flash but is DB-stored and editable."""
    s = client.get("/api/v1/advisor/settings").json()
    assert s["settings"]["web_search_model"] == "gemini-2.5-flash"   # shipped default
    assert s["settings"]["web_search_model_is_default"] is True

    client.put("/api/v1/advisor/settings", json={"web_search_model": "gemini-3.0-flash"})
    s2 = client.get("/api/v1/advisor/settings").json()
    assert s2["settings"]["web_search_model"] == "gemini-3.0-flash"
    assert s2["settings"]["web_search_model_is_default"] is False
    assert s2["web_search"]["model"] == "gemini-3.0-flash"           # status reflects it


def test_the_configured_search_model_is_the_one_actually_called(client, monkeypatch):
    from kblocallm import config as cfg
    monkeypatch.setenv("HELM_WEB_SEARCH_KEY", "k")
    cfg.get_settings.cache_clear()
    client.put("/api/v1/advisor/settings",
               json={"web_search_model": "some-new-model", "web_search_provider": "gemini"})

    from kblocallm.api import advisor as adv_api
    from kblocallm.web_search import SearchResult
    seen = {}

    async def fake_search(query, *, api_key, provider, model, **kw):
        seen["model"] = model
        return SearchResult(ok=True, answer="x", sources=[])
    monkeypatch.setattr(adv_api, "web_search", fake_search)

    client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true")
    assert seen["model"] == "some-new-model"   # the DB value, not a hardcoded default
    cfg.get_settings.cache_clear()


def test_settings_reports_search_status_but_never_the_key(client, monkeypatch):
    from kblocallm import config as cfg
    monkeypatch.setenv("HELM_WEB_SEARCH_KEY", "super-secret-value")
    cfg.get_settings.cache_clear()
    body = client.get("/api/v1/advisor/settings").json()
    assert body["web_search"]["enabled"] is True
    assert body["web_search"]["provider"] == "gemini"
    assert "super-secret-value" not in json.dumps(body)  # key never leaves the server
    cfg.get_settings.cache_clear()


# -- knowledge primer ------------------------------------------------------


def test_knowledge_primer_is_injected_as_authoritative_reference(client, monkeypatch):
    """When Kevin sets the primer, it goes into the prompt labeled authoritative,
    so concept teaching grounds in the vetted doc, not the model's recall."""
    client.put("/api/v1/advisor/settings",
               json={"knowledge_primer": "QUANTIZATION means storing weights in fewer bits."})
    captured = {}
    orig = client.stub.request

    async def spy(method, path, **kw):
        if "chat/completions" in path:
            captured["system"] = kw["json_body"]["messages"][0]["content"]
        return await orig(method, path, **kw)
    monkeypatch.setattr(client.stub, "request", spy)

    client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true")
    assert "REFERENCE PRIMER" in captured["system"]
    assert "storing weights in fewer bits" in captured["system"]


def test_primer_persists_and_reports_custom(client):
    client.put("/api/v1/advisor/settings", json={"knowledge_primer": "some primer text"})
    s = client.get("/api/v1/advisor/settings").json()["settings"]
    assert s["knowledge_primer_active"] is True
    assert s["knowledge_primer_is_default"] is False    # it's a custom override
    assert s["knowledge_primer"] == "some primer text"


# -- shipped defaults + operator management (Kevin's requirement) -----------


def test_the_architect_files_ship_as_editable_defaults(client):
    """Fresh install: the advisor's instructions and knowledge file are the
    architect's canonical docs, marked as defaults, and active out of the box."""
    s = client.get("/api/v1/advisor/settings").json()["settings"]
    assert s["persona_is_default"] is True
    assert "HELM Model Advisor" in s["persona_prompt"]          # from instructions.txt
    assert s["knowledge_primer_is_default"] is True
    assert s["knowledge_primer_active"] is True
    assert "MODEL ADVISOR KNOWLEDGE FILE" in s["knowledge_primer"]  # from 14d


def test_operator_override_wins_and_is_marked_custom(client):
    client.put("/api/v1/advisor/settings", json={"persona_prompt": "My own instructions."})
    s = client.get("/api/v1/advisor/settings").json()["settings"]
    assert s["persona_prompt"] == "My own instructions."
    assert s["persona_is_default"] is False


def test_restoring_the_shipped_default(client):
    """Clearing an override brings the architect's default back — the update
    path for when the architect ships a new version."""
    client.put("/api/v1/advisor/settings", json={"knowledge_primer": "custom"})
    assert client.get("/api/v1/advisor/settings").json()["settings"]["knowledge_primer_is_default"] is False
    client.put("/api/v1/advisor/settings", json={"knowledge_primer": "__default__"})
    s = client.get("/api/v1/advisor/settings").json()["settings"]
    assert s["knowledge_primer_is_default"] is True
    assert "MODEL ADVISOR KNOWLEDGE FILE" in s["knowledge_primer"]


def test_upload_a_knowledge_file(client):
    """Kevin attaches a .md rather than pasting 20KB."""
    r = client.post("/api/v1/advisor/knowledge-file",
                    files={"file": ("primer.md", b"# My primer\nQuant means fewer bits.", "text/markdown")})
    assert r.status_code == 200 and r.json()["attached"] is True
    assert r.json()["knowledge_primer_is_default"] is False
    s = client.get("/api/v1/advisor/settings").json()["settings"]
    assert "My primer" in s["knowledge_primer"]


def test_upload_rejects_a_giant_file(client):
    huge = b"x" * 300_000
    r = client.post("/api/v1/advisor/knowledge-file",
                    files={"file": ("big.md", huge, "text/markdown")})
    assert r.status_code == 422


def test_the_default_primer_is_injected_by_default(client, monkeypatch):
    """Out of the box, the shipped 14d primer is active, so the advisor teaches
    concepts from it without any operator setup."""
    captured = {}
    orig = client.stub.request

    async def spy(method, path, **kw):
        if "chat/completions" in path:
            captured["system"] = kw["json_body"]["messages"][0]["content"]
        return await orig(method, path, **kw)
    monkeypatch.setattr(client.stub, "request", spy)

    client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true")
    assert "REFERENCE PRIMER" in captured["system"]
    assert "QUANTIZATION" in captured["system"].upper()  # concept content from 14d


def test_an_empty_primer_produces_no_reference_section(client, monkeypatch):
    """If the default were absent and no override set, there is no empty
    REFERENCE PRIMER block — the section is conditional, not a stub."""
    from kblocallm import advisor_service
    monkeypatch.setattr(advisor_service, "DEFAULT_KNOWLEDGE_PRIMER", "")
    captured = {}
    orig = client.stub.request

    async def spy(method, path, **kw):
        if "chat/completions" in path:
            captured["system"] = kw["json_body"]["messages"][0]["content"]
        return await orig(method, path, **kw)
    monkeypatch.setattr(client.stub, "request", spy)

    client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true")
    assert "REFERENCE PRIMER" not in captured["system"]


def test_chat_about_an_unknown_model_is_a_404(client):
    r = client.post("/api/v1/advisor/chat", json={"model_id": "nope", "message": "hi"})
    assert r.status_code == 404


# -- saved summaries (Kevin's save/regenerate) ------------------------------


def test_no_saved_summary_by_default(client):
    r = client.post(f"/api/v1/advisor/analyze/{GEMMA}")
    assert r.json()["saved_summary"] is None


def test_save_then_analyze_returns_it_instantly(client):
    """The whole point: save the slow write-up, get it back instantly next
    time without regenerating."""
    client.post(f"/api/v1/advisor/summary/{GEMMA}",
                json={"text": "A saved analysis of the 26B.", "analysis_model": GEMMA})
    r = client.post(f"/api/v1/advisor/analyze/{GEMMA}")  # no synthesize — no model run
    saved = r.json()["saved_summary"]
    assert saved["text"] == "A saved analysis of the 26B."
    assert saved["analysis_model"] == GEMMA
    assert saved["saved"] is True and saved["generated_at"]


def test_saving_again_overwrites_regenerate(client):
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "first"})
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "second (regenerated)"})
    saved = client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()["saved_summary"]
    assert saved["text"] == "second (regenerated)"


def test_delete_a_saved_summary(client):
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "x"})
    client.delete(f"/api/v1/advisor/summary/{GEMMA}")
    assert client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()["saved_summary"] is None


def test_saving_empty_is_rejected(client):
    assert client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "   "}).status_code == 422


def test_a_fresh_generation_is_separate_from_the_saved_one(client):
    """Generate returns a FRESH synthesis and does NOT auto-save; the saved one
    (if any) comes back alongside it. The operator saves explicitly."""
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "the saved one"})
    r = client.post(f"/api/v1/advisor/analyze/{GEMMA}?synthesize=true").json()
    assert r["synthesis"]["available"] is True                # fresh
    assert r["saved_summary"]["text"] == "the saved one"      # still the old saved
    # Generation did not overwrite the save.
    assert client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()["saved_summary"]["text"] == "the saved one"


# -- Deep analysis has its OWN saved slot (QoL: save the deep write-up too) --


def test_summary_and_deep_are_saved_in_separate_slots(client):
    """The whole point: saving a Deep analysis must not clobber a saved Summary,
    and vice versa. Two kinds, two slots, one model."""
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "the fast summary", "kind": "summary"})
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "the deep teach", "kind": "deep"})
    r = client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()
    assert r["saved_summary"]["text"] == "the fast summary"
    assert r["saved_deep"]["text"] == "the deep teach"
    assert r["saved_deep"]["kind"] == "deep"


def test_saving_a_deep_analysis_does_not_touch_the_saved_summary(client):
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "summary stays", "kind": "summary"})
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "deep v1", "kind": "deep"})
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "deep v2", "kind": "deep"})  # regenerate deep
    r = client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()
    assert r["saved_summary"]["text"] == "summary stays"      # untouched by the deep saves
    assert r["saved_deep"]["text"] == "deep v2"               # deep overwrote its own slot only


def test_deleting_one_kind_leaves_the_other(client):
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "s", "kind": "summary"})
    client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "d", "kind": "deep"})
    client.delete(f"/api/v1/advisor/summary/{GEMMA}?kind=deep")
    r = client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()
    assert r["saved_summary"]["text"] == "s"                  # still there
    assert r["saved_deep"] is None                            # only the deep was removed


def test_an_unknown_saved_kind_is_rejected(client):
    r = client.post(f"/api/v1/advisor/summary/{GEMMA}", json={"text": "x", "kind": "bogus"})
    assert r.status_code == 422


def test_no_saved_deep_by_default(client):
    assert client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()["saved_deep"] is None


# -- Summary (§1.10): its own model, its own starved context ----------------


def test_summarize_uses_its_own_model_not_the_analyst(client):
    """Summary runs the SUMMARY model (Aevum by default), not the 26B analyst.
    Different feature, different model — that separation is the whole point."""
    r = client.post(f"/api/v1/advisor/summarize/{GEMMA}").json()
    assert r["generated"] is True and r["available"] is True
    assert r["model"] == AEVUM                       # Aevum, not the 26B analyst
    assert client.stub.last_chat_body["model"] == AEVUM
    assert "from the facts card" in r["label"]


def test_summarize_starves_the_context_no_inventory_search_or_primer(client):
    """The speed AND the grounding come from starvation: the summary prompt is
    the facts card only. None of the synthesis's fat context leaks in."""
    client.post(f"/api/v1/advisor/summarize/{GEMMA}")
    system = client.stub.last_chat_body["messages"][0]["content"]
    assert "FACTS CARD" in system
    # None of the deep-synthesis sources appear.
    assert "operators_other_owned_models" not in system
    assert "web_search_results" not in system
    assert "REFERENCE PRIMER" not in system
    assert "this_machine_detected_hardware" not in system


def test_summarize_caps_tokens_low(client):
    """A low cap so a runaway generation can't stretch the 'few seconds' run."""
    client.post(f"/api/v1/advisor/summarize/{GEMMA}")
    assert client.stub.last_chat_body["max_tokens"] <= 800


def test_summarize_has_its_own_locked_floor(client):
    """A summary-specific safety kernel, un-editable, rides in front of the
    instructions: facts card only, did not look anything up, not an analysis."""
    client.post(f"/api/v1/advisor/summarize/{GEMMA}")
    system = client.stub.last_chat_body["messages"][0]["content"]
    assert "ABSOLUTE" in system                       # the locked floor is present
    assert "did NOT look anything up" in system.replace("did not", "did NOT")


def test_summarize_failure_is_honest_not_faked(client):
    client.stub.chat_fails = True
    r = client.post(f"/api/v1/advisor/summarize/{GEMMA}").json()
    assert r["available"] is False and "unavailable" in r["note"]


def test_summarize_reasoning_starvation_is_named(client):
    """If the operator points Summary at a reasoning model that eats its budget,
    say so and point at Aevum — never a bare blank."""
    client.stub.chat_starved = True
    r = client.post(f"/api/v1/advisor/summarize/{GEMMA}").json()
    assert r["available"] is False and r["finish_reason"] == "length"
    assert "non-reasoning" in r["note"] and "Aevum" in r["note"]


def test_a_generated_summary_can_be_saved_and_returns_instantly(client):
    """The fast Summary is what the saved-snapshot slot now holds: generate,
    save, get it back instantly next open — with the summary model recorded."""
    gen = client.post(f"/api/v1/advisor/summarize/{GEMMA}").json()
    client.post(f"/api/v1/advisor/summary/{GEMMA}",
                json={"text": gen["text"], "analysis_model": gen["model"]})
    saved = client.post(f"/api/v1/advisor/analyze/{GEMMA}").json()["saved_summary"]
    assert saved["text"] == gen["text"]
    assert saved["analysis_model"] == AEVUM


# -- Summary settings: own model + own instruction box ----------------------


def test_summary_defaults_to_aevum_and_the_appendix_s_text(client):
    s = client.get("/api/v1/advisor/settings").json()
    assert s["settings"]["summary_model"] == AEVUM
    assert s["settings"]["summary_model_is_default"] is True
    assert s["settings"]["summary_prompt_is_default"] is True
    assert "HELM model Summary" in s["settings"]["summary_prompt"]


def test_the_summary_picker_does_not_flag_small_models_as_fabricating(client):
    """The analyst picker warns 'sub-2B fabricates'; the SUMMARY picker must not
    — a small model is the RIGHT tool here, and Aevum must read as fine."""
    s = client.get("/api/v1/advisor/settings").json()
    picker = {m["id"]: m for m in s["summary_models"]}
    assert AEVUM in picker
    assert picker[AEVUM]["not_recommended"] is False
    assert picker[AEVUM]["warning"] is None
    # And it is offered small-first.
    params = [m["params_b"] for m in s["summary_models"] if m["params_b"] is not None]
    assert params == sorted(params)


def test_summary_model_and_instructions_are_editable_and_restorable(client):
    client.put("/api/v1/advisor/settings", json={"summary_model": GEMMA})
    client.put("/api/v1/advisor/settings", json={"summary_prompt": "Be terse."})
    s = client.get("/api/v1/advisor/settings").json()["settings"]
    assert s["summary_model"] == GEMMA and s["summary_model_is_default"] is False
    assert s["summary_prompt"] == "Be terse." and s["summary_prompt_is_default"] is False
    # And the summarize path now honours the operator's pick.
    client.post(f"/api/v1/advisor/summarize/{GEMMA}")
    assert client.stub.last_chat_body["model"] == GEMMA
    # Restore both.
    client.put("/api/v1/advisor/settings", json={"summary_prompt": "__default__"})
    s2 = client.get("/api/v1/advisor/settings").json()["settings"]
    assert s2["summary_prompt_is_default"] is True


def test_editing_the_summary_box_does_not_touch_the_advisor_instructions(client):
    """The two boxes are independent: a summary edit must not bleed into the
    Analyze instructions, or the split is cosmetic."""
    before = client.get("/api/v1/advisor/settings").json()["settings"]["persona_prompt"]
    client.put("/api/v1/advisor/settings", json={"summary_prompt": "Summary only change."})
    after = client.get("/api/v1/advisor/settings").json()["settings"]
    assert after["persona_prompt"] == before                 # untouched
    assert after["summary_prompt"] == "Summary only change."


def test_the_default_summary_model_falls_back_to_the_smallest_when_aevum_absent(client, monkeypatch):
    """Portability, pulling the other way from Analyze: on a box without Aevum
    the DEFAULT summary model resolves to the SMALLEST owned model (Summary
    wants fast + cheap), not the largest."""
    from kblocallm.api import advisor as adv_api

    async def no_aevum(client_):
        return [
            {"id": "tiny-1b", "params_b": 1.0, "not_recommended": True, "warning": "x"},
            {"id": "hermes-8b", "params_b": 8.0, "not_recommended": False, "warning": None},
        ]
    monkeypatch.setattr(adv_api, "_analysis_model_options", no_aevum)

    client.post(f"/api/v1/advisor/summarize/{GEMMA}")
    assert client.stub.last_chat_body["model"] == "tiny-1b"   # smallest, not the 8B


# -- model card (the download/update "what am I getting" view) ---------------


def test_model_card_is_fetched_for_a_model_with_an_hf_repo(client, monkeypatch):
    """GEMMA's checkpoint resolves to unsloth/... so the card endpoint fetches it.
    The fetch is stubbed (no network); the point is the repo resolves and the
    card comes back."""
    from kblocallm import advisor_service

    async def fake_card(self, repo_id, *, force=False):
        return f"# Card for {repo_id}\nHow to run this model."
    monkeypatch.setattr(advisor_service.HFMetadataService, "fetch_card", fake_card)

    r = client.get(f"/api/v1/library/card/{GEMMA}").json()
    assert r["repo_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert "How to run this model" in r["card"]


def test_a_local_model_with_no_repo_has_no_card_not_an_invented_one(client):
    """Aevum is a local-path finetune with nothing remembered — honest absence."""
    r = client.get(f"/api/v1/library/card/{AEVUM}").json()
    assert r["repo_id"] is None and r["card"] is None
    assert "no HuggingFace repo" in r["note"].lower() or "custom" in r["note"].lower()


def test_a_remembered_checkpoint_restores_the_card_lookup(client, monkeypatch):
    """Same fix as external metadata: if HELM remembered the pull-time checkpoint,
    the card resolves from it even when Lemonade only has a local path."""
    from kblocallm import advisor_service
    from kblocallm.advisor_service import ModelSourceStore

    async def fake_card(self, repo_id, *, force=False):
        return f"# {repo_id} card"
    monkeypatch.setattr(advisor_service.HFMetadataService, "fetch_card", fake_card)
    ModelSourceStore(client.app.state.db).set(AEVUM, "someorg/aevum-gguf:Q4_K_M")

    r = client.get(f"/api/v1/library/card/{AEVUM}").json()
    assert r["repo_id"] == "someorg/aevum-gguf"
    assert "card" in r["card"].lower()


def test_the_card_frontmatter_is_stripped():
    from kblocallm.advisor_service import _strip_frontmatter
    body = _strip_frontmatter("---\nlicense: mit\ntags: [a, b]\n---\n\n# Title\nThe prose.")
    assert body.startswith("# Title") and "license: mit" not in body
