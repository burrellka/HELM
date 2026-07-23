"""Auth frame tests.

These aren't ceremony. `test_protected_route_requires_auth` and
`test_every_non_open_router_is_guarded` are the ones that matter: they're the
regression guard for HELM_SPEC_REVIEW.md F1 — the reason HELM has auth at all
is that it can unload the model six agents depend on and delete 20GB of
weights, and Cloudflare mTLS only guards the public path.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from kblocallm.auth import require_auth
from kblocallm.auth.hashing import hash_password, verify_password
from kblocallm.auth.tokens import (
    TokenError,
    decode_token,
    encode_token,
    resolve_jwt_secret,
)


# -- hashing ---------------------------------------------------------------


def test_password_roundtrip():
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong horse battery staple", h)


def test_verify_rejects_malformed_hash_instead_of_raising():
    """A corrupt auth.json must read as 'wrong password', not a 500."""
    assert verify_password("anything", "not-a-real-argon2-hash") is False


def test_hash_is_salted():
    """Same password, different hashes. Guards against anyone 'optimising'
    to an unsalted digest."""
    assert hash_password("same") != hash_password("same")


# -- tokens ----------------------------------------------------------------


def test_token_roundtrip():
    claims = decode_token("s3cret", encode_token("s3cret", kind="session"))
    assert claims["sub"] == "operator"
    assert claims["kind"] == "session"


def test_token_signed_with_another_secret_is_rejected():
    token = encode_token("secret-a")
    with pytest.raises(TokenError):
        decode_token("secret-b", token)


def test_garbage_token_is_rejected():
    with pytest.raises(TokenError):
        decode_token("s3cret", "not.a.jwt")


def test_empty_secret_is_rejected_not_ignored():
    """Fail closed. A missing secret must never mean 'skip verification'."""
    with pytest.raises(TokenError):
        encode_token("")
    with pytest.raises(TokenError):
        decode_token("", "whatever")


# -- secret resolution -----------------------------------------------------


def test_secret_is_generated_once_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("HELM_JWT_SECRET", raising=False)
    first = resolve_jwt_secret(tmp_path)
    second = resolve_jwt_secret(tmp_path)
    assert first == second, (
        "The secret must persist across restarts or every session cookie is "
        "invalidated on every container restart."
    )
    assert (tmp_path / "jwt_secret").exists()
    assert len(first) == 64  # 32 bytes hex


def test_env_secret_wins_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HELM_JWT_SECRET", "operator-managed-secret")
    assert resolve_jwt_secret(tmp_path) == "operator-managed-secret"


# -- the dependency --------------------------------------------------------


def _app_with_protected_route() -> FastAPI:
    app = FastAPI()

    @app.get("/open")
    def open_route():
        return {"ok": True}

    @app.get("/guarded", dependencies=[Depends(require_auth)])
    def guarded_route():
        return {"ok": True}

    app.state.jwt_secret = "test-secret"
    app.state.auth_disabled = False
    return app


def test_protected_route_requires_auth():
    with TestClient(_app_with_protected_route()) as c:
        assert c.get("/open").status_code == 200
        assert c.get("/guarded").status_code == 401


def test_protected_route_accepts_bearer():
    app = _app_with_protected_route()
    token = encode_token("test-secret", kind="api")
    with TestClient(app) as c:
        r = c.get("/guarded", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200


def test_protected_route_accepts_cookie():
    from kblocallm.auth.middleware import COOKIE_NAME
    app = _app_with_protected_route()
    token = encode_token("test-secret", kind="session")
    with TestClient(app) as c:
        c.cookies.set(COOKIE_NAME, token)
        assert c.get("/guarded").status_code == 200


def test_missing_jwt_secret_fails_closed_with_500_not_open():
    """A misconfigured app must refuse, never fall through to allowing the
    request. This is the difference between a bug and a breach."""
    app = _app_with_protected_route()
    app.state.jwt_secret = None
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/guarded").status_code == 500


# -- the app's real wiring -------------------------------------------------


def test_setup_then_login_flow(client):
    assert client.get("/api/v1/auth/status").json()["initialized"] is False
    # Guarded before setup.
    assert client.get("/api/v1/system/config").status_code == 401

    r = client.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
    assert r.status_code == 200
    # The setup call issued a session cookie, so we're in.
    assert client.get("/api/v1/system/config").status_code == 200

    status = client.get("/api/v1/auth/status").json()
    assert status["initialized"] is True and status["authenticated"] is True


def test_setup_refuses_once_initialized(client):
    client.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
    # Otherwise anyone reaching /setup could reset the operator's password
    # and the whole auth layer would be decorative.
    r = client.post("/api/v1/auth/setup", json={"password": "attacker-chosen"})
    assert r.status_code == 409


def test_short_password_rejected(client):
    assert client.post("/api/v1/auth/setup", json={"password": "short"}).status_code == 422


def test_wrong_password_rejected(client):
    client.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
    client.post("/api/v1/auth/logout")
    assert client.post("/api/v1/auth/login", json={"password": "nope"}).status_code == 401


def test_healthz_is_open_and_needs_no_upstream(client):
    """The container healthcheck must not depend on Lemonade being up —
    otherwise HELM restart-loops over an upstream problem restarting can't
    fix. (conftest points the upstream at a dead port.)"""
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_cookie_is_secure_when_the_proxy_says_https(client):
    """The reason _detect_scheme exists. NPM terminates TLS and speaks plain
    HTTP to the container, so the cookie's Secure flag must key off what the
    BROWSER is doing, not off the internal hop. URSA needed two hotfixes
    (0.13.2, 0.13.3) to land this."""
    r = client.post(
        "/api/v1/auth/setup",
        json={"password": "a-long-enough-pw"},
        headers={"X-Forwarded-Proto": "https"},
    )
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "secure" in cookie
    assert "samesite=strict" in cookie
    assert "httponly" in cookie


def test_cookie_is_not_secure_on_plain_http(client):
    """Otherwise the cookie is dropped on http and login silently no-ops —
    which is exactly how this failed the first time it was written."""
    r = client.post("/api/v1/auth/setup", json={"password": "a-long-enough-pw"})
    cookie = r.headers["set-cookie"].lower()
    assert "secure" not in cookie
    assert "httponly" in cookie  # still never readable from JS


@pytest.mark.parametrize(
    "headers,expected_https,expected_source",
    [
        ({"X-Forwarded-Proto": "https"}, True, "x-forwarded-proto"),
        ({"Origin": "https://helm.example.com"}, True, "origin"),
        ({"Referer": "https://helm.example.com/login"}, True, "referer"),
        ({"X-Forwarded-Proto": "http"}, False, "none"),
        ({}, False, "none"),
    ],
)
def test_scheme_detection_precedence(client, headers, expected_https, expected_source):
    """Origin/Referer are the load-bearing fallbacks: they keep the Secure
    flag correct when the proxy forgets X-Forwarded-Proto."""
    r = client.get("/api/v1/auth/status", headers=headers)
    assert r.json()["scheme_source"] == expected_source


#: Routes that are reachable without a session, and the reason each one has to
#: be. Adding to this list is a security decision — write down why.
OPEN_PREFIXES = (
    "/healthz",        # container healthcheck; must not depend on auth or upstream
    "/version",        # version self-introspection
    "/api/v1/auth/",   # the login endpoints; they check credentials themselves
    "/openapi", "/docs", "/redoc",  # FastAPI's own docs
)


def test_every_route_except_health_and_auth_rejects_an_unauthenticated_caller():
    """Drift guard for F1a. If someone adds a router without _AUTH_REQUIRED,
    this fails. On a control plane, an unguarded POST /unload is the whole
    ballgame — don't 'fix' this test by adding an exemption without a very
    good reason written down.

    This asserts BEHAVIOUR (call it, expect 401), not structure. The original
    version read `route.dependencies` off `app.routes`, and FastAPI 0.139
    stopped flattening included routers into that list — every real route
    became invisible and the check silently had nothing left to inspect. It
    failed closed, which is the only reason it was caught. A test that guards
    the most important property in the app must not rest on a private
    attribute that upstream is free to move; the OpenAPI schema is public API
    and a 401 is the property we actually care about.
    """
    from kblocallm.main import create_app

    app = create_app()
    checked: list[str] = []
    unguarded: list[str] = []

    with TestClient(app) as c:
        for path, operations in app.openapi()["paths"].items():
            if path.startswith(OPEN_PREFIXES):
                continue
            for method in operations:
                response = c.request(method.upper(), path)
                checked.append(f"{method.upper()} {path}")
                if response.status_code != 401:
                    unguarded.append(f"{method.upper()} {path} -> {response.status_code}")

    assert not unguarded, (
        f"Routes served without authentication: {unguarded}. Mount their router "
        f"with dependencies=_AUTH_REQUIRED in main.create_app()."
    )
    # A green light from an empty loop is how the previous version failed.
    # Assert it actually looked at something.
    assert checked, "The route sweep found nothing to check — the test is broken, not passing."


def test_the_guard_sweep_can_actually_see_the_dashboard_routes():
    """Companion to the above, and the specific regression that hid the
    breakage: the sweep must be able to SEE the Phase 1 routes. If this list
    ever comes back empty, the test above is passing vacuously again."""
    from kblocallm.main import create_app

    paths = set(create_app().openapi()["paths"])
    assert {"/api/v1/dashboard", "/api/v1/live", "/api/v1/system/config"} <= paths
