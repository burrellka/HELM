"""Auth endpoints — /api/v1/auth/*.

These are the ONLY routes mounted without `require_auth` (plus /healthz).
They check their own credentials internally.

Flow:
  first boot        -> GET  /auth/status  {initialized: false} -> UI shows /setup
  POST /auth/setup  -> sets the password, issues a session cookie
  POST /auth/login  -> verifies, issues a session cookie
  POST /auth/logout -> clears the cookie
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from .hashing import hash_password, verify_password
from .middleware import COOKIE_NAME
from .tokens import SESSION_LIFETIME, encode_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Short enough to type, long enough to matter. No complexity theater —
# length is what counts, and a password manager makes length free.
MIN_PASSWORD_LENGTH = 12


class SetupRequest(BaseModel):
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)


class LoginRequest(BaseModel):
    password: str


class AuthStatus(BaseModel):
    initialized: bool
    authenticated: bool
    auth_disabled: bool
    # Which signal told us the browser is on HTTPS. Surfaced so the UI can
    # warn about a misconfigured proxy instead of the operator silently
    # failing to log in. See _detect_scheme.
    scheme_source: str = "none"


def _detect_scheme(request: Request) -> tuple[bool, str]:
    """Is the OPERATOR'S BROWSER on HTTPS? Independent of the internal hop.

    Ported from URSA-OSCAR, where this took two hotfixes (0.13.2, 0.13.3) to
    get right. Read this before you "simplify" it — HELM sits behind NPM,
    which terminates TLS and speaks plain HTTP to the container, so the naive
    answers are both wrong:

      - `secure=True` always  -> the cookie is dropped on any http:// dev
        server and every test client. Login "does nothing", no error.
      - `request.url.scheme`  -> that's the NPM->container hop, which is
        http even when the browser is on https. Cookie is set non-Secure and
        rides in cleartext on the LAN leg.

    Precedence:
      1. url.scheme == https      — direct HTTPS to the API
      2. X-Forwarded-Proto: https — a correctly-configured proxy said so
      3. Origin: https://...      — browsers set this on POST; reliable on
                                    the actual login submit
      4. Referer: https://...     — set on most GETs, incl. the status probe
      5. otherwise                — plain HTTP

    3 and 4 are the load-bearing ones: they mean the Secure flag is still
    correct when the operator's proxy forgets to forward X-Forwarded-Proto,
    which NPM does not always do depending on how the host was created.
    """
    if os.environ.get("HELM_DEV_MODE", "false").lower() == "true":
        # Local dev on http://localhost:5173. Explicit opt-in only.
        return False, "dev_mode"
    if request.url.scheme == "https":
        return True, "url"
    if (request.headers.get("x-forwarded-proto") or "").lower() == "https":
        return True, "x-forwarded-proto"
    if (request.headers.get("origin") or "").lower().startswith("https://"):
        return True, "origin"
    if (request.headers.get("referer") or "").lower().startswith("https://"):
        return True, "referer"
    return False, "none"


def _issue_session(request: Request, response: Response, secret: str) -> None:
    token = encode_token(secret, kind="session")
    is_https, source = _detect_scheme(request)
    if not is_https:
        logger.warning(
            "Issuing a NON-Secure session cookie (scheme detection: %s). "
            "Expected in local dev; in production behind NPM it means the "
            "proxy isn't forwarding X-Forwarded-Proto and the session cookie "
            "will travel the LAN leg in cleartext.", source,
        )
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,                              # JS can never read it
        secure=is_https,                            # see _detect_scheme
        samesite="strict" if is_https else "lax",
        max_age=int(SESSION_LIFETIME.total_seconds()),
        path="/",
    )


@router.get("/status", response_model=AuthStatus)
def auth_status(request: Request) -> AuthStatus:
    """Unauthenticated by design — the login page needs to know whether to
    render setup or login before anyone has a token."""
    store = request.app.state.auth_store
    authed = False
    secret = getattr(request.app.state, "jwt_secret", None)
    token = request.cookies.get(COOKIE_NAME)
    if secret and token:
        from .tokens import TokenError, decode_token
        try:
            decode_token(secret, token)
            authed = True
        except TokenError:
            authed = False
    _is_https, scheme_source = _detect_scheme(request)
    return AuthStatus(
        initialized=store.is_initialized,
        authenticated=authed,
        auth_disabled=bool(getattr(request.app.state, "auth_disabled", False)),
        scheme_source=scheme_source,
    )


@router.post("/setup")
def setup(body: SetupRequest, request: Request, response: Response) -> dict:
    """First-boot password creation. Refuses once initialized — otherwise
    anyone who can reach /setup could reset the operator's password, which
    would make the whole auth layer decorative."""
    store = request.app.state.auth_store
    if store.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Already set up. To reset the password, delete auth.json from "
                "HELM's data directory and restart."
            ),
        )
    store.set_password_hash(hash_password(body.password))
    _issue_session(request, response, request.app.state.jwt_secret)
    logger.info("Operator password set; HELM is now initialized.")
    return {"ok": True}


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict:
    store = request.app.state.auth_store
    stored = store.get_password_hash()
    if not stored:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Not set up yet. Create a password first.",
        )
    if not verify_password(body.password, stored):
        # Deliberately identical to any other failure. Don't leak whether
        # the password was close, or how.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password.",
        )
    _issue_session(request, response, request.app.state.jwt_secret)
    return {"ok": True}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"ok": True}
