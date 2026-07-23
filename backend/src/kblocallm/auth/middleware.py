"""The auth dependency. Ported from URSA-OSCAR.

Reads a JWT from either:
  - the `helm_session` cookie (browser)
  - `Authorization: Bearer <token>` (scripts/agents)

Cookie wins when both are present.

Mount it at the ROUTER level, never per-endpoint:

    _AUTH_REQUIRED = [Depends(require_auth)]
    app.include_router(control.router, dependencies=_AUTH_REQUIRED)

That's deliberate and it's the single most important line in this file. On a
control plane where a forgotten decorator means an open `POST /unload`, auth
must hold by construction, not by the next developer remembering. See
HELM_SPEC_REVIEW.md F1a.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status

from .tokens import TokenError, decode_token

logger = logging.getLogger(__name__)

COOKIE_NAME = "helm_session"


def require_auth(request: Request) -> dict:
    """FastAPI dependency. Returns the JWT claims on success, raises 401
    otherwise."""
    # The documented escape hatch (config.disable_auth). Logged on every
    # single request, at WARNING, on purpose: an unauthenticated control
    # plane should be impossible to leave switched on by accident.
    if getattr(request.app.state, "auth_disabled", False):
        logger.warning(
            "AUTH DISABLED — %s %s served with no authentication. HELM can "
            "unload models and delete weights. This is only safe if the port "
            "is bound to 127.0.0.1 and the edge enforces mTLS.",
            request.method, request.url.path,
        )
        return {"sub": "operator", "kind": "session", "auth": "disabled"}

    secret = getattr(request.app.state, "jwt_secret", None)
    if not secret:
        # Misconfiguration. Fail closed and loudly — never fall through to
        # "well, no secret, so let it past".
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Auth not initialized: app.state.jwt_secret is missing. "
                "This is a server configuration error, not a login problem."
            ),
        )

    token = request.cookies.get(COOKIE_NAME)
    if not token:
        header = request.headers.get("authorization") or ""
        if header.lower().startswith("bearer "):
            token = header[7:].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return decode_token(secret, token)
    except TokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
