"""JWT issuance/verification + per-instance secret resolution.

Ported from URSA-OSCAR (`ursa_oscar/auth/tokens.py`), which has been running
this in production since its Phase 6.4. Kept deliberately close to the
original so a fix in one is trivially portable to the other — if you improve
this, tell the URSA dev.

HS256 over a per-instance secret. Single-operator system, so the subject is
always "operator"; there is no user table and there should not be one.

Two lifetimes:
  session (24h) — the browser cookie
  api     (90d) — for a script/agent calling HELM without a browser
                  (e.g. a warm-on-startup job, see spec Rule 5)
"""
from __future__ import annotations

import logging
import os
import secrets
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from jose import JWTError, jwt

logger = logging.getLogger(__name__)

JWT_SECRET_ENV = "HELM_JWT_SECRET"

SESSION_LIFETIME = timedelta(hours=24)
API_TOKEN_LIFETIME = timedelta(days=90)

_ALGORITHM = "HS256"
_SUBJECT = "operator"

TokenKind = Literal["session", "api"]


class TokenError(Exception):
    """Token can't be decoded or fails validation."""


def encode_token(
    secret: str,
    *,
    kind: TokenKind = "session",
    now: datetime | None = None,
) -> str:
    if not secret:
        raise TokenError("JWT secret is not configured")
    issued = now or datetime.now(timezone.utc)
    lifetime = SESSION_LIFETIME if kind == "session" else API_TOKEN_LIFETIME
    payload = {
        "sub": _SUBJECT,
        "kind": kind,
        "iat": int(issued.timestamp()),
        "exp": int((issued + lifetime).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def decode_token(secret: str, token: str) -> dict:
    """Verify + decode. Raises TokenError on ANY problem (expired, bad
    signature, malformed, wrong subject). Callers turn that into a 401 —
    they never need to tell the reasons apart at the auth boundary."""
    if not secret:
        raise TokenError("JWT secret is not configured")
    if not token:
        raise TokenError("No token provided")
    try:
        claims = jwt.decode(token, secret, algorithms=[_ALGORITHM])
    except JWTError as e:
        raise TokenError(f"Token rejected: {e}") from e
    if claims.get("sub") != _SUBJECT:
        raise TokenError(f"Unexpected subject: {claims.get('sub')!r}")
    if claims.get("kind") not in ("session", "api"):
        raise TokenError(f"Unknown token kind: {claims.get('kind')!r}")
    return claims


def resolve_jwt_secret(data_dir: Path) -> str:
    """env override > persisted file > first-boot generation.

    The house pattern (URSA uses it for the JWT secret, the Fernet master
    key, and service tokens): the operator should never have to perform a
    key ceremony to stand the app up, but must be able to take ownership of
    the secret if they want rotation/backup.
    """
    raw = os.environ.get(JWT_SECRET_ENV, "").strip()
    if raw:
        return raw

    data_dir.mkdir(parents=True, exist_ok=True)
    secret_path = data_dir / "jwt_secret"

    if secret_path.exists():
        try:
            existing = secret_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        except OSError as e:
            logger.exception(
                "jwt_secret at %s is unreadable (%s) — regenerating. Every "
                "existing session cookie will be invalidated.", secret_path, e,
            )

    secret = secrets.token_hex(32)
    secret_path.write_text(secret, encoding="utf-8")
    try:
        os.chmod(secret_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        # Non-POSIX filesystem (bind mount from Windows, etc.). The data dir
        # is operator-only there anyway.
        pass
    logger.info(
        "Generated per-instance JWT signing secret at %s (mode 0600). "
        "Persisted on the data volume; restarts reuse it. To manage it "
        "yourself, set %s and delete the file.", secret_path, JWT_SECRET_ENV,
    )
    return secret
