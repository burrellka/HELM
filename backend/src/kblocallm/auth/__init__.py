"""Operator auth for HELM.

Ported from URSA-OSCAR's Phase 6.4 auth. See HELM_SPEC_REVIEW.md F1 for why
HELM has this at all when the spec didn't call for it: HELM can unload the
model six agents depend on and delete 20GB of weights, and Cloudflare mTLS
only guards the public path — anything on the Docker network or your LAN reaches the
container port directly.
"""
from .middleware import COOKIE_NAME, require_auth  # noqa: F401
from .store import AuthStore  # noqa: F401
from .tokens import (  # noqa: F401
    TokenError,
    decode_token,
    encode_token,
    resolve_jwt_secret,
)
