"""Self-reported version.

House pattern (URSA 1.1.3): the container reads its own version from its
installed package metadata. Nothing has to keep an env var in sync with an
image tag, which is a chore that is always one deploy away from lying.

Bump `version` in pyproject.toml; everything downstream follows.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version


def get_version() -> str:
    try:
        return _pkg_version("kblocallm")
    except PackageNotFoundError:
        # Running from source without an install (pytest against src/).
        return "0.0.0-dev"
