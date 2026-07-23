"""Liveness + version. The only routes besides /auth that skip auth.

Kept dependency-free on purpose: the Docker healthcheck must be able to tell
"the process is up" from "the process is up but Lemonade is unreachable".
Conflating those makes the container restart-loop for an upstream problem it
cannot fix by restarting.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..version import get_version

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict:
    """Is this process alive? Nothing more. No DB, no upstream.

    Do NOT add an upstream check here. See the module docstring.
    """
    return {"ok": True, "version": get_version()}


@router.get("/version")
def version() -> dict:
    return {"version": get_version()}
