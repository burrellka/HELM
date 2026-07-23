"""Argon2id password hashing. Ported from URSA-OSCAR.

Self-describing hash strings ($argon2id$v=19$m=...$...) so verification reads
its parameters from the stored hash — retuning the cost later doesn't
invalidate existing passwords.
"""
from __future__ import annotations

from passlib.hash import argon2

_ARGON2 = argon2.using(type="ID")


def hash_password(plaintext: str) -> str:
    return _ARGON2.hash(plaintext)


def verify_password(plaintext: str, stored_hash: str) -> bool:
    """Constant-time verify. Returns False on a malformed stored hash too —
    the caller's response is 401 either way, and distinguishing "wrong
    password" from "corrupt hash file" at the auth boundary only helps an
    attacker."""
    try:
        return _ARGON2.verify(plaintext, stored_hash)
    except Exception:
        return False
