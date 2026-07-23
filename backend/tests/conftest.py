"""Test fixtures.

Note the shape: every test gets an isolated tmp data dir and the Settings
cache is cleared. Nothing here reaches the network, and NOTHING in this test
suite may ever touch the live box  — see README.md.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kblocallm.config import get_settings


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Isolated data dir + a cleared Settings cache."""
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("HELM_DATA_DIR", str(d))
    # Point the upstream at a black hole. If a test ever tries to reach the
    # real Lemonade, it fails fast instead of silently succeeding against
    # production.
    monkeypatch.setenv("HELM_LEMONADE_URL", "http://127.0.0.1:1")
    get_settings.cache_clear()
    yield d
    get_settings.cache_clear()


@pytest.fixture
def client(data_dir):
    from kblocallm.main import create_app
    with TestClient(create_app()) as c:
        yield c
