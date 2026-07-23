"""The data-dir preflight.

This exists because a real deploy died with
`sqlite3.OperationalError: unable to open database file` under forty lines of
lifespan traceback. The cause was mundane (a root-owned bind mount, a non-root
container) and the message said nothing about uids, the mount, or the fix.
"""
from __future__ import annotations

import os

import pytest

from kblocallm.config import Settings
from kblocallm.main import _preflight_data_dir


def test_a_writable_dir_passes_quietly(tmp_path):
    _preflight_data_dir(Settings(data_dir=tmp_path))


def test_a_missing_dir_is_created(tmp_path):
    target = tmp_path / "nested" / "data"
    _preflight_data_dir(Settings(data_dir=target))
    assert target.is_dir()


def test_the_probe_file_does_not_linger(tmp_path):
    _preflight_data_dir(Settings(data_dir=tmp_path))
    assert list(tmp_path.iterdir()) == [], "the write probe must clean up after itself"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions; the container is Linux")
def test_an_unwritable_dir_fails_with_an_actionable_message(tmp_path):
    """The whole point. The message must name the uid, the ownership, and the
    command that fixes it — not just that something went wrong."""
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o555)  # r-xr-xr-x: traversable, not writable
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _preflight_data_dir(Settings(data_dir=data))
        message = str(excinfo.value)
        assert "cannot write to its data directory" in message
        assert "Running as:" in message          # who we are
        assert "owned by uid=" in message        # who owns it
        assert "chown" in message                # what to do
        assert "PUID/PGID" in message            # the other lever
        assert "-wal" in message                 # why the DIRECTORY matters
    finally:
        data.chmod(0o755)  # so tmp_path cleanup can remove it


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions; the container is Linux")
def test_an_uncreatable_dir_names_the_bind_mount(tmp_path):
    parent = tmp_path / "locked"
    parent.mkdir()
    parent.chmod(0o555)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _preflight_data_dir(Settings(data_dir=parent / "data"))
        assert "bind mount" in str(excinfo.value)
    finally:
        parent.chmod(0o755)
