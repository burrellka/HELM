"""Persistence for golden config pins. Pairs with domain/golden.py.

Same split as the crash detector: the comparison in `domain.golden` is pure and
tested; this file is only reads and writes. One row per model, the operator's
blessed baseline, independent of what Lemonade currently has saved.
"""
from __future__ import annotations

from ..domain.golden import GoldenConfig
from .db import SQLiteManager


class GoldenConfigStore:
    def __init__(self, db: SQLiteManager) -> None:
        self._db = db

    def set(self, golden: GoldenConfig) -> None:
        with self._db.serialized() as conn:
            conn.execute(
                """
                INSERT INTO golden_configs
                    (model_id, ctx_size, llamacpp_backend, backend_build, llamacpp_args, note, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(model_id) DO UPDATE SET
                    ctx_size=excluded.ctx_size,
                    llamacpp_backend=excluded.llamacpp_backend,
                    backend_build=excluded.backend_build,
                    llamacpp_args=excluded.llamacpp_args,
                    note=excluded.note,
                    captured_at=datetime('now')
                """,
                (golden.model_id, golden.ctx_size, golden.llamacpp_backend,
                 golden.backend_build, golden.llamacpp_args, golden.note),
            )

    def get(self, model_id: str) -> GoldenConfig | None:
        with self._db.serialized() as conn:
            row = conn.execute(
                "SELECT model_id, ctx_size, llamacpp_backend, backend_build, "
                "llamacpp_args, note, captured_at FROM golden_configs WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        return _row(row) if row else None

    def all(self) -> dict[str, GoldenConfig]:
        with self._db.serialized() as conn:
            rows = conn.execute(
                "SELECT model_id, ctx_size, llamacpp_backend, backend_build, "
                "llamacpp_args, note, captured_at FROM golden_configs"
            ).fetchall()
        return {r[0]: _row(r) for r in rows}

    def delete(self, model_id: str) -> None:
        with self._db.serialized() as conn:
            conn.execute("DELETE FROM golden_configs WHERE model_id = ?", (model_id,))


def _row(row: tuple) -> GoldenConfig:
    return GoldenConfig(
        model_id=row[0], ctx_size=row[1], llamacpp_backend=row[2],
        backend_build=row[3], llamacpp_args=row[4] or "", note=row[5] or "",
        captured_at=row[6],
    )
