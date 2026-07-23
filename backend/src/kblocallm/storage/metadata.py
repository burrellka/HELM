"""Operator metadata: descriptions, favourites, tags, archived. Spec §5.3a.

**Lemonade tracks NONE of this.** It is HELM-local, keyed by Lemonade's model
id, and it has no upstream source of truth — which makes it the only data in
this system that cannot be rebuilt by re-reading the box. Treat it accordingly:
the migrations are append-only and nothing here deletes without being asked.

Keyed by model id because F9 confirmed `health.model_name == models.id`, so
one key serves both the catalog and the loaded state.
"""
from __future__ import annotations

from typing import Any

from .db import SQLiteManager


class MetadataStore:
    """CRUD for §5.3a. Small, boring, and the operator's own."""

    def __init__(self, db: SQLiteManager) -> None:
        self._db = db

    def all(self) -> dict[str, dict[str, Any]]:
        """Everything, keyed by model id, for the library join."""
        rows = self._db.execute(
            "SELECT model_id, description, favorite, archived FROM model_metadata"
        )
        out: dict[str, dict[str, Any]] = {
            r["model_id"]: {
                "description": r["description"],
                "favorite": bool(r["favorite"]),
                "archived": bool(r["archived"]),
                "tags": [],
            }
            for r in rows
        }
        for r in self._db.execute("SELECT model_id, tag FROM model_tags"):
            # A tag row can outlive its metadata row only if someone deleted
            # the parent by hand; tolerate it rather than KeyError the library.
            out.setdefault(r["model_id"], {
                "description": "", "favorite": False, "archived": False, "tags": [],
            })["tags"].append(r["tag"])
        return out

    def get(self, model_id: str) -> dict[str, Any]:
        return self.all().get(model_id, {
            "description": "", "favorite": False, "archived": False, "tags": [],
        })

    def update(
        self,
        model_id: str,
        *,
        description: str | None = None,
        favorite: bool | None = None,
        archived: bool | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Partial update. None means "leave alone", not "clear".

        The distinction matters: a PATCH that omits `description` must not
        wipe the note the operator wrote three months ago.
        """
        with self._db.serialized() as conn:
            conn.execute(
                "INSERT INTO model_metadata (model_id) VALUES (?) "
                "ON CONFLICT(model_id) DO NOTHING",
                (model_id,),
            )
            sets, params = [], []
            if description is not None:
                sets.append("description = ?")
                params.append(description)
            if favorite is not None:
                sets.append("favorite = ?")
                params.append(1 if favorite else 0)
            if archived is not None:
                sets.append("archived = ?")
                params.append(1 if archived else 0)
            if sets:
                sets.append("updated_at = datetime('now')")
                params.append(model_id)
                conn.execute(
                    f"UPDATE model_metadata SET {', '.join(sets)} WHERE model_id = ?",
                    tuple(params),
                )
            if tags is not None:
                # Replace wholesale: the UI sends the full tag set, and a
                # merge would make removing a tag impossible.
                conn.execute("DELETE FROM model_tags WHERE model_id = ?", (model_id,))
                cleaned = sorted({t.strip() for t in tags if t and t.strip()})
                conn.executemany(
                    "INSERT OR IGNORE INTO model_tags (model_id, tag) VALUES (?, ?)",
                    [(model_id, t) for t in cleaned],
                )
            conn.commit()
        return self.get(model_id)

    def known_tags(self) -> list[str]:
        """Every tag in use, for the filter chips."""
        return [r["tag"] for r in self._db.execute(
            "SELECT DISTINCT tag FROM model_tags ORDER BY tag"
        )]

    def export(self) -> dict[str, Any]:
        """§5.3a: portable and backup-able alongside the corpus.

        This is the only copy of the operator's own thinking about 161 models.
        Being able to get it out is not a nicety.
        """
        return {"version": 1, "metadata": self.all()}

    def import_(self, payload: dict[str, Any]) -> int:
        """Merge an export back in. Additive; does not delete."""
        count = 0
        for model_id, meta in (payload.get("metadata") or {}).items():
            if not isinstance(meta, dict):
                continue
            self.update(
                model_id,
                description=meta.get("description"),
                favorite=meta.get("favorite"),
                archived=meta.get("archived"),
                tags=meta.get("tags"),
            )
            count += 1
        return count


class WarmSetStore:
    """The explicit warm set. Phase 4 / F7.

    Deliberately its own store and deliberately empty by default. See
    migration 3 and `api/operations.py` for why warm-on-startup is
    operator-triggered only.
    """

    def __init__(self, db: SQLiteManager) -> None:
        self._db = db

    def all(self) -> list[str]:
        return [r["model_id"] for r in self._db.execute(
            "SELECT model_id FROM warm_set ORDER BY model_id"
        )]

    def add(self, model_id: str, note: str = "") -> None:
        self._db.execute(
            "INSERT INTO warm_set (model_id, note) VALUES (?, ?) "
            "ON CONFLICT(model_id) DO UPDATE SET note = excluded.note",
            (model_id, note),
        )

    def remove(self, model_id: str) -> None:
        self._db.execute("DELETE FROM warm_set WHERE model_id = ?", (model_id,))
