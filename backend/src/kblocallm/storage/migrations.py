"""Versioned schema migrations.

Pattern from URSA: a `schema_version` table, an ordered list of migrations,
applied in one transaction at startup. Idempotent — safe to run every boot.

Rules:
  - Migrations are append-only. Never edit a shipped one; add a new one.
  - Never destructive without an explicit operator action. HELM's DB holds
    the operator's own metadata (descriptions, tags, favourites) — that is
    NOT reconstructible from Lemonade, so a careless migration loses data
    that has no upstream source of truth.

The two tables here are the frame for spec §5.3a (operator metadata) and F8
(watchdog incident acknowledgement). Column choices are the scaffold's
opinion; the HELM dev should feel free to reshape them — the point is the
migration machinery, not the domain model.
"""
from __future__ import annotations

import logging

from .db import SQLiteManager

logger = logging.getLogger(__name__)


# (version, description, sql). Append only.
_MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "operator model metadata (spec 5.3a) + watchdog incident acks (review F8)",
        """
        -- Spec 5.3a. Lemonade tracks NONE of this; it is HELM-local and it
        -- is the operator's own data. Keyed by the Lemonade model id.
        --
        -- NOTE for the dev: which Lemonade field is the stable key across
        -- /health (`model_name`) and /models (`id`) is UNVERIFIED — see
        -- review F9. Confirm against the live box before this table has
        -- rows you care about; a key change later means a migration.
        CREATE TABLE IF NOT EXISTS model_metadata (
            model_id     TEXT PRIMARY KEY,
            description  TEXT    NOT NULL DEFAULT '',
            favorite     INTEGER NOT NULL DEFAULT 0,  -- 0/1
            archived     INTEGER NOT NULL DEFAULT 0,  -- 0/1, the LocalAI 'disabled' idea
            created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- Separate table, not a JSON blob in the row above: the operator
        -- wants to FILTER by tag (spec 5.3a unified search), and filtering
        -- by a JSON substring is how you end up matching 'karl-only' inside
        -- 'not-karl-only'.
        CREATE TABLE IF NOT EXISTS model_tags (
            model_id  TEXT NOT NULL,
            tag       TEXT NOT NULL,
            PRIMARY KEY (model_id, tag),
            FOREIGN KEY (model_id) REFERENCES model_metadata(model_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_model_tags_tag ON model_tags(tag);

        -- Review F8: watchdog_reset appears to be sticky, so a raw flag
        -- becomes a permanent warning nobody reads. Record when the operator
        -- acknowledged an incident so HELM can show unacked ones loudly and
        -- acked ones quietly.
        CREATE TABLE IF NOT EXISTS watchdog_acks (
            model_id     TEXT NOT NULL,
            acked_at     TEXT NOT NULL DEFAULT (datetime('now')),
            -- Distinguishes one incident from the next. Populate with
            -- whatever proves stable (pid is a decent candidate: a restarted
            -- backend gets a new one). See F8.
            incident_key TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (model_id, incident_key)
        );
        """,
    ),
    (
        2,
        "last-seen backend pids — the crash detector (handoff §3 / F8)",
        """
        -- HELM's crash detection lives here because Lemonade has no memory of
        -- it: /health reports the CURRENT pid and nothing about the previous
        -- one, so "did this backend restart" is only answerable by something
        -- that persists across polls. That something is this table.
        --
        -- Why pid and not watchdog_reset: the flag stayed FALSE through a real
        -- induced crash on the live box (SIGKILL, pid 2628794 -> 2646871),
        -- because a request lazily reloaded the backend before the watchdog
        -- polled. It corroborates; it does not detect. See domain/crash.py.
        --
        -- Keyed by model_name: that is /health's key, and F9 confirmed live
        -- that it shares a namespace with /models' `id` (2/2 matched), so the
        -- Phase 3 labels join lands on the same key with no migration.
        CREATE TABLE IF NOT EXISTS backend_state (
            model_name            TEXT PRIMARY KEY,
            -- NULL when the router reported no usable pid. Distinct from a
            -- pid of 0, which we never store: an unknown pid must not compare
            -- unequal to the next unknown pid and manufacture a restart.
            pid                   INTEGER,
            backend_alive         INTEGER NOT NULL DEFAULT 1,
            -- When we last saw the pid change. The timestamp is the point:
            -- a bare sticky flag becomes a permanent warning nobody reads,
            -- which costs you the NEXT incident (F8).
            last_restart_at       TEXT,
            last_restart_from     INTEGER,
            last_restart_watchdog INTEGER NOT NULL DEFAULT 0,
            first_seen            TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at            TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,
    ),
    (
        3,
        "warm set (Phase 4, F7) + benchmark results (§5.7)",
        """
        -- F7: warm-on-startup is OPERATOR-TRIGGERED ONLY. There is no restart
        -- signal in the API (14a RESULTS §3 confirmed none exists), and
        -- all_models_loaded == [] is ALSO the nuclear option's signature — so
        -- HELM must never infer "it restarted" and mass-load into a router
        -- that is mid-incident clearing hardware state.
        --
        -- The set is explicit and starts EMPTY. Not "everything that was
        -- pinned": pinned means "don't evict", which is a different intent
        -- from "bring this back after a restart".
        CREATE TABLE IF NOT EXISTS warm_set (
            model_id   TEXT PRIMARY KEY,
            -- Warming an NPU model unattended evicts the incumbent with no
            -- human present to see the warning (Rule 7). Stored so the
            -- refusal is data, not a special case buried in code.
            added_at   TEXT NOT NULL DEFAULT (datetime('now')),
            note       TEXT NOT NULL DEFAULT ''
        );

        -- §5.7: store benchmark results so model choices are evidence-based
        -- rather than folklore.
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id       TEXT NOT NULL,
            scenario       TEXT NOT NULL,
            started_at     TEXT NOT NULL,
            -- Nullable: a run that failed or was cancelled is still a fact
            -- worth keeping. A missing result is not a zero.
            tokens_per_second  REAL,
            time_to_first_token REAL,
            output_tokens  INTEGER,
            ok             INTEGER NOT NULL DEFAULT 0,
            error          TEXT NOT NULL DEFAULT '',
            -- What the model was actually running WHEN measured. A tok/s
            -- number without its slot/context configuration is not
            -- comparable to the next one, and folklore is exactly what
            -- happens when you compare them anyway.
            config_note    TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_benchmark_model ON benchmark_runs(model_id);
        """,
    ),
    (
        4,
        "config backups — the auto-backup + one-click restore rail (14c §2.3)",
        """
        -- Every config apply (per-model recipe_options OR global config.json)
        -- snapshots the PRIOR version here first, timestamped, before the
        -- write. That is Kevin's backup-first SOP as a table: a raw edit can
        -- always be walked back, and a trailing comma that bricks the server
        -- has a known-good version one click away.
        CREATE TABLE IF NOT EXISTS config_backups (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            -- 'model:<id>' for Level A, 'global' for Level B. Scopes the
            -- restore so a model backup can never be applied as global config.
            scope      TEXT NOT NULL,
            -- The full prior config text (JSON), verbatim. Restore replays it
            -- through the same validate-diff-confirm-apply pipeline.
            content    TEXT NOT NULL,
            -- What triggered the backup: 'apply' | 'import' | 'reset'. Context
            -- for the operator reading the restore history.
            reason     TEXT NOT NULL DEFAULT 'apply',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_config_backups_scope
            ON config_backups(scope, created_at DESC);
        """,
    ),
    (
        5,
        "Model Advisor: HF metadata cache + advisor settings (14c Part 1)",
        """
        -- Layer 2 cache (14c §1.2). The HuggingFace lookup is cached by
        -- resolved repo id with a TTL, so the async Analyze button is one
        -- fetch then instant on repeat, and a LAN-only control plane is not
        -- hammering the open web. status distinguishes 'found' from
        -- 'not_found' so a genuine "no external record" is cached too and not
        -- re-fetched every time.
        CREATE TABLE IF NOT EXISTS hf_cache (
            repo_id    TEXT PRIMARY KEY,
            status     TEXT NOT NULL,          -- 'found' | 'not_found' | 'error'
            payload    TEXT NOT NULL DEFAULT '{}',
            fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Advisor settings (14c §1.7). Two-layer prompt: the operational layer
        -- is LOCKED in code (ground-only, no fabricated specs, label
        -- provenance) and is NOT stored here; only the editable persona layer
        -- and the analysis-model choice are. Single-row by convention (id=1).
        CREATE TABLE IF NOT EXISTS advisor_settings (
            id             INTEGER PRIMARY KEY CHECK (id = 1),
            -- The operator's chosen analysis model. NULL means "use the
            -- default" (the resident 26B), resolved at call time. Never a
            -- hardcoded id — Kevin's rule: default to the 26B, but always
            -- let the operator pick in settings.
            analysis_model TEXT,
            -- Editable tone/framing layer. NULL = ship the default persona.
            persona_prompt TEXT,
            updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,
    ),
    (
        6,
        "Advisor knowledge primer — vetted 'how local models work' reference",
        """
        -- Kevin + the architect maintain a primer on how local models work
        -- (quantization, MoE, context, hardware fit). Pinned to the advisor's
        -- instructions and injected as an AUTHORITATIVE reference source, so
        -- the advisor teaches general concepts from a vetted document rather
        -- than its own recall. That tightens grounding: when the primer is
        -- present, concept explanations come from it; the model-specific
        -- no-fabrication floor is unchanged.
        --
        -- Single-row, alongside the other advisor settings. NULL = no primer
        -- yet, and the advisor falls back to plain textbook explanation.
        ALTER TABLE advisor_settings ADD COLUMN knowledge_primer TEXT;
        """,
    ),
    (
        7,
        "Advisor web-search provider + model (operator-editable, not env-only)",
        """
        -- The search MODEL and PROVIDER move out of env-only config into the
        -- DB so the operator can change them from the UI without a redeploy
        -- (e.g. gemini-2.5-flash -> a newer grounding model when Google ships
        -- one). The KEY stays in the environment: it is the secret, and a
        -- secret does not belong in a table the UI reads. NULL = use the
        -- shipped default (gemini / gemini-2.5-flash).
        ALTER TABLE advisor_settings ADD COLUMN web_search_provider TEXT;
        ALTER TABLE advisor_settings ADD COLUMN web_search_model TEXT;
        """,
    ),
    (
        8,
        "Saved advisor summaries — persist the slow generated write-up per model",
        """
        -- A generated summary takes real time on the analysis model, so once
        -- the operator likes one they SAVE it and it shows instantly on
        -- return, on both the Advisor tab and the model screen, with a
        -- Regenerate to refresh. Keyed by model id; one saved summary per
        -- model (Regenerate + Save overwrites). analysis_model and generated_at
        -- are kept so the UI can say "generated by X on <date>" — a saved
        -- summary is a dated snapshot over facts that were live at the time,
        -- not a live read.
        CREATE TABLE IF NOT EXISTS advisor_summaries (
            model_id       TEXT PRIMARY KEY,
            summary        TEXT NOT NULL,
            analysis_model TEXT,
            generated_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,
    ),
    (
        9,
        "Summary its own feature (14c §1.10): separate model + instruction box",
        """
        -- Summary and Analyze had converged into the same expensive synthesis.
        -- 14c §1.10 splits them: Summary becomes a fast, starved-context read
        -- with its OWN model and its OWN instruction box, independent of the
        -- Analyze/synthesis pair above. The live verify (2026-07-19) showed the
        -- 26B run is GENERATION-bound (prefill of the fat context is ~0.1s warm;
        -- the wall is the 26B generating thousands of reasoning+answer tokens at
        -- ~54 tok/s). So the win is a small non-reasoning model (Aevum, 226
        -- tok/s) + a low token cap; the starved context is kept for grounding a
        -- sub-1B model and trimming the cold-prefill, not for prefill speed.
        --
        -- NULL on either column = use the shipped default (Aevum-0.6B for the
        -- model; the Appendix S instruction text for the box). Same locked-floor
        -- / editable-persona split as Analyze: the summary safety kernel lives
        -- in code and is NOT stored here, so no edit can turn off "facts card
        -- only, no lookup, no fabrication".
        ALTER TABLE advisor_settings ADD COLUMN summary_model TEXT;
        ALTER TABLE advisor_settings ADD COLUMN summary_prompt TEXT;
        """,
    ),
    (
        10,
        "Saved advisor write-ups: keep a Summary AND a Deep analysis per model",
        """
        -- The saved-summary slot was one row per model, which meant Summary and
        -- Deep analysis fought over it. The operator wants to save BOTH (the fast
        -- read and the slow teach), so the table is rebuilt keyed by
        -- (model_id, kind) where kind is 'summary' or 'deep'. Existing saved rows
        -- carry forward as kind='summary' (that is what the slot held before this).
        -- Rebuild rather than ALTER because SQLite cannot change a primary key in
        -- place; the SELECT preserves every existing saved summary.
        CREATE TABLE advisor_summaries_v2 (
            model_id       TEXT NOT NULL,
            kind           TEXT NOT NULL DEFAULT 'summary',
            summary        TEXT NOT NULL,
            analysis_model TEXT,
            generated_at   TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (model_id, kind)
        );
        INSERT INTO advisor_summaries_v2 (model_id, kind, summary, analysis_model, generated_at)
            SELECT model_id, 'summary', summary, analysis_model, generated_at
            FROM advisor_summaries;
        DROP TABLE advisor_summaries;
        ALTER TABLE advisor_summaries_v2 RENAME TO advisor_summaries;
        """,
    ),
    (
        11,
        "Remember the HuggingFace checkpoint a model was pulled from",
        """
        -- After a download, Lemonade often reports the checkpoint as the on-disk
        -- FILE PATH, not the HuggingFace repo it came from — so the advisor's
        -- external-metadata lookup sees a local path and honestly says "no
        -- external record", even for a model that plainly came from HF. When a
        -- pull goes through HELM we DO know the origin (the operator/catalog
        -- supplied it), so we remember it here, keyed by model id, and use it as
        -- the repo source when Lemonade's own checkpoint no longer resolves.
        -- Forward-only: models pulled before this (or via Lemonade directly)
        -- aren't recorded, and that's fine — HELM never invents an origin.
        CREATE TABLE IF NOT EXISTS model_source (
            model_id     TEXT PRIMARY KEY,
            checkpoint   TEXT NOT NULL,
            recipe       TEXT,
            recorded_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,
    ),
    (
        12,
        "Cache HuggingFace model cards (the README) for the download/update view",
        """
        -- The operator should see what a model IS before downloading or updating
        -- it. HuggingFace's JSON metadata carries no usable description, but the
        -- model card README does — so HELM fetches that and shows it. Cached by
        -- repo id alongside the metadata cache, same TTL logic, so opening the
        -- dialog twice does not re-hit HF. `card` is the frontmatter-stripped,
        -- size-bounded markdown; a row with empty card means "fetched, no card".
        CREATE TABLE IF NOT EXISTS hf_card_cache (
            repo_id     TEXT PRIMARY KEY,
            card        TEXT NOT NULL DEFAULT '',
            fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,
    ),
    (
        13,
        "Golden config pins: the operator-blessed expected args + backend build",
        """
        -- A HELM-owned baseline of what a model SHOULD run with, independent of
        -- whatever Lemonade currently has saved. The point (KAIROS/KODIAK ask,
        -- 2026-10-29): a reload or Lemonade upgrade can silently drop a load arg
        -- — most importantly `--reasoning off`, the lever that keeps the local
        -- 26B answering directly instead of thinking — and nobody notices until
        -- answers regress. HELM cannot PREVENT the box changing it (HELM does
        -- not own lemond startup), but it can hold the golden and LOUDLY flag any
        -- drift of the running config from it. Detect-and-alert ONLY: no auto
        -- re-apply (that is a consequential write on inferred state — needs the
        -- architect's ruling, same line as warm-on-startup).
        --
        -- `backend_build` is the llama.cpp build the golden was captured against
        -- (e.g. 'llama-b11394', parsed from /health launch_command), so an
        -- upgrade that swaps the engine is visible too. `llamacpp_args` is the
        -- intended arg string at capture time; comparison tracks the
        -- golden-significant flags only, so Lemonade's injected
        -- --spec-type/--mmproj/--model-draft never read as drift.
        CREATE TABLE IF NOT EXISTS golden_configs (
            model_id         TEXT PRIMARY KEY,
            ctx_size         INTEGER,
            llamacpp_backend TEXT,
            backend_build    TEXT,
            llamacpp_args    TEXT NOT NULL DEFAULT '',
            note             TEXT NOT NULL DEFAULT '',
            captured_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,
    ),
]


def apply_migrations(db: SQLiteManager) -> None:
    """Bring the DB up to the latest schema version. Idempotent."""
    with db.serialized() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "  version INTEGER PRIMARY KEY,"
            "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        cur = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
        current = cur.fetchone()[0]
        cur.close()

        for version, description, sql in _MIGRATIONS:
            if version <= current:
                continue
            logger.info("Applying migration %d: %s", version, description)
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (version,),
            )
            conn.commit()

        if current < _MIGRATIONS[-1][0]:
            logger.info("Schema now at version %d", _MIGRATIONS[-1][0])
