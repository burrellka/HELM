"""FastAPI app factory.

The house pattern, lifted from URSA-OSCAR's main.py. The two things worth
copying exactly:

  1. `_AUTH_REQUIRED` applied at include_router level, so auth holds by
     construction. On a control plane a forgotten decorator is an open
     POST /unload. Do not auth per-endpoint. (Review F1a.)
  2. A loud, honest startup report. The operator should learn "management is
     disabled because there's no admin key" from the boot log, not from a
     button that 401s an hour later. (URSA 1.1.8's lesson.)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from .api import advisor, config, control, dashboard, health, library, operations, system
from .auth import AuthStore, require_auth, resolve_jwt_secret
from .auth.routes import router as auth_router
from .config import Settings, get_settings
from .storage.db import SQLiteManager
from .storage.migrations import apply_migrations
from .upstream.client import LemonadeClient
from .version import get_version

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Make our own INFO logs visible under uvicorn.

    Not boilerplate: uvicorn installs handlers for its own loggers and leaves
    the root at WARNING, so everything _startup_report logs at INFO vanishes
    and only the warnings survive. That's backwards — the startup report
    exists to be read, and "management: DISABLED" is more useful when it
    arrives next to the version and the upstream URL rather than alone.

    Attach a handler to the package logger only; don't reconfigure root and
    fight uvicorn for its own output.
    """
    pkg = logging.getLogger("kblocallm")
    if pkg.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
    pkg.addHandler(handler)
    pkg.setLevel(logging.INFO)
    pkg.propagate = False


def _startup_report(settings: Settings, client: LemonadeClient) -> None:
    """Say what state we're actually in, once, at boot.

    Every line here exists because a silent version of it cost someone time
    in a sibling app.
    """
    logger.info("HELM %s starting", get_version())
    logger.info("  data dir      : %s", settings.data_dir)

    # State the uid unprompted. This cost a deploy: the compose ran the
    # container as a uid that did not exist on the host, against a data dir
    # that was correctly owned all along, and the only symptom was sqlite
    # refusing to open the database. "Which user am I?" should never require a
    # failure to answer.
    uid = getattr(os, "getuid", lambda: -1)()
    if uid >= 0:
        gid = getattr(os, "getgid", lambda: -1)()
        logger.info(
            "  running as    : uid=%d gid=%d%s", uid, gid,
            " (root — writes to the data volume will be root-owned)" if uid == 0 else "",
        )

    logger.info("  lemonade      : %s", settings.lemonade_url)

    if client.has_admin_key:
        logger.info("  management    : ENABLED (admin key present)")
    else:
        # Not an error. Read-only HELM is a legitimate, supported mode — and
        # a perfectly good Phase 1 (spec §7). But it must be stated, because
        # the alternative is an operator clicking Unload and getting a bare
        # 401 with no clue why.
        logger.warning(
            "  management    : DISABLED — no HELM_LEMONADE_ADMIN_API_KEY set. "
            "Reads work; load/unload/pin/delete/config will refuse. Put your "
            "Lemonade admin key in the stack's .env to enable writes."
        )

    if settings.disable_auth:
        # Impossible to leave on by accident.
        logger.warning(
            "  operator auth : *** DISABLED *** — HELM_DISABLE_AUTH=true. "
            "Anyone who can reach this port can unload models and delete "
            "weights. This is only defensible if the port is bound to "
            "127.0.0.1 AND the edge enforces mTLS. See HELM_SPEC_REVIEW.md F1."
        )
    else:
        logger.info("  operator auth : enabled (session cookie / bearer)")


def _preflight_data_dir(settings: Settings) -> None:
    """Prove we can write to the data dir BEFORE sqlite tries to.

    Without this, a data dir the container cannot write produces
    `sqlite3.OperationalError: unable to open database file` under forty lines
    of lifespan traceback — which says nothing about uids, nothing about the
    mount, and nothing about what to do. That is the exact experience the
    startup report exists to prevent, and it cost a deploy.

    The common cause is mundane and invisible: the host dir was created with
    `sudo mkdir` and never chowned, so it is root-owned, and the container runs
    as a non-root `user:` that cannot write it. Root would have silently
    bypassed the permission check; a non-root uid does not.

    Note SQLite needs write access to the DIRECTORY, not just the file: it
    creates helm.db plus the -wal and -shm sidecars beside it. A read-only dir
    with a writable file still fails.
    """
    data_dir = settings.data_dir
    # getuid/getgid are POSIX-only; the test suite runs on Windows.
    uid = getattr(os, "getuid", lambda: -1)()
    gid = getattr(os, "getgid", lambda: -1)()

    def _who() -> str:
        return f"uid={uid} gid={gid}" if uid >= 0 else "this process"

    def _owner() -> str:
        try:
            st = data_dir.stat()
            return f"owned by uid={st.st_uid} gid={st.st_gid}, mode {oct(st.st_mode)[-3:]}"
        except OSError:
            return "and its ownership could not be read"

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"HELM cannot create its data directory {data_dir} ({e}). It is running as "
            f"{_who()}. In the container this path is a bind mount — check the host side of "
            f"the `volumes:` entry in the stack's compose file exists."
        ) from e

    probe = data_dir / ".helm-write-probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as e:
        raise RuntimeError(
            f"HELM cannot write to its data directory {data_dir}, so SQLite, the JWT secret "
            f"and the operator password have nowhere to live.\n"
            f"  Running as: {_who()}\n"
            f"  {data_dir} is {_owner()}\n"
            f"  Underlying error: {e}\n"
            f"This is almost always a uid mismatch on the bind mount: the host directory was "
            f"created as root and never chowned to the user the container runs as. Fix it by "
            f"chowning your mounted data directory to the container's user, e.g.:\n"
            f"  sudo chown -R $(id -u):$(id -g) /path/to/your/helm/data\n"
            f"  sudo chmod 775 /path/to/your/helm/data\n"
            f"Or set PUID/PGID in the stack's .env to the uid that already owns it "
            f"(check with: stat -c '%u %g' /path/to/your/helm/data). "
            f"Note SQLite needs to write the DIRECTORY, not just the file — it creates "
            f"helm.db plus -wal and -shm sidecars."
        ) from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    settings = get_settings()

    # Before anything touches the disk. A clear refusal here beats a sqlite
    # stack trace forty frames deep.
    _preflight_data_dir(settings)

    db = SQLiteManager(settings.db_path)
    apply_migrations(db)
    app.state.db = db

    # env > persisted file > first-boot autogen. No key ceremony.
    app.state.jwt_secret = resolve_jwt_secret(settings.data_dir)
    app.state.auth_store = AuthStore(store_path=settings.data_dir / "auth.json")
    app.state.auth_disabled = settings.disable_auth

    client = LemonadeClient(
        settings.lemonade_url,
        admin_api_key=settings.lemonade_admin_api_key,
        api_key=settings.lemonade_api_key,
    )
    app.state.lemonade = client

    _startup_report(settings, client)
    try:
        yield
    finally:
        await client.aclose()
        db.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="HELM",
        description="AI fleet control plane over Lemonade Server.",
        version=get_version(),
        lifespan=lifespan,
    )

    # No CORS middleware, deliberately.
    #
    # One nginx serves the SPA and proxies /api to this app, so every request
    # is same-origin and there is no cross-origin caller to permit. Adding
    # permissive CORS to a write-capable control plane would hand any website
    # the operator visits the ability to drive it. If you think you need CORS
    # here, you've probably split the origins by accident — fix that instead.
    # (House conflict #3 in HELM_SPEC_REVIEW.md.)

    # AUTH-OPEN, and these two only:
    #   health   — container healthcheck / uptime monitors
    #   auth     — the login endpoints themselves (they check credentials
    #              internally; /auth/status must answer before you have a token)
    app.include_router(health.router)
    app.include_router(auth_router)

    # Everything below requires auth, applied at the router level so a new
    # endpoint on an existing router CANNOT forget to be guarded.
    _AUTH_REQUIRED = [Depends(require_auth)]
    app.include_router(system.router, dependencies=_AUTH_REQUIRED)
    # Phase 1. Read-only today, but mounted guarded anyway: the Phase 2
    # control endpoints land on routers that come through here, and F1a's
    # whole point is that they cannot arrive unguarded.
    app.include_router(dashboard.router, dependencies=_AUTH_REQUIRED)
    # Phase 2/3/4. control and library carry every write HELM can make —
    # load, unload, pin, pull, delete — which is exactly why they are mounted
    # here rather than decorated per-endpoint. A forgotten decorator on
    # POST /control/unload is the whole ballgame (F1a).
    app.include_router(control.router, dependencies=_AUTH_REQUIRED)
    app.include_router(library.router, dependencies=_AUTH_REQUIRED)
    app.include_router(operations.router, dependencies=_AUTH_REQUIRED)
    app.include_router(config.router, dependencies=_AUTH_REQUIRED)
    app.include_router(advisor.router, dependencies=_AUTH_REQUIRED)

    return app


app = create_app()
