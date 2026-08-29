"""Settings — env-driven, one source of truth.

House rule (from URSA): nothing reads os.environ directly outside this
module. Everything else takes a Settings. That's what makes the app
testable without monkeypatching the environment.

Every var is prefixed HELM_ so a shared .env on the Dockge host can't
collide with another stack's vars.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HELM_", extra="ignore")

    # --- HELM's own state -------------------------------------------------
    # SQLite lives here, alongside jwt_secret and auth.json. In the container
    # this is /data, bind-mounted from a host directory you choose (see
    # infra/docker-compose.production.yml).
    #
    # SQLite by design: HELM stores operator metadata and settings (key/value-ish,
    # low volume, transactional), not analytics — SQLite is the right tool.
    data_dir: Path = Field(default=Path("/data"))

    # --- Upstream: Lemonade ----------------------------------------------
    # Your Lemonade Server's URL. Override with HELM_LEMONADE_URL. If Lemonade is
    # a host service rather than a container, use the host's LAN IP.
    lemonade_url: str = Field(default="http://localhost:13305")

    # Gates Lemonade's management/write routes. HELM's backend holds this;
    # it must NEVER be sent to the browser (spec §4, rule 9).
    #
    # Empty is a valid, supported state: HELM runs read-only and the UI
    # says so. That's deliberate — see startup_report() in main.py. It is
    # NOT an error to run HELM without write access.
    lemonade_admin_api_key: str = Field(default="")

    # Set only if Lemonade's global LEMONADE_API_KEY is ever enabled (it
    # gates ALL routes including reads, and breaks Lemonade's own bundled
    # UI — spec §4). Usually unset.
    lemonade_api_key: str = Field(default="")

    # --- HELM's own auth --------------------------------------------------
    # See HELM_SPEC_REVIEW.md F1. The spec as written has no operator auth;
    # this scaffold ships it on by default because HELM can unload the model
    # six agents depend on and delete 20GB of weights.
    #
    # Leave unset and the app auto-generates a per-instance secret at
    # <data_dir>/jwt_secret on first boot. No key ceremony.
    jwt_secret: str = Field(default="")

    # Escape hatch, NOT a default. If the decision is ever "mTLS at the edge
    # is the only gate", set this true AND bind the container port to
    # 127.0.0.1 in compose. Logged loudly at every boot so it can't be
    # forgotten. Read F1 before you touch this.
    disable_auth: bool = Field(default=False)

    # --- Model Advisor web search (14c Part 1, ported from KAIROS) ---------
    # Grounded lookup for the advisor. Gemini Google-Search grounding by
    # default: a real server-side Google Search returning a cited answer, so
    # the advisor's "what's said online" stays grounded, not recalled.
    #
    # This is a SEPARATE key from any model or admin key (KAIROS's rule: the
    # search key is its own credential with its own quota). Free tier is
    # 1,500 grounded requests/day on gemini-2.5-flash. Empty = search off, and
    # the advisor says a live lookup was not run rather than inventing one.
    #
    # Only the KEY lives here (it is the secret). The provider and model are
    # operator-editable from the advisor UI and stored in the DB, so they can
    # change without a redeploy — see advisor_service.AdvisorSettingsStore.
    web_search_key: str = Field(default="")

    # --- Local GPU sensors (temp/power) -----------------------------------
    # Lemonade's /api/v1/system-stats exposes gpu_percent but NOT temperature
    # or power. When helm-api runs ON the GPU host, HELM reads those two from
    # the kernel's hwmon sysfs directly — they read correctly on every AMD GPU
    # even where the utilization gauge is stuck (a documented gfx1151 driver
    # bug, ROCm #6035/#5750). This is the DRM class root it globs under; the
    # default is correct in-container when /sys is present. On a box where HELM
    # is NOT co-located with the GPU (or a non-Linux dev host) the path is
    # absent and temp/power simply read as unavailable — never faked.
    drm_root: Path = Field(default=Path("/sys/class/drm"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "helm.db"

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.web_search_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
