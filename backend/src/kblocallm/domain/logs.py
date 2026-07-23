"""Log line classification. Spec §5.6.

The log pane's value is not the text; it is spotting the four events that
explain everything else, in a stream that is mostly noise:

  - `(BackendWatchdog)` — a backend crashed and was restarted.
  - `evict` / `Unloading models` — including the **nuclear option**: a backend
    failure makes the router evict EVERYTHING to clear hardware state. When
    that fires, every model went cold and every agent is about to notice.
  - `initializing slots, n_slots = N` / `new slot, n_ctx = N` — proof a load
    actually took effect, and the only place the REAL slot count is stated by
    the thing doing the work rather than derived by us.
  - `Largest memory pool:` — the line that must read ~120. This is the 80GB
    saga's tell, printed at every startup, and it went unread for months.

Pure and table-tested: the highlighting is the feature, so it needs to be
provable without a running lemond.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class LogEvent(str, Enum):
    WATCHDOG = "watchdog"
    #: Ordinary LRU eviction. Expected under capacity pressure.
    EVICT = "evict"
    #: The nuclear option: everything evicted to clear hardware state.
    MASS_EVICT = "mass_evict"
    #: A load taking effect, with the backend's own slot/context numbers.
    SLOTS = "slots"
    #: The pool figure at startup.
    POOL = "pool"
    ERROR = "error"
    PLAIN = "plain"


@dataclass(frozen=True)
class LogLine:
    text: str
    event: LogEvent = LogEvent.PLAIN
    #: Shown to the operator. Explains why the line was worth catching.
    note: str | None = None


_SLOT_RE = re.compile(r"n_slots\s*=\s*(\d+)|new slot.*n_ctx\s*=\s*(\d+)", re.I)
_POOL_RE = re.compile(r"largest memory pool[:\s]+([\d.]+)", re.I)


def classify(line: str) -> LogLine:
    """Classify one log line. Never raises; an unclassified line is PLAIN."""
    if not line or not line.strip():
        return LogLine(text=line)

    low = line.lower()

    if "backendwatchdog" in low:
        return LogLine(
            text=line, event=LogEvent.WATCHDOG,
            note=(
                "The watchdog acted on a backend. Corroborates a crash — but note it stays "
                "silent when a request reloads a dead backend first, so its absence proves "
                "nothing."
            ),
        )

    # Order matters: the nuclear option also contains the word "unloading",
    # so the broader mass-evict test must run BEFORE the ordinary evict test
    # or every catastrophe reads as routine housekeeping.
    if "unloading models" in low or "unloading all" in low or (
        "evict" in low and ("all" in low or "everything" in low)
    ):
        return LogLine(
            text=line, event=LogEvent.MASS_EVICT,
            note=(
                "The router evicted EVERYTHING to clear hardware state after a backend "
                "failure. Every model just went cold. Do not mass-load into this — the router "
                "is mid-incident."
            ),
        )

    if "evict" in low:
        return LogLine(
            text=line, event=LogEvent.EVICT,
            note="LRU eviction: the router made room by dropping the least recently used unpinned model.",
        )

    match = _SLOT_RE.search(line)
    if match:
        value = match.group(1) or match.group(2)
        return LogLine(
            text=line, event=LogEvent.SLOTS,
            note=(
                f"The backend states its own configuration here ({value}). This is ground "
                f"truth for the slot/context maths, straight from llama.cpp."
            ),
        )

    match = _POOL_RE.search(line)
    if match:
        try:
            pool = float(match.group(1))
        except ValueError:
            pool = None
        note = "The memory pool llama.cpp actually sees."
        if pool is not None and pool < 16:
            note = (
                f"⚠ The pool reads {pool} GB. That is the BIOS carve-out, not the unified "
                f"pool — the 80GB saga's exact signature. Check enable_dgpu_gtt."
            )
        return LogLine(text=line, event=LogEvent.POOL, note=note)

    if any(token in low for token in ("error", "traceback", "failed", "exception")):
        return LogLine(text=line, event=LogEvent.ERROR)

    return LogLine(text=line)
