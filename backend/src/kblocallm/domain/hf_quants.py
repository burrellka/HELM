"""Grouping HuggingFace GGUF files into pickable quants.

Three traps a flat "every .gguf is a quant" list walks straight into, all of
which the ISTA Flash-Next and Gemma repos ship at once:

1. **Split GGUFs.** A quant shipped as `name-00001-of-00002.gguf` +
   `name-00002-of-00002.gguf` is ONE model, not two. Offering each shard as its
   own download gets the operator one shard; the load then fails
   `failed to open ...-00002-of-00002.gguf` and the failed load evicts the
   fleet. We group shards into one option, sum the sizes, and register the
   checkpoint on shard `00001` — llama.cpp finds the rest in the same directory.

2. **Subdirectories.** Repos organise quants as `IQ3_XXS/...gguf`,
   `Q2_0/...gguf`. A non-recursive tree read sees none of them. The caller must
   fetch the tree with `?recursive=true`; this module groups whatever paths it
   is handed, nested or not.

3. **Companions.** `mmproj-*` (vision projector), `*imatrix*` (calibration), and
   `mtp-*`/`draft-*`/`eagle-*` (speculative draft heads) are not standalone
   weights. Offering them as quants pulls a file llama-server can't start from.

Pure and table-tested: `group_gguf_files` takes `(path, size_bytes)` pairs and
returns one `QuantGroup` per real quant. The API layer adds the quant badge,
fit math, and pull identity on top.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# `stem-00001-of-00002.gguf`. Five digits each is the llama.cpp/HF convention.
# Case-insensitive on the extension; the index/total widths are fixed at 5.
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)


def is_companion(path: str) -> bool:
    """True for a .gguf that is NOT standalone weights (projector/calibration/
    draft head). Matches the filename, so a nested path is handled."""
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    return (
        "mmproj" in low
        or "imatrix" in low
        or base.startswith(("mtp-", "mtp_", "draft-", "draft_", "eagle-", "eagle_"))
        or "-mtp-" in base
        or "-draft" in base
    )


@dataclass
class QuantGroup:
    #: Stable grouping key (directory + shard stem, or the full path for a
    #: single file). Not shown; used only to dedup.
    key: str
    #: The checkpoint file to register on: shard `00001` for a split, else the
    #: single file. llama.cpp loads the rest of a split from this one.
    primary: str
    #: Every file to download, shard order. One entry for a non-split quant.
    shards: list[str] = field(default_factory=list)
    #: Summed size in bytes across all shards; None if any shard size is unknown
    #: (we never show a confident total built on a guessed part).
    size_bytes: int | None = None
    #: The containing directory, if the files are nested (e.g. "IQ3_XXS"). A
    #: good display/quant hint when the filename alone is ambiguous.
    subdir: str | None = None

    @property
    def is_split(self) -> bool:
        return len(self.shards) > 1

    @property
    def fallback_label(self) -> str:
        """A human label when the quant can't be decoded from the filename:
        the subdirectory if nested, else the shard stem."""
        if self.subdir:
            return self.subdir
        base = self.primary.rsplit("/", 1)[-1]
        m = _SHARD_RE.match(base)
        return m.group("stem") if m else base.removesuffix(".gguf").removesuffix(".GGUF")


def group_gguf_files(entries: list[tuple[str, int | None]]) -> list[QuantGroup]:
    """Group `(path, size_bytes)` pairs into one `QuantGroup` per real quant.

    Companions are dropped. Split shards collapse into one group with the summed
    size and the `00001` primary. Order of returned groups is by ascending total
    size (unknown sizes last), so the picker leads with the leanest option.
    """
    groups: dict[str, QuantGroup] = {}
    # Remember insertion order of shard (idx, path) so we can pick 00001 and
    # present shards in order even if the tree listed them out of order.
    shard_members: dict[str, list[tuple[int, str]]] = {}

    for path, size in entries:
        if not path.lower().endswith(".gguf") or is_companion(path):
            continue

        directory, _, base = path.rpartition("/")
        subdir = directory.rsplit("/", 1)[-1] if directory else None
        m = _SHARD_RE.match(base)

        if m:
            # One group per (directory, stem): shards of the same quant share both.
            key = f"{directory}/{m.group('stem')}"
            idx = int(m.group("idx"))
        else:
            key = path
            idx = 1

        g = groups.get(key)
        if g is None:
            g = QuantGroup(key=key, primary=path, subdir=subdir, size_bytes=0)
            groups[key] = g
            shard_members[key] = []
        shard_members[key].append((idx, path))

        # Sum sizes, but once any part is unknown the total is unknown — not a
        # smaller number that looks authoritative.
        if g.size_bytes is not None:
            g.size_bytes = (g.size_bytes + size) if isinstance(size, int) else None

    for key, g in groups.items():
        members = sorted(shard_members[key])
        g.shards = [p for _, p in members]
        # Primary = the lowest shard index present (normally 00001). If 00001 is
        # missing we still register on the lowest we have; the load-time shard
        # check is what refuses an incomplete set.
        g.primary = members[0][1]

    return sorted(
        groups.values(),
        key=lambda x: (x.size_bytes is None, x.size_bytes or 0),
    )
