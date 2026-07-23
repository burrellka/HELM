"""Resolve a model to a HuggingFace repo. Spec 14c §1.8 (guard 1).

Repo resolution is error-prone and a wrong guess is a confident lie: attach the
wrong card and the whole advisor is grounded on someone else's model. So this
is deliberately CONSERVATIVE. It resolves a repo only when the model's own
checkpoint carries an explicit `org/repo` id. Anything else, a local file path,
a bare FLM tag, an ambiguous name, resolves to "no confident repo", which the
advisor renders as "no external record" and backs by local facts alone.

It does NOT fuzzy-match a name to a plausible repo. `Aevum-0.6B-Finetuned` is
Kevin's own model and is not on HuggingFace; guessing `some-org/Aevum` for it
would be exactly the fabrication 14c §1.8 forbids. Absent data is shown as
absent.

Pure and table-tested. The actual HF fetch lives in the Layer 2 service; this
only decides WHICH repo (if any) to fetch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

#: An explicit HuggingFace repo id: `org/name`, each segment a normal token.
#: No leading slash (that is a filesystem path), no backslash, no drive letter.
_REPO_RE = re.compile(r"^([A-Za-z0-9][\w.-]+)/([A-Za-z0-9][\w.-]+)$")


class RepoConfidence(str, Enum):
    #: The checkpoint carried an explicit org/repo. Safe to fetch.
    EXPLICIT = "explicit"
    #: No repo could be resolved without guessing. The advisor shows local
    #: facts only and says "no external record".
    NONE = "none"


@dataclass(frozen=True)
class RepoResolution:
    confidence: RepoConfidence
    repo_id: str | None = None
    #: Why we resolved (or did not). Shown to the operator so the provenance
    #: of the external card is auditable.
    reason: str = ""


def _looks_like_path(text: str) -> bool:
    return (
        text.startswith("/")
        or text.startswith("~")
        or "\\" in text
        or bool(re.match(r"^[A-Za-z]:", text))  # windows drive
    )


def resolve_repo(checkpoint: str | None, model_id: str | None = None) -> RepoResolution:
    """Decide the HF repo for a model, or decline to guess.

    The checkpoint is the trustworthy source: Lemonade stores the pull source
    there, e.g. `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M`. We take the part
    before the first colon (the quant tag) and accept it only if it is a clean
    `org/repo`.
    """
    if checkpoint and not _looks_like_path(checkpoint):
        # Strip a trailing `:quant` or `:tag` and any `.gguf`.
        head = checkpoint.split(":", 1)[0].strip()
        head = re.sub(r"\.gguf$", "", head, flags=re.IGNORECASE)
        m = _REPO_RE.match(head)
        if m:
            return RepoResolution(
                RepoConfidence.EXPLICIT, repo_id=f"{m.group(1)}/{m.group(2)}",
                reason=f"Resolved from the checkpoint's explicit repo id ({head}).",
            )

    if checkpoint and _looks_like_path(checkpoint):
        return RepoResolution(
            RepoConfidence.NONE,
            reason=(
                "The checkpoint is a local file path, so there is no HuggingFace repo to look up. "
                "This is typically a custom or local model."
            ),
        )

    return RepoResolution(
        RepoConfidence.NONE,
        reason=(
            "No explicit repo id in the model's metadata, and HELM does not guess one from the "
            "name, because a wrong guess would attach the wrong model's card. Showing local "
            "facts only."
        ),
    )
