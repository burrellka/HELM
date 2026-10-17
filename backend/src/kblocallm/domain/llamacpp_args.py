"""The shared `llamacpp_args` parser. Review F4.

This is the highest-risk function in the product, and it is worth being blunt
about why. `recipe_options.llamacpp_args` is a free-text string the operator
can edit. HELM divides `ctx_size` by whatever `--parallel` it finds in there.
If the parse MISSES, the old naive code fell back to `default 1`, so
`per_slot = ctx_size = 524288`, and the dashboard displayed EIGHT TIMES the
real context with total confidence. That is the exact "worse than no UI"
failure the spec's §6 opens with.

So the contract here is: a value we are sure of, or a visible `unknown`.
Never a plausible number. `ParseStatus.MALFORMED` exists so the UI has
something honest to render, and `slots_from_args` refuses to guess.

**One parser, two consumers.** The Phase 2 reserved-arg validator (spec §5.3)
must reject `--ctx-size`, `-c`, AND `--ctx-size=4096` — the same three
spellings this module already normalizes. It uses `find_flag`, not its own
regex. Two hand-rolled arg parsers in one codebase WILL drift; that is URSA
1.1.15's bug class (two hand-maintained lists disagreed, the wrong value was
silent). If you need a new flag read, add an alias tuple and a test row here.
Do not write a second parser.

llama.cpp spellings this handles, all of which appear in the wild:
    --parallel 8      --parallel=8      -np 8      --parallel  8
    --parallel 8 --parallel 4   (last wins, matching llama.cpp itself)
    --parallel abc    (malformed -> unknown, never 1)
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import Enum

# llama.cpp's own aliases. `-np` is the short form and it is NOT hypothetical:
# it is the form most people type, and the form a naive `--parallel` regex
# misses, silently, by a factor of 8.
PARALLEL_FLAGS: tuple[str, ...] = ("--parallel", "-np")

# Sampling flags Lemonade stores inside llamacpp_args (verified live: the box's
# recipe_options carry `--temp 0.6 --top-p 0.85 --top-k 20 --min-p 0.0
# --repeat-penalty 1.0`). The settings editor lifts these into labeled fields
# instead of a raw text box, and this ONE module owns their spelling so the
# read and the compose can never disagree (F4).
TEMP_FLAGS: tuple[str, ...] = ("--temp", "--temperature")
TOP_K_FLAGS: tuple[str, ...] = ("--top-k",)
TOP_P_FLAGS: tuple[str, ...] = ("--top-p",)
MIN_P_FLAGS: tuple[str, ...] = ("--min-p",)
REPEAT_PENALTY_FLAGS: tuple[str, ...] = ("--repeat-penalty",)
#: Every sampling flag, for stripping them out of the "extra args" escape hatch
#: so they show in their own fields and never appear twice.
SAMPLING_FLAGS: tuple[str, ...] = (
    TEMP_FLAGS + TOP_K_FLAGS + TOP_P_FLAGS + MIN_P_FLAGS + REPEAT_PENALTY_FLAGS
)

# Reserved args (spec §5.3) — Lemonade hard-errors on these because they are
# how it auto-wires MTP, vision and device placement. Listed here, next to the
# parser, so the Phase 2 validator reads them from the same module that knows
# how to spell them. Not used in Phase 1; do not delete.
RESERVED_FLAGS: tuple[str, ...] = (
    "--ctx-size", "-c",
    "--mmproj",
    "--model-draft", "-md",
    "--spec-draft-model",
    "--device", "-dev",
    "--port",
    "--jinja",
    "--embedding",
    "--metrics",
    "--model", "-m",
)


#: Flags Lemonade removed or renamed, mapped to their replacement. Typing one
#: is no longer a no-op warning: in 2026.42.1 `--no-mmap` is `error: invalid
#: argument: --no-mmap`, llama-server exits, and the failed load evicts the
#: fleet (confirmed on the box). So HELM blocks these in the planner BEFORE the
#: load request, with the replacement named. One map, next to the parser, same
#: F4 reason as RESERVED_FLAGS: the spelling lives in exactly one place.
DEPRECATED_FLAGS: dict[str, str] = {
    "--no-mmap": "--load-mode none",
}


class ParseStatus(str, Enum):
    """Why you did or didn't get a value. All three are load-bearing.

    ABSENT and MALFORMED must NOT collapse into each other: absent means
    "llama.cpp's documented default applies", malformed means "the operator
    typed something we cannot read and we refuse to guess". The first is a
    number we can stand behind; the second must reach the screen as `unknown`.
    """

    FOUND = "found"
    ABSENT = "absent"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class IntFlag:
    """The result of looking up an integer flag."""

    status: ParseStatus
    value: int | None = None
    #: The token we could not read, kept for the UI/logs. "--parallel abc"
    #: is far more actionable than "unknown".
    raw: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is ParseStatus.FOUND


@dataclass(frozen=True)
class FlagOccurrence:
    """One appearance of a flag. `value` is None for a bare/boolean flag."""

    flag: str
    value: str | None


def tokenize(args: str | None) -> list[str] | None:
    """Split an args string into tokens, `--flag=value` kept intact.

    Returns None when the string cannot be tokenized at all (unbalanced
    quotes). None means "we do not know what this says" and every lookup on
    it must degrade to MALFORMED — NOT to an empty list, which would look
    exactly like "the operator set no flags" and hand us the 8x bug through
    the back door.
    """
    if args is None:
        return []
    if not args.strip():
        return []
    try:
        # posix=True so quoting works the way a shell would; the operator is
        # pasting from a terminal.
        return shlex.split(args, posix=True)
    except ValueError:
        return None


def find_flag(args: str | None, aliases: tuple[str, ...]) -> list[FlagOccurrence]:
    """Every occurrence of any alias, in order. The generic primitive.

    Both the `--parallel` read and the Phase 2 reserved-arg validator go
    through here, which is the whole point of F4: one place that knows
    `--ctx-size`, `-c` and `--ctx-size=4096` are the same flag.

    An untokenizable string yields no occurrences; callers that care about
    the difference between "not there" and "cannot read" must check
    `tokenize` themselves. `lookup_int` does.
    """
    tokens = tokenize(args)
    if tokens is None:
        return []

    found: list[FlagOccurrence] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # `--flag=value`. Split on the FIRST `=` only: a value may legitimately
        # contain one (`--override-kv foo=bar`).
        if "=" in tok:
            name, _, value = tok.partition("=")
            if name in aliases:
                found.append(FlagOccurrence(flag=name, value=value))
            i += 1
            continue

        if tok in aliases:
            # The next token is this flag's value only if it isn't itself a
            # flag. Deliberately NOT "starts with -": `--parallel -1` is a
            # negative VALUE, not a following flag, and treating it as a flag
            # would report ABSENT for an argument the operator plainly wrote.
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and not _looks_like_flag(nxt):
                found.append(FlagOccurrence(flag=tok, value=nxt))
                i += 2
                continue
            # Bare flag: either a boolean (`--jinja`) or a value flag the
            # operator left dangling. value=None; the caller decides which.
            found.append(FlagOccurrence(flag=tok, value=None))
            i += 1
            continue

        i += 1

    return found


def _looks_like_flag(token: str) -> bool:
    """True for `--temp` / `-np`, False for `8` / `-1` / `-0.5`.

    The int/float check is what separates a negative value from a short flag.
    Without it, `--parallel -1` parses as a bare `--parallel` followed by a
    mystery, and HELM reports the wrong thing about an argument that is right
    there in the string.
    """
    if not token.startswith("-"):
        return False
    try:
        float(token)
    except ValueError:
        return True
    return False


def lookup_int(args: str | None, aliases: tuple[str, ...]) -> IntFlag:
    """Read an integer flag. Last occurrence wins. Never guesses.

    Last-wins matches llama.cpp's own behaviour: it parses argv left to right
    and the later assignment overwrites the earlier one. `--parallel 8
    --parallel 4` really does run 4 slots, so HELM must say 4. The spec called
    this "undefined"; it isn't, and picking the first would be a confident lie.
    """
    if tokenize(args) is None:
        return IntFlag(status=ParseStatus.MALFORMED, raw=args)

    occurrences = find_flag(args, aliases)
    if not occurrences:
        return IntFlag(status=ParseStatus.ABSENT)

    last = occurrences[-1]
    if last.value is None:
        # `--parallel` with nothing after it. The operator meant something;
        # we don't know what. Refuse.
        return IntFlag(status=ParseStatus.MALFORMED, raw=last.flag)
    try:
        return IntFlag(status=ParseStatus.FOUND, value=int(last.value), raw=last.value)
    except ValueError:
        return IntFlag(
            status=ParseStatus.MALFORMED, raw=f"{last.flag} {last.value}",
        )


def lookup_float(args: str | None, aliases: tuple[str, ...]) -> float | None:
    """Read a float-valued flag (temp, top-p, min-p, repeat-penalty). Last wins.

    Returns None for absent, malformed, or unparseable — the settings editor
    treats all three as "field empty, use the model default", which is the
    honest behaviour: an empty sampling field means "do not set it", not zero.
    Uses the same tokenizer and flag finder as everything else (F4).
    """
    if tokenize(args) is None:
        return None
    occurrences = find_flag(args, aliases)
    if not occurrences or occurrences[-1].value is None:
        return None
    try:
        return float(occurrences[-1].value)
    except ValueError:
        return None


def parallel_slots(args: str | None) -> IntFlag:
    """How many llama.cpp slots this args string asks for.

    ABSENT is a real answer, not a failure: llama.cpp's documented default for
    `--parallel` is 1, so an args string without it means one slot. The caller
    (see `domain.concurrency`) is responsible for saying "1 slot (default)"
    rather than "1 slot", because those are different claims and only one of
    them is something the operator typed.

    A non-positive count is MALFORMED, not FOUND. `--parallel 0` divides
    ctx_size by zero; there is no honest number on the other side of that.
    """
    flag = lookup_int(args, PARALLEL_FLAGS)
    if flag.ok and (flag.value is None or flag.value < 1):
        return IntFlag(
            status=ParseStatus.MALFORMED,
            raw=f"--parallel {flag.value}",
        )
    return flag


def strip_flag(args: str | None, aliases: tuple[str, ...]) -> str:
    """Return `args` with every occurrence of `aliases` (and its value) removed.

    The settings editor needs this: the slots field owns `--parallel`, so the
    raw-args box must NOT also contain it — the planner rejects a value with
    two sources, because a value with two sources is one that silently
    disagrees with itself.

    Here rather than in the frontend deliberately. A regex in a React component
    that "removes --parallel" is a SECOND arg parser, and F4's whole point is
    that two hand-rolled parsers drift: this one would have to know about
    `-np`, `--parallel=8` and quoting all over again, and would get it subtly
    wrong the first time someone typed the short form.
    """
    tokens = tokenize(args)
    if not tokens:
        # None (untokenizable) or empty. Returning the original string on a
        # parse failure would hand back the flag we were asked to remove; the
        # caller gets nothing rather than something wrong.
        return "" if tokens is None else ""

    kept: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        name = tok.partition("=")[0] if "=" in tok else tok
        if name in aliases:
            if "=" in tok:
                i += 1
                continue
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 2 if (nxt is not None and not _looks_like_flag(nxt)) else 1
            continue
        kept.append(tok)
        i += 1

    # shlex.join would quote defensively and churn the string the operator
    # typed; a plain join round-trips the common case unchanged.
    return " ".join(kept)


def find_reserved(args: str | None) -> list[str]:
    """Reserved args present in this string. Spec §5.3, Phase 2.

    Here in Phase 1 because it is the second consumer F4 requires, and its
    existence is what stops someone writing a second parser in Phase 2. It is
    tested; it is not yet wired to an endpoint.
    """
    hits: list[str] = []
    for occurrence in find_flag(args, RESERVED_FLAGS):
        if occurrence.flag not in hits:
            hits.append(occurrence.flag)
    return hits


def find_deprecated(args: str | None) -> list[tuple[str, str]]:
    """Deprecated/removed flags present, each paired with its replacement.

    Same mechanism as `find_reserved` (one parser, F4), but these are blocked
    for a different reason: they are not Lemonade-owned, they are simply gone,
    and passing one now hard-fails the load and evicts the fleet.
    """
    hits: list[tuple[str, str]] = []
    seen: set[str] = set()
    for occurrence in find_flag(args, tuple(DEPRECATED_FLAGS)):
        if occurrence.flag not in seen:
            seen.add(occurrence.flag)
            hits.append((occurrence.flag, DEPRECATED_FLAGS[occurrence.flag]))
    return hits
