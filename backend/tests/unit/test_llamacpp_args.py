"""The F4 test table.

Every row here is a spelling that, if misparsed, makes HELM display a
confidently wrong context size. The review's table is reproduced verbatim as
`test_the_review_f4_table` — if that test ever goes red, the dashboard is
lying by a factor of `--parallel`.
"""
from __future__ import annotations

import pytest

from kblocallm.domain.llamacpp_args import (
    PARALLEL_FLAGS,
    SAMPLING_FLAGS,
    TEMP_FLAGS,
    TOP_P_FLAGS,
    ParseStatus,
    find_reserved,
    lookup_float,
    lookup_int,
    parallel_slots,
    strip_flag,
    tokenize,
)


# -- the review's own table, verbatim --------------------------------------


@pytest.mark.parametrize(
    "args,expected",
    [
        ("--parallel 8", 8),
        ("--parallel=8", 8),           # naive regex: MISSES -> 8x wrong
        ("-np 8", 8),                  # naive regex: MISSES -> 8x wrong
        ("--parallel  8", 8),          # double space
        ("--parallel 8 --parallel 4", 4),  # last wins, as llama.cpp does
    ],
)
def test_the_review_f4_table(args, expected):
    result = parallel_slots(args)
    assert result.ok, f"{args!r} must parse; a miss here is an 8x lie"
    assert result.value == expected


def test_malformed_value_is_unknown_never_a_number():
    """The whole point of F4. `--parallel abc` must not become 1."""
    result = parallel_slots("--parallel abc")
    assert result.status is ParseStatus.MALFORMED
    assert result.value is None
    assert "abc" in (result.raw or ""), "keep the bad token; 'unknown' alone isn't actionable"


# -- absent vs malformed, which must never collapse ------------------------


@pytest.mark.parametrize("args", ["", "   ", None, "--temp 1.0 --top-k 64"])
def test_no_parallel_flag_reads_as_absent_not_malformed(args):
    """Aevum on the live box runs exactly this: llamacpp args with sampling
    flags and no --parallel. That is one slot by llama.cpp's default, and a
    real answer — distinct from 'we cannot read this'."""
    assert parallel_slots(args).status is ParseStatus.ABSENT


def test_dangling_flag_is_malformed_not_absent():
    """`--parallel` with no value: the operator meant something and we don't
    know what. Guessing 1 here is the 8x bug."""
    assert parallel_slots("--temp 1.0 --parallel").status is ParseStatus.MALFORMED


def test_unbalanced_quotes_degrade_to_malformed_not_to_empty():
    """An untokenizable string must NOT look like 'no flags set'. That path
    silently returns the default and reintroduces the bug through the back
    door."""
    assert tokenize('--parallel 8 --foo "unclosed') is None
    assert parallel_slots('--parallel 8 --foo "unclosed').status is ParseStatus.MALFORMED


# -- values that break naive parsers ---------------------------------------


@pytest.mark.parametrize("args", ["--parallel 0", "--parallel -1", "--parallel=0"])
def test_non_positive_slot_count_is_malformed(args):
    """There is no honest per-slot context on the other side of a divide by
    zero, and a negative slot count is nonsense the operator should see."""
    assert parallel_slots(args).status is ParseStatus.MALFORMED


def test_negative_value_is_read_as_a_value_not_a_following_flag():
    """`-1` is a value, not a short flag. Generic lookup keeps it; only
    parallel_slots rejects it, and rejects it for being non-positive rather
    than for being unreadable."""
    result = lookup_int("--ctx-size -1", ("--ctx-size",))
    assert result.ok and result.value == -1


def test_equals_form_with_a_value_containing_equals():
    """Split on the first `=` only — llama.cpp really does take
    `--override-kv key=value`."""
    result = lookup_int("--override-kv foo=bar --parallel=4", ("--parallel",))
    assert result.ok and result.value == 4


def test_flag_prefix_is_not_a_match():
    """`--parallel-something` is a different flag. A substring match ('--parallel'
    in args) — which is what the old CLI dashboard did — matches it and reads
    the wrong value."""
    assert parallel_slots("--parallel-foo 8").status is ParseStatus.ABSENT


def test_quoted_args_survive():
    result = parallel_slots('--chat-template "a b c" --parallel 8')
    assert result.ok and result.value == 8


def test_the_live_box_gemma_args_parse():
    """Verbatim from /api/v1/health on the Lemonade host, 2026-07-16. The 26B six
    agents depend on: ctx_size 524288 / 8 slots = 65536 per slot."""
    result = parallel_slots("--parallel 8 --temp 1.0 --top-k 64 --top-p 0.95")
    assert result.ok and result.value == 8


# -- the second consumer (F4: one parser, two consumers) -------------------


@pytest.mark.parametrize(
    "args,expected",
    [
        # The live 26B: --parallel goes, the sampling settings stay untouched.
        ("--parallel 8 --temp 1.0 --top-k 64 --top-p 0.95", "--temp 1.0 --top-k 64 --top-p 0.95"),
        ("--parallel=8 --temp 1.0", "--temp 1.0"),        # equals form
        ("-np 8 --temp 1.0", "--temp 1.0"),               # short form
        ("--temp 1.0 --parallel 8", "--temp 1.0"),        # trailing
        ("--temp 1.0", "--temp 1.0"),                     # nothing to strip
        ("--parallel 8", ""),                             # nothing left
        ("", ""),
        (None, ""),
        ("--parallel 8 --parallel 4 --temp 1.0", "--temp 1.0"),  # every occurrence
    ],
)
def test_strip_flag_removes_parallel_in_every_spelling(args, expected):
    """The settings editor's prefill rides on this. A regex in the frontend
    would be a second arg parser, would miss `-np`, and would silently leave a
    duplicate --parallel the planner then rejects."""
    assert strip_flag(args, PARALLEL_FLAGS) == expected


def test_strip_flag_on_unparseable_args_returns_nothing_rather_than_the_flag():
    """Handing back the original string on a parse failure would return the
    very flag we were asked to remove."""
    assert strip_flag('--parallel 8 --foo "unclosed', PARALLEL_FLAGS) == ""


def test_strip_flag_leaves_lookalike_flags_alone():
    assert strip_flag("--parallel-foo 8 --temp 1.0", PARALLEL_FLAGS) == "--parallel-foo 8 --temp 1.0"


@pytest.mark.parametrize(
    "args,expected",
    [
        ("--ctx-size 4096", ["--ctx-size"]),
        ("--ctx-size=4096", ["--ctx-size"]),   # the form a regex for '--ctx-size ' misses
        ("-c 4096", ["-c"]),                    # the short form
        ("--parallel 8 --temp 1.0", []),
        ("--jinja", ["--jinja"]),               # bare boolean, still reserved
    ],
)
def test_reserved_flags_are_found_in_every_spelling(args, expected):
    """Phase 2's validator rides on this. It must catch --ctx-size, -c AND
    --ctx-size=4096 — and it must do so through the SAME parser, or the two
    lists drift and the divergence is silent (URSA 1.1.15)."""
    assert find_reserved(args) == expected


# -- lookup_float: the settings editor's sampling reads --------------------


@pytest.mark.parametrize(
    "args,aliases,expected",
    [
        ("--temp 0.7", TEMP_FLAGS, 0.7),
        ("--temperature 0.7", TEMP_FLAGS, 0.7),   # the long alias too
        ("--temp=0.7", TEMP_FLAGS, 0.7),          # the = spelling
        ("--top-p 0.9 --temp 0.7", TEMP_FLAGS, 0.7),
        ("--temp 0.2 --temp 0.8", TEMP_FLAGS, 0.8),  # last wins, like llama.cpp
        ("--repeat-penalty 1", TOP_P_FLAGS, None),   # absent -> None, never 0.0
        ("--temp abc", TEMP_FLAGS, None),            # malformed -> None, never a guess
        ("--temp", TEMP_FLAGS, None),                # dangling -> None
        ("'--temp 0.7", TEMP_FLAGS, None),           # untokenizable -> None
    ],
)
def test_lookup_float_reads_or_honestly_declines(args, aliases, expected):
    assert lookup_float(args, aliases) == expected


def test_sampling_flags_bundle_covers_every_field():
    """The strip in _settings_from removes the whole bundle in one pass; if a
    flag is missing from SAMPLING_FLAGS it would leak back into extra_args."""
    for alias in ("--temp", "--top-k", "--top-p", "--min-p", "--repeat-penalty"):
        assert alias in SAMPLING_FLAGS


def test_stripping_the_sampling_bundle_clears_them_all():
    left = strip_flag(
        "--parallel 2 --temp 0.7 --top-k 40 --min-p 0.05 --keep-me on",
        PARALLEL_FLAGS + SAMPLING_FLAGS,
    )
    assert left == "--keep-me on"
