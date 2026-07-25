"""Per-model throughput from Lemonade's Prometheus `/metrics` (14c §1.12).

The dashboard's throughput strip reads `/stats`, which is WHOLE-SERVER — the
last request against whichever model. `/metrics` carries the same numbers keyed
`lemonade_model_*{model_name="…"}`, so the per-model detail view can show each
model's own last-request rate, TTFT, and lifetime counters.

Two things this module refuses to get wrong:

- **The fabricated 1,000,000 tok/s.** Same artifact as `/stats`: a 1-token
  response over a near-zero elapsed time. On the live box LFM2.5 reads exactly
  `lemonade_model_tokens_per_second … 1000000`. It is not a rate, and it gets
  the same plausibility guard the dashboard uses — surfaced as "no measurement"
  with the reason, never printed beside "tok/s".
- **Absent vs zero.** A model with no series (never served a request, or a
  non-llm) yields None fields, not 0 — the F-series discipline. A metric we did
  not read is unknown, not a measured zero.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .dashboard import PLAUSIBLE_MAX_TOKENS_PER_SECOND

#: One Prometheus sample line: `name{labels} value`. Comments (`# …`) are skipped
#: before this ever runs. The value is the last whitespace-separated token.
_LINE = re.compile(r"^(?P<name>lemonade_model_[a-z_]+)\{(?P<labels>[^}]*)\}\s+(?P<value>\S+)\s*$")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class ModelMetrics:
    model_name: str
    checkpoint: str | None = None
    #: Last-request rate. None when unread; see rate_implausible for the 1e6 case.
    tokens_per_second: float | None = None
    rate_implausible: bool = False
    rate_note: str | None = None
    #: Seconds. Lemonade reports it in seconds; the UI converts to ms.
    time_to_first_token: float | None = None
    requests_total: int | None = None
    input_tokens_total: int | None = None
    output_tokens_total: int | None = None
    #: Last request's output token count — only kept to explain an implausible rate.
    last_output_tokens: int | None = None


def _f(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f


def _i(v: str | None) -> int | None:
    f = _f(v)
    return int(f) if f is not None else None


def parse_model_metrics(text: str | None) -> dict[str, ModelMetrics]:
    """Group Lemonade's `/metrics` into per-model throughput.

    Returns model_name -> ModelMetrics. A model absent from `/metrics` is simply
    absent from the map (the caller renders "unavailable"), never a zeroed row.
    """
    if not isinstance(text, str) or not text:
        return {}

    # model_name -> {metric_suffix: raw_value}, plus a captured checkpoint label.
    by_model: dict[str, dict[str, str]] = {}
    checkpoints: dict[str, str] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels = dict(_LABEL.findall(m.group("labels")))
        name = labels.get("model_name")
        if not name:
            continue
        suffix = m.group("name")[len("lemonade_model_"):]
        by_model.setdefault(name, {})[suffix] = m.group("value")
        if labels.get("checkpoint"):
            checkpoints[name] = labels["checkpoint"]

    out: dict[str, ModelMetrics] = {}
    for name, series in by_model.items():
        rate = _f(series.get("tokens_per_second"))
        implausible = rate is not None and rate > PLAUSIBLE_MAX_TOKENS_PER_SECOND
        last_out = _i(series.get("output_tokens"))
        note = None
        if implausible:
            note = (
                f"Lemonade reports {rate:,.0f} tok/s for this model's last request, which "
                f"is not a real rate — it is a tiny token count over a near-zero elapsed "
                f"time"
                + (f" (that request produced {last_out} token(s))." if last_out is not None else ".")
                + " Treated as no measurement."
            )
        out[name] = ModelMetrics(
            model_name=name,
            checkpoint=checkpoints.get(name),
            tokens_per_second=rate,
            rate_implausible=implausible,
            rate_note=note,
            time_to_first_token=_f(series.get("time_to_first_token_seconds")),
            requests_total=_i(series.get("requests_total")),
            input_tokens_total=_i(series.get("input_tokens_total")),
            output_tokens_total=_i(series.get("output_tokens_total")),
            last_output_tokens=last_out,
        )
    return out
