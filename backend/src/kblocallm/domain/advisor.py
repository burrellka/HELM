"""Model Advisor, Layer 1: local facts. Spec 14c Part 1 §1.1.

**This layer never hallucinates, because no model runs.** It is deterministic
arithmetic and string parsing over the catalog HELM already reads: decode the
quant, count the parameters, compute the memory footprint, check the fit
against the real pool. Every output here is a fact or a clearly-labeled
estimate, never a recalled opinion. Layers 2 (HuggingFace metadata) and 3 (LLM
synthesis) sit on top; this is the floor they cannot contradict.

The quant decoder is the direct answer to "I don't get q4 vs q8": parse the
quant string, explain it in plain terms, and compute the concrete size impact
for THIS model. The fit calc turns "will it run" into a number, not an opinion.

Pure functions only. Table-tested. If it needs the network or a model, it
belongs in the Layer 2 or Layer 3 module, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Bytes per weight by quant, from 14c §1.1 (verified against llama.cpp's own
# figures). These drive the size-impact math. Keyed by a normalized quant
# string; the decoder falls back to the leading digit for families not listed.
_BYTES_PER_WEIGHT: dict[str, float] = {
    "f32": 4.0, "fp32": 4.0,
    "f16": 2.0, "fp16": 2.0, "bf16": 2.0,
    "q8_0": 1.0, "q8": 1.0,
    "q6_k": 0.82, "q6": 0.82,
    "q5_k_m": 0.71, "q5_k_s": 0.67, "q5_0": 0.69, "q5_1": 0.75, "q5": 0.70,
    "q4_k_m": 0.56, "q4_k_s": 0.53, "q4_k_xl": 0.62, "q4_0": 0.56, "q4_1": 0.61, "q4": 0.56,
    "q3_k_m": 0.43, "q3_k_s": 0.40, "q3_k_l": 0.46, "q3": 0.43,
    "q2_k": 0.33, "q2": 0.33,
    # NPU / FastFlowLM kernel quant. Roughly q4-class, but it is a compiled
    # kernel, not a GGUF quant, so it is decoded and labeled distinctly.
    "q4nx": 0.56,
}

#: Rough per-family fallback when the exact suffix is not in the table.
_FALLBACK_BY_DIGIT: dict[str, float] = {
    "8": 1.0, "6": 0.82, "5": 0.70, "4": 0.56, "3": 0.43, "2": 0.33,
}

_QUANT_RE = re.compile(
    r"\b(?:UD-)?(F(?:P)?(?:16|32)|BF16|Q\d(?:_K(?:_[MSL]|_XL)?|_[01])?|IQ\d[A-Z_]*|Q4NX)\b",
    re.IGNORECASE,
)

# Params: "26B", "26B-A4B" (MoE total+active), "0.6B", "300m", "1.2b".
_PARAM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*([BM])\b", re.IGNORECASE)
_ACTIVE_RE = re.compile(r"\bA(\d+(?:\.\d+)?)\s*B\b", re.IGNORECASE)


class QuantClass(str, Enum):
    NEAR_LOSSLESS = "near_lossless"   # q8, fp16
    HIGH = "high"                     # q6, q5
    BALANCED = "balanced"             # q4 (the common sweet spot)
    LOSSY = "lossy"                   # q3, q2
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class QuantInfo:
    #: The quant token as found, e.g. "Q4_K_M".
    label: str
    bytes_per_weight: float | None
    cls: QuantClass
    #: Whether it is a k-quant family (improved), and the M/S/XL size variant.
    k_quant: bool = False
    #: True for the NPU compiled kernel quant, which is not a GGUF quant.
    npu_kernel: bool = False
    explanation: str = ""
    tradeoff: str = ""


def decode_quant(text: str | None) -> QuantInfo | None:
    """Find and explain the quant in a model id or checkpoint string.

    Returns None when there is no quant token at all (e.g. a bare FLM name like
    `qwen3.5:2b`). None means "not stated", never a guessed default.
    """
    if not text:
        return None
    match = _QUANT_RE.search(text)
    if not match:
        return None

    label = match.group(1).upper()
    key = label.lower().replace("ud-", "")
    bpw = _BYTES_PER_WEIGHT.get(key)
    if bpw is None:
        digit = re.search(r"q(\d)", key)
        bpw = _FALLBACK_BY_DIGIT.get(digit.group(1)) if digit else None

    npu = key == "q4nx"
    k_quant = "_k" in key
    cls = _classify_quant(key, bpw)
    return QuantInfo(
        label=label, bytes_per_weight=bpw, cls=cls, k_quant=k_quant, npu_kernel=npu,
        explanation=_quant_explanation(label, cls, k_quant, npu),
        tradeoff=_quant_tradeoff(cls),
    )


def _classify_quant(key: str, bpw: float | None) -> QuantClass:
    if key.startswith(("f16", "fp16", "bf16", "f32", "fp32", "q8")):
        return QuantClass.NEAR_LOSSLESS
    if key.startswith(("q6", "q5")):
        return QuantClass.HIGH
    if key.startswith("q4"):
        return QuantClass.BALANCED
    if key.startswith(("q3", "q2")):
        return QuantClass.LOSSY
    return QuantClass.UNKNOWN if bpw is None else QuantClass.BALANCED


def _quant_explanation(label: str, cls: QuantClass, k_quant: bool, npu: bool) -> str:
    if npu:
        return (
            f"{label} is FastFlowLM's compiled NPU kernel quant, roughly q4-class in size. It is "
            f"not a GGUF quant, so it does not swap quant levels the way the GPU models do."
        )
    base = {
        QuantClass.NEAR_LOSSLESS: f"{label} is near-lossless and the largest. Quality is essentially the full model.",
        QuantClass.HIGH: f"{label} keeps most of the quality of the full model at noticeably less memory.",
        QuantClass.BALANCED: f"{label} is the common sweet spot: about half the memory of q8 with a small quality cost and more speed.",
        QuantClass.LOSSY: f"{label} trades visible quality for memory. Use it only when memory is genuinely tight.",
        QuantClass.UNKNOWN: f"{label} is a quantization HELM does not have a size figure for.",
    }[cls]
    if k_quant:
        base += " The _K_ family is an improved k-quant (M is medium, S small, XL extra)."
    return base


def _quant_tradeoff(cls: QuantClass) -> str:
    if cls is QuantClass.BALANCED:
        return (
            "On a 120GB pool, memory is rarely the constraint, so a higher quant (q6 or q8) is "
            "usually affordable if you want more quality. You are not forced down to q4 here."
        )
    if cls is QuantClass.LOSSY:
        return "On a 120GB pool you almost never need to go this low. A higher quant will fit."
    if cls is QuantClass.NEAR_LOSSLESS:
        return "This is the quality ceiling. Only drop below it if you want more speed or headroom."
    return ""


@dataclass(frozen=True)
class ParamInfo:
    #: Total parameters in billions.
    total_b: float
    #: Active parameters per token, for MoE models (the "A3B" in "35B-A3B").
    #: Equal to total for dense models.
    active_b: float
    is_moe: bool = False


def parse_params(name: str | None) -> ParamInfo | None:
    """Extract parameter counts from a model name. Handles MoE total+active.

    "Gemma-4-26B-A4B" -> total 26.0, active 4.0, MoE. "8B" -> 8.0 dense.
    "300m" -> 0.3. None when no count is stated (never a guess).
    """
    if not name:
        return None
    match = _PARAM_RE.search(name)
    if not match:
        return None
    value = float(match.group(1))
    total = value / 1000.0 if match.group(2).upper() == "M" else value

    active_match = _ACTIVE_RE.search(name)
    if active_match:
        return ParamInfo(total_b=total, active_b=float(active_match.group(1)), is_moe=True)
    return ParamInfo(total_b=total, active_b=total, is_moe=False)


class FitVerdict(str, Enum):
    FITS_RESIDENT = "fits_resident"       # room alongside what is already loaded
    FITS_IF_EVICT = "fits_if_evict"       # loads only if something is evicted
    TOO_BIG = "too_big"                   # exceeds the pool even alone
    UNKNOWN = "unknown"                   # no footprint figure available


@dataclass(frozen=True)
class FitEstimate:
    verdict: FitVerdict
    #: GB the model itself needs. A FACT when from disk size or a measured
    #: delta; an estimate otherwise. `measured` says which.
    footprint_gb: float | None
    #: True when footprint_gb came from a real resident measurement (Phase 4
    #: memory attribution), not an estimate.
    measured: bool
    pool_gb: float | None
    #: GB currently in use, so the operator sees the headroom.
    used_gb: float | None = None
    detail: str = ""


def compute_fit(
    *,
    size_gb: float | None,
    pool_gb: float | None,
    used_gb: float | None = None,
    measured_gb: float | None = None,
) -> FitEstimate:
    """Will this model fit, as a number. 14c §1.1.

    Prefers a measured resident delta (ground truth) over the on-disk size
    (a good proxy, since weights dominate). KV cache is not added here as a
    fabricated number: the on-disk size already reflects the weights, and a
    precise KV figure needs per-model architecture HELM does not have, so the
    honest output is "weights fit, context adds on top" rather than a made-up
    total. When nothing is known (FLM models carry no size), it says so.
    """
    footprint = measured_gb if measured_gb is not None else size_gb
    measured = measured_gb is not None

    if footprint is None or pool_gb is None:
        return FitEstimate(
            verdict=FitVerdict.UNKNOWN, footprint_gb=footprint, measured=measured,
            pool_gb=pool_gb,
            detail=(
                "No footprint figure available (NPU models carry no catalog size, and this model "
                "has no measured resident delta yet). Load it once and HELM can measure it."
            ),
        )

    free = pool_gb - (used_gb or 0.0)
    source = "measured resident use" if measured else "on-disk weights size"
    if footprint <= free:
        verdict = FitVerdict.FITS_RESIDENT
        detail = (
            f"Needs about {footprint:.1f} GB ({source}); roughly {free:.1f} GB is free of the "
            f"{pool_gb:.0f} GB pool, so it fits alongside what is already loaded. Context adds on "
            f"top of the weights."
        )
    elif footprint <= pool_gb:
        verdict = FitVerdict.FITS_IF_EVICT
        detail = (
            f"Needs about {footprint:.1f} GB ({source}), more than the {free:.1f} GB free right "
            f"now, so it loads only if the router evicts something first. It fits the {pool_gb:.0f} "
            f"GB pool on its own."
        )
    else:
        verdict = FitVerdict.TOO_BIG
        detail = (
            f"Needs about {footprint:.1f} GB ({source}), more than the whole {pool_gb:.0f} GB pool. "
            f"It will not load here."
        )
    return FitEstimate(
        verdict=verdict, footprint_gb=footprint, measured=measured,
        pool_gb=pool_gb, used_gb=used_gb, detail=detail,
    )


def quant_size_impact(size_gb: float, quant: QuantInfo, target: str = "q8_0") -> str | None:
    """Concrete size impact: this model at its quant versus at `target`.

    Turns the abstract "q4 is smaller" into "this model is 17 GB at q4_K_M and
    would be about 30 GB at q8". Returns None when either quant has no known
    bytes-per-weight (no fabricated comparison).
    """
    if quant is None:
        return None
    target_bpw = _BYTES_PER_WEIGHT.get(target.lower())
    if not quant.bytes_per_weight or not target_bpw or quant.bytes_per_weight == 0:
        return None
    scaled = size_gb * (target_bpw / quant.bytes_per_weight)
    return (
        f"At {quant.label} this model is about {size_gb:.1f} GB. At {target.upper()} it would be "
        f"roughly {scaled:.1f} GB."
    )
