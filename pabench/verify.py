"""The per-format correctness gate.

Every library's decode is compared against a `soundfile`-decoded reference before its
timings are allowed to count. WAV and FLAC are graded sample-exact within a tolerance
set by the source bit depth; MP3 cannot be compared sample-exact (decoders disagree
about encoder delay), so it is graded by a relaxed length + level gate instead. See
`docs/refactor-design.md` for the rationale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from pabench.corpus import SAMPLE_RATE, Fmt

# rtol, atol per wav/flac subtype. Fixed by the design doc; never widen these.
_EXACT_TOLERANCES: dict[str, tuple[float, float]] = {
    "PCM_16": (0.0, 1.5 / 32768),
    "PCM_24": (0.0, 1.5 / 8388608),
    "FLOAT": (1e-5, 1e-7),
}

_MP3_MAX_LENGTH_DIFF_S = 0.05
_MP3_MAX_LEVEL_DIFF_DB = 0.5
_MIN_RMS = 1e-12  # guards the relaxed gate's log ratio against a zero/near-zero RMS


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str | None
    gate: str  # "exact" or "relaxed"


def _to_array(data: object) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def compare(
    reference: object,
    candidate: object,
    fmt: Fmt,
    sample_rate: int = SAMPLE_RATE,
) -> VerifyResult:
    """Compare a candidate decode against the reference decode for `fmt`.

    `sample_rate` is only used by the mp3 relaxed gate, to convert its 50 ms length
    allowance into samples; it defaults to the corpus's standard rate so existing call
    sites are unaffected when the corpus isn't 44100 Hz.

    Raises `ValueError` if `fmt`'s subtype is not one this gate knows how to grade
    (mp3 is always graded by the relaxed gate regardless of its subtype).
    """
    if fmt.container == "mp3":
        return _compare_relaxed(reference, candidate, sample_rate)
    return _compare_exact(reference, candidate, fmt)


def _compare_exact(reference: object, candidate: object, fmt: Fmt) -> VerifyResult:
    tolerance = _EXACT_TOLERANCES.get(fmt.subtype)
    if tolerance is None:
        raise ValueError(f"unknown format for the correctness gate: {fmt.subtype!r}")
    rtol, atol = tolerance

    ref = _to_array(reference)
    cand = _to_array(candidate)

    if ref.shape != cand.shape:
        return VerifyResult(
            ok=False,
            reason=f"shape mismatch: reference {ref.shape} vs candidate {cand.shape}",
            gate="exact",
        )

    if np.allclose(cand, ref, rtol=rtol, atol=atol):
        return VerifyResult(ok=True, reason=None, gate="exact")

    max_abs_diff = float(np.max(np.abs(cand.astype(np.float64) - ref.astype(np.float64))))
    return VerifyResult(
        ok=False,
        reason=(
            f"max abs diff {max_abs_diff:.3g} exceeds tolerance (rtol={rtol}, atol={atol:.3g})"
        ),
        gate="exact",
    )


def _compare_relaxed(reference: object, candidate: object, sample_rate: int) -> VerifyResult:
    ref = _to_array(reference).astype(np.float64)
    cand = _to_array(candidate).astype(np.float64)

    ref_frames = ref.shape[-1]
    cand_frames = cand.shape[-1]
    max_diff_samples = _MP3_MAX_LENGTH_DIFF_S * sample_rate
    frame_diff = abs(ref_frames - cand_frames)
    if frame_diff > max_diff_samples:
        return VerifyResult(
            ok=False,
            reason=(
                f"length difference {frame_diff} samples exceeds "
                f"{max_diff_samples:.0f} samples (50 ms at {sample_rate} Hz)"
            ),
            gate="relaxed",
        )

    common = min(ref_frames, cand_frames)
    ref_trimmed = ref[..., :common]
    cand_trimmed = cand[..., :common]

    ref_rms = float(np.sqrt(np.mean(np.square(ref_trimmed))))
    cand_rms = float(np.sqrt(np.mean(np.square(cand_trimmed))))

    if ref_rms < _MIN_RMS:
        return VerifyResult(
            ok=False,
            reason=f"reference RMS {ref_rms:.3g} is too small to compute a level ratio",
            gate="relaxed",
        )

    if cand_rms < _MIN_RMS:
        level_diff_db = -math.inf
    else:
        level_diff_db = 20.0 * math.log10(cand_rms / ref_rms)

    if abs(level_diff_db) > _MP3_MAX_LEVEL_DIFF_DB:
        return VerifyResult(
            ok=False,
            reason=(f"level difference {level_diff_db:.2f} dB exceeds {_MP3_MAX_LEVEL_DIFF_DB} dB"),
            gate="relaxed",
        )

    return VerifyResult(ok=True, reason=None, gate="relaxed")
