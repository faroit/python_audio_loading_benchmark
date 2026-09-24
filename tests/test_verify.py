import numpy as np
import pytest

from pabench.corpus import Fmt
from pabench.verify import VerifyResult, compare, compare_seek

WAV_PCM16 = Fmt("wav", "PCM_16")
WAV_PCM24 = Fmt("wav", "PCM_24")
WAV_FLOAT = Fmt("wav", "FLOAT")
FLAC_PCM16 = Fmt("flac", "PCM_16")
MP3 = Fmt("mp3", "MP3")

ATOL_PCM16 = 1.5 / 32768
ATOL_PCM24 = 1.5 / 8388608


def _stereo_noise(seed: int, frames: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=(2, frames)).astype(np.float32)


# ---- exact gate: wav/flac ----------------------------------------------------


def test_identical_passes_exact_gate():
    ref = _stereo_noise(0)
    result = compare(ref, ref.copy(), WAV_PCM16)
    assert result.ok is True
    assert result.reason is None
    assert result.gate == "exact"


def test_downmix_rejected():
    ref = _stereo_noise(1)
    mono = ref.mean(axis=0, keepdims=True)
    downmixed = np.repeat(mono, 2, axis=0)  # both channels replaced by the average
    result = compare(ref, downmixed, WAV_PCM16)
    assert result.ok is False
    assert result.gate == "exact"


def test_truncation_rejected():
    ref = _stereo_noise(2)
    truncated = ref[:, :-100]
    result = compare(ref, truncated, WAV_PCM16)
    assert result.ok is False


def test_channel_swap_rejected():
    ref = _stereo_noise(3)
    swapped = ref[::-1, :].copy()
    result = compare(ref, swapped, WAV_PCM16)
    assert result.ok is False


def test_pcm16_rescale_passes():
    # 32768/32767 is a legitimate int16-normalisation convention difference.
    ref = _stereo_noise(4)
    rescaled = (ref.astype(np.float64) * (32768.0 / 32767.0)).astype(np.float32)
    result = compare(ref, rescaled, WAV_PCM16)
    assert result.ok is True


def test_pcm16_2x_gain_fails():
    ref = _stereo_noise(5)
    gained = ref * 2.0
    result = compare(ref, gained, WAV_PCM16)
    assert result.ok is False


@pytest.mark.parametrize(
    "fmt,atol",
    [
        (WAV_PCM16, ATOL_PCM16),
        (WAV_PCM24, ATOL_PCM24),
    ],
)
def test_boundary_just_below_tolerance_passes(fmt, atol):
    ref = np.zeros((2, 500), dtype=np.float64)
    cand = ref + (atol * 0.9)
    result = compare(ref, cand, fmt)
    assert result.ok is True


@pytest.mark.parametrize(
    "fmt,atol",
    [
        (WAV_PCM16, ATOL_PCM16),
        (WAV_PCM24, ATOL_PCM24),
    ],
)
def test_boundary_just_above_tolerance_fails(fmt, atol):
    ref = np.zeros((2, 500), dtype=np.float64)
    cand = ref + (atol * 1.1)
    result = compare(ref, cand, fmt)
    assert result.ok is False


def test_float_gate_identical_passes():
    ref = _stereo_noise(6)
    result = compare(ref, ref.copy(), WAV_FLOAT)
    assert result.ok is True


def test_flac_uses_pcm16_style_tolerance():
    ref = _stereo_noise(7)
    rescaled = (ref.astype(np.float64) * (32768.0 / 32767.0)).astype(np.float32)
    result = compare(ref, rescaled, FLAC_PCM16)
    assert result.ok is True


def test_gate_is_exact_for_wav():
    ref = _stereo_noise(8)
    result = compare(ref, ref.copy(), WAV_PCM16)
    assert result.gate == "exact"


# ---- relaxed gate: mp3 --------------------------------------------------------


def test_mp3_delay_shift_passes():
    ref = _stereo_noise(9, frames=44100)
    # Simulate encoder delay: candidate is 1000 samples shorter (~22.7 ms, < 50 ms).
    candidate = ref[:, :-1000]
    result = compare(ref, candidate, MP3)
    assert result.ok is True
    assert result.gate == "relaxed"


def test_mp3_level_error_fails():
    ref = _stereo_noise(10, frames=44100)
    candidate = ref * 2.0  # +6 dB
    result = compare(ref, candidate, MP3)
    assert result.ok is False
    assert result.gate == "relaxed"


def test_mp3_length_difference_beyond_50ms_fails():
    ref = _stereo_noise(11, frames=44100)
    # 3000 samples at 44100 Hz is ~68 ms, beyond the 50 ms allowance.
    candidate = ref[:, :-3000]
    result = compare(ref, candidate, MP3)
    assert result.ok is False
    assert result.gate == "relaxed"


def test_mp3_zero_reference_rms_does_not_blow_up():
    ref = np.zeros((1, 44100), dtype=np.float32)
    candidate = np.zeros((1, 44100), dtype=np.float32)
    result = compare(ref, candidate, MP3)
    # Must not raise (e.g. a log(0) domain error); either verdict is acceptable
    # as long as it is a clean VerifyResult.
    assert isinstance(result, VerifyResult)
    assert result.gate == "relaxed"


def test_gate_is_relaxed_for_mp3():
    ref = _stereo_noise(12, frames=44100)
    result = compare(ref, ref.copy(), MP3)
    assert result.gate == "relaxed"


def test_mp3_length_gate_respects_explicit_sample_rate():
    # 500 samples is ~11 ms at 44100 Hz (passes) but ~62.5 ms at 8000 Hz (fails):
    # the same absolute sample difference trips the gate differently depending on
    # which sample rate the 50 ms allowance is computed against.
    ref = _stereo_noise(14, frames=8000)
    candidate = ref[:, :-500]

    at_corpus_rate = compare(ref, candidate, MP3, sample_rate=44100)
    at_low_rate = compare(ref, candidate, MP3, sample_rate=8000)

    assert at_corpus_rate.ok is True
    assert at_low_rate.ok is False


# ---- unknown format -----------------------------------------------------------


def test_unknown_format_raises_value_error_naming_it():
    ref = _stereo_noise(13)
    weird = Fmt("wav", "PCM_8")
    with pytest.raises(ValueError, match="PCM_8"):
        compare(ref, ref.copy(), weird)


def test_compare_seek_tolerates_a_one_frame_shift():
    """Seconds-based seek APIs round differently; one frame apart is a convention."""
    rng = np.random.default_rng(3)
    ref = rng.uniform(-0.5, 0.5, size=(1, 2000)).astype(np.float32)
    shifted = ref[:, 1:].copy()
    padded = np.concatenate([shifted, ref[:, :1]], axis=1)
    assert compare_seek(ref, padded, Fmt("wav", "FLOAT")).ok


def test_compare_seek_still_rejects_a_real_seek_error():
    """A genuinely wrong offset is orders of magnitude beyond one frame."""
    rng = np.random.default_rng(4)
    ref = rng.uniform(-0.5, 0.5, size=(1, 2000)).astype(np.float32)
    wrong = rng.uniform(-0.5, 0.5, size=(1, 2000)).astype(np.float32)
    assert not compare_seek(ref, wrong, Fmt("wav", "FLOAT")).ok


def test_compare_seek_exact_match_passes_unshifted():
    rng = np.random.default_rng(5)
    ref = rng.uniform(-0.5, 0.5, size=(1, 2000)).astype(np.float32)
    assert compare_seek(ref, ref.copy(), Fmt("wav", "FLOAT")).ok
