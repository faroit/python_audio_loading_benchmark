"""Adapters for the eleven audio-loading libraries this benchmark measures.

Each probe function in `PROBES` imports one library, wires up its full-file decode
and (where the library supports it) seek decode callables, and smoke-tests them by
actually decoding a small temporary WAV file. A probe that imports successfully but
fails to decode is downgraded to `available=False` with the decode error recorded,
rather than reported as working on the strength of a successful import alone --
`torchcodec` is the motivating case: it imports cleanly and only fails once it tries
to load its native FFmpeg bindings.

Every `full`/`seek` callable returns whatever its library natively hands back --
numpy array, torch tensor, or anything `pabench.canonical.to_tensor` accepts.
Converting that to the canonical float32 (channels, frames) tensor, and doing so
inside the timed region, is the caller's job (see `docs/refactor-design.md`), not
this module's.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

import numpy as np
import soundfile as sf

from pabench.canonical import Layout

_SMOKE_SAMPLE_RATE = 8000
_SMOKE_FRAMES = 800  # 100 ms
_SMOKE_CHANNELS = 2

_WAV_FLAC_MP3 = frozenset({"wav", "flac", "mp3"})
_WAV_ONLY = frozenset({"wav"})


@dataclass(frozen=True)
class Loader:
    name: str
    layout: Layout
    full: Callable[[Path], object] | None
    seek: Callable[[Path, float, float], object] | None
    version: str | None
    available: bool
    error: str | None
    formats: frozenset[str]
    notes: str | None


def _package_version(module: object, dist_name: str) -> str | None:
    """`module.__version__` if present, else the installed distribution's version."""
    attr_version = getattr(module, "__version__", None)
    if attr_version is not None:
        return str(attr_version)
    try:
        return _dist_version(dist_name)
    except PackageNotFoundError:
        return None


def _make_smoke_wav(directory: Path) -> Path:
    path = directory / "smoke.wav"
    rng = np.random.default_rng(0)
    data = rng.uniform(-0.5, 0.5, size=(_SMOKE_FRAMES, _SMOKE_CHANNELS)).astype(np.float32)
    sf.write(str(path), data, _SMOKE_SAMPLE_RATE, subtype="PCM_16")
    return path


def _smoke_test(loader: Loader) -> Loader:
    """Decode a tiny temp WAV with `loader.full`; downgrade to unavailable on failure.

    Import success does not imply decode success, so every probed loader is
    exercised against a real file before it is trusted as available.
    """
    if not loader.available or loader.full is None:
        return loader
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_smoke_wav(Path(tmp))
            loader.full(path)
    except Exception as exc:  # any decode failure disqualifies the loader (see module docstring)
        return replace(loader, available=False, error=f"{type(exc).__name__}: {exc}")
    return loader


def _unavailable(
    name: str,
    layout: Layout,
    formats: frozenset[str],
    notes: str | None,
    exc: Exception,
) -> Loader:
    return Loader(
        name=name,
        layout=layout,
        full=None,
        seek=None,
        version=None,
        available=False,
        error=f"{type(exc).__name__}: {exc}",
        formats=formats,
        notes=notes,
    )


def _scale_to_float32(raw: np.ndarray) -> np.ndarray:
    """Scale an integer PCM array to float32 in [-1, 1]; pass float arrays through.

    Integer full scale is derived from the dtype's bit width (e.g. int16 -> 32768,
    int32 -> 2147483648), which matches how `scipy.io.wavfile` represents 24-bit PCM
    (left-justified into a 32-bit container) as well as plain 16-bit PCM.
    """
    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    if raw.dtype.kind == "f":
        return raw.astype(np.float32)
    if raw.dtype.kind != "i":
        raise ValueError(f"unsupported integer PCM dtype: {raw.dtype}")
    full_scale = 2.0 ** (8 * raw.dtype.itemsize - 1)
    return raw.astype(np.float32) / full_scale


# ---- probes -----------------------------------------------------------------------


def _probe_soundfile() -> Loader:
    name = "soundfile"
    layout: Layout = "frames_first"
    formats = _WAV_FLAC_MP3
    notes = "the correctness reference every other loader is checked against"
    try:
        import soundfile as libsndfile
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        data, _ = libsndfile.read(str(path), dtype="float32", always_2d=True)
        return data

    def seek(path: Path, start_seconds: float, duration_seconds: float) -> object:
        info = libsndfile.info(str(path))
        start_frame = round(start_seconds * info.samplerate)
        num_frames = round(duration_seconds * info.samplerate)
        data, _ = libsndfile.read(
            str(path),
            start=start_frame,
            frames=num_frames,
            dtype="float32",
            always_2d=True,
        )
        return data

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=seek,
        version=_package_version(libsndfile, "soundfile"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


def _probe_librosa() -> Loader:
    name = "librosa"
    layout: Layout = "channels_first"
    formats = _WAV_FLAC_MP3
    notes = None
    try:
        import librosa
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        data, _ = librosa.load(str(path), sr=None, mono=False)
        return data

    def seek(path: Path, start_seconds: float, duration_seconds: float) -> object:
        data, _ = librosa.load(
            str(path),
            sr=None,
            mono=False,
            offset=start_seconds,
            duration=duration_seconds,
        )
        return data

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=seek,
        version=_package_version(librosa, "librosa"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


def _probe_scipy() -> Loader:
    name = "scipy"
    layout: Layout = "frames_first"
    formats = _WAV_ONLY
    notes = "WAV only; scipy.io.wavfile has no seek API"
    try:
        import scipy
        from scipy.io import wavfile
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        _, raw = wavfile.read(str(path))
        return _scale_to_float32(raw)

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=None,
        version=_package_version(scipy, "scipy"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


def _probe_scipy_mmap() -> Loader:
    name = "scipy_mmap"
    layout: Layout = "frames_first"
    formats = _WAV_ONLY
    notes = (
        "WAV only, memory-mapped; scipy's mmap mode cannot open 24-bit "
        "('3-byte container') WAV, which surfaces as a decode error for that subtype"
    )
    try:
        import scipy
        from scipy.io import wavfile
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        _, raw = wavfile.read(str(path), mmap=True)
        return _scale_to_float32(raw)

    def seek(path: Path, start_seconds: float, duration_seconds: float) -> object:
        rate, raw = wavfile.read(str(path), mmap=True)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        start_frame = round(start_seconds * rate)
        num_frames = round(duration_seconds * rate)
        sliced = raw[start_frame : start_frame + num_frames]
        return _scale_to_float32(sliced)

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=seek,
        version=_package_version(scipy, "scipy"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


def _probe_audioread() -> Loader:
    name = "audioread"
    layout: Layout = "frames_first"
    formats = _WAV_FLAC_MP3
    notes = (
        "full-file only, no seek API; audioread always yields 16-bit PCM buffers, "
        "so it cannot pass the exact gate for 24-bit or float32 WAV sources"
    )
    try:
        import audioread
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        with audioread.audio_open(str(path)) as f:
            channels = f.channels
            chunks = [np.frombuffer(buf, dtype="<i2") for buf in f]
        signal = np.concatenate(chunks) if chunks else np.zeros(0, dtype="<i2")
        return signal.astype(np.float32).reshape(-1, channels) / 32768.0

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=None,
        version=_package_version(audioread, "audioread"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


def _probe_pedalboard() -> Loader:
    name = "pedalboard"
    layout: Layout = "channels_first"
    formats = _WAV_FLAC_MP3
    notes = None
    try:
        import pedalboard
        from pedalboard.io import AudioFile
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        with AudioFile(str(path)) as f:
            # (channels, samples); never index [0], which would silently keep
            # only the first channel (see docs/refactor-design.md).
            return f.read(f.frames)

    def seek(path: Path, start_seconds: float, duration_seconds: float) -> object:
        with AudioFile(str(path)) as f:
            start_frame = round(start_seconds * f.samplerate)
            num_frames = round(duration_seconds * f.samplerate)
            f.seek(start_frame)
            return f.read(num_frames)

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=seek,
        version=_package_version(pedalboard, "pedalboard"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


def _probe_torchcodec() -> Loader:
    name = "torchcodec"
    layout: Layout = "channels_first"
    formats = _WAV_FLAC_MP3
    notes = (
        "imports cleanly even when its native FFmpeg bindings can't load; only the "
        "decode smoke test below catches that (see docs/refactor-design.md)"
    )
    try:
        import torchcodec
        from torchcodec.decoders import AudioDecoder
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        return AudioDecoder(str(path)).get_all_samples().data

    def seek(path: Path, start_seconds: float, duration_seconds: float) -> object:
        decoder = AudioDecoder(str(path))
        stop_seconds = start_seconds + duration_seconds
        result = decoder.get_samples_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )
        return result.data

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=seek,
        version=_package_version(torchcodec, "torchcodec"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


def _probe_audiolab() -> Loader:
    name = "audiolab"
    layout: Layout = "channels_first"
    formats = _WAV_FLAC_MP3
    notes = None
    try:
        import audiolab
        from audiolab import load_audio
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        # `load_audio` is the documented one-shot entry point; the package's
        # `Reader.pull()` is a separate streaming API and is not used here (see
        # docs/refactor-design.md).
        data, _ = load_audio(path, dtype=np.float32)
        return data

    def seek(path: Path, start_seconds: float, duration_seconds: float) -> object:
        data, _ = load_audio(
            path, offset=start_seconds, duration=duration_seconds, dtype=np.float32
        )
        return data

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=seek,
        version=_package_version(audiolab, "audiolab"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


def _probe_sphn() -> Loader:
    name = "sphn"
    layout: Layout = "channels_first"
    formats = _WAV_FLAC_MP3
    notes = (
        "MP3 decode returns a different frame count than soundfile's reference "
        "(decoder-delay disagreement); graded by the relaxed MP3 gate"
    )
    try:
        import sphn
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        data, _ = sphn.read(str(path))
        return data

    def seek(path: Path, start_seconds: float, duration_seconds: float) -> object:
        data, _ = sphn.read(str(path), start_sec=start_seconds, duration_sec=duration_seconds)
        return data

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=seek,
        version=_package_version(sphn, "sphn"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


def _probe_audiosample() -> Loader:
    name = "audiosample"
    layout: Layout = "channels_first"
    formats = _WAV_ONLY
    # Its own parser handles integer-PCM WAV; everything else (float WAV, flac, mp3)
    # falls through to PyAV, which raises AttributeError against PyAV 18. Declared
    # WAV-only at container level, so float WAV still reaches the loader and is
    # reported as an error with that reason rather than hidden.
    notes = (
        "integer-PCM WAV only: float WAV, FLAC and MP3 go through its PyAV path, "
        "which is incompatible with PyAV 18 (Flags.FAST_SEEK)"
    )
    try:
        import audiosample
        from audiosample import AudioSample
    except Exception as exc:
        return _unavailable(name, layout, formats, notes, exc)

    def full(path: Path) -> object:
        # Kept as numpy (not `.as_tensor()`) so it goes through the same
        # `to_tensor` conversion as every other loader (see
        # docs/refactor-design.md).
        return AudioSample(str(path)).as_numpy()

    def seek(path: Path, start_seconds: float, duration_seconds: float) -> object:
        sample = AudioSample(str(path))
        excerpt = sample[start_seconds : start_seconds + duration_seconds]
        return excerpt.as_numpy()

    loader = Loader(
        name=name,
        layout=layout,
        full=full,
        seek=seek,
        version=_package_version(audiosample, "audiosample"),
        available=True,
        error=None,
        formats=formats,
        notes=notes,
    )
    return _smoke_test(loader)


PROBES: dict[str, Callable[[], Loader]] = {
    "soundfile": _probe_soundfile,
    "librosa": _probe_librosa,
    "scipy": _probe_scipy,
    "scipy_mmap": _probe_scipy_mmap,
    "audioread": _probe_audioread,
    "pedalboard": _probe_pedalboard,
    "torchcodec": _probe_torchcodec,
    "audiolab": _probe_audiolab,
    "audiosample": _probe_audiosample,
    "sphn": _probe_sphn,
}


def available_loaders(names: list[str] | None = None) -> list[Loader]:
    """Probe and return loaders, in the order given (or `PROBES`' order if omitted).

    Every requested loader is probed and returned, whether or not it ends up
    `available` -- callers that only want working loaders should filter on
    `.available` themselves. Raises `KeyError` naming any name that isn't one of
    the eleven registered probes.
    """
    selected = list(PROBES.keys()) if names is None else list(names)
    unknown = [n for n in selected if n not in PROBES]
    if unknown:
        raise KeyError(f"unknown loader name(s): {unknown!r}; known: {sorted(PROBES)}")
    return [PROBES[n]() for n in selected]
