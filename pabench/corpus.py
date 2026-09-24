"""Deterministic generation of the WAV/FLAC/MP3 corpus used by the benchmark.

Every file in the corpus is synthetic white noise, seeded per-spec so that
regenerating a single file reproduces exactly the bytes it had inside a full
sweep of the default corpus.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
PEAK_AMPLITUDE = 0.5
DURATIONS_S: tuple[int, ...] = (1, 10, 60, 300)
CHANNEL_COUNTS: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class Fmt:
    container: str  # "wav" / "flac" / "mp3"
    subtype: str  # "PCM_16" / "FLOAT" / "MP3"

    @property
    def key(self) -> str:
        subtype_part = self.subtype.lower().replace("_", "")
        if subtype_part == self.container:
            return self.container
        return f"{self.container}_{subtype_part}"


FORMATS: tuple[Fmt, ...] = (
    Fmt("wav", "PCM_16"),
    Fmt("wav", "FLOAT"),
    Fmt("flac", "PCM_16"),
    Fmt("mp3", "MP3"),
)


@dataclass(frozen=True)
class CorpusSpec:
    duration_s: int
    channels: int
    fmt: Fmt
    sample_rate: int = SAMPLE_RATE

    @property
    def filename(self) -> str:
        return f"{self.duration_s}s_{self.channels}ch_{self.fmt.key}.{self.fmt.container}"

    @property
    def frames(self) -> int:
        return self.duration_s * self.sample_rate


def _build_default_specs() -> tuple[CorpusSpec, ...]:
    specs = []
    for duration_s in DURATIONS_S:
        for channels in CHANNEL_COUNTS:
            for fmt in FORMATS:
                specs.append(CorpusSpec(duration_s=duration_s, channels=channels, fmt=fmt))
    return tuple(specs)


DEFAULT_SPECS: tuple[CorpusSpec, ...] = _build_default_specs()


def _spec_seed(seed: int, spec: CorpusSpec) -> int:
    """Derive a deterministic per-spec seed from the sweep seed and the spec's identity.

    Using a stable hash (rather than the builtin `hash`, which is randomised per
    process) means regenerating one file in isolation reproduces the same bytes
    it had as part of a full sweep.
    """
    key = f"{seed}:{spec.duration_s}:{spec.channels}:{spec.sample_rate}:{spec.fmt.key}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _synthesize(spec: CorpusSpec, seed: int) -> np.ndarray:
    """White noise in [-PEAK_AMPLITUDE, PEAK_AMPLITUDE], shape (frames, channels)."""
    rng = np.random.default_rng(_spec_seed(seed, spec))
    shape = (spec.frames, spec.channels)
    return rng.uniform(-PEAK_AMPLITUDE, PEAK_AMPLITUDE, size=shape).astype(np.float32)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _write_mp3(spec: CorpusSpec, data: np.ndarray, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "source.wav"
        sf.write(str(tmp_wav), data, spec.sample_rate, subtype="PCM_16")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(tmp_wav),
                "-b:a",
                "192k",
                str(dest),
            ],
            check=True,
        )


def generate(
    corpus_dir: Path,
    specs: tuple[CorpusSpec, ...] = DEFAULT_SPECS,
    seed: int = 0,
) -> list[Path]:
    """Generate every spec's audio file under `corpus_dir`, returning the written paths."""
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for spec in specs:
        dest = corpus_dir / spec.filename
        data = _synthesize(spec, seed)

        if spec.fmt.container == "mp3":
            _write_mp3(spec, data, dest)
        else:
            sf.write(str(dest), data, spec.sample_rate, subtype=spec.fmt.subtype)

        paths.append(dest)

    return paths


def corpus_files(
    corpus_dir: Path,
    specs: tuple[CorpusSpec, ...] = DEFAULT_SPECS,
) -> list[tuple[CorpusSpec, Path]]:
    """Resolve each spec to its path under `corpus_dir`, raising if any file is missing."""
    corpus_dir = Path(corpus_dir)
    result: list[tuple[CorpusSpec, Path]] = []
    missing: list[Path] = []

    for spec in specs:
        path = corpus_dir / spec.filename
        if not path.exists():
            missing.append(path)
        result.append((spec, path))

    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing corpus file(s): {names}. Run `pabench gen` to generate the corpus."
        )

    return result
