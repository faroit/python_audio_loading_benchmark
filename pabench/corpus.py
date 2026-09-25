"""Deterministic generation of the WAV/FLAC/MP3 corpus used by the benchmark.

Every file in the corpus is synthetic white noise, seeded per-spec so that
regenerating a single file reproduces exactly the bytes it had inside a full
sweep of the default corpus.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

#: FLAC metadata block type for SEEKTABLE.
_FLAC_SEEKTABLE_BLOCK = 3

SAMPLE_RATE = 44100
PEAK_AMPLITUDE = 0.5
DURATIONS_S: tuple[int, ...] = (1, 10, 60, 300)
CHANNEL_COUNTS: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class Fmt:
    container: str  # "wav" / "flac" / "mp3"
    subtype: str  # "PCM_16" / "MP3"

    @property
    def key(self) -> str:
        subtype_part = self.subtype.lower().replace("_", "")
        if subtype_part == self.container:
            return self.container
        return f"{self.container}_{subtype_part}"


FORMATS: tuple[Fmt, ...] = (
    Fmt("wav", "PCM_16"),
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


SEEKTABLE_INSTALL_HINT = """\
WARNING: metaflac not found, so the generated FLAC files will have no SEEKTABLE.

Libraries must then locate a seek position by binary-searching the frames instead of
jumping straight to it, which is not what a FLAC from the wild looks like -- the
reference encoder writes a seektable by default. The FLAC seek numbers from this corpus
describe the harder case, and the report records which case was measured.

metaflac ships with the `flac` package:

    Debian / Ubuntu   sudo apt install flac
    Fedora / RHEL     sudo dnf install flac
    Arch              sudo pacman -S flac
    macOS (Homebrew)  brew install flac
    conda             conda install -c conda-forge flac

Then regenerate the corpus (existing files are not rewritten):

    rm -rf <corpus-dir> && pabench gen
"""


def metaflac_available() -> bool:
    """Whether `metaflac` is on PATH, for writing FLAC SEEKTABLE blocks."""
    return shutil.which("metaflac") is not None


def has_seektable(path: Path) -> bool:
    """Whether a FLAC file carries a SEEKTABLE metadata block."""
    with open(path, "rb") as handle:
        if handle.read(4) != b"fLaC":
            return False
        while True:
            header = handle.read(4)
            if len(header) < 4:
                return False
            last, block_type = header[0] >> 7, header[0] & 0x7F
            if block_type == _FLAC_SEEKTABLE_BLOCK:
                return True
            if last:
                return False
            handle.seek(int.from_bytes(header[1:4], "big"), 1)


def _add_seektable(dest: Path) -> None:
    """Add one seekpoint per second, the layout the reference `flac` encoder writes.

    libsndfile writes no SEEKTABLE, so a corpus built with `soundfile` alone would
    make every library seek the hard way — binary search over frames — which is not
    what a FLAC from the wild looks like. `metaflac` is the only tool at hand that
    can add one; when it is absent the file is still valid and the run records that
    its FLAC files had no seektable.
    """
    subprocess.run(
        ["metaflac", "--add-seekpoint=1s", str(dest)],
        check=True,
        capture_output=True,
    )


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
    flac_seektable: bool = True,
) -> list[Path]:
    """Generate every spec's audio file under `corpus_dir`, returning the written paths.

    Args:
        corpus_dir: Directory to write into; created if absent.
        specs: Which files to generate.
        seed: Base seed; each spec reseeds from it, so one regenerated file matches
            the copy a full sweep would have produced.
        flac_seektable: Add a SEEKTABLE to FLAC files when `metaflac` is available.
            True matches a FLAC from the wild, which the reference encoder gives a
            seektable. False leaves the libsndfile default of none, so a library has
            to seek by binary search over frames -- set it to measure that difference.
    """
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    if flac_seektable and not metaflac_available():
        wants_flac = any(spec.fmt.container == "flac" for spec in specs)
        if wants_flac:
            print(SEEKTABLE_INSTALL_HINT, file=sys.stderr)

    paths: list[Path] = []
    for spec in specs:
        dest = corpus_dir / spec.filename
        data = _synthesize(spec, seed)

        if spec.fmt.container == "mp3":
            _write_mp3(spec, data, dest)
        else:
            sf.write(str(dest), data, spec.sample_rate, subtype=spec.fmt.subtype)
            if spec.fmt.container == "flac" and flac_seektable and metaflac_available():
                _add_seektable(dest)

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
