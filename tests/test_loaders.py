"""Tests for the eight library adapters.

No optional library is required to run this suite: a loader for a library that
isn't installed, or whose native components fail to load (torchcodec on a machine
missing `DYLD_FALLBACK_LIBRARY_PATH` is the motivating case, see
`docs/refactor-design.md`), simply probes as `available=False`, and the round-trip
and seek tests below are parametrized only over the loaders that came back
available in *this* environment.

The round-trip test exercises one representative `Fmt` per container a loader
claims in `formats` (`formats` is container-level: "wav"/"flac"/"mp3"), rather than
every wav bit-depth variant in `pabench.corpus.FORMATS`. Bit-depth-specific
correctness (e.g. audioread's inability to exceed 16-bit precision, or scipy_mmap's
inability to open 24-bit WAV) is a real property of these libraries, surfaced by
the full benchmark sweep in a later task rather than asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
import torch

from pabench.canonical import to_tensor
from pabench.corpus import DEFAULT_SPECS, CorpusSpec, Fmt, ffmpeg_available, generate
from pabench.loaders import PROBES, Loader, _smoke_test, available_loaders
from pabench.verify import compare

EXPECTED_LOADER_NAMES = {
    "soundfile",
    "librosa",
    "scipy",
    "scipy_mmap",
    "pydub",
    "audioread",
    "pedalboard",
    "torchcodec",
}

# One representative Fmt per container, used for the per-loader round-trip/seek
# tests (loader.formats only records containers, not wav bit depths).
REPRESENTATIVE_FMT: dict[str, Fmt] = {
    "wav": Fmt("wav", "PCM_16"),
    "flac": Fmt("flac", "PCM_16"),
    "mp3": Fmt("mp3", "MP3"),
}

SEEK_START_S = 0.2
SEEK_DURATION_S = 0.3
# Lossy mp3 seeking can be off by a sample or two; everything else must be exact.
MP3_SEEK_FRAME_TOLERANCE = 2


# ---- registry -----------------------------------------------------------------


def test_probes_has_exactly_the_eight_expected_names():
    assert set(PROBES.keys()) == EXPECTED_LOADER_NAMES


def test_available_loaders_with_no_names_probes_all_eight():
    loaders = available_loaders()
    assert {loader.name for loader in loaders} == EXPECTED_LOADER_NAMES
    assert all(isinstance(loader, Loader) for loader in loaders)


def test_available_loaders_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        available_loaders(["not_a_real_loader"])


def test_available_loaders_filters_to_requested_names():
    loaders = available_loaders(["soundfile"])
    assert [loader.name for loader in loaders] == ["soundfile"]


def test_soundfile_is_always_available_reference():
    # soundfile is a hard (non-optional) dependency and the correctness reference,
    # so it must always probe as available in this environment.
    (loader,) = available_loaders(["soundfile"])
    assert loader.available is True
    assert loader.error is None


# ---- a loader whose callable always raises probes as unavailable --------------


def test_broken_callable_downgrades_to_unavailable_with_error():
    def always_raises(path: Path) -> object:
        raise RuntimeError("boom")

    broken = Loader(
        name="broken",
        layout="frames_first",
        full=always_raises,
        seek=None,
        version="0.0",
        available=True,
        error=None,
        formats=frozenset({"wav"}),
        notes=None,
    )
    probed = _smoke_test(broken)
    assert probed.available is False
    assert probed.error is not None
    assert "boom" in probed.error


def test_smoke_test_is_a_noop_for_an_already_unavailable_loader():
    unavailable = Loader(
        name="broken",
        layout="frames_first",
        full=None,
        seek=None,
        version=None,
        available=False,
        error="ImportError: no such module",
        formats=frozenset({"wav"}),
        notes=None,
    )
    assert _smoke_test(unavailable) == unavailable


# ---- fixtures -------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("loader_corpus")
    specs = tuple(
        spec
        for spec in DEFAULT_SPECS
        if spec.duration_s == 1
        and spec.fmt in REPRESENTATIVE_FMT.values()
        and (spec.fmt.container != "mp3" or ffmpeg_available())
    )
    generate(directory, specs=specs)
    return directory


def _spec_for(fmt: Fmt, channels: int = 2) -> CorpusSpec:
    return next(
        spec
        for spec in DEFAULT_SPECS
        if spec.duration_s == 1 and spec.channels == channels and spec.fmt == fmt
    )


def _reference_tensor(path: Path) -> torch.Tensor:
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return to_tensor(data, "frames_first")


# All eight loaders, probed once at collection time so availability here drives
# which round-trip/seek cases get generated.
ALL_LOADERS: dict[str, Loader] = {loader.name: loader for loader in available_loaders()}
AVAILABLE_LOADER_NAMES = sorted(name for name, loader in ALL_LOADERS.items() if loader.available)


def _container_cases() -> list[tuple[str, str]]:
    cases = []
    for name in AVAILABLE_LOADER_NAMES:
        for container in sorted(ALL_LOADERS[name].formats):
            if container in REPRESENTATIVE_FMT:
                cases.append((name, container))
    return cases


CONTAINER_CASES = _container_cases()
SEEK_CASES = [
    (name, container) for name, container in CONTAINER_CASES if ALL_LOADERS[name].seek is not None
]


@pytest.mark.parametrize(
    "loader_name,container",
    CONTAINER_CASES,
    ids=[f"{name}-{container}" for name, container in CONTAINER_CASES],
)
def test_full_decode_matches_reference_under_gate(loader_name, container, corpus_dir):
    if container == "mp3" and not ffmpeg_available():
        pytest.skip("ffmpeg not available to build the mp3 fixture")

    loader = ALL_LOADERS[loader_name]
    fmt = REPRESENTATIVE_FMT[container]
    spec = _spec_for(fmt)
    path = corpus_dir / spec.filename

    reference = _reference_tensor(path)
    candidate = to_tensor(loader.full(path), loader.layout)

    result = compare(reference, candidate, fmt)
    assert result.ok, result.reason


@pytest.mark.parametrize(
    "loader_name,container",
    SEEK_CASES,
    ids=[f"{name}-{container}" for name, container in SEEK_CASES],
)
def test_seek_returns_expected_frame_count(loader_name, container, corpus_dir):
    if container == "mp3" and not ffmpeg_available():
        pytest.skip("ffmpeg not available to build the mp3 fixture")

    loader = ALL_LOADERS[loader_name]
    fmt = REPRESENTATIVE_FMT[container]
    spec = _spec_for(fmt)
    path = corpus_dir / spec.filename

    result = loader.seek(path, SEEK_START_S, SEEK_DURATION_S)
    tensor = to_tensor(result, loader.layout)
    actual_frames = tensor.shape[-1]

    expected_frames = round(SEEK_DURATION_S * spec.sample_rate)
    tolerance = MP3_SEEK_FRAME_TOLERANCE if container == "mp3" else 0

    assert abs(actual_frames - expected_frames) <= tolerance, (
        f"{loader_name}/{container}: expected ~{expected_frames} frames, got {actual_frames}"
    )
