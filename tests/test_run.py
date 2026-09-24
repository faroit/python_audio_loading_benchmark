"""Tests for the run loop that turns loaders plus a corpus into results.

Uses small hand-built stub `Loader`s rather than the real nine adapters, so these
tests exercise the run loop's own logic (availability/format/seek gating,
verify-then-time ordering, error containment) without depending on which
optional libraries happen to be installed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import soundfile as sf

from pabench.corpus import CorpusSpec, Fmt, generate
from pabench.loaders import Loader
from pabench.run import Record, run, seek_offset, write_results

WAV_PCM16 = Fmt("wav", "PCM_16")


def _spec(duration_s: int, channels: int, fmt: Fmt = WAV_PCM16) -> CorpusSpec:
    return CorpusSpec(duration_s=duration_s, channels=channels, fmt=fmt)


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    generate(directory, specs=(_spec(10, 2), _spec(1, 1)))
    return directory


def _soundfile_full(path: Path) -> object:
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return data


def _soundfile_seek(path: Path, start_seconds: float, duration_seconds: float) -> object:
    info = sf.info(str(path))
    start_frame = round(start_seconds * info.samplerate)
    num_frames = round(duration_seconds * info.samplerate)
    data, _ = sf.read(
        str(path), start=start_frame, frames=num_frames, dtype="float32", always_2d=True
    )
    return data


def _correct_loader(name: str = "correct", seek: bool = True) -> Loader:
    return Loader(
        name=name,
        layout="frames_first",
        full=_soundfile_full,
        seek=_soundfile_seek if seek else None,
        version="1.0",
        available=True,
        error=None,
        formats=frozenset({"wav"}),
        notes=None,
    )


def _unavailable_loader() -> Loader:
    return Loader(
        name="broken_import",
        layout="frames_first",
        full=None,
        seek=None,
        version=None,
        available=False,
        error="ImportError: no such module",
        formats=frozenset({"wav"}),
        notes=None,
    )


def _downmixing_loader() -> Loader:
    def full(path: Path) -> object:
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
        return data.mean(axis=1, keepdims=True)  # collapses stereo to mono: wrong shape

    return Loader(
        name="downmixer",
        layout="frames_first",
        full=full,
        seek=None,
        version="0.1",
        available=True,
        error=None,
        formats=frozenset({"wav"}),
        notes=None,
    )


def _raising_loader() -> Loader:
    def full(path: Path) -> object:
        raise RuntimeError("kaboom")

    return Loader(
        name="raiser",
        layout="frames_first",
        full=full,
        seek=None,
        version="0.1",
        available=True,
        error=None,
        formats=frozenset({"wav"}),
        notes=None,
    )


def _wrong_container_loader() -> Loader:
    return Loader(
        name="flac_only",
        layout="frames_first",
        full=_soundfile_full,
        seek=_soundfile_seek,
        version="1.0",
        available=True,
        error=None,
        formats=frozenset({"flac"}),  # never claims wav
        notes=None,
    )


def _records_for(results: dict, library: str) -> list[dict]:
    return [r for r in results["records"] if r["library"] == library]


# ---- seek_offset ------------------------------------------------------------


def test_seek_offset_is_deterministic():
    spec = _spec(60, 2)
    assert seek_offset(spec) == seek_offset(spec)


def test_seek_offset_lands_in_middle_half():
    spec = _spec(60, 2)
    offset = seek_offset(spec)
    assert 15.0 <= offset <= 45.0  # [0.25, 0.75] * 60s
    assert offset + 1.0 <= spec.duration_s


def test_seek_offset_on_a_one_second_file_leaves_room_for_the_read():
    spec = _spec(1, 1)
    offset = seek_offset(spec)
    assert offset >= 0.0
    assert offset + 1.0 <= spec.duration_s


# ---- run(): correctness of the loop's own logic ------------------------------


def test_correct_loader_yields_ok_for_both_benches(corpus_dir):
    results = run(corpus_dir, [_correct_loader()], specs=(_spec(10, 2),), repeat=2)
    records = _records_for(results, "correct")
    assert {r["bench"] for r in records} == {"full", "seek"}
    for record in records:
        assert record["status"] == "ok"
        assert record["median_ms"] is not None
        assert record["min_ms"] is not None
        assert record["max_ms"] is not None
        assert record["realtime_factor"] is not None
        assert record["gate"] == "exact"
        assert record["reason"] is None


def test_unavailable_loader_is_recorded_with_its_error(corpus_dir):
    results = run(corpus_dir, [_unavailable_loader()], specs=(_spec(10, 2),), repeat=2)
    records = _records_for(results, "broken_import")
    assert len(records) == 2  # one per bench: not dropped
    for record in records:
        assert record["status"] == "unavailable"
        assert record["reason"] == "ImportError: no such module"
        assert record["median_ms"] is None


def test_downmixing_loader_is_incorrect_with_timings_withheld(corpus_dir):
    results = run(corpus_dir, [_downmixing_loader()], specs=(_spec(10, 2),), benches=("full",))
    (record,) = _records_for(results, "downmixer")
    assert record["status"] == "incorrect"
    assert record["gate"] == "exact"
    assert record["reason"] is not None
    assert record["median_ms"] is None
    assert record["min_ms"] is None
    assert record["max_ms"] is None


def test_raising_loader_is_error_and_run_continues(corpus_dir):
    results = run(
        corpus_dir,
        [_raising_loader(), _correct_loader()],
        specs=(_spec(10, 2),),
        benches=("full",),
    )
    (raiser_record,) = _records_for(results, "raiser")
    assert raiser_record["status"] == "error"
    assert "RuntimeError" in raiser_record["reason"]
    assert "kaboom" in raiser_record["reason"]
    assert raiser_record["median_ms"] is None

    # The run did not abort: the second loader still produced a real result.
    (correct_record,) = _records_for(results, "correct")
    assert correct_record["status"] == "ok"


def test_seekless_loader_is_unsupported_for_seek_but_ok_for_full(corpus_dir):
    results = run(corpus_dir, [_correct_loader(seek=False)], specs=(_spec(10, 2),), repeat=2)
    records = {r["bench"]: r for r in _records_for(results, "correct")}
    assert records["full"]["status"] == "ok"
    assert records["seek"]["status"] == "unsupported"
    assert records["seek"]["median_ms"] is None


def test_loader_not_claiming_the_container_is_unsupported(corpus_dir):
    results = run(corpus_dir, [_wrong_container_loader()], specs=(_spec(10, 2),), benches=("full",))
    (record,) = _records_for(results, "flac_only")
    assert record["status"] == "unsupported"
    assert record["median_ms"] is None


def test_empty_specs_do_not_raise(corpus_dir):
    results = run(corpus_dir, [_correct_loader()], specs=())
    assert results["records"] == []


def test_records_carry_the_corpus_axes(corpus_dir):
    spec = _spec(10, 2)
    results = run(corpus_dir, [_correct_loader()], specs=(spec,), benches=("full",))
    (record,) = _records_for(results, "correct")
    assert record["file"] == spec.filename
    assert record["duration_s"] == spec.duration_s
    assert record["channels"] == spec.channels
    assert record["format"] == spec.fmt.key


def test_progress_is_called_once_per_library_file_pair(corpus_dir):
    calls = []
    run(
        corpus_dir,
        [_correct_loader(), _unavailable_loader()],
        specs=(_spec(10, 2), _spec(1, 1)),
        progress=lambda library, filename: calls.append((library, filename)),
    )
    # 2 libraries * 2 files = 4 calls, regardless of how many benches each yields.
    assert len(calls) == 4


def test_reference_is_decoded_once_per_file_not_once_per_pair(corpus_dir, monkeypatch):
    import pabench.run as run_module

    call_count = {"n": 0}
    original = run_module._decode_reference

    def counting_decode_reference(path):
        call_count["n"] += 1
        return original(path)

    monkeypatch.setattr(run_module, "_decode_reference", counting_decode_reference)

    run(
        corpus_dir,
        [_correct_loader("a"), _correct_loader("b")],
        specs=(_spec(10, 2),),
        benches=("full", "seek"),
        repeat=1,
    )
    # 2 libraries * 1 file * 2 benches = 4 (library, file, bench) triples, but
    # the reference is decoded once, for the one file, and reused for all of them.
    assert call_count["n"] == 1


# ---- Record / write_results --------------------------------------------------


def test_record_is_a_frozen_dataclass_with_expected_fields():
    record = Record(
        library="x",
        file="f.wav",
        duration_s=1,
        channels=1,
        format="wav",
        bench="full",
        status="ok",
        median_ms=1.0,
        min_ms=1.0,
        max_ms=1.0,
        realtime_factor=1.0,
        gate="exact",
        reason=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.status = "error"  # type: ignore[misc]


def test_write_results_round_trips_through_json(tmp_path):
    results = {
        "platform": {"os": "test"},
        "records": [
            {
                "library": "x",
                "file": "f.wav",
                "duration_s": 1,
                "channels": 1,
                "format": "wav",
                "bench": "full",
                "status": "ok",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
                "realtime_factor": 10.0,
                "gate": "exact",
                "reason": None,
            }
        ],
    }
    out_path = tmp_path / "nested" / "results.json"
    write_results(results, out_path)

    assert out_path.exists()
    import json

    with out_path.open() as f:
        loaded = json.load(f)
    assert loaded == results


def test_platform_block_includes_expected_keys(corpus_dir):
    results = run(corpus_dir, [_correct_loader()], specs=(_spec(10, 2),), benches=("full",))
    platform = results["platform"]
    for key in ("os", "machine", "python_version", "torch_version", "libraries"):
        assert key in platform
    assert "correct" in platform["libraries"]
    assert platform["libraries"]["correct"]["available"] is True
