"""Tests for the run loop that turns loaders plus a corpus into results.

Uses small hand-built stub `Loader`s rather than the real adapters, so these
tests exercise the run loop's own logic (availability/format/seek gating,
verify-then-time ordering, error containment) without depending on which
optional libraries happen to be installed.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import pytest
import soundfile as sf

from pabench.corpus import CorpusSpec, Fmt, generate
from pabench.loaders import Loader
from pabench.run import SEEK_DURATIONS, Record, platform_block, run, seek_offset, write_results

WAV_PCM16 = Fmt("wav", "PCM_16")


def _spec(duration_s: int, channels: int, fmt: Fmt = WAV_PCM16) -> CorpusSpec:
    return CorpusSpec(duration_s=duration_s, channels=channels, fmt=fmt)


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    generate(directory, specs=(_spec(10, 2), _spec(1, 1)))
    return directory


@pytest.fixture()
def five_second_corpus_dir(tmp_path: Path) -> Path:
    """A 5 s file: long enough for the 1s/3s seek chunks, too short for 10s/30s."""
    directory = tmp_path / "corpus5"
    generate(directory, specs=(_spec(5, 2),))
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


def _soundfile_from_bytes(raw: bytes) -> object:
    data, _ = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return data


def _correct_loader(name: str = "correct", seek: bool = True, from_bytes: bool = False) -> Loader:
    return Loader(
        name=name,
        layout="frames_first",
        full=_soundfile_full,
        seek=_soundfile_seek if seek else None,
        from_bytes=_soundfile_from_bytes if from_bytes else None,
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
    assert seek_offset(spec, 1.0) == seek_offset(spec, 1.0)


def test_seek_offset_lands_in_middle_half():
    spec = _spec(60, 2)
    offset = seek_offset(spec, 1.0)
    assert 15.0 <= offset <= 45.0  # [0.25, 0.75] * 60s
    assert offset + 1.0 <= spec.duration_s


def test_seek_offset_on_a_one_second_file_leaves_room_for_the_read():
    spec = _spec(1, 1)
    offset = seek_offset(spec, 1.0)
    assert offset >= 0.0
    assert offset + 1.0 <= spec.duration_s


def test_seek_offset_respects_the_chunk_length_argument():
    spec = _spec(60, 2)
    for chunk in SEEK_DURATIONS:
        offset = seek_offset(spec, chunk)
        assert offset >= 0.0
        # Reading `chunk` seconds from `offset` must never run past the end of the file.
        assert offset + chunk <= spec.duration_s + 1e-9


def test_seek_offset_never_reads_past_the_end_for_a_short_file():
    spec = _spec(10, 2)
    for chunk in (1.0, 3.0, 10.0):
        offset = seek_offset(spec, chunk)
        assert offset >= 0.0
        assert offset + chunk <= spec.duration_s + 1e-9


def test_seek_offset_when_chunk_equals_the_whole_file_returns_zero():
    """Degenerate case: no room to bias towards the middle half at all."""
    spec = _spec(10, 2)
    offset = seek_offset(spec, 10.0)
    assert offset == 0.0
    assert offset + 10.0 <= spec.duration_s + 1e-9


def test_seek_offset_quantises_to_a_sample_boundary():
    spec = _spec(60, 2)
    for chunk in SEEK_DURATIONS:
        offset = seek_offset(spec, chunk)
        frame = offset * spec.sample_rate
        assert abs(frame - round(frame)) < 1e-6


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
    # 1 full + 3 seek chunk lengths (1/3/10s all fit inside a 10s file): not dropped.
    assert len(records) == 4
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


# ---- seek chunk-duration axis --------------------------------------------------


def test_seek_bench_skips_chunk_durations_longer_than_the_file(five_second_corpus_dir):
    results = run(
        five_second_corpus_dir,
        [_correct_loader()],
        specs=(_spec(5, 2),),
        benches=("seek",),
        repeat=1,
    )
    records = _records_for(results, "correct")
    # Only 1s/3s fit inside a 5s file; 10s/30s must not be emitted at all (not
    # "unsupported" -- simply absent).
    assert sorted(r["seek_seconds"] for r in records) == [1.0, 3.0]
    assert all(r["status"] == "ok" for r in records)


def test_seek_bench_emits_one_record_per_applicable_chunk_duration(corpus_dir):
    results = run(
        corpus_dir,
        [_correct_loader()],
        specs=(_spec(10, 2),),
        benches=("seek",),
        repeat=1,
    )
    records = _records_for(results, "correct")
    assert sorted(r["seek_seconds"] for r in records) == [1.0, 3.0, 10.0]
    assert all(r["status"] == "ok" for r in records)


def test_seek_durations_argument_filters_which_chunks_are_used(corpus_dir):
    results = run(
        corpus_dir,
        [_correct_loader()],
        specs=(_spec(10, 2),),
        benches=("seek",),
        seek_durations=(3.0,),
        repeat=1,
    )
    records = _records_for(results, "correct")
    assert [r["seek_seconds"] for r in records] == [3.0]


def test_full_and_bytes_records_have_seek_seconds_none(corpus_dir):
    results = run(
        corpus_dir,
        [_correct_loader(from_bytes=True)],
        specs=(_spec(10, 2),),
        benches=("full", "bytes"),
        repeat=1,
    )
    records = _records_for(results, "correct")
    assert records
    assert all(r["seek_seconds"] is None for r in records)


def test_loader_not_claiming_the_container_is_unsupported(corpus_dir):
    results = run(corpus_dir, [_wrong_container_loader()], specs=(_spec(10, 2),), benches=("full",))
    (record,) = _records_for(results, "flac_only")
    assert record["status"] == "unsupported"
    assert record["median_ms"] is None


# ---- bytes bench --------------------------------------------------------------


def test_loader_without_from_bytes_is_unsupported_for_bytes_but_unaffected_for_others(corpus_dir):
    results = run(
        corpus_dir,
        [_correct_loader()],  # from_bytes=False (the default): no from_bytes at all
        specs=(_spec(10, 2),),
        benches=("full", "seek", "bytes"),
        repeat=1,
    )
    records = {r["bench"]: r for r in _records_for(results, "correct")}
    assert records["bytes"]["status"] == "unsupported"
    assert records["bytes"]["reason"] == "correct has no from_bytes implementation"
    assert records["bytes"]["median_ms"] is None
    assert records["full"]["status"] == "ok"
    assert records["seek"]["status"] == "ok"


def test_loader_with_from_bytes_yields_ok_for_the_bytes_bench(corpus_dir):
    results = run(
        corpus_dir,
        [_correct_loader(from_bytes=True)],
        specs=(_spec(10, 2),),
        benches=("bytes",),
        repeat=2,
    )
    (record,) = _records_for(results, "correct")
    assert record["status"] == "ok"
    assert record["median_ms"] is not None
    assert record["realtime_factor"] is not None
    assert record["gate"] == "exact"
    assert record["reason"] is None


def test_bytes_bench_reads_the_file_once_and_passes_bytes_not_a_path(corpus_dir, monkeypatch):
    """The file must be read into memory once, outside the timed region.

    Asserts both halves of that requirement: the loader's callable is handed a
    `bytes` object (never a `Path`), and the file is read from disk exactly once
    regardless of how many timed trials run.
    """
    received_types: list[type] = []

    def tracking_from_bytes(raw: object) -> object:
        received_types.append(type(raw))
        return _soundfile_from_bytes(raw)

    loader = _correct_loader(from_bytes=True)
    loader = dataclasses.replace(loader, from_bytes=tracking_from_bytes)

    read_calls = {"n": 0}
    original_read_bytes = Path.read_bytes

    def counting_read_bytes(self: Path, *args: object, **kwargs: object) -> bytes:
        read_calls["n"] += 1
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

    results = run(corpus_dir, [loader], specs=(_spec(10, 2),), benches=("bytes",), repeat=3)
    (record,) = _records_for(results, "correct")

    assert record["status"] == "ok"
    assert received_types and all(t is bytes for t in received_types)
    # 1 correctness check + 1 warmup + 3 timed trials call the decode closure 5
    # times, but the file itself is read from disk exactly once.
    assert read_calls["n"] == 1


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
    # `seek_seconds` defaults to None so existing constructions (no chunk length
    # named) keep working.
    assert record.seek_seconds is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.status = "error"  # type: ignore[misc]


def test_record_seek_seconds_can_be_set_explicitly():
    record = Record(
        library="x",
        file="f.wav",
        duration_s=10,
        channels=2,
        format="wav",
        bench="seek",
        status="ok",
        median_ms=1.0,
        min_ms=1.0,
        max_ms=1.0,
        realtime_factor=1.0,
        gate="exact",
        reason=None,
        seek_seconds=3.0,
    )
    assert record.seek_seconds == 3.0


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


def test_platform_block_reads_seektable_state_off_the_corpus(tmp_path):
    """Reported from the files that were measured, not from whether metaflac exists."""
    from pabench.corpus import CorpusSpec, Fmt, generate, metaflac_available

    spec = CorpusSpec(duration_s=1, channels=1, fmt=Fmt("flac", "PCM_16"))
    generate(tmp_path, (spec,), flac_seektable=False)
    assert platform_block([], tmp_path)["flac_seektables"] is False

    if metaflac_available():
        other = tmp_path / "with"
        generate(other, (spec,))
        assert platform_block([], other)["flac_seektables"] is True


def test_platform_block_reports_unknown_seektable_state_without_a_corpus():
    assert platform_block([])["flac_seektables"] is None
