"""The run loop: loaders + a corpus -> a results dict, and its JSON serialisation.

For every (library, file, bench) triple this produces exactly one `Record`, except
for `bench == "seek"`: a seek chunk length has real consequences for what is being
measured (`docs/refactor-design.md`), so `run()` produces one `Record` per
(library, file, chunk length) for every chunk length in `SEEK_DURATIONS` (or
whatever subset the caller passes) that actually fits the file -- a chunk longer
than the file is skipped entirely rather than recorded with an invented status. A
library that isn't available, doesn't claim a file's container, has no seek
implementation, or has no `from_bytes` implementation never touches the decoder
or the correctness gate for that triple -- it's recorded as
`unavailable`/`unsupported` and the run moves on. A
library that is exercised is verified once against a `soundfile` reference
before it is timed; a verification failure withholds timings (`incorrect`), and
an exception anywhere in decode or verification is caught and recorded
(`error`) without aborting the run for the libraries that follow. See
`docs/refactor-design.md`, "Error handling" and "Measurement".

No bare `except`: `except Exception` is used deliberately here (see the
`pabench/run.py` entry in `pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]`)
because a third-party decoder can raise anything, and the whole point of this
module is that one broken library never takes the run down with it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import soundfile as sf
import torch

from pabench.canonical import to_tensor
from pabench.corpus import DEFAULT_SPECS, CorpusSpec, corpus_files, has_seektable
from pabench.loaders import Loader
from pabench.timing import DEFAULT_REPEAT, measure, realtime_factor
from pabench.verify import compare, compare_seek

Bench = Literal["full", "seek", "bytes"]
Status = Literal["ok", "unavailable", "incorrect", "error", "unsupported"]

#: Chunk lengths the seek bench measures. A seek's cost is
#: `fixed_locate_cost + per_second_decode_cost * chunk_length`; at 1s the fixed cost
#: dominates for every library, at 30s the decode cost does, and one chunk length
#: cannot show that tradeoff. `run()` measures a given (file, chunk) pair only when
#: `chunk <= file duration` -- see `run()` and `seek_offset`.
SEEK_DURATIONS: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0)

ProgressFn = Callable[[str, str], None]  # (library name, corpus filename) -> None


@dataclass(frozen=True)
class Record:
    library: str
    file: str
    duration_s: int
    channels: int
    format: str
    bench: Bench
    status: Status
    median_ms: float | None
    min_ms: float | None
    max_ms: float | None
    realtime_factor: float | None
    gate: str | None
    reason: str | None
    #: Interquartile range in ms. Preferred over `max_ms - min_ms` for judging noise:
    #: the full range grows with the trial count, so ranges from runs with different
    #: `--repeat` values are not comparable, while the IQR is stable.
    iqr_ms: float | None = None
    #: The seek chunk length this record measures, `None` for `full`/`bytes` records.
    #: Placed last with a default so existing positional/keyword constructions of
    #: `Record` (predating the chunk-duration axis) keep working unchanged.
    seek_seconds: float | None = None


def seek_offset(spec: CorpusSpec, chunk_seconds: float, seed: int = 0) -> float:
    """A deterministic seek start (seconds) landing in the middle half of the file.

    Chosen so that reading `chunk_seconds` seconds starting at the returned offset
    never runs past the end of the file: the start frame is clamped to
    `[0, total_frames - read_frames]`, biased towards the middle half
    (`[0.25, 0.75] * duration_s`) whenever that band leaves room for the read.

    Degenerate case: when the file is too short to leave any room at all (a
    `duration_s <= chunk_seconds` file -- including `chunk_seconds == duration_s`,
    i.e. the chunk is the whole file), the middle half can't be honoured without
    reading past the end, so this returns `0.0` rather than a negative or
    out-of-range offset.

    The result is deliberately computed as an integer frame count divided by
    `spec.sample_rate`, not as an arbitrary float in the middle-half band: two
    libraries can legitimately convert the same offset-in-seconds to different
    frame indices (soundfile-based loaders round; librosa truncates), and for
    white noise even a 1-frame misalignment looks like total decorrelation
    under the exact gate. Landing exactly on a frame boundary means every
    reasonable rounding convention recovers the same integer frame, so the
    gate is actually exercising decode correctness rather than an artifact of
    this function's own choice of offset.
    """
    total_frames = spec.frames
    read_frames = round(chunk_seconds * spec.sample_rate)
    max_start_frame = max(0, total_frames - read_frames)
    if max_start_frame <= 0:
        return 0.0

    middle_lo_frame = min(round(spec.duration_s * 0.25 * spec.sample_rate), max_start_frame)
    middle_hi_frame = min(round(spec.duration_s * 0.75 * spec.sample_rate), max_start_frame)
    if middle_hi_frame <= middle_lo_frame:
        start_frame = middle_lo_frame
    else:
        key = (
            f"{seed}:{spec.duration_s}:{spec.channels}:{spec.sample_rate}:"
            f"{spec.fmt.key}:{chunk_seconds}"
        )
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        start_frame = middle_lo_frame + round(fraction * (middle_hi_frame - middle_lo_frame))

    return start_frame / spec.sample_rate


def _decode_reference(path: Path) -> torch.Tensor:
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return to_tensor(data, "frames_first")


def _slice_reference(
    reference: torch.Tensor, start_seconds: float, duration_seconds: float, sample_rate: int
) -> torch.Tensor:
    start_frame = round(start_seconds * sample_rate)
    num_frames = round(duration_seconds * sample_rate)
    return reference[:, start_frame : start_frame + num_frames]


def _ffmpeg_version_line() -> str | None:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    except Exception:  # ffmpeg missing, not on PATH, or misbehaving: not fatal
        return None
    lines = result.stdout.splitlines()
    return lines[0] if lines else None


def _corpus_has_flac_seektables(corpus_dir: Path | None) -> bool | None:
    """Whether the corpus's FLAC files actually carry seektables.

    Read off the files rather than inferred from whether `metaflac` is installed:
    a corpus can be generated with seektables suppressed, or carried over from
    another machine, and the report should describe the files that were measured.
    """
    if corpus_dir is None:
        return None
    flacs = sorted(Path(corpus_dir).glob("*.flac"))
    if not flacs:
        return None
    return all(has_seektable(path) for path in flacs)


def platform_block(loaders: list[Loader], corpus_dir: Path | None = None) -> dict:
    """OS/interpreter/library facts worth recording alongside a set of results."""
    return {
        "os": _platform.platform(),
        "machine": _platform.machine(),
        "python_version": _platform.python_version(),
        "torch_version": torch.__version__,
        "ffmpeg_version": _ffmpeg_version_line(),
        # Whether the FLAC corpus carries SEEKTABLE blocks. Without metaflac the files
        # are still valid but every library must seek by binary search, which is not
        # what a FLAC from the wild looks like, so the seek numbers mean something
        # different and the report says so.
        "flac_seektables": _corpus_has_flac_seektables(corpus_dir),
        "dyld_fallback_library_path": os.environ.get("DYLD_FALLBACK_LIBRARY_PATH"),
        "libraries": {
            loader.name: {
                "version": loader.version,
                "available": loader.available,
                "notes": loader.notes,
                "error": loader.error,
            }
            for loader in loaders
        },
    }


def _decode_fn(
    loader: Loader, path: Path, bench: Bench, start_seconds: float, chunk_seconds: float | None
) -> Callable[[], object]:
    if bench == "full":
        return lambda: to_tensor(loader.full(path), loader.layout)
    if bench == "bytes":
        # Read once, here, before the decode closure is ever called (including the
        # correctness check below and every warmup/timed trial in `measure`): the
        # "bytes" bench measures decode-from-memory, not file reading, so the file
        # read itself must never fall inside the timed region (see
        # docs/refactor-design.md).
        raw_bytes = path.read_bytes()
        return lambda: to_tensor(loader.from_bytes(raw_bytes), loader.layout)
    assert chunk_seconds is not None  # every "seek" call site supplies a chunk length
    return lambda: to_tensor(loader.seek(path, start_seconds, chunk_seconds), loader.layout)


def _unmeasured(
    *,
    library: str,
    spec: CorpusSpec,
    bench: Bench,
    status: Status,
    reason: str | None,
    gate: str | None = None,
    seek_seconds: float | None = None,
) -> Record:
    return Record(
        library=library,
        file=spec.filename,
        duration_s=spec.duration_s,
        channels=spec.channels,
        format=spec.fmt.key,
        bench=bench,
        status=status,
        median_ms=None,
        min_ms=None,
        max_ms=None,
        iqr_ms=None,
        realtime_factor=None,
        gate=gate,
        reason=reason,
        seek_seconds=seek_seconds,
    )


def _build_record(
    loader: Loader,
    spec: CorpusSpec,
    path: Path,
    bench: Bench,
    repeat: int,
    reference_for: Callable[[Path], torch.Tensor],
    chunk_seconds: float | None = None,
) -> Record:
    """Build one record. `chunk_seconds` is the seek chunk length for `bench == "seek"`,
    `None` for `full`/`bytes`; the caller (`run()`) is responsible for only calling this
    for a `(spec, chunk_seconds)` pair where the chunk actually fits the file.
    """
    if not loader.available:
        return _unmeasured(
            library=loader.name,
            spec=spec,
            bench=bench,
            status="unavailable",
            reason=loader.error,
            seek_seconds=chunk_seconds,
        )

    if spec.fmt.container not in loader.formats:
        return _unmeasured(
            library=loader.name,
            spec=spec,
            bench=bench,
            status="unsupported",
            reason=f"{loader.name} does not decode {spec.fmt.container!r} files",
            seek_seconds=chunk_seconds,
        )

    if bench == "seek" and loader.seek is None:
        return _unmeasured(
            library=loader.name,
            spec=spec,
            bench=bench,
            status="unsupported",
            reason=f"{loader.name} has no seek implementation",
            seek_seconds=chunk_seconds,
        )

    if bench == "bytes" and loader.from_bytes is None:
        return _unmeasured(
            library=loader.name,
            spec=spec,
            bench=bench,
            status="unsupported",
            reason=f"{loader.name} has no from_bytes implementation",
        )

    start_seconds = seek_offset(spec, chunk_seconds) if bench == "seek" else 0.0
    decode = _decode_fn(loader, path, bench, start_seconds, chunk_seconds)

    try:
        full_reference = reference_for(path)
        if bench == "seek":
            reference = _slice_reference(
                full_reference, start_seconds, chunk_seconds, spec.sample_rate
            )
        else:
            reference = full_reference
        candidate = decode()
        if bench == "seek":
            # Seconds-to-frame rounding differs between libraries; allow one frame.
            verify_result = compare_seek(
                reference, candidate, spec.fmt, sample_rate=spec.sample_rate
            )
        else:
            verify_result = compare(reference, candidate, spec.fmt, sample_rate=spec.sample_rate)
    except Exception as exc:  # any decoder/verification failure is recorded, never raised
        return _unmeasured(
            library=loader.name,
            spec=spec,
            bench=bench,
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
            seek_seconds=chunk_seconds,
        )

    if not verify_result.ok:
        return _unmeasured(
            library=loader.name,
            spec=spec,
            bench=bench,
            status="incorrect",
            reason=verify_result.reason,
            gate=verify_result.gate,
            seek_seconds=chunk_seconds,
        )

    try:
        timing = measure(decode, repeat=repeat)
    except Exception as exc:  # decode succeeded once above but can still fail under repeat
        return _unmeasured(
            library=loader.name,
            spec=spec,
            bench=bench,
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
            gate=verify_result.gate,
            seek_seconds=chunk_seconds,
        )

    # "bytes" is a full-file decode differing only in where the input comes from, so
    # it is measured against the whole file's duration exactly like "full".
    if bench in ("full", "bytes"):
        audio_seconds = float(spec.duration_s)
    else:
        audio_seconds = chunk_seconds
    return Record(
        library=loader.name,
        file=spec.filename,
        duration_s=spec.duration_s,
        channels=spec.channels,
        format=spec.fmt.key,
        bench=bench,
        status="ok",
        median_ms=timing.median_ms,
        min_ms=timing.min_ms,
        max_ms=timing.max_ms,
        iqr_ms=timing.iqr_ms,
        realtime_factor=realtime_factor(audio_seconds, timing.median_ms),
        gate=verify_result.gate,
        reason=None,
        seek_seconds=chunk_seconds if bench == "seek" else None,
    )


def run(
    corpus_dir: Path,
    loaders: list[Loader],
    specs: tuple[CorpusSpec, ...] = DEFAULT_SPECS,
    repeat: int = DEFAULT_REPEAT,
    benches: tuple[Bench, ...] = ("full", "seek"),
    seek_durations: tuple[float, ...] = SEEK_DURATIONS,
    progress: ProgressFn | None = None,
) -> dict:
    """Run every loader against every corpus file/bench, returning JSON-ready results.

    `{"platform": platform_block(loaders), "records": [...]}`, where each record is
    a plain dict (`dataclasses.asdict` of a `Record`), ready for `write_results`.

    The `soundfile` reference for a file is decoded once and reused for every
    library and bench that needs it (a `seek` reference is a slice of that same
    decode, not a second file read).

    The `seek` bench runs once per chunk length in `seek_durations` per file, but
    only for chunk lengths that fit inside the file (`chunk <= spec.duration_s`): a
    chunk longer than the file makes no sense, so that combination is skipped
    entirely rather than recorded with some invented status.
    """
    corpus = corpus_files(corpus_dir, specs)

    reference_cache: dict[Path, torch.Tensor] = {}

    def reference_for(path: Path) -> torch.Tensor:
        cached = reference_cache.get(path)
        if cached is not None:
            return cached
        tensor = _decode_reference(path)
        reference_cache[path] = tensor
        return tensor

    records: list[Record] = []
    for loader in loaders:
        for spec, path in corpus:
            if progress is not None:
                progress(loader.name, spec.filename)
            for bench in benches:
                if bench != "seek":
                    records.append(_build_record(loader, spec, path, bench, repeat, reference_for))
                    continue
                for chunk_seconds in sorted(seek_durations):
                    if chunk_seconds > spec.duration_s:
                        continue
                    records.append(
                        _build_record(
                            loader, spec, path, bench, repeat, reference_for, chunk_seconds
                        )
                    )

    return {
        "platform": platform_block(loaders, corpus_dir),
        "records": [asdict(record) for record in records],
    }


def write_results(results: dict, path: Path) -> None:
    """Write `results` as indented JSON to `path`, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
