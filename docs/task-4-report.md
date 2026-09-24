# Task 4 report — FFmpeg resolution and the run loop

Date: 2026-09-24
Branch: `refactor/torch-uv-seek`

## Summary

Completed the Task 3 ruff cleanup (per-file `BLE001` ignores instead of a repo-wide
one), then implemented `pabench/ffmpeg_env.py` and `pabench/run.py` with their test
suites. Final state: **116 tests pass**, `uv run ruff check .` and
`uv run ruff format --check .` both clean.

## Fix 0 — `BLE001` scoped to per-file ignores

`pyproject.toml`'s `[tool.ruff.lint]` had a repo-wide `ignore = ["BLE001"]` (added in
Task 3) permitting blind `except Exception` everywhere. Replaced it with
`[tool.ruff.lint.per-file-ignores]` naming exactly `pabench/loaders.py` and
`pabench/run.py` — the two modules whose design requires catching any exception a
third-party decoder can raise. Confirmed `uv run ruff check .` still passes with no
new findings anywhere else in the package (there weren't any latent broad-except
usages outside those two files).

## `pabench/ffmpeg_env.py`

`find_ffmpeg_lib_dir() -> str | None` checks `/opt/homebrew/lib`, `/usr/local/lib`,
`/opt/local/lib` in order for a `libavutil*.dylib` glob match. `ensure_ffmpeg_libs()`
is a no-op off macOS, when `DYLD_FALLBACK_LIBRARY_PATH` is already set, when the
sentinel env var `_PABENCH_FFMPEG_LIBS_ENSURED` is present, or when no candidate
directory is found; otherwise it sets both variables and `os.execv(sys.executable,
[sys.executable, *sys.argv])`s once. Its docstring states the "never call this from
`main()`" constraint explicitly, and it is not wired into anything — Task 5's
`console_main` is the only intended caller.

**Deviation (addition, not in the plan's file list):** added `tests/test_ffmpeg_env.py`,
even though `docs/refactor-plan.md`'s Task 4 section only lists `tests/test_run.py`
as a new test file. Reason: `ensure_ffmpeg_libs` is a small but load-bearing, genuinely
dangerous function (it replaces the process image), and it seemed wrong to ship it
with zero test coverage when it's straightforward to test safely — every test either
exercises a no-op branch (returns before `os.execv`) or monkeypatches `os.execv`
itself to record the call instead of performing it, so the real interpreter-restart
path is never reached. 8 tests, none of which risk restarting the test runner.

Verified live on this machine (manually, outside pytest — this function must never
run inside the test process): with `DYLD_FALLBACK_LIBRARY_PATH` unset,
`find_ffmpeg_lib_dir()` returns `/opt/homebrew/lib`, matching Task 3's finding that
this exact fix resolves torchcodec's native-library load failure here.

## `pabench/run.py`

`Bench = Literal["full", "seek"]`; `Record` frozen dataclass with the fields listed
in the plan, in the given order. `SEEK_SECONDS = 1.0`. `platform_block(loaders)`
returns OS/machine/Python/torch version, the first line of `ffmpeg -version` (`None`
if ffmpeg isn't runnable), `DYLD_FALLBACK_LIBRARY_PATH`, and a
`version`/`available`/`notes`/`error` block per loader. `run(...)` returns
`{"platform": ..., "records": [...]}` with records as plain JSON-ready dicts
(`dataclasses.asdict` of each `Record`). `write_results` writes indented JSON,
creating parent directories.

One `Record` per `(library, file, bench)` triple. Per triple: unavailable library ->
`unavailable` (loader's own error as `reason`); unclaimed container -> `unsupported`;
`seek` requested but `loader.seek is None` -> `unsupported`; otherwise verify once
(catching any exception as `error`), then, only if verification passed, time via
`pabench.timing.measure` (also exception-guarded) and report `ok` with the gate used.
A verification failure records `incorrect` with the gate and reason, timings withheld
(`None`). The `soundfile` reference is decoded once per file via `_decode_reference`
and cached across every library and bench that needs it; the `seek` reference is a
slice of that same cached tensor (`_slice_reference`), not a second file read.
`progress(library_name, filename)` is called once per `(library, file)` pair, before
its benches are built, if given.

### `seek_offset` and a real bug this design caught

`seek_offset(spec, seed=0)` computes a deterministic **frame index** (SHA-256-seeded,
biased into `[0.25, 0.75] * duration_s`, clamped so `start_frame + read_frames <=
total_frames`) and returns `start_frame / spec.sample_rate`. Degenerate 1 s files
(`duration_s <= SEEK_SECONDS`) return `0.0`, which exactly fits since the corpus's
shortest duration equals `SEEK_SECONDS`.

Returning an integer-frame-aligned offset, rather than an arbitrary float in the
middle-half band, was **not** the first thing I tried — it's a fix for a real bug
this task's own end-to-end sanity check caught. My first version of `seek_offset`
returned an arbitrary float. Running `run()` against the real nine loaders (see
below) showed `librosa`'s `seek` bench spuriously `incorrect` for several
mono/duration combinations, with a max-abs-diff near 1.0 — essentially total
decorrelation for white noise. Root cause: `soundfile`'s own seek (used both by the
`soundfile` loader and by this module's reference-slicing) converts
`start_seconds -> frame` with `round()`, while `librosa.load(..., offset=...)`
truncates. For an arbitrary offset like `6.085572345206106 s` at 44100 Hz these
disagree by exactly one frame (`268374` vs `268373`), and one frame of misalignment
in white noise looks like a completely different signal under the exact gate. This
is a real, verifiable cross-library rounding difference, but it is an artifact of
which offset `seek_offset` happens to choose, not a defect in either library or in
the run loop's error handling — so the fix is to choose offsets that both rounding
conventions agree on, which any exact-frame-boundary offset guarantees (dividing an
integer frame count by the integer sample rate and multiplying back recovers the
same integer well within floating-point precision, for both `round()` and
truncation). After the fix, the same end-to-end check showed `librosa` fully `ok`
across all combinations. This is called out here rather than filed as one of the
plan's two named expected failures because it isn't a library limitation at all —
it was a bug in this module, now fixed, not a result to report as "honest incorrect."

## Tests

`tests/test_run.py` (16 tests): correct stub loader -> `ok` both benches with timing
fields populated; unavailable loader -> one record per bench, `unavailable`, carries
`loader.error`; downmixing loader (stereo file, mono-average adapter) -> `incorrect`,
`gate="exact"`, all timing fields `None`; raising loader -> `error` with the
exception type and message in `reason`, and a second loader in the same run still
produces a real `ok` record (run doesn't abort); a `seek=None` loader -> `ok` for
`full`, `unsupported` for `seek`; a loader whose `formats` never claims the corpus
file's container -> `unsupported`; empty `specs` -> `records == []`, no raise;
records carry `file`/`duration_s`/`channels`/`format` from the spec; `progress` is
called exactly once per `(library, file)` pair regardless of bench count; the
`soundfile` reference is decoded exactly once for one file even with 2 libraries x 2
benches (monkeypatches `pabench.run._decode_reference` to count calls);
`seek_offset` determinism and the 1 s degenerate case; `Record` is a frozen dataclass
(`dataclasses.FrozenInstanceError` on mutation); `write_results` round-trips through
real JSON on disk, including creating a nested parent directory that didn't exist;
`platform_block` includes the expected top-level keys and a per-library
`available` flag.

`tests/test_ffmpeg_env.py` (8 tests, addition — see deviation above):
`find_ffmpeg_lib_dir` returns `None` when no candidate matches, finds a directory
containing a `libavutil*.dylib`, and prefers the first matching candidate in order.
`ensure_ffmpeg_libs` no-ops (asserting `os.execv` is never even reached) off macOS,
when the fallback var is already set, when the sentinel is present, and when no
directory is found; and, on the one active path, calls `os.execv` exactly once with
`sys.executable` and sets both env vars, verified via a monkeypatched `os.execv`
that records the call instead of performing it (with explicit cleanup of the two env
vars it sets directly, since `monkeypatch.setenv`/`delenv` never observed that
mutation and won't undo it).

## End-to-end sanity check against the real nine loaders (not part of the test suite)

Ran `run()` by hand against `available_loaders()` (all nine probe as available on
this machine with `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` set) over a small
corpus spanning both durations either side of `SEEK_SECONDS`, both channel counts,
all three wav subtypes, and mp3. Every record landed on `ok`, `unsupported`, or one
of the two failures Task 3 already documented and this task was told to expect and
not "fix":

- `audioread`: `incorrect` on `PCM_24` and `FLOAT` wav (8 records) — 16-bit-only
  internal precision.
- `scipy_mmap`: `error` on `PCM_24` wav (8 records) — can't `mmap` a 24-bit
  ("3-byte container") wav at all.

No other library produced anything other than `ok`/`unsupported` across the sweep.
This was a manual check (not committed as a test) since it depends on the optional
`libs` extra and a real `ffmpeg`/Homebrew install; it's recorded here as evidence,
not as part of the automated suite.

## Deviations summary

1. Added `tests/test_ffmpeg_env.py`, not listed in the plan's Task 4 file list — see
   above.
2. `seek_offset` returns an offset quantized to an exact frame boundary
   (`start_frame / sample_rate`) rather than an arbitrary float within the
   middle-half band. The plan only specifies "landing inside the middle half" and a
   sensible 1 s degenerate case; it doesn't anticipate the cross-library rounding
   issue this choice avoids. Documented in code and above.
3. `run()`'s `records` are plain dicts (`dataclasses.asdict(Record)`), not `Record`
   instances, so the return value is directly `json.dumps`-able by `write_results`
   without a custom encoder. The plan states the shape as
   `{"platform": ..., "records": [...]}` without specifying the element type.

## Anomaly observed, not caused by this task

`git status` at commit time shows `docs/refactor-design.md` and `docs/refactor-plan.md`
modified on disk (removing `PCM_24` from the corpus/gate description), and an
untracked `docs/task-1-report.md`. **I did not make these changes** — I only read
these two files (with `Read`, never `Edit`/`Write`) and quoted their original content
(including `PCM_24`) verbatim in this report. Their on-disk modification timestamps
fall inside this session's window but before `pabench/run.py` was written, which
means something other than my own tool calls touched them while I was working —
most likely a concurrent process/session with access to the same working directory.
`pabench/corpus.py` itself is untouched and still defines `PCM_24` as part of
`FORMATS`/`DEFAULT_SPECS`, and my implementation (and the end-to-end check above)
is consistent with the original, unmodified design doc I was given, which explicitly
retains `PCM_24`. I left both files and the untracked report as-is on disk and did
**not** include them in this task's commit — only the files this task actually
produced or intentionally changed are staged. Flagging this so it isn't mistaken for
part of Task 4, and so whoever owns that edit can reconcile or commit it separately.

## Commands and final output

```
$ uv run pytest -q
116 passed in ~4s

$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
21 files already formatted
```

## Files

- `pyproject.toml` — `BLE001` moved to `per-file-ignores` (this task's own change)
- `pabench/ffmpeg_env.py`, `tests/test_ffmpeg_env.py` (new)
- `pabench/run.py`, `tests/test_run.py` (new)
- Not touched, not committed by this task, flagged above: `docs/refactor-design.md`,
  `docs/refactor-plan.md` (both modified on disk by something other than this
  session), `docs/task-1-report.md` (untracked, pre-existing)
