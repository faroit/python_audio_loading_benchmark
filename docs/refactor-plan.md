# Refactor implementation plan

**Spec:** `docs/refactor-design.md` (read it; it is the binding authority)
**Branch:** `refactor/torch-uv-seek`

## Global constraints

- Python `>=3.12`. `uv` only — never `pip`, `venv`, or bare `python`. No `requirements.txt`.
- Package is `pabench/` at the repo root, flat (not `src/`). Tests in `tests/`.
- Every loader is timed to a **`float32` channels-first torch tensor**; the conversion is
  inside the timed region.
- Default repeat 7, one untimed warmup, `gc.collect()` between trials,
  `time.perf_counter_ns`. Report median, min, max, realtime factor.
- **No bare `except`.** Every failure is recorded with its reason and the run continues.
- Correctness gate tolerances, fixed: `PCM_16` → `atol=1.5/32768`, `PCM_24` →
  `atol=1.5/8388608`, `FLOAT`/`flac-float` → `rtol=1e-5, atol=1e-7`; `mp3` → duration
  within 50 ms and RMS within 0.5 dB on the common length.
- Corpus: 44100 Hz; durations 1/10/60/300 s; mono and stereo; `wav` PCM_16/PCM_24/FLOAT,
  `flac`, `mp3`. Seeded, never committed, gitignored.
- **Write everything fresh.** Do not copy code from any other project. Do not reference
  or read any directory outside this repository.
- Run `uv run pytest` before each commit; it must pass.

---

### Task 1 — scaffolding and corpus

**Create:** `pyproject.toml`, `.gitignore` (add `corpus/`, `results/*.json`),
`pabench/__init__.py`, `pabench/corpus.py`, `tests/test_corpus.py`.
**Delete:** `requirements.txt`, `Dockerfile`, `generate_audio.sh`, `run.sh`.

`pyproject.toml`: project `pabench`, `requires-python = ">=3.12"`, console script
`pabench = "pabench.cli:console_main"`, dependencies `numpy`, `soundfile`, `torch`,
`matplotlib`; optional-dependency group `libs` holding `librosa`, `scipy`, `pydub`,
`audioread`, `stempeg`, `pedalboard`, `torchcodec`; dependency-group `test` with `pytest`.
Build backend `hatchling`. Include `[tool.ruff]` with `line-length = 100`.

**Produces:**
- `Fmt` — frozen dataclass: `container: str` (`"wav"`/`"flac"`/`"mp3"`), `subtype: str`
  (`"PCM_16"`/`"PCM_24"`/`"FLOAT"`/`"MP3"`), with property `key -> str` (e.g. `wav_pcm16`).
- `FORMATS: tuple[Fmt, ...]` — the five entries.
- `CorpusSpec` — frozen dataclass: `duration_s: int`, `channels: int`, `fmt: Fmt`,
  `sample_rate: int = 44100`; properties `filename -> str`, `frames -> int`.
- `DEFAULT_SPECS: tuple[CorpusSpec, ...]` — 4 durations x 2 channels x 5 formats = 40.
- `generate(corpus_dir, specs=DEFAULT_SPECS, seed=0) -> list[Path]` — WAV/FLAC via
  `soundfile`, MP3 by writing a temp WAV and invoking `ffmpeg -b:a 192k`. Reseed per spec
  so one regenerated file matches the full sweep. Peak 0.5 to avoid integer clipping.
- `corpus_files(corpus_dir, specs=DEFAULT_SPECS) -> list[tuple[CorpusSpec, Path]]`,
  raising `FileNotFoundError` whose message contains `pabench gen`.
- `ffmpeg_available() -> bool`.

**Tests:** 40 default specs with the expected axis values; filenames unique; generated
WAV/FLAC match requested rate/channels/frames/subtype; generation deterministic; peak
within 0.5; MP3 written and non-empty (skip if ffmpeg missing); `corpus_files` raises
naming `pabench gen`.

---

### Task 2 — canonical conversion and the correctness gate

**Create:** `pabench/canonical.py`, `pabench/verify.py`, `tests/test_canonical.py`,
`tests/test_verify.py`.

`canonical.py` produces:
- `Layout = Literal["channels_first", "frames_first"]`
- `to_tensor(data, layout) -> torch.Tensor` — accepts numpy arrays, torch tensors, and
  1-D input (gains a channel axis); returns contiguous `float32` `(channels, frames)`.
  Raises `ValueError` on unknown layout or >2 dimensions.

`verify.py` produces:
- `VerifyResult` — frozen dataclass `ok: bool`, `reason: str | None`, `gate: str`
  (`"exact"` or `"relaxed"`).
- `compare(reference, candidate, fmt) -> VerifyResult` where `fmt` is a `Fmt`.
  Exact gate for wav/flac using the tolerance table in the global constraints; relaxed
  gate for mp3: trim both to the common length, fail if the length difference exceeds
  50 ms of samples, fail if `20*log10(rms_candidate/rms_reference)` exceeds 0.5 dB.
  Unknown format raises `ValueError` naming it.

**Tests:** conversion from both layouts, 1-D promotion, dtype/contiguity, torch input,
rejections. Gate: identical passes; downmix, truncation, channel swap rejected; a
32768/32767 rescale passes on PCM_16 but a 2x gain fails; PCM_24 boundary just-below
passes and just-above fails; mp3 gate accepts a 1000-sample delay shift and rejects a
6 dB level error; unknown format raises.

---

### Task 3 — timing and the nine loaders

**Create:** `pabench/timing.py`, `pabench/loaders.py`, `tests/test_timing.py`,
`tests/test_loaders.py`. **Delete:** the old `loaders.py` and `utils.py`.

`timing.py` produces `DEFAULT_REPEAT = 7`, `Timing(median_ms, min_ms, max_ms, repeat)`,
`measure(fn, repeat=DEFAULT_REPEAT, clock=time.perf_counter_ns) -> Timing` (warmup
excluded, `gc.collect()` between trials, `repeat < 1` raises), and
`realtime_factor(audio_seconds, wall_ms) -> float`.

`loaders.py` produces:
- `Loader` — frozen dataclass: `name`, `layout`, `full: Callable[[Path], object] | None`,
  `seek: Callable[[Path, float, float], object] | None` (path, start_seconds,
  duration_seconds), `version`, `available: bool`, `error: str | None`,
  `formats: frozenset[str]` (containers it supports), `notes: str | None`.
- `PROBES: dict[str, Callable[[], Loader]]` for the nine libraries in the spec's table.
- `available_loaders(names=None) -> list[Loader]`, raising `KeyError` on an unknown name.
- A shared smoke test: after importing, each probe decodes a short temporary WAV and
  downgrades to `available=False` carrying the error if that fails. **Import success does
  not imply decode success** — torchcodec imports cleanly and then fails to load its
  native libraries, so an import-only probe reports a broken library as working.
- `scipy` sets `seek=None`; `pydub` and `audioread` set `seek=None`. `scipy_mmap` seeks by
  slicing the memmap. `pedalboard` uses `AudioFile(...)` as a context manager with
  `.seek(frame)` then `.read(n)` and returns `(channels, samples)` unmodified — do not
  index `[0]`. `torchcodec` uses `AudioDecoder(path).get_all_samples().data` and
  `.get_samples_played_in_range(start_seconds=, stop_seconds=)`.
- `formats` must be honest: `scipy` and `scipy_mmap` are WAV-only.

**Tests:** timing aggregation with an injected clock, warmup excluded, `repeat=0` raises.
Loaders: registry contains the nine names; unknown name raises; a loader whose callable
raises probes unavailable; for each available loader, decoding a generated fixture in
every format it claims matches the soundfile reference under the right gate, and the seek
result has the expected frame count (+/- one frame for mp3).

---

### Task 4 — FFmpeg resolution and the run loop

**Create:** `pabench/ffmpeg_env.py`, `pabench/run.py`, `tests/test_run.py`.

`ffmpeg_env.py` produces `find_ffmpeg_lib_dir() -> str | None` (searching
`/opt/homebrew/lib`, `/usr/local/lib`, `/opt/local/lib` for `libavutil*.dylib`) and
`ensure_ffmpeg_libs() -> None`, which on macOS only, when `DYLD_FALLBACK_LIBRARY_PATH` is
unset and a directory is found, sets it plus a sentinel env var and `os.execv`s once.
**It must never be called from `main()`** — only from the console entry point — because
calling it in-process replaces the interpreter, which restarts pytest mid-suite.

`run.py` produces:
- `Bench = Literal["full", "seek"]`
- `Record` — `library`, `file`, `duration_s`, `channels`, `format`, `bench`, `status`
  (`ok`/`unavailable`/`incorrect`/`error`/`unsupported`), `median_ms`, `min_ms`, `max_ms`,
  `realtime_factor`, `gate`, `reason`. Unmeasured axes are `None`, never fabricated.
- `seek_offset(spec, seed=0) -> float` — deterministic, inside the middle half of the file.
- `platform_block(loaders) -> dict` — OS, machine, Python, torch version, ffmpeg binary
  version, `DYLD_FALLBACK_LIBRARY_PATH`, and per-library version/availability/notes/error.
- `run(corpus_dir, loaders, specs=DEFAULT_SPECS, repeat=DEFAULT_REPEAT, benches=("full","seek"), progress=None) -> dict`
  returning `{"platform": ..., "records": [...]}`. Order: verify once per (library, file),
  then time. A library that does not support a container, or has no seek callable, yields
  `status="unsupported"` for those cells. The soundfile reference is decoded **once per
  file**, not once per library-file pair.
- `write_results(results, path)`.

**Tests:** a correct stub library yields ok records for both benches; an unavailable one
is recorded not dropped and carries its error; a downmixing one is `incorrect` with
withheld timings; a raising one is `error` and does not abort the run; a seek-less one is
`unsupported` for seek but ok for full; empty specs do not raise; records carry the corpus
axes; results JSON round-trips.

---

### Task 5 — report, CLI, README

**Create:** `pabench/report.py`, `pabench/cli.py`, `tests/test_report.py`,
`tests/test_cli.py`. **Rewrite:** `README.md`. **Delete:** `benchmark_np.py`,
`benchmark_tf.py`, `benchmark_pytorch.py`, `benchmark_metadata.py`, `plot.py`.

`report.py` produces `render_markdown(results) -> str` and
`write_plots(results, out_dir) -> list[Path]`. The markdown carries a platform table, a
library table (version, availability, notes, error), one table per (format, bench) with
libraries as columns and one row per (duration, channels), cells showing median plus
spread and the status word where a measurement was withheld, a list of unavailable and
incorrect libraries with reasons, an explicit note that MP3 is graded by the relaxed gate,
a measurement-noise section computing the observed `(max-min)/median` distribution, and a
statement that timings are warm-cache over synthetic noise. Plots: log-log duration vs
median ms, one line per library, faceted by format, one figure per bench, matplotlib with
the `Agg` backend. Before writing plotting code, load the `dataviz` skill and follow it;
label lines directly and de-collide the labels.

`cli.py` produces `main(argv=None) -> int` and `console_main() -> int` (the latter calls
`ensure_ffmpeg_libs()` then `main()`). Subcommands `gen`, `run`, `report`, `all`, sharing
`--corpus-dir` (default `corpus`), `--durations`, `--channels`, `--formats`; `run` also
takes `--library` (repeatable), `--bench`, `--repeat`, `--out`. Unknown library exits 2
with a message on stderr, never a traceback. A missing corpus is caught and reported.
`all` generates only what is missing, runs both benches, and writes the report.
Progress goes to stderr, one line per (library, file); stdout stays clean.

`README.md` documents: what is measured and how, the single command, the library table
with the seek column, the dropped libraries and why (aubio and soxbindings fail to build
under uv; torchaudio delegates to torchcodec; TensorFlow support removed), the per-format
gate including MP3's weaker guarantee, the automatic FFmpeg resolution, and the warm-cache
caveat. Do not paste results the benchmark has not produced.

**Tests:** report renders from a fixture including unavailable/incorrect/unsupported rows;
asserts an exact row format and a known median under the right column; plots created and
non-empty; CLI gen/run/report round-trip on a one-file corpus; unknown library exits 2;
missing corpus exits nonzero mentioning `pabench gen`.
