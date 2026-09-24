# python_audio_loading_benchmark — refactor design

Date: 2026-09-24
Branch: `refactor/torch-uv-seek`

## Purpose

Modernise the benchmark so its numbers can be trusted and reproduced: one command,
`uv`-managed dependencies, PyTorch as the only target tensor type, and a seek benchmark
alongside the existing full-file one.

## Why the current harness understates the problem

Three properties of the existing code make its published numbers hard to defend, and
they set the requirements below:

1. `benchmark_pytorch.py` wraps every measurement in `try: ... except: continue`. A
   library that raises disappears from the results rather than being reported as broken,
   so a reader cannot distinguish "not measured" from "failed".
2. Nothing verifies that a library decoded the file correctly. A loader that returns
   int16, a single channel, or a truncated signal simply posts a fast time and wins.
3. Timing is a single `time.time()` span over `repeat` passes of a DataLoader, divided by
   the file count. There is no warmup, no per-trial dispersion, and the mean of three
   passes hides run-to-run noise that is comparable to the differences being reported.

## Scope

In scope: full-file decode and seek (excerpt) decode, to torch tensors, over locally
generated WAV/FLAC/MP3 files.

Out of scope, deliberately: numpy and TensorFlow targets (removed), metadata-only
benchmarks (the existing `benchmark_metadata.py` is dropped with the numpy harness),
multiprocess DataLoader throughput, and any network or streaming source.

## Libraries

Selected by one rule: it must install with `uv` on Python 3.12 and import cleanly.

| library | full-file | seek | notes |
| --- | --- | --- | --- |
| `soundfile` | yes | yes | libsndfile; the correctness reference |
| `librosa` | yes | yes | 1.0.0; `offset=`/`duration=` |
| `scipy.io.wavfile` | yes | no | WAV only, no seek API |
| `scipy` memmap | yes | yes | seek is a memmap slice |
| `pydub` | yes | no | no native seek |
| `audioread` | yes | no | no native seek |
| `pedalboard` | yes | yes | `AudioFile.seek` + `read` |
| `torchcodec` | yes | yes | `get_samples_played_in_range` |

Dropped, with reasons recorded in the README so the removals are not silent:

- `aubio` — `uv` cannot build it (`Failed to build aubio==0.4.9`).
- `soxbindings` — `uv` cannot build it (`Failed to build soxbindings==1.2.3`).
- `torchaudio` — superseded by `torchcodec`; from 2.9 `torchaudio.load` delegates to
  torchcodec, so keeping both would report one decoder twice.
- `stempeg` — subprocess-per-call; measures process startup, not decoding.
- `tensorflow`, `tensorflow_io` — the TensorFlow target is removed entirely.

`pedalboard` follows upstream PR #21 in intent, but not in code: that PR's loader ends
`f.read(f.frames)[0]`, which keeps only the first channel. The documented API returns
`(channels, samples)` and is used unmodified here.

### Seek benchmark membership

`scipy` (non-memmap), `pydub` and `audioread` have no seek API and are excluded from the
seek benchmark rather than measured as read-everything-then-slice. The README states the
exclusion and the reason, so their absence is not mistaken for an oversight.

## Corpus

Generated locally, never committed. Seeded so a regenerated corpus is byte-identical.

- Sample rate 44100.
- Durations 1, 10, 60, 300 s.
- Channels: mono and stereo.
- Formats: `wav` in `PCM_16` and `FLOAT`; `flac` (16-bit); `mp3` (CBR 192k).

WAV and FLAC are written with `soundfile`; MP3 is encoded with the `ffmpeg` binary, which
is the one non-Python prerequisite.

Bit depth remains an axis, as `PCM_16` versus `FLOAT`, because it changes the ranking and
not merely the magnitude: FFmpeg-backed decoders are markedly slower converting 16-bit PCM
to float than they are reading float32, which a single-subtype corpus would report as a
flat property of "wav". 24-bit was dropped from the sweep: it sits between the two cases
and added a third of the corpus for no distinct finding.

## Measurement

Target type is a `float32` channels-first torch tensor for every library. Loaders that
return numpy are converted, and the conversion is inside the timed region, because a
caller who wants a tensor pays for it.

Two benchmarks:

- **full** — decode the entire file.
- **seek** — decode 1 s starting at a seeded offset in the middle half of the file.
  The offset is fixed per file so every library does identical work.

### Protocol

- One untimed warmup per (library, file, benchmark), which also warms the page cache.
- Then `--repeat` timed trials, default 7.
- `gc.collect()` between trials; `time.perf_counter_ns` for timing.
- Reported: median (primary), min, max, and the realtime factor.
- Warm page cache throughout, stated in the report.

### Correctness gate

Every library is checked against a `soundfile` reference before its timings count. A
failing library is reported as incorrect with the reason, and its timings are withheld.

The gate is per-format, because one tolerance cannot be right for all three:

- `wav`/`flac` — sample-exact within a tolerance set by the source bit depth: 1.5 LSB for
  integer subtypes, `atol=1e-7` for float32. The integer allowance exists because
  normalisation conventions legitimately differ by one LSB between libraries (dividing by
  32767 versus 32768); it still rejects downmixing, truncation, channel swaps and any
  gain error above roughly 0.001 dB.
- `mp3` — a relaxed gate. MP3 decoders disagree on encoder delay, so decoded lengths
  differ by around a thousand samples and samples never match bit-for-bit. The gate
  checks that the duration is within 50 ms of the reference and that the RMS level is
  within 0.5 dB after trimming both signals to their common length. The report labels
  MP3 results as graded by the relaxed gate, so the weaker guarantee is visible rather
  than implied to be the same check.

## Structure

```
pabench/
  __init__.py
  corpus.py      # deterministic corpus generation
  loaders.py     # one adapter per library: full + seek
  canonical.py   # anything -> float32 (channels, frames) torch tensor
  verify.py      # per-format correctness gate
  timing.py      # warmup + repeat + aggregation
  run.py         # orchestration; produces results.json
  report.py      # markdown tables + plots
  cli.py         # gen / run / report / all
tests/
pyproject.toml   # uv-managed; no requirements.txt
```

Removed: `benchmark_np.py`, `benchmark_tf.py`, `benchmark_pytorch.py`,
`benchmark_metadata.py`, `loaders.py`, `utils.py`, `plot.py`, `run.sh`,
`generate_audio.sh`, `requirements.txt`, `Dockerfile`.

## Entry point

`uv run pabench all` regenerates the corpus if needed, runs both benchmarks, and writes
the report. `gen`, `run` and `report` are available separately; `report` never re-runs a
benchmark.

## Error handling

- A library that fails to import is recorded unavailable with its error and skipped.
- A library that fails the correctness gate has its timings withheld and the reason
  recorded.
- A library that raises mid-measurement is recorded as errored for that cell.
- No bare `except`. A run never aborts because one library is broken, and nothing that
  failed is silently absent from the report.

## Prerequisites

Python 3.12 and `uv`; the `ffmpeg` binary for MP3 encoding during corpus generation.

FFmpeg's shared libraries are located automatically for `torchcodec`, which loads a
native library built against a specific FFmpeg major (4 through 9). On macOS those
libraries carry no `LC_RPATH`, so the loader consults `DYLD_FALLBACK_LIBRARY_PATH`, whose
default omits Homebrew's `/opt/homebrew/lib`. The CLI finds the directory and re-executes
once with the variable set; an explicit value is respected. This cannot be fixed in
process: dyld reads the variable at start, and a `ctypes` preload does not help because
dyld re-searches the rpath list when loading a dependent dylib.

## Testing

pytest, with no optional library required to run the suite: adapter tests use
`pytest.importorskip`. Covered: corpus determinism and written properties; canonical
conversion from both layout conventions; the gate accepting a 1-LSB convention shift
while rejecting a downmix, a truncation and a real gain error; MP3's relaxed gate
accepting a delay-shifted signal and rejecting a wrong one; timing aggregation with an
injected clock; report rendering from a fixture including unavailable and failed entries;
CLI argument handling.

## Licensing

All code is written for this repository. Nothing is copied from any other codebase.
