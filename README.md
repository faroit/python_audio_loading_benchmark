# pabench — Python audio-loading benchmark

`pabench` measures how fast nine Python audio-decoding libraries can get a WAV,
FLAC, or MP3 file into a **`float32`, channels-first PyTorch tensor** — the
representation almost every audio model wants — both for a **full-file decode**
and for a **seek (excerpt) decode**. It is `uv`-managed, has no untracked
"it worked on my machine" state, and checks every library's output for
correctness before it lets that library's timing count.

See `docs/refactor-design.md` for the full design rationale and
`docs/refactor-plan.md` for how it was built.

## What is measured

For every (library, corpus file) pair, `pabench` runs two benchmarks:

- **full** — decode the entire file to a tensor.
- **seek** — decode a fixed 1-second excerpt starting at a deterministic offset
  in the middle half of the file, via whichever seek API the library exposes.
  Libraries with no seek API are excluded from this benchmark rather than
  measured as read-everything-then-slice (see the library table below).

Every library's output — numpy array, torch tensor, or anything else it hands
back — is converted to `float32` `(channels, frames)` **inside the timed
region**, because a caller who wants a tensor pays for that conversion.

### Protocol

1. One **untimed warmup** decode per (library, file, benchmark), which also
   warms the OS page cache for that file.
2. **7 timed trials** by default (`--repeat`), each preceded by `gc.collect()`
   so a garbage-collection pause in one trial can't leak into the next, timed
   with `time.perf_counter_ns`.
3. The **median** trial is the headline number; min and max are reported
   alongside it as the spread.
4. Every library is checked against a `soundfile`-decoded reference **before**
   its timings are allowed to count (see "Correctness gate" below). A library
   that fails the gate is reported as incorrect, with the reason, and its
   timings are withheld — never silently dropped.

Timings are **warm-page-cache**, over a **locally generated, synthetic
white-noise corpus** (44.1 kHz, 1/10/60/300 s, mono/stereo). They describe
decode speed against this machine's page cache and this synthetic corpus —
not cold-disk I/O, and not real program material. See "Caveats" below.

## Quickstart

```bash
uv sync --extra libs
uv run pabench all
```

This is the single reproduce command: it generates only the corpus files that
are missing, runs both benchmarks against every library that's installed and
importable, and writes a Markdown report plus plots under `results/`. No
audio is committed to the repository — the corpus is generated locally and
gitignored (`corpus/`).

Each subcommand is also available on its own:

```bash
uv run pabench gen                     # generate the corpus
uv run pabench run --library soundfile # run one library
uv run pabench report                  # render results.json -> report.md + plots
```

All four subcommands (`gen`, `run`, `report`, `all`) accept `--corpus-dir`,
`--durations`, `--channels`, and `--formats` to work on a subset of the
default 32-file corpus (4 durations x 2 channel counts x 4 formats); `run`
additionally takes repeatable `--library`, `--bench {full,seek,both}`,
`--repeat`, and `--out`. Run `uv run pabench <subcommand> --help` for the
full list.

FFmpeg's shared libraries (needed by `torchcodec`) are located automatically:
on macOS, `pabench`'s console entry point finds Homebrew/MacPorts FFmpeg and
re-executes itself once with `DYLD_FALLBACK_LIBRARY_PATH` set, since dyld
only reads that variable at process start and can't be fixed up once the
interpreter is already running (see `pabench/ffmpeg_env.py` and
`docs/refactor-design.md`).

## Prerequisites

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/).
- The `ffmpeg` binary on `PATH`, for MP3 corpus generation.

## Libraries

| Library | Full-file | Seek | Notes |
| --- | --- | --- | --- |
| [`soundfile`](https://pysoundfile.readthedocs.io/) | yes | yes | libsndfile; the correctness reference every other library is checked against |
| [`librosa`](https://librosa.org/) | yes | yes | `offset=`/`duration=` |
| [`scipy.io.wavfile`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html) | yes | no | WAV only; no seek API |
| `scipy.io.wavfile` (memmap) | yes | yes | WAV only; seek is a memmap slice |
| [`pydub`](https://github.com/jiaaro/pydub) | yes | no | no native seek |
| [`audioread`](https://github.com/beetbox/audioread) | yes | no | no native seek; always yields 16-bit PCM, so it cannot pass the exact gate against a float32 or 24-bit source |
| [`pedalboard`](https://github.com/spotify/pedalboard) | yes | yes | `AudioFile.seek` + `.read` |
| [`torchcodec`](https://github.com/pytorch/torchcodec) | yes | yes | `get_samples_played_in_range`; imports cleanly even when its native FFmpeg bindings can't load, so `pabench` smoke-tests a real decode before trusting it as available |

A library is selected by one rule: it must install with `uv` on Python 3.12
and import cleanly. Every library above is probed on every run, whether or
not it's installed — an unavailable library is reported as such, with its
import error, never silently omitted (see `docs/refactor-design.md`, "Error
handling").

### Dropped libraries

These appeared in earlier versions of this benchmark and are deliberately not
here:

- **`aubio`**, **`soxbindings`** — `uv` cannot build either of them
  (`Failed to build aubio==0.4.9`, `Failed to build soxbindings==1.2.3`).
- **`torchaudio`** — from PyTorch 2.9, `torchaudio.load()` delegates to
  `torchcodec` internally, so keeping both would report the same decoder
  twice under two names.
- **TensorFlow / `tensorflow_io`, and the plain-numpy target** — this
  benchmark is torch-only now; loading to a numpy array or a TensorFlow
  tensor is out of scope.

## Correctness gate

No library's timing counts until its decode is checked against a
`soundfile`-decoded reference. The tolerance is per format, because one
tolerance isn't right for all three:

- **WAV / FLAC** — sample-exact, within a tolerance set by the source bit
  depth: 1.5 LSB for 16-bit PCM (`atol = 1.5/32768`), `atol=1e-7` for
  float32. The 1-LSB allowance exists because normalisation conventions
  legitimately differ between libraries (dividing by 32767 vs. 32768); it
  still rejects downmixing, truncation, channel swaps, and any real gain
  error.
- **MP3 — a relaxed gate, and a weaker guarantee.** MP3 decoders disagree on
  encoder delay, so decoded lengths differ by roughly a thousand samples and
  samples never match bit-for-bit, even between two correct decoders. MP3 is
  therefore graded on **decoded duration within 50 ms of the reference** and
  **RMS level within 0.5 dB** after trimming both signals to their common
  length. A library that passes the MP3 gate is verified to be reading the
  right audio at roughly the right level — **not** verified sample-exact the
  way its WAV/FLAC results are. The report labels every MP3 result with this
  weaker gate rather than leaving the distinction implicit.

Bit depth remains a corpus axis (`PCM_16` vs. `FLOAT` WAV) because it changes
which library wins, not just the magnitude: FFmpeg-backed decoders are
markedly slower converting 16-bit PCM to float than reading float32 directly.
24-bit WAV was in an earlier version of this sweep and has been dropped: it
sits between the two remaining cases and added a third of the corpus for no
distinct finding.

## Caveats

- **Warm page cache.** Every trial (after the untimed warmup) reads a file
  that the OS has already cached. This measures decode speed, not storage
  I/O or cold-start latency.
- **Synthetic corpus.** The corpus is generated white noise, not music or
  speech. It is reproducible and free of licensing concerns, but a decoder's
  relative speed on real program material — silence runs, transients,
  particular sample-rate or bit-depth combinations in the wild — may differ.
- **Single machine, single process.** No multiprocessing, no DataLoader
  batching; that would measure a different thing (I/O-bound batch throughput)
  than per-call decode latency.

## Results

Measured on an Apple M-series Mac (macOS 26.5, arm64), Python 3.12, **median of 15 trials**
after one untimed warmup, warm page cache, stereo files. Full tables including mono and
per-cell spread are in [`results/report.md`](results/report.md).

![full decode](results/full.png)

![seek decode](results/seek.png)

512 records: 360 measured, 144 unsupported (a library that cannot read that container, or
cannot seek), 8 incorrect. Dispersion, measured as **IQR / median**, has a median of 0.064
and a 90th percentile of 0.215, so **treat differences below roughly 10% as ties**. The
interquartile range is used rather than max−min because the full range grows with the
trial count, which makes runs with different `--repeat` values incomparable.

### Full-file decode, stereo, 300 s, median ms

| format | soundfile | scipy | scipy_mmap | pedalboard | torchcodec | librosa | audioread | pydub |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wav_pcm16 | 53.84 | 26.98 | 26.23 | **20.96** | 51.12 | 49.15 | 76.42 | 28.88 |
| wav_float | 22.10 | 28.09 | 24.32 | **20.96** | 39.05 | 21.95 | — | 194.07 |
| flac | 167.35 | — | — | **155.07** | 191.98 | 184.29 | 272.25 | 198.82 |
| mp3 | **166.47** | — | — | 295.36 | 223.56 | 188.37 | 388.06 | 392.39 |

### Seek — decode 1 s from an offset, stereo, median ms

| format / duration | soundfile | scipy_mmap | pedalboard | torchcodec | librosa |
|---|---:|---:|---:|---:|---:|
| wav_pcm16, 300 s | 0.36 | 5.08 | **0.22** | 25.09 | 0.40 |
| wav_float, 300 s | 0.36 | 5.58 | **0.23** | 0.75 | 0.29 |
| flac, 300 s | 0.91 | — | **0.86** | 2.69 | 1.01 |
| mp3, 1 s | 0.81 | — | 1.35 | 1.28 | **0.76** |
| mp3, 300 s | 3.71 | — | 95.55 | **2.77** | 3.84 |

### Findings

**1. Seeking separates these libraries far more than full-file decoding.** For full reads
the good implementations sit within a factor of two of each other. For excerpts,
`soundfile`, `pedalboard` and `librosa` return a 1 s slice of a 300 s WAV in 0.2–0.4 ms —
*independent of file length* — while others scale with the file. If you build training
batches from excerpts, this is the table that matters. The previous version of this
benchmark never measured it.

**2. `pedalboard` does not seek in MP3.** Its excerpt time grows with file duration —
1.35, 3.87, 24.14, 95.55 ms at 1/10/60/300 s — the signature of decoding from the start
and discarding. It seeks properly in WAV and FLAC (flat at ~0.2 and ~0.9 ms). `torchcodec`
returns the same 300 s MP3 excerpt in 2.77 ms, **34x faster**, so the container is not the
obstacle.

**3. `torchcodec` pays a large penalty on 16-bit PCM.** Full-file it needs 10.42 ms for a
1 s `wav_pcm16` file where `soundfile` needs 0.32 ms. Seeking, it costs 25.09 ms on
`wav_pcm16` against 0.75 ms on `wav_float` — a **33x gap** for the same duration, channel
count and container. The cost tracks the s16 conversion path, not WAV itself. Given
`torchcodec` is otherwise the strongest MP3 seeker here, this is worth knowing before
standardising on 16-bit WAV for a torch pipeline.

**4. `audioread` cannot round-trip float32 audio.** Its output differs from the reference
by exactly 1.53e-05 = 1/65536 — the int16 quantisation step — on every float WAV file; it
decodes at 16-bit precision internally. Those 8 results are reported `incorrect` and their
timings withheld. No earlier version of this benchmark could detect this, because none
verified output at all.

**5. `librosa` tracks `soundfile` closely** on WAV (21.95 vs 22.10 ms on float), which is
expected: it delegates to it. It is not an independent decoder.

**6. `scipy` is WAV-only.** Its memmap variant is competitive for full reads but a poor
seeker here (5+ ms), since slicing a memmap still faults in pages through the OS.

## Development

```bash
uv sync --extra libs --group test
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

No optional library is required to run the test suite: a loader for a
library that isn't installed (or whose native components fail to load) is
recorded `available=False`, and the tests that need a real decode are
parametrized only over the libraries that came back available in the current
environment.

## Authors

@faroit, @hagenw

## Contribution

We encourage interested users to contribute to this repository in the issue
section and via pull requests. Particularly interesting are notifications of
new tools and new versions of existing packages.
