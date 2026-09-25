# pabench — Python audio-loading benchmark

`pabench` measures how fast eleven Python audio-decoding libraries can get a WAV,
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
uv sync
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
| [`audioread`](https://github.com/beetbox/audioread) | yes | no | no native seek; always yields 16-bit PCM, so it cannot pass the exact gate against a float32 or 24-bit source |
| [`pedalboard`](https://github.com/spotify/pedalboard) | yes | yes | `AudioFile.seek` + `.read` |
| [`torchcodec`](https://github.com/pytorch/torchcodec) | yes | yes | `get_samples_played_in_range`; imports cleanly even when its native FFmpeg bindings can't load, so `pabench` smoke-tests a real decode before trusting it as available |
| [`audiolab`](https://github.com/pengzhendong/audiolab) | yes | yes | PyAV-backed; `load_audio(path, dtype=np.float32)`, `offset=`/`duration=` for seek |
| [`audiosample`](https://github.com/deepdub-ai/audiosample) | **WAV only** | yes | its compressed-format path goes through PyAV and is incompatible with PyAV 18 (`Flags.FAST_SEEK`); slice by seconds (`a[start:stop]`) for seek |
| [`sphn`](https://github.com/kyutai-labs/sphn) | yes | yes | Rust-backed; `sphn.read(path, start_sec=, duration_sec=)` for seek |

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
- **`stempeg`** — it shells out to an `ffmpeg` subprocess per call, so every
  cell measured process startup rather than decoding: a floor of roughly
  90–250 ms in every format and duration, including a 1 s file. It is a
  container-aware tool rather than a loader, and leaving it in only
  compressed the log axis for everything else.
- **`pydub`** — consistently at the back of the field, and by margins far
  outside the noise floor: at 300 s stereo it needed 165.81 ms on float WAV
  where the leaders needed ~21 ms, and 336.08 ms on MP3 against ~170 ms. It
  also has no seek API, so it could never appear in the seek benchmark,
  which is where this tool's interesting results are.

Both `stempeg` and `pydub` were measured before being removed; the numbers
above are from this benchmark, not from reputation. Adding either back is a
matter of restoring its probe in `pabench/loaders.py` and its entry in the
`libs` optional-dependency group.

## Where the corpus lives

The corpus is generated and read from `--corpus-dir` (default `corpus/`), so pointing it
at another disk benchmarks that disk's files:

```shell
uv run pabench all --corpus-dir /Volumes/fast-nvme/pabench-corpus
```

`PABENCH_CORPUS_DIR` supplies the default, so it does not have to be repeated across
subcommands:

```shell
export PABENCH_CORPUS_DIR=/Volumes/fast-nvme/pabench-corpus
uv run pabench gen && uv run pabench run && uv run pabench report --out results/report.md
```

Note that this changes which disk holds the files, not what is being measured. Every
timing runs against a **warm page cache** by design — an untimed warmup call precedes the
timed trials — because the question here is how fast each library *decodes*, not how fast
the storage is. Cold-cache numbers would be dominated by I/O and would mostly rank disks.

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

> **These numbers are from one developer laptop that was in normal use during the run.**
> Treat them as indicative. Re-run on your own hardware with `uv run pabench all`; the
> report regenerates itself and the findings below state which effects reproduce.

Apple M-series Mac (macOS 26.5, arm64), Python 3.12, median of 15 trials after one untimed
warmup, warm page cache, stereo. Full tables including mono and per-cell spread are in
[`results/report.md`](results/report.md).

![full decode](results/full.png)

![seek decode](results/seek.png)

480 records over ten libraries: 352 measured, 128 unsupported, **0 incorrect, 0 errored** —
every library that ran decoded correctly. Dispersion (IQR / median) has a median of 0.063
and a 90th percentile of 0.201, so **treat differences below roughly 10% as ties**.

### Full-file decode, stereo, 300 s, median ms (lower is better)

| format | ranking |
|---|---|
| wav_pcm16 | pedalboard 21.56 · scipy_mmap 22.97 · audiosample 23.95 · scipy 25.67 · librosa 49.59 · soundfile 52.07 · torchcodec 52.37 · audiolab 60.20 · audioread 76.94 · sphn 144.38 |
| flac | pedalboard 158.40 · librosa 172.43 · soundfile 173.89 · audiolab 183.75 · torchcodec 196.09 · audioread 279.09 · sphn 323.73 |
| mp3 | librosa 171.90 · soundfile 176.60 · audiolab 190.97 · torchcodec 221.85 · pedalboard 294.20 · audioread 387.96 · sphn 397.54 |

### Seek — decode 1 s from an offset, stereo, 300 s file, median ms

| format | ranking |
|---|---|
| wav_pcm16 | audiosample 0.21 · pedalboard 0.22 · librosa 0.37 · soundfile 0.45 · sphn 0.50 · audiolab 0.57 · scipy_mmap 0.73 · **torchcodec 24.96** |
| flac | librosa 0.87 · pedalboard 0.89 · soundfile 0.99 · audiolab 1.03 · sphn 1.30 · torchcodec 2.64 |
| mp3 | sphn 2.10 · torchcodec 2.94 · audiolab 3.70 · librosa 3.73 · soundfile 3.80 · **pedalboard 94.51** |

### Findings

**1. Seeking separates these libraries far more than full-file decoding.** For full reads
the good implementations sit within a factor of two or three; for excerpts the spread is
two orders of magnitude. If you assemble training batches from excerpts, the seek table is
the one that matters, and it is the one earlier versions of this benchmark never measured.

**2. `pedalboard` does not seek in MP3.** Its excerpt time grows with file duration —
1.34, 3.88, 24.32, 94.51 ms at 1/10/60/300 s — the signature of decoding from the start and
discarding. It seeks properly in WAV and FLAC (flat at ~0.22 and ~0.89 ms) and is the
fastest full-file reader in two of three formats, so this is specifically an MP3 seek
limitation. **Reproduced across three independent runs** (92.0, 94.5, 95.6 ms), while
`sphn` and `torchcodec` return the same excerpt in 2–3 ms.

**3. `torchcodec` is far slower seeking uncompressed 16-bit WAV than compressed audio.**
24.96 ms for a 1 s excerpt of a 300 s WAV, against 2.64 ms for FLAC and 2.94 ms for MP3 —
roughly 10x slower on the *easier* format, and flat from 10 s onward (11.91, 24.93, 24.99,
24.96 ms). **Reproduced across runs** (24.20, 24.96 ms). Worth knowing before standardising
on 16-bit WAV for a torch pipeline.

**4. `sphn`'s seek is strong on compressed formats but its full-file reads are slow here**
(144 ms for a 300 s WAV, the slowest in the field). Its full-file figures were also the
least stable across runs — the same FLAC cell measured 119 ms, 324 ms and ~170 ms on
repeat — so **treat sphn's full-file numbers as unreliable on this machine** and re-measure
before drawing conclusions. Its seek results were consistent.

**5. `librosa` tracks `soundfile` closely**, as expected — it delegates to it. It is not an
independent decoder, and the two should not be read as corroborating each other.

**6. `scipy` is WAV-only.** Its memmap variant is competitive for full reads but a mediocre
seeker (0.73 ms), since slicing a memmap still faults pages in through the OS.

**7. Nothing failed the correctness gate in this run.** An earlier sweep that included
float32 WAV caught `audioread` decoding at 16-bit precision internally (off by exactly
1/65536) and `audiosample` raising on its PyAV path. Both are invisible to the current
corpus, which is 16-bit only — the limitations still exist, the benchmark just no longer
exercises them.

## Development

```bash
uv sync --group test
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
