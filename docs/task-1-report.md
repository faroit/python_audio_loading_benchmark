# Task 1 report — scaffolding and corpus

**Delete before the branch is finished** (per instructions, this file is for the user only).

## What was implemented

- `pyproject.toml`: `pabench` project, `requires-python = ">=3.12"`, console script
  `pabench = "pabench.cli:console_main"` (module doesn't exist yet — deliberate, per
  instructions), deps `numpy`/`soundfile`/`torch`/`matplotlib`, optional-dependency group
  `libs` (`librosa`, `scipy`, `pydub`, `audioread`, `stempeg`, `pedalboard`, `torchcodec`),
  dependency-group `test` (`pytest`), `hatchling` build backend, `[tool.ruff] line-length =
  100`.
- `.gitignore`: added a `pabench` section with `corpus/` and `results/*.json`.
- `pabench/__init__.py`: docstring + `__version__`.
- `pabench/corpus.py`: `Fmt` (frozen dataclass, `container`/`subtype`/`key` property),
  `FORMATS` (5 entries: `wav_pcm16`, `wav_pcm24`, `wav_float`, `flac_pcm16`, `mp3`),
  `CorpusSpec` (frozen dataclass, `duration_s`/`channels`/`fmt`/`sample_rate=44100`,
  `filename`/`frames` properties), `DEFAULT_SPECS` (40 = 4 durations x 2 channels x 5
  formats), `generate()`, `corpus_files()`, `ffmpeg_available()`.
- `tests/test_corpus.py`: 14 tests covering everything the plan's Task 1 test list asks
  for (spec axis values, unique filenames, WAV/FLAC properties matching request,
  determinism, single-file-matches-full-sweep, peak amplitude, MP3 written/non-empty
  guarded by `ffmpeg_available()`, `corpus_files` raising `FileNotFoundError` mentioning
  `pabench gen`).
- Deleted `requirements.txt`, `Dockerfile`, `generate_audio.sh`, `run.sh` via `git rm`.

Also added (not explicitly requested, judged necessary/harmless):
- `[tool.hatch.build.targets.wheel] packages = ["pabench"]` in `pyproject.toml` — hatchling
  auto-detected this fine without it (project name normalizes to `pabench`, matching the
  package dir), but I left it explicit for robustness since it's a flat, non-`src/` layout.
- `.python-version` pinning `3.12` (created by `uv python pin 3.12`) and `uv.lock`, both
  committed as normal `uv` project artifacts.

## Commands run and output (abridged)

```
$ uv python pin 3.12
Pinned `.python-version` to `3.12`

$ uv sync --group test
Resolved 80 packages in 279ms
Building pabench @ file:///.../python_audio_loading_benchmark
Installed 29 packages ...

$ uv run pytest tests/test_corpus.py -q      # before implementing corpus.py
ModuleNotFoundError: No module named 'pabench.corpus'
1 error in 0.81s

$ uv run pytest -v                           # after implementing
14 passed in 0.13s

$ uv run --with ruff ruff check pabench tests --line-length 100
All checks passed!
```

Final `uv run pytest` (repo root, full suite): **14 passed**.

## Deviations, ambiguities, and judgment calls

1. **`Fmt.key` derivation rule wasn't spelled out** — the plan gives one example
   (`wav_pcm16`) but not the general rule for `wav_float` or `mp3`. I implemented:
   lowercase the subtype, strip underscores, and if that equals the container name use the
   container alone (so `mp3`/`MP3` -> `"mp3"`, not `"mp3_mp3"`); otherwise
   `f"{container}_{subtype_part}"`. This gives `wav_pcm16`, `wav_pcm24`, `wav_float`,
   `flac_pcm16`, `mp3`, matching the plan's one given example and design doc's five listed
   formats.
2. **Filename scheme wasn't specified** — I chose `"{duration_s}s_{channels}ch_{fmt.key}.
   {container}"` (e.g. `10s_2ch_wav_pcm16.wav`, `300s_1ch_mp3.mp3`). Any later task that
   parses filenames back into specs should use `corpus_files`/`CorpusSpec.filename` rather
   than re-deriving the pattern, since this is my own invented convention, not a spec
   requirement.
3. **Per-spec seeding mechanism** — the plan says "reseed per spec" but not how to combine
   the sweep `seed` with a spec's identity. I used
   `int.from_bytes(sha256(f"{seed}:{duration_s}:{channels}:{sample_rate}:{fmt.key}"), "big")`
   fed to `numpy.random.default_rng`. I deliberately avoided Python's builtin `hash()`
   because string hashing is randomized per-process (`PYTHONHASHSEED`) unless disabled,
   which would have broken the determinism requirement across separate `uv run` processes.
4. **Noise distribution** — plan says "peak 0.5", not a distribution. I used
   `rng.uniform(-0.5, 0.5, ...)`, which bounds the peak at exactly 0.5 by construction
   (simpler and more robust for the "peak within 0.5" test than scaling a Gaussian).
5. **MP3 intermediate WAV subtype** — plan doesn't specify what subtype the temp WAV before
   ffmpeg encoding should use. I used `PCM_16` for the intermediate write; ffmpeg then
   encodes to MP3 at `-b:a 192k` as specified. This only affects the temp file, which is
   deleted immediately after encoding.
6. Nothing in the plan's Task 1 section appeared wrong; the design doc and plan agreed on
   all corpus axes (44100 Hz; durations 1/10/60/300; mono/stereo; wav PCM_16/PCM_24/FLOAT,
   flac 16-bit, mp3 CBR 192k).

## Not done (intentionally out of scope for Task 1)

- No `pabench/cli.py`, `loaders.py`, `canonical.py`, `verify.py`, `timing.py`, `run.py`,
  `report.py` — these belong to Tasks 2-5.
- `corpus/` was never created at the repo root; all corpus tests use `tmp_path`.
