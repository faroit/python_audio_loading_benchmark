# Task 2 report — canonical conversion and the correctness gate

**Delete before the branch is finished** (per instructions, this file is for the user
only, like `docs/task-1-report.md`).

## What was implemented

- `pabench/canonical.py`:
  - `Layout = Literal["channels_first", "frames_first"]`.
  - `_to_numpy(data)` (private helper): coerces `torch.Tensor` (via
    `.detach().cpu().numpy()`), `np.ndarray` (passed through), any object exposing a
    callable `.numpy()`, or falls back to `np.asarray`.
  - `to_tensor(data, layout)`: validates `layout` first (raises `ValueError` naming the
    bad value before looking at `data` at all); 1-D input gets a leading axis
    (`(1, frames)`) since a single channel has no first-vs-channels ambiguity;
    `frames_first` 2-D input is transposed; anything with more than 2 dimensions raises
    `ValueError`. The array is finished with a single `np.ascontiguousarray(arr,
    dtype=np.float32)`, which is a no-op (no copy) when `arr` is already a contiguous
    float32 array, and copies only when the transpose or a dtype cast requires it — this
    is the "no needless copy" instruction from the task, and `torch.from_numpy` is then
    called on a guaranteed-contiguous, guaranteed-supported-dtype array so it never
    raises.
- `pabench/verify.py`:
  - `VerifyResult` — frozen dataclass `ok: bool`, `reason: str | None`, `gate: str`.
  - `compare(reference, candidate, fmt)`: dispatches to the exact gate for
    wav/flac (keyed by `fmt.subtype`) or the relaxed gate for `fmt.container == "mp3"`.
  - Exact gate: shape mismatch is an explicit, separate failure reason (so a downmix or
    truncation that changes shape is rejected on that armature rather than by luck of
    `np.allclose`'s broadcasting); otherwise `np.allclose(candidate, reference, rtol,
    atol)` with the fixed per-subtype tolerances from the design doc. Unknown subtype
    raises `ValueError` naming the subtype.
  - Relaxed (mp3) gate: length check first (fails if the frame-count difference exceeds
    50 ms of samples), then trims both to the common length and compares RMS in dB,
    guarding both a near-zero reference RMS (raises would otherwise come from
    `math.log10`) and a near-zero candidate RMS (treated as `-inf` dB, which correctly
    fails the gate rather than raising).
- `tests/test_canonical.py` (12 tests): both layouts, 1-D promotion under both layouts,
  values-survive-transpose with an asymmetric fixture, dtype, contiguity, torch tensor
  input, an object exposing `.numpy()`, unknown layout rejected, 3-D rejected.
- `tests/test_verify.py` (18 tests, 2 of which are `@pytest.mark.parametrize` x2):
  identical passes; downmix/truncation/channel-swap rejected; the 32768/32767 rescale
  passes on PCM_16; a 2x gain fails; PCM_16/PCM_24 boundary just-below passes and
  just-above fails; FLOAT gate; FLAC uses the PCM_16-style tolerance; mp3 delay-shift
  (length only, via truncation) passes; mp3 6 dB level error fails; mp3 length
  difference beyond 50 ms fails; mp3 zero-RMS reference doesn't raise; `gate` is
  `"relaxed"` for mp3 and `"exact"` for wav; unknown format raises `ValueError` naming
  it.

## Commands and output

```
$ uv run pytest tests/test_canonical.py tests/test_verify.py -q
..............................                                           [100%]
30 passed in 0.82s

$ uv run pytest -q
............................................                             [100%]
44 passed in 0.63s
```

(44 = Task 1's 14 + Task 2's 30.)

Tests were written and run first against the not-yet-existing modules to confirm they
failed for the right reason (`ModuleNotFoundError: No module named 'pabench.canonical'`
/ `'pabench.verify'`), before any implementation code was written.

`ruff` is not installed in this environment (`uv run ruff check ...` → "Failed to
spawn: `ruff`: No such file or directory"; it isn't in the `test` dependency-group or a
base dependency), so line length was checked manually instead: no line in the four new
files exceeds 100 characters (the repo's `[tool.ruff] line-length = 100`).

## Ambiguity found, and how it was resolved

The plan's Task 2 signature is fixed as `compare(reference, candidate, fmt)` — no
`sample_rate` parameter — but the mp3 relaxed gate is specified as "length difference
must be within 50 ms of samples at the reference sample rate". Since `Fmt` (Task 1's
type, not redefined here) carries only `container`/`subtype`, not a sample rate, and the
whole corpus is generated at a single fixed rate, `verify.py` imports `SAMPLE_RATE`
(44100) from `pabench.corpus` and uses that as "the reference sample rate" for the 50 ms
→ samples conversion, rather than adding an out-of-spec parameter to `compare`. This
matches the corpus design (`SAMPLE_RATE = 44100` is the only sample rate the corpus
generator ever writes) and keeps the signature exactly as specified.

## Deviations

None from the specified public API (`Layout`, `to_tensor`, `VerifyResult`, `compare`).
Everything else added (the `_to_numpy`/`_to_array` helpers, the tolerance/threshold
module-level constants) is private implementation detail, not part of the task's
required surface.

## Note

Per the same instruction as `docs/task-1-report.md`: this file (and
`docs/task-1-report.md`) should be deleted before the branch is finished — these are
scratch reports for the human reviewer during development, not documentation meant to
ship.
