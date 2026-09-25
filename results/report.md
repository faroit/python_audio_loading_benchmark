# pabench results

Timings below are wall-clock, warm-page-cache decodes of a locally generated **synthetic white-noise corpus** (see `docs/refactor-design.md`): one untimed warmup per (library, file, benchmark) that also warms the OS page cache, followed by the repeated, timed trials each cell's spread is drawn from. They describe decode speed against this machine's page cache and this synthetic corpus, not cold-disk I/O and not real program material.

## Platform

| Field | Value |
| --- | --- |
| OS | Linux-6.8.0-136-generic-x86_64-with-glibc2.39 |
| Machine | x86_64 |
| Python | 3.12.3 |
| PyTorch | 2.14.0+cu130 |
| FFmpeg | ffmpeg version n8.0.1-48-g0592be14ff-20260116 Copyright (c) 2000-2025 the FFmpeg developers |
| DYLD_FALLBACK_LIBRARY_PATH | - |

## Libraries

| Library | Version | Available | Seek | Notes | Error |
| --- | --- | --- | --- | --- | --- |
| soundfile | 0.14.0 | yes | yes | the correctness reference every other loader is checked against | - |
| librosa | - | no | n/a | - | ModuleNotFoundError: No module named 'librosa' |
| scipy | - | no | n/a | WAV only; scipy.io.wavfile has no seek API | ModuleNotFoundError: No module named 'scipy' |
| scipy_mmap | - | no | n/a | WAV only, memory-mapped; scipy's mmap mode cannot open 24-bit ('3-byte container') WAV, which surfaces as a decode error for that subtype | ModuleNotFoundError: No module named 'scipy' |
| audioread | - | no | n/a | full-file only, no seek API; audioread always yields 16-bit PCM buffers, so it cannot pass the exact gate for 24-bit or float32 WAV sources | ModuleNotFoundError: No module named 'audioread' |
| pedalboard | - | no | n/a | - | ModuleNotFoundError: No module named 'pedalboard' |
| torchcodec | - | no | n/a | imports cleanly even when its native FFmpeg bindings can't load; only the decode smoke test below catches that (see docs/refactor-design.md) | ModuleNotFoundError: No module named 'torchcodec' |
| audiolab | - | no | n/a | - | ModuleNotFoundError: No module named 'audiolab' |
| audiosample | - | no | n/a | integer-PCM WAV only: float WAV, FLAC and MP3 go through its PyAV path, which is incompatible with PyAV 18 (Flags.FAST_SEEK) | ModuleNotFoundError: No module named 'audiosample' |
| sphn | - | no | n/a | MP3 decode returns a different frame count than soundfile's reference (decoder-delay disagreement); graded by the relaxed MP3 gate | ModuleNotFoundError: No module named 'sphn' |

**MP3 is graded by a relaxed gate** (decoded duration within 50 ms of the reference, RMS level within 0.5 dB on the common length) rather than the sample-exact gate WAV and FLAC are held to (1.5 LSB for integer PCM, `atol=1e-7` for float32), because MP3 decoders disagree on encoder delay and never match sample-for-sample. MP3 results therefore carry a weaker correctness guarantee than the WAV/FLAC results in this report.

## wav_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.28 ms [0.26–0.45] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1 | 2 | 0.43 ms [0.42–0.80] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 1 | 1.23 ms [1.20–1.27] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 2 | 2.80 ms [2.71–2.88] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 1 | 6.72 ms [6.55–7.07] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 2 | 16.80 ms [16.55–17.73] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 1 | 41.59 ms [41.11–42.14] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 2 | 124.71 ms [124.32–125.20] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

## wav_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.34 ms [0.32–0.35] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1 | 2 | 0.48 ms [0.48–0.49] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 1 | 0.33 ms [0.33–0.35] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 2 | 0.49 ms [0.48–0.50] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 1 | 0.34 ms [0.33–0.35] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 2 | 0.51 ms [0.50–0.53] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 1 | 0.35 ms [0.33–0.39] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 2 | 0.51 ms [0.49–0.55] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.69 ms [0.69–0.70] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1 | 2 | 1.16 ms [1.15–1.16] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 1 | 5.19 ms [5.18–5.27] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 2 | 10.00 ms [9.98–10.03] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 1 | 30.28 ms [30.02–31.68] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 2 | 60.34 ms [60.14–62.53] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 1 | 159.63 ms [159.39–161.20] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 2 | 341.20 ms [340.89–341.45] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

## flac_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.75 ms [0.74–0.76] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1 | 2 | 1.23 ms [1.23–1.84] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 1 | 0.87 ms [0.86–0.88] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 2 | 1.40 ms [1.39–1.40] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 1 | 0.83 ms [0.82–0.83] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 2 | 1.46 ms [1.46–1.48] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 1 | 0.84 ms [0.83–0.89] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 2 | 1.47 ms [1.45–1.55] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.96 ms [0.96–1.12] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1 | 2 | 1.52 ms [1.51–1.54] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 1 | 7.70 ms [7.66–7.76] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 2 | 13.00 ms [12.89–13.13] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 1 | 45.12 ms [44.98–45.33] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 2 | 77.07 ms [76.84–78.25] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 1 | 231.18 ms [230.91–232.41] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 2 | 423.50 ms [422.42–424.66] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

## mp3 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.03 ms [1.03–1.04] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1 | 2 | 1.59 ms [1.57–1.62] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 1 | 1.30 ms [1.30–1.32] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 10 | 2 | 1.81 ms [1.79–1.85] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 1 | 2.59 ms [2.56–2.62] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 60 | 2 | 2.57 ms [2.56–2.65] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 1 | 4.17 ms [4.16–4.19] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 300 | 2 | 5.03 ms [5.01–5.08] | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

## Unavailable and incorrect libraries

Unavailable libraries:

- **audiolab**: ModuleNotFoundError: No module named 'audiolab'
- **audioread**: ModuleNotFoundError: No module named 'audioread'
- **audiosample**: ModuleNotFoundError: No module named 'audiosample'
- **librosa**: ModuleNotFoundError: No module named 'librosa'
- **pedalboard**: ModuleNotFoundError: No module named 'pedalboard'
- **scipy**: ModuleNotFoundError: No module named 'scipy'
- **scipy_mmap**: ModuleNotFoundError: No module named 'scipy'
- **sphn**: ModuleNotFoundError: No module named 'sphn'
- **torchcodec**: ModuleNotFoundError: No module named 'torchcodec'

## Measurement noise

Across 48 `ok` measurements, the observed spread `(max - min) / median` ranges from 0.1% to 10.7%, with a median of 0.7%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings.
