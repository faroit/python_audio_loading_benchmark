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
| librosa | 1.0.0 | yes | yes | - | - |
| scipy | 1.18.1 | yes | no | WAV only; scipy.io.wavfile has no seek API | - |
| scipy_mmap | 1.18.1 | yes | yes | WAV only, memory-mapped; scipy's mmap mode cannot open 24-bit ('3-byte container') WAV, which surfaces as a decode error for that subtype | - |
| audioread | 3.1.0 | yes | no | full-file only, no seek API; audioread always yields 16-bit PCM buffers, so it cannot pass the exact gate for 24-bit or float32 WAV sources | - |
| pedalboard | 0.9.25 | yes | yes | - | - |
| torchcodec | 0.16.0+cu130 | yes | yes | imports cleanly even when its native FFmpeg bindings can't load; only the decode smoke test below catches that (see docs/refactor-design.md) | - |
| audiolab | 0.5.2 | yes | yes | - | - |
| audiosample | 2.2.12 | yes | yes | integer-PCM WAV only: float WAV, FLAC and MP3 go through its PyAV path, which is incompatible with PyAV 18 (Flags.FAST_SEEK) | - |
| sphn | 0.2.1 | yes | yes | MP3 decode returns a different frame count than soundfile's reference (decoder-delay disagreement); graded by the relaxed MP3 gate | - |

**MP3 is graded by a relaxed gate** (decoded duration within 50 ms of the reference, RMS level within 0.5 dB on the common length) rather than the sample-exact gate WAV and FLAC are held to (1.5 LSB for integer PCM, `atol=1e-7` for float32), because MP3 decoders disagree on encoder delay and never match sample-for-sample. MP3 results therefore carry a weaker correctness guarantee than the WAV/FLAC results in this report.

## wav_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.37 ms [0.36–0.41] | 0.37 ms [0.36–0.38] | 0.23 ms [0.23–0.25] | 0.30 ms [0.29–0.32] | 0.40 ms [0.38–0.41] | 0.18 ms [0.18–0.20] | 6.67 ms [6.62–6.84] | 0.58 ms [0.57–0.60] | 0.22 ms [0.22–0.23] | 0.24 ms [0.24–0.25] |
| 1 | 2 | 0.55 ms [0.53–0.72] | 0.54 ms [0.53–0.55] | 0.33 ms [0.32–0.34] | 0.39 ms [0.38–0.40] | 0.54 ms [0.52–0.58] | 0.23 ms [0.23–0.24] | 7.91 ms [7.80–8.05] | 0.76 ms [0.72–0.79] | 0.32 ms [0.32–0.33] | 0.39 ms [0.38–0.40] |
| 10 | 1 | 1.43 ms [1.39–1.47] | 1.44 ms [1.40–1.46] | 0.77 ms [0.76–0.81] | 0.78 ms [0.76–0.83] | 2.15 ms [2.12–2.21] | 0.65 ms [0.64–0.67] | 9.62 ms [9.52–9.72] | 2.05 ms [2.01–2.10] | 0.62 ms [0.61–0.64] | 1.66 ms [1.65–1.68] |
| 10 | 2 | 3.29 ms [3.27–3.36] | 3.31 ms [3.21–3.36] | 1.76 ms [1.75–1.82] | 1.77 ms [1.75–1.82] | 3.73 ms [3.71–3.76] | 1.25 ms [1.24–1.28] | 11.24 ms [11.15–11.41] | 3.80 ms [3.75–3.87] | 1.58 ms [1.57–1.61] | 3.00 ms [2.99–3.10] |
| 60 | 1 | 7.21 ms [7.14–7.29] | 7.27 ms [7.15–7.38] | 3.76 ms [3.68–3.87] | 3.66 ms [3.64–3.76] | 12.05 ms [11.97–12.12] | 3.44 ms [3.43–3.56] | 18.26 ms [18.05–18.49] | 9.59 ms [9.53–9.70] | 2.90 ms [2.89–2.95] | 9.51 ms [9.48–9.54] |
| 60 | 2 | 18.93 ms [18.82–19.11] | 19.07 ms [18.86–19.25] | 10.96 ms [10.74–11.06] | 10.76 ms [10.46–10.82] | 22.28 ms [22.21–22.35] | 7.03 ms [6.92–7.27] | 29.04 ms [28.85–29.98] | 19.61 ms [19.44–19.75] | 8.88 ms [8.85–8.93] | 19.37 ms [19.26–19.49] |
| 300 | 1 | 43.79 ms [43.36–44.18] | 42.94 ms [42.81–43.22] | 31.73 ms [31.67–31.86] | 28.15 ms [28.10–28.30] | 72.79 ms [72.50–73.16] | 24.62 ms [24.13–24.82] | 86.97 ms [78.33–100.90] | 53.61 ms [52.81–53.91] | 20.40 ms [19.93–20.66] | 78.93 ms [78.67–79.12] |
| 300 | 2 | 125.49 ms [125.12–125.78] | 125.29 ms [124.78–125.97] | 108.14 ms [107.82–108.62] | 95.08 ms [94.88–97.00] | 168.24 ms [167.83–169.30] | 51.45 ms [51.32–51.58] | 154.27 ms [149.14–166.33] | 111.29 ms [110.72–111.67] | 98.58 ms [98.21–98.87] | 150.91 ms [150.53–151.27] |

## wav_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.45 ms [0.43–0.56] | 0.37 ms [0.35–0.39] | unsupported | 0.31 ms [0.30–0.31] | unsupported | 0.18 ms [0.18–0.19] | 6.65 ms [6.52–6.74] | 0.56 ms [0.54–0.57] | 0.24 ms [0.24–0.24] | 0.22 ms [0.22–0.23] |
| 1 | 2 | 0.61 ms [0.60–0.64] | 0.54 ms [0.54–0.56] | unsupported | 0.40 ms [0.39–0.42] | unsupported | 0.24 ms [0.23–0.25] | 7.95 ms [7.85–8.14] | 0.73 ms [0.72–0.77] | 0.34 ms [0.33–0.35] | 0.36 ms [0.35–0.41] |
| 10 | 1 | 0.45 ms [0.43–0.54] | 0.38 ms [0.36–0.38] | unsupported | 0.31 ms [0.31–0.33] | unsupported | 0.18 ms [0.18–0.19] | 8.00 ms [7.87–8.12] | 0.56 ms [0.55–0.58] | 0.24 ms [0.24–0.25] | 0.23 ms [0.23–0.24] |
| 10 | 2 | 0.62 ms [0.60–0.64] | 0.54 ms [0.53–0.56] | unsupported | 0.40 ms [0.38–0.42] | unsupported | 0.24 ms [0.24–0.25] | 8.07 ms [7.97–8.16] | 0.72 ms [0.71–0.74] | 0.34 ms [0.33–0.35] | 0.35 ms [0.34–0.35] |
| 60 | 1 | 0.44 ms [0.44–0.59] | 0.38 ms [0.36–0.39] | unsupported | 0.32 ms [0.30–0.33] | unsupported | 0.18 ms [0.18–0.19] | 7.91 ms [7.81–8.03] | 0.55 ms [0.55–0.56] | 0.24 ms [0.24–0.25] | 0.23 ms [0.22–0.23] |
| 60 | 2 | 0.62 ms [0.59–0.63] | 0.54 ms [0.53–0.55] | unsupported | 0.40 ms [0.39–0.41] | unsupported | 0.24 ms [0.24–0.25] | 8.26 ms [8.20–8.32] | 0.72 ms [0.70–0.76] | 0.35 ms [0.34–0.35] | 0.37 ms [0.36–0.44] |
| 300 | 1 | 0.45 ms [0.44–0.47] | 0.37 ms [0.36–0.39] | unsupported | 0.32 ms [0.31–0.33] | unsupported | 0.19 ms [0.18–0.19] | 8.05 ms [7.99–8.11] | 0.56 ms [0.53–0.60] | 0.25 ms [0.25–0.27] | 0.23 ms [0.22–0.24] |
| 300 | 2 | 0.61 ms [0.61–0.63] | 0.54 ms [0.54–0.56] | unsupported | 0.40 ms [0.39–0.42] | unsupported | 0.24 ms [0.23–0.24] | 8.17 ms [8.04–8.23] | 0.74 ms [0.70–0.75] | 0.34 ms [0.32–0.35] | 0.36 ms [0.35–0.36] |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.81 ms [0.81–0.84] | 0.78 ms [0.77–0.79] | unsupported | unsupported | 18.02 ms [17.12–18.80] | 0.45 ms [0.45–0.46] | 1.39 ms [1.35–1.41] | 1.03 ms [1.02–1.04] | unsupported | 0.50 ms [0.49–0.51] |
| 1 | 2 | 1.31 ms [1.30–1.60] | 1.26 ms [1.25–1.27] | unsupported | unsupported | 18.50 ms [17.86–19.70] | 0.80 ms [0.79–0.81] | 1.92 ms [1.87–1.93] | 1.55 ms [1.51–1.56] | unsupported | 0.92 ms [0.90–0.92] |
| 10 | 1 | 5.36 ms [5.33–5.41] | 5.36 ms [5.33–5.40] | unsupported | unsupported | 21.54 ms [20.66–22.48] | 3.31 ms [3.27–3.36] | 7.04 ms [7.02–7.11] | 6.06 ms [6.03–6.09] | unsupported | 3.86 ms [3.84–3.88] |
| 10 | 2 | 10.52 ms [10.47–10.79] | 10.45 ms [10.42–10.51] | unsupported | unsupported | 25.75 ms [24.49–26.48] | 6.89 ms [6.87–7.01] | 12.09 ms [12.05–12.17] | 11.09 ms [11.07–11.11] | unsupported | 7.68 ms [7.67–7.88] |
| 60 | 1 | 30.72 ms [30.66–30.98] | 30.73 ms [30.64–31.02] | unsupported | unsupported | 34.82 ms [33.19–37.59] | 19.39 ms [19.30–19.50] | 37.67 ms [37.54–37.86] | 33.43 ms [33.28–33.50] | unsupported | 22.41 ms [22.36–22.50] |
| 60 | 2 | 62.08 ms [61.94–62.21] | 62.04 ms [61.89–62.28] | unsupported | unsupported | 51.00 ms [49.76–52.40] | 40.82 ms [40.63–41.10] | 67.39 ms [67.28–67.69] | 63.23 ms [63.16–63.49] | unsupported | 47.25 ms [47.21–47.37] |
| 300 | 1 | 161.29 ms [160.01–161.59] | 160.71 ms [159.83–161.84] | unsupported | unsupported | 109.47 ms [107.12–110.97] | 104.69 ms [104.43–105.07] | 197.89 ms [197.21–198.05] | 172.76 ms [172.60–173.32] | unsupported | 144.83 ms [144.76–145.35] |
| 300 | 2 | 340.71 ms [340.22–341.04] | 340.72 ms [340.24–341.14] | unsupported | unsupported | 230.96 ms [228.12–233.29] | 221.15 ms [220.64–221.62] | 364.86 ms [364.24–365.40] | 328.66 ms [328.45–329.13] | unsupported | 291.70 ms [291.45–292.51] |

## flac_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.89 ms [0.88–0.92] | 0.78 ms [0.77–0.88] | unsupported | unsupported | unsupported | 0.46 ms [0.45–0.46] | 1.36 ms [1.35–1.47] | 1.00 ms [0.97–1.01] | unsupported | 0.47 ms [0.47–0.49] |
| 1 | 2 | 1.39 ms [1.36–1.50] | 1.26 ms [1.24–1.26] | unsupported | unsupported | unsupported | 0.81 ms [0.80–0.82] | 1.90 ms [1.86–1.93] | 1.52 ms [1.49–1.53] | unsupported | 0.89 ms [0.88–0.90] |
| 10 | 1 | 1.02 ms [1.01–1.03] | 0.95 ms [0.94–0.96] | unsupported | unsupported | unsupported | 0.49 ms [0.48–0.50] | 2.22 ms [2.18–2.23] | 1.13 ms [1.11–1.14] | unsupported | 0.47 ms [0.46–0.47] |
| 10 | 2 | 1.56 ms [1.55–1.58] | 1.47 ms [1.47–1.49] | unsupported | unsupported | unsupported | 0.90 ms [0.88–0.90] | 4.09 ms [4.04–4.12] | 1.69 ms [1.65–1.71] | unsupported | 0.95 ms [0.95–0.96] |
| 60 | 1 | 0.96 ms [0.95–0.99] | 0.89 ms [0.88–0.90] | unsupported | unsupported | unsupported | 0.50 ms [0.50–0.54] | 2.18 ms [2.13–2.20] | 1.07 ms [1.07–1.09] | unsupported | 0.52 ms [0.51–0.53] |
| 60 | 2 | 1.61 ms [1.58–1.62] | 1.53 ms [1.51–1.55] | unsupported | unsupported | unsupported | 0.93 ms [0.91–0.94] | 3.66 ms [3.60–3.70] | 1.74 ms [1.69–1.75] | unsupported | 1.02 ms [1.01–1.04] |
| 300 | 1 | 0.95 ms [0.94–0.98] | 0.88 ms [0.88–0.90] | unsupported | unsupported | unsupported | 0.50 ms [0.49–0.50] | 2.21 ms [2.16–2.25] | 1.06 ms [1.04–1.09] | unsupported | 0.54 ms [0.53–0.57] |
| 300 | 2 | 1.60 ms [1.60–1.64] | 1.52 ms [1.50–1.53] | unsupported | unsupported | unsupported | 0.93 ms [0.92–0.94] | 4.14 ms [4.12–4.16] | 1.73 ms [1.69–1.74] | unsupported | 1.01 ms [1.00–1.02] |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.11 ms [1.08–1.17] | 1.07 ms [1.07–1.09] | unsupported | unsupported | 18.65 ms [17.41–19.00] | 1.93 ms [1.90–1.95] | 1.76 ms [1.75–1.79] | 1.29 ms [1.27–1.57] | unsupported | 0.83 ms [0.82–0.85] |
| 1 | 2 | 1.71 ms [1.65–1.71] | 1.65 ms [1.63–1.68] | unsupported | unsupported | 18.33 ms [17.51–19.35] | 2.73 ms [2.70–2.74] | 2.25 ms [2.21–2.28] | 1.87 ms [1.85–1.90] | unsupported | 1.52 ms [1.52–1.53] |
| 10 | 1 | 7.89 ms [7.86–7.99] | 7.89 ms [7.87–7.93] | unsupported | unsupported | 26.95 ms [25.97–28.42] | 17.35 ms [17.20–17.45] | 8.99 ms [8.95–9.06] | 8.94 ms [8.92–9.04] | unsupported | 6.88 ms [6.85–6.98] |
| 10 | 2 | 13.39 ms [13.37–13.54] | 13.40 ms [13.37–13.61] | unsupported | unsupported | 32.31 ms [30.84–32.71] | 24.92 ms [24.82–25.02] | 13.24 ms [13.20–14.48] | 14.71 ms [14.68–14.76] | unsupported | 13.53 ms [13.48–13.74] |
| 60 | 1 | 45.73 ms [45.60–45.96] | 45.67 ms [45.49–45.76] | unsupported | unsupported | 72.06 ms [70.70–72.61] | 103.24 ms [103.10–103.48] | 49.01 ms [48.90–49.17] | 50.92 ms [50.87–50.97] | unsupported | 40.13 ms [40.04–40.28] |
| 60 | 2 | 79.22 ms [78.99–79.53] | 79.36 ms [79.06–79.58] | unsupported | unsupported | 104.64 ms [103.13–104.91] | 148.27 ms [147.72–148.76] | 74.37 ms [74.20–74.60] | 84.77 ms [84.69–84.84] | unsupported | 81.88 ms [81.77–82.01] |
| 300 | 1 | 232.99 ms [232.87–233.62] | 233.72 ms [233.29–234.29] | unsupported | unsupported | 290.92 ms [288.32–293.63] | 523.47 ms [522.73–524.69] | 253.09 ms [252.96–253.50] | 260.21 ms [259.81–260.49] | unsupported | 233.49 ms [233.19–233.72] |
| 300 | 2 | 422.50 ms [422.09–422.68] | 422.33 ms [422.02–422.78] | unsupported | unsupported | 498.98 ms [495.49–502.48] | 752.33 ms [751.73–752.98] | 398.54 ms [397.73–399.14] | 436.92 ms [436.46–437.40] | unsupported | 465.29 ms [464.65–466.73] |

## mp3 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.20 ms [1.18–1.21] | 1.07 ms [1.07–1.12] | unsupported | unsupported | unsupported | 1.90 ms [1.88–1.97] | 1.72 ms [1.69–1.75] | 1.26 ms [1.24–1.28] | unsupported | 0.79 ms [0.79–0.83] |
| 1 | 2 | 1.79 ms [1.77–2.07] | 1.64 ms [1.62–1.66] | unsupported | unsupported | unsupported | 2.69 ms [2.67–2.76] | 2.21 ms [2.17–2.24] | 1.84 ms [1.81–1.85] | unsupported | 1.48 ms [1.47–1.50] |
| 10 | 1 | 1.44 ms [1.42–1.51] | 1.35 ms [1.34–1.36] | unsupported | unsupported | unsupported | 11.31 ms [11.28–11.48] | 1.96 ms [1.90–1.99] | 1.52 ms [1.50–1.54] | unsupported | 0.84 ms [0.82–0.91] |
| 10 | 2 | 1.91 ms [1.90–1.93] | 1.83 ms [1.80–1.85] | unsupported | unsupported | unsupported | 9.21 ms [9.15–9.23] | 2.40 ms [2.38–2.42] | 2.01 ms [1.99–2.04] | unsupported | 1.51 ms [1.48–1.56] |
| 60 | 1 | 2.72 ms [2.70–2.74] | 2.66 ms [2.64–2.69] | unsupported | unsupported | unsupported | 74.73 ms [74.58–74.97] | 3.06 ms [2.98–3.09] | 2.73 ms [2.70–2.80] | unsupported | 1.01 ms [0.99–1.03] |
| 60 | 2 | 2.72 ms [2.68–2.79] | 2.61 ms [2.59–2.63] | unsupported | unsupported | unsupported | 61.24 ms [61.02–61.40] | 3.06 ms [2.97–3.17] | 2.74 ms [2.72–2.78] | unsupported | 1.64 ms [1.63–1.65] |
| 300 | 1 | 4.40 ms [4.38–4.45] | 4.29 ms [4.24–4.32] | unsupported | unsupported | unsupported | 157.52 ms [157.22–157.83] | 4.32 ms [4.25–4.45] | 4.25 ms [4.18–4.36] | unsupported | 1.18 ms [1.18–1.21] |
| 300 | 2 | 5.22 ms [5.20–5.23] | 5.12 ms [5.10–5.14] | unsupported | unsupported | unsupported | 237.59 ms [236.92–237.88] | 5.02 ms [4.95–5.13] | 5.08 ms [5.01–5.10] | unsupported | 1.91 ms [1.90–1.98] |

## Unavailable and incorrect libraries

Every probed library was available and passed the correctness gate.

## Measurement noise

Across 352 `ok` measurements, the observed dispersion `IQR / median` ranges from 0.0% to 23.2%, with a median of 0.8%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings. The interquartile range is reported rather than max-min because the full range grows with the trial count, which would make runs using different `--repeat` values incomparable.
