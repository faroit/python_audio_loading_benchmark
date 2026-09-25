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
| 1 | 1 | 0.37 ms [0.35–0.61] | 0.36 ms [0.36–0.41] | 0.23 ms [0.22–0.26] | 0.30 ms [0.30–0.36] | 0.39 ms [0.38–0.39] | 0.18 ms [0.18–0.21] | 6.69 ms [6.59–6.77] | 0.57 ms [0.56–0.60] | 0.23 ms [0.22–0.23] | 0.25 ms [0.24–0.26] |
| 1 | 2 | 0.54 ms [0.52–0.77] | 0.53 ms [0.52–0.55] | 0.32 ms [0.32–0.36] | 0.39 ms [0.38–0.41] | 0.53 ms [0.51–0.57] | 0.23 ms [0.23–0.25] | 7.98 ms [7.82–8.15] | 0.76 ms [0.72–0.77] | 0.32 ms [0.31–0.34] | 0.39 ms [0.38–0.40] |
| 10 | 1 | 1.42 ms [1.38–1.47] | 1.42 ms [1.39–1.46] | 0.75 ms [0.73–0.76] | 0.77 ms [0.75–0.80] | 2.16 ms [2.14–2.20] | 0.67 ms [0.66–0.68] | 9.61 ms [9.48–9.81] | 2.02 ms [1.98–2.09] | 0.62 ms [0.61–0.65] | 1.68 ms [1.67–1.72] |
| 10 | 2 | 3.29 ms [3.23–3.39] | 3.26 ms [3.20–3.30] | 1.72 ms [1.70–1.78] | 1.78 ms [1.77–1.83] | 3.72 ms [3.69–4.64] | 1.23 ms [1.22–1.31] | 11.49 ms [11.39–11.70] | 3.76 ms [3.73–3.90] | 1.59 ms [1.57–1.65] | 2.95 ms [2.93–3.09] |
| 60 | 1 | 7.19 ms [7.08–7.86] | 7.17 ms [7.07–7.22] | 3.74 ms [3.69–3.85] | 3.67 ms [3.61–3.70] | 12.12 ms [12.04–12.27] | 3.43 ms [3.41–3.56] | 18.58 ms [17.86–19.02] | 9.50 ms [9.36–9.76] | 2.87 ms [2.82–2.95] | 10.07 ms [10.01–10.22] |
| 60 | 2 | 18.90 ms [18.72–19.14] | 18.99 ms [18.79–19.16] | 10.90 ms [10.70–11.16] | 10.78 ms [10.55–10.84] | 22.22 ms [21.94–22.37] | 7.04 ms [6.86–7.31] | 29.35 ms [29.04–29.72] | 19.63 ms [19.49–19.76] | 9.18 ms [9.13–9.23] | 19.37 ms [19.25–19.55] |
| 300 | 1 | 43.88 ms [43.47–44.20] | 42.77 ms [42.53–43.27] | 31.63 ms [31.38–31.81] | 28.09 ms [28.02–28.28] | 72.80 ms [72.49–73.04] | 25.47 ms [25.06–25.63] | 83.58 ms [74.85–103.20] | 54.03 ms [53.30–54.82] | 20.41 ms [19.95–20.62] | 76.21 ms [76.06–77.44] |
| 300 | 2 | 125.28 ms [124.31–125.92] | 124.82 ms [123.99–125.32] | 107.95 ms [107.76–108.23] | 94.93 ms [94.68–96.26] | 166.96 ms [166.40–167.47] | 51.51 ms [51.33–53.25] | 151.74 ms [149.88–175.05] | 111.42 ms [110.28–112.13] | 98.73 ms [98.42–99.24] | 145.31 ms [144.85–145.90] |

## wav_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.44 ms [0.43–0.58] | 0.36 ms [0.35–0.38] | unsupported | 0.31 ms [0.30–0.32] | unsupported | 0.18 ms [0.18–0.26] | 6.66 ms [6.58–6.78] | 0.55 ms [0.53–0.57] | 0.24 ms [0.24–0.26] | 0.22 ms [0.22–0.23] |
| 1 | 2 | 0.62 ms [0.60–0.71] | 0.53 ms [0.52–0.55] | unsupported | 0.39 ms [0.38–0.41] | unsupported | 0.24 ms [0.23–0.24] | 7.95 ms [7.73–8.05] | 0.73 ms [0.71–0.77] | 0.34 ms [0.33–0.35] | 0.36 ms [0.35–0.37] |
| 10 | 1 | 0.44 ms [0.43–0.57] | 0.37 ms [0.36–0.38] | unsupported | 0.31 ms [0.30–0.33] | unsupported | 0.18 ms [0.18–0.19] | 8.02 ms [7.88–8.14] | 0.55 ms [0.53–0.57] | 0.24 ms [0.24–0.26] | 0.23 ms [0.22–0.24] |
| 10 | 2 | 0.62 ms [0.60–0.70] | 0.54 ms [0.52–0.56] | unsupported | 0.40 ms [0.38–0.42] | unsupported | 0.24 ms [0.24–0.25] | 8.09 ms [7.98–8.16] | 0.73 ms [0.69–0.76] | 0.34 ms [0.34–0.35] | 0.37 ms [0.36–0.37] |
| 60 | 1 | 0.44 ms [0.42–0.45] | 0.37 ms [0.36–0.38] | unsupported | 0.31 ms [0.30–0.34] | unsupported | 0.18 ms [0.18–0.19] | 7.93 ms [7.76–8.11] | 0.55 ms [0.54–0.58] | 0.24 ms [0.24–0.26] | 0.22 ms [0.22–0.23] |
| 60 | 2 | 0.61 ms [0.59–0.68] | 0.54 ms [0.52–0.56] | unsupported | 0.40 ms [0.40–0.41] | unsupported | 0.24 ms [0.23–0.24] | 8.30 ms [8.14–8.34] | 0.72 ms [0.70–0.75] | 0.34 ms [0.34–0.47] | 0.37 ms [0.37–0.40] |
| 300 | 1 | 0.44 ms [0.43–0.47] | 0.37 ms [0.35–0.38] | unsupported | 0.32 ms [0.31–0.41] | unsupported | 0.19 ms [0.18–0.19] | 8.07 ms [7.98–8.14] | 0.55 ms [0.53–0.57] | 0.25 ms [0.24–0.26] | 0.23 ms [0.22–0.23] |
| 300 | 2 | 0.62 ms [0.60–0.63] | 0.54 ms [0.52–0.55] | unsupported | 0.40 ms [0.38–0.42] | unsupported | 0.24 ms [0.23–0.25] | 8.19 ms [8.09–11.20] | 0.73 ms [0.70–0.75] | 0.34 ms [0.33–0.35] | 0.37 ms [0.36–0.42] |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.81 ms [0.80–0.84] | 0.78 ms [0.76–0.87] | unsupported | unsupported | 17.84 ms [16.87–19.23] | 0.45 ms [0.44–0.46] | 1.40 ms [1.36–1.44] | 1.02 ms [1.00–1.05] | unsupported | 0.50 ms [0.49–0.52] |
| 1 | 2 | 1.30 ms [1.29–1.59] | 1.26 ms [1.25–1.32] | unsupported | unsupported | 18.50 ms [17.59–19.42] | 0.80 ms [0.79–0.81] | 1.95 ms [1.89–2.02] | 1.53 ms [1.50–1.60] | unsupported | 0.92 ms [0.90–0.96] |
| 10 | 1 | 5.38 ms [5.34–5.46] | 5.35 ms [5.32–5.46] | unsupported | unsupported | 21.49 ms [20.30–22.77] | 3.28 ms [3.26–3.33] | 6.94 ms [6.91–7.07] | 6.08 ms [6.04–6.14] | unsupported | 3.82 ms [3.79–3.85] |
| 10 | 2 | 10.52 ms [10.49–10.59] | 10.47 ms [10.42–10.57] | unsupported | unsupported | 24.92 ms [23.93–26.13] | 6.87 ms [6.83–6.92] | 12.07 ms [12.03–12.23] | 11.06 ms [11.04–11.19] | unsupported | 7.73 ms [7.70–7.81] |
| 60 | 1 | 30.86 ms [30.77–31.38] | 30.80 ms [30.64–30.96] | unsupported | unsupported | 34.64 ms [33.35–36.20] | 19.27 ms [19.18–19.43] | 38.10 ms [37.50–38.30] | 33.37 ms [33.33–33.53] | unsupported | 23.02 ms [22.96–23.13] |
| 60 | 2 | 62.15 ms [62.03–62.24] | 62.21 ms [62.02–62.48] | unsupported | unsupported | 51.01 ms [49.48–53.10] | 40.95 ms [40.74–41.18] | 67.47 ms [67.00–67.62] | 63.18 ms [63.06–63.71] | unsupported | 47.47 ms [47.30–47.67] |
| 300 | 1 | 161.16 ms [160.87–161.67] | 160.99 ms [160.58–161.34] | unsupported | unsupported | 109.77 ms [107.82–111.63] | 104.67 ms [104.29–105.21] | 211.47 ms [200.01–223.55] | 174.26 ms [173.81–174.49] | unsupported | 146.00 ms [145.80–146.54] |
| 300 | 2 | 339.98 ms [339.46–340.36] | 340.98 ms [340.67–342.94] | unsupported | unsupported | 228.05 ms [226.11–234.70] | 221.33 ms [220.61–222.75] | 390.50 ms [389.34–393.92] | 329.03 ms [328.75–329.48] | unsupported | 289.95 ms [289.36–292.53] |

## flac_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.88 ms [0.87–0.98] | 0.78 ms [0.76–0.79] | unsupported | unsupported | unsupported | 0.46 ms [0.45–0.46] | 1.38 ms [1.34–1.42] | 0.99 ms [0.97–1.02] | unsupported | 0.48 ms [0.47–0.49] |
| 1 | 2 | 1.38 ms [1.37–1.47] | 1.26 ms [1.24–1.26] | unsupported | unsupported | unsupported | 0.81 ms [0.80–0.83] | 1.93 ms [1.91–1.97] | 1.50 ms [1.47–1.53] | unsupported | 0.89 ms [0.89–0.91] |
| 10 | 1 | 1.02 ms [1.01–1.03] | 0.95 ms [0.93–0.96] | unsupported | unsupported | unsupported | 0.49 ms [0.48–0.50] | 2.23 ms [2.18–2.28] | 1.13 ms [1.11–1.19] | unsupported | 0.47 ms [0.46–0.55] |
| 10 | 2 | 1.56 ms [1.53–1.57] | 1.48 ms [1.45–1.49] | unsupported | unsupported | unsupported | 0.90 ms [0.89–0.99] | 4.12 ms [4.08–4.16] | 1.68 ms [1.63–1.75] | unsupported | 0.96 ms [0.95–0.97] |
| 60 | 1 | 0.96 ms [0.95–0.97] | 0.89 ms [0.87–0.89] | unsupported | unsupported | unsupported | 0.49 ms [0.49–0.51] | 2.20 ms [2.13–2.23] | 1.07 ms [1.04–1.10] | unsupported | 0.52 ms [0.51–0.58] |
| 60 | 2 | 1.61 ms [1.59–1.67] | 1.53 ms [1.51–1.54] | unsupported | unsupported | unsupported | 0.92 ms [0.91–0.94] | 3.67 ms [3.60–3.73] | 1.73 ms [1.68–1.76] | unsupported | 1.04 ms [1.03–1.07] |
| 300 | 1 | 0.95 ms [0.95–0.98] | 0.88 ms [0.87–0.89] | unsupported | unsupported | unsupported | 0.49 ms [0.48–0.50] | 2.23 ms [2.18–2.29] | 1.07 ms [1.02–1.10] | unsupported | 0.54 ms [0.53–0.59] |
| 300 | 2 | 1.60 ms [1.58–1.61] | 1.52 ms [1.51–1.53] | unsupported | unsupported | unsupported | 0.92 ms [0.91–0.94] | 4.21 ms [4.17–4.29] | 1.72 ms [1.67–1.74] | unsupported | 1.02 ms [1.01–1.29] |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.10 ms [1.07–1.22] | 1.08 ms [1.06–1.11] | unsupported | unsupported | 17.68 ms [16.77–19.86] | 1.92 ms [1.90–1.94] | 1.78 ms [1.72–1.82] | 1.28 ms [1.26–1.29] | unsupported | 0.83 ms [0.82–0.85] |
| 1 | 2 | 1.65 ms [1.62–1.68] | 1.70 ms [1.66–1.71] | unsupported | unsupported | 18.78 ms [17.11–19.81] | 2.73 ms [2.70–2.76] | 2.26 ms [2.21–2.35] | 1.89 ms [1.84–1.94] | unsupported | 1.54 ms [1.52–1.62] |
| 10 | 1 | 7.86 ms [7.85–7.93] | 7.88 ms [7.83–7.96] | unsupported | unsupported | 26.72 ms [25.59–27.86] | 17.34 ms [17.28–17.55] | 9.06 ms [9.02–9.16] | 8.94 ms [8.91–9.00] | unsupported | 6.94 ms [6.87–7.02] |
| 10 | 2 | 13.43 ms [13.36–13.54] | 13.42 ms [13.38–13.55] | unsupported | unsupported | 31.78 ms [30.53–33.00] | 24.84 ms [24.77–25.13] | 13.30 ms [13.25–13.40] | 14.72 ms [14.65–14.90] | unsupported | 13.51 ms [13.47–16.59] |
| 60 | 1 | 45.66 ms [45.48–45.89] | 45.78 ms [45.55–46.05] | unsupported | unsupported | 72.03 ms [70.71–72.88] | 103.33 ms [102.98–103.72] | 49.19 ms [49.08–49.70] | 50.71 ms [50.63–50.92] | unsupported | 40.79 ms [40.58–40.96] |
| 60 | 2 | 79.21 ms [79.05–79.46] | 79.38 ms [79.15–79.56] | unsupported | unsupported | 103.28 ms [102.54–104.46] | 148.37 ms [148.05–148.66] | 74.50 ms [74.35–74.76] | 84.83 ms [84.65–84.98] | unsupported | 82.07 ms [81.92–82.17] |
| 300 | 1 | 232.24 ms [231.83–232.54] | 231.79 ms [231.58–232.16] | unsupported | unsupported | 293.72 ms [291.97–298.26] | 523.05 ms [522.02–523.65] | 253.61 ms [252.97–254.80] | 260.89 ms [260.41–261.33] | unsupported | 233.37 ms [232.74–234.37] |
| 300 | 2 | 422.66 ms [422.27–423.37] | 422.65 ms [422.07–423.48] | unsupported | unsupported | 500.88 ms [497.14–505.41] | 753.00 ms [752.04–754.00] | 401.56 ms [399.60–404.14] | 438.00 ms [437.75–438.24] | unsupported | 462.85 ms [462.56–464.25] |

## mp3 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.19 ms [1.18–1.28] | 1.09 ms [1.06–1.12] | unsupported | unsupported | unsupported | 1.90 ms [1.88–1.92] | 1.75 ms [1.71–1.79] | 1.25 ms [1.25–1.28] | unsupported | 0.80 ms [0.78–0.81] |
| 1 | 2 | 1.74 ms [1.71–2.01] | 1.70 ms [1.67–1.73] | unsupported | unsupported | unsupported | 2.69 ms [2.66–2.70] | 2.25 ms [2.20–2.29] | 1.86 ms [1.82–1.92] | unsupported | 1.49 ms [1.48–1.51] |
| 10 | 1 | 1.43 ms [1.42–1.46] | 1.35 ms [1.34–1.40] | unsupported | unsupported | unsupported | 11.32 ms [11.24–11.43] | 1.98 ms [1.92–2.05] | 1.53 ms [1.51–1.55] | unsupported | 0.83 ms [0.83–0.85] |
| 10 | 2 | 1.92 ms [1.89–1.98] | 1.82 ms [1.81–1.94] | unsupported | unsupported | unsupported | 9.19 ms [9.10–9.23] | 2.41 ms [2.34–2.49] | 2.01 ms [1.99–2.04] | unsupported | 1.53 ms [1.49–1.55] |
| 60 | 1 | 2.72 ms [2.69–2.76] | 2.64 ms [2.59–2.66] | unsupported | unsupported | unsupported | 74.80 ms [74.53–74.99] | 2.99 ms [2.89–3.08] | 2.75 ms [2.72–2.78] | unsupported | 1.01 ms [0.99–1.08] |
| 60 | 2 | 2.71 ms [2.68–2.81] | 2.61 ms [2.59–2.66] | unsupported | unsupported | unsupported | 61.32 ms [61.19–61.44] | 3.02 ms [2.92–3.05] | 2.76 ms [2.71–2.84] | unsupported | 1.65 ms [1.63–1.70] |
| 300 | 1 | 4.34 ms [4.31–4.37] | 4.25 ms [4.24–4.30] | unsupported | unsupported | unsupported | 157.44 ms [157.15–158.03] | 4.31 ms [4.13–4.46] | 4.30 ms [4.22–4.39] | unsupported | 1.19 ms [1.17–1.21] |
| 300 | 2 | 5.21 ms [5.18–5.38] | 5.09 ms [5.06–5.12] | unsupported | unsupported | unsupported | 237.79 ms [237.31–238.22] | 4.92 ms [4.84–5.19] | 5.14 ms [5.09–5.19] | unsupported | 1.93 ms [1.91–1.96] |

## Unavailable and incorrect libraries

Every probed library was available and passed the correctness gate.

## Measurement noise

Across 352 `ok` measurements, the observed dispersion `IQR / median` ranges from 0.0% to 19.1%, with a median of 0.9%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings. The interquartile range is reported rather than max-min because the full range grows with the trial count, which would make runs using different `--repeat` values incomparable.
