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
| FLAC seektables | absent (seeking must scan frames) |

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
| 1 | 1 | 0.38 ms [0.36–0.56] | 0.38 ms [0.37–0.40] | 0.26 ms [0.22–0.28] | 0.30 ms [0.28–0.32] | 0.40 ms [0.37–0.44] | 0.19 ms [0.18–0.26] | 7.30 ms [6.55–7.89] | 0.60 ms [0.58–0.61] | 0.23 ms [0.22–0.25] | 0.25 ms [0.24–0.26] |
| 1 | 2 | 0.56 ms [0.54–0.81] | 0.54 ms [0.52–0.59] | 0.33 ms [0.32–0.39] | 0.43 ms [0.40–0.46] | 0.54 ms [0.53–0.63] | 0.25 ms [0.24–0.28] | 8.48 ms [7.90–9.25] | 0.78 ms [0.75–0.80] | 0.32 ms [0.32–0.34] | 0.39 ms [0.38–0.40] |
| 10 | 1 | 1.45 ms [1.42–1.50] | 1.45 ms [1.41–1.49] | 0.82 ms [0.79–0.88] | 0.83 ms [0.77–0.90] | 2.64 ms [2.21–2.77] | 0.71 ms [0.70–0.85] | 9.72 ms [9.65–9.99] | 2.05 ms [1.99–2.14] | 0.64 ms [0.62–0.68] | 1.71 ms [1.69–1.75] |
| 10 | 2 | 3.35 ms [3.29–3.41] | 3.43 ms [3.36–3.50] | 1.91 ms [1.84–2.01] | 1.94 ms [1.87–2.07] | 3.93 ms [3.84–4.58] | 1.39 ms [1.32–1.67] | 11.57 ms [11.45–11.73] | 3.91 ms [3.84–4.04] | 1.66 ms [1.64–1.67] | 3.06 ms [3.04–3.17] |
| 60 | 1 | 7.39 ms [7.31–7.56] | 7.43 ms [7.38–7.53] | 4.03 ms [3.90–4.21] | 3.75 ms [3.70–3.87] | 13.11 ms [12.43–15.11] | 3.99 ms [3.72–4.63] | 18.93 ms [18.18–19.23] | 9.97 ms [9.64–10.13] | 3.04 ms [3.00–3.14] | 9.75 ms [9.71–9.91] |
| 60 | 2 | 19.57 ms [19.23–20.08] | 19.56 ms [19.48–19.79] | 11.36 ms [11.18–11.59] | 11.15 ms [10.95–11.77] | 23.82 ms [22.56–24.93] | 7.96 ms [7.57–9.04] | 29.84 ms [29.53–31.70] | 19.89 ms [19.66–20.44] | 9.21 ms [9.17–9.30] | 18.42 ms [18.38–18.60] |
| 300 | 1 | 44.66 ms [44.21–45.27] | 43.80 ms [43.58–43.95] | 34.35 ms [33.20–37.52] | 29.97 ms [29.21–31.19] | 79.92 ms [77.88–87.37] | 27.43 ms [26.11–28.27] | 106.75 ms [84.65–109.12] | 53.81 ms [53.35–54.60] | 20.86 ms [20.54–21.16] | 82.42 ms [82.23–82.78] |
| 300 | 2 | 126.38 ms [126.10–126.70] | 127.54 ms [126.99–128.44] | 122.33 ms [118.35–124.99] | 102.43 ms [101.03–103.57] | 182.11 ms [174.66–185.50] | 55.96 ms [55.51–56.72] | 154.16 ms [151.89–174.35] | 112.30 ms [111.53–112.90] | 100.42 ms [100.05–100.91] | 155.32 ms [154.93–156.88] |

## wav_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.44 ms [0.43–0.48] | 0.38 ms [0.37–0.40] | unsupported | 0.32 ms [0.29–0.39] | unsupported | 0.21 ms [0.19–0.25] | 7.53 ms [6.68–7.86] | 0.57 ms [0.55–0.59] | 0.25 ms [0.24–0.26] | 0.23 ms [0.22–0.29] |
| 1 | 2 | 0.63 ms [0.59–0.71] | 0.54 ms [0.53–0.56] | unsupported | 0.46 ms [0.41–0.49] | unsupported | 0.25 ms [0.24–0.28] | 8.01 ms [7.85–9.40] | 0.74 ms [0.72–0.79] | 0.34 ms [0.33–0.36] | 0.36 ms [0.35–0.37] |
| 10 | 1 | 0.44 ms [0.43–0.80] | 0.42 ms [0.36–0.48] | unsupported | 0.34 ms [0.29–0.38] | unsupported | 0.21 ms [0.18–0.25] | 8.12 ms [8.00–8.43] | 0.56 ms [0.54–0.59] | 0.24 ms [0.24–0.25] | 0.23 ms [0.23–0.23] |
| 10 | 2 | 0.63 ms [0.62–0.65] | 0.54 ms [0.52–0.56] | unsupported | 0.41 ms [0.39–0.45] | unsupported | 0.25 ms [0.24–0.32] | 8.12 ms [8.06–8.25] | 0.72 ms [0.70–0.76] | 0.35 ms [0.33–0.39] | 0.36 ms [0.36–0.37] |
| 60 | 1 | 0.45 ms [0.44–0.46] | 0.37 ms [0.35–0.42] | unsupported | 0.32 ms [0.29–0.34] | unsupported | 0.20 ms [0.19–0.26] | 7.97 ms [7.86–8.05] | 0.55 ms [0.53–0.59] | 0.25 ms [0.24–0.25] | 0.23 ms [0.22–0.24] |
| 60 | 2 | 0.63 ms [0.62–0.65] | 0.54 ms [0.53–0.56] | unsupported | 0.39 ms [0.38–0.43] | unsupported | 0.26 ms [0.25–0.30] | 8.34 ms [8.23–8.42] | 0.74 ms [0.69–0.77] | 0.34 ms [0.33–0.35] | 0.38 ms [0.37–0.39] |
| 300 | 1 | 0.45 ms [0.43–0.46] | 0.36 ms [0.35–0.38] | unsupported | 0.30 ms [0.29–0.33] | unsupported | 0.19 ms [0.18–0.22] | 8.10 ms [7.97–8.20] | 0.56 ms [0.53–0.58] | 0.25 ms [0.24–0.27] | 0.23 ms [0.22–0.24] |
| 300 | 2 | 0.62 ms [0.61–0.64] | 0.55 ms [0.52–0.59] | unsupported | 0.39 ms [0.38–0.45] | unsupported | 0.27 ms [0.25–0.30] | 8.17 ms [8.08–8.35] | 0.73 ms [0.70–0.75] | 0.34 ms [0.33–0.36] | 0.37 ms [0.35–0.39] |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.82 ms [0.81–1.04] | 0.79 ms [0.77–0.82] | unsupported | unsupported | 14.99 ms [14.16–16.00] | 0.45 ms [0.44–0.50] | 1.47 ms [1.39–1.59] | 1.04 ms [1.02–1.08] | unsupported | 0.50 ms [0.49–0.51] |
| 1 | 2 | 1.33 ms [1.32–1.36] | 1.26 ms [1.25–1.27] | unsupported | unsupported | 17.04 ms [14.68–18.18] | 0.82 ms [0.81–0.92] | 2.07 ms [1.96–2.21] | 1.56 ms [1.51–1.60] | unsupported | 0.92 ms [0.91–0.99] |
| 10 | 1 | 5.45 ms [5.41–5.47] | 5.41 ms [5.38–6.57] | unsupported | unsupported | 20.27 ms [17.28–23.24] | 3.55 ms [3.33–3.64] | 6.93 ms [6.89–7.04] | 6.15 ms [6.13–6.23] | unsupported | 3.91 ms [3.89–3.93] |
| 10 | 2 | 10.65 ms [10.62–10.74] | 10.53 ms [10.51–10.55] | unsupported | unsupported | 21.28 ms [19.60–24.94] | 7.23 ms [7.05–7.59] | 12.07 ms [12.02–12.13] | 11.20 ms [11.17–11.28] | unsupported | 7.83 ms [7.79–7.93] |
| 60 | 1 | 31.10 ms [30.93–31.20] | 32.69 ms [31.99–33.26] | unsupported | unsupported | 33.16 ms [30.82–35.88] | 20.21 ms [20.05–20.31] | 38.10 ms [37.43–38.60] | 33.55 ms [33.42–33.60] | unsupported | 22.66 ms [22.61–22.76] |
| 60 | 2 | 62.81 ms [62.67–63.09] | 62.33 ms [62.25–62.55] | unsupported | unsupported | 48.98 ms [47.59–50.67] | 42.82 ms [42.57–45.07] | 67.93 ms [67.79–68.80] | 63.81 ms [63.80–64.05] | unsupported | 46.55 ms [46.45–46.73] |
| 300 | 1 | 162.74 ms [162.44–163.35] | 161.55 ms [161.11–161.75] | unsupported | unsupported | 115.55 ms [108.17–156.38] | 108.93 ms [107.95–110.03] | 212.62 ms [205.47–226.58] | 174.15 ms [173.75–174.36] | unsupported | 146.81 ms [146.70–147.17] |
| 300 | 2 | 342.77 ms [342.04–344.67] | 359.17 ms [357.84–362.02] | unsupported | unsupported | 238.54 ms [233.16–252.63] | 230.83 ms [230.10–234.35] | 369.83 ms [368.90–370.95] | 331.44 ms [331.31–332.75] | unsupported | 295.24 ms [294.69–296.12] |

## flac_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.90 ms [0.88–0.92] | 0.79 ms [0.78–0.80] | unsupported | unsupported | unsupported | 0.47 ms [0.45–0.51] | 1.40 ms [1.36–1.57] | 1.01 ms [1.00–1.03] | unsupported | 0.48 ms [0.47–0.49] |
| 1 | 2 | 1.42 ms [1.40–1.80] | 1.26 ms [1.24–1.31] | unsupported | unsupported | unsupported | 0.83 ms [0.81–0.89] | 1.99 ms [1.89–2.02] | 1.53 ms [1.48–1.58] | unsupported | 0.89 ms [0.89–0.91] |
| 10 | 1 | 1.03 ms [1.02–1.04] | 1.14 ms [0.94–1.17] | unsupported | unsupported | unsupported | 0.52 ms [0.50–0.57] | 2.23 ms [2.18–2.30] | 1.14 ms [1.13–1.20] | unsupported | 0.47 ms [0.47–0.49] |
| 10 | 2 | 1.57 ms [1.55–1.59] | 1.48 ms [1.48–1.49] | unsupported | unsupported | unsupported | 0.92 ms [0.90–1.00] | 4.09 ms [4.05–4.14] | 1.69 ms [1.66–1.72] | unsupported | 0.96 ms [0.96–1.01] |
| 60 | 1 | 0.96 ms [0.95–1.08] | 0.90 ms [0.87–1.07] | unsupported | unsupported | unsupported | 0.51 ms [0.50–0.57] | 2.20 ms [2.16–2.63] | 1.07 ms [1.04–1.09] | unsupported | 0.52 ms [0.51–0.53] |
| 60 | 2 | 1.62 ms [1.61–1.63] | 1.53 ms [1.52–1.55] | unsupported | unsupported | unsupported | 0.95 ms [0.93–1.05] | 3.66 ms [3.62–3.74] | 1.74 ms [1.70–1.77] | unsupported | 1.05 ms [1.03–1.05] |
| 300 | 1 | 0.95 ms [0.94–0.97] | 0.88 ms [0.87–0.93] | unsupported | unsupported | unsupported | 0.51 ms [0.49–0.59] | 2.23 ms [2.21–2.28] | 1.06 ms [1.02–1.09] | unsupported | 0.54 ms [0.54–0.55] |
| 300 | 2 | 1.61 ms [1.60–1.63] | 1.52 ms [1.51–1.80] | unsupported | unsupported | unsupported | 0.94 ms [0.93–1.05] | 4.18 ms [4.14–4.23] | 1.73 ms [1.69–1.81] | unsupported | 1.02 ms [1.02–1.04] |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.10 ms [1.08–1.15] | 1.08 ms [1.06–1.10] | unsupported | unsupported | 15.28 ms [14.12–16.22] | 1.97 ms [1.91–2.30] | 1.81 ms [1.78–2.05] | 1.30 ms [1.29–1.36] | unsupported | 0.83 ms [0.82–0.84] |
| 1 | 2 | 1.73 ms [1.70–1.74] | 1.64 ms [1.61–1.66] | unsupported | unsupported | 15.95 ms [14.42–18.49] | 2.78 ms [2.72–3.30] | 2.33 ms [2.27–2.65] | 1.87 ms [1.85–1.90] | unsupported | 1.54 ms [1.53–1.63] |
| 10 | 1 | 7.92 ms [7.89–7.97] | 7.87 ms [7.84–7.92] | unsupported | unsupported | 24.79 ms [23.57–28.18] | 18.30 ms [17.60–18.68] | 9.02 ms [8.95–9.17] | 8.93 ms [8.92–9.06] | unsupported | 6.92 ms [6.86–6.96] |
| 10 | 2 | 13.42 ms [13.36–13.58] | 13.41 ms [13.35–13.47] | unsupported | unsupported | 30.42 ms [27.28–32.12] | 26.04 ms [25.97–27.30] | 13.32 ms [13.28–13.50] | 14.80 ms [14.77–14.90] | unsupported | 13.67 ms [13.62–13.79] |
| 60 | 1 | 45.64 ms [45.56–45.76] | 46.67 ms [46.35–46.90] | unsupported | unsupported | 68.63 ms [66.71–71.16] | 107.70 ms [107.46–110.10] | 49.36 ms [49.05–50.17] | 50.76 ms [50.69–50.95] | unsupported | 40.26 ms [40.11–40.40] |
| 60 | 2 | 79.63 ms [79.31–79.98] | 79.43 ms [79.31–79.63] | unsupported | unsupported | 104.18 ms [101.04–124.21] | 155.85 ms [155.02–160.46] | 74.96 ms [74.69–78.41] | 85.31 ms [85.22–85.75] | unsupported | 81.78 ms [81.64–82.14] |
| 300 | 1 | 232.66 ms [232.39–233.27] | 231.19 ms [230.90–231.46] | unsupported | unsupported | 308.96 ms [297.00–378.87] | 545.32 ms [544.45–552.68] | 254.96 ms [254.58–255.82] | 260.06 ms [259.81–260.36] | unsupported | 235.83 ms [235.62–236.30] |
| 300 | 2 | 424.69 ms [424.32–427.40] | 434.65 ms [434.34–438.28] | unsupported | unsupported | 506.30 ms [500.48–614.00] | 792.88 ms [780.74–798.94] | 402.02 ms [401.33–403.39] | 438.57 ms [438.15–444.16] | unsupported | 472.63 ms [471.68–474.39] |

## mp3 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.20 ms [1.17–1.45] | 1.13 ms [1.06–1.46] | unsupported | unsupported | unsupported | 2.02 ms [1.89–2.21] | 1.75 ms [1.72–1.99] | 1.27 ms [1.26–1.30] | unsupported | 0.79 ms [0.78–0.81] |
| 1 | 2 | 1.82 ms [1.79–2.05] | 1.64 ms [1.63–1.71] | unsupported | unsupported | unsupported | 2.72 ms [2.68–3.21] | 2.26 ms [2.23–2.32] | 1.84 ms [1.81–1.88] | unsupported | 1.49 ms [1.49–1.51] |
| 10 | 1 | 1.46 ms [1.43–1.64] | 1.32 ms [1.32–1.33] | unsupported | unsupported | unsupported | 12.15 ms [11.36–12.90] | 1.97 ms [1.91–1.98] | 1.53 ms [1.50–1.55] | unsupported | 0.83 ms [0.82–0.85] |
| 10 | 2 | 1.92 ms [1.91–1.94] | 1.81 ms [1.80–1.85] | unsupported | unsupported | unsupported | 10.05 ms [9.19–10.31] | 2.43 ms [2.35–2.51] | 2.03 ms [2.00–2.04] | unsupported | 1.51 ms [1.50–1.53] |
| 60 | 1 | 2.72 ms [2.69–2.74] | 2.55 ms [2.53–3.06] | unsupported | unsupported | unsupported | 77.77 ms [77.42–79.81] | 3.01 ms [2.94–3.09] | 2.72 ms [2.69–2.82] | unsupported | 1.00 ms [0.99–1.08] |
| 60 | 2 | 2.76 ms [2.73–2.78] | 2.59 ms [2.57–2.61] | unsupported | unsupported | unsupported | 64.76 ms [63.57–65.41] | 3.07 ms [2.96–3.11] | 2.79 ms [2.73–2.82] | unsupported | 1.68 ms [1.67–1.69] |
| 300 | 1 | 4.39 ms [4.35–4.52] | 4.12 ms [4.10–4.13] | unsupported | unsupported | unsupported | 164.41 ms [163.39–168.40] | 4.28 ms [4.22–4.44] | 4.29 ms [4.22–4.32] | unsupported | 1.23 ms [1.21–1.28] |
| 300 | 2 | 5.34 ms [5.28–5.36] | 5.90 ms [5.01–6.03] | unsupported | unsupported | unsupported | 251.10 ms [250.52–252.51] | 4.99 ms [4.93–5.08] | 5.15 ms [5.10–5.18] | unsupported | 1.99 ms [1.98–2.01] |

## Unavailable and incorrect libraries

Every probed library was available and passed the correctness gate.

## Measurement noise

Across 352 `ok` measurements, the observed dispersion `IQR / median` ranges from 0.1% to 19.1%, with a median of 1.5%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings. The interquartile range is reported rather than max-min because the full range grows with the trial count, which would make runs using different `--repeat` values incomparable.
