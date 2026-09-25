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
| 1 | 1 | 0.36 ms [0.35–0.40] | 0.36 ms [0.34–0.38] | 0.23 ms [0.22–0.24] | 0.28 ms [0.26–0.29] | 0.39 ms [0.37–0.39] | 0.18 ms [0.18–0.23] | 6.68 ms [6.58–6.81] | 0.57 ms [0.56–0.58] | 0.22 ms [0.22–0.23] | 0.24 ms [0.23–0.25] |
| 1 | 2 | 0.53 ms [0.52–0.54] | 0.53 ms [0.51–0.54] | 0.32 ms [0.32–0.33] | 0.37 ms [0.36–0.39] | 0.54 ms [0.53–0.54] | 0.22 ms [0.22–0.24] | 7.90 ms [7.82–8.04] | 0.74 ms [0.73–0.78] | 0.32 ms [0.31–0.32] | 0.38 ms [0.37–0.38] |
| 10 | 1 | 1.40 ms [1.38–1.44] | 1.43 ms [1.38–1.45] | 0.78 ms [0.78–0.80] | 0.83 ms [0.79–0.85] | 2.19 ms [2.17–2.23] | 0.66 ms [0.65–0.67] | 9.51 ms [9.43–10.05] | 2.02 ms [1.99–2.06] | 0.61 ms [0.60–0.63] | 1.67 ms [1.65–1.80] |
| 10 | 2 | 3.25 ms [3.19–3.27] | 3.32 ms [3.25–3.39] | 1.78 ms [1.76–1.80] | 1.91 ms [1.71–2.16] | 3.83 ms [3.75–3.86] | 1.23 ms [1.22–1.29] | 11.24 ms [11.16–11.37] | 3.79 ms [3.75–3.88] | 1.59 ms [1.57–1.63] | 2.98 ms [2.97–3.05] |
| 60 | 1 | 7.18 ms [7.09–7.25] | 7.31 ms [7.27–7.62] | 3.85 ms [3.82–3.87] | 3.88 ms [3.80–4.12] | 12.34 ms [12.27–12.39] | 3.52 ms [3.50–3.61] | 17.81 ms [17.74–18.05] | 9.37 ms [9.12–9.54] | 2.84 ms [2.83–2.95] | 10.25 ms [10.22–10.37] |
| 60 | 2 | 18.90 ms [18.74–18.97] | 19.01 ms [18.94–19.17] | 11.29 ms [10.87–11.84] | 11.09 ms [10.77–11.57] | 22.36 ms [22.23–22.60] | 7.19 ms [7.14–7.33] | 29.06 ms [28.98–30.34] | 19.38 ms [19.20–19.55] | 9.08 ms [9.06–9.15] | 19.28 ms [19.22–19.52] |
| 300 | 1 | 43.42 ms [42.95–43.59] | 42.77 ms [42.39–42.95] | 32.90 ms [31.90–37.46] | 28.13 ms [28.03–28.41] | 73.43 ms [72.94–74.37] | 24.77 ms [24.52–24.88] | 100.53 ms [83.45–104.48] | 52.91 ms [52.43–53.48] | 20.70 ms [20.38–20.91] | 76.16 ms [76.07–76.29] |
| 300 | 2 | 126.53 ms [126.43–127.46] | 127.08 ms [126.70–127.47] | 116.35 ms [114.11–117.62] | 97.61 ms [97.34–98.42] | 169.68 ms [169.14–172.47] | 53.58 ms [53.15–54.79] | 154.56 ms [153.94–156.85] | 113.75 ms [113.35–114.15] | 103.15 ms [103.02–103.55] | 147.82 ms [147.46–148.45] |

## wav_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.43 ms [0.42–0.62] | 0.36 ms [0.35–0.37] | unsupported | 0.29 ms [0.29–0.31] | unsupported | 0.18 ms [0.18–0.22] | 6.60 ms [6.53–6.72] | 0.54 ms [0.53–0.56] | 0.24 ms [0.24–0.24] | 0.22 ms [0.22–0.22] |
| 1 | 2 | 0.60 ms [0.58–0.62] | 0.57 ms [0.55–0.67] | unsupported | 0.38 ms [0.37–0.38] | unsupported | 0.23 ms [0.23–0.24] | 7.94 ms [7.79–8.03] | 0.72 ms [0.69–0.73] | 0.34 ms [0.32–0.34] | 0.39 ms [0.35–0.41] |
| 10 | 1 | 0.44 ms [0.43–0.50] | 0.36 ms [0.35–0.40] | unsupported | 0.30 ms [0.29–0.32] | unsupported | 0.18 ms [0.17–0.19] | 7.97 ms [7.84–8.04] | 0.54 ms [0.54–0.58] | 0.24 ms [0.24–0.25] | 0.23 ms [0.22–0.24] |
| 10 | 2 | 0.60 ms [0.58–0.61] | 0.57 ms [0.55–0.58] | unsupported | 0.39 ms [0.38–0.41] | unsupported | 0.24 ms [0.23–0.25] | 8.01 ms [7.95–8.09] | 0.71 ms [0.68–0.76] | 0.33 ms [0.33–0.34] | 0.34 ms [0.33–0.35] |
| 60 | 1 | 0.43 ms [0.42–0.53] | 0.37 ms [0.36–0.38] | unsupported | 0.31 ms [0.30–0.35] | unsupported | 0.19 ms [0.18–0.21] | 7.96 ms [7.82–8.00] | 0.58 ms [0.57–0.60] | 0.24 ms [0.24–0.25] | 0.22 ms [0.22–0.23] |
| 60 | 2 | 0.61 ms [0.59–0.63] | 0.53 ms [0.52–0.54] | unsupported | 0.40 ms [0.39–0.41] | unsupported | 0.25 ms [0.24–0.27] | 8.23 ms [8.19–8.35] | 0.73 ms [0.71–0.77] | 0.33 ms [0.32–0.36] | 0.37 ms [0.37–0.37] |
| 300 | 1 | 0.44 ms [0.42–0.45] | 0.36 ms [0.35–0.37] | unsupported | 0.31 ms [0.29–0.31] | unsupported | 0.18 ms [0.18–0.19] | 8.02 ms [7.99–8.06] | 0.53 ms [0.52–0.55] | 0.25 ms [0.24–0.25] | 0.22 ms [0.22–0.23] |
| 300 | 2 | 0.62 ms [0.60–0.63] | 0.54 ms [0.52–0.55] | unsupported | 0.40 ms [0.39–0.42] | unsupported | 0.25 ms [0.24–0.26] | 8.27 ms [8.23–8.29] | 0.74 ms [0.71–0.78] | 0.36 ms [0.34–0.39] | 0.37 ms [0.36–0.38] |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.80 ms [0.79–0.81] | 0.77 ms [0.76–0.78] | unsupported | unsupported | 17.85 ms [17.25–18.48] | 0.46 ms [0.45–0.52] | 1.39 ms [1.37–1.42] | 1.01 ms [0.99–1.02] | unsupported | 0.49 ms [0.48–0.49] |
| 1 | 2 | 1.29 ms [1.28–1.36] | 1.25 ms [1.24–1.28] | unsupported | unsupported | 18.92 ms [18.39–19.54] | 0.80 ms [0.79–0.88] | 1.93 ms [1.91–2.00] | 1.53 ms [1.52–1.54] | unsupported | 0.90 ms [0.89–0.92] |
| 10 | 1 | 5.37 ms [5.34–5.46] | 5.35 ms [5.34–5.39] | unsupported | unsupported | 21.40 ms [20.38–22.95] | 3.34 ms [3.32–3.38] | 6.85 ms [6.82–6.89] | 6.07 ms [6.03–6.11] | unsupported | 3.85 ms [3.85–3.89] |
| 10 | 2 | 10.49 ms [10.47–10.60] | 10.59 ms [10.57–10.62] | unsupported | unsupported | 25.06 ms [24.25–26.37] | 6.92 ms [6.91–7.08] | 11.95 ms [11.90–12.00] | 11.10 ms [11.09–11.30] | unsupported | 7.72 ms [7.68–7.75] |
| 60 | 1 | 30.83 ms [30.77–34.55] | 30.98 ms [30.87–31.06] | unsupported | unsupported | 33.84 ms [33.65–35.76] | 19.67 ms [19.60–19.75] | 37.44 ms [37.14–37.71] | 33.38 ms [33.30–33.55] | unsupported | 23.11 ms [23.07–23.23] |
| 60 | 2 | 62.11 ms [62.03–62.47] | 62.83 ms [62.54–64.95] | unsupported | unsupported | 51.32 ms [49.31–52.43] | 41.94 ms [41.05–42.44] | 67.40 ms [67.33–67.50] | 63.40 ms [63.30–63.48] | unsupported | 47.50 ms [47.49–47.66] |
| 300 | 1 | 162.53 ms [162.29–164.44] | 161.91 ms [161.14–162.03] | unsupported | unsupported | 110.95 ms [109.29–111.40] | 106.67 ms [106.39–106.85] | 204.59 ms [202.34–229.04] | 175.08 ms [174.76–175.47] | unsupported | 145.87 ms [145.62–146.39] |
| 300 | 2 | 341.61 ms [341.49–343.90] | 342.92 ms [342.45–343.40] | unsupported | unsupported | 234.10 ms [232.07–285.51] | 223.57 ms [223.17–224.16] | 367.89 ms [367.19–368.44] | 332.25 ms [332.00–332.73] | unsupported | 289.33 ms [288.98–289.66] |

## flac_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.87 ms [0.85–0.89] | 0.77 ms [0.76–0.77] | unsupported | unsupported | unsupported | 0.45 ms [0.45–0.52] | 1.35 ms [1.32–1.38] | 0.98 ms [0.98–1.00] | unsupported | 0.47 ms [0.46–0.48] |
| 1 | 2 | 1.36 ms [1.35–1.38] | 1.25 ms [1.24–1.29] | unsupported | unsupported | unsupported | 0.81 ms [0.80–0.82] | 1.88 ms [1.86–1.90] | 1.50 ms [1.48–1.52] | unsupported | 0.88 ms [0.87–0.89] |
| 10 | 1 | 1.00 ms [0.99–1.02] | 0.95 ms [0.93–0.97] | unsupported | unsupported | unsupported | 0.50 ms [0.48–0.50] | 2.19 ms [2.17–2.21] | 1.12 ms [1.11–1.14] | unsupported | 0.46 ms [0.45–0.47] |
| 10 | 2 | 1.54 ms [1.53–1.64] | 1.53 ms [1.51–1.55] | unsupported | unsupported | unsupported | 0.91 ms [0.90–0.93] | 4.09 ms [4.08–4.11] | 1.68 ms [1.67–1.73] | unsupported | 0.94 ms [0.93–0.94] |
| 60 | 1 | 0.94 ms [0.94–0.95] | 0.89 ms [0.88–0.91] | unsupported | unsupported | unsupported | 0.50 ms [0.49–0.50] | 2.16 ms [2.15–2.18] | 1.06 ms [1.05–1.09] | unsupported | 0.51 ms [0.50–0.51] |
| 60 | 2 | 1.59 ms [1.59–1.70] | 1.80 ms [1.54–1.85] | unsupported | unsupported | unsupported | 0.92 ms [0.91–0.93] | 3.66 ms [3.62–3.73] | 1.72 ms [1.72–1.75] | unsupported | 1.01 ms [1.01–1.02] |
| 300 | 1 | 0.94 ms [0.94–0.96] | 0.87 ms [0.86–0.88] | unsupported | unsupported | unsupported | 0.50 ms [0.49–0.51] | 2.23 ms [2.22–2.55] | 1.06 ms [1.04–1.07] | unsupported | 0.55 ms [0.54–0.55] |
| 300 | 2 | 1.59 ms [1.58–1.61] | 1.52 ms [1.51–1.52] | unsupported | unsupported | unsupported | 0.92 ms [0.91–1.04] | 4.13 ms [4.10–4.18] | 1.72 ms [1.70–1.77] | unsupported | 1.01 ms [1.00–1.08] |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.05 ms [1.04–1.19] | 1.05 ms [1.05–1.06] | unsupported | unsupported | 17.88 ms [17.08–19.48] | 1.91 ms [1.90–1.92] | 1.75 ms [1.75–1.80] | 1.26 ms [1.26–1.27] | unsupported | 0.81 ms [0.81–0.84] |
| 1 | 2 | 1.65 ms [1.62–1.78] | 1.65 ms [1.63–1.67] | unsupported | unsupported | 18.08 ms [17.34–19.32] | 2.73 ms [2.72–2.74] | 2.24 ms [2.21–2.29] | 1.85 ms [1.85–1.89] | unsupported | 1.52 ms [1.51–1.53] |
| 10 | 1 | 7.85 ms [7.83–7.95] | 7.91 ms [7.89–7.93] | unsupported | unsupported | 26.34 ms [25.50–28.26] | 17.48 ms [17.33–17.51] | 8.94 ms [8.90–9.01] | 8.89 ms [8.88–8.92] | unsupported | 6.87 ms [6.82–7.03] |
| 10 | 2 | 13.40 ms [13.35–13.47] | 13.55 ms [13.40–15.77] | unsupported | unsupported | 31.94 ms [31.25–32.52] | 24.93 ms [24.81–25.05] | 13.29 ms [13.24–13.39] | 14.73 ms [14.70–14.80] | unsupported | 13.53 ms [13.50–13.59] |
| 60 | 1 | 45.79 ms [45.69–45.94] | 45.81 ms [45.73–45.85] | unsupported | unsupported | 72.28 ms [70.03–74.14] | 103.63 ms [103.56–105.17] | 49.21 ms [49.13–50.13] | 50.83 ms [50.64–50.88] | unsupported | 40.92 ms [40.78–41.07] |
| 60 | 2 | 79.10 ms [79.02–79.51] | 80.18 ms [79.34–80.42] | unsupported | unsupported | 104.13 ms [98.09–105.33] | 148.16 ms [147.49–148.21] | 74.47 ms [74.32–74.80] | 85.00 ms [84.82–85.20] | unsupported | 81.91 ms [81.73–81.99] |
| 300 | 1 | 234.19 ms [234.06–234.63] | 233.18 ms [232.97–233.40] | unsupported | unsupported | 294.74 ms [293.12–298.22] | 522.48 ms [522.37–522.73] | 254.79 ms [254.07–254.94] | 260.45 ms [260.06–260.83] | unsupported | 233.11 ms [232.94–233.48] |
| 300 | 2 | 423.17 ms [422.86–423.50] | 423.10 ms [422.85–423.35] | unsupported | unsupported | 500.00 ms [496.32–502.57] | 751.78 ms [751.21–753.16] | 399.66 ms [398.73–400.64] | 438.29 ms [438.14–438.57] | unsupported | 463.21 ms [462.65–464.11] |

## mp3 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.14 ms [1.14–1.16] | 1.06 ms [1.05–1.06] | unsupported | unsupported | unsupported | 1.89 ms [1.88–1.94] | 1.73 ms [1.69–1.77] | 1.23 ms [1.23–1.25] | unsupported | 0.79 ms [0.78–0.80] |
| 1 | 2 | 1.72 ms [1.70–1.99] | 1.64 ms [1.61–1.74] | unsupported | unsupported | unsupported | 2.69 ms [2.67–2.75] | 2.21 ms [2.18–2.26] | 1.83 ms [1.81–1.84] | unsupported | 1.47 ms [1.47–1.51] |
| 10 | 1 | 1.42 ms [1.42–1.43] | 1.34 ms [1.32–1.38] | unsupported | unsupported | unsupported | 11.38 ms [11.31–12.47] | 1.92 ms [1.88–1.93] | 1.50 ms [1.49–1.53] | unsupported | 0.82 ms [0.81–0.83] |
| 10 | 2 | 1.91 ms [1.88–1.92] | 1.85 ms [1.83–1.86] | unsupported | unsupported | unsupported | 9.19 ms [9.15–9.29] | 2.37 ms [2.33–2.46] | 2.00 ms [1.96–2.01] | unsupported | 1.50 ms [1.49–1.51] |
| 60 | 1 | 2.83 ms [2.81–2.84] | 2.71 ms [2.69–2.72] | unsupported | unsupported | unsupported | 74.94 ms [74.85–75.28] | 3.03 ms [2.96–3.07] | 2.80 ms [2.77–2.83] | unsupported | 1.11 ms [1.10–1.13] |
| 60 | 2 | 2.72 ms [2.71–2.77] | 2.65 ms [2.63–2.69] | unsupported | unsupported | unsupported | 61.26 ms [60.97–61.43] | 2.99 ms [2.94–3.00] | 2.78 ms [2.74–2.78] | unsupported | 1.66 ms [1.65–1.73] |
| 300 | 1 | 4.61 ms [4.56–4.63] | 4.51 ms [4.48–4.57] | unsupported | unsupported | unsupported | 157.65 ms [157.30–157.85] | 4.46 ms [4.35–4.51] | 4.45 ms [4.40–4.55] | unsupported | 1.41 ms [1.40–1.43] |
| 300 | 2 | 5.18 ms [5.17–5.26] | 5.05 ms [5.04–5.08] | unsupported | unsupported | unsupported | 237.48 ms [237.17–237.71] | 4.86 ms [4.84–4.99] | 5.05 ms [5.00–5.10] | unsupported | 1.89 ms [1.88–1.91] |

## Unavailable and incorrect libraries

Every probed library was available and passed the correctness gate.

## Measurement noise

Across 352 `ok` measurements, the observed dispersion `IQR / median` ranges from 0.0% to 14.7%, with a median of 0.8%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings. The interquartile range is reported rather than max-min because the full range grows with the trial count, which would make runs using different `--repeat` values incomparable.
