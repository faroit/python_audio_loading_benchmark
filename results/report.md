# pabench results

Timings below are wall-clock, warm-page-cache decodes of a locally generated **synthetic white-noise corpus** (see `docs/refactor-design.md`): one untimed warmup per (library, file, benchmark) that also warms the OS page cache, followed by the repeated, timed trials each cell's spread is drawn from. They describe decode speed against this machine's page cache and this synthetic corpus, not cold-disk I/O and not real program material.

## Platform

| Field | Value |
| --- | --- |
| OS | macOS-26.5.1-arm64-arm-64bit |
| Machine | arm64 |
| Python | 3.12.11 |
| PyTorch | 2.14.0 |
| FFmpeg | ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers |
| DYLD_FALLBACK_LIBRARY_PATH | /opt/homebrew/lib |

## Libraries

| Library | Version | Available | Seek | Notes | Error |
| --- | --- | --- | --- | --- | --- |
| soundfile | 0.14.0 | yes | yes | the correctness reference every other loader is checked against | - |
| librosa | 1.0.0 | yes | yes | - | - |
| scipy | 1.18.1 | yes | no | WAV only; scipy.io.wavfile has no seek API | - |
| scipy_mmap | 1.18.1 | yes | yes | WAV only, memory-mapped; scipy's mmap mode cannot open 24-bit ('3-byte container') WAV, which surfaces as a decode error for that subtype | - |
| audioread | 3.1.0 | yes | no | full-file only, no seek API; audioread always yields 16-bit PCM buffers, so it cannot pass the exact gate for 24-bit or float32 WAV sources | - |
| pedalboard | 0.9.25 | yes | yes | - | - |
| torchcodec | 0.16.0 | yes | yes | imports cleanly even when its native FFmpeg bindings can't load; only the decode smoke test below catches that (see docs/refactor-design.md) | - |
| audiolab | 0.5.2 | yes | yes | - | - |
| audiosample | 2.2.12 | yes | yes | integer-PCM WAV only: float WAV, FLAC and MP3 go through its PyAV path, which is incompatible with PyAV 18 (Flags.FAST_SEEK) | - |
| sphn | 0.2.1 | yes | yes | MP3 decode returns a different frame count than soundfile's reference (decoder-delay disagreement); graded by the relaxed MP3 gate | - |

**MP3 is graded by a relaxed gate** (decoded duration within 50 ms of the reference, RMS level within 0.5 dB on the common length) rather than the sample-exact gate WAV and FLAC are held to (1.5 LSB for integer PCM, `atol=1e-7` for float32), because MP3 decoders disagree on encoder delay and never match sample-for-sample. MP3 results therefore carry a weaker correctness guarantee than the WAV/FLAC results in this report.

## wav_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.25 ms [0.21–0.28] | 0.32 ms [0.21–5.76] | 0.19 ms [0.14–0.42] | 0.33 ms [0.25–6.19] | 0.34 ms [0.25–7.13] | 0.22 ms [0.17–0.52] | 5.63 ms [5.51–5.91] | 0.46 ms [0.37–0.63] | 0.14 ms [0.12–3.45] | 0.37 ms [0.19–0.82] |
| 1 | 2 | 0.35 ms [0.31–0.42] | 0.39 ms [0.33–0.44] | 0.25 ms [0.20–0.28] | 0.31 ms [0.26–4.72] | 0.40 ms [0.36–0.47] | 0.25 ms [0.22–0.31] | 12.24 ms [10.58–62.33] | 0.51 ms [0.44–0.57] | 0.24 ms [0.19–0.27] | 0.35 ms [0.31–0.41] |
| 10 | 1 | 0.89 ms [0.84–1.02] | 0.93 ms [0.88–0.98] | 0.45 ms [0.41–0.77] | 0.44 ms [0.40–5.16] | 1.47 ms [1.40–1.56] | 0.52 ms [0.47–0.57] | 13.28 ms [13.14–13.74] | 1.35 ms [1.21–1.43] | 0.40 ms [0.36–0.54] | 1.71 ms [1.24–2.83] |
| 10 | 2 | 1.89 ms [1.79–2.22] | 1.90 ms [1.82–2.22] | 1.02 ms [0.93–3.86] | 1.04 ms [0.99–6.19] | 2.73 ms [2.69–3.07] | 0.86 ms [0.79–0.94] | 26.04 ms [25.72–28.64] | 2.33 ms [2.24–2.81] | 0.94 ms [0.87–1.28] | 2.43 ms [2.23–2.49] |
| 60 | 1 | 4.43 ms [4.36–4.66] | 4.47 ms [4.31–4.76] | 2.04 ms [1.99–6.92] | 1.65 ms [1.55–5.89] | 9.11 ms [8.03–11.15] | 2.19 ms [2.03–2.30] | 15.84 ms [15.45–19.90] | 6.33 ms [5.85–10.65] | 1.77 ms [1.74–4.16] | 6.96 ms [6.33–9.33] |
| 60 | 2 | 9.97 ms [9.88–11.43] | 9.92 ms [9.80–12.78] | 5.31 ms [5.08–7.60] | 4.44 ms [4.34–8.97] | 15.87 ms [15.42–18.55] | 4.39 ms [4.12–4.55] | 31.15 ms [29.75–33.86] | 12.30 ms [12.08–17.59] | 5.09 ms [4.84–12.29] | 14.55 ms [12.26–54.59] |
| 300 | 1 | 21.72 ms [20.75–24.79] | 21.15 ms [20.68–60.01] | 9.18 ms [8.81–10.64] | 7.04 ms [6.81–14.72] | 40.19 ms [39.19–44.21] | 10.27 ms [9.68–23.35] | 28.59 ms [27.98–33.72] | 27.94 ms [27.68–29.95] | 9.04 ms [8.44–40.84] | 34.56 ms [32.59–51.92] |
| 300 | 2 | 52.07 ms [49.45–60.07] | 49.59 ms [49.06–80.34] | 25.67 ms [24.92–48.44] | 22.97 ms [21.01–29.51] | 76.94 ms [75.94–84.07] | 21.56 ms [20.79–32.14] | 52.37 ms [50.85–55.94] | 60.20 ms [58.03–124.35] | 23.95 ms [23.31–26.63] | 144.38 ms [132.87–162.02] |

## wav_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.32 ms [0.27–0.41] | 0.32 ms [0.25–1.80] | unsupported | 0.30 ms [0.23–9.04] | unsupported | 0.20 ms [0.16–0.26] | 5.46 ms [5.41–5.56] | 0.42 ms [0.33–2.90] | 0.21 ms [0.14–0.49] | 0.27 ms [0.24–0.40] |
| 1 | 2 | 0.42 ms [0.34–0.48] | 0.43 ms [0.31–2.54] | unsupported | 0.29 ms [0.25–4.43] | unsupported | 0.25 ms [0.21–0.31] | 11.91 ms [11.26–16.18] | 0.49 ms [0.42–0.58] | 0.22 ms [0.18–0.28] | 0.33 ms [0.33–0.44] |
| 10 | 1 | 0.31 ms [0.24–0.39] | 0.27 ms [0.22–0.30] | unsupported | 0.24 ms [0.23–5.32] | unsupported | 0.24 ms [0.18–0.25] | 12.73 ms [12.66–12.91] | 0.42 ms [0.35–3.00] | 0.19 ms [0.14–0.23] | 0.25 ms [0.23–0.32] |
| 10 | 2 | 0.46 ms [0.42–0.91] | 0.38 ms [0.33–0.44] | unsupported | 0.35 ms [0.26–0.57] | unsupported | 0.25 ms [0.20–0.30] | 24.93 ms [24.66–28.50] | 0.47 ms [0.39–0.55] | 0.22 ms [0.19–0.27] | 0.33 ms [0.31–0.36] |
| 60 | 1 | 0.32 ms [0.28–0.37] | 0.26 ms [0.22–0.34] | unsupported | 0.30 ms [0.23–4.62] | unsupported | 0.22 ms [0.19–0.30] | 12.50 ms [12.42–12.84] | 0.42 ms [0.34–0.71] | 0.19 ms [0.14–0.48] | 0.16 ms [0.13–0.19] |
| 60 | 2 | 0.40 ms [0.34–0.46] | 0.34 ms [0.31–0.71] | unsupported | 0.41 ms [0.39–4.99] | unsupported | 0.23 ms [0.21–0.29] | 24.99 ms [24.71–26.78] | 0.51 ms [0.44–0.60] | 0.21 ms [0.19–0.26] | 0.23 ms [0.19–0.86] |
| 300 | 1 | 0.31 ms [0.27–0.67] | 0.22 ms [0.21–0.28] | unsupported | 0.48 ms [0.40–2.68] | unsupported | 0.23 ms [0.20–0.34] | 12.72 ms [12.59–13.24] | 0.36 ms [0.31–0.74] | 0.17 ms [0.14–0.21] | 0.16 ms [0.13–0.30] |
| 300 | 2 | 0.45 ms [0.42–0.55] | 0.37 ms [0.33–0.42] | unsupported | 0.73 ms [0.63–4.32] | unsupported | 0.22 ms [0.20–0.28] | 24.96 ms [24.55–29.08] | 0.57 ms [0.50–1.69] | 0.21 ms [0.19–0.25] | 0.50 ms [0.42–0.55] |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.49 ms [0.47–0.59] | 0.57 ms [0.52–0.88] | unsupported | unsupported | 1.62 ms [1.57–11.13] | 0.57 ms [0.50–3.29] | 0.81 ms [0.77–0.93] | 0.68 ms [0.64–0.76] | unsupported | 0.44 ms [0.39–0.55] |
| 1 | 2 | 0.79 ms [0.76–0.94] | 0.80 ms [0.77–1.02] | unsupported | unsupported | 2.09 ms [2.01–2.22] | 0.87 ms [0.78–1.20] | 1.40 ms [1.27–1.71] | 0.92 ms [0.87–6.04] | unsupported | 0.75 ms [0.67–0.87] |
| 10 | 1 | 3.39 ms [3.35–3.48] | 3.36 ms [3.31–3.42] | unsupported | unsupported | 5.58 ms [5.48–6.32] | 2.81 ms [2.71–2.91] | 3.89 ms [3.80–4.04] | 3.80 ms [3.75–3.91] | unsupported | 3.41 ms [2.98–4.03] |
| 10 | 2 | 5.93 ms [5.83–6.37] | 5.92 ms [5.82–6.35] | unsupported | unsupported | 10.38 ms [10.21–11.97] | 5.56 ms [5.48–5.85] | 6.96 ms [6.82–7.63] | 6.48 ms [6.27–9.38] | unsupported | 5.46 ms [5.39–6.04] |
| 60 | 1 | 19.28 ms [19.20–19.74] | 19.24 ms [19.17–19.63] | unsupported | unsupported | 27.78 ms [27.24–30.16] | 15.08 ms [14.90–15.54] | 20.72 ms [20.58–22.51] | 21.00 ms [20.89–21.53] | unsupported | 11.45 ms [9.94–18.88] |
| 60 | 2 | 34.22 ms [33.91–36.96] | 33.93 ms [33.70–57.90] | unsupported | unsupported | 56.94 ms [55.78–60.13] | 31.80 ms [31.68–60.10] | 39.57 ms [38.65–46.05] | 37.34 ms [36.50–45.90] | unsupported | 23.76 ms [21.36–35.99] |
| 300 | 1 | 96.45 ms [95.28–173.07] | 95.54 ms [95.35–173.85] | unsupported | unsupported | 132.58 ms [130.75–138.88] | 74.34 ms [73.88–80.10] | 102.54 ms [100.46–111.60] | 105.38 ms [103.47–159.16] | unsupported | 61.62 ms [60.16–69.05] |
| 300 | 2 | 173.89 ms [168.36–251.41] | 172.43 ms [168.80–213.40] | unsupported | unsupported | 279.09 ms [273.45–328.69] | 158.40 ms [156.45–234.69] | 196.09 ms [192.39–211.79] | 183.75 ms [179.62–189.35] | unsupported | 323.73 ms [229.49–406.67] |

## flac_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.62 ms [0.55–0.91] | 0.56 ms [0.47–0.87] | unsupported | unsupported | unsupported | 0.57 ms [0.52–4.58] | 0.75 ms [0.67–0.85] | 0.70 ms [0.61–10.68] | unsupported | 0.48 ms [0.43–0.69] |
| 1 | 2 | 0.86 ms [0.78–1.52] | 0.77 ms [0.74–0.82] | unsupported | unsupported | unsupported | 0.89 ms [0.80–0.94] | 1.21 ms [1.13–1.82] | 0.91 ms [0.87–0.97] | unsupported | 0.75 ms [0.67–1.07] |
| 10 | 1 | 0.66 ms [0.57–0.81] | 0.63 ms [0.59–0.69] | unsupported | unsupported | unsupported | 0.62 ms [0.57–0.82] | 1.33 ms [1.25–1.42] | 0.68 ms [0.66–0.74] | unsupported | 0.43 ms [0.39–0.80] |
| 10 | 2 | 0.94 ms [0.85–1.05] | 0.93 ms [0.87–1.03] | unsupported | unsupported | unsupported | 0.88 ms [0.82–0.91] | 2.65 ms [2.56–2.83] | 0.99 ms [0.96–1.08] | unsupported | 0.80 ms [0.72–0.96] |
| 60 | 1 | 0.58 ms [0.55–1.41] | 0.55 ms [0.52–0.60] | unsupported | unsupported | unsupported | 0.56 ms [0.54–0.62] | 1.39 ms [1.24–1.58] | 0.69 ms [0.63–1.06] | unsupported | 0.28 ms [0.27–0.32] |
| 60 | 2 | 0.97 ms [0.93–1.06] | 0.88 ms [0.85–1.23] | unsupported | unsupported | unsupported | 0.93 ms [0.86–2.53] | 2.34 ms [2.21–4.57] | 1.06 ms [0.98–1.15] | unsupported | 0.50 ms [0.49–0.71] |
| 300 | 1 | 0.60 ms [0.56–1.11] | 0.54 ms [0.49–0.61] | unsupported | unsupported | unsupported | 0.56 ms [0.50–0.70] | 1.34 ms [1.28–1.51] | 0.68 ms [0.61–0.89] | unsupported | 0.30 ms [0.28–0.56] |
| 300 | 2 | 0.99 ms [0.95–1.11] | 0.87 ms [0.85–0.93] | unsupported | unsupported | unsupported | 0.89 ms [0.84–3.53] | 2.64 ms [2.57–3.00] | 1.03 ms [0.98–1.53] | unsupported | 1.30 ms [0.96–1.54] |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.50 ms [0.48–0.61] | 0.54 ms [0.49–0.61] | unsupported | unsupported | 1.47 ms [1.30–1.85] | 1.08 ms [0.99–1.13] | 1.10 ms [1.01–1.16] | 0.65 ms [0.60–0.81] | unsupported | 0.94 ms [0.71–1.33] |
| 1 | 2 | 0.80 ms [0.77–1.85] | 0.82 ms [0.77–0.92] | unsupported | unsupported | 2.04 ms [1.97–2.14] | 1.35 ms [1.30–1.41] | 1.54 ms [1.40–2.36] | 1.00 ms [0.92–1.50] | unsupported | 1.25 ms [1.22–1.68] |
| 10 | 1 | 3.33 ms [3.24–5.60] | 3.29 ms [3.21–4.09] | unsupported | unsupported | 8.27 ms [8.13–10.57] | 7.21 ms [7.10–8.06] | 5.08 ms [5.01–5.19] | 3.93 ms [3.84–4.38] | unsupported | 5.67 ms [4.75–7.88] |
| 10 | 2 | 6.73 ms [6.09–80.64] | 6.01 ms [5.91–6.71] | unsupported | unsupported | 13.74 ms [13.45–17.92] | 9.98 ms [9.94–10.17] | 7.81 ms [7.68–10.18] | 6.77 ms [6.62–7.59] | unsupported | 5.89 ms [5.85–14.59] |
| 60 | 1 | 18.35 ms [18.24–21.20] | 18.30 ms [18.21–18.73] | unsupported | unsupported | 46.71 ms [46.20–57.23] | 41.29 ms [41.14–45.07] | 28.73 ms [27.63–63.74] | 21.75 ms [21.24–23.99] | unsupported | 22.27 ms [20.34–26.86] |
| 60 | 2 | 34.38 ms [33.87–35.21] | 33.92 ms [33.87–35.62] | unsupported | unsupported | 78.10 ms [76.90–85.78] | 58.34 ms [58.08–61.55] | 43.96 ms [43.25–48.91] | 39.45 ms [38.08–41.77] | unsupported | 38.73 ms [37.66–58.05] |
| 300 | 1 | 91.22 ms [90.60–98.24] | 91.39 ms [90.47–100.08] | unsupported | unsupported | 229.67 ms [226.52–259.55] | 206.26 ms [204.01–291.00] | 138.99 ms [135.15–149.50] | 105.92 ms [104.92–265.71] | unsupported | 103.61 ms [101.26–128.33] |
| 300 | 2 | 176.60 ms [169.37–194.68] | 171.90 ms [169.10–184.41] | unsupported | unsupported | 387.96 ms [383.58–412.56] | 294.20 ms [288.69–367.46] | 221.85 ms [215.43–245.76] | 190.97 ms [187.48–206.25] | unsupported | 397.54 ms [345.70–456.88] |

## mp3 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.57 ms [0.56–0.67] | 0.60 ms [0.53–0.69] | unsupported | unsupported | unsupported | 1.08 ms [0.97–1.98] | 1.02 ms [0.93–1.21] | 0.69 ms [0.63–1.06] | unsupported | 0.70 ms [0.69–0.98] |
| 1 | 2 | 0.91 ms [0.83–12.50] | 0.80 ms [0.79–0.93] | unsupported | unsupported | unsupported | 1.34 ms [1.27–1.42] | 1.39 ms [1.26–1.98] | 0.94 ms [0.91–1.03] | unsupported | 1.36 ms [1.20–1.67] |
| 10 | 1 | 0.80 ms [0.71–0.91] | 0.76 ms [0.67–0.85] | unsupported | unsupported | unsupported | 4.85 ms [4.72–4.99] | 1.03 ms [1.01–1.55] | 0.82 ms [0.78–0.94] | unsupported | 0.72 ms [0.65–0.99] |
| 10 | 2 | 1.07 ms [0.97–10.40] | 0.98 ms [0.87–1.39] | unsupported | unsupported | unsupported | 3.88 ms [3.81–4.03] | 1.48 ms [1.30–2.01] | 1.06 ms [1.02–1.12] | unsupported | 0.69 ms [0.68–0.73] |
| 60 | 1 | 1.85 ms [1.80–2.56] | 1.75 ms [1.72–1.79] | unsupported | unsupported | unsupported | 30.31 ms [29.99–37.81] | 1.71 ms [1.56–2.25] | 1.97 ms [1.90–2.12] | unsupported | 0.51 ms [0.48–0.53] |
| 60 | 2 | 1.68 ms [1.53–2.03] | 1.55 ms [1.49–1.63] | unsupported | unsupported | unsupported | 24.32 ms [24.22–26.09] | 1.67 ms [1.61–1.99] | 1.71 ms [1.65–4.64] | unsupported | 0.78 ms [0.75–0.86] |
| 300 | 1 | 3.31 ms [3.24–6.90] | 3.20 ms [3.13–5.76] | unsupported | unsupported | unsupported | 63.05 ms [62.82–198.47] | 2.27 ms [2.20–2.53] | 3.36 ms [3.21–8.97] | unsupported | 1.58 ms [0.87–1.81] |
| 300 | 2 | 3.80 ms [3.64–9.99] | 3.73 ms [3.64–9.08] | unsupported | unsupported | unsupported | 94.51 ms [93.55–103.53] | 2.94 ms [2.80–3.59] | 3.70 ms [3.66–4.24] | unsupported | 2.10 ms [1.66–17.03] |

## Unavailable and incorrect libraries

Every probed library was available and passed the correctness gate.

## Measurement noise

Across 352 `ok` measurements, the observed spread `(max - min) / median` ranges from 0.4% to 1395.2%, with a median of 6.3%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings.
