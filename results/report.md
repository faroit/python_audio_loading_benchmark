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
| pydub | 0.25.1 | yes | no | full-file only; pydub has no seek API | - |
| audioread | 3.1.0 | yes | no | full-file only, no seek API; audioread always yields 16-bit PCM buffers, so it cannot pass the exact gate for 24-bit or float32 WAV sources | - |
| pedalboard | 0.9.25 | yes | yes | - | - |
| torchcodec | 0.16.0 | yes | yes | imports cleanly even when its native FFmpeg bindings can't load; only the decode smoke test below catches that (see docs/refactor-design.md) | - |
| audiolab | 0.5.2 | yes | yes | - | - |
| audiosample | 2.2.12 | yes | yes | WAV only here: its PyAV path is incompatible with PyAV 18 (Flags.FAST_SEEK) | - |
| sphn | 0.2.1 | yes | yes | MP3 decode returns a different frame count than soundfile's reference (decoder-delay disagreement); graded by the relaxed MP3 gate | - |

**MP3 is graded by a relaxed gate** (decoded duration within 50 ms of the reference, RMS level within 0.5 dB on the common length) rather than the sample-exact gate WAV and FLAC are held to (1.5 LSB for integer PCM, `atol=1e-7` for float32), because MP3 decoders disagree on encoder delay and never match sample-for-sample. MP3 results therefore carry a weaker correctness guarantee than the WAV/FLAC results in this report.

## wav_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.28 ms [0.25–0.50] | 0.25 ms [0.22–0.34] | 0.18 ms [0.15–0.26] | 0.22 ms [0.17–4.63] | 0.17 ms [0.15–0.24] | 0.25 ms [0.24–0.27] | 0.18 ms [0.17–0.23] | 5.34 ms [5.32–5.66] | 0.35 ms [0.32–0.47] | 0.19 ms [0.14–0.41] | 0.16 ms [0.13–0.19] |
| 1 | 2 | 0.30 ms [0.28–0.38] | 0.31 ms [0.29–0.37] | 0.19 ms [0.18–0.22] | 0.31 ms [0.24–7.13] | 0.21 ms [0.20–0.25] | 0.38 ms [0.36–3.96] | 0.21 ms [0.20–0.23] | 10.27 ms [10.21–10.53] | 0.42 ms [0.41–0.49] | 0.23 ms [0.19–0.46] | 0.25 ms [0.22–0.34] |
| 10 | 1 | 0.83 ms [0.79–1.03] | 0.89 ms [0.86–1.30] | 0.40 ms [0.37–0.46] | 0.43 ms [0.38–5.92] | 0.48 ms [0.45–0.59] | 1.44 ms [1.41–1.49] | 0.48 ms [0.44–0.56] | 13.04 ms [12.92–13.49] | 1.13 ms [1.12–1.28] | 0.61 ms [0.38–3.77] | 0.80 ms [0.77–0.93] |
| 10 | 2 | 1.74 ms [1.71–1.88] | 1.84 ms [1.76–2.24] | 0.95 ms [0.93–1.03] | 0.90 ms [0.85–5.22] | 1.04 ms [1.01–1.20] | 2.63 ms [2.61–4.32] | 0.80 ms [0.79–0.91] | 24.97 ms [24.82–25.40] | 2.72 ms [2.17–6.50] | 1.06 ms [0.93–1.23] | 1.46 ms [1.43–2.20] |
| 60 | 1 | 4.21 ms [4.17–4.43] | 4.32 ms [4.26–4.56] | 1.92 ms [1.80–2.47] | 1.53 ms [1.48–6.94] | 2.19 ms [2.11–2.66] | 8.04 ms [7.92–8.24] | 2.09 ms [2.05–2.45] | 15.32 ms [15.11–16.75] | 5.67 ms [5.65–6.15] | 1.80 ms [1.74–2.14] | 6.01 ms [5.86–6.96] |
| 60 | 2 | 9.72 ms [9.60–10.90] | 9.80 ms [9.64–10.24] | 4.97 ms [4.90–5.78] | 4.54 ms [4.26–8.70] | 5.63 ms [5.50–5.99] | 15.12 ms [15.03–15.65] | 4.16 ms [4.11–4.53] | 29.18 ms [28.96–83.96] | 11.81 ms [11.74–12.34] | 4.84 ms [4.77–6.59] | 13.96 ms [12.59–15.06] |
| 300 | 1 | 20.58 ms [20.29–21.89] | 21.09 ms [20.80–23.60] | 8.75 ms [8.63–10.35] | 6.81 ms [6.64–11.28] | 10.66 ms [10.22–11.64] | 39.72 ms [38.98–43.85] | 9.58 ms [9.50–10.44] | 28.88 ms [28.03–37.55] | 27.24 ms [27.13–29.67] | 8.69 ms [8.54–9.00] | 30.14 ms [27.95–39.80] |
| 300 | 2 | 49.09 ms [47.62–60.08] | 48.51 ms [47.61–112.25] | 24.52 ms [24.13–28.89] | 20.55 ms [20.26–25.37] | 27.69 ms [27.45–30.02] | 75.30 ms [74.54–141.86] | 20.46 ms [20.28–22.37] | 50.31 ms [49.64–53.31] | 57.38 ms [57.23–75.56] | 24.18 ms [23.92–27.51] | 56.77 ms [54.82–142.82] |

## wav_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.27 ms [0.26–0.38] | 0.24 ms [0.22–0.29] | unsupported | 0.23 ms [0.19–6.01] | unsupported | unsupported | 0.18 ms [0.17–0.23] | 5.29 ms [5.25–5.41] | 0.46 ms [0.38–5.70] | 0.20 ms [0.16–0.24] | 0.15 ms [0.14–6.92] |
| 1 | 2 | 0.37 ms [0.33–0.47] | 0.33 ms [0.30–0.36] | unsupported | 0.27 ms [0.24–5.22] | unsupported | unsupported | 0.20 ms [0.18–0.21] | 10.21 ms [10.16–10.60] | 0.40 ms [0.38–0.47] | 0.24 ms [0.22–0.33] | 0.25 ms [0.21–0.36] |
| 10 | 1 | 0.33 ms [0.28–0.58] | 0.22 ms [0.21–0.27] | unsupported | 3.93 ms [0.22–5.55] | unsupported | unsupported | 0.22 ms [0.19–0.26] | 12.57 ms [12.33–14.17] | 0.34 ms [0.30–0.58] | 0.22 ms [0.15–0.50] | 0.14 ms [0.13–0.17] |
| 10 | 2 | 0.34 ms [0.33–0.37] | 0.37 ms [0.33–0.41] | unsupported | 0.27 ms [0.22–4.94] | unsupported | unsupported | 0.22 ms [0.20–0.28] | 24.14 ms [23.91–25.71] | 0.42 ms [0.40–0.54] | 0.27 ms [0.22–0.34] | 0.25 ms [0.22–0.42] |
| 60 | 1 | 0.26 ms [0.24–0.42] | 0.23 ms [0.21–5.12] | unsupported | 0.28 ms [0.26–6.78] | unsupported | unsupported | 0.19 ms [0.17–0.29] | 12.29 ms [12.15–12.87] | 0.31 ms [0.27–0.69] | 0.15 ms [0.14–0.21] | 0.14 ms [0.13–0.22] |
| 60 | 2 | 0.34 ms [0.32–0.41] | 0.31 ms [0.30–0.37] | unsupported | 0.39 ms [0.34–5.35] | unsupported | unsupported | 0.22 ms [0.21–0.27] | 24.69 ms [24.29–27.87] | 0.40 ms [0.38–0.49] | 0.23 ms [0.20–0.29] | 0.22 ms [0.22–0.26] |
| 300 | 1 | 0.25 ms [0.25–0.35] | 0.25 ms [0.21–2.63] | unsupported | 0.40 ms [0.39–5.21] | unsupported | unsupported | 0.17 ms [0.16–0.31] | 13.86 ms [12.47–21.73] | 0.29 ms [0.29–0.33] | 0.15 ms [0.14–1.31] | 0.18 ms [0.15–0.23] |
| 300 | 2 | 0.37 ms [0.35–1.96] | 0.32 ms [0.30–0.38] | unsupported | 0.73 ms [0.61–5.04] | unsupported | unsupported | 0.23 ms [0.20–0.29] | 24.20 ms [24.13–24.38] | 0.38 ms [0.38–0.43] | 0.20 ms [0.19–4.91] | 0.22 ms [0.21–0.25] |

## wav_float / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.17 ms [0.16–0.22] | 0.17 ms [0.16–0.22] | 0.16 ms [0.14–4.21] | 0.19 ms [0.18–4.88] | 52.27 ms [51.46–57.23] | incorrect | 0.20 ms [0.18–2.52] | 0.46 ms [0.45–0.51] | 0.35 ms [0.30–0.39] | error | 0.16 ms [0.15–3.05] |
| 1 | 2 | 0.23 ms [0.20–0.30] | 0.23 ms [0.20–0.39] | 0.19 ms [0.18–0.25] | 3.83 ms [0.23–5.58] | 53.05 ms [51.64–59.05] | incorrect | 0.23 ms [0.21–0.30] | 0.53 ms [0.49–0.66] | 0.33 ms [0.31–0.37] | error | 0.24 ms [0.23–0.29] |
| 10 | 1 | 0.40 ms [0.37–0.44] | 0.39 ms [0.33–0.46] | 0.38 ms [0.36–0.42] | 0.37 ms [0.34–4.97] | 56.66 ms [53.99–66.11] | incorrect | 0.56 ms [0.53–0.60] | 1.16 ms [1.07–1.24] | 0.82 ms [0.66–2.96] | error | 0.89 ms [0.87–1.02] |
| 10 | 2 | 0.85 ms [0.80–0.93] | 0.91 ms [0.85–2.07] | 0.93 ms [0.88–1.16] | 0.77 ms [0.72–5.68] | 56.76 ms [55.81–59.38] | incorrect | 0.91 ms [0.88–1.12] | 1.56 ms [1.48–1.62] | 1.15 ms [1.11–3.31] | error | 1.68 ms [1.52–2.12] |
| 60 | 1 | 1.53 ms [1.46–1.62] | 1.54 ms [1.49–3.83] | 1.75 ms [1.64–1.92] | 1.24 ms [1.06–9.99] | 64.38 ms [62.94–73.10] | incorrect | 2.49 ms [2.34–2.75] | 3.92 ms [3.84–4.10] | 2.29 ms [2.22–2.42] | error | 6.58 ms [6.31–7.57] |
| 60 | 2 | 4.38 ms [4.30–4.49] | 4.44 ms [4.36–5.20] | 4.81 ms [4.76–5.51] | 3.57 ms [3.42–7.86] | 76.83 ms [75.53–82.36] | incorrect | 4.37 ms [4.30–6.26] | 7.32 ms [6.86–9.66] | 4.99 ms [4.89–5.12] | error | 12.67 ms [12.45–14.06] |
| 300 | 1 | 7.14 ms [6.95–8.61] | 7.49 ms [6.99–7.87] | 8.52 ms [8.20–9.76] | 5.23 ms [4.90–9.89] | 112.29 ms [108.94–128.94] | incorrect | 11.71 ms [11.50–12.95] | 19.10 ms [18.27–27.48] | 10.01 ms [9.91–10.30] | error | 34.73 ms [31.17–40.93] |
| 300 | 2 | 21.76 ms [21.37–23.40] | 21.63 ms [21.33–22.78] | 24.03 ms [23.81–27.23] | 17.15 ms [16.37–22.00] | 165.81 ms [163.12–183.86] | incorrect | 21.26 ms [20.94–64.23] | 34.96 ms [33.31–36.52] | 23.15 ms [23.03–24.69] | error | 62.42 ms [58.59–79.63] |

## wav_float / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.26 ms [0.22–0.34] | 0.23 ms [0.18–0.71] | unsupported | 0.20 ms [0.19–10.13] | unsupported | unsupported | 0.20 ms [0.18–3.90] | 0.41 ms [0.37–0.53] | 0.29 ms [0.26–0.34] | error | 0.16 ms [0.15–0.19] |
| 1 | 2 | 0.27 ms [0.26–0.33] | 0.26 ms [0.22–0.28] | unsupported | 4.75 ms [0.24–9.11] | unsupported | unsupported | 0.23 ms [0.22–0.27] | 0.52 ms [0.42–1.79] | 0.32 ms [0.30–0.35] | error | 0.25 ms [0.23–0.31] |
| 10 | 1 | 0.23 ms [0.22–0.76] | 0.18 ms [0.16–0.24] | unsupported | 0.23 ms [0.21–4.68] | unsupported | unsupported | 0.24 ms [0.20–0.32] | 0.54 ms [0.50–0.81] | 0.38 ms [0.32–0.61] | error | 0.16 ms [0.15–0.20] |
| 10 | 2 | 0.27 ms [0.25–0.33] | 0.25 ms [0.22–0.29] | unsupported | 0.29 ms [0.26–4.71] | unsupported | unsupported | 0.27 ms [0.24–0.29] | 0.61 ms [0.59–0.70] | 0.39 ms [0.35–0.42] | error | 0.24 ms [0.22–0.28] |
| 60 | 1 | 0.22 ms [0.21–0.25] | 0.18 ms [0.16–0.24] | unsupported | 0.32 ms [0.28–4.98] | unsupported | unsupported | 0.20 ms [0.19–0.25] | 0.50 ms [0.48–0.67] | 0.26 ms [0.24–0.33] | error | 0.16 ms [0.15–0.16] |
| 60 | 2 | 0.29 ms [0.28–0.71] | 0.23 ms [0.21–0.26] | unsupported | 0.41 ms [0.35–5.15] | unsupported | unsupported | 0.23 ms [0.22–0.29] | 0.66 ms [0.61–0.77] | 0.33 ms [0.32–0.40] | error | 0.22 ms [0.22–0.26] |
| 300 | 1 | 0.22 ms [0.21–0.25] | 0.21 ms [0.18–2.09] | unsupported | 0.59 ms [0.55–5.95] | unsupported | unsupported | 0.19 ms [0.17–0.24] | 0.60 ms [0.57–0.74] | 0.24 ms [0.24–0.34] | error | 0.17 ms [0.15–0.20] |
| 300 | 2 | 0.29 ms [0.26–0.35] | 0.27 ms [0.23–0.29] | unsupported | 0.95 ms [0.91–5.50] | unsupported | unsupported | 0.27 ms [0.23–0.37] | 0.64 ms [0.60–1.02] | 0.31 ms [0.30–3.21] | error | 0.24 ms [0.23–0.39] |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.52 ms [0.46–0.57] | 0.50 ms [0.45–0.92] | unsupported | unsupported | 52.51 ms [51.79–55.32] | 1.59 ms [1.42–1.85] | 0.50 ms [0.46–0.56] | 0.81 ms [0.74–0.88] | 0.60 ms [0.56–0.76] | unsupported | 0.27 ms [0.26–0.31] |
| 1 | 2 | 0.70 ms [0.68–0.79] | 0.73 ms [0.72–2.38] | unsupported | unsupported | 52.79 ms [51.76–54.61] | 2.02 ms [1.92–2.08] | 0.77 ms [0.75–0.83] | 1.07 ms [1.01–1.22] | 0.83 ms [0.81–0.86] | unsupported | 0.46 ms [0.45–0.77] |
| 10 | 1 | 3.25 ms [3.21–5.05] | 3.31 ms [3.27–3.48] | unsupported | unsupported | 56.43 ms [53.70–57.64] | 5.41 ms [5.28–5.47] | 2.79 ms [2.69–3.24] | 3.84 ms [3.71–4.19] | 3.70 ms [3.63–4.04] | unsupported | 1.78 ms [1.76–1.83] |
| 10 | 2 | 5.62 ms [5.60–5.88] | 5.91 ms [5.78–6.11] | unsupported | unsupported | 57.15 ms [55.65–61.68] | 10.04 ms [10.00–10.20] | 5.49 ms [5.38–5.80] | 6.65 ms [6.46–6.88] | 6.42 ms [6.23–30.99] | unsupported | 3.47 ms [3.43–3.55] |
| 60 | 1 | 18.84 ms [18.78–19.02] | 19.32 ms [19.13–78.35] | unsupported | unsupported | 64.94 ms [64.10–72.27] | 27.11 ms [26.91–28.42] | 14.78 ms [14.66–15.65] | 20.26 ms [20.04–22.14] | 20.68 ms [20.56–21.06] | unsupported | 11.94 ms [11.28–12.58] |
| 60 | 2 | 33.34 ms [33.16–35.28] | 33.89 ms [33.28–35.53] | unsupported | unsupported | 78.41 ms [76.44–96.66] | 54.90 ms [54.84–58.48] | 31.37 ms [31.08–90.24] | 37.99 ms [37.70–39.18] | 35.91 ms [35.54–38.37] | unsupported | 24.45 ms [24.09–26.07] |
| 300 | 1 | 94.15 ms [93.57–96.89] | 94.18 ms [93.72–166.59] | unsupported | unsupported | 119.20 ms [117.53–129.70] | 130.81 ms [130.27–135.84] | 73.13 ms [72.74–76.40] | 100.61 ms [99.11–107.08] | 101.49 ms [101.19–103.78] | unsupported | 56.33 ms [52.27–58.66] |
| 300 | 2 | 167.03 ms [165.91–172.12] | 167.57 ms [165.73–253.17] | unsupported | unsupported | 170.06 ms [167.44–183.97] | 274.97 ms [270.58–404.50] | 155.12 ms [154.21–216.37] | 188.90 ms [187.50–192.71] | 179.27 ms [177.54–184.40] | unsupported | 119.33 ms [115.71–126.65] |

## flac_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.53 ms [0.49–0.59] | 0.49 ms [0.46–0.55] | unsupported | unsupported | unsupported | unsupported | 0.49 ms [0.47–0.59] | 0.68 ms [0.66–0.80] | 0.57 ms [0.55–0.63] | unsupported | 0.27 ms [0.25–0.30] |
| 1 | 2 | 0.76 ms [0.72–0.83] | 0.75 ms [0.71–1.00] | unsupported | unsupported | unsupported | unsupported | 0.77 ms [0.75–0.88] | 1.06 ms [1.01–1.11] | 0.84 ms [0.80–0.93] | unsupported | 0.47 ms [0.45–0.51] |
| 10 | 1 | 0.61 ms [0.56–0.68] | 0.56 ms [0.54–0.60] | unsupported | unsupported | unsupported | unsupported | 0.54 ms [0.51–0.62] | 1.35 ms [1.28–1.48] | 0.70 ms [0.65–0.86] | unsupported | 0.26 ms [0.25–0.29] |
| 10 | 2 | 0.83 ms [0.81–0.88] | 0.88 ms [0.83–2.81] | unsupported | unsupported | unsupported | unsupported | 0.85 ms [0.82–0.97] | 2.43 ms [2.41–2.82] | 0.98 ms [0.94–1.21] | unsupported | 0.49 ms [0.48–0.57] |
| 60 | 1 | 0.53 ms [0.52–0.58] | 0.57 ms [0.51–0.61] | unsupported | unsupported | unsupported | unsupported | 0.55 ms [0.53–0.62] | 1.23 ms [1.17–1.35] | 0.63 ms [0.60–3.12] | unsupported | 0.29 ms [0.26–0.36] |
| 60 | 2 | 0.86 ms [0.82–1.17] | 0.88 ms [0.86–0.96] | unsupported | unsupported | unsupported | unsupported | 0.87 ms [0.85–0.95] | 2.09 ms [2.05–2.31] | 0.97 ms [0.93–1.18] | unsupported | 0.53 ms [0.53–0.57] |
| 300 | 1 | 0.55 ms [0.50–0.62] | 0.58 ms [0.55–2.45] | unsupported | unsupported | unsupported | unsupported | 0.54 ms [0.47–0.62] | 1.33 ms [1.26–1.49] | 0.58 ms [0.57–0.65] | unsupported | 0.30 ms [0.29–0.31] |
| 300 | 2 | 0.90 ms [0.87–0.95] | 0.91 ms [0.85–1.20] | unsupported | unsupported | unsupported | unsupported | 0.86 ms [0.84–0.97] | 2.52 ms [2.50–2.73] | 0.97 ms [0.93–1.06] | unsupported | 0.53 ms [0.51–0.64] |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.48 ms [0.45–0.53] | 0.53 ms [0.51–0.59] | unsupported | unsupported | 52.93 ms [51.88–54.50] | 1.48 ms [1.35–4.88] | 0.96 ms [0.93–1.05] | 0.93 ms [0.93–1.08] | 0.58 ms [0.55–0.67] | unsupported | 0.42 ms [0.41–0.45] |
| 1 | 2 | 0.75 ms [0.70–0.84] | 0.79 ms [0.75–0.87] | unsupported | unsupported | 53.38 ms [52.30–63.46] | 1.96 ms [1.94–3.28] | 1.27 ms [1.23–1.43] | 1.33 ms [1.18–1.38] | 0.84 ms [0.83–0.89] | unsupported | 0.74 ms [0.73–0.79] |
| 10 | 1 | 3.11 ms [3.00–3.24] | 3.18 ms [3.13–3.30] | unsupported | unsupported | 59.76 ms [58.10–65.93] | 8.03 ms [7.96–8.78] | 7.01 ms [6.97–7.17] | 4.97 ms [4.91–5.21] | 3.80 ms [3.67–4.23] | unsupported | 3.32 ms [3.27–3.40] |
| 10 | 2 | 5.70 ms [5.66–6.73] | 6.08 ms [5.97–19.17] | unsupported | unsupported | 61.78 ms [60.85–66.84] | 13.42 ms [13.33–13.84] | 9.93 ms [9.80–10.47] | 7.76 ms [7.62–8.53] | 6.51 ms [6.40–7.46] | unsupported | 6.30 ms [6.26–6.35] |
| 60 | 1 | 17.94 ms [17.88–18.56] | 18.23 ms [17.96–18.78] | unsupported | unsupported | 87.38 ms [86.25–94.61] | 45.50 ms [44.65–47.43] | 40.58 ms [40.43–42.52] | 27.24 ms [26.96–29.68] | 20.73 ms [20.67–21.83] | unsupported | 21.22 ms [20.88–24.22] |
| 60 | 2 | 33.58 ms [33.28–35.62] | 34.25 ms [34.08–94.48] | unsupported | unsupported | 111.13 ms [108.01–174.25] | 76.49 ms [75.91–91.18] | 57.25 ms [57.11–59.95] | 43.77 ms [43.05–115.45] | 36.93 ms [36.80–39.93] | unsupported | 40.00 ms [39.30–42.85] |
| 300 | 1 | 91.18 ms [88.69–105.64] | 88.89 ms [88.30–89.70] | unsupported | unsupported | 230.62 ms [227.47–236.87] | 224.83 ms [221.93–287.49] | 203.11 ms [200.81–221.72] | 135.52 ms [132.51–168.31] | 102.81 ms [102.20–107.69] | unsupported | 104.61 ms [100.93–108.14] |
| 300 | 2 | 172.82 ms [167.14–330.79] | 169.59 ms [166.68–177.57] | unsupported | unsupported | 336.08 ms [329.03–403.17] | 381.93 ms [377.53–409.19] | 286.26 ms [284.31–292.03] | 217.86 ms [211.82–238.17] | 186.40 ms [182.83–193.41] | unsupported | 200.96 ms [197.15–208.19] |

## mp3 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.51 ms [0.49–0.55] | 0.50 ms [0.45–0.53] | unsupported | unsupported | unsupported | unsupported | 0.95 ms [0.93–1.05] | 0.85 ms [0.80–0.89] | 0.57 ms [0.56–1.62] | unsupported | 0.42 ms [0.40–0.58] |
| 1 | 2 | 0.80 ms [0.76–0.86] | 0.80 ms [0.77–0.95] | unsupported | unsupported | unsupported | unsupported | 1.30 ms [1.20–2.32] | 1.26 ms [1.20–1.37] | 0.84 ms [0.82–0.89] | unsupported | 0.72 ms [0.70–0.76] |
| 10 | 1 | 0.75 ms [0.69–0.82] | 0.73 ms [0.68–0.76] | unsupported | unsupported | unsupported | unsupported | 4.66 ms [4.63–4.83] | 1.01 ms [0.94–1.11] | 0.79 ms [0.75–1.30] | unsupported | 0.43 ms [0.42–0.47] |
| 10 | 2 | 0.89 ms [0.86–0.94] | 0.92 ms [0.85–0.99] | unsupported | unsupported | unsupported | unsupported | 3.83 ms [3.79–3.95] | 1.40 ms [1.31–1.58] | 0.99 ms [0.96–1.03] | unsupported | 0.73 ms [0.72–0.78] |
| 60 | 1 | 1.74 ms [1.70–1.81] | 1.78 ms [1.70–1.85] | unsupported | unsupported | unsupported | unsupported | 29.66 ms [29.51–30.48] | 1.61 ms [1.52–1.85] | 1.82 ms [1.79–2.16] | unsupported | 0.59 ms [0.55–0.61] |
| 60 | 2 | 1.51 ms [1.49–1.58] | 1.54 ms [1.50–1.84] | unsupported | unsupported | unsupported | unsupported | 23.96 ms [23.89–26.74] | 2.24 ms [1.74–4.86] | 1.60 ms [1.56–3.57] | unsupported | 0.80 ms [0.78–0.88] |
| 300 | 1 | 3.23 ms [3.15–10.37] | 3.11 ms [3.07–5.46] | unsupported | unsupported | unsupported | unsupported | 62.94 ms [62.10–69.88] | 2.33 ms [2.23–43.47] | 3.19 ms [3.15–3.24] | unsupported | 0.65 ms [0.64–0.68] |
| 300 | 2 | 3.75 ms [3.63–4.04] | 3.55 ms [3.52–3.70] | unsupported | unsupported | unsupported | unsupported | 92.01 ms [91.62–159.70] | 2.75 ms [2.59–2.96] | 3.68 ms [3.63–5.74] | unsupported | 0.98 ms [0.96–1.03] |

## Unavailable and incorrect libraries

Libraries that failed the correctness gate for at least one cell:

- **audioread** (wav_float, full): max abs diff 1.53e-05 exceeds tolerance (rtol=1e-05, atol=1e-07)

## Measurement noise

Across 504 `ok` measurements, the observed spread `(max - min) / median` ranges from 0.2% to 1777.5%, with a median of 4.5%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings.
