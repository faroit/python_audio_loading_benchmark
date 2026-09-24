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

**MP3 is graded by a relaxed gate** (decoded duration within 50 ms of the reference, RMS level within 0.5 dB on the common length) rather than the sample-exact gate WAV and FLAC are held to (1.5 LSB for integer PCM, `atol=1e-7` for float32), because MP3 decoders disagree on encoder delay and never match sample-for-sample. MP3 results therefore carry a weaker correctness guarantee than the WAV/FLAC results in this report.

## wav_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.22 ms [0.21–0.26] | 0.29 ms [0.22–5.96] | 0.21 ms [0.15–0.48] | 0.31 ms [0.26–4.94] | 0.18 ms [0.15–0.23] | 0.33 ms [0.31–0.41] | 0.18 ms [0.17–0.56] | 5.51 ms [5.48–6.04] |
| 1 | 2 | 0.32 ms [0.31–0.59] | 0.36 ms [0.32–0.41] | 0.25 ms [0.22–3.72] | 0.49 ms [0.35–5.30] | 0.27 ms [0.21–0.34] | 0.38 ms [0.36–0.45] | 0.22 ms [0.20–0.32] | 10.42 ms [10.21–11.18] |
| 10 | 1 | 0.84 ms [0.82–0.90] | 0.84 ms [0.81–0.93] | 0.48 ms [0.44–1.67] | 0.53 ms [0.48–4.19] | 0.46 ms [0.43–0.51] | 1.44 ms [1.37–1.68] | 0.49 ms [0.46–0.54] | 13.09 ms [12.91–13.34] |
| 10 | 2 | 1.76 ms [1.73–1.85] | 1.85 ms [1.79–5.88] | 1.04 ms [0.98–1.42] | 1.28 ms [0.95–6.14] | 1.14 ms [1.04–1.53] | 2.64 ms [2.62–2.77] | 0.86 ms [0.83–1.07] | 25.06 ms [24.85–27.82] |
| 60 | 1 | 4.27 ms [4.24–4.71] | 4.33 ms [4.28–4.63] | 1.93 ms [1.86–21.95] | 1.96 ms [1.65–7.55] | 2.31 ms [2.19–5.00] | 7.99 ms [7.87–8.37] | 2.30 ms [2.07–2.94] | 15.58 ms [15.39–18.32] |
| 60 | 2 | 9.77 ms [9.62–10.26] | 10.26 ms [9.90–19.80] | 5.78 ms [5.10–7.10] | 9.04 ms [4.87–11.30] | 5.85 ms [5.60–6.28] | 15.24 ms [14.94–16.34] | 4.60 ms [4.13–5.14] | 29.91 ms [29.15–65.78] |
| 300 | 1 | 20.83 ms [20.60–58.89] | 21.12 ms [20.70–28.87] | 9.62 ms [8.82–15.62] | 13.70 ms [8.86–16.42] | 11.30 ms [10.35–18.28] | 41.71 ms [40.71–49.22] | 10.52 ms [9.63–13.69] | 28.48 ms [27.74–38.63] |
| 300 | 2 | 53.84 ms [48.05–144.48] | 49.15 ms [48.19–51.03] | 26.98 ms [24.35–32.47] | 26.23 ms [25.13–42.16] | 28.88 ms [27.91–116.00] | 76.42 ms [74.45–87.67] | 20.96 ms [20.21–25.25] | 51.12 ms [49.80–67.43] |

## wav_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.26 ms [0.24–0.29] | 0.28 ms [0.22–0.56] | unsupported | 0.33 ms [0.24–4.91] | unsupported | unsupported | 0.18 ms [0.17–3.72] | 5.43 ms [5.29–7.66] |
| 1 | 2 | 0.38 ms [0.35–13.82] | 0.37 ms [0.31–0.57] | unsupported | 0.42 ms [0.33–4.72] | unsupported | unsupported | 0.24 ms [0.20–0.30] | 10.26 ms [10.16–10.45] |
| 10 | 1 | 0.27 ms [0.25–0.33] | 0.22 ms [0.21–0.34] | unsupported | 0.33 ms [0.25–4.83] | unsupported | unsupported | 0.24 ms [0.22–0.30] | 12.46 ms [12.34–14.25] |
| 10 | 2 | 0.36 ms [0.34–0.43] | 0.34 ms [0.32–0.38] | unsupported | 5.00 ms [3.90–6.37] | unsupported | unsupported | 0.26 ms [0.21–0.30] | 24.22 ms [24.05–25.60] |
| 60 | 1 | 0.27 ms [0.26–0.33] | 0.25 ms [0.22–0.30] | unsupported | 0.36 ms [0.31–5.92] | unsupported | unsupported | 0.20 ms [0.17–0.24] | 12.33 ms [12.22–12.97] |
| 60 | 2 | 0.35 ms [0.35–0.36] | 0.37 ms [0.33–0.49] | unsupported | 0.60 ms [0.41–6.48] | unsupported | unsupported | 0.26 ms [0.20–0.42] | 24.91 ms [24.25–26.37] |
| 300 | 1 | 0.33 ms [0.27–0.41] | 0.26 ms [0.23–0.30] | unsupported | 5.30 ms [0.56–10.35] | unsupported | unsupported | 0.21 ms [0.17–0.33] | 14.62 ms [12.37–26.70] |
| 300 | 2 | 0.36 ms [0.35–0.37] | 0.40 ms [0.34–3.42] | unsupported | 5.08 ms [4.36–12.77] | unsupported | unsupported | 0.22 ms [0.19–1.42] | 25.09 ms [24.32–28.94] |

## wav_float / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.17 ms [0.16–0.23] | 0.22 ms [0.18–0.29] | 0.19 ms [0.16–3.27] | 0.30 ms [0.24–5.13] | 51.27 ms [50.06–55.57] | incorrect | 0.19 ms [0.17–0.33] | 0.55 ms [0.46–0.68] |
| 1 | 2 | 0.30 ms [0.24–0.53] | 0.28 ms [0.24–0.31] | 0.22 ms [0.20–0.34] | 0.46 ms [0.32–5.70] | 52.79 ms [51.30–61.63] | incorrect | 0.25 ms [0.22–0.32] | 0.57 ms [0.52–0.68] |
| 10 | 1 | 0.41 ms [0.38–0.53] | 0.44 ms [0.40–0.94] | 0.45 ms [0.41–0.62] | 5.06 ms [0.54–6.34] | 55.47 ms [52.07–99.77] | incorrect | 0.61 ms [0.54–0.86] | 1.07 ms [1.03–1.76] |
| 10 | 2 | 0.87 ms [0.80–0.88] | 0.92 ms [0.87–1.07] | 0.93 ms [0.88–1.26] | 5.58 ms [4.99–7.16] | 57.75 ms [56.80–89.28] | incorrect | 0.93 ms [0.89–1.03] | 1.60 ms [1.55–1.79] |
| 60 | 1 | 1.55 ms [1.53–1.99] | 1.55 ms [1.50–1.62] | 1.85 ms [1.66–2.20] | 2.22 ms [1.21–7.88] | 69.97 ms [66.10–97.89] | incorrect | 2.47 ms [2.40–2.70] | 4.32 ms [4.11–4.87] |
| 60 | 2 | 4.44 ms [4.34–4.55] | 4.57 ms [4.46–5.77] | 5.14 ms [4.87–6.47] | 7.45 ms [4.08–9.91] | 83.95 ms [81.17–102.48] | incorrect | 4.39 ms [4.31–4.62] | 7.35 ms [6.96–8.36] |
| 300 | 1 | 7.24 ms [7.06–8.14] | 7.16 ms [7.02–8.63] | 9.08 ms [8.76–14.21] | 10.34 ms [5.48–12.42] | 113.60 ms [109.21–146.03] | incorrect | 12.18 ms [11.53–44.72] | 19.46 ms [17.93–28.99] |
| 300 | 2 | 22.10 ms [21.44–24.58] | 21.95 ms [21.69–25.14] | 28.09 ms [27.10–31.53] | 24.32 ms [21.95–30.74] | 194.07 ms [174.12–398.60] | incorrect | 20.96 ms [20.82–22.30] | 39.05 ms [34.59–43.79] |

## wav_float / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.23 ms [0.21–0.31] | 0.23 ms [0.18–0.50] | unsupported | 0.34 ms [0.26–4.69] | unsupported | unsupported | 0.26 ms [0.19–5.46] | 0.43 ms [0.38–0.56] |
| 1 | 2 | 0.32 ms [0.29–0.39] | 0.30 ms [0.25–0.40] | unsupported | 0.35 ms [0.31–4.81] | unsupported | unsupported | 0.28 ms [0.23–0.33] | 0.57 ms [0.47–2.50] |
| 10 | 1 | 0.25 ms [0.23–0.32] | 0.24 ms [0.19–0.37] | unsupported | 4.23 ms [0.27–7.74] | unsupported | unsupported | 0.22 ms [0.18–0.33] | 0.52 ms [0.49–0.63] |
| 10 | 2 | 0.29 ms [0.27–0.33] | 0.25 ms [0.23–0.30] | unsupported | 4.23 ms [0.37–7.04] | unsupported | unsupported | 0.23 ms [0.20–0.30] | 0.76 ms [0.67–0.94] |
| 60 | 1 | 0.23 ms [0.20–0.74] | 0.22 ms [0.17–0.40] | unsupported | 5.78 ms [0.33–24.77] | unsupported | unsupported | 0.25 ms [0.20–0.29] | 0.52 ms [0.48–0.63] |
| 60 | 2 | 0.28 ms [0.26–0.34] | 0.29 ms [0.24–3.37] | unsupported | 4.10 ms [0.49–6.34] | unsupported | unsupported | 0.29 ms [0.23–0.52] | 0.75 ms [0.63–2.20] |
| 300 | 1 | 0.25 ms [0.23–1.80] | 0.18 ms [0.17–0.40] | unsupported | 5.24 ms [4.13–9.11] | unsupported | unsupported | 0.25 ms [0.21–0.31] | 0.52 ms [0.48–0.65] |
| 300 | 2 | 0.36 ms [0.32–0.54] | 0.29 ms [0.23–0.43] | unsupported | 5.58 ms [5.02–6.10] | unsupported | unsupported | 0.23 ms [0.22–0.25] | 0.75 ms [0.69–0.88] |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.54 ms [0.48–0.76] | 0.55 ms [0.49–1.21] | unsupported | unsupported | 51.84 ms [50.75–56.23] | 1.27 ms [1.21–3.12] | 0.54 ms [0.49–0.68] | 0.83 ms [0.79–0.89] |
| 1 | 2 | 0.74 ms [0.72–0.82] | 0.72 ms [0.71–0.85] | unsupported | unsupported | 52.15 ms [50.84–56.54] | 1.69 ms [1.65–5.68] | 0.86 ms [0.78–0.95] | 1.12 ms [1.07–1.64] |
| 10 | 1 | 3.32 ms [3.29–3.41] | 3.38 ms [3.31–4.11] | unsupported | unsupported | 56.24 ms [54.12–61.60] | 5.28 ms [5.15–6.65] | 2.75 ms [2.67–3.54] | 4.20 ms [3.95–4.70] |
| 10 | 2 | 5.74 ms [5.69–6.21] | 5.82 ms [5.72–6.31] | unsupported | unsupported | 60.10 ms [57.36–67.17] | 9.70 ms [9.64–9.79] | 5.55 ms [5.44–8.13] | 6.85 ms [6.79–7.73] |
| 60 | 1 | 18.89 ms [18.86–19.02] | 19.10 ms [18.89–21.53] | unsupported | unsupported | 70.23 ms [67.63–84.55] | 26.74 ms [26.51–29.36] | 14.79 ms [14.73–15.61] | 20.38 ms [20.22–23.17] |
| 60 | 2 | 33.41 ms [33.24–33.74] | 34.68 ms [33.40–49.38] | unsupported | unsupported | 84.19 ms [78.48–99.21] | 56.61 ms [55.23–67.12] | 31.90 ms [31.32–39.87] | 39.77 ms [38.07–43.32] |
| 300 | 1 | 98.70 ms [94.08–105.84] | 96.62 ms [94.40–104.16] | unsupported | unsupported | 119.82 ms [116.04–131.74] | 133.06 ms [130.79–150.01] | 73.90 ms [72.84–110.20] | 101.82 ms [99.33–109.73] |
| 300 | 2 | 167.35 ms [165.88–217.79] | 184.29 ms [176.02–289.85] | unsupported | unsupported | 198.82 ms [189.51–235.42] | 272.25 ms [267.67–284.09] | 155.07 ms [154.22–240.52] | 191.98 ms [189.80–304.95] |

## flac_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.54 ms [0.49–1.07] | 0.60 ms [0.51–0.80] | unsupported | unsupported | unsupported | unsupported | 0.59 ms [0.50–0.70] | 0.86 ms [0.75–1.28] |
| 1 | 2 | 0.78 ms [0.76–1.22] | 0.74 ms [0.70–0.85] | unsupported | unsupported | unsupported | unsupported | 0.87 ms [0.81–1.05] | 1.04 ms [0.97–1.47] |
| 10 | 1 | 0.60 ms [0.58–0.68] | 0.60 ms [0.56–0.94] | unsupported | unsupported | unsupported | unsupported | 0.60 ms [0.52–0.73] | 1.43 ms [1.31–2.24] |
| 10 | 2 | 0.89 ms [0.86–1.36] | 0.87 ms [0.83–0.91] | unsupported | unsupported | unsupported | unsupported | 0.94 ms [0.89–2.23] | 2.59 ms [2.53–2.95] |
| 60 | 1 | 0.55 ms [0.54–0.59] | 0.54 ms [0.52–0.62] | unsupported | unsupported | unsupported | unsupported | 0.57 ms [0.50–0.86] | 1.24 ms [1.22–1.41] |
| 60 | 2 | 0.91 ms [0.89–0.97] | 0.87 ms [0.84–0.93] | unsupported | unsupported | unsupported | unsupported | 0.90 ms [0.85–1.71] | 2.29 ms [2.17–2.45] |
| 300 | 1 | 0.62 ms [0.55–2.66] | 0.56 ms [0.49–0.59] | unsupported | unsupported | unsupported | unsupported | 0.55 ms [0.50–4.36] | 1.40 ms [1.26–1.78] |
| 300 | 2 | 0.91 ms [0.86–0.97] | 1.01 ms [0.95–2.35] | unsupported | unsupported | unsupported | unsupported | 0.86 ms [0.83–4.27] | 2.69 ms [2.58–3.49] |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.47 ms [0.46–0.52] | 0.58 ms [0.54–0.96] | unsupported | unsupported | 54.15 ms [51.81–59.37] | 1.25 ms [1.22–1.38] | 0.98 ms [0.93–1.13] | 1.01 ms [0.94–1.39] |
| 1 | 2 | 0.75 ms [0.73–0.83] | 0.74 ms [0.74–0.84] | unsupported | unsupported | 52.74 ms [51.67–65.89] | 1.84 ms [1.81–1.90] | 1.32 ms [1.24–1.66] | 1.33 ms [1.26–1.51] |
| 10 | 1 | 3.14 ms [3.12–3.19] | 3.23 ms [3.16–3.32] | unsupported | unsupported | 62.77 ms [58.67–67.34] | 7.85 ms [7.77–8.36] | 7.09 ms [6.95–7.89] | 5.07 ms [4.95–5.63] |
| 10 | 2 | 5.80 ms [5.75–7.12] | 5.89 ms [5.79–6.09] | unsupported | unsupported | 71.28 ms [62.60–87.88] | 13.24 ms [13.12–14.20] | 9.98 ms [9.80–11.15] | 7.69 ms [7.58–8.32] |
| 60 | 1 | 17.99 ms [17.94–19.29] | 18.16 ms [17.93–18.75] | unsupported | unsupported | 91.25 ms [88.08–127.46] | 46.32 ms [44.44–48.22] | 40.61 ms [40.49–61.83] | 27.01 ms [26.90–27.40] |
| 60 | 2 | 33.44 ms [33.28–36.18] | 33.51 ms [33.27–34.63] | unsupported | unsupported | 114.23 ms [110.05–247.10] | 76.87 ms [76.47–87.60] | 59.32 ms [57.18–72.01] | 44.20 ms [43.22–45.67] |
| 300 | 1 | 88.85 ms [88.40–95.66] | 90.19 ms [88.64–94.17] | unsupported | unsupported | 238.11 ms [224.45–303.49] | 224.10 ms [221.17–230.83] | 205.29 ms [201.12–249.84] | 136.86 ms [133.24–172.14] |
| 300 | 2 | 166.47 ms [165.09–182.81] | 188.37 ms [167.32–206.62] | unsupported | unsupported | 392.39 ms [370.31–444.28] | 388.06 ms [377.17–523.02] | 295.36 ms [285.01–352.72] | 223.56 ms [212.56–238.14] |

## mp3 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.56 ms [0.51–0.70] | 0.58 ms [0.56–0.86] | unsupported | unsupported | unsupported | unsupported | 1.03 ms [0.94–1.40] | 0.97 ms [0.87–1.35] |
| 1 | 2 | 0.81 ms [0.80–0.95] | 0.74 ms [0.74–0.77] | unsupported | unsupported | unsupported | unsupported | 1.35 ms [1.23–1.83] | 1.28 ms [1.15–1.90] |
| 10 | 1 | 0.72 ms [0.71–0.81] | 0.69 ms [0.66–1.56] | unsupported | unsupported | unsupported | unsupported | 4.76 ms [4.67–5.16] | 0.99 ms [0.97–1.64] |
| 10 | 2 | 0.92 ms [0.91–1.00] | 0.93 ms [0.87–1.00] | unsupported | unsupported | unsupported | unsupported | 3.87 ms [3.80–5.99] | 1.28 ms [1.25–1.44] |
| 60 | 1 | 1.78 ms [1.76–1.85] | 1.84 ms [1.76–2.05] | unsupported | unsupported | unsupported | unsupported | 29.66 ms [29.53–30.99] | 1.54 ms [1.51–3.67] |
| 60 | 2 | 1.57 ms [1.53–1.64] | 1.54 ms [1.49–1.64] | unsupported | unsupported | unsupported | unsupported | 24.14 ms [23.91–32.21] | 1.78 ms [1.59–2.15] |
| 300 | 1 | 3.36 ms [3.19–12.76] | 3.20 ms [3.10–4.27] | unsupported | unsupported | unsupported | unsupported | 62.12 ms [61.90–63.89] | 2.31 ms [2.18–2.93] |
| 300 | 2 | 3.71 ms [3.58–4.95] | 3.84 ms [3.62–5.29] | unsupported | unsupported | unsupported | unsupported | 95.55 ms [92.04–101.98] | 2.77 ms [2.63–2.95] |

## Unavailable and incorrect libraries

Libraries that failed the correctness gate for at least one cell:

- **audioread** (wav_float, full): max abs diff 1.53e-05 exceeds tolerance (rtol=1e-05, atol=1e-07)

## Measurement noise

Across 360 `ok` measurements, the observed spread `(max - min) / median` ranges from 0.2% to 1295.2%, with a median of 6.4%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings.
