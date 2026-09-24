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
| stempeg | 0.2.6 | yes | yes | - | - |
| pedalboard | 0.9.25 | yes | yes | - | - |
| torchcodec | 0.16.0 | yes | yes | imports cleanly even when its native FFmpeg bindings can't load; only the decode smoke test below catches that (see docs/refactor-design.md) | - |

**MP3 is graded by a relaxed gate** (decoded duration within 50 ms of the reference, RMS level within 0.5 dB on the common length) rather than the sample-exact gate WAV and FLAC are held to (1.5 LSB for integer PCM, `atol=1e-7` for float32), because MP3 decoders disagree on encoder delay and never match sample-for-sample. MP3 results therefore carry a weaker correctness guarantee than the WAV/FLAC results in this report.

## wav_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | stempeg | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.22 ms [0.21–0.26] | 0.22 ms [0.21–0.24] | 0.14 ms [0.14–0.15] | 4.71 ms [0.19–6.88] | 0.14 ms [0.14–0.16] | 0.25 ms [0.24–0.30] | 106.43 ms [105.55–109.92] | 0.17 ms [0.16–0.19] | 5.15 ms [5.13–5.17] |
| 1 | 2 | 0.29 ms [0.29–0.30] | 0.29 ms [0.29–0.30] | 0.18 ms [0.18–0.21] | 0.26 ms [0.22–3.80] | 0.20 ms [0.20–0.23] | 0.39 ms [0.35–0.40] | 122.24 ms [120.46–123.17] | 0.21 ms [0.18–0.25] | 10.05 ms [10.00–10.14] |
| 10 | 1 | 0.78 ms [0.77–0.81] | 0.82 ms [0.77–0.87] | 0.37 ms [0.37–0.38] | 0.37 ms [0.35–0.39] | 0.44 ms [0.43–0.49] | 1.33 ms [1.31–1.40] | 129.31 ms [128.34–134.50] | 0.44 ms [0.42–0.47] | 12.65 ms [12.63–12.71] |
| 10 | 2 | 1.66 ms [1.64–1.77] | 1.78 ms [1.64–2.67] | 0.87 ms [0.85–0.94] | 0.80 ms [0.80–5.58] | 1.00 ms [0.98–1.06] | 2.52 ms [2.52–2.54] | 166.64 ms [166.12–170.40] | 0.75 ms [0.74–0.77] | 24.57 ms [24.45–24.71] |
| 60 | 1 | 4.13 ms [4.10–4.26] | 4.36 ms [4.22–5.62] | 1.74 ms [1.69–1.79] | 1.42 ms [1.41–1.45] | 2.09 ms [2.06–2.51] | 7.92 ms [7.81–8.02] | 136.06 ms [135.55–139.10] | 1.98 ms [1.97–2.00] | 14.98 ms [14.96–15.05] |
| 60 | 2 | 9.60 ms [9.53–9.64] | 10.98 ms [9.98–12.80] | 4.87 ms [4.80–6.81] | 4.18 ms [4.10–8.40] | 5.56 ms [5.52–5.92] | 14.97 ms [14.86–16.81] | 181.26 ms [180.02–184.93] | 4.07 ms [4.02–4.10] | 28.76 ms [28.67–30.51] |
| 300 | 1 | 20.48 ms [20.42–20.57] | 21.57 ms [21.14–21.86] | 8.53 ms [8.37–8.73] | 6.48 ms [6.45–6.65] | 10.19 ms [10.10–10.33] | 38.71 ms [38.46–39.13] | 175.34 ms [173.07–184.19] | 9.53 ms [9.49–10.61] | 26.99 ms [26.97–27.16] |
| 300 | 2 | 47.94 ms [47.90–48.09] | 47.94 ms [47.88–54.66] | 23.95 ms [23.73–24.51] | 19.85 ms [19.81–24.59] | 27.30 ms [27.24–27.58] | 73.98 ms [73.88–74.26] | 248.36 ms [245.62–250.91] | 20.12 ms [20.08–29.68] | 49.11 ms [48.85–52.75] |

## wav_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | stempeg | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.25 ms [0.24–0.26] | 0.21 ms [0.20–0.21] | unsupported | 0.22 ms [0.19–4.73] | unsupported | unsupported | 106.38 ms [104.56–107.61] | 0.17 ms [0.16–0.20] | 5.07 ms [5.04–5.41] |
| 1 | 2 | 0.34 ms [0.33–0.36] | 0.29 ms [0.28–0.31] | unsupported | 0.24 ms [0.23–0.24] | unsupported | unsupported | 121.38 ms [120.70–126.39] | 0.20 ms [0.19–0.22] | 9.96 ms [9.89–10.20] |
| 10 | 1 | 0.26 ms [0.24–0.31] | 0.22 ms [0.21–0.25] | unsupported | 0.20 ms [0.17–4.76] | unsupported | unsupported | 128.15 ms [126.64–132.44] | 0.17 ms [0.16–0.19] | 12.21 ms [12.18–14.26] |
| 10 | 2 | 0.34 ms [0.33–0.35] | 0.30 ms [0.28–0.35] | unsupported | 0.28 ms [0.24–4.83] | unsupported | unsupported | 163.76 ms [162.70–167.36] | 0.20 ms [0.19–0.23] | 23.90 ms [23.74–24.00] |
| 60 | 1 | 0.25 ms [0.25–0.26] | 0.21 ms [0.19–0.25] | unsupported | 0.23 ms [0.22–0.23] | unsupported | unsupported | 128.65 ms [127.00–133.96] | 0.17 ms [0.16–0.19] | 12.02 ms [11.98–12.75] |
| 60 | 2 | 0.34 ms [0.33–0.35] | 0.30 ms [0.27–0.34] | unsupported | 0.32 ms [0.31–0.33] | unsupported | unsupported | 167.01 ms [166.04–169.30] | 0.19 ms [0.19–0.20] | 24.03 ms [24.01–24.17] |
| 300 | 1 | 0.25 ms [0.24–0.28] | 0.23 ms [0.21–0.27] | unsupported | 0.37 ms [0.35–4.16] | unsupported | unsupported | 132.88 ms [132.26–139.92] | 0.16 ms [0.16–0.17] | 12.19 ms [12.18–12.62] |
| 300 | 2 | 0.36 ms [0.34–0.41] | 0.31 ms [0.29–0.34] | unsupported | 0.57 ms [0.55–4.32] | unsupported | unsupported | 176.52 ms [175.50–180.46] | 0.20 ms [0.19–0.21] | 23.89 ms [23.86–24.04] |

## wav_float / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | stempeg | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.16 ms [0.15–0.16] | 0.16 ms [0.15–0.17] | 0.13 ms [0.12–0.13] | 0.18 ms [0.18–0.19] | 49.85 ms [49.58–51.33] | incorrect | 91.90 ms [90.63–96.31] | 0.17 ms [0.16–0.18] | 0.46 ms [0.41–0.54] |
| 1 | 2 | 0.24 ms [0.21–0.26] | 0.21 ms [0.19–0.23] | 0.18 ms [0.17–0.22] | 0.25 ms [0.21–4.06] | 50.10 ms [49.78–53.43] | incorrect | 91.68 ms [90.96–94.48] | 0.21 ms [0.20–0.22] | 0.47 ms [0.46–0.58] |
| 10 | 1 | 0.38 ms [0.37–0.38] | 0.43 ms [0.37–2.34] | 0.36 ms [0.34–0.37] | 0.33 ms [0.31–4.82] | 51.83 ms [50.90–53.13] | incorrect | 93.73 ms [92.63–101.93] | 0.49 ms [0.47–0.50] | 1.01 ms [0.99–1.05] |
| 10 | 2 | 0.84 ms [0.83–0.87] | 0.84 ms [0.81–0.93] | 0.86 ms [0.82–0.89] | 0.68 ms [0.67–1.01] | 55.97 ms [53.83–58.52] | incorrect | 96.32 ms [95.37–99.43] | 0.84 ms [0.83–0.85] | 1.47 ms [1.46–1.48] |
| 60 | 1 | 1.51 ms [1.42–1.54] | 1.47 ms [1.40–1.58] | 1.72 ms [1.70–1.82] | 1.09 ms [1.08–1.22] | 61.43 ms [61.06–62.70] | incorrect | 101.12 ms [100.50–103.12] | 2.39 ms [2.38–2.56] | 3.86 ms [3.83–6.20] |
| 60 | 2 | 4.29 ms [4.23–4.56] | 4.26 ms [4.13–4.96] | 4.68 ms [4.64–4.74] | 3.40 ms [3.34–6.51] | 74.35 ms [73.83–75.13] | incorrect | 109.80 ms [109.34–111.91] | 4.28 ms [4.24–4.47] | 6.71 ms [6.67–11.26] |
| 300 | 1 | 6.96 ms [6.85–7.02] | 6.89 ms [6.81–6.96] | 8.05 ms [7.97–8.12] | 4.65 ms [4.57–9.04] | 114.54 ms [107.80–122.39] | incorrect | 135.05 ms [133.47–137.21] | 11.43 ms [11.28–11.58] | 17.80 ms [17.57–17.88] |
| 300 | 2 | 21.12 ms [20.93–21.49] | 21.04 ms [20.87–21.15] | 23.79 ms [23.72–24.23] | 19.84 ms [16.26–22.62] | 160.21 ms [157.85–162.76] | incorrect | 173.63 ms [172.68–177.85] | 20.88 ms [20.77–26.42] | 33.88 ms [33.26–34.55] |

## wav_float / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | stempeg | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.22 ms [0.20–0.24] | 0.16 ms [0.16–0.17] | unsupported | 0.19 ms [0.17–0.19] | unsupported | unsupported | 91.74 ms [91.14–92.82] | 0.19 ms [0.18–0.21] | 0.36 ms [0.35–0.38] |
| 1 | 2 | 0.30 ms [0.27–0.30] | 0.22 ms [0.21–0.23] | unsupported | 0.24 ms [0.21–4.82] | unsupported | unsupported | 91.78 ms [91.65–93.90] | 0.21 ms [0.20–0.22] | 0.42 ms [0.41–0.44] |
| 10 | 1 | 0.21 ms [0.21–0.22] | 0.18 ms [0.17–0.22] | unsupported | 0.20 ms [0.19–0.22] | unsupported | unsupported | 93.17 ms [92.32–97.01] | 0.19 ms [0.17–0.22] | 0.46 ms [0.44–0.47] |
| 10 | 2 | 0.30 ms [0.27–0.42] | 0.22 ms [0.22–0.23] | unsupported | 0.27 ms [0.25–4.34] | unsupported | unsupported | 93.97 ms [93.37–97.55] | 0.21 ms [0.21–0.21] | 0.58 ms [0.56–0.61] |
| 60 | 1 | 0.22 ms [0.22–0.23] | 0.18 ms [0.16–0.22] | unsupported | 0.26 ms [0.25–0.27] | unsupported | unsupported | 93.56 ms [92.87–98.92] | 0.18 ms [0.17–0.23] | 0.45 ms [0.44–0.46] |
| 60 | 2 | 0.28 ms [0.27–0.31] | 0.22 ms [0.20–0.23] | unsupported | 0.38 ms [0.37–4.48] | unsupported | unsupported | 97.15 ms [95.35–99.67] | 0.21 ms [0.21–0.22] | 0.61 ms [0.57–0.72] |
| 300 | 1 | 0.22 ms [0.22–0.22] | 0.17 ms [0.16–0.17] | unsupported | 0.53 ms [0.50–5.07] | unsupported | unsupported | 100.21 ms [99.70–107.96] | 0.18 ms [0.17–0.19] | 0.46 ms [0.45–0.50] |
| 300 | 2 | 0.28 ms [0.26–0.49] | 0.22 ms [0.21–0.25] | unsupported | 0.82 ms [0.81–5.38] | unsupported | unsupported | 103.23 ms [102.68–105.19] | 0.22 ms [0.21–0.22] | 0.62 ms [0.61–0.71] |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | stempeg | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.45 ms [0.43–0.46] | 0.45 ms [0.44–0.50] | unsupported | unsupported | 50.24 ms [49.93–52.83] | 1.10 ms [1.08–1.17] | 91.92 ms [91.24–102.75] | 0.48 ms [0.47–0.49] | 0.70 ms [0.69–0.81] |
| 1 | 2 | 0.70 ms [0.68–0.73] | 0.68 ms [0.66–0.70] | unsupported | unsupported | 50.65 ms [50.20–53.23] | 1.56 ms [1.54–1.59] | 93.86 ms [92.79–109.13] | 0.74 ms [0.72–0.83] | 0.99 ms [0.98–1.00] |
| 10 | 1 | 3.08 ms [3.05–3.32] | 3.20 ms [3.18–3.38] | unsupported | unsupported | 52.25 ms [52.05–56.76] | 4.96 ms [4.93–4.98] | 93.12 ms [92.77–99.58] | 2.51 ms [2.49–2.57] | 3.57 ms [3.54–3.67] |
| 10 | 2 | 5.37 ms [5.34–5.46] | 5.81 ms [5.69–6.03] | unsupported | unsupported | 54.30 ms [53.93–55.42] | 9.64 ms [9.61–9.70] | 96.34 ms [94.70–104.12] | 5.25 ms [5.22–5.63] | 6.62 ms [6.51–10.70] |
| 60 | 1 | 18.55 ms [18.50–18.61] | 19.08 ms [18.81–20.27] | unsupported | unsupported | 62.81 ms [62.50–63.45] | 26.99 ms [26.94–27.47] | 105.36 ms [103.35–118.83] | 14.54 ms [14.49–14.56] | 20.10 ms [20.01–20.24] |
| 60 | 2 | 33.04 ms [32.96–33.18] | 33.37 ms [33.03–34.35] | unsupported | unsupported | 73.84 ms [73.47–81.08] | 54.09 ms [53.94–54.29] | 114.24 ms [113.66–116.22] | 31.06 ms [30.87–32.58] | 37.63 ms [37.49–37.77] |
| 300 | 1 | 93.32 ms [93.01–96.63] | 93.46 ms [93.33–93.51] | unsupported | unsupported | 113.46 ms [113.00–116.84] | 128.89 ms [128.70–130.49] | 156.75 ms [154.72–161.10] | 72.66 ms [72.50–73.09] | 99.25 ms [98.36–100.41] |
| 300 | 2 | 165.59 ms [165.25–167.78] | 166.02 ms [165.21–167.80] | unsupported | unsupported | 167.65 ms [165.57–169.45] | 270.27 ms [267.91–274.72] | 211.92 ms [209.23–224.77] | 154.48 ms [154.18–167.02] | 189.29 ms [188.22–191.52] |

## flac_pcm16 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | stempeg | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.48 ms [0.48–0.50] | 0.43 ms [0.42–0.44] | unsupported | unsupported | unsupported | unsupported | 93.12 ms [91.49–94.24] | 0.47 ms [0.46–0.55] | 0.63 ms [0.62–0.73] |
| 1 | 2 | 0.73 ms [0.70–0.78] | 0.68 ms [0.65–0.72] | unsupported | unsupported | unsupported | unsupported | 95.51 ms [91.96–97.36] | 0.72 ms [0.71–0.73] | 0.91 ms [0.91–0.93] |
| 10 | 1 | 0.58 ms [0.55–0.65] | 0.51 ms [0.50–0.64] | unsupported | unsupported | unsupported | unsupported | 93.55 ms [91.59–99.55] | 0.48 ms [0.47–0.49] | 1.17 ms [1.16–1.22] |
| 10 | 2 | 0.83 ms [0.81–0.84] | 0.81 ms [0.79–0.89] | unsupported | unsupported | unsupported | unsupported | 92.88 ms [92.00–99.77] | 0.79 ms [0.78–0.82] | 2.40 ms [2.37–2.73] |
| 60 | 1 | 0.53 ms [0.51–0.61] | 0.53 ms [0.50–0.55] | unsupported | unsupported | unsupported | unsupported | 94.69 ms [93.40–96.78] | 0.49 ms [0.48–0.54] | 1.14 ms [1.14–1.16] |
| 60 | 2 | 0.87 ms [0.85–0.89] | 0.81 ms [0.79–0.82] | unsupported | unsupported | unsupported | unsupported | 98.25 ms [97.65–105.14] | 0.79 ms [0.78–0.80] | 2.06 ms [2.01–2.17] |
| 300 | 1 | 0.55 ms [0.54–0.62] | 0.49 ms [0.47–0.50] | unsupported | unsupported | unsupported | unsupported | 109.06 ms [107.58–112.79] | 0.49 ms [0.49–0.52] | 1.15 ms [1.14–1.16] |
| 300 | 2 | 0.85 ms [0.83–0.86] | 0.80 ms [0.78–0.82] | unsupported | unsupported | unsupported | unsupported | 132.59 ms [131.03–136.50] | 0.81 ms [0.80–0.82] | 2.39 ms [2.37–3.01] |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | stempeg | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.44 ms [0.43–0.45] | 0.45 ms [0.43–0.47] | unsupported | unsupported | 50.76 ms [50.45–52.63] | 1.15 ms [1.14–1.21] | 92.35 ms [91.63–95.80] | 0.90 ms [0.88–0.92] | 0.88 ms [0.88–0.91] |
| 1 | 2 | 0.70 ms [0.69–0.72] | 0.70 ms [0.69–0.70] | unsupported | unsupported | 51.22 ms [50.95–51.67] | 1.69 ms [1.68–1.76] | 92.94 ms [91.88–100.95] | 1.15 ms [1.15–1.20] | 1.18 ms [1.14–1.22] |
| 10 | 1 | 2.94 ms [2.93–2.97] | 3.08 ms [2.97–3.20] | unsupported | unsupported | 56.40 ms [56.15–65.73] | 7.70 ms [7.64–7.74] | 97.82 ms [97.28–102.98] | 6.97 ms [6.91–7.04] | 4.83 ms [4.81–4.85] |
| 10 | 2 | 5.40 ms [5.36–5.70] | 5.86 ms [5.85–5.88] | unsupported | unsupported | 59.90 ms [59.37–67.72] | 13.47 ms [13.22–13.65] | 100.92 ms [99.63–103.21] | 9.78 ms [9.72–9.80] | 7.58 ms [7.53–7.68] |
| 60 | 1 | 17.60 ms [17.53–17.62] | 18.35 ms [18.34–18.42] | unsupported | unsupported | 85.18 ms [84.25–87.23] | 44.46 ms [44.32–44.67] | 126.76 ms [125.41–131.99] | 40.44 ms [40.38–40.72] | 26.84 ms [26.72–27.44] |
| 60 | 2 | 32.19 ms [32.17–32.29] | 33.22 ms [33.12–33.54] | unsupported | unsupported | 107.06 ms [106.51–111.39] | 76.99 ms [75.80–78.35] | 149.33 ms [148.69–152.81] | 57.21 ms [57.03–59.09] | 42.77 ms [42.70–43.24] |
| 300 | 1 | 89.04 ms [88.19–91.22] | 88.47 ms [88.29–89.24] | unsupported | unsupported | 226.44 ms [224.16–232.76] | 220.35 ms [219.35–221.43] | 268.78 ms [267.37–282.77] | 201.27 ms [200.86–202.62] | 132.79 ms [132.54–134.29] |
| 300 | 2 | 165.23 ms [164.94–165.93] | 165.43 ms [165.20–168.12] | unsupported | unsupported | 334.22 ms [331.52–339.69] | 380.98 ms [380.05–383.40] | 382.94 ms [380.35–385.27] | 285.99 ms [284.53–399.20] | 211.96 ms [211.24–214.68] |

## mp3 / seek

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | pydub | audioread | stempeg | pedalboard | torchcodec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.52 ms [0.50–0.60] | 0.45 ms [0.45–0.50] | unsupported | unsupported | unsupported | unsupported | 92.71 ms [91.91–99.98] | 0.87 ms [0.87–0.89] | 0.84 ms [0.80–1.10] |
| 1 | 2 | 0.75 ms [0.74–0.76] | 0.69 ms [0.69–0.70] | unsupported | unsupported | unsupported | unsupported | 92.13 ms [92.01–100.22] | 1.17 ms [1.15–1.26] | 1.06 ms [1.05–1.08] |
| 10 | 1 | 0.67 ms [0.66–0.67] | 0.67 ms [0.61–0.88] | unsupported | unsupported | unsupported | unsupported | 95.25 ms [94.76–98.14] | 4.57 ms [4.56–4.64] | 0.95 ms [0.90–1.02] |
| 10 | 2 | 0.85 ms [0.84–0.86] | 0.86 ms [0.82–0.88] | unsupported | unsupported | unsupported | unsupported | 95.57 ms [94.09–97.90] | 3.67 ms [3.66–3.81] | 1.20 ms [1.17–1.48] |
| 60 | 1 | 1.67 ms [1.66–1.69] | 1.69 ms [1.67–1.85] | unsupported | unsupported | unsupported | unsupported | 113.54 ms [112.95–119.77] | 29.52 ms [29.50–29.62] | 1.44 ms [1.43–1.46] |
| 60 | 2 | 1.45 ms [1.41–1.45] | 1.42 ms [1.40–1.52] | unsupported | unsupported | unsupported | unsupported | 113.66 ms [112.22–121.55] | 23.80 ms [23.75–23.90] | 1.55 ms [1.51–1.64] |
| 300 | 1 | 2.96 ms [2.95–3.02] | 2.98 ms [2.96–3.00] | unsupported | unsupported | unsupported | unsupported | 140.57 ms [138.66–153.04] | 62.00 ms [61.95–63.84] | 2.13 ms [2.10–2.15] |
| 300 | 2 | 3.44 ms [3.39–3.46] | 3.48 ms [3.45–3.53] | unsupported | unsupported | unsupported | unsupported | 176.21 ms [175.05–179.62] | 92.60 ms [91.84–132.02] | 2.48 ms [2.45–2.61] |

## Unavailable and incorrect libraries

Libraries that failed the correctness gate for at least one cell:

- **audioread** (wav_float, full): max abs diff 1.53e-05 exceeds tolerance (rtol=1e-05, atol=1e-07)

## Measurement noise

Across 424 `ok` measurements, the observed spread `(max - min) / median` ranges from 0.2% to 2350.9%, with a median of 5.5%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings.
