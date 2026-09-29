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
| FLAC seektables | present (seeking can jump) |

## Libraries

| Library | Version | Available | Seek | Bytes | Notes | Error |
| --- | --- | --- | --- | --- | --- | --- |
| soundfile | 0.14.0 | yes | yes | yes | the correctness reference every other loader is checked against | - |
| librosa | 1.0.0 | yes | yes | yes | - | - |
| scipy | 1.18.1 | yes | no | yes | WAV only; scipy.io.wavfile has no seek API | - |
| scipy_mmap | 1.18.1 | yes | yes | no | WAV only, memory-mapped; scipy's mmap mode cannot open 24-bit ('3-byte container') WAV, which surfaces as a decode error for that subtype | - |
| audioread | 3.1.0 | yes | no | no | full-file only, no seek API; audioread always yields 16-bit PCM buffers, so it cannot pass the exact gate for 24-bit or float32 WAV sources | - |
| pedalboard | 0.9.25 | yes | yes | yes | - | - |
| torchcodec | 0.16.0 | yes | yes | yes | imports cleanly even when its native FFmpeg bindings can't load; only the decode smoke test below catches that (see docs/refactor-design.md) | - |
| audiolab | 0.5.2 | yes | yes | yes | - | - |
| audiosample | 2.2.12 | yes | yes | yes | integer-PCM WAV only: float WAV, FLAC and MP3 go through its PyAV path, which is incompatible with PyAV 18 (Flags.FAST_SEEK) | - |
| sphn | 0.2.1 | yes | yes | no | MP3 decode returns a different frame count than soundfile's reference (decoder-delay disagreement); graded by the relaxed MP3 gate. Opus decode dispatches to sphn.read_opus (sphn.read raises on Opus), which returns 960 extra samples -- 20 ms at 48 kHz -- versus soundfile's reference, also graded by the relaxed gate; read_opus takes no start/duration arguments, so sphn has no seek for Opus though it seeks wav/flac/mp3. sphn.read_opus_bytes exists and works, but sphn has no in-memory decode for the other three containers, so from_bytes is left unset rather than wired up for Opus alone | - |

**MP3 is graded by a relaxed gate** (decoded duration within 50 ms of the reference, RMS level within 0.5 dB on the common length) rather than the sample-exact gate WAV and FLAC are held to (1.5 LSB for integer PCM, `atol=1e-7` for float32), because MP3 decoders disagree on encoder delay and never match sample-for-sample. MP3 results therefore carry a weaker correctness guarantee than the WAV/FLAC results in this report.

## wav_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.32 ms [0.29–0.39] | 0.30 ms [0.28–0.40] | 0.15 ms [0.14–0.19] | 0.28 ms [0.22–5.10] | 0.27 ms [0.26–0.33] | 0.22 ms [0.20–0.24] | 5.72 ms [5.58–6.36] | 0.32 ms [0.30–0.39] | 0.15 ms [0.14–0.19] | 0.19 ms [0.17–3.22] |
| 1 | 2 | 0.42 ms [0.39–0.59] | 0.40 ms [0.38–0.42] | 0.19 ms [0.18–0.22] | 0.28 ms [0.26–4.88] | 0.38 ms [0.36–0.42] | 0.21 ms [0.20–0.25] | 10.46 ms [10.36–11.43] | 0.43 ms [0.40–0.49] | 0.24 ms [0.20–0.44] | 0.28 ms [0.23–0.38] |
| 10 | 1 | 1.07 ms [0.93–1.26] | 0.92 ms [0.92–0.99] | 0.45 ms [0.41–4.73] | 0.41 ms [0.35–4.51] | 1.48 ms [1.45–1.51] | 0.45 ms [0.41–0.54] | 13.27 ms [13.05–15.94] | 1.15 ms [1.13–1.27] | 0.45 ms [0.40–0.53] | 0.80 ms [0.79–0.81] |
| 10 | 2 | 2.02 ms [1.87–2.16] | 1.89 ms [1.86–1.97] | 0.95 ms [0.92–1.02] | 0.95 ms [0.89–4.73] | 2.72 ms [2.65–2.74] | 0.79 ms [0.79–0.88] | 25.09 ms [25.00–25.12] | 2.27 ms [2.17–40.81] | 0.94 ms [0.89–1.03] | 1.64 ms [1.49–1.97] |
| 60 | 1 | 4.60 ms [4.54–6.88] | 4.88 ms [4.49–5.80] | 1.84 ms [1.78–1.95] | 1.55 ms [1.53–6.16] | 8.20 ms [7.95–8.41] | 2.03 ms [2.02–2.33] | 15.65 ms [15.28–17.35] | 6.29 ms [5.77–13.16] | 1.81 ms [1.71–2.17] | 7.27 ms [7.14–7.37] |
| 60 | 2 | 10.73 ms [10.38–11.34] | 10.88 ms [10.19–19.77] | 4.88 ms [4.83–5.56] | 4.61 ms [4.30–9.67] | 15.19 ms [15.15–15.76] | 4.13 ms [4.10–4.23] | 29.77 ms [29.28–29.99] | 12.59 ms [11.85–13.98] | 4.78 ms [4.74–5.16] | 10.99 ms [10.43–11.38] |
| 300 | 1 | 22.77 ms [22.65–22.98] | 20.72 ms [20.67–21.59] | 8.61 ms [8.51–8.74] | 7.76 ms [6.82–21.62] | 41.19 ms [39.03–44.35] | 9.64 ms [9.52–9.68] | 27.42 ms [27.22–27.58] | 27.30 ms [27.19–27.70] | 9.11 ms [8.41–9.94] | 30.84 ms [29.23–31.64] |
| 300 | 2 | 51.35 ms [50.98–54.53] | 49.30 ms [48.27–52.67] | 24.08 ms [23.99–24.23] | 20.93 ms [20.14–24.99] | 74.59 ms [74.45–79.48] | 20.74 ms [20.25–21.34] | 51.14 ms [49.06–54.58] | 57.52 ms [57.44–58.10] | 25.71 ms [23.94–26.74] | 72.11 ms [57.88–84.41] |

## wav_pcm16 / seek

| Duration (s) | Channels | Chunk (s) | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.0 | 0.38 ms [0.35–2.00] | 0.31 ms [0.26–0.56] | unsupported | 0.24 ms [0.21–5.08] | unsupported | 0.23 ms [0.20–0.24] | 5.29 ms [5.25–5.52] | 0.29 ms [0.28–0.30] | 0.20 ms [0.15–0.26] | 0.15 ms [0.14–0.19] |
| 1 | 2 | 1.0 | 0.47 ms [0.46–1.11] | 0.37 ms [0.36–0.50] | unsupported | 0.29 ms [0.26–0.34] | unsupported | 0.25 ms [0.20–0.29] | 11.27 ms [11.03–11.49] | 0.39 ms [0.38–0.40] | 0.23 ms [0.20–0.25] | 0.26 ms [0.25–0.38] |
| 10 | 1 | 1.0 | 0.37 ms [0.34–0.58] | 0.29 ms [0.27–0.33] | unsupported | 0.25 ms [0.20–9.71] | unsupported | 0.19 ms [0.17–0.45] | 12.52 ms [12.50–13.98] | 0.29 ms [0.29–0.31] | 0.15 ms [0.14–0.20] | 0.19 ms [0.15–0.25] |
| 10 | 1 | 3.0 | 0.54 ms [0.49–0.67] | 0.44 ms [0.42–0.59] | unsupported | 0.30 ms [0.25–5.20] | unsupported | 0.30 ms [0.28–0.32] | 12.66 ms [12.57–13.17] | 0.49 ms [0.49–0.50] | 0.22 ms [0.20–4.13] | 0.28 ms [0.27–0.30] |
| 10 | 1 | 10.0 | 1.23 ms [0.98–2.34] | 0.97 ms [0.91–1.06] | unsupported | 0.43 ms [0.40–5.29] | unsupported | 0.49 ms [0.46–0.54] | 12.89 ms [12.79–12.99] | 1.12 ms [1.11–1.13] | 0.49 ms [0.38–0.70] | 0.75 ms [0.73–0.84] |
| 10 | 2 | 1.0 | 0.51 ms [0.46–0.70] | 0.38 ms [0.36–0.43] | unsupported | 0.35 ms [0.27–5.23] | unsupported | 0.21 ms [0.20–0.25] | 24.19 ms [24.13–24.55] | 0.53 ms [0.48–0.76] | 0.28 ms [0.20–0.32] | 0.25 ms [0.22–0.27] |
| 10 | 2 | 3.0 | 0.88 ms [0.80–0.97] | 0.76 ms [0.73–1.01] | unsupported | 0.40 ms [0.39–0.51] | unsupported | 0.32 ms [0.31–0.35] | 24.36 ms [24.28–24.38] | 0.91 ms [0.88–1.99] | 0.41 ms [0.36–0.66] | 0.55 ms [0.51–0.62] |
| 10 | 2 | 10.0 | 2.29 ms [2.05–3.92] | 1.98 ms [1.89–2.24] | unsupported | 1.18 ms [0.91–7.50] | unsupported | 0.78 ms [0.78–0.82] | 24.85 ms [24.79–24.89] | 2.22 ms [2.17–2.36] | 0.88 ms [0.87–1.23] | 1.43 ms [1.39–1.56] |
| 60 | 1 | 1.0 | 0.35 ms [0.34–0.43] | 0.31 ms [0.29–0.35] | unsupported | 4.79 ms [0.27–5.18] | unsupported | 0.17 ms [0.16–0.18] | 12.53 ms [12.30–12.92] | 0.34 ms [0.32–0.39] | 0.17 ms [0.15–0.25] | 0.15 ms [0.14–0.18] |
| 60 | 1 | 3.0 | 0.53 ms [0.51–0.72] | 0.50 ms [0.42–0.54] | unsupported | 0.37 ms [0.33–0.54] | unsupported | 0.22 ms [0.22–0.23] | 12.42 ms [12.35–12.67] | 0.59 ms [0.53–0.76] | 0.25 ms [0.20–0.44] | 0.28 ms [0.28–0.29] |
| 60 | 1 | 10.0 | 1.00 ms [0.97–1.86] | 0.95 ms [0.90–1.55] | unsupported | 5.08 ms [0.46–5.19] | unsupported | 0.44 ms [0.44–0.47] | 12.70 ms [12.69–13.66] | 1.25 ms [1.16–1.51] | 0.45 ms [0.41–0.50] | 0.74 ms [0.73–0.74] |
| 60 | 1 | 30.0 | 2.58 ms [2.53–2.81] | 2.58 ms [2.32–3.19] | unsupported | 0.95 ms [0.92–5.09] | unsupported | 1.08 ms [1.06–1.26] | 13.93 ms [13.70–14.07] | 3.12 ms [3.04–3.58] | 0.96 ms [0.93–1.04] | 2.11 ms [2.06–2.56] |
| 60 | 2 | 1.0 | 0.51 ms [0.45–0.64] | 0.39 ms [0.36–0.72] | unsupported | 5.04 ms [0.34–5.93] | unsupported | 0.27 ms [0.23–0.36] | 26.30 ms [25.43–27.53] | 0.50 ms [0.47–0.99] | 0.21 ms [0.20–0.30] | 0.21 ms [0.20–0.21] |
| 60 | 2 | 3.0 | 0.79 ms [0.76–0.88] | 0.72 ms [0.71–0.87] | unsupported | 0.56 ms [0.45–8.31] | unsupported | 0.40 ms [0.36–0.49] | 25.87 ms [25.42–26.47] | 0.91 ms [0.85–1.18] | 0.42 ms [0.35–0.48] | 0.48 ms [0.47–0.49] |
| 60 | 2 | 10.0 | 2.06 ms [1.94–2.26] | 2.04 ms [1.94–2.25] | unsupported | 1.01 ms [0.94–6.51] | unsupported | 0.83 ms [0.79–0.88] | 26.69 ms [25.41–27.75] | 2.36 ms [2.28–3.01] | 0.89 ms [0.87–0.97] | 1.37 ms [1.36–1.43] |
| 60 | 2 | 30.0 | 5.46 ms [5.33–5.72] | 5.45 ms [5.11–12.64] | unsupported | 2.33 ms [2.27–6.93] | unsupported | 2.14 ms [2.11–2.18] | 27.69 ms [26.96–28.86] | 6.60 ms [6.30–6.94] | 2.54 ms [2.45–2.68] | 3.94 ms [3.92–4.01] |
| 300 | 1 | 1.0 | 0.36 ms [0.34–0.48] | 0.29 ms [0.25–0.31] | unsupported | 0.45 ms [0.40–5.20] | unsupported | 0.18 ms [0.15–0.19] | 12.49 ms [12.40–12.84] | 0.29 ms [0.28–0.39] | 0.15 ms [0.14–0.16] | 0.15 ms [0.14–0.16] |
| 300 | 1 | 3.0 | 0.55 ms [0.50–0.82] | 0.40 ms [0.37–0.52] | unsupported | 4.69 ms [0.50–5.35] | unsupported | 0.25 ms [0.22–0.45] | 12.60 ms [12.57–12.76] | 0.61 ms [0.58–4.84] | 0.23 ms [0.19–0.27] | 0.27 ms [0.27–0.28] |
| 300 | 1 | 10.0 | 1.02 ms [0.99–3.73] | 0.93 ms [0.88–0.94] | unsupported | 0.61 ms [0.59–6.32] | unsupported | 0.53 ms [0.47–0.70] | 12.92 ms [12.85–13.08] | 1.29 ms [1.18–1.91] | 0.38 ms [0.37–0.40] | 0.74 ms [0.73–0.79] |
| 300 | 1 | 30.0 | 2.53 ms [2.46–5.25] | 2.31 ms [2.21–7.15] | unsupported | 1.07 ms [1.03–5.51] | unsupported | 1.44 ms [1.17–1.88] | 13.92 ms [13.83–18.02] | 3.05 ms [3.00–3.13] | 0.93 ms [0.90–1.01] | 2.09 ms [2.08–2.54] |
| 300 | 2 | 1.0 | 0.48 ms [0.45–0.59] | 0.37 ms [0.34–0.39] | unsupported | 0.66 ms [0.63–7.06] | unsupported | 0.21 ms [0.20–0.24] | 24.56 ms [24.41–24.88] | 0.39 ms [0.38–0.42] | 0.21 ms [0.20–0.26] | 0.29 ms [0.26–0.36] |
| 300 | 2 | 3.0 | 0.83 ms [0.77–1.02] | 0.71 ms [0.69–0.76] | unsupported | 0.79 ms [0.75–0.84] | unsupported | 0.32 ms [0.32–0.34] | 24.62 ms [24.43–25.32] | 0.78 ms [0.78–0.80] | 0.37 ms [0.35–0.72] | 0.50 ms [0.48–0.66] |
| 300 | 2 | 10.0 | 1.96 ms [1.93–2.13] | 1.83 ms [1.80–1.95] | unsupported | 5.69 ms [1.31–6.67] | unsupported | 0.79 ms [0.78–0.81] | 25.12 ms [25.00–25.21] | 2.14 ms [2.12–5.75] | 1.00 ms [0.91–1.36] | 1.57 ms [1.44–1.97] |
| 300 | 2 | 30.0 | 5.56 ms [5.45–6.07] | 5.00 ms [4.98–6.03] | unsupported | 2.67 ms [2.60–7.90] | unsupported | 2.33 ms [2.18–2.86] | 26.99 ms [26.79–29.34] | 6.02 ms [5.99–6.09] | 2.50 ms [2.47–5.75] | 4.01 ms [3.95–12.21] |

## wav_pcm16 / bytes

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.24 ms [0.21–0.29] | 0.23 ms [0.21–0.35] | 0.07 ms [0.06–0.10] | unsupported | unsupported | 0.29 ms [0.26–0.40] | 5.73 ms [5.47–17.73] | 0.24 ms [0.23–0.31] | 0.10 ms [0.07–0.12] | unsupported |
| 1 | 2 | 0.36 ms [0.31–0.53] | 0.30 ms [0.29–0.37] | 0.10 ms [0.10–0.12] | unsupported | unsupported | 0.29 ms [0.27–0.33] | 11.04 ms [10.69–11.53] | 0.33 ms [0.33–0.36] | 0.12 ms [0.10–3.98] | unsupported |
| 10 | 1 | 0.82 ms [0.79–1.16] | 0.79 ms [0.77–0.97] | 0.23 ms [0.22–0.25] | unsupported | unsupported | 0.53 ms [0.52–1.73] | 12.98 ms [12.92–13.16] | 1.01 ms [1.00–1.03] | 0.28 ms [0.26–0.32] | unsupported |
| 10 | 2 | 1.77 ms [1.69–2.00] | 1.80 ms [1.66–2.00] | 0.65 ms [0.64–5.57] | unsupported | unsupported | 0.74 ms [0.72–0.79] | 24.87 ms [24.82–24.96] | 2.09 ms [2.02–2.44] | 0.61 ms [0.60–0.73] | unsupported |
| 60 | 1 | 4.29 ms [3.96–5.26] | 4.05 ms [3.92–5.57] | 1.41 ms [1.14–3.91] | unsupported | unsupported | 1.79 ms [1.76–1.83] | 15.11 ms [15.06–15.40] | 5.53 ms [5.36–5.93] | 1.15 ms [1.11–1.30] | unsupported |
| 60 | 2 | 9.86 ms [9.50–11.54] | 9.51 ms [8.97–11.85] | 3.67 ms [3.65–3.85] | unsupported | unsupported | 3.63 ms [3.45–3.77] | 30.33 ms [29.84–31.48] | 11.33 ms [11.18–12.12] | 3.60 ms [3.56–4.04] | unsupported |
| 300 | 1 | 20.54 ms [19.88–21.18] | 18.85 ms [18.59–35.68] | 5.52 ms [5.47–5.94] | unsupported | unsupported | 8.14 ms [8.05–10.53] | 27.03 ms [26.44–31.22] | 27.74 ms [26.98–29.72] | 5.53 ms [5.41–7.90] | unsupported |
| 300 | 2 | 47.03 ms [46.45–50.02] | 44.14 ms [43.98–44.55] | 18.27 ms [18.03–19.03] | unsupported | unsupported | 16.81 ms [16.73–18.06] | 47.74 ms [47.13–48.33] | 53.85 ms [53.44–54.68] | 18.15 ms [17.84–19.89] | unsupported |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.61 ms [0.57–0.72] | 0.60 ms [0.57–4.60] | unsupported | unsupported | 1.90 ms [1.85–2.00] | 0.54 ms [0.51–0.69] | 1.01 ms [0.93–2.10] | 0.59 ms [0.56–4.30] | unsupported | 0.27 ms [0.26–4.23] |
| 1 | 2 | 0.88 ms [0.84–1.13] | 0.81 ms [0.78–0.82] | unsupported | unsupported | 2.30 ms [2.24–2.49] | 0.78 ms [0.76–0.88] | 1.31 ms [1.19–1.39] | 0.82 ms [0.80–0.84] | unsupported | 0.51 ms [0.49–0.56] |
| 10 | 1 | 3.87 ms [3.67–4.02] | 3.56 ms [3.39–3.64] | unsupported | unsupported | 5.87 ms [5.77–5.97] | 2.85 ms [2.77–3.11] | 3.89 ms [3.83–4.13] | 3.71 ms [3.66–3.76] | unsupported | 1.78 ms [1.76–1.90] |
| 10 | 2 | 6.44 ms [6.06–6.99] | 6.28 ms [6.10–6.55] | unsupported | unsupported | 10.46 ms [10.31–10.70] | 5.39 ms [5.35–5.50] | 6.82 ms [6.81–6.92] | 6.35 ms [6.27–7.38] | unsupported | 3.55 ms [3.52–3.64] |
| 60 | 1 | 20.43 ms [20.14–20.81] | 19.85 ms [19.73–20.46] | unsupported | unsupported | 27.75 ms [27.41–27.96] | 14.76 ms [14.74–14.86] | 20.71 ms [20.44–20.97] | 23.04 ms [21.84–41.63] | unsupported | 11.38 ms [10.81–12.26] |
| 60 | 2 | 35.87 ms [35.46–36.47] | 33.71 ms [33.56–36.04] | unsupported | unsupported | 56.58 ms [55.42–142.88] | 31.14 ms [31.05–31.20] | 38.80 ms [37.94–40.35] | 37.48 ms [36.95–38.66] | unsupported | 24.07 ms [22.66–24.76] |
| 300 | 1 | 107.25 ms [99.93–111.12] | 96.32 ms [94.24–182.66] | unsupported | unsupported | 134.63 ms [129.49–137.36] | 72.92 ms [72.68–79.19] | 117.40 ms [105.04–137.51] | 108.00 ms [105.38–109.94] | unsupported | 62.00 ms [56.57–104.51] |
| 300 | 2 | 183.90 ms [178.69–189.59] | 168.69 ms [166.35–176.26] | unsupported | unsupported | 273.20 ms [270.00–275.00] | 154.92 ms [154.35–160.62] | 189.41 ms [188.76–202.08] | 186.24 ms [185.27–188.73] | unsupported | 119.97 ms [117.54–125.10] |

## flac_pcm16 / seek

| Duration (s) | Channels | Chunk (s) | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.0 | 0.72 ms [0.62–1.22] | 0.58 ms [0.53–0.72] | unsupported | unsupported | unsupported | 0.57 ms [0.52–0.81] | 0.85 ms [0.83–1.57] | 0.67 ms [0.58–0.78] | unsupported | 0.30 ms [0.27–0.33] |
| 1 | 2 | 1.0 | 0.95 ms [0.89–1.42] | 0.80 ms [0.78–0.99] | unsupported | unsupported | unsupported | 0.76 ms [0.75–0.77] | 1.05 ms [1.01–1.15] | 0.80 ms [0.79–0.82] | unsupported | 0.50 ms [0.47–0.52] |
| 10 | 1 | 1.0 | 0.76 ms [0.69–0.98] | 0.65 ms [0.63–0.86] | unsupported | unsupported | unsupported | 0.63 ms [0.57–0.77] | 1.28 ms [1.24–1.37] | 0.62 ms [0.61–0.63] | unsupported | 0.31 ms [0.27–0.32] |
| 10 | 1 | 3.0 | 1.40 ms [1.28–5.59] | 1.30 ms [1.24–1.62] | unsupported | unsupported | unsupported | 1.04 ms [1.03–1.07] | 1.97 ms [1.92–2.03] | 1.33 ms [1.31–1.33] | unsupported | 0.63 ms [0.60–0.65] |
| 10 | 1 | 10.0 | 4.06 ms [3.57–4.86] | 3.49 ms [3.43–3.79] | unsupported | unsupported | unsupported | 2.68 ms [2.66–2.70] | 3.72 ms [3.69–3.78] | 3.66 ms [3.64–3.66] | unsupported | 1.76 ms [1.70–1.83] |
| 10 | 2 | 1.0 | 1.09 ms [1.02–1.38] | 1.06 ms [1.00–2.43] | unsupported | unsupported | unsupported | 0.83 ms [0.80–0.88] | 2.16 ms [2.12–5.10] | 1.07 ms [1.02–2.06] | unsupported | 0.51 ms [0.48–0.52] |
| 10 | 2 | 3.0 | 2.31 ms [2.18–3.36] | 2.37 ms [2.14–2.70] | unsupported | unsupported | unsupported | 1.83 ms [1.81–1.88] | 4.08 ms [4.07–4.11] | 2.45 ms [2.38–4.33] | unsupported | 1.33 ms [1.18–3.16] |
| 10 | 2 | 10.0 | 6.45 ms [6.06–6.93] | 6.17 ms [6.01–6.27] | unsupported | unsupported | unsupported | 5.42 ms [5.35–5.48] | 6.65 ms [6.59–6.67] | 6.40 ms [6.27–6.86] | unsupported | 3.54 ms [3.44–3.70] |
| 60 | 1 | 1.0 | 0.69 ms [0.64–0.89] | 0.63 ms [0.56–1.59] | unsupported | unsupported | unsupported | 0.52 ms [0.49–0.65] | 1.32 ms [1.26–1.52] | 0.72 ms [0.64–0.85] | unsupported | 0.28 ms [0.26–0.33] |
| 60 | 1 | 3.0 | 1.45 ms [1.35–1.51] | 1.28 ms [1.24–1.40] | unsupported | unsupported | unsupported | 1.03 ms [1.01–1.04] | 2.77 ms [2.66–2.83] | 1.55 ms [1.42–1.72] | unsupported | 0.63 ms [0.62–0.68] |
| 60 | 1 | 10.0 | 3.98 ms [3.57–4.39] | 3.66 ms [3.53–4.65] | unsupported | unsupported | unsupported | 2.70 ms [2.68–2.71] | 4.46 ms [4.41–4.54] | 3.90 ms [3.85–4.07] | unsupported | 1.79 ms [1.78–1.81] |
| 60 | 1 | 30.0 | 10.78 ms [10.24–15.15] | 10.13 ms [10.05–10.36] | unsupported | unsupported | unsupported | 7.52 ms [7.50–7.61] | 11.20 ms [10.86–12.06] | 11.46 ms [11.04–24.92] | unsupported | 5.07 ms [5.00–5.23] |
| 60 | 2 | 1.0 | 1.03 ms [0.96–1.39] | 0.84 ms [0.80–0.88] | unsupported | unsupported | unsupported | 0.82 ms [0.81–0.84] | 4.55 ms [4.52–4.58] | 0.94 ms [0.93–0.99] | unsupported | 0.49 ms [0.48–0.56] |
| 60 | 2 | 3.0 | 2.25 ms [2.21–3.79] | 1.99 ms [1.96–2.11] | unsupported | unsupported | unsupported | 1.87 ms [1.86–1.88] | 4.84 ms [4.82–7.94] | 2.24 ms [2.23–2.27] | unsupported | 1.19 ms [1.18–1.24] |
| 60 | 2 | 10.0 | 6.41 ms [6.29–6.75] | 5.90 ms [5.83–6.03] | unsupported | unsupported | unsupported | 5.44 ms [5.42–29.40] | 9.15 ms [8.66–9.53] | 6.56 ms [6.52–6.59] | unsupported | 3.47 ms [3.45–3.52] |
| 60 | 2 | 30.0 | 18.87 ms [18.58–22.22] | 17.02 ms [16.94–17.82] | unsupported | unsupported | unsupported | 18.91 ms [16.05–37.61] | 20.79 ms [20.64–21.31] | 19.76 ms [19.67–20.08] | unsupported | 12.11 ms [11.40–12.32] |
| 300 | 1 | 1.0 | 0.77 ms [0.72–3.35] | 0.60 ms [0.55–0.63] | unsupported | unsupported | unsupported | 0.67 ms [0.57–4.09] | 1.57 ms [1.45–1.60] | 0.65 ms [0.62–0.69] | unsupported | 0.36 ms [0.34–0.40] |
| 300 | 1 | 3.0 | 1.38 ms [1.33–1.53] | 1.19 ms [1.17–1.24] | unsupported | unsupported | unsupported | 1.02 ms [1.01–1.22] | 2.06 ms [2.00–2.66] | 1.40 ms [1.39–1.72] | unsupported | 0.72 ms [0.66–0.90] |
| 300 | 1 | 10.0 | 3.81 ms [3.74–5.76] | 3.45 ms [3.35–3.93] | unsupported | unsupported | unsupported | 2.95 ms [2.85–8.02] | 4.76 ms [4.67–4.78] | 3.82 ms [3.78–3.94] | unsupported | 1.81 ms [1.79–1.99] |
| 300 | 1 | 30.0 | 10.43 ms [10.09–10.69] | 9.71 ms [9.66–11.56] | unsupported | unsupported | unsupported | 7.60 ms [7.55–7.66] | 11.49 ms [11.22–13.61] | 10.89 ms [10.84–10.93] | unsupported | 5.35 ms [5.32–9.90] |
| 300 | 2 | 1.0 | 1.07 ms [1.01–1.52] | 0.90 ms [0.83–1.00] | unsupported | unsupported | unsupported | 0.83 ms [0.83–0.86] | 2.15 ms [2.11–2.15] | 0.90 ms [0.89–0.91] | unsupported | 0.47 ms [0.46–0.56] |
| 300 | 2 | 3.0 | 2.14 ms [2.07–2.37] | 2.09 ms [1.99–2.14] | unsupported | unsupported | unsupported | 1.86 ms [1.85–1.88] | 3.53 ms [3.46–6.02] | 2.23 ms [2.19–2.25] | unsupported | 1.18 ms [1.16–1.33] |
| 300 | 2 | 10.0 | 6.38 ms [6.32–7.23] | 5.99 ms [5.94–6.71] | unsupported | unsupported | unsupported | 5.51 ms [5.47–5.79] | 8.62 ms [8.56–8.84] | 6.51 ms [6.49–6.64] | unsupported | 3.68 ms [3.54–9.64] |
| 300 | 2 | 30.0 | 18.67 ms [17.96–63.29] | 16.92 ms [16.83–17.19] | unsupported | unsupported | unsupported | 15.79 ms [15.70–20.44] | 20.59 ms [20.44–26.61] | 19.35 ms [19.29–22.69] | unsupported | 12.25 ms [11.34–19.59] |

## flac_pcm16 / bytes

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.53 ms [0.50–0.63] | 0.50 ms [0.48–5.94] | unsupported | unsupported | unsupported | 0.50 ms [0.48–0.54] | 1.16 ms [0.93–2.03] | 0.53 ms [0.51–0.57] | unsupported | unsupported |
| 1 | 2 | 0.78 ms [0.76–0.86] | 0.72 ms [0.68–0.90] | unsupported | unsupported | unsupported | 0.71 ms [0.70–0.74] | 1.36 ms [1.17–1.68] | 0.75 ms [0.74–0.81] | unsupported | unsupported |
| 10 | 1 | 3.73 ms [3.40–43.61] | 3.32 ms [3.25–3.45] | unsupported | unsupported | unsupported | 2.49 ms [2.47–2.54] | 3.76 ms [3.73–3.84] | 3.54 ms [3.53–3.54] | unsupported | unsupported |
| 10 | 2 | 6.30 ms [5.83–6.43] | 6.12 ms [5.76–6.47] | unsupported | unsupported | unsupported | 5.01 ms [5.00–5.04] | 6.65 ms [6.62–6.66] | 6.19 ms [6.16–7.23] | unsupported | unsupported |
| 60 | 1 | 20.54 ms [20.26–21.16] | 19.40 ms [19.35–19.60] | unsupported | unsupported | unsupported | 13.81 ms [13.75–14.07] | 20.11 ms [19.96–21.24] | 21.07 ms [20.93–24.30] | unsupported | unsupported |
| 60 | 2 | 35.24 ms [34.76–109.71] | 32.68 ms [32.63–34.12] | unsupported | unsupported | unsupported | 38.63 ms [29.43–49.18] | 37.54 ms [37.22–37.92] | 36.26 ms [36.20–36.33] | unsupported | unsupported |
| 300 | 1 | 99.99 ms [97.55–100.52] | 100.41 ms [94.56–103.23] | unsupported | unsupported | unsupported | 69.32 ms [68.11–74.76] | 99.76 ms [98.14–106.04] | 103.32 ms [102.96–103.43] | unsupported | unsupported |
| 300 | 2 | 175.48 ms [171.70–178.59] | 163.12 ms [162.64–174.70] | unsupported | unsupported | unsupported | 146.31 ms [143.69–173.66] | 186.36 ms [185.69–264.91] | 182.05 ms [181.60–186.65] | unsupported | unsupported |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.59 ms [0.57–0.62] | 0.62 ms [0.57–0.77] | unsupported | unsupported | 1.56 ms [1.53–6.36] | 1.02 ms [0.97–1.07] | 1.18 ms [1.13–1.81] | 0.57 ms [0.56–0.60] | unsupported | 0.44 ms [0.42–0.52] |
| 1 | 2 | 0.90 ms [0.85–0.96] | 0.87 ms [0.81–0.89] | unsupported | unsupported | 2.12 ms [2.09–2.23] | 1.28 ms [1.24–1.36] | 1.43 ms [1.38–1.49] | 0.87 ms [0.84–0.95] | unsupported | 0.75 ms [0.74–0.82] |
| 10 | 1 | 3.64 ms [3.45–4.40] | 3.43 ms [3.27–3.52] | unsupported | unsupported | 8.44 ms [8.30–9.60] | 7.09 ms [6.99–7.44] | 5.16 ms [5.07–5.54] | 3.63 ms [3.61–3.71] | unsupported | 3.40 ms [3.32–3.63] |
| 10 | 2 | 6.56 ms [6.01–6.73] | 6.55 ms [6.09–6.65] | unsupported | unsupported | 13.64 ms [13.41–13.92] | 9.91 ms [9.83–12.04] | 8.55 ms [8.04–8.70] | 6.95 ms [6.42–8.19] | unsupported | 6.57 ms [6.34–6.81] |
| 60 | 1 | 19.99 ms [19.73–21.80] | 19.02 ms [18.99–19.30] | unsupported | unsupported | 46.17 ms [44.96–66.89] | 40.54 ms [40.48–66.81] | 27.21 ms [27.05–28.70] | 21.15 ms [20.65–24.76] | unsupported | 21.23 ms [21.02–21.36] |
| 60 | 2 | 36.81 ms [36.04–39.34] | 33.63 ms [33.60–35.37] | unsupported | unsupported | 81.59 ms [78.37–171.96] | 58.39 ms [57.35–86.05] | 43.89 ms [43.22–100.25] | 36.99 ms [36.88–37.07] | unsupported | 40.74 ms [40.65–43.26] |
| 300 | 1 | 96.45 ms [94.56–132.24] | 93.45 ms [90.65–106.21] | unsupported | unsupported | 222.20 ms [221.19–239.52] | 201.09 ms [200.90–216.87] | 134.05 ms [133.42–139.04] | 103.01 ms [102.71–104.76] | unsupported | 103.61 ms [103.21–104.68] |
| 300 | 2 | 180.62 ms [175.87–216.07] | 169.63 ms [166.20–173.35] | unsupported | unsupported | 381.49 ms [378.96–510.93] | 292.82 ms [286.15–313.07] | 218.31 ms [211.43–242.23] | 183.74 ms [183.27–190.40] | unsupported | 207.80 ms [201.90–222.74] |

## mp3 / seek

| Duration (s) | Channels | Chunk (s) | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.0 | 0.71 ms [0.64–1.14] | 0.56 ms [0.55–0.71] | unsupported | unsupported | unsupported | 0.95 ms [0.94–0.97] | 1.05 ms [1.00–1.08] | 0.54 ms [0.53–0.55] | unsupported | 0.45 ms [0.43–0.54] |
| 1 | 2 | 1.0 | 0.96 ms [0.91–1.59] | 0.82 ms [0.73–1.00] | unsupported | unsupported | unsupported | 1.23 ms [1.22–1.26] | 1.31 ms [1.25–1.40] | 0.84 ms [0.83–0.87] | unsupported | 0.79 ms [0.77–0.87] |
| 10 | 1 | 1.0 | 0.88 ms [0.81–1.20] | 0.77 ms [0.72–0.97] | unsupported | unsupported | unsupported | 4.67 ms [4.58–5.12] | 1.12 ms [1.04–1.28] | 0.72 ms [0.72–0.74] | unsupported | 0.49 ms [0.47–0.63] |
| 10 | 1 | 3.0 | 1.53 ms [1.47–1.87] | 1.39 ms [1.35–1.75] | unsupported | unsupported | unsupported | 5.97 ms [5.92–6.08] | 2.03 ms [1.95–2.34] | 1.44 ms [1.43–1.55] | unsupported | 1.12 ms [1.07–1.41] |
| 10 | 1 | 10.0 | 3.51 ms [3.45–3.86] | 3.32 ms [3.28–3.54] | unsupported | unsupported | unsupported | 6.98 ms [6.94–7.01] | 5.07 ms [4.89–5.36] | 3.67 ms [3.61–3.75] | unsupported | 3.32 ms [3.27–3.75] |
| 10 | 2 | 1.0 | 1.28 ms [1.11–1.36] | 1.05 ms [1.01–1.32] | unsupported | unsupported | unsupported | 5.92 ms [5.87–8.29] | 1.30 ms [1.28–1.37] | 1.13 ms [1.04–1.38] | unsupported | 0.79 ms [0.78–0.83] |
| 10 | 2 | 3.0 | 2.39 ms [2.27–2.46] | 2.23 ms [2.20–2.45] | unsupported | unsupported | unsupported | 8.80 ms [8.79–8.95] | 2.76 ms [2.72–2.94] | 2.39 ms [2.35–2.43] | unsupported | 2.03 ms [1.99–2.67] |
| 10 | 2 | 10.0 | 6.50 ms [6.26–7.05] | 7.08 ms [6.03–7.69] | unsupported | unsupported | unsupported | 9.82 ms [9.80–9.84] | 7.55 ms [7.49–7.59] | 7.00 ms [6.43–7.71] | unsupported | 7.01 ms [6.42–10.89] |
| 60 | 1 | 1.0 | 2.06 ms [1.95–2.30] | 1.79 ms [1.78–2.01] | unsupported | unsupported | unsupported | 28.54 ms [28.46–28.86] | 1.72 ms [1.62–1.93] | 1.88 ms [1.76–1.96] | unsupported | 0.52 ms [0.51–0.53] |
| 60 | 1 | 3.0 | 2.35 ms [2.13–2.45] | 2.09 ms [2.02–2.64] | unsupported | unsupported | unsupported | 21.35 ms [21.31–21.48] | 2.49 ms [2.28–2.96] | 2.22 ms [2.18–2.29] | unsupported | 1.13 ms [1.11–1.13] |
| 60 | 1 | 10.0 | 4.29 ms [4.06–5.71] | 4.11 ms [3.88–23.22] | unsupported | unsupported | unsupported | 19.58 ms [19.50–19.85] | 5.08 ms [5.07–5.30] | 4.76 ms [4.33–6.30] | unsupported | 3.31 ms [3.29–3.37] |
| 60 | 1 | 30.0 | 10.77 ms [10.59–11.91] | 10.65 ms [10.49–12.17] | unsupported | unsupported | unsupported | 32.00 ms [31.92–32.07] | 13.73 ms [13.63–13.84] | 11.38 ms [11.07–12.49] | unsupported | 9.54 ms [9.51–9.65] |
| 60 | 2 | 1.0 | 2.01 ms [1.75–2.50] | 1.61 ms [1.57–1.66] | unsupported | unsupported | unsupported | 26.97 ms [26.64–39.26] | 1.86 ms [1.69–5.19] | 1.65 ms [1.63–1.66] | unsupported | 0.80 ms [0.80–0.83] |
| 60 | 2 | 3.0 | 3.08 ms [2.82–4.12] | 2.71 ms [2.64–2.88] | unsupported | unsupported | unsupported | 34.61 ms [28.49–77.40] | 3.13 ms [3.11–6.19] | 2.86 ms [2.84–2.88] | unsupported | 2.02 ms [2.01–2.02] |
| 60 | 2 | 10.0 | 7.32 ms [7.13–9.50] | 6.56 ms [6.51–6.83] | unsupported | unsupported | unsupported | 38.40 ms [33.64–47.89] | 7.88 ms [7.83–7.99] | 7.17 ms [7.12–7.40] | unsupported | 6.29 ms [6.28–6.31] |
| 60 | 2 | 30.0 | 19.52 ms [18.67–65.11] | 18.27 ms [17.66–20.50] | unsupported | unsupported | unsupported | 53.79 ms [52.63–62.03] | 21.86 ms [21.78–24.32] | 19.45 ms [19.43–19.89] | unsupported | 19.33 ms [19.14–20.43] |
| 300 | 1 | 1.0 | 6.97 ms [6.88–9.67] | 6.86 ms [6.64–12.13] | unsupported | unsupported | unsupported | 139.23 ms [138.99–152.91] | 3.83 ms [3.74–3.96] | 6.47 ms [6.45–6.64] | unsupported | 0.96 ms [0.93–1.01] |
| 300 | 1 | 3.0 | 4.45 ms [4.07–9.39] | 3.96 ms [3.80–4.47] | unsupported | unsupported | unsupported | 62.88 ms [62.70–68.49] | 3.07 ms [3.04–3.40] | 4.07 ms [3.89–8.03] | unsupported | 1.30 ms [1.27–1.53] |
| 300 | 1 | 10.0 | 9.91 ms [9.07–12.02] | 9.53 ms [8.78–13.64] | unsupported | unsupported | unsupported | 130.51 ms [130.35–131.45] | 7.92 ms [7.42–11.46] | 9.01 ms [8.93–9.98] | unsupported | 3.71 ms [3.68–3.78] |
| 300 | 1 | 30.0 | 15.14 ms [13.43–17.34] | 13.21 ms [12.72–14.06] | unsupported | unsupported | unsupported | 96.78 ms [96.64–96.91] | 15.26 ms [15.05–15.59] | 14.96 ms [13.99–20.94] | unsupported | 10.00 ms [9.86–17.24] |
| 300 | 2 | 1.0 | 4.15 ms [4.06–4.85] | 3.94 ms [3.84–4.09] | unsupported | unsupported | unsupported | 103.82 ms [100.37–130.98] | 2.74 ms [2.67–2.82] | 4.01 ms [3.90–11.63] | unsupported | 1.27 ms [1.04–5.65] |
| 300 | 2 | 3.0 | 6.34 ms [6.20–6.58] | 5.86 ms [5.82–6.00] | unsupported | unsupported | unsupported | 133.87 ms [131.10–232.38] | 4.50 ms [4.46–9.96] | 6.14 ms [6.00–6.20] | unsupported | 2.31 ms [2.27–2.55] |
| 300 | 2 | 10.0 | 9.92 ms [9.71–12.63] | 9.17 ms [9.11–18.59] | unsupported | unsupported | unsupported | 121.65 ms [118.06–126.97] | 9.13 ms [9.09–9.15] | 9.82 ms [9.74–13.83] | unsupported | 6.98 ms [6.60–14.14] |
| 300 | 2 | 30.0 | 25.86 ms [24.45–28.41] | 23.04 ms [22.75–23.31] | unsupported | unsupported | unsupported | 216.77 ms [215.97–221.34] | 24.49 ms [24.39–29.96] | 24.54 ms [24.47–25.32] | unsupported | 20.49 ms [19.72–22.64] |

## mp3 / bytes

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.51 ms [0.48–0.74] | 0.55 ms [0.48–0.89] | unsupported | unsupported | unsupported | 0.93 ms [0.93–0.97] | 1.13 ms [1.06–1.25] | 0.48 ms [0.48–0.49] | unsupported | unsupported |
| 1 | 2 | 0.79 ms [0.76–0.92] | 0.77 ms [0.76–0.82] | unsupported | unsupported | unsupported | 1.24 ms [1.23–1.30] | 1.49 ms [1.29–2.01] | 0.78 ms [0.77–0.81] | unsupported | unsupported |
| 10 | 1 | 3.21 ms [3.15–6.30] | 3.21 ms [3.09–3.37] | unsupported | unsupported | unsupported | 6.99 ms [6.92–7.47] | 5.80 ms [5.00–9.26] | 3.47 ms [3.45–3.53] | unsupported | unsupported |
| 10 | 2 | 6.06 ms [5.95–7.05] | 6.26 ms [5.90–6.52] | unsupported | unsupported | unsupported | 9.79 ms [9.78–10.00] | 7.96 ms [7.76–8.19] | 6.35 ms [6.23–6.61] | unsupported | unsupported |
| 60 | 1 | 18.84 ms [18.32–21.04] | 20.35 ms [18.69–24.93] | unsupported | unsupported | unsupported | 40.29 ms [40.22–40.44] | 26.78 ms [26.73–27.61] | 20.05 ms [19.74–20.95] | unsupported | unsupported |
| 60 | 2 | 35.63 ms [34.64–37.14] | 33.89 ms [33.59–35.86] | unsupported | unsupported | unsupported | 65.60 ms [61.00–72.81] | 42.58 ms [42.51–43.00] | 35.99 ms [35.97–36.15] | unsupported | unsupported |
| 300 | 1 | 93.04 ms [90.67–98.36] | 100.04 ms [91.37–179.46] | unsupported | unsupported | unsupported | 201.13 ms [200.39–202.77] | 133.90 ms [132.68–142.27] | 100.56 ms [97.49–101.29] | unsupported | unsupported |
| 300 | 2 | 177.07 ms [174.92–184.27] | 170.12 ms [162.06–177.31] | unsupported | unsupported | unsupported | 299.19 ms [287.00–308.27] | 212.22 ms [211.69–217.69] | 180.41 ms [178.83–186.93] | unsupported | unsupported |

## opus / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 5.26 ms [4.88–5.70] | 5.17 ms [4.88–43.58] | unsupported | unsupported | 3.06 ms [2.87–3.19] | 3.13 ms [3.11–10.50] | 1.75 ms [1.64–1.99] | 6.21 ms [6.19–6.24] | unsupported | 1.01 ms [0.99–1.04] |
| 1 | 2 | 7.54 ms [7.32–8.31] | 7.39 ms [7.08–7.94] | unsupported | unsupported | 3.99 ms [3.87–4.22] | 4.02 ms [3.93–4.16] | 2.21 ms [2.12–2.38] | 9.43 ms [9.32–9.70] | unsupported | 1.47 ms [1.43–1.63] |
| 10 | 1 | 28.97 ms [28.66–29.58] | 28.13 ms [27.96–29.19] | unsupported | unsupported | 15.19 ms [14.86–19.30] | 15.34 ms [15.26–15.67] | 9.87 ms [9.63–10.13] | 28.83 ms [28.79–29.03] | unsupported | 9.37 ms [8.85–9.59] |
| 10 | 2 | 45.34 ms [43.34–47.69] | 43.59 ms [42.84–47.03] | unsupported | unsupported | 24.04 ms [23.70–24.24] | 23.77 ms [23.71–24.33] | 14.72 ms [14.71–23.58] | 44.03 ms [44.00–45.19] | unsupported | 13.09 ms [13.04–13.86] |
| 60 | 1 | 161.22 ms [159.03–168.00] | 166.88 ms [160.88–168.24] | unsupported | unsupported | 79.48 ms [79.23–84.02] | 82.50 ms [82.38–90.72] | 56.34 ms [54.22–59.34] | 154.64 ms [153.63–156.72] | unsupported | 53.82 ms [53.09–54.73] |
| 60 | 2 | 249.96 ms [246.84–296.55] | 231.22 ms [228.93–241.72] | unsupported | unsupported | 135.73 ms [135.25–147.57] | 135.29 ms [134.53–142.29] | 84.35 ms [84.15–86.69] | 235.29 ms [231.86–238.62] | unsupported | 85.55 ms [83.36–99.18] |
| 300 | 1 | 808.06 ms [785.45–857.28] | 759.58 ms [745.46–810.27] | unsupported | unsupported | 396.38 ms [393.15–407.34] | 408.86 ms [408.13–427.42] | 271.99 ms [265.43–279.86] | 752.36 ms [749.76–762.99] | unsupported | 275.21 ms [265.65–282.00] |
| 300 | 2 | 1269.41 ms [1200.76–1370.39] | 1156.30 ms [1133.07–1223.94] | unsupported | unsupported | 669.33 ms [667.58–678.82] | 690.97 ms [672.66–705.09] | 429.03 ms [418.95–442.29] | 1143.85 ms [1125.13–1163.85] | unsupported | 425.29 ms [413.19–478.98] |

## opus / seek

| Duration (s) | Channels | Chunk (s) | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.0 | 5.28 ms [4.85–6.45] | 5.04 ms [4.73–5.41] | unsupported | unsupported | unsupported | 3.08 ms [3.04–3.12] | 1.59 ms [1.42–1.70] | 4.55 ms [4.54–4.61] | unsupported | unsupported |
| 1 | 2 | 1.0 | 7.91 ms [7.49–11.07] | 7.26 ms [7.03–7.68] | unsupported | unsupported | unsupported | 3.98 ms [3.94–4.20] | 1.97 ms [1.92–5.82] | 6.94 ms [6.93–6.98] | unsupported | unsupported |
| 10 | 1 | 1.0 | 4.97 ms [4.86–5.27] | 4.70 ms [4.58–4.85] | unsupported | unsupported | unsupported | 3.61 ms [3.59–3.62] | 3.14 ms [3.11–3.16] | 4.56 ms [4.54–4.97] | unsupported | unsupported |
| 10 | 1 | 3.0 | 10.41 ms [10.21–10.47] | 9.92 ms [9.82–10.02] | unsupported | unsupported | unsupported | 6.33 ms [6.21–6.45] | 4.96 ms [4.92–5.00] | 9.62 ms [9.56–10.31] | unsupported | unsupported |
| 10 | 1 | 10.0 | 29.13 ms [28.70–79.66] | 28.07 ms [27.86–28.30] | unsupported | unsupported | unsupported | 15.46 ms [15.31–15.94] | 9.48 ms [9.45–9.51] | 27.88 ms [26.95–47.60] | unsupported | unsupported |
| 10 | 2 | 1.0 | 7.60 ms [7.31–9.42] | 7.47 ms [7.09–8.24] | unsupported | unsupported | unsupported | 4.48 ms [4.43–4.51] | 4.01 ms [3.98–4.04] | 6.79 ms [6.72–7.02] | unsupported | unsupported |
| 10 | 2 | 3.0 | 15.30 ms [14.96–15.45] | 16.11 ms [14.80–17.45] | unsupported | unsupported | unsupported | 8.86 ms [8.75–8.93] | 7.03 ms [6.96–7.16] | 14.69 ms [14.63–14.78] | unsupported | unsupported |
| 10 | 2 | 10.0 | 44.75 ms [43.45–46.02] | 46.93 ms [44.36–241.14] | unsupported | unsupported | unsupported | 23.73 ms [23.68–23.90] | 14.63 ms [14.51–15.01] | 41.82 ms [40.95–62.96] | unsupported | unsupported |
| 60 | 1 | 1.0 | 5.09 ms [4.89–5.38] | 5.24 ms [4.94–6.63] | unsupported | unsupported | unsupported | 7.31 ms [7.03–8.22] | 2.85 ms [2.81–3.22] | 4.62 ms [4.57–4.71] | unsupported | unsupported |
| 60 | 1 | 3.0 | 10.36 ms [10.09–11.92] | 10.99 ms [9.99–12.28] | unsupported | unsupported | unsupported | 9.67 ms [9.42–9.99] | 4.46 ms [4.37–6.58] | 9.64 ms [9.47–9.71] | unsupported | unsupported |
| 60 | 1 | 10.0 | 29.07 ms [28.74–32.09] | 28.29 ms [28.13–31.85] | unsupported | unsupported | unsupported | 18.66 ms [18.56–18.78] | 11.69 ms [10.65–12.08] | 32.10 ms [29.94–68.12] | unsupported | unsupported |
| 60 | 1 | 30.0 | 84.45 ms [81.43–92.15] | 80.08 ms [79.72–82.90] | unsupported | unsupported | unsupported | 46.81 ms [44.24–50.53] | 30.20 ms [28.68–30.53] | 97.78 ms [85.56–104.54] | unsupported | unsupported |
| 60 | 2 | 1.0 | 7.46 ms [7.22–7.58] | 6.67 ms [6.63–6.78] | unsupported | unsupported | unsupported | 7.86 ms [7.82–7.94] | 4.80 ms [4.60–5.79] | 6.80 ms [6.79–10.30] | unsupported | unsupported |
| 60 | 2 | 3.0 | 15.43 ms [14.91–17.54] | 14.23 ms [14.17–14.45] | unsupported | unsupported | unsupported | 12.15 ms [12.00–15.33] | 6.56 ms [6.41–7.09] | 14.33 ms [14.28–14.61] | unsupported | unsupported |
| 60 | 2 | 10.0 | 43.61 ms [42.91–49.16] | 40.59 ms [39.98–44.08] | unsupported | unsupported | unsupported | 27.16 ms [27.03–30.46] | 16.91 ms [16.52–17.81] | 41.04 ms [40.30–44.35] | unsupported | unsupported |
| 60 | 2 | 30.0 | 123.36 ms [120.89–167.39] | 118.17 ms [115.52–126.07] | unsupported | unsupported | unsupported | 69.54 ms [69.41–70.71] | 43.81 ms [43.59–44.71] | 118.89 ms [117.07–120.18] | unsupported | unsupported |
| 300 | 1 | 1.0 | 4.85 ms [4.78–5.23] | 4.59 ms [4.52–4.79] | unsupported | unsupported | unsupported | 22.73 ms [22.54–23.98] | 3.68 ms [3.63–6.22] | 4.57 ms [4.56–4.59] | unsupported | unsupported |
| 300 | 1 | 3.0 | 10.54 ms [10.03–11.16] | 10.01 ms [9.54–10.26] | unsupported | unsupported | unsupported | 26.89 ms [25.88–67.52] | 5.06 ms [4.88–5.36] | 9.72 ms [9.68–9.97] | unsupported | unsupported |
| 300 | 1 | 10.0 | 28.66 ms [28.34–28.82] | 26.78 ms [26.71–27.32] | unsupported | unsupported | unsupported | 35.47 ms [34.80–84.86] | 11.92 ms [11.52–17.48] | 27.45 ms [27.40–31.19] | unsupported | unsupported |
| 300 | 1 | 30.0 | 80.35 ms [80.13–81.03] | 80.11 ms [76.47–83.99] | unsupported | unsupported | unsupported | 75.26 ms [63.26–102.40] | 29.24 ms [28.19–30.12] | 77.74 ms [77.50–83.55] | unsupported | unsupported |
| 300 | 2 | 1.0 | 7.20 ms [7.08–7.43] | 6.94 ms [6.69–9.69] | unsupported | unsupported | unsupported | 23.66 ms [23.55–24.80] | 3.81 ms [3.71–3.93] | 6.79 ms [6.74–6.82] | unsupported | unsupported |
| 300 | 2 | 3.0 | 15.17 ms [14.90–15.36] | 14.63 ms [14.15–15.64] | unsupported | unsupported | unsupported | 28.27 ms [27.93–30.85] | 7.92 ms [7.80–9.19] | 14.42 ms [14.25–14.99] | unsupported | unsupported |
| 300 | 2 | 10.0 | 42.49 ms [42.33–48.88] | 40.55 ms [40.20–54.63] | unsupported | unsupported | unsupported | 42.91 ms [42.74–52.06] | 16.03 ms [15.90–22.01] | 42.07 ms [41.96–42.14] | unsupported | unsupported |
| 300 | 2 | 30.0 | 121.68 ms [121.08–158.82] | 118.31 ms [114.81–124.99] | unsupported | unsupported | unsupported | 87.63 ms [85.15–90.17] | 44.05 ms [43.94–44.77] | 117.56 ms [114.87–123.24] | unsupported | unsupported |

## opus / bytes

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 5.08 ms [4.65–5.48] | 4.81 ms [4.62–5.11] | unsupported | unsupported | unsupported | 2.89 ms [2.88–6.06] | 1.52 ms [1.49–1.70] | 6.36 ms [6.17–16.52] | unsupported | unsupported |
| 1 | 2 | 7.52 ms [7.07–8.27] | 6.98 ms [6.93–7.04] | unsupported | unsupported | unsupported | 3.76 ms [3.72–3.77] | 2.24 ms [2.13–2.65] | 9.60 ms [9.54–9.70] | unsupported | unsupported |
| 10 | 1 | 28.71 ms [28.49–29.86] | 28.06 ms [27.92–29.16] | unsupported | unsupported | unsupported | 13.98 ms [13.94–14.03] | 9.59 ms [9.54–9.60] | 28.82 ms [28.71–30.19] | unsupported | unsupported |
| 10 | 2 | 43.71 ms [43.33–59.90] | 43.58 ms [42.31–45.25] | unsupported | unsupported | unsupported | 22.30 ms [22.27–22.35] | 14.61 ms [14.58–14.67] | 43.86 ms [43.33–44.92] | unsupported | unsupported |
| 60 | 1 | 161.52 ms [159.02–239.34] | 161.67 ms [152.50–401.54] | unsupported | unsupported | unsupported | 75.88 ms [75.64–76.48] | 57.95 ms [54.85–58.96] | 158.22 ms [156.00–160.82] | unsupported | unsupported |
| 60 | 2 | 243.53 ms [242.20–254.73] | 232.63 ms [229.42–234.78] | unsupported | unsupported | unsupported | 125.60 ms [125.27–126.47] | 84.64 ms [84.15–86.66] | 237.82 ms [234.25–239.34] | unsupported | unsupported |
| 300 | 1 | 814.98 ms [794.64–848.96] | 750.42 ms [743.54–866.26] | unsupported | unsupported | unsupported | 399.24 ms [370.25–570.67] | 281.90 ms [267.31–571.30] | 754.52 ms [749.85–756.40] | unsupported | unsupported |
| 300 | 2 | 1226.99 ms [1186.91–1318.65] | 1136.18 ms [1128.76–1160.75] | unsupported | unsupported | unsupported | 637.37 ms [628.85–648.37] | 424.20 ms [417.64–527.81] | 1149.38 ms [1137.11–1165.81] | unsupported | unsupported |

## Unavailable and incorrect libraries

Every probed library was available and passed the correctness gate.

## Measurement noise

Across 1024 `ok` measurements, the observed dispersion `IQR / median` ranges from 0.1% to 1386.4%, with a median of 3.4%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings. The interquartile range is reported rather than max-min because the full range grows with the trial count, which would make runs using different `--repeat` values incomparable.
