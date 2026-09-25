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
| torchcodec | 0.16.0+cu130 | yes | yes | yes | imports cleanly even when its native FFmpeg bindings can't load; only the decode smoke test below catches that (see docs/refactor-design.md) | - |
| audiolab | 0.5.2 | yes | yes | yes | - | - |
| audiosample | 2.2.12 | yes | yes | yes | integer-PCM WAV only: float WAV, FLAC and MP3 go through its PyAV path, which is incompatible with PyAV 18 (Flags.FAST_SEEK) | - |
| sphn | 0.2.1 | yes | yes | no | MP3 decode returns a different frame count than soundfile's reference (decoder-delay disagreement); graded by the relaxed MP3 gate | - |

**MP3 is graded by a relaxed gate** (decoded duration within 50 ms of the reference, RMS level within 0.5 dB on the common length) rather than the sample-exact gate WAV and FLAC are held to (1.5 LSB for integer PCM, `atol=1e-7` for float32), because MP3 decoders disagree on encoder delay and never match sample-for-sample. MP3 results therefore carry a weaker correctness guarantee than the WAV/FLAC results in this report.

## wav_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.41 ms [0.38–0.47] | 0.40 ms [0.37–0.42] | 0.24 ms [0.24–0.25] | 0.32 ms [0.31–0.33] | 0.41 ms [0.40–0.41] | 0.19 ms [0.18–0.20] | 6.64 ms [6.59–6.72] | 0.58 ms [0.58–0.61] | 0.23 ms [0.23–0.24] | 0.27 ms [0.26–0.31] |
| 1 | 2 | 0.56 ms [0.55–0.57] | 0.55 ms [0.53–0.58] | 0.34 ms [0.33–0.37] | 0.41 ms [0.39–0.42] | 0.55 ms [0.54–0.56] | 0.24 ms [0.24–0.25] | 7.98 ms [7.88–8.19] | 0.77 ms [0.75–0.79] | 0.34 ms [0.33–0.34] | 0.40 ms [0.39–0.40] |
| 10 | 1 | 1.47 ms [1.46–2.07] | 1.44 ms [1.42–1.49] | 0.79 ms [0.79–0.83] | 0.80 ms [0.79–0.81] | 2.20 ms [2.16–2.25] | 0.70 ms [0.69–0.72] | 9.75 ms [9.68–9.79] | 2.06 ms [2.03–2.10] | 0.66 ms [0.65–0.69] | 1.66 ms [1.64–1.69] |
| 10 | 2 | 3.42 ms [3.39–3.50] | 3.45 ms [3.38–3.51] | 1.82 ms [1.82–1.95] | 1.82 ms [1.79–1.86] | 3.81 ms [3.78–3.85] | 1.32 ms [1.32–1.33] | 11.68 ms [11.50–11.97] | 3.88 ms [3.84–3.93] | 1.67 ms [1.65–1.73] | 3.08 ms [3.05–3.09] |
| 60 | 1 | 7.36 ms [7.29–7.42] | 7.36 ms [7.32–7.47] | 3.99 ms [3.94–4.16] | 3.83 ms [3.78–3.87] | 12.37 ms [12.29–12.47] | 3.61 ms [3.60–3.76] | 18.82 ms [18.15–19.16] | 9.62 ms [9.47–9.83] | 2.98 ms [2.94–3.36] | 9.90 ms [9.85–10.02] |
| 60 | 2 | 19.74 ms [19.63–19.98] | 19.66 ms [19.47–19.82] | 11.56 ms [11.35–11.61] | 11.23 ms [10.97–11.26] | 23.06 ms [22.95–23.11] | 7.47 ms [7.44–7.67] | 29.93 ms [29.74–30.56] | 19.98 ms [19.49–20.56] | 9.35 ms [9.33–9.41] | 19.47 ms [19.39–19.70] |
| 300 | 1 | 44.99 ms [44.66–45.20] | 43.71 ms [43.52–43.96] | 32.37 ms [32.33–32.73] | 28.95 ms [28.89–29.15] | 73.87 ms [73.63–74.28] | 25.65 ms [25.48–27.39] | 80.44 ms [76.24–105.06] | 53.97 ms [53.18–54.22] | 20.89 ms [20.47–20.93] | 83.62 ms [83.41–83.95] |
| 300 | 2 | 128.87 ms [128.57–129.67] | 129.26 ms [128.94–129.54] | 111.93 ms [111.65–113.05] | 98.58 ms [98.37–98.68] | 171.75 ms [171.55–172.06] | 53.07 ms [52.71–54.46] | 154.53 ms [154.10–155.58] | 113.05 ms [111.95–113.27] | 100.51 ms [100.39–101.34] | 154.91 ms [154.50–155.30] |

## wav_pcm16 / seek

| Duration (s) | Channels | Chunk (s) | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.0 | 0.49 ms [0.44–0.52] | 0.41 ms [0.38–0.43] | unsupported | 0.33 ms [0.32–0.33] | unsupported | 0.19 ms [0.19–0.20] | 6.66 ms [6.56–6.84] | 0.56 ms [0.54–0.59] | 0.25 ms [0.25–0.27] | 0.23 ms [0.23–0.26] |
| 1 | 2 | 1.0 | 0.62 ms [0.61–0.72] | 0.57 ms [0.53–0.58] | unsupported | 0.41 ms [0.39–0.43] | unsupported | 0.25 ms [0.24–0.25] | 8.02 ms [7.85–8.16] | 0.74 ms [0.72–0.76] | 0.35 ms [0.34–0.37] | 0.37 ms [0.37–0.39] |
| 10 | 1 | 1.0 | 0.46 ms [0.45–0.51] | 0.39 ms [0.37–0.42] | unsupported | 0.33 ms [0.31–0.35] | unsupported | 0.19 ms [0.19–0.20] | 8.05 ms [7.97–8.23] | 0.55 ms [0.55–0.56] | 0.25 ms [0.24–0.27] | 0.24 ms [0.24–0.25] |
| 10 | 1 | 3.0 | 0.71 ms [0.69–0.74] | 0.63 ms [0.61–0.63] | unsupported | 0.42 ms [0.41–0.43] | unsupported | 0.30 ms [0.29–0.31] | 8.36 ms [8.27–8.45] | 0.97 ms [0.93–0.98] | 0.34 ms [0.32–0.35] | 0.53 ms [0.52–0.54] |
| 10 | 1 | 10.0 | 1.58 ms [1.53–1.65] | 1.43 ms [1.40–1.48] | unsupported | 0.80 ms [0.78–0.82] | unsupported | 0.70 ms [0.69–0.71] | 9.65 ms [9.56–9.72] | 2.02 ms [2.01–2.08] | 0.68 ms [0.67–0.78] | 1.56 ms [1.55–1.59] |
| 10 | 2 | 1.0 | 0.67 ms [0.65–0.95] | 0.55 ms [0.54–0.58] | unsupported | 0.41 ms [0.40–0.42] | unsupported | 0.25 ms [0.24–0.26] | 8.18 ms [8.05–8.22] | 0.74 ms [0.72–0.75] | 0.35 ms [0.34–0.42] | 0.37 ms [0.36–0.38] |
| 10 | 2 | 3.0 | 1.25 ms [1.20–1.26] | 1.15 ms [1.12–1.33] | unsupported | 0.72 ms [0.69–0.79] | unsupported | 0.48 ms [0.47–0.50] | 9.03 ms [8.87–9.13] | 1.52 ms [1.49–1.56] | 0.64 ms [0.62–0.65] | 0.88 ms [0.87–0.88] |
| 10 | 2 | 10.0 | 3.48 ms [3.46–3.55] | 3.44 ms [3.40–3.47] | unsupported | 1.83 ms [1.81–1.90] | unsupported | 1.32 ms [1.30–1.33] | 11.65 ms [11.52–11.89] | 3.86 ms [3.80–3.91] | 1.69 ms [1.67–1.75] | 2.87 ms [2.85–2.95] |
| 60 | 1 | 1.0 | 0.47 ms [0.44–0.50] | 0.38 ms [0.36–0.40] | unsupported | 0.32 ms [0.32–0.35] | unsupported | 0.19 ms [0.18–0.19] | 7.98 ms [7.95–8.11] | 0.56 ms [0.55–0.59] | 0.25 ms [0.25–0.26] | 0.24 ms [0.23–0.24] |
| 60 | 1 | 3.0 | 0.69 ms [0.68–0.75] | 0.63 ms [0.62–0.84] | unsupported | 0.45 ms [0.44–0.47] | unsupported | 0.30 ms [0.29–0.31] | 8.27 ms [8.14–8.37] | 0.96 ms [0.95–0.98] | 0.34 ms [0.33–0.34] | 0.53 ms [0.51–0.54] |
| 60 | 1 | 10.0 | 1.52 ms [1.48–1.58] | 1.48 ms [1.43–1.52] | unsupported | 0.83 ms [0.81–0.85] | unsupported | 0.69 ms [0.69–0.71] | 9.52 ms [9.41–9.63] | 2.06 ms [2.02–2.09] | 0.67 ms [0.66–0.70] | 1.56 ms [1.54–1.61] |
| 60 | 1 | 30.0 | 3.89 ms [3.85–3.93] | 3.79 ms [3.76–3.83] | unsupported | 2.02 ms [2.00–2.10] | unsupported | 1.83 ms [1.81–1.91] | 13.21 ms [13.10–13.39] | 5.03 ms [4.98–5.14] | 1.58 ms [1.56–1.61] | 4.55 ms [4.52–4.63] |
| 60 | 2 | 1.0 | 0.67 ms [0.62–0.68] | 0.54 ms [0.53–0.56] | unsupported | 0.45 ms [0.43–0.47] | unsupported | 0.26 ms [0.25–0.28] | 8.36 ms [8.29–8.39] | 0.77 ms [0.72–0.82] | 0.35 ms [0.34–0.36] | 0.39 ms [0.39–0.39] |
| 60 | 2 | 3.0 | 1.26 ms [1.23–1.31] | 1.15 ms [1.14–1.16] | unsupported | 0.71 ms [0.70–0.75] | unsupported | 0.48 ms [0.47–0.49] | 9.04 ms [8.91–9.22] | 1.54 ms [1.50–1.56] | 0.64 ms [0.62–0.68] | 0.93 ms [0.91–0.94] |
| 60 | 2 | 10.0 | 3.51 ms [3.48–3.61] | 3.42 ms [3.38–3.48] | unsupported | 1.85 ms [1.83–1.89] | unsupported | 1.34 ms [1.33–1.36] | 11.82 ms [11.77–11.92] | 3.89 ms [3.79–4.05] | 1.69 ms [1.67–1.72] | 2.86 ms [2.84–2.89] |
| 60 | 2 | 30.0 | 10.03 ms [9.90–10.23] | 9.88 ms [9.74–10.02] | unsupported | 5.49 ms [5.45–5.61] | unsupported | 3.72 ms [3.69–3.89] | 19.35 ms [19.25–19.49] | 10.32 ms [10.12–10.38] | 4.70 ms [4.69–4.78] | 8.77 ms [8.73–8.82] |
| 300 | 1 | 1.0 | 0.49 ms [0.46–0.51] | 0.38 ms [0.37–0.40] | unsupported | 0.37 ms [0.36–0.37] | unsupported | 0.20 ms [0.19–0.20] | 8.12 ms [8.08–8.26] | 0.55 ms [0.52–0.60] | 0.25 ms [0.25–0.27] | 0.24 ms [0.23–0.25] |
| 300 | 1 | 3.0 | 0.73 ms [0.72–0.75] | 0.63 ms [0.62–0.69] | unsupported | 0.45 ms [0.42–0.46] | unsupported | 0.31 ms [0.30–0.31] | 8.49 ms [8.46–8.51] | 0.97 ms [0.94–1.00] | 0.34 ms [0.34–0.35] | 0.54 ms [0.53–0.55] |
| 300 | 1 | 10.0 | 1.58 ms [1.54–1.60] | 1.47 ms [1.42–1.50] | unsupported | 0.83 ms [0.82–0.85] | unsupported | 0.71 ms [0.70–0.72] | 9.61 ms [9.54–9.73] | 2.06 ms [2.04–2.11] | 0.68 ms [0.67–0.70] | 1.58 ms [1.58–1.62] |
| 300 | 1 | 30.0 | 3.90 ms [3.85–3.98] | 3.85 ms [3.79–3.89] | unsupported | 2.01 ms [2.00–2.08] | unsupported | 1.85 ms [1.84–1.91] | 13.00 ms [12.78–13.08] | 5.19 ms [5.15–5.25] | 1.60 ms [1.57–1.66] | 4.58 ms [4.55–4.64] |
| 300 | 2 | 1.0 | 0.63 ms [0.61–0.68] | 0.58 ms [0.55–0.59] | unsupported | 0.44 ms [0.41–0.60] | unsupported | 0.25 ms [0.24–0.27] | 8.23 ms [8.15–8.35] | 0.72 ms [0.70–0.77] | 0.35 ms [0.34–0.37] | 0.38 ms [0.38–0.39] |
| 300 | 2 | 3.0 | 1.22 ms [1.20–1.26] | 1.17 ms [1.12–1.20] | unsupported | 0.71 ms [0.71–0.75] | unsupported | 0.48 ms [0.48–0.49] | 9.03 ms [8.91–12.67] | 1.52 ms [1.48–1.53] | 0.64 ms [0.63–0.67] | 0.88 ms [0.87–0.91] |
| 300 | 2 | 10.0 | 3.48 ms [3.43–3.61] | 3.44 ms [3.39–3.49] | unsupported | 1.88 ms [1.82–1.89] | unsupported | 1.30 ms [1.29–1.36] | 11.68 ms [11.54–11.86] | 3.85 ms [3.79–3.89] | 1.66 ms [1.64–1.68] | 2.92 ms [2.91–2.93] |
| 300 | 2 | 30.0 | 9.89 ms [9.75–10.01] | 9.81 ms [9.71–9.90] | unsupported | 5.38 ms [5.31–5.59] | unsupported | 3.62 ms [3.61–3.74] | 19.32 ms [19.13–19.75] | 10.18 ms [10.13–10.27] | 4.62 ms [4.62–4.73] | 8.36 ms [8.31–8.46] |

## wav_pcm16 / bytes

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.43 ms [0.42–0.45] | 0.40 ms [0.36–0.41] | 0.18 ms [0.17–0.18] | unsupported | unsupported | 0.18 ms [0.18–0.19] | 6.64 ms [6.59–6.73] | 0.58 ms [0.57–0.60] | 0.16 ms [0.16–0.16] | unsupported |
| 1 | 2 | 0.56 ms [0.54–0.58] | 0.53 ms [0.51–0.56] | 0.27 ms [0.27–0.29] | unsupported | unsupported | 0.23 ms [0.23–0.24] | 7.90 ms [7.80–8.02] | 0.76 ms [0.74–0.78] | 0.25 ms [0.25–0.26] | unsupported |
| 10 | 1 | 1.47 ms [1.42–1.58] | 1.41 ms [1.37–1.49] | 0.68 ms [0.67–0.69] | unsupported | unsupported | 0.70 ms [0.68–0.72] | 9.56 ms [9.43–9.67] | 2.09 ms [2.06–2.16] | 0.53 ms [0.52–0.53] | unsupported |
| 10 | 2 | 3.28 ms [3.20–3.45] | 3.31 ms [3.27–3.34] | 1.65 ms [1.64–1.66] | unsupported | unsupported | 1.28 ms [1.25–1.32] | 11.40 ms [11.24–11.48] | 3.75 ms [3.64–3.90] | 1.42 ms [1.40–1.43] | unsupported |
| 60 | 1 | 6.81 ms [6.60–6.89] | 6.86 ms [6.69–7.04] | 3.62 ms [3.53–3.73] | unsupported | unsupported | 3.59 ms [3.56–3.70] | 18.24 ms [17.40–18.48] | 9.37 ms [9.26–9.61] | 2.66 ms [2.62–2.68] | unsupported |
| 60 | 2 | 18.25 ms [17.93–18.70] | 18.32 ms [18.12–18.55] | 20.52 ms [20.46–20.68] | unsupported | unsupported | 7.35 ms [7.08–8.02] | 28.91 ms [28.54–29.05] | 18.54 ms [18.32–18.77] | 8.55 ms [8.52–8.61] | unsupported |
| 300 | 1 | 42.22 ms [41.85–42.76] | 41.17 ms [40.57–41.50] | 30.86 ms [30.74–30.97] | unsupported | unsupported | 24.93 ms [24.76–25.30] | 101.55 ms [82.74–104.06] | 52.34 ms [51.72–52.95] | 19.71 ms [19.22–19.98] | unsupported |
| 300 | 2 | 123.12 ms [122.29–123.57] | 122.69 ms [122.21–124.06] | 126.15 ms [125.71–126.69] | unsupported | unsupported | 77.65 ms [77.35–78.44] | 149.21 ms [148.23–150.11] | 107.52 ms [106.65–107.79] | 102.93 ms [102.82–103.11] | unsupported |

## flac_pcm16 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.83 ms [0.82–0.84] | 0.79 ms [0.78–0.94] | unsupported | unsupported | 18.36 ms [17.92–18.59] | 0.46 ms [0.45–0.47] | 1.42 ms [1.36–1.53] | 1.04 ms [1.02–1.04] | unsupported | 0.51 ms [0.50–0.52] |
| 1 | 2 | 1.32 ms [1.30–1.40] | 1.27 ms [1.26–1.29] | unsupported | unsupported | 18.92 ms [18.46–19.86] | 0.84 ms [0.82–0.91] | 1.95 ms [1.88–1.97] | 1.56 ms [1.54–1.58] | unsupported | 0.93 ms [0.93–0.94] |
| 10 | 1 | 5.44 ms [5.42–5.51] | 5.40 ms [5.36–5.54] | unsupported | unsupported | 21.26 ms [20.62–22.05] | 3.34 ms [3.32–3.38] | 6.97 ms [6.91–6.99] | 6.19 ms [6.16–6.24] | unsupported | 3.89 ms [3.86–3.90] |
| 10 | 2 | 10.70 ms [10.65–10.79] | 10.62 ms [10.56–10.70] | unsupported | unsupported | 25.12 ms [24.70–26.33] | 7.00 ms [6.99–7.08] | 12.14 ms [12.07–12.18] | 11.38 ms [11.34–11.53] | unsupported | 7.83 ms [7.80–7.87] |
| 60 | 1 | 31.07 ms [31.00–31.16] | 30.92 ms [30.81–31.14] | unsupported | unsupported | 34.63 ms [34.12–35.93] | 19.54 ms [19.42–19.69] | 37.81 ms [37.53–38.47] | 34.76 ms [34.73–34.86] | unsupported | 22.60 ms [22.56–22.65] |
| 60 | 2 | 63.02 ms [62.90–63.14] | 62.90 ms [62.79–62.95] | unsupported | unsupported | 52.41 ms [50.95–53.68] | 41.48 ms [41.31–41.62] | 67.86 ms [67.80–68.00] | 66.41 ms [66.22–66.46] | unsupported | 47.39 ms [47.32–47.48] |
| 300 | 1 | 162.73 ms [162.58–163.59] | 161.64 ms [161.03–161.79] | unsupported | unsupported | 111.13 ms [109.76–112.21] | 104.87 ms [104.56–105.23] | 225.04 ms [220.07–226.05] | 181.19 ms [180.90–181.46] | unsupported | 147.39 ms [146.83–147.62] |
| 300 | 2 | 344.22 ms [343.93–346.36] | 344.41 ms [343.02–344.97] | unsupported | unsupported | 232.41 ms [230.80–236.68] | 222.91 ms [222.36–223.18] | 367.91 ms [367.37–390.59] | 345.13 ms [344.93–345.57] | unsupported | 293.68 ms [293.43–294.56] |

## flac_pcm16 / seek

| Duration (s) | Channels | Chunk (s) | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.0 | 0.90 ms [0.89–0.92] | 0.80 ms [0.79–0.81] | unsupported | unsupported | unsupported | 0.46 ms [0.46–0.49] | 1.38 ms [1.35–1.41] | 1.01 ms [0.99–1.05] | unsupported | 0.49 ms [0.49–0.49] |
| 1 | 2 | 1.0 | 1.41 ms [1.40–1.70] | 1.28 ms [1.27–1.32] | unsupported | unsupported | unsupported | 0.83 ms [0.82–0.85] | 1.94 ms [1.88–1.97] | 1.52 ms [1.49–1.54] | unsupported | 0.90 ms [0.89–0.90] |
| 10 | 1 | 1.0 | 1.04 ms [1.02–1.27] | 0.96 ms [0.95–0.98] | unsupported | unsupported | unsupported | 0.50 ms [0.49–0.50] | 2.25 ms [2.19–2.30] | 1.14 ms [1.12–1.15] | unsupported | 0.49 ms [0.48–0.49] |
| 10 | 1 | 3.0 | 2.00 ms [1.98–2.26] | 1.92 ms [1.90–1.95] | unsupported | unsupported | unsupported | 1.14 ms [1.13–1.15] | 3.49 ms [3.46–3.51] | 2.35 ms [2.34–2.37] | unsupported | 1.21 ms [1.21–1.22] |
| 10 | 1 | 10.0 | 5.57 ms [5.54–5.59] | 5.40 ms [5.36–5.51] | unsupported | unsupported | unsupported | 3.34 ms [3.31–3.37] | 6.99 ms [6.97–7.05] | 6.16 ms [6.15–6.22] | unsupported | 3.71 ms [3.71–3.73] |
| 10 | 2 | 1.0 | 1.75 ms [1.73–1.77] | 1.63 ms [1.62–1.67] | unsupported | unsupported | unsupported | 0.90 ms [0.90–0.92] | 3.61 ms [3.50–4.12] | 1.82 ms [1.82–1.83] | unsupported | 0.91 ms [0.90–0.92] |
| 10 | 2 | 3.0 | 3.71 ms [3.69–3.79] | 3.62 ms [3.60–3.67] | unsupported | unsupported | unsupported | 2.23 ms [2.22–2.26] | 6.94 ms [6.83–7.04] | 4.14 ms [4.12–4.21] | unsupported | 2.41 ms [2.39–2.58] |
| 10 | 2 | 10.0 | 10.78 ms [10.74–10.87] | 10.64 ms [10.59–10.75] | unsupported | unsupported | unsupported | 7.05 ms [6.97–7.09] | 12.18 ms [12.12–12.26] | 11.36 ms [11.34–11.40] | unsupported | 7.62 ms [7.61–7.66] |
| 60 | 1 | 1.0 | 0.98 ms [0.96–1.03] | 0.91 ms [0.89–0.91] | unsupported | unsupported | unsupported | 0.51 ms [0.50–0.53] | 2.23 ms [2.16–2.25] | 1.10 ms [1.06–1.12] | unsupported | 0.50 ms [0.49–0.57] |
| 60 | 1 | 3.0 | 2.04 ms [2.01–2.12] | 1.98 ms [1.96–1.99] | unsupported | unsupported | unsupported | 1.15 ms [1.14–1.17] | 4.53 ms [4.49–4.55] | 2.37 ms [2.34–2.41] | unsupported | 1.29 ms [1.28–1.36] |
| 60 | 1 | 10.0 | 5.64 ms [5.57–5.68] | 5.54 ms [5.53–5.64] | unsupported | unsupported | unsupported | 3.39 ms [3.37–3.41] | 8.06 ms [8.04–8.12] | 6.41 ms [6.40–6.44] | unsupported | 3.78 ms [3.78–3.82] |
| 60 | 1 | 30.0 | 15.88 ms [15.84–15.98] | 15.75 ms [15.66–15.80] | unsupported | unsupported | unsupported | 9.87 ms [9.81–9.96] | 20.11 ms [20.02–20.19] | 18.01 ms [17.95–18.15] | unsupported | 11.02 ms [10.99–11.10] |
| 60 | 2 | 1.0 | 1.60 ms [1.54–1.62] | 1.48 ms [1.45–1.48] | unsupported | unsupported | unsupported | 0.91 ms [0.90–0.92] | 7.04 ms [6.98–7.10] | 1.68 ms [1.67–1.70] | unsupported | 0.98 ms [0.97–0.99] |
| 60 | 2 | 3.0 | 3.72 ms [3.68–3.81] | 3.64 ms [3.59–3.66] | unsupported | unsupported | unsupported | 2.30 ms [2.29–2.31] | 8.05 ms [8.00–8.13] | 4.12 ms [4.10–4.18] | unsupported | 2.48 ms [2.47–2.49] |
| 60 | 2 | 10.0 | 10.97 ms [10.96–11.02] | 10.85 ms [10.82–10.89] | unsupported | unsupported | unsupported | 7.11 ms [7.07–7.18] | 14.97 ms [14.95–15.09] | 11.81 ms [11.77–11.90] | unsupported | 7.78 ms [7.76–7.81] |
| 60 | 2 | 30.0 | 31.88 ms [31.86–32.19] | 31.77 ms [31.62–31.95] | unsupported | unsupported | unsupported | 20.96 ms [20.87–21.12] | 37.02 ms [36.80–37.25] | 34.70 ms [34.66–34.82] | unsupported | 22.89 ms [22.82–23.24] |
| 300 | 1 | 1.0 | 1.03 ms [1.02–1.08] | 0.98 ms [0.95–1.01] | unsupported | unsupported | unsupported | 0.53 ms [0.53–0.55] | 2.57 ms [2.52–2.60] | 1.12 ms [1.08–1.18] | unsupported | 0.53 ms [0.52–0.54] |
| 300 | 1 | 3.0 | 2.10 ms [2.06–2.18] | 1.98 ms [1.96–2.00] | unsupported | unsupported | unsupported | 1.17 ms [1.16–1.17] | 3.53 ms [3.46–3.65] | 2.42 ms [2.41–2.50] | unsupported | 1.25 ms [1.25–1.26] |
| 300 | 1 | 10.0 | 5.68 ms [5.62–5.70] | 5.58 ms [5.54–5.68] | unsupported | unsupported | unsupported | 3.39 ms [3.37–3.46] | 8.35 ms [8.33–8.52] | 6.36 ms [6.32–6.39] | unsupported | 3.81 ms [3.79–4.29] |
| 300 | 1 | 30.0 | 15.92 ms [15.80–16.06] | 15.81 ms [15.71–15.86] | unsupported | unsupported | unsupported | 9.87 ms [9.80–9.96] | 20.48 ms [20.30–20.52] | 18.03 ms [17.98–18.13] | unsupported | 11.04 ms [11.02–11.07] |
| 300 | 2 | 1.0 | 1.59 ms [1.58–1.60] | 1.53 ms [1.52–1.55] | unsupported | unsupported | unsupported | 0.93 ms [0.93–0.93] | 3.68 ms [3.66–3.71] | 1.70 ms [1.65–1.72] | unsupported | 0.92 ms [0.91–0.95] |
| 300 | 2 | 3.0 | 3.62 ms [3.59–3.64] | 3.52 ms [3.50–3.56] | unsupported | unsupported | unsupported | 2.27 ms [2.26–2.29] | 6.02 ms [5.97–6.11] | 4.13 ms [4.08–4.15] | unsupported | 2.38 ms [2.38–2.41] |
| 300 | 2 | 10.0 | 10.92 ms [10.87–10.99] | 10.87 ms [10.82–10.88] | unsupported | unsupported | unsupported | 7.12 ms [7.04–7.19] | 15.11 ms [15.07–15.17] | 11.80 ms [11.79–11.88] | unsupported | 7.73 ms [7.71–7.74] |
| 300 | 2 | 30.0 | 31.74 ms [31.65–31.89] | 31.59 ms [31.57–31.78] | unsupported | unsupported | unsupported | 20.73 ms [20.66–20.83] | 36.99 ms [36.91–37.11] | 34.52 ms [34.40–34.59] | unsupported | 22.59 ms [22.57–22.66] |

## flac_pcm16 / bytes

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.83 ms [0.82–0.85] | 0.79 ms [0.78–0.80] | unsupported | unsupported | unsupported | 0.46 ms [0.45–0.49] | 1.39 ms [1.37–1.43] | 1.04 ms [1.02–1.06] | unsupported | unsupported |
| 1 | 2 | 1.33 ms [1.31–1.35] | 1.28 ms [1.27–1.30] | unsupported | unsupported | unsupported | 0.80 ms [0.79–0.85] | 1.91 ms [1.88–1.93] | 1.55 ms [1.54–1.56] | unsupported | unsupported |
| 10 | 1 | 5.53 ms [5.51–5.57] | 5.50 ms [5.45–5.51] | unsupported | unsupported | unsupported | 3.23 ms [3.22–3.38] | 6.88 ms [6.83–6.99] | 6.31 ms [6.28–6.32] | unsupported | unsupported |
| 10 | 2 | 10.69 ms [10.66–10.73] | 10.68 ms [10.60–10.72] | unsupported | unsupported | unsupported | 6.88 ms [6.77–7.02] | 11.94 ms [11.87–12.00] | 11.47 ms [11.45–11.55] | unsupported | unsupported |
| 60 | 1 | 31.66 ms [31.60–31.78] | 31.54 ms [31.50–31.63] | unsupported | unsupported | unsupported | 19.21 ms [19.03–19.51] | 37.24 ms [37.10–37.81] | 35.70 ms [35.64–35.83] | unsupported | unsupported |
| 60 | 2 | 63.14 ms [63.01–63.21] | 63.13 ms [63.02–63.21] | unsupported | unsupported | unsupported | 40.88 ms [40.76–41.27] | 67.16 ms [67.06–67.41] | 66.92 ms [66.87–67.05] | unsupported | unsupported |
| 300 | 1 | 165.91 ms [165.63–166.08] | 165.49 ms [165.43–166.66] | unsupported | unsupported | unsupported | 103.20 ms [102.58–104.10] | 219.43 ms [202.64–222.32] | 185.90 ms [185.63–186.13] | unsupported | unsupported |
| 300 | 2 | 347.33 ms [346.78–347.83] | 346.48 ms [346.28–346.83] | unsupported | unsupported | unsupported | 241.87 ms [241.12–242.73] | 387.12 ms [386.14–387.76] | 348.94 ms [348.70–349.52] | unsupported | unsupported |

## mp3 / full

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.07 ms [1.07–1.08] | 1.07 ms [1.06–1.13] | unsupported | unsupported | 17.88 ms [16.92–18.99] | 1.92 ms [1.90–1.94] | 1.79 ms [1.76–1.81] | 1.28 ms [1.27–1.30] | unsupported | 0.84 ms [0.83–0.86] |
| 1 | 2 | 1.65 ms [1.64–1.68] | 1.64 ms [1.63–1.67] | unsupported | unsupported | 18.63 ms [17.31–19.18] | 2.72 ms [2.69–2.76] | 2.28 ms [2.25–2.34] | 1.88 ms [1.84–1.91] | unsupported | 1.53 ms [1.52–1.55] |
| 10 | 1 | 7.85 ms [7.82–7.98] | 7.84 ms [7.83–7.90] | unsupported | unsupported | 27.46 ms [26.47–28.35] | 17.33 ms [17.25–17.52] | 9.10 ms [9.08–9.23] | 8.91 ms [8.90–9.02] | unsupported | 6.86 ms [6.78–6.90] |
| 10 | 2 | 13.51 ms [13.47–13.60] | 13.47 ms [13.40–13.55] | unsupported | unsupported | 31.87 ms [30.91–33.09] | 24.95 ms [24.75–25.02] | 13.41 ms [13.35–13.60] | 14.68 ms [14.65–14.79] | unsupported | 13.55 ms [13.50–13.57] |
| 60 | 1 | 45.43 ms [45.37–45.58] | 45.44 ms [45.42–45.57] | unsupported | unsupported | 71.69 ms [70.76–72.09] | 103.60 ms [103.22–103.79] | 49.23 ms [49.21–49.49] | 50.90 ms [50.86–51.07] | unsupported | 40.06 ms [39.95–40.26] |
| 60 | 2 | 79.56 ms [79.41–79.99] | 79.73 ms [79.50–80.04] | unsupported | unsupported | 103.84 ms [102.90–104.83] | 148.60 ms [148.19–148.77] | 74.90 ms [74.75–74.95] | 84.73 ms [84.60–84.88] | unsupported | 81.70 ms [81.57–82.02] |
| 300 | 1 | 231.85 ms [231.63–232.10] | 231.76 ms [231.62–231.94] | unsupported | unsupported | 294.28 ms [290.80–297.77] | 522.50 ms [521.73–523.77] | 255.80 ms [255.62–256.19] | 260.92 ms [260.30–262.36] | unsupported | 236.11 ms [235.82–236.65] |
| 300 | 2 | 425.24 ms [425.04–425.40] | 423.08 ms [422.54–423.39] | unsupported | unsupported | 499.37 ms [497.17–503.40] | 752.81 ms [751.66–753.22] | 404.28 ms [403.78–404.92] | 436.84 ms [436.63–437.27] | unsupported | 468.21 ms [468.05–469.10] |

## mp3 / seek

| Duration (s) | Channels | Chunk (s) | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.0 | 1.17 ms [1.16–1.39] | 1.08 ms [1.07–1.11] | unsupported | unsupported | unsupported | 1.91 ms [1.89–1.92] | 1.75 ms [1.75–1.78] | 1.26 ms [1.25–1.28] | unsupported | 0.80 ms [0.79–0.82] |
| 1 | 2 | 1.0 | 1.74 ms [1.71–1.76] | 1.65 ms [1.64–1.66] | unsupported | unsupported | unsupported | 2.67 ms [2.66–2.73] | 2.28 ms [2.24–2.33] | 1.83 ms [1.80–1.88] | unsupported | 1.49 ms [1.48–1.50] |
| 10 | 1 | 1.0 | 1.41 ms [1.41–1.44] | 1.32 ms [1.32–1.33] | unsupported | unsupported | unsupported | 11.13 ms [11.08–11.31] | 2.01 ms [2.00–2.06] | 1.52 ms [1.49–1.54] | unsupported | 0.84 ms [0.83–0.85] |
| 10 | 1 | 3.0 | 2.90 ms [2.88–2.91] | 2.80 ms [2.78–2.82] | unsupported | unsupported | unsupported | 14.58 ms [14.53–14.66] | 3.55 ms [3.52–3.60] | 3.33 ms [3.30–3.34] | unsupported | 2.17 ms [2.14–2.21] |
| 10 | 1 | 10.0 | 7.93 ms [7.90–7.94] | 7.83 ms [7.81–7.86] | unsupported | unsupported | unsupported | 17.31 ms [17.23–17.41] | 9.05 ms [9.02–9.10] | 8.93 ms [8.89–8.97] | unsupported | 6.67 ms [6.64–6.89] |
| 10 | 2 | 1.0 | 1.99 ms [1.96–2.00] | 1.89 ms [1.88–1.95] | unsupported | unsupported | unsupported | 14.65 ms [14.58–14.74] | 2.50 ms [2.44–2.55] | 2.08 ms [2.05–2.09] | unsupported | 1.51 ms [1.51–1.53] |
| 10 | 2 | 3.0 | 4.66 ms [4.65–4.83] | 4.55 ms [4.53–4.60] | unsupported | unsupported | unsupported | 22.11 ms [22.03–22.27] | 5.07 ms [5.01–5.14] | 5.18 ms [5.17–5.20] | unsupported | 4.20 ms [4.19–4.22] |
| 10 | 2 | 10.0 | 13.54 ms [13.48–13.70] | 13.48 ms [13.39–13.67] | unsupported | unsupported | unsupported | 24.87 ms [24.69–25.02] | 13.66 ms [13.64–13.81] | 14.63 ms [14.60–14.82] | unsupported | 13.34 ms [13.31–13.47] |
| 60 | 1 | 1.0 | 2.63 ms [2.60–2.71] | 2.53 ms [2.51–2.56] | unsupported | unsupported | unsupported | 72.06 ms [71.94–72.25] | 3.11 ms [3.04–3.14] | 2.71 ms [2.69–2.73] | unsupported | 1.04 ms [1.03–1.05] |
| 60 | 1 | 3.0 | 3.71 ms [3.67–3.73] | 3.62 ms [3.60–3.63] | unsupported | unsupported | unsupported | 54.00 ms [53.84–54.09] | 4.31 ms [4.25–4.41] | 4.12 ms [4.11–4.13] | unsupported | 2.29 ms [2.27–2.30] |
| 60 | 1 | 10.0 | 8.66 ms [8.64–8.80] | 8.56 ms [8.54–8.67] | unsupported | unsupported | unsupported | 49.44 ms [49.31–49.61] | 9.37 ms [9.36–9.50] | 9.62 ms [9.60–9.68] | unsupported | 6.87 ms [6.79–7.04] |
| 60 | 1 | 30.0 | 23.66 ms [23.60–23.73] | 23.55 ms [23.53–23.61] | unsupported | unsupported | unsupported | 81.20 ms [81.00–81.53] | 24.64 ms [24.58–24.66] | 26.41 ms [26.34–26.52] | unsupported | 19.93 ms [19.89–20.07] |
| 60 | 2 | 1.0 | 2.75 ms [2.71–2.77] | 2.66 ms [2.65–2.74] | unsupported | unsupported | unsupported | 67.80 ms [67.64–68.06] | 3.20 ms [3.07–3.25] | 2.84 ms [2.82–2.90] | unsupported | 1.63 ms [1.63–1.65] |
| 60 | 2 | 3.0 | 5.28 ms [5.26–5.33] | 5.20 ms [5.18–5.23] | unsupported | unsupported | unsupported | 67.37 ms [67.19–67.47] | 5.59 ms [5.55–5.63] | 5.81 ms [5.75–5.86] | unsupported | 4.24 ms [4.20–4.24] |
| 60 | 2 | 10.0 | 14.51 ms [14.49–14.58] | 14.41 ms [14.37–14.45] | unsupported | unsupported | unsupported | 83.87 ms [83.58–83.99] | 14.18 ms [14.13–14.34] | 15.66 ms [15.55–15.79] | unsupported | 13.55 ms [13.54–13.57] |
| 60 | 2 | 30.0 | 41.04 ms [40.89–42.06] | 40.92 ms [40.80–40.98] | unsupported | unsupported | unsupported | 135.94 ms [135.52–136.09] | 38.80 ms [38.78–38.93] | 43.75 ms [43.66–43.88] | unsupported | 39.86 ms [39.82–39.90] |
| 300 | 1 | 1.0 | 8.25 ms [8.19–8.27] | 8.16 ms [8.08–8.24] | unsupported | unsupported | unsupported | 355.32 ms [355.05–355.85] | 7.79 ms [7.70–8.06] | 8.25 ms [8.16–8.38] | unsupported | 1.89 ms [1.87–1.91] |
| 300 | 1 | 3.0 | 5.84 ms [5.78–5.86] | 5.71 ms [5.69–5.73] | unsupported | unsupported | unsupported | 159.72 ms [159.68–159.96] | 6.08 ms [5.96–6.15] | 6.22 ms [6.18–6.24] | unsupported | 2.62 ms [2.60–2.69] |
| 300 | 1 | 10.0 | 14.26 ms [14.24–14.33] | 14.19 ms [14.16–14.23] | unsupported | unsupported | unsupported | 333.68 ms [333.32–333.91] | 14.29 ms [14.15–14.38] | 15.19 ms [15.15–15.33] | unsupported | 7.68 ms [7.62–7.83] |
| 300 | 1 | 30.0 | 26.96 ms [26.89–27.06] | 26.90 ms [26.83–26.92] | unsupported | unsupported | unsupported | 247.22 ms [246.94–247.71] | 27.54 ms [27.44–27.72] | 29.75 ms [29.68–29.87] | unsupported | 20.53 ms [20.40–20.62] |
| 300 | 2 | 1.0 | 5.49 ms [5.49–5.57] | 5.36 ms [5.35–5.41] | unsupported | unsupported | unsupported | 260.13 ms [259.74–260.25] | 5.47 ms [5.41–5.55] | 5.55 ms [5.48–5.57] | unsupported | 2.10 ms [2.08–2.16] |
| 300 | 2 | 3.0 | 9.06 ms [9.04–9.14] | 8.93 ms [8.89–9.02] | unsupported | unsupported | unsupported | 338.93 ms [338.39–339.41] | 8.80 ms [8.60–8.91] | 9.54 ms [9.51–9.61] | unsupported | 4.82 ms [4.75–4.88] |
| 300 | 2 | 10.0 | 17.65 ms [17.59–17.73] | 17.47 ms [17.40–17.53] | unsupported | unsupported | unsupported | 307.12 ms [306.71–307.68] | 16.84 ms [16.79–16.98] | 18.78 ms [18.67–18.81] | unsupported | 14.08 ms [14.07–14.17] |
| 300 | 2 | 30.0 | 46.90 ms [46.81–47.43] | 46.60 ms [46.37–46.73] | unsupported | unsupported | unsupported | 563.50 ms [563.32–563.96] | 43.99 ms [43.78–44.28] | 49.72 ms [49.57–49.90] | unsupported | 41.06 ms [40.97–41.32] |

## mp3 / bytes

| Duration (s) | Channels | soundfile | librosa | scipy | scipy_mmap | audioread | pedalboard | torchcodec | audiolab | audiosample | sphn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.10 ms [1.09–1.11] | 1.10 ms [1.09–1.11] | unsupported | unsupported | unsupported | 1.95 ms [1.92–1.96] | 1.76 ms [1.72–1.83] | 1.31 ms [1.29–1.35] | unsupported | unsupported |
| 1 | 2 | 1.68 ms [1.66–1.69] | 1.69 ms [1.67–1.69] | unsupported | unsupported | unsupported | 2.78 ms [2.75–2.78] | 2.23 ms [2.20–2.24] | 1.88 ms [1.87–1.92] | unsupported | unsupported |
| 10 | 1 | 8.04 ms [8.03–8.05] | 8.05 ms [8.03–8.19] | unsupported | unsupported | unsupported | 17.42 ms [17.32–17.48] | 9.04 ms [9.01–9.14] | 9.17 ms [9.15–9.22] | unsupported | unsupported |
| 10 | 2 | 13.66 ms [13.64–13.78] | 13.67 ms [13.58–13.74] | unsupported | unsupported | unsupported | 24.94 ms [24.84–25.02] | 13.33 ms [13.30–13.39] | 14.95 ms [14.93–14.98] | unsupported | unsupported |
| 60 | 1 | 46.73 ms [46.65–46.89] | 46.75 ms [46.70–46.89] | unsupported | unsupported | unsupported | 103.49 ms [103.35–103.67] | 49.05 ms [48.99–49.38] | 52.39 ms [52.18–52.52] | unsupported | unsupported |
| 60 | 2 | 80.77 ms [80.66–81.01] | 80.74 ms [80.61–80.92] | unsupported | unsupported | unsupported | 148.66 ms [148.23–149.23] | 74.61 ms [74.53–74.96] | 86.18 ms [86.10–86.60] | unsupported | unsupported |
| 300 | 1 | 238.62 ms [238.47–238.79] | 238.47 ms [238.33–238.87] | unsupported | unsupported | unsupported | 522.56 ms [522.02–523.13] | 254.19 ms [253.58–254.84] | 269.02 ms [268.53–270.17] | unsupported | unsupported |
| 300 | 2 | 431.38 ms [431.10–431.90] | 429.24 ms [428.86–429.54] | unsupported | unsupported | unsupported | 753.22 ms [752.66–753.47] | 402.32 ms [401.28–403.31] | 444.74 ms [444.40–445.19] | unsupported | unsupported |

## Unavailable and incorrect libraries

Every probed library was available and passed the correctness gate.

## Measurement noise

Across 808 `ok` measurements, the observed dispersion `IQR / median` ranges from 0.0% to 14.5%, with a median of 0.7%. That is the measurement-noise floor of this run: differences between libraries, formats, or durations smaller than this floor are noise, not rankings. The interquartile range is reported rather than max-min because the full range grows with the trial count, which would make runs using different `--repeat` values incomparable.
