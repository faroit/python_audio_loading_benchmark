"""Tests for `pabench.report`: markdown rendering and plot generation.

The fixture results dict below is hand-built (not produced by `pabench.run.run`)
so these tests exercise `render_markdown`/`write_plots` in isolation, with a
deliberate mix of `ok`, `unavailable`, `incorrect`, and `unsupported` records
across three benches and three formats -- exactly the mix a real run produces.

`soundfile`, `flaky`, `wavonly`, and `ghost` are fixture library names, not the
the real adapters; nothing here imports `pabench.loaders`.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from pabench.report import render_markdown, write_plots

WAV = "wav_pcm16"
FLAC = "flac_pcm16"
MP3 = "mp3"


def _ok(library, fmt, bench, duration_s, channels, median_ms, min_ms, max_ms, gate="exact"):
    return {
        "library": library,
        "file": f"{duration_s}s_{channels}ch_{fmt}.{'mp3' if fmt == MP3 else fmt.split('_')[0]}",
        "duration_s": duration_s,
        "channels": channels,
        "format": fmt,
        "bench": bench,
        "status": "ok",
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "realtime_factor": (duration_s if bench in ("full", "bytes") else 1.0)
        / (median_ms / 1000.0),
        "gate": gate,
        "reason": None,
    }


def _withheld(library, fmt, bench, duration_s, channels, status, reason, gate=None):
    return {
        "library": library,
        "file": f"{duration_s}s_{channels}ch_{fmt}.{'mp3' if fmt == MP3 else fmt.split('_')[0]}",
        "duration_s": duration_s,
        "channels": channels,
        "format": fmt,
        "bench": bench,
        "status": status,
        "median_ms": None,
        "min_ms": None,
        "max_ms": None,
        "realtime_factor": None,
        "gate": gate,
        "reason": reason,
    }


PLATFORM = {
    "os": "macOS-15.0-arm64",
    "machine": "arm64",
    "python_version": "3.12.5",
    "torch_version": "2.9.0",
    "ffmpeg_version": "ffmpeg version 8.1.1",
    "dyld_fallback_library_path": "/opt/homebrew/lib",
    "libraries": {
        "soundfile": {
            "version": "0.12.1",
            "available": True,
            "notes": "the correctness reference every other loader is checked against",
            "error": None,
        },
        "flaky": {"version": "1.2.3", "available": True, "notes": None, "error": None},
        "wavonly": {"version": "0.1", "available": True, "notes": "WAV only", "error": None},
        "ghost": {
            "version": None,
            "available": False,
            "notes": None,
            "error": "ImportError: no module named ghost",
        },
    },
}

RECORDS = [
    # soundfile: correct everywhere, full + seek
    _ok("soundfile", WAV, "full", 1, 1, median_ms=2.10, min_ms=2.00, max_ms=2.30),
    _ok("soundfile", WAV, "full", 10, 2, median_ms=12.34, min_ms=11.90, max_ms=12.80),
    _ok("soundfile", FLAC, "full", 1, 1, median_ms=3.50, min_ms=3.30, max_ms=3.80),
    _ok("soundfile", MP3, "full", 1, 1, median_ms=4.00, min_ms=3.90, max_ms=4.20, gate="relaxed"),
    _ok("soundfile", WAV, "seek", 10, 2, median_ms=1.20, min_ms=1.10, max_ms=1.30),
    # flaky: ok on some cells, incorrect on one, no seek support at all
    _ok("flaky", WAV, "full", 1, 1, median_ms=2.50, min_ms=2.40, max_ms=2.60),
    _withheld(
        "flaky",
        WAV,
        "full",
        10,
        2,
        status="incorrect",
        reason="shape mismatch: reference (2, 441000) vs candidate (1, 441000)",
        gate="exact",
    ),
    _ok("flaky", FLAC, "full", 1, 1, median_ms=3.60, min_ms=3.45, max_ms=3.75),
    _ok("flaky", MP3, "full", 1, 1, median_ms=4.10, min_ms=3.95, max_ms=4.25, gate="relaxed"),
    _withheld(
        "flaky",
        WAV,
        "seek",
        10,
        2,
        status="unsupported",
        reason="flaky has no seek implementation",
    ),
    # wavonly: WAV-only, but does support seek
    _ok("wavonly", WAV, "full", 1, 1, median_ms=2.05, min_ms=1.95, max_ms=2.15),
    _ok("wavonly", WAV, "full", 10, 2, median_ms=15.20, min_ms=14.80, max_ms=15.90),
    _withheld(
        "wavonly",
        FLAC,
        "full",
        1,
        1,
        status="unsupported",
        reason="wavonly does not decode 'flac' files",
    ),
    _withheld(
        "wavonly",
        MP3,
        "full",
        1,
        1,
        status="unsupported",
        reason="wavonly does not decode 'mp3' files",
    ),
    _ok("wavonly", WAV, "seek", 10, 2, median_ms=1.10, min_ms=1.05, max_ms=1.18),
    # ghost: never available, recorded (not dropped) for every cell
    _withheld(
        "ghost",
        WAV,
        "full",
        1,
        1,
        status="unavailable",
        reason="ImportError: no module named ghost",
    ),
    _withheld(
        "ghost",
        WAV,
        "full",
        10,
        2,
        status="unavailable",
        reason="ImportError: no module named ghost",
    ),
    _withheld(
        "ghost",
        FLAC,
        "full",
        1,
        1,
        status="unavailable",
        reason="ImportError: no module named ghost",
    ),
    _withheld(
        "ghost",
        MP3,
        "full",
        1,
        1,
        status="unavailable",
        reason="ImportError: no module named ghost",
    ),
    _withheld(
        "ghost",
        WAV,
        "seek",
        10,
        2,
        status="unavailable",
        reason="ImportError: no module named ghost",
    ),
    # bytes bench, appended (not interleaved above) so existing positional indices
    # into RECORDS elsewhere in this file are unaffected: soundfile supports it,
    # flaky and wavonly don't, ghost is unavailable.
    _ok("soundfile", WAV, "bytes", 1, 1, median_ms=1.80, min_ms=1.70, max_ms=1.95),
    _withheld(
        "flaky",
        WAV,
        "bytes",
        1,
        1,
        status="unsupported",
        reason="flaky has no from_bytes implementation",
    ),
    _withheld(
        "wavonly",
        WAV,
        "bytes",
        1,
        1,
        status="unsupported",
        reason="wavonly has no from_bytes implementation",
    ),
    _withheld(
        "ghost",
        WAV,
        "bytes",
        1,
        1,
        status="unavailable",
        reason="ImportError: no module named ghost",
    ),
]

RESULTS = {"platform": PLATFORM, "records": RECORDS}


def _table_rows(markdown: str, heading: str) -> list[str]:
    lines = markdown.splitlines()
    start = lines.index(heading) + 1
    rows = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            rows.append(line)
    return rows


def _row_as_dict(header_row: str, data_row: str) -> dict[str, str]:
    headers = [c.strip() for c in header_row.strip("|").split("|")]
    cells = [c.strip() for c in data_row.strip("|").split("|")]
    return dict(zip(headers, cells))


# ---- render_markdown: structure -----------------------------------------------


def test_render_markdown_includes_platform_fields():
    markdown = render_markdown(RESULTS)
    assert "macOS-15.0-arm64" in markdown
    assert "3.12.5" in markdown
    assert "2.9.0" in markdown
    assert "/opt/homebrew/lib" in markdown


def test_render_markdown_library_table_lists_every_library_with_seek_support():
    markdown = render_markdown(RESULTS)
    rows = _table_rows(markdown, "## Libraries")
    header, data_rows = rows[0], rows[2:]
    by_library = {}
    for row in data_rows:
        as_dict = _row_as_dict(header, row)
        by_library[as_dict["Library"]] = as_dict

    assert by_library["soundfile"]["Seek"] == "yes"
    assert by_library["soundfile"]["Available"] == "yes"
    assert by_library["flaky"]["Seek"] == "no"
    assert by_library["wavonly"]["Seek"] == "yes"
    assert by_library["ghost"]["Seek"] == "n/a"
    assert by_library["ghost"]["Available"] == "no"
    assert by_library["ghost"]["Error"] == "ImportError: no module named ghost"


def test_render_markdown_library_table_lists_every_library_with_bytes_support():
    markdown = render_markdown(RESULTS)
    rows = _table_rows(markdown, "## Libraries")
    header, data_rows = rows[0], rows[2:]
    by_library = {}
    for row in data_rows:
        as_dict = _row_as_dict(header, row)
        by_library[as_dict["Library"]] = as_dict

    assert by_library["soundfile"]["Bytes"] == "yes"
    assert by_library["flaky"]["Bytes"] == "no"
    assert by_library["wavonly"]["Bytes"] == "no"
    assert by_library["ghost"]["Bytes"] == "n/a"


def test_cross_table_cell_is_median_with_spread_not_a_bare_median():
    markdown = render_markdown(RESULTS)
    rows = _table_rows(markdown, f"## {WAV} / full")
    header = rows[0]
    row_10_2 = next(r for r in rows[2:] if r.strip("|").split("|")[0].strip() == "10")
    as_dict = _row_as_dict(header, row_10_2)

    # The exact format matters: a bare "12.34" would satisfy a looser check, but
    # the spec requires the spread to be visible in the same cell.
    assert as_dict["soundfile"] == "12.34 ms [11.90–12.80]"
    # A transposed table (rows/columns swapped, or columns in the wrong order)
    # would put wavonly's number under soundfile or vice versa.
    assert as_dict["wavonly"] == "15.20 ms [14.80–15.90]"
    assert as_dict["flaky"] == "incorrect"
    assert as_dict["ghost"] == "unavailable"


def test_cross_table_row_for_a_different_duration_is_not_confused_with_the_other():
    markdown = render_markdown(RESULTS)
    rows = _table_rows(markdown, f"## {WAV} / full")
    header = rows[0]
    row_1_1 = next(r for r in rows[2:] if r.strip("|").split("|")[0].strip() == "1")
    as_dict = _row_as_dict(header, row_1_1)
    assert as_dict["soundfile"] == "2.10 ms [2.00–2.30]"
    assert as_dict["flaky"] == "2.50 ms [2.40–2.60]"
    assert as_dict["wavonly"] == "2.05 ms [1.95–2.15]"
    assert as_dict["ghost"] == "unavailable"


def test_unsupported_cells_show_the_status_word():
    markdown = render_markdown(RESULTS)
    rows = _table_rows(markdown, f"## {FLAC} / full")
    header = rows[0]
    (row,) = [r for r in rows[2:]]
    as_dict = _row_as_dict(header, row)
    assert as_dict["wavonly"] == "unsupported"


def test_seek_table_only_has_the_one_measured_duration_channels_combo():
    markdown = render_markdown(RESULTS)
    rows = _table_rows(markdown, f"## {WAV} / seek")
    data_rows = rows[2:]
    assert len(data_rows) == 1
    as_dict = _row_as_dict(rows[0], data_rows[0])
    assert as_dict["soundfile"] == "1.20 ms [1.10–1.30]"
    assert as_dict["flaky"] == "unsupported"
    assert as_dict["wavonly"] == "1.10 ms [1.05–1.18]"
    assert as_dict["ghost"] == "unavailable"


def test_bytes_cross_table_renders_without_special_casing():
    markdown = render_markdown(RESULTS)
    rows = _table_rows(markdown, f"## {WAV} / bytes")
    header, data_rows = rows[0], rows[2:]
    (row,) = data_rows
    as_dict = _row_as_dict(header, row)
    assert as_dict["soundfile"] == "1.80 ms [1.70–1.95]"
    assert as_dict["flaky"] == "unsupported"
    assert as_dict["wavonly"] == "unsupported"
    assert as_dict["ghost"] == "unavailable"


def test_unavailable_and_incorrect_section_lists_reasons():
    markdown = render_markdown(RESULTS)
    section = markdown[markdown.index("## Unavailable and incorrect libraries") :]
    assert "ghost" in section
    assert "ImportError: no module named ghost" in section
    assert "flaky" in section
    assert "shape mismatch" in section


def test_mp3_relaxed_gate_is_called_out():
    markdown = render_markdown(RESULTS)
    assert "relaxed gate" in markdown.lower()
    assert "50 ms" in markdown
    assert "0.5 dB" in markdown


def test_warm_cache_and_synthetic_corpus_are_stated():
    markdown = render_markdown(RESULTS)
    lowered = markdown.lower()
    assert "warm" in lowered and "cache" in lowered
    assert "synthetic" in lowered


def test_measurement_noise_section_matches_independent_computation():
    ratios = [(r["max_ms"] - r["min_ms"]) / r["median_ms"] for r in RECORDS if r["status"] == "ok"]
    expected_median_pct = statistics.median(ratios) * 100
    expected_min_pct = min(ratios) * 100
    expected_max_pct = max(ratios) * 100

    markdown = render_markdown(RESULTS)
    section = markdown[markdown.index("## Measurement noise") :]

    assert f"{expected_min_pct:.1f}%" in section
    assert f"{expected_max_pct:.1f}%" in section
    assert f"{expected_median_pct:.1f}%" in section
    assert "not a" in section.lower() or "not rankings" in section.lower()


def test_measurement_noise_section_handles_no_ok_records():
    results = {"platform": PLATFORM, "records": [RECORDS[14]]}  # a single unavailable record
    markdown = render_markdown(results)
    section = markdown[markdown.index("## Measurement noise") :]
    assert "no" in section.lower()


# ---- write_plots ----------------------------------------------------------------


def test_write_plots_creates_non_empty_files_for_every_bench(tmp_path: Path):
    paths = write_plots(RESULTS, tmp_path)
    assert len(paths) == 3  # "full", "seek" and "bytes" all appear in RECORDS
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_plots_creates_a_bytes_png(tmp_path: Path):
    write_plots(RESULTS, tmp_path)
    bytes_png = tmp_path / "bytes.png"
    assert bytes_png.exists()
    assert bytes_png.stat().st_size > 0


def test_write_plots_does_not_raise_on_none_timings(tmp_path: Path):
    # RECORDS already includes unavailable/incorrect/unsupported rows with
    # median_ms=None; this just asserts the whole fixture (not a filtered-down
    # ok-only subset) survives write_plots without raising.
    paths = write_plots(RESULTS, tmp_path / "plots")
    assert paths
    for path in paths:
        assert path.stat().st_size > 0


def test_write_plots_on_all_withheld_records_still_produces_a_file(tmp_path: Path):
    only_withheld = [r for r in RECORDS if r["status"] != "ok"]
    results = {"platform": PLATFORM, "records": only_withheld}
    paths = write_plots(results, tmp_path)
    assert paths
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
