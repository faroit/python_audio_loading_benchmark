"""Markdown report and plots rendered from a `pabench.run.run` results dict.

Both `render_markdown` and `write_plots` consume exactly the dict `pabench.run.run`
returns -- `{"platform": ..., "records": [...]}`, where each record has the same
keys as `pabench.run.Record` -- whether that dict comes straight out of `run()` or
was round-tripped through `pabench.run.write_results`/`json.load`. Neither function
re-runs a benchmark or reads a corpus file; every number they print already lives
in `results`. See `docs/refactor-design.md`, "Measurement" and the correctness gate
section, for why a status word stands in for a number, and why MP3 carries a
weaker correctness guarantee than WAV/FLAC.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must precede importing pyplot: no display is ever available here

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pabench.corpus import FORMATS

Records = list[dict[str, Any]]

_FORMAT_ORDER = {fmt.key: index for index, fmt in enumerate(FORMATS)}
_BENCH_ORDER = {"full": 0, "seek": 1, "bytes": 2}

_EN_DASH = "–"


def _format_sort_key(format_key: str) -> tuple[int, str]:
    return (_FORMAT_ORDER.get(format_key, len(_FORMAT_ORDER)), format_key)


def _bench_sort_key(bench: str) -> tuple[int, str]:
    return (_BENCH_ORDER.get(bench, len(_BENCH_ORDER)), bench)


def _library_order(results: dict) -> list[str]:
    """Libraries in `platform.libraries` order, then any seen only in `records`."""
    platform_libraries = results.get("platform", {}).get("libraries") or {}
    seen_in_records = {r["library"] for r in results.get("records", [])}
    ordered = [name for name in platform_libraries if name in seen_in_records]
    ordered += sorted(seen_in_records - set(ordered))
    return ordered


def _format_cell(record: dict) -> str:
    """A single results-table cell: median+spread if measured, else the status word."""
    if record["status"] == "ok":
        return (
            f"{record['median_ms']:.2f} ms [{record['min_ms']:.2f}{_EN_DASH}{record['max_ms']:.2f}]"
        )
    return record["status"]


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _seektable_label(value: bool | None) -> str | None:
    """Describe the corpus's FLAC seektable state in words rather than a bare bool.

    Which case was measured changes what a FLAC seek number means: with a seektable a
    library jumps, without one it binary-searches the frames.
    """
    if value is None:
        return None
    return "present (seeking can jump)" if value else "absent (seeking must scan frames)"


def _platform_table(platform: dict) -> str:
    fields = [
        ("OS", platform.get("os")),
        ("Machine", platform.get("machine")),
        ("Python", platform.get("python_version")),
        ("PyTorch", platform.get("torch_version")),
        ("FFmpeg", platform.get("ffmpeg_version")),
        ("DYLD_FALLBACK_LIBRARY_PATH", platform.get("dyld_fallback_library_path")),
        ("FLAC seektables", _seektable_label(platform.get("flac_seektables"))),
    ]
    rows = [[label, str(value) if value is not None else "-"] for label, value in fields]
    return _markdown_table(["Field", "Value"], rows)


def _capability_support(library: str, bench: str, records: Records, no_support_reason: str) -> str:
    """Whether `library` has a working `bench` implementation, inferred from its records.

    "no" if any `bench` record for this library carries the fixed reason
    `pabench.run` gives a loader lacking that capability; "n/a" if the library is
    unavailable, or was never actually attempted for `bench` (no supported container
    in this corpus); "yes" otherwise (at least one attempt was made and didn't hit
    that reason).
    """
    bench_records = [r for r in records if r["library"] == library and r["bench"] == bench]
    if not bench_records:
        return "n/a"
    if any(r["reason"] == no_support_reason for r in bench_records):
        return "no"
    if all(r["status"] in ("unavailable", "unsupported") for r in bench_records):
        return "n/a"
    return "yes"


def _seek_support(library: str, records: Records) -> str:
    """Whether `library` has a working seek implementation, inferred from its records."""
    return _capability_support(library, "seek", records, f"{library} has no seek implementation")


def _bytes_support(library: str, records: Records) -> str:
    """Whether `library` has a working `from_bytes` implementation, inferred from its records."""
    return _capability_support(
        library, "bytes", records, f"{library} has no from_bytes implementation"
    )


def _library_table(results: dict) -> str:
    platform_libraries = results.get("platform", {}).get("libraries") or {}
    records = results.get("records", [])
    rows = []
    for name in _library_order(results):
        info = platform_libraries.get(name, {})
        rows.append(
            [
                name,
                str(info.get("version") or "-"),
                "yes" if info.get("available") else "no",
                _seek_support(name, records),
                _bytes_support(name, records),
                str(info.get("notes") or "-"),
                str(info.get("error") or "-"),
            ]
        )
    return _markdown_table(
        ["Library", "Version", "Available", "Seek", "Bytes", "Notes", "Error"], rows
    )


def _cross_table(format_key: str, bench: str, records: Records, libraries: list[str]) -> str:
    """One table for a (format, bench) pair: libraries as columns.

    Rows are (duration, channels) for `full`/`bytes`, and (duration, channels, chunk
    length) for `seek` -- the chunk length is a distinct axis with real consequences
    for what is being measured (a seek's fixed cost dominates at a short chunk, its
    per-second decode cost at a long one), so two chunk lengths for the same file are
    two separate rows here, never averaged into one.
    """
    subset = [r for r in records if r["format"] == format_key and r["bench"] == bench]

    if bench == "seek":
        cell_by_key = {
            (r["duration_s"], r["channels"], r.get("seek_seconds"), r["library"]): r for r in subset
        }
        axes = sorted(
            {(r["duration_s"], r["channels"], r.get("seek_seconds")) for r in subset},
            key=lambda axis: (axis[0], axis[1], axis[2] if axis[2] is not None else -1.0),
        )
        header = ["Duration (s)", "Channels", "Chunk (s)", *libraries]
        rows = []
        for duration_s, channels, chunk_seconds in axes:
            row = [str(duration_s), str(channels), str(chunk_seconds)]
            for library in libraries:
                record = cell_by_key.get((duration_s, channels, chunk_seconds, library))
                row.append(_format_cell(record) if record is not None else "-")
            rows.append(row)
        return _markdown_table(header, rows)

    cell_by_key = {(r["duration_s"], r["channels"], r["library"]): r for r in subset}
    axes = sorted({(r["duration_s"], r["channels"]) for r in subset})

    rows = []
    for duration_s, channels in axes:
        row = [str(duration_s), str(channels)]
        for library in libraries:
            record = cell_by_key.get((duration_s, channels, library))
            row.append(_format_cell(record) if record is not None else "-")
        rows.append(row)

    return _markdown_table(["Duration (s)", "Channels", *libraries], rows)


def _unavailable_and_incorrect(records: Records) -> str:
    unavailable: dict[str, str] = {}
    incorrect: list[tuple[str, str, str, str]] = []
    seen_incorrect: set[tuple[str, str, str, str]] = set()

    for r in records:
        if r["status"] == "unavailable" and r["library"] not in unavailable:
            unavailable[r["library"]] = r["reason"] or "no reason recorded"
        if r["status"] == "incorrect":
            key = (r["library"], r["format"], r["bench"], r["reason"] or "no reason recorded")
            if key not in seen_incorrect:
                seen_incorrect.add(key)
                incorrect.append(key)

    lines: list[str] = []
    if unavailable:
        lines.append("Unavailable libraries:")
        lines.append("")
        lines.extend(f"- **{name}**: {unavailable[name]}" for name in sorted(unavailable))
        lines.append("")
    if incorrect:
        lines.append("Libraries that failed the correctness gate for at least one cell:")
        lines.append("")
        for library, format_key, bench, reason in sorted(incorrect):
            lines.append(f"- **{library}** ({format_key}, {bench}): {reason}")
        lines.append("")
    if not unavailable and not incorrect:
        lines.append("Every probed library was available and passed the correctness gate.")
    return "\n".join(lines).rstrip()


def _noise_ratios(records: Records) -> list[float]:
    """Per-measurement dispersion as IQR divided by median.

    The interquartile range is used rather than the full range because the range
    grows with the trial count by construction -- more trials mean more chances to
    sample an outlier -- so ranges from runs with different `--repeat` values cannot
    be compared. The IQR is stable across trial counts.
    """
    ratios = []
    for r in records:
        if r["status"] != "ok":
            continue
        median = r.get("median_ms")
        if not median:  # None or 0: nothing to divide by
            continue
        spread = r.get("iqr_ms")
        if spread is None:  # results produced before iqr_ms existed
            spread = r["max_ms"] - r["min_ms"]
        ratios.append(spread / median)
    return ratios


def _noise_section(records: Records) -> str:
    ratios = _noise_ratios(records)
    if not ratios:
        return (
            "No `ok` measurements are present in this report, so no measurement-noise "
            "floor could be computed."
        )
    median_pct = statistics.median(ratios) * 100
    min_pct = min(ratios) * 100
    max_pct = max(ratios) * 100
    return (
        f"Across {len(ratios)} `ok` measurements, the observed dispersion "
        "`IQR / median` ranges from "
        f"{min_pct:.1f}% to {max_pct:.1f}%, with a median of {median_pct:.1f}%. "
        "That is the measurement-noise floor of this run: differences between "
        "libraries, formats, or durations smaller than this floor are noise, not "
        "rankings. The interquartile range is reported rather than max-min because "
        "the full range grows with the trial count, which would make runs using "
        "different `--repeat` values incomparable."
    )


def render_markdown(results: dict) -> str:
    """Render the full markdown report for `results`.

    `results` is the dict `pabench.run.run` returns (or the same dict round-tripped
    through JSON). This never re-runs a benchmark and never reads a corpus file --
    every number printed already lives in `results`.
    """
    platform = results.get("platform", {})
    records = results.get("records", [])
    libraries = _library_order(results)

    format_bench_pairs = sorted(
        {(r["format"], r["bench"]) for r in records},
        key=lambda pair: (_format_sort_key(pair[0]), _bench_sort_key(pair[1])),
    )

    sections = [
        "# pabench results",
        "",
        (
            "Timings below are wall-clock, warm-page-cache decodes of a locally "
            "generated **synthetic white-noise corpus** (see `docs/refactor-design.md`): "
            "one untimed warmup per (library, file, benchmark) that also warms the OS "
            "page cache, followed by the repeated, timed trials each cell's spread is "
            "drawn from. They describe decode speed against this machine's page cache "
            "and this synthetic corpus, not cold-disk I/O and not real program material."
        ),
        "",
        "## Platform",
        "",
        _platform_table(platform),
        "",
        "## Libraries",
        "",
        _library_table(results),
        "",
        (
            "**MP3 is graded by a relaxed gate** (decoded duration within 50 ms of the "
            "reference, RMS level within 0.5 dB on the common length) rather than the "
            "sample-exact gate WAV and FLAC are held to (1.5 LSB for integer PCM, "
            "`atol=1e-7` for float32), because MP3 decoders disagree on encoder delay "
            "and never match sample-for-sample. MP3 results therefore carry a weaker "
            "correctness guarantee than the WAV/FLAC results in this report."
        ),
        "",
    ]

    for format_key, bench in format_bench_pairs:
        sections.append(f"## {format_key} / {bench}")
        sections.append("")
        sections.append(_cross_table(format_key, bench, records, libraries))
        sections.append("")

    sections.extend(
        [
            "## Unavailable and incorrect libraries",
            "",
            _unavailable_and_incorrect(records),
            "",
            "## Measurement noise",
            "",
            _noise_section(records),
            "",
        ]
    )

    return "\n".join(sections)


# ---- plots ------------------------------------------------------------------
#
# Loaded per docs/refactor-design.md and docs/refactor-plan.md: the `dataviz`
# skill was read before any of this was written. Its categorical palette (fixed
# hue order, never cycled) is reused here; a 9th-or-later series folds into a
# repeated hue with a dashed line rather than generating a new, unvalidated hue
# (see the skill's palette.md, "A 9th series is never a generated hue").

_PALETTE: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
_MUTED_TEXT = "#52514e"
_GRIDLINE = "#e1e0d9"


def _library_styles(style_order: list[str]) -> tuple[dict, dict]:
    """Colour and dash maps keyed by library name, stable across every figure.

    Keying by name rather than by position matters: a figure that draws fewer
    libraries (the seek grid omits those that cannot seek) must not repaint the
    ones that remain. The palette has eight validated hues; past eight, a dash
    pattern carries the difference, so identity never rests on a ninth colour
    that would sit too close to an existing one.
    """
    # seaborn's "colorblind" palette, sized to the library count so every library gets
    # its own hue and no line needs a dash to disambiguate a reused colour. Lines stay
    # solid; marker shape varies so identity survives a greyscale print.
    hues = sns.color_palette("colorblind", max(len(style_order), 1)).as_hex()
    markers_cycle = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
    colors, markers = {}, {}
    for index, name in enumerate(style_order):
        colors[name] = hues[index % len(hues)]
        markers[name] = markers_cycle[index % len(markers_cycle)]
    return colors, markers


def _plot_bench(records: Records, bench: str, out_dir: Path, style_order: list[str]) -> Path:
    """Render one bench as a seaborn FacetGrid: formats across, channels down."""
    plotted = [
        r
        for r in records
        if r["bench"] == bench and r["status"] == "ok" and r["median_ms"] is not None
    ]
    out_path = Path(out_dir) / f"{bench}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not plotted:
        fig = plt.figure(figsize=(6, 2))
        fig.text(0.5, 0.5, f"no {bench} data", ha="center", va="center", color=_MUTED_TEXT)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    frame = pd.DataFrame(
        [
            {
                "duration": r["duration_s"],
                "median_ms": r["median_ms"],
                "library": r["library"],
                "format": r["format"],
                "channels": f"{r['channels']}ch",
            }
            for r in plotted
        ]
    )
    # Only libraries actually drawn reach the legend; a library that cannot seek is
    # an unsupported operation, not a line someone should hunt for.
    present = [name for name in style_order if name in set(frame["library"])]
    colors, markers = _library_styles(style_order)
    formats = [f for f in sorted({*frame["format"]}, key=_format_sort_key)]

    sns.set_theme(
        style="whitegrid",
        rc={"grid.color": _GRIDLINE, "grid.linewidth": 0.6},
    )
    grid = sns.relplot(
        data=frame,
        x="duration",
        y="median_ms",
        hue="library",
        style="library",
        hue_order=present,
        style_order=present,
        palette={name: colors[name] for name in present},
        markers={name: markers[name] for name in present},
        dashes=False,
        col="format",
        col_order=formats,
        row="channels",
        row_order=sorted({*frame["channels"]}),
        kind="line",
        markersize=5,
        linewidth=1.8,
        height=3.1,
        aspect=1.3,
        facet_kws={"sharey": "row", "legend_out": True},
    )
    grid.set(xscale="log", yscale="log")
    # Minor gridlines on both log axes: on a log-log plot the decade lines alone leave
    # most of the plane unreferenced, and reading a value between them is guesswork.
    for ax in grid.axes.flat:
        ax.grid(True, which="major", color=_GRIDLINE, linewidth=0.7)
        ax.grid(True, which="minor", color=_GRIDLINE, linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)
    grid.figure.subplots_adjust(wspace=0.22, hspace=0.28)
    grid.set_axis_labels("duration (s)", "median (ms)")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    grid.figure.suptitle(
        f"{bench} decode: duration vs. median time (log-log, lower is better)",
        y=1.02,
        color="#0b0b0b",
    )
    if grid.legend is not None:
        grid.legend.set_title("library")
    for ax in grid.axes.flat:
        ax.tick_params(colors=_MUTED_TEXT, labelsize=8)
        ax.xaxis.label.set_color(_MUTED_TEXT)
        ax.yaxis.label.set_color(_MUTED_TEXT)

    grid.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(grid.figure)
    return out_path


def _plot_seek_scaling(records: Records, out_dir: Path, style_order: list[str]) -> Path:
    """`seek_scaling.png`: chunk duration vs. median time, one line per library.

    This is the figure the chunk-duration axis exists for: a seek's cost is
    `fixed_locate_cost + per_second_decode_cost * chunk_length`, and plotting median
    time against chunk length on log-log axes shows the fixed-cost intercept and the
    per-second slope directly, which a single fixed chunk length cannot (see
    `docs/refactor-design.md`).

    Restricted to the longest file duration present and stereo files -- the cleanest
    signal, and the one comparison this figure is meant to make -- faceted by format,
    with one line per library. Only `bench == "seek"`, `status == "ok"` records with a
    timing are plotted; a run with no such data (or none at the longest duration in
    stereo) yields a placeholder figure rather than no file, matching `_plot_bench`.
    """
    out_path = Path(out_dir) / "seek_scaling.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        r
        for r in records
        if r["bench"] == "seek"
        and r["status"] == "ok"
        and r["median_ms"] is not None
        and r.get("seek_seconds") is not None
        and r["channels"] == 2
    ]

    if not candidates:
        fig = plt.figure(figsize=(6, 2))
        fig.text(0.5, 0.5, "no seek data", ha="center", va="center", color=_MUTED_TEXT)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    longest_duration = max(r["duration_s"] for r in candidates)
    plotted = [r for r in candidates if r["duration_s"] == longest_duration]

    frame = pd.DataFrame(
        [
            {
                "chunk_seconds": r["seek_seconds"],
                "median_ms": r["median_ms"],
                "library": r["library"],
                "format": r["format"],
            }
            for r in plotted
        ]
    )
    present = [name for name in style_order if name in set(frame["library"])]
    colors, markers = _library_styles(style_order)
    formats = [f for f in sorted({*frame["format"]}, key=_format_sort_key)]

    sns.set_theme(
        style="whitegrid",
        rc={"grid.color": _GRIDLINE, "grid.linewidth": 0.6},
    )
    grid = sns.relplot(
        data=frame,
        x="chunk_seconds",
        y="median_ms",
        hue="library",
        style="library",
        hue_order=present,
        style_order=present,
        palette={name: colors[name] for name in present},
        markers={name: markers[name] for name in present},
        dashes=False,
        col="format",
        col_order=formats,
        kind="line",
        markersize=5,
        linewidth=1.8,
        height=3.1,
        aspect=1.3,
        facet_kws={"sharey": False, "legend_out": True},
    )
    grid.set(xscale="log", yscale="log")
    # Minor gridlines on both log axes: on a log-log plot the decade lines alone leave
    # most of the plane unreferenced, and reading a value between them is guesswork.
    for ax in grid.axes.flat:
        ax.grid(True, which="major", color=_GRIDLINE, linewidth=0.7)
        ax.grid(True, which="minor", color=_GRIDLINE, linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)
    grid.figure.subplots_adjust(wspace=0.22)
    grid.set_axis_labels("chunk duration (s)", "median (ms)")
    grid.set_titles(col_template="{col_name}")
    grid.figure.suptitle(
        f"seek scaling at {longest_duration}s, stereo: chunk duration vs. "
        "median time (log-log, lower is better)",
        y=1.02,
        color="#0b0b0b",
    )
    if grid.legend is not None:
        grid.legend.set_title("library")
    for ax in grid.axes.flat:
        ax.tick_params(colors=_MUTED_TEXT, labelsize=8)
        ax.xaxis.label.set_color(_MUTED_TEXT)
        ax.yaxis.label.set_color(_MUTED_TEXT)

    grid.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(grid.figure)
    return out_path


def write_plots(results: dict, out_dir: Path) -> list[Path]:
    """Write one figure per bench present in `results["records"]`, plus `seek_scaling.png`.

    Each per-bench figure is a seaborn `FacetGrid`: a log-log grid of duration versus
    median decode time (ms), formats across the columns and channel counts down the
    rows, one line per library, with a single legend outside the axes. Only
    `status == "ok"` records with a timing are plotted; withheld (`None`) timings are
    skipped without raising, and a bench with no plottable data yields a placeholder
    figure rather than no file.

    `seek_scaling.png` is always written in addition (see `_plot_seek_scaling`): the
    seek chunk-duration axis's own figure, restricted to the longest file duration and
    stereo channels.
    """
    records = results.get("records", [])
    benches = sorted({r["bench"] for r in records}, key=_bench_sort_key)
    # One stable slot per library across every figure, so a library keeps its colour
    # whether or not a given figure draws it.
    style_order = sorted({r["library"] for r in records})
    out_dir = Path(out_dir)
    paths = [_plot_bench(records, bench, out_dir, style_order) for bench in benches]
    paths.append(_plot_seek_scaling(records, out_dir, style_order))
    return paths
