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

import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must precede importing pyplot: no display is ever available here

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pabench.corpus import FORMATS

Records = list[dict[str, Any]]

_FORMAT_ORDER = {fmt.key: index for index, fmt in enumerate(FORMATS)}
_BENCH_ORDER = {"full": 0, "seek": 1}

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


def _platform_table(platform: dict) -> str:
    fields = [
        ("OS", platform.get("os")),
        ("Machine", platform.get("machine")),
        ("Python", platform.get("python_version")),
        ("PyTorch", platform.get("torch_version")),
        ("FFmpeg", platform.get("ffmpeg_version")),
        ("DYLD_FALLBACK_LIBRARY_PATH", platform.get("dyld_fallback_library_path")),
    ]
    rows = [[label, str(value) if value is not None else "-"] for label, value in fields]
    return _markdown_table(["Field", "Value"], rows)


def _seek_support(library: str, records: Records) -> str:
    """Whether `library` has a working seek implementation, inferred from its records.

    "no" if any seek-bench record for this library carries the fixed reason
    `pabench.run` gives a seek-less loader; "n/a" if the library is unavailable, or
    was never actually attempted for seek (no supported container in this corpus);
    "yes" otherwise (at least one seek attempt was made and didn't hit that reason).
    """
    seek_records = [r for r in records if r["library"] == library and r["bench"] == "seek"]
    if not seek_records:
        return "n/a"
    no_seek_reason = f"{library} has no seek implementation"
    if any(r["reason"] == no_seek_reason for r in seek_records):
        return "no"
    if all(r["status"] in ("unavailable", "unsupported") for r in seek_records):
        return "n/a"
    return "yes"


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
                str(info.get("notes") or "-"),
                str(info.get("error") or "-"),
            ]
        )
    return _markdown_table(["Library", "Version", "Available", "Seek", "Notes", "Error"], rows)


def _cross_table(format_key: str, bench: str, records: Records, libraries: list[str]) -> str:
    """One table for a (format, bench) pair: libraries as columns, (duration, channels) rows."""
    subset = [r for r in records if r["format"] == format_key and r["bench"] == bench]
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


def _library_style(index: int) -> dict:
    """Style for a library's stable slot.

    `index` must come from a library's position in the *whole* results set, never
    from its position in a filtered subset: colour follows the entity, so a figure
    that happens to draw fewer libraries must not repaint the survivors.
    """
    return {
        "color": _PALETTE[index % len(_PALETTE)],
        "linestyle": "-" if index < len(_PALETTE) else "--",
    }


def _style_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.6, color=_GRIDLINE)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRIDLINE)
    ax.tick_params(colors=_MUTED_TEXT, labelsize=7)


def _direct_label(ax, end_points: list[tuple[float, float, str, str]]) -> None:
    """Label each line at the panel's right edge, nudging apart labels that collide.

    `end_points` is `(x, y, library, color)` for each line's last plotted point.
    Labels are sorted by y and pushed apart in log-space by a fixed minimum gap;
    a thin leader line in the series color connects a nudged label back to its
    actual data point, so a moved label doesn't read as belonging to a neighbour.
    """
    if not end_points:
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=_MUTED_TEXT,
            fontsize=8,
        )
        return

    points = sorted(end_points, key=lambda p: p[1])
    log_ys = [math.log10(p[1]) for p in points]
    min_gap = 0.105  # log10 units; wide enough for 7pt text at this figure height
    adjusted = list(log_ys)
    for i in range(1, len(adjusted)):
        if adjusted[i] - adjusted[i - 1] < min_gap:
            adjusted[i] = adjusted[i - 1] + min_gap

    x_max = max(p[0] for p in points)
    ax.set_xlim(ax.get_xlim()[0], x_max * 2.2)

    for (x, y, library, color), log_y in zip(points, adjusted):
        label_y = 10**log_y
        if abs(label_y - y) > 1e-9 * max(abs(y), 1e-9):
            ax.plot([x, x_max], [y, label_y], linewidth=0.6, linestyle=":", color=color, alpha=0.6)
        ax.annotate(
            library,
            xy=(x_max, label_y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=7,
            color=_MUTED_TEXT,
            annotation_clip=False,
        )


def _plot_cell(ax, cell_records: Records, libraries: list[str], style_order: list[str]) -> None:
    _style_axis(ax)
    end_points: list[tuple[float, float, str, str]] = []
    for library in libraries:
        index = style_order.index(library)
        lib_records = sorted(
            (
                r
                for r in cell_records
                if r["library"] == library and r["status"] == "ok" and r["median_ms"]
            ),
            key=lambda r: r["duration_s"],
        )
        if not lib_records:
            continue
        xs = [r["duration_s"] for r in lib_records]
        ys = [r["median_ms"] for r in lib_records]
        style = _library_style(index)
        ax.plot(xs, ys, linewidth=2, marker="o", markersize=4, **style)
        end_points.append((xs[-1], ys[-1], library, style["color"]))

    _direct_label(ax, end_points)


def _plot_bench(records: Records, bench: str, out_dir: Path, style_order: list[str]) -> Path:
    bench_records = [r for r in records if r["bench"] == bench]
    # Only libraries with at least one plottable point belong in this figure: a
    # library that cannot seek has no line in the seek figure, and listing it in
    # the legend would imply a missing line rather than an unsupported operation.
    libraries = sorted(
        {r["library"] for r in bench_records if r["status"] == "ok" and r["median_ms"]}
    )
    formats = sorted({r["format"] for r in bench_records}, key=_format_sort_key)
    channels_list = sorted({r["channels"] for r in bench_records})

    n_rows = max(len(channels_list), 1)
    n_cols = max(len(formats), 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for row, channels in enumerate(channels_list or [None]):
        for col, format_key in enumerate(formats or [None]):
            ax = axes[row][col]
            cell_records = [
                r for r in bench_records if r["channels"] == channels and r["format"] == format_key
            ]
            _plot_cell(ax, cell_records, libraries, style_order)
            if row == 0:
                ax.set_title(format_key or "-", color=_MUTED_TEXT, fontsize=10)
            if col == 0:
                ylabel = f"{channels}ch median (ms)" if channels is not None else "median (ms)"
                ax.set_ylabel(ylabel, color=_MUTED_TEXT, fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("duration (s)", color=_MUTED_TEXT, fontsize=8)

    fig.suptitle(f"{bench} decode: duration vs. median time (log-log)", color="#0b0b0b")

    if libraries:
        handles = [
            Line2D(
                [0], [0], label=library, linewidth=2, **_library_style(style_order.index(library))
            )
            for library in libraries
        ]
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
            frameon=False,
            title="library",
        )

    fig.tight_layout(rect=(0, 0, 0.86, 0.95))

    out_path = Path(out_dir) / f"{bench}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_plots(results: dict, out_dir: Path) -> list[Path]:
    """Write one figure per bench present in `results["records"]` under `out_dir`.

    Each figure is a log-log grid of duration versus median decode time (ms),
    faceted by format (columns) and channel count (rows), with one line per
    library and a direct label at each line's right edge. Only `status == "ok"`
    records are plotted; records with withheld (`None`) timings are skipped
    without raising, and a panel with no `ok` data at all is still rendered
    (as an empty, labelled "no data" panel) rather than omitted.
    """
    records = results.get("records", [])
    benches = sorted({r["bench"] for r in records}, key=_bench_sort_key)
    # One stable slot per library across every figure, so a library keeps its colour
    # whether or not a given figure draws it.
    style_order = sorted({r["library"] for r in records})
    out_dir = Path(out_dir)
    return [_plot_bench(records, bench, out_dir, style_order) for bench in benches]
