"""Command-line interface: `pabench gen|run|report|all`.

`main(argv=None) -> int` is what tests call directly and what `console_main` calls
after the FFmpeg-visibility fix has already run. `main` must never call
`pabench.ffmpeg_env.ensure_ffmpeg_libs` itself: that function `os.execv`s the
running interpreter on its one active (macOS) path, which replaces the process
image in place. Calling it from code a test suite calls directly would restart
the suite mid-run rather than return -- see `pabench/ffmpeg_env.py`. Only
`console_main`, the actual `pabench` console-script entry point
(`pyproject.toml`), is allowed to call it, exactly once, before any argument
parsing happens.

Argument validation (an unknown `--library` or `--formats` value) is handled by
`argparse`'s own `choices`, so an unknown name is a normal parse error: a one-line
usage message on stderr and exit code 2, never a Python traceback. A missing
corpus surfaces as `FileNotFoundError` from `pabench.corpus.corpus_files` (via
`pabench.run.run`), whose message already names `pabench gen`; `run`/`all` catch
it, print it as-is, and exit 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pabench.corpus import DEFAULT_SPECS, FORMATS, CorpusSpec, generate
from pabench.ffmpeg_env import ensure_ffmpeg_libs
from pabench.loaders import PROBES, available_loaders
from pabench.report import render_markdown, write_plots
from pabench.run import run as run_benchmark
from pabench.run import write_results
from pabench.timing import DEFAULT_REPEAT

#: Environment variable that supplies the default for `--corpus-dir`, so a corpus on
#: another disk does not have to be named again on every subcommand.
CORPUS_DIR_ENV = "PABENCH_CORPUS_DIR"

#: "both" is kept as a backwards-compatible alias for "full,seek" (it predates the
#: "bytes" bench); "all" is the new alias for every bench, "full,seek,bytes".
_BENCH_CHOICES = ("full", "seek", "bytes", "both", "all")
_FORMAT_CHOICES = tuple(fmt.key for fmt in FORMATS)


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    """`--corpus-dir`/`--durations`/`--channels`/`--formats`, shared by all four subcommands."""
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get(CORPUS_DIR_ENV, "corpus"),
        help=(
            "directory holding the corpus (default: %(default)s; "
            f"set {CORPUS_DIR_ENV} to change it without passing this flag). "
            "Point it at another disk to generate and read the corpus there."
        ),
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=None,
        metavar="SECONDS",
        help="restrict to these durations (default: all)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="restrict to these channel counts (default: all)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        choices=_FORMAT_CHOICES,
        metavar="FORMAT",
        help="restrict to these format keys (default: all)",
    )


def _select_specs(args: argparse.Namespace) -> tuple[CorpusSpec, ...]:
    """`DEFAULT_SPECS` filtered by `--durations`/`--channels`/`--formats`, if given."""
    durations = set(args.durations) if args.durations else None
    channels = set(args.channels) if args.channels else None
    formats = set(args.formats) if args.formats else None

    def _matches(spec: CorpusSpec) -> bool:
        return (
            (durations is None or spec.duration_s in durations)
            and (channels is None or spec.channels in channels)
            and (formats is None or spec.fmt.key in formats)
        )

    return tuple(spec for spec in DEFAULT_SPECS if _matches(spec))


def _bench_tuple(bench: str) -> tuple[str, ...]:
    if bench == "both":
        return ("full", "seek")
    if bench == "all":
        return ("full", "seek", "bytes")
    return (bench,)


def _progress(library: str, filename: str) -> None:
    print(f"[{library}] {filename}", file=sys.stderr)


def _cmd_gen(args: argparse.Namespace) -> int:
    specs = _select_specs(args)
    paths = generate(Path(args.corpus_dir), specs=specs)
    print(f"generated {len(paths)} corpus file(s) under {args.corpus_dir}", file=sys.stderr)
    return 0


def _missing_libraries_message(loaders: list) -> str:
    """Explain which libraries could not be loaded, and how to proceed."""
    missing = [loader for loader in loaders if not loader.available]
    lines = [
        (
            f"{len(missing)} of {len(loaders)} libraries could not be loaded, so this "
            "run would compare a subset and report the rest as unavailable:"
        ),
        "",
    ]
    lines += [f"  {loader.name}: {loader.error}" for loader in missing]
    lines += [
        "",
        (
            "Install everything with `uv sync` (the libraries under test are required "
            "dependencies, not an extra)."
        ),
        (
            "If a library genuinely cannot be installed on this platform, re-run with "
            "--allow-missing to benchmark the rest and record these as unavailable."
        ),
    ]
    return "\n".join(lines)


def _cmd_run(args: argparse.Namespace) -> int:
    specs = _select_specs(args)
    loaders = available_loaders(args.library)
    benches = _bench_tuple(args.bench)

    # Refuse by default rather than publishing a report whose cells are mostly
    # "unavailable": that reads as a benchmark result and is not one.
    if not args.allow_missing and any(not loader.available for loader in loaders):
        print(_missing_libraries_message(loaders), file=sys.stderr)
        return 2
    try:
        results = run_benchmark(
            Path(args.corpus_dir),
            loaders,
            specs=specs,
            repeat=args.repeat,
            benches=benches,
            progress=_progress,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    write_results(results, Path(args.out))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _filter_records(results: dict, specs: tuple[CorpusSpec, ...]) -> dict:
    """Restrict an already-loaded results dict to the axes named by `specs`."""
    allowed = {(spec.duration_s, spec.channels, spec.fmt.key) for spec in specs}
    records = [
        r
        for r in results.get("records", [])
        if (r["duration_s"], r["channels"], r["format"]) in allowed
    ]
    return {"platform": results.get("platform", {}), "records": records}


def _cmd_report(args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    try:
        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError as exc:
        print(
            f"{exc}. Run `pabench run` (or `pabench all`) first to produce a results file.",
            file=sys.stderr,
        )
        return 1

    if args.durations or args.channels or args.formats:
        results = _filter_records(results, _select_specs(args))

    out_path = Path(args.out)
    if out_path.is_dir():
        print(
            f"--out must name a markdown file, not a directory: {out_path}. "
            f"Try --out {out_path / 'report.md'}",
            file=sys.stderr,
        )
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(results), encoding="utf-8")
    write_plots(results, out_path.parent)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    """Generate only the corpus files that are missing, run all three benches, report."""
    specs = _select_specs(args)
    corpus_dir = Path(args.corpus_dir)
    missing = tuple(spec for spec in specs if not (corpus_dir / spec.filename).exists())
    if missing:
        generate(corpus_dir, specs=missing)
        print(
            f"generated {len(missing)} missing corpus file(s) under {args.corpus_dir}",
            file=sys.stderr,
        )

    loaders = available_loaders(None)
    try:
        results = run_benchmark(
            corpus_dir,
            loaders,
            specs=specs,
            repeat=args.repeat,
            benches=("full", "seek", "bytes"),
            progress=_progress,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_path = Path(args.out)
    write_results(results, out_path)
    report_path = out_path.parent / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(results), encoding="utf-8")
    write_plots(results, out_path.parent)
    print(f"wrote {out_path} and {report_path}", file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pabench", description="Benchmark Python audio-loading libraries against torch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("gen", help="generate the corpus")
    _add_filter_args(gen_parser)
    gen_parser.set_defaults(func=_cmd_gen)

    run_parser = subparsers.add_parser("run", help="run the benchmark, writing results.json")
    _add_filter_args(run_parser)
    run_parser.add_argument(
        "--library",
        action="append",
        choices=sorted(PROBES),
        metavar="NAME",
        help="restrict to this library (repeatable; default: all eleven)",
    )
    run_parser.add_argument("--bench", choices=_BENCH_CHOICES, default="both")
    run_parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    run_parser.add_argument("--out", default="results/results.json")
    run_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="benchmark the libraries that did load instead of refusing; the rest "
        "are recorded as unavailable with their import errors",
    )
    run_parser.set_defaults(func=_cmd_run)

    report_parser = subparsers.add_parser(
        "report", help="render the markdown report and plots from an existing results.json"
    )
    _add_filter_args(report_parser)
    report_parser.add_argument("--results", default="results/results.json")
    report_parser.add_argument("--out", default="results/report.md")
    report_parser.set_defaults(func=_cmd_report)

    all_parser = subparsers.add_parser(
        "all",
        help="generate what's missing, run all three benchmarks (all libraries), write the report",
    )
    _add_filter_args(all_parser)
    all_parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    all_parser.add_argument("--out", default="results/results.json")
    all_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="benchmark the libraries that did load instead of refusing; the rest "
        "are recorded as unavailable with their import errors",
    )
    all_parser.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` (`sys.argv[1:]` if omitted) and run the selected subcommand.

    Never calls `ensure_ffmpeg_libs` -- see the module docstring. An `argparse`
    parse error (unknown subcommand, unknown `--library`/`--formats` value, `--help`)
    raises `SystemExit`; that is caught here and turned into this function's `int`
    return, rather than left to propagate and kill the calling process, so `main`
    can be called directly (as every test in `tests/test_cli.py` does) without
    also exiting the test runner.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    return args.func(args)


def console_main() -> int:
    """The `pabench` console-script entry point (see `pyproject.toml`).

    Fixes FFmpeg's dyld visibility for torchcodec *before* anything else runs,
    which on its one active path replaces the process via `os.execv`. `main`
    itself must never do this (see the module docstring); only this function may.
    """
    ensure_ffmpeg_libs()
    return main()


if __name__ == "__main__":
    sys.exit(console_main())
