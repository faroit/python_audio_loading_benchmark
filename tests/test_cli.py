"""Tests for `pabench.cli`.

Every test here calls `main(argv)` directly, never `console_main`: `console_main`
calls `pabench.ffmpeg_env.ensure_ffmpeg_libs`, which `os.execv`s the interpreter on
its one active macOS path and would restart this very test suite mid-run (see
`pabench/ffmpeg_env.py`, `pabench/cli.py`). `test_main_never_calls_ensure_ffmpeg_libs`
below guards that boundary directly, the same way `tests/test_ffmpeg_env.py` guards
`os.execv` itself.

`soundfile` is used as the one real library exercised end-to-end: it is a base
dependency (`pyproject.toml`), never an optional one, so it is always installed
wherever this suite runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pabench import cli
from pabench.cli import console_main, main
from pabench.loaders import PROBES

# A single, cheap spec (1 second, mono, PCM_16 WAV) used by every test below that
# needs a real (but tiny) corpus.
_TINY_FILTERS = ["--durations", "1", "--channels", "1", "--formats", "wav_pcm16"]
_TINY_FILENAME = "1s_1ch_wav_pcm16.wav"


def _gen(corpus_dir: Path) -> int:
    return main(["gen", "--corpus-dir", str(corpus_dir), *_TINY_FILTERS])


# ---- gen / run / report round trip -------------------------------------------


def test_gen_writes_only_the_selected_corpus_file(tmp_path):
    corpus_dir = tmp_path / "corpus"
    rc = _gen(corpus_dir)
    assert rc == 0
    written = list(corpus_dir.iterdir())
    assert [p.name for p in written] == [_TINY_FILENAME]


def test_run_writes_results_for_only_the_requested_library(tmp_path):
    corpus_dir = tmp_path / "corpus"
    results_path = tmp_path / "results.json"
    assert _gen(corpus_dir) == 0

    rc = main(
        [
            "run",
            "--corpus-dir",
            str(corpus_dir),
            *_TINY_FILTERS,
            "--library",
            "soundfile",
            "--bench",
            "full",
            "--repeat",
            "1",
            "--out",
            str(results_path),
        ]
    )
    assert rc == 0
    assert results_path.exists()

    data = json.loads(results_path.read_text())
    assert data["records"]
    assert {r["library"] for r in data["records"]} == {"soundfile"}
    assert {r["bench"] for r in data["records"]} == {"full"}
    (record,) = data["records"]
    assert record["status"] == "ok"
    assert record["median_ms"] is not None


def test_report_renders_markdown_and_plots_from_an_existing_results_file(tmp_path):
    corpus_dir = tmp_path / "corpus"
    results_path = tmp_path / "results.json"
    report_path = tmp_path / "report.md"
    assert _gen(corpus_dir) == 0
    assert (
        main(
            [
                "run",
                "--corpus-dir",
                str(corpus_dir),
                *_TINY_FILTERS,
                "--library",
                "soundfile",
                "--repeat",
                "1",
                "--out",
                str(results_path),
            ]
        )
        == 0
    )

    rc = main(["report", "--results", str(results_path), "--out", str(report_path)])
    assert rc == 0
    assert report_path.exists()
    markdown = report_path.read_text()
    assert "soundfile" in markdown
    assert (report_path.parent / "full.png").exists()
    assert (report_path.parent / "seek.png").exists()


def test_report_never_touches_the_corpus_or_reruns_anything(tmp_path, monkeypatch):
    # report only reads results.json; deleting the corpus dir must not matter.
    corpus_dir = tmp_path / "corpus"
    results_path = tmp_path / "results.json"
    report_path = tmp_path / "report.md"
    assert _gen(corpus_dir) == 0
    assert (
        main(
            [
                "run",
                "--corpus-dir",
                str(corpus_dir),
                *_TINY_FILTERS,
                "--library",
                "soundfile",
                "--repeat",
                "1",
                "--out",
                str(results_path),
            ]
        )
        == 0
    )

    def _boom(*args, **kwargs):
        raise AssertionError("report must never call the run loop")

    monkeypatch.setattr(cli, "run_benchmark", _boom)
    rc = main(["report", "--results", str(results_path), "--out", str(report_path)])
    assert rc == 0
    assert report_path.exists()


# ---- all: the single reproduce command ---------------------------------------


def test_all_generates_runs_and_reports_end_to_end_on_a_tiny_corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    out_path = tmp_path / "results.json"

    rc = main(
        [
            "all",
            "--corpus-dir",
            str(corpus_dir),
            *_TINY_FILTERS,
            "--repeat",
            "1",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert (corpus_dir / _TINY_FILENAME).exists()
    assert out_path.exists()
    assert (out_path.parent / "report.md").exists()
    assert (out_path.parent / "full.png").exists()
    assert (out_path.parent / "seek.png").exists()

    data = json.loads(out_path.read_text())
    # every registered library was probed, whether or not it ended up available
    assert {r["library"] for r in data["records"]} == set(PROBES)


def test_all_does_not_regenerate_a_file_that_already_exists(tmp_path):
    corpus_dir = tmp_path / "corpus"
    assert _gen(corpus_dir) == 0
    original = (corpus_dir / _TINY_FILENAME).read_bytes()

    rc = main(
        [
            "all",
            "--corpus-dir",
            str(corpus_dir),
            *_TINY_FILTERS,
            "--repeat",
            "1",
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 0
    # byte-identical: `all` must not have regenerated (and thus rewritten) it
    assert (corpus_dir / _TINY_FILENAME).read_bytes() == original


# ---- error handling: exit codes and messages ---------------------------------


def test_unknown_library_exits_2_with_a_stderr_message_not_a_traceback(capsys):
    rc = main(["run", "--library", "not_a_real_library"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err
    assert captured.err.strip() != ""


def test_unknown_format_exits_2_with_a_stderr_message_not_a_traceback(capsys):
    rc = main(["gen", "--formats", "not_a_real_format"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err
    assert captured.err.strip() != ""


def test_missing_corpus_exits_nonzero_and_mentions_pabench_gen(tmp_path, capsys):
    empty_corpus_dir = tmp_path / "no_such_corpus"
    rc = main(
        [
            "run",
            "--corpus-dir",
            str(empty_corpus_dir),
            *_TINY_FILTERS,
            "--library",
            "soundfile",
        ]
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert "Traceback" not in captured.err
    assert "pabench gen" in captured.err


# ---- progress and stdout/stderr separation -----------------------------------


def test_run_progress_goes_to_stderr_and_stdout_stays_clean(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    assert _gen(corpus_dir) == 0
    capsys.readouterr()  # discard gen's own stderr output

    rc = main(
        [
            "run",
            "--corpus-dir",
            str(corpus_dir),
            *_TINY_FILTERS,
            "--library",
            "soundfile",
            "--repeat",
            "1",
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert "soundfile" in captured.err
    assert _TINY_FILENAME in captured.err


# ---- the ensure_ffmpeg_libs / main boundary ----------------------------------


def test_main_never_calls_ensure_ffmpeg_libs(monkeypatch, tmp_path):
    def _boom():
        raise AssertionError("main() must never call ensure_ffmpeg_libs (see pabench/cli.py)")

    monkeypatch.setattr(cli, "ensure_ffmpeg_libs", _boom)
    corpus_dir = tmp_path / "corpus"
    rc = main(["gen", "--corpus-dir", str(corpus_dir), *_TINY_FILTERS])
    assert rc == 0


def test_console_main_calls_ensure_ffmpeg_libs_then_main(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "ensure_ffmpeg_libs", lambda: calls.append("ensure"))
    monkeypatch.setattr(cli, "main", lambda: calls.append("main") or 0)
    assert console_main() == 0
    assert calls == ["ensure", "main"]


def test_no_command_exits_2_not_a_traceback(capsys):
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("argv", [["--help"], ["run", "--help"]])
def test_help_exits_0(argv, capsys):
    rc = main(argv)
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out != ""
