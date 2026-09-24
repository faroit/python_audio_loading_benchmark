"""Tests for FFmpeg lib-dir discovery and the macOS dyld fallback-path fix.

`ensure_ffmpeg_libs` calls `os.execv` on its one active path, which replaces the
current process image -- calling the real thing here would replace the test
runner. Every test below either exercises a no-op branch (which returns before
`os.execv` is reached) or monkeypatches `os.execv` itself to record the call
instead of performing it, so the test process is never at risk.
"""

from __future__ import annotations

from pathlib import Path

from pabench import ffmpeg_env
from pabench.ffmpeg_env import find_ffmpeg_lib_dir


def test_find_ffmpeg_lib_dir_returns_none_when_no_candidate_has_libavutil(monkeypatch):
    monkeypatch.setattr(ffmpeg_env, "_CANDIDATE_LIB_DIRS", ("/no/such/dir",))
    assert find_ffmpeg_lib_dir() is None


def test_find_ffmpeg_lib_dir_finds_a_matching_directory(tmp_path: Path, monkeypatch):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "libavutil.61.dylib").write_bytes(b"")
    monkeypatch.setattr(ffmpeg_env, "_CANDIDATE_LIB_DIRS", ("/no/such/dir", str(lib_dir)))
    assert find_ffmpeg_lib_dir() == str(lib_dir)


def test_find_ffmpeg_lib_dir_prefers_the_first_matching_candidate(tmp_path: Path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "libavutil.61.dylib").write_bytes(b"")
    (second / "libavutil.60.dylib").write_bytes(b"")
    monkeypatch.setattr(ffmpeg_env, "_CANDIDATE_LIB_DIRS", (str(first), str(second)))
    assert find_ffmpeg_lib_dir() == str(first)


def _no_execv(monkeypatch):
    """Fail the test loudly if `os.execv` is ever actually reached."""

    def _boom(*args, **kwargs):
        raise AssertionError("os.execv must not be called in this branch")

    monkeypatch.setattr(ffmpeg_env.os, "execv", _boom)


def test_ensure_ffmpeg_libs_is_a_noop_off_macos(monkeypatch):
    _no_execv(monkeypatch)
    monkeypatch.setattr(ffmpeg_env.sys, "platform", "linux")
    monkeypatch.delenv(ffmpeg_env._FALLBACK_ENV_VAR, raising=False)
    monkeypatch.delenv(ffmpeg_env._SENTINEL_ENV_VAR, raising=False)
    ffmpeg_env.ensure_ffmpeg_libs()  # must not raise, must not exec


def test_ensure_ffmpeg_libs_is_a_noop_when_fallback_already_set(monkeypatch):
    _no_execv(monkeypatch)
    monkeypatch.setattr(ffmpeg_env.sys, "platform", "darwin")
    monkeypatch.setenv(ffmpeg_env._FALLBACK_ENV_VAR, "/already/set")
    monkeypatch.delenv(ffmpeg_env._SENTINEL_ENV_VAR, raising=False)
    ffmpeg_env.ensure_ffmpeg_libs()


def test_ensure_ffmpeg_libs_is_a_noop_when_sentinel_present(monkeypatch):
    _no_execv(monkeypatch)
    monkeypatch.setattr(ffmpeg_env.sys, "platform", "darwin")
    monkeypatch.delenv(ffmpeg_env._FALLBACK_ENV_VAR, raising=False)
    monkeypatch.setenv(ffmpeg_env._SENTINEL_ENV_VAR, "1")
    ffmpeg_env.ensure_ffmpeg_libs()


def test_ensure_ffmpeg_libs_is_a_noop_when_nothing_found(monkeypatch):
    _no_execv(monkeypatch)
    monkeypatch.setattr(ffmpeg_env.sys, "platform", "darwin")
    monkeypatch.delenv(ffmpeg_env._FALLBACK_ENV_VAR, raising=False)
    monkeypatch.delenv(ffmpeg_env._SENTINEL_ENV_VAR, raising=False)
    monkeypatch.setattr(ffmpeg_env, "find_ffmpeg_lib_dir", lambda: None)
    ffmpeg_env.ensure_ffmpeg_libs()


def test_ensure_ffmpeg_libs_execs_once_with_the_variable_set(monkeypatch):
    monkeypatch.setattr(ffmpeg_env.sys, "platform", "darwin")
    monkeypatch.delenv(ffmpeg_env._FALLBACK_ENV_VAR, raising=False)
    monkeypatch.delenv(ffmpeg_env._SENTINEL_ENV_VAR, raising=False)
    monkeypatch.setattr(ffmpeg_env, "find_ffmpeg_lib_dir", lambda: "/opt/homebrew/lib")

    calls = []

    def fake_execv(executable, args):
        calls.append((executable, args))

    monkeypatch.setattr(ffmpeg_env.os, "execv", fake_execv)

    try:
        ffmpeg_env.ensure_ffmpeg_libs()

        assert len(calls) == 1
        assert ffmpeg_env.os.environ[ffmpeg_env._FALLBACK_ENV_VAR] == "/opt/homebrew/lib"
        assert ffmpeg_env.os.environ[ffmpeg_env._SENTINEL_ENV_VAR] == "1"
    finally:
        # ensure_ffmpeg_libs sets these directly on os.environ (that's the whole
        # point -- they must survive into the execv'd process), so monkeypatch's
        # own env fixtures never saw the mutation and won't undo it. Clean up by
        # hand so this test can't leak state into whatever runs after it.
        ffmpeg_env.os.environ.pop(ffmpeg_env._FALLBACK_ENV_VAR, None)
        ffmpeg_env.os.environ.pop(ffmpeg_env._SENTINEL_ENV_VAR, None)
