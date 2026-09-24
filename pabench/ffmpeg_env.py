"""Make Homebrew/MacPorts FFmpeg visible to dyld before torchcodec loads it.

torchcodec ships a native library built against a specific FFmpeg major version
(4 through 9). On macOS that native library carries no `LC_RPATH`, so dyld falls
back to searching `DYLD_FALLBACK_LIBRARY_PATH`, whose OS default
(`$HOME/lib:/usr/local/lib:/lib:/usr/lib`) omits Homebrew's `/opt/homebrew/lib`.
A perfectly compatible FFmpeg install is then present but invisible to dyld, and
torchcodec fails to load with a `Library not loaded` error.

This cannot be fixed from inside an already-running process: dyld reads
`DYLD_FALLBACK_LIBRARY_PATH` once, at process start, and a `ctypes` preload of the
library does not help either, because dyld re-searches the fallback path itself
when a *dependent* dylib is loaded rather than reusing an already-loaded image.
Restarting the interpreter with the variable set is the only remedy, which is
what `ensure_ffmpeg_libs` does, via `os.execv`, at most once.
"""

from __future__ import annotations

import glob
import os
import sys

# Search order matches common install prefixes: Homebrew on Apple Silicon,
# Homebrew/other package managers on Intel Macs, and MacPorts.
_CANDIDATE_LIB_DIRS: tuple[str, ...] = (
    "/opt/homebrew/lib",
    "/usr/local/lib",
    "/opt/local/lib",
)

_FALLBACK_ENV_VAR = "DYLD_FALLBACK_LIBRARY_PATH"
# Set on the restarted process alongside DYLD_FALLBACK_LIBRARY_PATH so a second
# call to ensure_ffmpeg_libs() (e.g. because the caller sets the fallback var
# itself and also calls this function) can never re-exec in a loop.
_SENTINEL_ENV_VAR = "_PABENCH_FFMPEG_LIBS_ENSURED"


def find_ffmpeg_lib_dir() -> str | None:
    """Return the first candidate directory containing a `libavutil*.dylib`, else None."""
    for directory in _CANDIDATE_LIB_DIRS:
        if glob.glob(os.path.join(directory, "libavutil*.dylib")):
            return directory
    return None


def ensure_ffmpeg_libs() -> None:
    """Restart the process once, on macOS, with `DYLD_FALLBACK_LIBRARY_PATH` set.

    A no-op in every one of these cases:
    - not running on macOS,
    - `DYLD_FALLBACK_LIBRARY_PATH` is already set (an explicit value, from the
      environment or from a previous call, is always respected as-is), or
    - no candidate FFmpeg lib directory is found.

    The sentinel environment variable guards against ever re-executing more than
    once: the restarted process inherits it, so a second call in that process
    (however it might occur) is also a no-op.

    Caller contract: this function must be called only from the console entry
    point (`pabench.cli.console_main`), never from `main()` or from anything a
    test suite calls. `os.execv` replaces the current process image in place; if
    it were reachable from code under test, running the test suite would restart
    the test runner itself mid-suite instead of returning.
    """
    if sys.platform != "darwin":
        return
    if _SENTINEL_ENV_VAR in os.environ:
        return
    if _FALLBACK_ENV_VAR in os.environ:
        return

    lib_dir = find_ffmpeg_lib_dir()
    if lib_dir is None:
        return

    os.environ[_FALLBACK_ENV_VAR] = lib_dir
    os.environ[_SENTINEL_ENV_VAR] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
