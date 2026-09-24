"""Warmup + repeat + aggregation for timing a decode call.

`measure` runs one untimed warmup call (which also warms the OS page cache for the
file being decoded) followed by `repeat` timed trials, with a `gc.collect()` between
each trial so a garbage-collection pause in one trial cannot leak into the next. The
clock is injectable so the aggregation logic itself can be tested deterministically,
without relying on real wall-clock sleeps.
"""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_REPEAT = 7

_NS_PER_MS = 1_000_000.0


@dataclass(frozen=True)
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float
    repeat: int


def measure[T](
    fn: Callable[[], T],
    repeat: int = DEFAULT_REPEAT,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> Timing:
    """Time `repeat` calls to `fn`, after one untimed warmup call.

    `clock` must return an integer count of nanoseconds, matching the contract of
    `time.perf_counter_ns` (the default). Raises `ValueError` if `repeat < 1`.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")

    fn()  # untimed warmup: not included in the statistics, warms the page cache

    durations_ms: list[float] = []
    for _ in range(repeat):
        gc.collect()
        start = clock()
        fn()
        end = clock()
        durations_ms.append((end - start) / _NS_PER_MS)

    return Timing(
        median_ms=statistics.median(durations_ms),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
        repeat=repeat,
    )


def realtime_factor(audio_seconds: float, wall_ms: float) -> float:
    """How many seconds of audio were decoded per second of wall time.

    A realtime factor of 20 means a 1-second clip decoded in 50 ms.
    """
    wall_seconds = wall_ms / 1000.0
    return audio_seconds / wall_seconds
