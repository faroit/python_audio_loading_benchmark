import pytest

from pabench.timing import DEFAULT_REPEAT, Timing, measure, realtime_factor


def _fake_clock(timestamps_ns: list[int]):
    """A deterministic clock that yields each timestamp in order, then raises."""
    it = iter(timestamps_ns)

    def clock() -> int:
        return next(it)

    return clock


def test_default_repeat_is_seven():
    assert DEFAULT_REPEAT == 7


def test_measure_aggregates_over_injected_clock():
    # Five timed trials of 10, 20, 5, 15, 12 ms -> median 12, min 5, max 20.
    durations_ms = [10, 20, 5, 15, 12]
    timestamps_ns = []
    t = 0
    for d in durations_ms:
        timestamps_ns.append(t)
        t += int(d * 1_000_000)
        timestamps_ns.append(t)
    clock = _fake_clock(timestamps_ns)

    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    timing = measure(fn, repeat=5, clock=clock)

    assert isinstance(timing, Timing)
    assert timing.median_ms == pytest.approx(12)
    assert timing.min_ms == pytest.approx(5)
    assert timing.max_ms == pytest.approx(20)
    assert timing.repeat == 5


def test_warmup_is_excluded_from_statistics():
    # Exactly 2 * repeat timestamps are supplied. If the warmup call were also
    # timed, the clock would be exhausted before the last trial and raise
    # StopIteration instead of completing.
    repeat = 3
    timestamps_ns = [0, 1_000_000, 1_000_000, 2_000_000, 2_000_000, 3_000_000]
    clock = _fake_clock(timestamps_ns)

    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    timing = measure(fn, repeat=repeat, clock=clock)

    assert timing.repeat == repeat
    # One untimed warmup call plus one call per timed trial.
    assert calls["n"] == repeat + 1
    # All supplied timestamps were consumed, and no more were requested.
    with pytest.raises(StopIteration):
        clock()


def test_repeat_zero_raises_value_error():
    with pytest.raises(ValueError):
        measure(lambda: None, repeat=0)


def test_repeat_negative_raises_value_error():
    with pytest.raises(ValueError):
        measure(lambda: None, repeat=-1)


def test_realtime_factor():
    # 10 s of audio decoded in 500 ms of wall time is 20x realtime.
    assert realtime_factor(10.0, 500.0) == pytest.approx(20.0)


def test_realtime_factor_slower_than_realtime():
    # 1 s of audio decoded in 2000 ms of wall time is 0.5x realtime.
    assert realtime_factor(1.0, 2000.0) == pytest.approx(0.5)
