from archiver.cache import SaveThrottle


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_it_fires_on_the_count_trigger():
    clock = FakeClock()
    throttle = SaveThrottle(min_interval_s=1000.0, min_dirty=3, clock=clock)
    assert throttle.record() is False
    assert throttle.record() is False
    assert throttle.record() is True
    assert throttle.record() is False


def test_it_fires_on_the_time_trigger():
    clock = FakeClock()
    throttle = SaveThrottle(min_interval_s=5.0, min_dirty=1000, clock=clock)
    assert throttle.record() is False
    clock.now = 5.0
    assert throttle.record() is True


def test_flush_writes_only_what_is_pending():
    clock = FakeClock()
    throttle = SaveThrottle(min_interval_s=1000.0, min_dirty=1000, clock=clock)
    assert throttle.flush() is False, "nothing changed, nothing to write"
    throttle.record()
    assert throttle.flush() is True
    assert throttle.flush() is False
