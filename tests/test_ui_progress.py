from archiver.ui_status import RunProgress, compute_rate, format_eta, progress_line


def test_queued_is_what_is_neither_done_nor_running():
    p = RunProgress(total=100, completed=10, in_flight=4)
    assert p.queued == 86


def test_queued_never_goes_negative():
    assert RunProgress(total=2, completed=2, in_flight=4).queued == 0


def test_no_completions_means_no_rate():
    assert compute_rate([], now=100.0, started_at=40.0) is None


def test_rate_is_completions_over_the_time_actually_covered():
    # three files in the sixty seconds since the run started
    assert compute_rate(
        [50.0, 70.0, 90.0], now=100.0, started_at=40.0, window_s=60.0
    ) == 3 / 60


def test_a_burst_does_not_inflate_the_rate():
    """Four files finishing within a second of each other is a wave arriving,
    not the run suddenly going four times faster."""
    burst = [98.0, 98.2, 98.4, 98.6]
    rate = compute_rate(burst, now=99.0, started_at=39.0, window_s=60.0)
    assert rate == 4 / 60


def test_the_window_never_reaches_back_before_the_run_started():
    # nine seconds in, one file done: one per nine seconds, not one per sixty
    assert compute_rate([95.0], now=100.0, started_at=91.0, window_s=60.0) == 1 / 9


def test_rate_ignores_completions_outside_the_window():
    assert compute_rate(
        [0.0, 95.0], now=100.0, started_at=0.0, window_s=60.0
    ) == 1 / 60


def test_eta_formats_by_magnitude():
    assert format_eta(None) == ""
    assert format_eta(0) == ""
    assert format_eta(45) == "~45s left"
    assert format_eta(600) == "~10m left"
    assert format_eta(3840) == "~1h04m left"


def test_progress_line_without_a_rate_omits_the_estimate():
    line = progress_line(
        RunProgress(total=810, completed=12, in_flight=4), rate=None, total_files=810
    )
    assert line == (
        "files: 810 • queued: 794 • in flight: 4 • done: 12 • skipped: 0 • error: 0"
    )


def test_progress_line_with_a_rate_shows_throughput_and_eta():
    line = progress_line(
        RunProgress(total=810, completed=12, in_flight=4), rate=0.2, total_files=810
    )
    assert "0.20 file/s" in line
    # 798 left at 0.2/s is 3990s
    assert "~1h06m left" in line


def test_the_table_total_can_differ_from_the_run_total():
    """A table can hold files classified in an earlier session; the run counts
    only its own targets."""
    line = progress_line(
        RunProgress(total=40, completed=10, in_flight=2), rate=None, total_files=810
    )
    assert "files: 810" in line
    assert "queued: 28" in line
