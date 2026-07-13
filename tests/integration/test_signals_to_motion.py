import pytest
import numpy as np

from beb_soc_model import build_speed_profile
from gtfs_to_segment import build_segments, load_shapes_for_trips, stop_times_for_trip
from traffic_signals import add_traffic_signals


def test_signal_counts_flow_to_motion_sublinks_and_wait_diagnostics(
    integration_gtfs_zip, integration_tables, integration_motion
):
    _routes, trips, stops, _calendar, _calendar_dates = integration_tables
    rows = stop_times_for_trip(integration_gtfs_zip, "t1")
    shapes, shape_by_trip = load_shapes_for_trips(integration_gtfs_zip, trips, ["t1"])
    segments = build_segments(rows, stops, shape_points=shapes[shape_by_trip["t1"]])
    before = [(s.length_m, s.run_time_s, s.speed_cap_ms, s.passengers, s.grade) for s in segments]

    no_signal_profile, no_signal_diag = build_speed_profile(
        segments[0], motion_params=integration_motion, return_diagnostics=True
    )

    add_traffic_signals(
        segments,
        rows,
        stops,
        count_map={("sdup", "s2"): 1, ("s2", "s3"): 0},
        source_map={("sdup", "s2"): "cache", ("s2", "s3"): "cache"},
        shape_points=shapes[shape_by_trip["t1"]],
        verbose=False,
    )

    assert [(s.length_m, s.run_time_s, s.speed_cap_ms, s.passengers, s.grade) for s in segments] == before
    assert [s.n_signals for s in segments] == [1, 0]
    assert [s.signal_source for s in segments] == ["cache", "cache"]

    signal_profile, signal_diag = build_speed_profile(
        segments[0], motion_params=integration_motion, return_diagnostics=True
    )
    original_run_time = segments[0].run_time_s
    reduced_seg = segments[0]
    reduced_seg.run_time_s = 75.0
    reduced_profile, reduced_diag = build_speed_profile(
        reduced_seg, motion_params=integration_motion, return_diagnostics=True
    )

    assert no_signal_diag["n_effective_signal_stops"] == 0
    assert signal_diag["n_motion_sublinks"] == 2
    assert signal_diag["n_effective_signal_stops"] == 1
    assert signal_diag["signal_wait_s"] == pytest.approx(10.0)
    assert signal_diag["actual_profile_time_s"] == pytest.approx(original_run_time)
    assert reduced_diag["signal_wait_s"] < reduced_diag["signal_wait_requested_s"]
    assert reduced_diag["signal_wait_reduced_s"] > 0.0
    assert reduced_diag["schedule_delay_s"] == pytest.approx(0.0)
    assert reduced_diag["actual_profile_time_s"] == pytest.approx(75.0)
    assert np.count_nonzero(signal_profile[1] == 0.0) > np.count_nonzero(
        no_signal_profile[1] == 0.0
    )
    assert len(reduced_profile[0]) > 0
