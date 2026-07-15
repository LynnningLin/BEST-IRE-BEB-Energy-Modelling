import numpy as np
import pytest

from gtfs_to_segment import build_segments, load_shapes_for_trips, stop_times_for_trip
from speed_caps import add_speed_caps
from beb_soc_model import build_speed_profile


def test_speed_cap_assignment_flows_to_motion_diagnostics(
    integration_gtfs_zip, integration_tables, integration_motion
):
    _routes, trips, stops, _calendar, _calendar_dates = integration_tables
    rows = stop_times_for_trip(integration_gtfs_zip, "t1")
    shapes, shape_by_trip = load_shapes_for_trips(integration_gtfs_zip, trips, ["t1"])
    segments = build_segments(rows, stops, shape_points=shapes[shape_by_trip["t1"]])
    before = [(s.length_m, s.run_time_s, s.passengers, s.grade, s.n_signals) for s in segments]

    add_speed_caps(
        segments,
        rows,
        stops,
        cap_map={("sdup", "s2"): 6.0, ("s2", "s3"): 4.0},
        source_map={("sdup", "s2"): "cache_low", ("s2", "s3"): "cache_tight"},
        shape_points=shapes[shape_by_trip["t1"]],
        verbose=False,
    )

    assert [(s.length_m, s.run_time_s, s.passengers, s.grade, s.n_signals) for s in segments] == before
    assert [s.speed_cap_ms for s in segments] == [6.0, 4.0]

    first_profile, first_diag = build_speed_profile(
        segments[0], motion_params=integration_motion, return_diagnostics=True
    )
    second_profile, second_diag = build_speed_profile(
        segments[1], motion_params=integration_motion, return_diagnostics=True
    )

    assert first_diag["speed_cap_source"] == "cache_low"
    assert second_diag["speed_cap_source"] == "cache_tight"
    assert first_diag["target_cruise_ms"] < first_diag["speed_cap_ms"]
    assert first_diag["schedule_infeasible"] is False
    assert second_diag["schedule_infeasible"] is True
    assert np.max(first_profile[1]) <= 6.0 + 1e-9
    assert np.max(second_profile[1]) <= 4.0 + 1e-9
    assert second_diag["schedule_delay_s"] > 0.0
