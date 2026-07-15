import pytest

from beb_soc_model import build_speed_profile
from gtfs_to_segment import build_segments, load_shapes_for_trips, stop_times_for_trip


def test_gtfs_segment_runtime_is_preserved_by_feasible_motion_profile(
    integration_gtfs_zip, integration_tables, integration_motion
):
    _routes, trips, stops, _calendar, _calendar_dates = integration_tables
    rows = stop_times_for_trip(integration_gtfs_zip, "t1")
    shapes, shape_by_trip = load_shapes_for_trips(integration_gtfs_zip, trips, ["t1"])
    [first, _second] = build_segments(
        rows, stops, shape_points=shapes[shape_by_trip["t1"]]
    )

    profile, diag = build_speed_profile(
        first, motion_params=integration_motion, return_diagnostics=True
    )

    assert diag["scheduled_run_time_s"] == first.run_time_s
    assert diag["actual_profile_time_s"] == pytest.approx(first.run_time_s)
    assert diag["schedule_infeasible"] is False
    assert diag["speed_cap_source"] == "config_default"
    assert max(profile[1]) <= diag["speed_cap_ms"] + 1e-9
