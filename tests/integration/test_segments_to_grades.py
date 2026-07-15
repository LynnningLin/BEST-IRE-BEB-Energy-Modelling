import pytest

from beb_soc_model import Segment
from gtfs_to_segment import add_grades_from_dem, build_segments, load_shapes_for_trips, stop_times_for_trip


def test_grade_assignment_preserves_segment_alignment_with_duplicate_stop(
    integration_gtfs_zip, integration_tables, missing_last_elevation
):
    _routes, trips, stops, _calendar, _calendar_dates = integration_tables
    rows = stop_times_for_trip(integration_gtfs_zip, "t1")
    shapes, shape_by_trip = load_shapes_for_trips(integration_gtfs_zip, trips, ["t1"])
    segments = build_segments(rows, stops, shape_points=shapes[shape_by_trip["t1"]])
    before = [(s.length_m, s.run_time_s, s.dwell_s) for s in segments]

    graded = add_grades_from_dem(
        segments,
        rows,
        stops,
        elevation_data=missing_last_elevation,
        shape_points=shapes[shape_by_trip["t1"]],
        verbose=False,
    )

    assert graded is segments
    assert len(graded) == 2
    assert [(s.length_m, s.run_time_s, s.dwell_s) for s in graded] == before
    assert graded[0].grade == pytest.approx(0.02, abs=1e-4)
    assert graded[1].grade == 0.0, "missing s3 elevation should only flatten s2->s3"


def test_grade_assignment_rejects_segment_pair_misalignment(integration_tables, full_elevation):
    _routes, _trips, stops, _calendar, _calendar_dates = integration_tables
    rows = [
        {"stop_id": "s1", "arrival_time": "08:00:00", "departure_time": "08:00:00"},
        {"stop_id": "s2", "arrival_time": "08:02:00", "departure_time": "08:02:00"},
        {"stop_id": "s3", "arrival_time": "08:04:00", "departure_time": "08:04:00"},
    ]

    with pytest.raises(ValueError, match="segment/pair mismatch"):
        add_grades_from_dem(
            [Segment(length_m=500.0)],
            rows,
            stops,
            elevation_data=full_elevation,
            verbose=False,
        )
