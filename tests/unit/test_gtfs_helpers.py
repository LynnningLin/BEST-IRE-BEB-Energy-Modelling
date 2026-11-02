from datetime import date

import pandas as pd
import pytest

from beb_soc_model import Segment
from gtfs_to_segment import (
    _repaired_shape_rows,
    _shape_points_with_cumdist,
    build_segments,
    gtfs_time_to_seconds,
    haversine_m,
    implausible_intervals,
    iter_valid_stop_pairs,
    resolve_route_id,
    seconds_to_hhmmss,
    select_trips_by_start,
    service_ids_for_date,
    service_ids_for_day,
    stop_times_for_trips,
    trip_start_seconds,
    trips_for_route,
)


def _rows(trip_id="t1", start="08:00:00", second="08:05:00", third="08:10:00"):
    return [
        {
            "trip_id": trip_id,
            "arrival_time": start,
            "departure_time": start,
            "stop_id": "a",
            "stop_sequence": "1",
        },
        {
            "trip_id": trip_id,
            "arrival_time": second,
            "departure_time": "08:05:30",
            "stop_id": "b",
            "stop_sequence": "2",
        },
        {
            "trip_id": trip_id,
            "arrival_time": third,
            "departure_time": "08:10:45",
            "stop_id": "c",
            "stop_sequence": "3",
        },
    ]


def _stops():
    return pd.DataFrame(
        [
            {"stop_id": "a", "stop_lat": 0.0, "stop_lon": 0.0},
            {"stop_id": "b", "stop_lat": 0.0, "stop_lon": 0.001},
            {"stop_id": "c", "stop_lat": 0.001, "stop_lon": 0.001},
        ]
    )


def test_gtfs_time_round_trips_exact_values():
    assert gtfs_time_to_seconds("01:02:03") == 3723
    assert gtfs_time_to_seconds("25:10:00") == 90600
    assert seconds_to_hhmmss(90600) == "25:10:00"


def test_trip_start_and_stop_time_ordering(minimal_gtfs_zip):
    feed = minimal_gtfs_zip(
        overrides={
            "stop_times.txt": [
                {
                    "trip_id": "t1",
                    "arrival_time": "08:10:00",
                    "departure_time": "08:10:00",
                    "stop_id": "s3",
                    "stop_sequence": "3",
                },
                {
                    "trip_id": "t1",
                    "arrival_time": "08:00:00",
                    "departure_time": "08:00:00",
                    "stop_id": "s1",
                    "stop_sequence": "1",
                },
                {
                    "trip_id": "t1",
                    "arrival_time": "08:05:00",
                    "departure_time": "08:05:00",
                    "stop_id": "s2",
                    "stop_sequence": "2",
                },
            ]
        }
    )

    rows = stop_times_for_trips(feed, ["t1"])["t1"]

    assert [r["stop_sequence"] for r in rows] == ["1", "2", "3"]
    assert trip_start_seconds(rows) == 8 * 3600


def test_nearest_trip_selection_is_ordered_and_within_tolerance():
    by_trip = {
        "late": _rows("late", "09:00:00"),
        "early": _rows("early", "07:00:00"),
        "mid": _rows("mid", "08:00:00"),
    }

    assert select_trips_by_start(by_trip) == ["early", "mid", "late"]
    assert select_trips_by_start(by_trip, ["08:04"], tolerance_s=5 * 60) == ["mid"]
    assert select_trips_by_start(by_trip, ["08:20"], tolerance_s=5 * 60) == []


def test_weekday_date_exceptions_and_direction_filtering():
    calendar = pd.DataFrame(
        [
            {
                "service_id": "weekday",
                "monday": 1,
                "tuesday": 0,
                "wednesday": 0,
                "thursday": 0,
                "friday": 0,
                "saturday": 0,
                "sunday": 0,
                "start_date": "20250101",
                "end_date": "20251231",
            },
            {
                "service_id": "removed",
                "monday": 1,
                "tuesday": 0,
                "wednesday": 0,
                "thursday": 0,
                "friday": 0,
                "saturday": 0,
                "sunday": 0,
                "start_date": "20250101",
                "end_date": "20251231",
            },
        ]
    )
    calendar_dates = pd.DataFrame(
        [
            {"service_id": "special", "date": "20250707", "exception_type": 1},
            {"service_id": "removed", "date": "20250707", "exception_type": 2},
        ]
    )
    trips = pd.DataFrame(
        [
            {"trip_id": "a", "route_id": "r1", "direction_id": 0, "service_id": "weekday"},
            {"trip_id": "b", "route_id": "r1", "direction_id": 1, "service_id": "weekday"},
            {"trip_id": "c", "route_id": "r2", "direction_id": 0, "service_id": "weekday"},
        ]
    )

    assert service_ids_for_day(calendar, "monday") == {"weekday", "removed"}
    assert service_ids_for_date(calendar, calendar_dates, date(2025, 7, 7)) == {
        "weekday",
        "special",
    }
    assert trips_for_route(trips, "r1", direction_id=0, service_ids={"weekday"}) == ["a"]
    assert trips_for_route(trips, "r1", direction_id=None, service_ids={"weekday"}) == [
        "a",
        "b",
    ]


def test_shape_projection_haversine_fallback_duplicate_filter_and_segment_times():
    rows = _rows()
    stops = _stops()
    shape = _shape_points_with_cumdist(
        [(0.0, 0.0, 1), (0.001, 0.0, 2), (0.001, 0.001, 3), (0.0, 0.001, 4)]
    )

    shaped = list(iter_valid_stop_pairs(rows, stops, shape_points=shape))
    fallback = list(iter_valid_stop_pairs(rows, stops))
    segments = build_segments(rows, stops)

    assert shaped[0][2] > haversine_m(0.0, 0.0, 0.0, 0.001)
    assert fallback[0][2] == pytest.approx(haversine_m(0.0, 0.0, 0.0, 0.001))
    assert segments[0].run_time_s == 300
    assert segments[0].dwell_s == 30

    colocated = pd.DataFrame(
        [
            {"stop_id": "a", "stop_lat": 0.0, "stop_lon": 0.0},
            {"stop_id": "b", "stop_lat": 0.0, "stop_lon": 0.0},
        ]
    )
    assert list(iter_valid_stop_pairs(rows[:2], colocated)) == []


def _routes_sharing_a_short_name():
    return pd.DataFrame(
        [
            {"route_id": "3 W4 a", "route_short_name": "W4", "route_long_name": "Dublin"},
            {"route_id": "WFRD W4 c b", "route_short_name": "W4", "route_long_name": "Waterford"},
            {"route_id": "2 208 c a", "route_short_name": "208", "route_long_name": "Cork"},
        ]
    )


def test_exact_route_id_selects_one_of_several_routes_sharing_a_short_name():
    routes = _routes_sharing_a_short_name()

    assert resolve_route_id(routes, "WFRD W4 c b") == ("WFRD W4 c b", "Waterford")
    assert resolve_route_id(routes, "208") == ("2 208 c a", "Cork")


def test_ambiguous_short_name_is_rejected_and_lists_the_route_ids():
    with pytest.raises(ValueError, match="ambiguous") as error:
        resolve_route_id(_routes_sharing_a_short_name(), "W4")

    assert "3 W4 a" in str(error.value) and "WFRD W4 c b" in str(error.value)


def test_unknown_route_selector_is_rejected():
    with pytest.raises(ValueError, match="No route"):
        resolve_route_id(_routes_sharing_a_short_name(), "999")


def _shape_with_a_phantom_loop():
    return [
        (0.0, 0.0, 1),
        (0.0, 0.001, 2),
        (0.0, 0.002, 3),
        (0.0, 0.003, 4),
        (0.0, 0.002, 5),
        (0.0, 0.001, 6),
        (0.0, 0.002, 7),
        (0.0, 0.003, 8),
        (0.0, 0.004, 9),
    ]


def test_shape_repair_removes_a_phantom_loop_between_coincident_points():
    rows = _shape_with_a_phantom_loop()

    repaired = _repaired_shape_rows(rows, [[2, 6]])

    assert [row[2] for row in repaired] == [1, 2, 6, 7, 8, 9]
    original_m = _shape_points_with_cumdist(rows)[-1][2]
    repaired_m = _shape_points_with_cumdist(repaired)[-1][2]
    assert repaired_m == pytest.approx(haversine_m(0.0, 0.0, 0.0, 0.004))
    assert original_m - repaired_m == pytest.approx(4 * haversine_m(0.0, 0.0, 0.0, 0.001))


def test_shape_without_a_repair_is_returned_unchanged():
    rows = _shape_with_a_phantom_loop()

    assert _repaired_shape_rows(rows, None) is rows
    assert _repaired_shape_rows(rows, []) is rows


@pytest.mark.parametrize("repair", [[2, 7], [2, 99]], ids=["points_differ", "missing_point"])
def test_shape_repair_fails_when_its_points_no_longer_join(repair):
    with pytest.raises(ValueError, match="no longer joins"):
        _repaired_shape_rows(_shape_with_a_phantom_loop(), [repair])


def test_implausible_intervals_flags_only_impossible_schedules():
    segments = [
        Segment(length_m=500.0, run_time_s=60.0),
        Segment(length_m=23_650.0, run_time_s=240.0),
        Segment(length_m=860.0, run_time_s=30.0),
        Segment(length_m=500.0, run_time_s=0.0),
        Segment(length_m=500.0, run_time_s=None),
    ]

    assert implausible_intervals(segments) == [1]
