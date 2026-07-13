import pandas as pd
import pytest

from gtfs_to_segment import build_segments, haversine_m


def test_build_segments_sets_runtime_dwell_passengers_and_cruise_speed():
    stops = pd.DataFrame(
        [
            {"stop_id": "a", "stop_lat": 0.0, "stop_lon": 0.0},
            {"stop_id": "b", "stop_lat": 0.0, "stop_lon": 0.001},
        ]
    )
    rows = [
        {
            "stop_id": "a",
            "arrival_time": "08:00:00",
            "departure_time": "08:00:00",
        },
        {
            "stop_id": "b",
            "arrival_time": "08:02:00",
            "departure_time": "08:02:30",
        },
    ]

    [seg] = build_segments(rows, stops, passengers=33, cruise_factor=1.2)

    length = haversine_m(0.0, 0.0, 0.0, 0.001)
    assert seg.length_m == pytest.approx(length)
    assert seg.run_time_s == 120
    assert seg.dwell_s == 30
    assert seg.passengers == 33
    assert seg.v_cruise_ms == pytest.approx(max((length / 120) * 1.2, 3.0))


def test_build_segments_uses_fallback_cruise_for_non_positive_runtime_and_clamps_dwell():
    stops = pd.DataFrame(
        [
            {"stop_id": "a", "stop_lat": 0.0, "stop_lon": 0.0},
            {"stop_id": "b", "stop_lat": 0.0, "stop_lon": 0.001},
        ]
    )
    rows = [
        {"stop_id": "a", "arrival_time": "08:00:00", "departure_time": "08:05:00"},
        {"stop_id": "b", "arrival_time": "08:04:00", "departure_time": "08:03:00"},
    ]

    [seg] = build_segments(rows, stops)

    assert seg.run_time_s == -60
    assert seg.v_cruise_ms == 11.0
    assert seg.dwell_s == 0.0
