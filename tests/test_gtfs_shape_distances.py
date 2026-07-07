import sys
import types
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

sys.modules.setdefault(
    "srtm", types.SimpleNamespace(get_data=lambda *args, **kwargs: None)
)

from gtfs_to_segment import (  # noqa: E402
    _shape_points_with_cumdist,
    haversine_m,
    iter_valid_stop_pairs,
)


def _trip_rows():
    return [
        {
            "trip_id": "t1",
            "stop_id": "a",
            "arrival_time": "08:00:00",
            "departure_time": "08:00:00",
            "stop_sequence": "1",
        },
        {
            "trip_id": "t1",
            "stop_id": "b",
            "arrival_time": "08:05:00",
            "departure_time": "08:05:30",
            "stop_sequence": "2",
        },
    ]


def _stops():
    return pd.DataFrame(
        [
            {"stop_id": "a", "stop_lat": 0.0, "stop_lon": 0.0},
            {"stop_id": "b", "stop_lat": 0.001, "stop_lon": 0.001},
        ]
    )


def test_stop_pair_length_uses_shape_polyline_distance():
    shape_points = _shape_points_with_cumdist(
        [
            (0.0, 0.0, 1),
            (0.0, 0.001, 2),
            (0.001, 0.001, 3),
        ]
    )

    pairs = list(iter_valid_stop_pairs(_trip_rows(), _stops(), shape_points))

    expected = (
        haversine_m(0.0, 0.0, 0.0, 0.001)
        + haversine_m(0.0, 0.001, 0.001, 0.001)
    )
    straight_line = haversine_m(0.0, 0.0, 0.001, 0.001)
    assert len(pairs) == 1
    assert pairs[0][2] == expected
    assert pairs[0][2] > straight_line


def test_stop_pair_length_falls_back_to_haversine_without_shape():
    pairs = list(iter_valid_stop_pairs(_trip_rows(), _stops()))

    assert len(pairs) == 1
    assert pairs[0][2] == haversine_m(0.0, 0.0, 0.001, 0.001)
