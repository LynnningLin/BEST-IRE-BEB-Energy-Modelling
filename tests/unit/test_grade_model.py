import pandas as pd
import pytest
from math import degrees

from beb_soc_model import Segment
from gtfs_to_segment import add_grades_from_dem


class ElevationStub:
    def __init__(self, by_lat_lon):
        self.by_lat_lon = by_lat_lon

    def get_elevation(self, latitude, longitude):
        return self.by_lat_lon.get((latitude, longitude))


def _trip_rows(stop_ids):
    rows = []
    for i, stop_id in enumerate(stop_ids):
        minute = i * 5
        rows.append(
            {
                "trip_id": "t1",
                "stop_id": stop_id,
                "arrival_time": f"08:{minute:02d}:00",
                "departure_time": f"08:{minute:02d}:00",
                "stop_sequence": str(i + 1),
            }
        )
    return rows


def _stops_for_500m_pairs():
    step_deg = degrees(500.0 / 6_371_000.0)
    return pd.DataFrame(
        [
            {"stop_id": "a", "stop_lat": 0.0, "stop_lon": 0.0},
            {"stop_id": "b", "stop_lat": step_deg, "stop_lon": 0.0},
            {"stop_id": "c", "stop_lat": 2 * step_deg, "stop_lon": 0.0},
        ]
    )


@pytest.mark.parametrize(
    ("start_elev", "end_elev", "expected"),
    [(100.0, 100.0, 0.0), (100.0, 110.0, 0.02), (110.0, 100.0, -0.02)],
)
def test_grade_from_synthetic_elevations(start_elev, end_elev, expected):
    stops = _stops_for_500m_pairs().iloc[:2].copy()
    elevations = ElevationStub(
        {
            (stops.iloc[0].stop_lat, stops.iloc[0].stop_lon): start_elev,
            (stops.iloc[1].stop_lat, stops.iloc[1].stop_lon): end_elev,
        }
    )
    seg = Segment(length_m=500.0)

    add_grades_from_dem(
        [seg], _trip_rows(["a", "b"]), stops, elevation_data=elevations, verbose=False
    )

    assert seg.grade == pytest.approx(expected, abs=2e-5)


def test_missing_elevation_falls_back_to_flat_grade():
    stops = _stops_for_500m_pairs().iloc[:2].copy()
    seg = Segment(length_m=500.0, grade=0.5)

    add_grades_from_dem(
        [seg],
        _trip_rows(["a", "b"]),
        stops,
        elevation_data=ElevationStub({}),
        verbose=False,
    )

    assert seg.grade == 0.0


def test_extreme_grade_is_clamped_to_configured_limit():
    stops = _stops_for_500m_pairs().iloc[:2].copy()
    elevations = ElevationStub(
        {
            (stops.iloc[0].stop_lat, stops.iloc[0].stop_lon): 0.0,
            (stops.iloc[1].stop_lat, stops.iloc[1].stop_lon): 200.0,
        }
    )
    seg = Segment(length_m=500.0)

    add_grades_from_dem(
        [seg],
        _trip_rows(["a", "b"]),
        stops,
        elevation_data=elevations,
        max_abs_grade=0.15,
        verbose=False,
    )

    assert seg.grade == pytest.approx(0.15)


def test_grade_alignment_survives_removed_colocated_stop():
    stops = pd.DataFrame(
        [
            {"stop_id": "a", "stop_lat": 0.0, "stop_lon": 0.0},
            {"stop_id": "dup", "stop_lat": 0.0, "stop_lon": 0.0},
            {"stop_id": "b", "stop_lat": degrees(500.0 / 6_371_000.0), "stop_lon": 0.0},
            {"stop_id": "c", "stop_lat": degrees(1000.0 / 6_371_000.0), "stop_lon": 0.0},
        ]
    )
    elevations = ElevationStub(
        {
            (row.stop_lat, row.stop_lon): elev
            for row, elev in zip(stops.itertuples(index=False), [100.0, 100.0, 110.0, 90.0])
        }
    )
    segments = [Segment(length_m=500.0), Segment(length_m=500.0)]

    add_grades_from_dem(
        segments,
        _trip_rows(["a", "dup", "b", "c"]),
        stops,
        elevation_data=elevations,
        verbose=False,
    )

    assert [s.grade for s in segments] == pytest.approx([0.02, -0.04], abs=2e-5)
