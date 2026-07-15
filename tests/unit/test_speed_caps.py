import pandas as pd
import pytest

from beb_soc_model import Segment
from speed_caps import (
    MIN_CAP_KMH,
    _EdgeIndex,
    _harmonic_cap_for_polyline,
    _parse_maxspeed_kmh,
    add_speed_caps,
    load_speed_cap_cache,
    resolve_speed_caps,
    save_speed_cap_cache,
)


def test_parse_maxspeed_values():
    assert _parse_maxspeed_kmh("50") == 50.0
    assert _parse_maxspeed_kmh("30 mph") == pytest.approx(48.28032)
    assert _parse_maxspeed_kmh("IE:urban") == 50.0
    assert _parse_maxspeed_kmh(["80", "50"]) == 50.0
    assert _parse_maxspeed_kmh("signals") is None
    assert _parse_maxspeed_kmh(0) is None
    assert _parse_maxspeed_kmh(-10) is None


def _mixed_limit_geometry():
    lat0, lon0 = 51.0, -8.0
    dlat = 1.0 / 111_320.0
    poly = [(lat0, lon0), (lat0 + 1000 * dlat, lon0)]
    edges = [
        ([(lat0, lon0), (lat0 + 800 * dlat, lon0)], 50.0, True),
        ([(lat0 + 800 * dlat, lon0), (lat0 + 1000 * dlat, lon0)], 30.0, True),
    ]
    return poly, edges


def test_harmonic_speed_cap_for_known_mixed_limit_route():
    poly, edges = _mixed_limit_geometry()
    cap_kmh, coverage, tagged = _harmonic_cap_for_polyline(
        poly, _EdgeIndex(edges), snap_radius_m=25.0, sample_step_m=20.0
    )
    expected = 1000.0 / (800.0 / 50.0 + 200.0 / 30.0)

    assert cap_kmh == pytest.approx(expected, abs=0.15)
    assert coverage == pytest.approx(1.0)
    assert tagged == pytest.approx(1.0)


def test_resolver_sources_fallbacks_and_clamps():
    poly, tagged_edges = _mixed_limit_geometry()
    lat0, lon0 = poly[0]
    dlat = 1.0 / 111_320.0
    coords = {
        "a": poly[0],
        "b": poly[-1],
        "c": (lat0 + 2000 * dlat, lon0),
        "x": (52.0, -8.0),
        "y": (52.01, -8.0),
    }
    geom_map = {
        ("a", "b"): poly,
        ("b", "c"): [(lat0 + 1000 * dlat, lon0), (lat0 + 2000 * dlat, lon0)],
        ("x", "y"): [coords["x"], coords["y"]],
    }
    imputed_edges = tagged_edges + [
        (geom_map[("b", "c")], 45.0, False),
    ]

    caps, sources, stats = resolve_speed_caps(
        [("a", "b", 1000.0), ("b", "c", 1000.0), ("x", "y", 1000.0)],
        coords,
        geom_map=geom_map,
        cache={},
        fetch_fn=lambda *args, **kwargs: imputed_edges,
        verbose=False,
    )

    assert sources[("a", "b")] == "osm"
    assert sources[("b", "c")] == "imputed"
    assert sources[("x", "y")] == "fallback_low_coverage"
    assert caps[("x", "y")] == pytest.approx(50.0 / 3.6)
    assert stats["osm"] == 1 and stats["imputed"] == 1 and stats["fallback_low_coverage"] == 1

    fallback_caps, fallback_sources, _ = resolve_speed_caps(
        [("a", "b", 1000.0)],
        coords,
        cache={},
        fetch_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
        verbose=False,
    )
    assert fallback_sources[("a", "b")] == "fallback"
    assert fallback_caps[("a", "b")] == pytest.approx(50.0 / 3.6)

    low_caps, _, _ = resolve_speed_caps(
        [("a", "b", 1000.0)],
        coords,
        geom_map={("a", "b"): poly},
        cache={},
        fetch_fn=lambda *args, **kwargs: [(poly, 3.0, True)],
        verbose=False,
    )
    high_caps, _, _ = resolve_speed_caps(
        [("a", "b", 1000.0)],
        coords,
        geom_map={("a", "b"): poly},
        cache={},
        max_cap_kmh=90.0,
        fetch_fn=lambda *args, **kwargs: [(poly, 160.0, True)],
        verbose=False,
    )
    assert low_caps[("a", "b")] == pytest.approx(MIN_CAP_KMH / 3.6)
    assert high_caps[("a", "b")] == pytest.approx(90.0 / 3.6)


def test_cache_round_trip_and_per_segment_assignment(tmp_path):
    cache = {
        ("a", "b"): {
            "speed_cap_ms": 13.8889,
            "speed_cap_kmh": 50.0,
            "source": "osm",
            "coverage_frac": 1.0,
            "snap_radius_m": 25.0,
            "sample_step_m": 20.0,
            "length_m": 500.0,
            "fetched_utc": "2026-01-01T00:00:00Z",
        }
    }
    path = tmp_path / "caps.csv"
    save_speed_cap_cache(path, cache)

    loaded = load_speed_cap_cache(path)

    assert loaded[("a", "b")]["speed_cap_ms"] == pytest.approx(13.889, abs=1e-3)

    stops = pd.DataFrame(
        [
            {"stop_id": "a", "stop_lat": 51.0, "stop_lon": -8.0},
            {"stop_id": "b", "stop_lat": 51.0045, "stop_lon": -8.0},
        ]
    )
    rows = [
        {"stop_id": "a", "arrival_time": "08:00:00", "departure_time": "08:00:00"},
        {"stop_id": "b", "arrival_time": "08:05:00", "departure_time": "08:05:00"},
    ]
    segments = add_speed_caps(
        [Segment(length_m=500.0)],
        rows,
        stops,
        cap_map={("a", "b"): 13.8889},
        source_map={("a", "b"): "osm"},
        verbose=False,
    )

    assert segments[0].speed_cap_ms == pytest.approx(13.8889)
    assert segments[0].speed_cap_source == "osm"
