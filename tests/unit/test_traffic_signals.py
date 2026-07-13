import pytest

from beb_soc_model import Segment, _signal_stop_count, _signal_stop_prob_for_segment
from traffic_signals import (
    _clustered_count,
    _pair_signal_count,
    _pair_signal_positions,
    _polyline_signal_count,
    _polyline_signal_positions,
    add_traffic_signals,
    resolve_signal_counts,
)


COORDS = {"a": (51.0, -8.0), "b": (51.0045, -8.0), "c": (51.0090, -8.0)}


def _rows():
    return [
        {"stop_id": "a", "arrival_time": "08:00:00", "departure_time": "08:00:00"},
        {"stop_id": "b", "arrival_time": "08:05:00", "departure_time": "08:05:00"},
    ]


def _stops():
    import pandas as pd

    return pd.DataFrame(
        [
            {"stop_id": "a", "stop_lat": COORDS["a"][0], "stop_lon": COORDS["a"][1]},
            {"stop_id": "b", "stop_lat": COORDS["b"][0], "stop_lon": COORDS["b"][1]},
        ]
    )


def test_signal_pair_geometry_strict_relaxed_and_endpoints():
    a, b = COORDS["a"], COORDS["b"]

    assert _pair_signal_count(a, b, [], 30.0) == 0
    assert _pair_signal_count(a, b, [(51.00225, -8.0)], 30.0) == 1
    assert _pair_signal_count(a, b, [(51.00225, -8.0010)], 30.0) == 0
    assert _pair_signal_count(a, b, [(51.00225, -8.0010)], 80.0) == 1
    assert _pair_signal_positions(a, b, [a], 30.0) == [0.0]
    assert _pair_signal_positions(a, b, [b], 30.0) == []


def test_shared_stop_signal_is_not_double_counted_across_adjacent_pairs():
    signal_at_b = [COORDS["b"]]

    upstream = _pair_signal_count(COORDS["a"], COORDS["b"], signal_at_b, 30.0)
    downstream = _pair_signal_count(COORDS["b"], COORDS["c"], signal_at_b, 30.0)

    assert upstream == 0
    assert downstream == 1


def test_nearby_signal_nodes_are_clustered():
    assert _clustered_count([100.0, 105.0, 160.0], cluster_radius_m=20.0) == 2


def test_signals_on_curved_polyline_are_counted():
    poly = [(51.0, -8.0), (51.0, -7.995), (51.0045, -7.995)]
    signal = [(51.0, -7.997)]

    positions = _polyline_signal_positions(poly, signal, snap_radius_m=30.0)

    assert len(positions) == 1
    assert _polyline_signal_count(poly, signal, snap_radius_m=30.0) == 1


def test_cache_lookup_fallback_density_and_add_assignment():
    cache = {
        ("a", "b"): {
            "n_signals": 2,
            "source": "osm",
            "snap_radius_m": 30.0,
            "relaxed_snap_radius_m": 60.0,
            "cluster_radius_m": 35.0,
            "length_m": 500.0,
        }
    }

    counts, sources, stats = resolve_signal_counts(
        [("a", "b", 500.0)],
        COORDS,
        cache=cache,
        snap_radius_m=30.0,
        relaxed_snap_radius_m=60.0,
        cluster_radius_m=35.0,
        fetch_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fetch")),
        verbose=False,
    )

    assert stats["cache_hit"] == 1
    assert counts[("a", "b")] == 2
    assert sources[("a", "b")] == "osm"

    segments = add_traffic_signals(
        [Segment(length_m=500.0)],
        _rows(),
        _stops(),
        count_map=counts,
        source_map=sources,
        verbose=False,
    )
    assert segments[0].n_signals == 2
    assert segments[0].signal_source == "osm"


def test_fallback_signal_density_when_fetch_fails():
    counts, sources, stats = resolve_signal_counts(
        [("a", "b", 1000.0)],
        COORDS,
        cache={},
        fallback_per_km=2.0,
        fetch_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
        verbose=False,
    )

    assert counts[("a", "b")] == 2
    assert sources[("a", "b")] == "fallback"
    assert stats["fallback"] == 1


def test_deterministic_stop_count_and_probability_lookup():
    seg = Segment(length_m=500.0, from_stop_departure_time="25:15:00")

    assert _signal_stop_count(seg, n_signals=3, stop_prob=0.0) == 0
    assert _signal_stop_count(seg, n_signals=3, stop_prob=1.0) == 3
    assert _signal_stop_count(seg, n_signals=5, stop_prob=0.4) == _signal_stop_count(
        seg, n_signals=5, stop_prob=0.4
    )

    prob, hour, source = _signal_stop_prob_for_segment(
        seg, default_prob=0.2, stop_prob_by_hour={1: 0.7}
    )
    fallback, _, fallback_source = _signal_stop_prob_for_segment(
        seg, default_prob=0.2, stop_prob_by_hour={8: 0.7}
    )

    assert (prob, hour, source) == (0.7, 1, "hourly")
    assert (fallback, fallback_source) == (0.2, "constant_fallback")
