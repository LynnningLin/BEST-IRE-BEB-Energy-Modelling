import pytest
import requests

import traffic_signals
from beb_soc_model import Segment, _signal_stop_count, _signal_stop_prob_for_segment
from traffic_signals import (
    OSM_SIGNAL_TAG_QUERIES,
    SIGNAL_EVENT_CACHE_COLUMNS,
    _clustered_count,
    _pair_signal_count,
    _pair_signal_positions,
    _polyline_signal_count,
    _polyline_signal_positions,
    fetch_traffic_signals_bbox,
    load_signal_cache,
    load_signal_event_cache,
    add_traffic_signals,
    resolve_signal_counts,
    route_stop_pairs,
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


def test_resolve_persists_clustered_signal_events_and_derives_counts(tmp_path):
    summary_path = tmp_path / "traffic_signals.csv"
    events_path = tmp_path / "traffic_signal_events.csv"
    pair = ("shape:s1|a|b", "a", "b", 500.0, "s1")
    poly = [COORDS["a"], COORDS["b"]]
    signals = [
        {"lat": 51.0010, "lon": -8.0, "osm_id": "n1", "signal_type": "vehicle"},
        {"lat": 51.0011, "lon": -8.0, "osm_id": "n2", "signal_type": "vehicle"},
        {"lat": 51.0035, "lon": -8.0, "osm_id": "n3", "signal_type": "crossing"},
    ]

    counts, sources, stats = resolve_signal_counts(
        [pair],
        COORDS,
        cache={},
        cache_path=summary_path,
        event_cache={},
        event_cache_path=events_path,
        geom_map={pair[0]: poly},
        snap_radius_m=30.0,
        relaxed_snap_radius_m=60.0,
        cluster_radius_m=35.0,
        fetch_fn=lambda *args, **kwargs: signals,
        verbose=False,
    )

    events = load_signal_event_cache(events_path)
    rows = events[pair[0]]
    assert counts[pair[0]] == 2
    assert sources[pair[0]] == "osm"
    assert stats["event_rows"] == 2
    assert len(rows) == counts[pair[0]]
    assert [r["distance_along_segment_m"] for r in rows] == sorted(
        r["distance_along_segment_m"] for r in rows
    )
    assert all(0.0 <= r["distance_along_segment_m"] < r["segment_length_m"] for r in rows)
    assert rows[0]["cluster_size"] == 2
    assert rows[0]["raw_osm_ids"] == "n1|n2"
    assert rows[0]["source"] == "osm"
    assert set(SIGNAL_EVENT_CACHE_COLUMNS).issuperset(rows[0].keys())

    again, _, _ = resolve_signal_counts(
        [pair],
        COORDS,
        cache=None,
        cache_path=summary_path,
        event_cache=None,
        event_cache_path=events_path,
        geom_map={pair[0]: poly},
        fetch_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fetch")),
        verbose=False,
    )
    again_rows = load_signal_event_cache(events_path)[pair[0]]
    assert again[pair[0]] == 2
    assert [r["signal_event_id"] for r in again_rows] == [
        r["signal_event_id"] for r in rows
    ]


def test_relaxed_events_remain_distinguishable(tmp_path):
    pair = ("shape:s1|a|b", "a", "b", 500.0, "s1")
    poly = [COORDS["a"], COORDS["b"]]
    signals = [{"lat": 51.00225, "lon": -8.00065, "osm_id": "wide"}]

    counts, sources, _ = resolve_signal_counts(
        [pair],
        COORDS,
        cache={},
        event_cache={},
        event_cache_path=tmp_path / "events.csv",
        geom_map={pair[0]: poly},
        snap_radius_m=30.0,
        relaxed_snap_radius_m=80.0,
        cluster_radius_m=35.0,
        fetch_fn=lambda *args, **kwargs: signals,
        verbose=False,
    )

    rows = load_signal_event_cache(tmp_path / "events.csv")[pair[0]]
    assert counts[pair[0]] == 1
    assert sources[pair[0]] == "osm_relaxed"
    assert rows[0]["source"] == "osm_relaxed"
    assert rows[0]["snap_distance_m"] > 30.0


def test_exact_boundary_event_is_downstream_only_and_diagnostic_is_clear():
    pairs = [
        ("shape:s1|a|b", "a", "b", 500.0, "s1"),
        ("shape:s1|b|c", "b", "c", 500.0, "s1"),
    ]
    geom = {pairs[0][0]: [COORDS["a"], COORDS["b"]],
            pairs[1][0]: [COORDS["b"], COORDS["c"]]}

    counts, _sources, stats = resolve_signal_counts(
        pairs,
        COORDS,
        cache={},
        event_cache={},
        geom_map=geom,
        fetch_fn=lambda *args, **kwargs: [{"lat": COORDS["b"][0], "lon": COORDS["b"][1]}],
        verbose=False,
    )

    assert counts[pairs[0][0]] == 0
    assert counts[pairs[1][0]] == 1
    assert stats["adjacent_raw_signal_duplicates"] == 0


def test_same_stop_pair_shape_variants_do_not_share_events():
    import pandas as pd

    stops = pd.DataFrame([
        {"stop_id": "a", "stop_lat": 51.0, "stop_lon": -8.0},
        {"stop_id": "b", "stop_lat": 51.0045, "stop_lon": -8.0},
    ])
    rows1 = [{"stop_id": "a"}, {"stop_id": "b"}]
    rows2 = [{"stop_id": "a"}, {"stop_id": "b"}]
    shape1 = [(51.0, -8.0, 0.0), (51.0045, -8.0, 500.0)]
    shape2 = [(51.0, -8.0, 0.0), (51.0, -7.995, 340.0), (51.0045, -7.995, 840.0)]

    pairs, geom = route_stop_pairs(
        [("t1", rows1), ("t2", rows2)],
        stops,
        {"t1": shape1, "t2": shape2},
        shape_id_by_trip={"t1": "shape_a", "t2": "shape_b"},
        include_geometry_id=True,
    )

    assert len(pairs) == 2
    assert len({p[0] for p in pairs}) == 2
    assert set(geom) == {p[0] for p in pairs}


def test_add_traffic_signals_attaches_events_for_actual_geometry():
    import pandas as pd

    stops = pd.DataFrame([
        {"stop_id": "a", "stop_lat": 51.0, "stop_lon": -8.0},
        {"stop_id": "b", "stop_lat": 51.0045, "stop_lon": -8.0},
    ])
    rows = [{"stop_id": "a"}, {"stop_id": "b"}]
    shape = [(51.0, -8.0, 0.0), (51.0045, -8.0, 500.0)]
    geom_key = "shape:shape1|a|b"
    events = [
        {"signal_event_id": "s2", "distance_along_segment_m": 170.0},
        {"signal_event_id": "s1", "distance_along_segment_m": 100.0},
    ]

    segments = add_traffic_signals(
        [Segment(length_m=500.0)],
        rows,
        stops,
        count_map={geom_key: 2},
        source_map={geom_key: "osm"},
        event_map={geom_key: events},
        shape_points=shape,
        shape_id="shape1",
        verbose=False,
    )

    seg = segments[0]
    assert seg.segment_geometry_id == geom_key
    assert seg.n_signals == 2
    assert [e["signal_event_id"] for e in seg.signal_events] == ["s1", "s2"]


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


class _Response:
    def __init__(self, payload=None, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self.payload is None:
            raise requests.exceptions.JSONDecodeError("not json", "<html>", 0)
        return self.payload


def _overpass(monkeypatch, *responses):
    """Serve canned Overpass responses in order, recording each query sent."""
    queries = []
    pending = iter(responses)

    def post(url, data, headers, timeout):
        queries.append(data["data"])
        return next(pending)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(traffic_signals.time, "sleep", lambda seconds: None)
    return queries


def _node(osm_id, lat, tags):
    return {"type": "node", "id": osm_id, "lat": lat, "lon": -8.0, "tags": tags}


def test_signal_fetch_sends_every_tag_in_one_pinned_query(monkeypatch):
    queries = _overpass(monkeypatch, _Response({"elements": []}))

    assert fetch_traffic_signals_bbox(52, 51, -7, -8, snapshot="2026-08-26T16:04:56Z") == []
    assert len(queries) == 1
    assert '[date:"2026-08-26T16:04:56Z"]' in queries[0]
    for _label, tags in OSM_SIGNAL_TAG_QUERIES:
        for key, value in tags.items():
            assert f'["{key}"="{value}"]' in queries[0]


def test_signal_fetch_labels_nodes_and_way_centroids_and_drops_duplicates(monkeypatch):
    way = {
        "type": "way",
        "id": 9,
        "tags": {"crossing:signals": "yes"},
        "geometry": [{"lat": 51.2, "lon": -8.0}, {"lat": 51.3, "lon": -8.0}],
    }
    elements = [
        _node(2, 51.0, {"crossing_ref": "pelican", "crossing": "traffic_signals"}),
        _node(1, 51.1, {"highway": "traffic_signals"}),
        _node(3, 51.1000001, {"crossing": "traffic_signals"}),
        way,
    ]
    _overpass(monkeypatch, _Response({"elements": elements}))

    points = {p["osm_id"]: p for p in fetch_traffic_signals_bbox(52, 51, -7, -9)}

    assert set(points) == {"node:1", "node:2", "way:9"}
    assert points["node:1"]["signal_type"] == "vehicle_signals"
    assert points["node:2"]["signal_type"] == "ped_crossing_signals"
    assert points["way:9"]["lat"] == pytest.approx(51.25)


@pytest.mark.parametrize(
    "failed",
    [
        _Response({"elements": [_node(1, 51.0, {"highway": "traffic_signals"})],
                   "remark": "runtime error: Query timed out"}),
        _Response({"elements": [], "remark": "runtime error: out of memory"}),
        _Response(None),
        _Response({"elements": []}, status=504),
        _Response({"version": 0.6}),
    ],
    ids=["partial_remark", "empty_remark", "html_200", "http_504", "no_elements"],
)
def test_signal_fetch_treats_incomplete_responses_as_failures(monkeypatch, failed):
    queries = _overpass(monkeypatch, *[failed] * traffic_signals.OVERPASS_ATTEMPTS)

    with pytest.raises(RuntimeError, match="OSM signal query failed"):
        fetch_traffic_signals_bbox(52, 51, -7, -8)
    assert len(queries) == traffic_signals.OVERPASS_ATTEMPTS


def test_signal_fetch_retries_a_failed_response(monkeypatch):
    elements = [_node(1, 51.0, {"highway": "traffic_signals"})]
    _overpass(monkeypatch, _Response({}, status=429), _Response({"elements": elements}))

    assert len(fetch_traffic_signals_bbox(52, 51, -7, -8)) == 1


def test_invalid_osm_snapshot_is_rejected():
    with pytest.raises(ValueError, match="OSM snapshot"):
        fetch_traffic_signals_bbox(52, 51, -7, -8, snapshot='2026"];node;out;')


def _fallback_row():
    return {
        "n_signals": 7,
        "source": "fallback",
        "snap_radius_m": 30.0,
        "relaxed_snap_radius_m": 60.0,
        "cluster_radius_m": 35.0,
        "length_m": 500.0,
    }


def test_cached_fallback_row_is_retried_and_replaced():
    cache = {("a", "b"): _fallback_row()}

    counts, sources, stats = resolve_signal_counts(
        [("a", "b", 500.0)], COORDS, cache=cache, event_cache={},
        fetch_fn=lambda *args, **kwargs: [(51.00225, -8.0)], verbose=False,
    )

    assert stats["cache_hit"] == 0
    assert (counts[("a", "b")], sources[("a", "b")]) == (1, "osm")
    assert cache[("a", "b")]["source"] == "osm"


def test_failed_fetch_is_not_saved_as_cached_data(tmp_path):
    path = tmp_path / "signals.csv"

    _, sources, _ = resolve_signal_counts(
        [("a", "b", 1000.0)], COORDS, cache={}, event_cache={}, cache_path=path,
        event_cache_path=tmp_path / "events.csv",
        fetch_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
        verbose=False,
    )

    assert sources[("a", "b")] == "fallback"
    assert load_signal_cache(path) == {}


# Two consecutive intervals on a straight north-going road, a -> b -> c.
UPSTREAM = [COORDS["a"], COORDS["b"]]
DOWNSTREAM = [COORDS["b"], COORDS["c"]]
METRES_NORTH = 1.0 / 111_195.0


def test_signal_just_before_a_stop_is_counted_once_upstream():
    signal = [(COORDS["b"][0] - 20 * METRES_NORTH, COORDS["b"][1])]

    assert len(_polyline_signal_positions(UPSTREAM, signal, 30.0)) == 1
    assert _polyline_signal_positions(DOWNSTREAM, signal, 30.0) == []


def test_signal_just_after_a_stop_is_counted_once_downstream():
    signal = [(COORDS["b"][0] + 20 * METRES_NORTH, COORDS["b"][1])]
    positions = _polyline_signal_positions(DOWNSTREAM, signal, 30.0)

    assert _polyline_signal_positions(UPSTREAM, signal, 30.0) == []
    assert positions == pytest.approx([20.0], abs=0.1)


def test_signal_exactly_at_a_stop_stays_downstream():
    signal = [COORDS["b"]]

    assert _polyline_signal_positions(UPSTREAM, signal, 30.0) == []
    assert _polyline_signal_positions(DOWNSTREAM, signal, 30.0) == [0.0]


def test_signal_behind_the_first_stop_is_not_on_the_trip():
    signal = [(COORDS["a"][0] - 20 * METRES_NORTH, COORDS["a"][1])]

    assert _polyline_signal_positions(UPSTREAM, signal, 30.0) == []


def test_polyline_matches_chord_rule_for_a_signal_behind_the_start():
    signal = [(COORDS["a"][0] - 20 * METRES_NORTH, COORDS["a"][1])]

    chord = _pair_signal_positions(COORDS["a"], COORDS["b"], signal, 30.0)
    polyline = _polyline_signal_positions(UPSTREAM, signal, 30.0)

    assert chord == polyline == []


def test_event_clusters_exclude_a_signal_behind_the_start():
    signal = [(COORDS["b"][0] - 20 * METRES_NORTH, COORDS["b"][1])]

    events = traffic_signals._cluster_signal_events(
        "g", "b", "c", "s", DOWNSTREAM, signal, "osm", 30.0, 60.0, 35.0, "now"
    )

    assert events == []
