import json
import math

import pytest

import signal_inventory as si
from signal_placement import PlacementUnavailable
from traffic_signals import load_signal_cache, resolve_signal_counts


def node(i, x, y, **tags):
    # About 1.1 m per 1e-5 degrees, so x and y read as local metres.
    return {"type": "node", "id": i, "lat": 52.0 + y * 1e-5 / 1.11,
            "lon": -8.0 + x * 1e-5 / (1.11 * math.cos(math.radians(52.0))), "tags": tags}


def way(i, nodes, **tags):
    return {"type": "way", "id": i, "nodes": nodes,
            "tags": {"highway": "secondary", **tags}}


def shape_along(graph, nodes):
    points, cum, last = [], 0.0, None
    for n in nodes:
        lat, lon = graph.node_ll[n]
        if last is not None:
            cum += math.dist(graph.node_xy[last], graph.node_xy[n])
        points.append((lat, lon, cum))
        last = n
    return points


def traverse(elements, nodes, crossing_ways=()):
    graph = si.RoadGraph(elements)
    shape = shape_along(graph, nodes)
    match = si.match_shape(graph, shape, 0.0, shape[-1][2])
    return graph, match, si.path_controls(graph, match, crossing_ways)


def straight_road(*signals):
    """A two-way road along x, node ids 1.. every 20 m, with signal tags by node id."""
    tagged = dict(signals)
    elements = [node(i, (i - 1) * 20.0, 0.0, **tagged.get(i, {})) for i in range(1, 22)]
    return elements + [way(100, list(range(1, 22)))]


CROSSING_TAGS = {"highway": "crossing", "crossing": "traffic_signals"}
FORWARD = {"highway": "traffic_signals", "traffic_signals:direction": "forward"}
BACKWARD = {"highway": "traffic_signals", "traffic_signals:direction": "backward"}


def test_match_follows_the_road_through_a_turn():
    elements = [node(1, 0, 0), node(2, 100, 0), node(3, 200, 0), node(4, 100, 100), node(5, 100, 200)]
    elements += [way(10, [1, 2, 3]), way(11, [2, 4, 5])]
    graph, match, _ = traverse(elements, [1, 2, 4, 5])
    assert [n for n, _c in match.nodes] == [2, 4]
    assert match.gaps == ()
    assert match.contraflow_m == 0.0


def test_match_takes_the_carriageway_for_the_direction_of_travel():
    # Two one-way carriageways 15 m apart. The shape runs between them, so distance
    # alone cannot choose; only the legal direction can.
    elements = [node(i, (i - 1) * 50.0, 0.0) for i in range(1, 6)]
    elements += [node(i + 10, (i - 1) * 50.0, 15.0) for i in range(1, 6)]
    elements += [way(10, [1, 2, 3, 4, 5], oneway="yes"), way(11, [15, 14, 13, 12, 11], oneway="yes")]
    graph = si.RoadGraph(elements)
    shape = [(lat, lon, c) for (lat, lon), c in
             [(si._shape_position(shape_along(graph, [1, 2, 3, 4, 5]), c), c) for c in (0.0, 100.0, 200.0)]]
    shift = 7.5e-5 / 1.11
    shape = [(lat + shift, lon, c) for lat, lon, c in shape]
    match = si.match_shape(graph, shape, 0.0, 200.0)
    assert {n for n, _c in match.nodes} <= {1, 2, 3, 4, 5}


def test_stretch_with_no_road_is_reported_as_a_gap():
    elements = [node(i, (i - 1) * 50.0, 0.0) for i in range(1, 4)]
    elements += [node(i, 300.0 + (i - 4) * 50.0, 0.0) for i in range(4, 7)]
    elements += [way(10, [1, 2, 3]), way(11, [4, 5, 6])]
    graph = si.RoadGraph(elements)
    match = si.match_shape(graph, shape_along(graph, [1, 2, 3, 4, 5, 6]), 0.0, 400.0)
    assert len(match.gaps) == 1
    start, end = match.gaps[0]
    assert start >= 95.0 and end <= 305.0


def test_signal_beside_a_gap_is_found():
    elements = [node(1, 0, 0), node(2, 50, 0), node(3, 250, 0), node(4, 300, 0),
                node(5, 150, 20, highway="traffic_signals"), node(6, 150, 80),
                way(10, [1, 2]), way(11, [3, 4]), way(12, [5, 6])]
    graph = si.RoadGraph(elements)
    match = si.match_shape(graph, shape_along(graph, [1, 2, 3, 4]), 0.0, 300.0)
    assert si.signals_near_gaps(graph, match, shape_along(graph, [1, 2, 3, 4])) == [5]


def test_signal_on_a_side_road_is_not_on_the_bus_road():
    elements = [node(1, 0, 0), node(2, 100, 0), node(3, 200, 0),
                node(4, 100, 8, highway="traffic_signals"), node(5, 100, 100),
                way(10, [1, 2, 3]), way(11, [2, 4, 5])]
    _graph, _match, controls = traverse(elements, [1, 2, 3])
    assert controls == []


@pytest.mark.parametrize("nodes, role", [
    (list(range(1, 22)), si.STOP_LINE),
    (list(range(21, 0, -1)), si.OPPOSING),
])
def test_direction_tag_is_read_against_travel(nodes, role):
    _graph, _match, controls = traverse(straight_road((10, FORWARD)), nodes)
    assert [c.role for c in controls] == [role]


def test_opposing_tag_is_taken_literally_on_a_one_way():
    elements = [node(i, (i - 1) * 20.0, 0.0, **(BACKWARD if i == 3 else {})) for i in range(1, 6)]
    _graph, _match, controls = traverse(elements + [way(10, [1, 2, 3, 4, 5], oneway="yes")],
                                        [1, 2, 3, 4, 5])
    assert [c.role for c in controls] == [si.OPPOSING]


def test_untagged_signal_on_a_one_way_holds_the_bus():
    elements = [node(i, (i - 1) * 20.0, 0.0, **({"highway": "traffic_signals"} if i == 3 else {}))
                for i in range(1, 6)]
    _graph, _match, controls = traverse(elements + [way(10, [1, 2, 3, 4, 5], oneway="yes")],
                                        [1, 2, 3, 4, 5])
    assert (controls[0].role, controls[0].reason) == (si.STOP_LINE, "one_way_has_one_direction")


def test_opposing_tag_stands_for_a_bus_contraflow_on_a_one_way():
    elements = [node(i, (i - 1) * 20.0, 0.0, **(FORWARD if i == 3 else {})) for i in range(1, 6)]
    _graph, match, controls = traverse(elements + [way(10, [1, 2, 3, 4, 5], oneway="yes")],
                                       [5, 4, 3, 2, 1])
    assert match.contraflow_m > 0.0
    assert [c.role for c in controls] == [si.OPPOSING]


def test_crossing_holds_both_directions():
    forward = traverse(straight_road((10, CROSSING_TAGS)), list(range(1, 22)))[2]
    backward = traverse(straight_road((10, CROSSING_TAGS)), list(range(21, 0, -1)))[2]
    assert [len(si.stop_points(c)) for c in (forward, backward)] == [1, 1]


def test_own_line_crossing_and_opposing_line_are_one_stop_at_the_line():
    road = straight_road((5, FORWARD), (6, CROSSING_TAGS), (7, BACKWARD))
    _graph, _match, controls = traverse(road, list(range(1, 22)))
    points = si.stop_points(controls)
    assert len(points) == 1
    assert si.stop_position(points[0]).osm_id == "node:5"
    assert si.position_kind(si.stop_position(points[0])) == "mapped_stop_line"


@pytest.mark.parametrize("far_side, stops", [(150.0, 1), (195.0, 2)])
def test_wide_junction_is_one_stop_only_within_the_junction_span(far_side, stops):
    elements = [node(1, 0, 0), node(2, 100, 0, **FORWARD), node(3, 105, 0, **CROSSING_TAGS),
                node(4, far_side, 0, **CROSSING_TAGS), node(5, far_side + 5, 0, **BACKWARD),
                node(6, 300, 0), way(10, [1, 2, 3, 4, 5, 6])]
    _graph, _match, controls = traverse(elements, [1, 2, 3, 4, 5, 6])
    assert controls[2].chainage_m - controls[1].chainage_m > si.SAME_CONTROL_GAP_M
    assert len(si.stop_points(controls)) == stops


def dual_carriageway(crossing_x, side_road="secondary"):
    """One-way road with a stop line at 100 m and a side road at 130 m."""
    elements = [node(1, 0, 0), node(2, 100, 0, **FORWARD), node(3, 130, 0),
                node(4, crossing_x, 0, **CROSSING_TAGS), node(5, 400, 0), node(6, 130, 100)]
    return elements + [way(10, [1, 2, 3, 4, 5], oneway="yes"),
                       way(11, [3, 6], highway=side_road)]


@pytest.mark.parametrize("crossing_x, side_road, stops", [
    (150.0, "secondary", 1),    # just past the junction the stop line controls
    (150.0, "service", 2),      # a driveway is not a junction
    (175.0, "secondary", 2),    # too far past it to belong to it
])
def test_far_side_crossing_needs_a_junction_just_behind_it(crossing_x, side_road, stops):
    _graph, _match, controls = traverse(dual_carriageway(crossing_x, side_road), [1, 2, 3, 4, 5])
    assert controls[1].chainage_m - controls[0].chainage_m > si.SAME_CONTROL_GAP_M
    assert len(si.stop_points(controls)) == stops


def test_stop_line_after_a_far_side_crossing_is_the_next_junction():
    elements = dual_carriageway(150.0)
    elements[3:5] = [node(4, 150, 0, **CROSSING_TAGS), node(5, 175, 0, **FORWARD)]
    elements += [node(8, 400, 0)]
    elements[6] = way(10, [1, 2, 3, 4, 5, 8], oneway="yes")
    _graph, _match, controls = traverse(elements, [1, 2, 3, 4, 5, 8])
    points = si.stop_points(controls)
    assert [[c.osm_id for c in point] for point in points] == [["node:2", "node:4"], ["node:5"]]


def test_far_side_crossing_past_an_opposing_line_is_a_new_stop():
    elements = [node(1, 0, 0), node(2, 100, 0, **FORWARD), node(3, 130, 0), node(4, 140, 0, **BACKWARD),
                node(5, 150, 0, **CROSSING_TAGS), node(6, 400, 0), node(7, 130, 100),
                way(10, [1, 2, 3, 4, 5, 6]), way(11, [3, 7])]
    _graph, _match, controls = traverse(elements, [1, 2, 3, 4, 5, 6])
    assert len(si.stop_points(controls)) == 2


def test_direction_tagged_junction_node_is_a_stop_line_not_a_centre():
    elements = straight_road((10, FORWARD)) + [node(50, 180, 100), way(101, [10, 50])]
    _graph, _match, controls = traverse(elements, list(range(1, 22)))
    assert si.position_kind(controls[0]) == "mapped_stop_line"


def test_next_junction_after_an_opposing_line_is_a_new_stop_however_close():
    road = straight_road((5, FORWARD), (6, BACKWARD), (7, FORWARD), (8, BACKWARD))
    _graph, _match, controls = traverse(road, list(range(1, 22)))
    assert len(si.stop_points(controls)) == 2


def test_close_stop_lines_with_no_opposing_line_are_one_stop():
    road = straight_road((5, FORWARD), (6, FORWARD))
    _graph, _match, controls = traverse(road, list(range(1, 22)))
    assert len(si.stop_points(controls)) == 1


def test_controls_beyond_the_gap_are_separate_stops():
    road = straight_road((5, FORWARD), (8, FORWARD))
    _graph, _match, controls = traverse(road, list(range(1, 22)))
    assert controls[1].chainage_m - controls[0].chainage_m == pytest.approx(60.0, abs=1.0)
    assert len(si.stop_points(controls)) == 2


def test_centre_mapped_junction_with_crossings_is_one_stop_at_the_near_crossing():
    elements = straight_road((9, CROSSING_TAGS), (10, {"highway": "traffic_signals"}), (11, CROSSING_TAGS))
    elements += [node(50, 180, 100), way(101, [10, 50])]
    _graph, _match, controls = traverse(elements, list(range(1, 22)))
    points = si.stop_points(controls)
    assert len(points) == 1
    assert si.stop_position(points[0]).osm_id == "node:9"


def test_centre_alone_keeps_the_centre_position():
    elements = straight_road((10, {"highway": "traffic_signals"}))
    elements += [node(50, 180, 100), way(101, [10, 50])]
    _graph, _match, controls = traverse(elements, list(range(1, 22)))
    point = si.stop_points(controls)[0]
    assert si.position_kind(si.stop_position(point)) == "junction_centre"


def test_return_visit_to_a_signal_is_a_second_stop():
    elements = [node(1, 0, 0), node(2, 100, 0), node(3, 200, 0, highway="traffic_signals"),
                node(4, 300, 0), node(5, 300, 100), node(6, 100, 100)]
    elements += [way(10, [1, 2, 3, 4], oneway="no"), way(11, [4, 5, 6, 2], oneway="no")]
    _graph, _match, controls = traverse(elements, [1, 2, 3, 4, 5, 6, 2, 3, 4])
    assert [c.osm_id for c in controls] == ["node:3", "node:3"]
    assert len(si.stop_points(controls)) == 2


def test_crossing_mapped_only_as_a_footway_counts_where_it_meets_the_path():
    road = straight_road()
    graph = si.RoadGraph(road)
    a = graph.node_ll[10]
    ways = [{"osm_id": "way:7", "points": [(a[0] - 2e-4, a[1] + 1e-5), (a[0] + 2e-4, a[1] + 1e-5)]}]
    _graph, _match, controls = traverse(road, list(range(1, 22)), ways)
    assert [(c.osm_id, c.role) for c in controls] == [("way:7", si.CROSSING)]
    assert len(si.stop_points(controls)) == 1


def test_signal_before_the_first_stop_is_not_served():
    road = straight_road((3, FORWARD))
    graph = si.RoadGraph(road)
    shape = shape_along(graph, list(range(1, 22)))
    match = si.match_shape(graph, shape, 100.0, shape[-1][2])
    assert si.path_controls(graph, match) == []


def test_interval_ownership_is_half_open_at_stops():
    road = straight_road((6, FORWARD))
    graph = si.RoadGraph(road)
    a, b = graph.node_ll[1], graph.node_ll[6]
    point = [si.Control("node:6", 100.0, si.STOP_LINE, "test", False)]
    intervals = [si.Interval("first", "s1", "s2", "x", 100.0, (a, b), 0.0),
                 si.Interval("second", "s2", "s3", "x", 300.0, (b, graph.node_ll[21]), 100.0)]
    radii = dict(snap_radius_m=30.0, relaxed_snap_radius_m=60.0, cluster_radius_m=35.0)
    events = si.interval_events([point], intervals, graph, "2026-08-26T16:04:56Z", radii)
    assert events["first"] == []
    assert [e["distance_along_segment_m"] for e in events["second"]] == [0.0]
    assert events["second"][0]["source"] == si.RULE_SOURCE


def test_every_rule_row_loads_through_the_runner_cache(tmp_path):
    summaries = {"g": dict(segment_geometry_id="g", shape_id="x", from_stop_id="a", to_stop_id="b",
                           n_signals=0, source=si.RULE_SOURCE, snap_radius_m=30.0,
                           relaxed_snap_radius_m=60.0, cluster_radius_m=35.0, length_m=10.0,
                           fetched_utc="t")}
    si.save_signal_cache(tmp_path / "signals.csv", summaries)
    assert load_signal_cache(tmp_path / "signals.csv")["g"]["source"] == si.RULE_SOURCE


def test_frozen_inventory_refuses_a_miss():
    with pytest.raises(ValueError, match="frozen signal inventory"):
        resolve_signal_counts([("g", "a", "b", 100.0, "s")], {"a": (52.0, -8.0), "b": (52.0, -8.1)},
                              cache={}, event_cache={}, fetch_fn=None, verbose=False, frozen=True)


def test_saved_response_is_reused_without_a_fetch(tmp_path):
    calls = []

    def fetch(query):
        calls.append(query)
        return [{"type": "node", "id": 1}]

    first = si.saved_elements("q", tmp_path, "r", fetch=fetch)
    second = si.saved_elements("q", tmp_path, "r", fetch=fetch)
    assert first == second and len(calls) == 1
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["query"] == "q" and saved["route"] == "r"


def test_unavailable_error_is_importable():
    # The rule feeds the same gate F1-U already reads.
    assert issubclass(PlacementUnavailable, ValueError)
