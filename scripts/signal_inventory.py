"""
signal_inventory.py
================================================================================
Rule-based signal inventory: N is the number of distinct signal-controlled stop
points a bus on one directed pattern must obey on one traversal.

The legacy count in traffic_signals.py takes every signal within a radius of the
shape and merges by a 35 m chain. That cannot tell the bus's own road from a side
road, or its own stop line from the opposing one. This module matches each GTFS
shape onto the directed OSM road graph first, so "on the bus's road" is a yes/no
question about the path, and then applies the control rules below.

Everything is computed from saved Overpass responses at the pinned snapshot, so a
build needs no network once the responses exist.
================================================================================
"""

import argparse
import csv
import hashlib
import heapq
import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, box

from traffic_signals import (
    DEFAULT_RELAXED_SNAP_RADIUS_M, DEFAULT_SIGNAL_CLUSTER_RADIUS_M, DEFAULT_SNAP_RADIUS_M,
    OSM_SIGNAL_TAG_QUERIES, OVERPASS_TIMEOUT_S, _event_id, _haversine_m, _interp_on_polyline,
    _overpass_elements, _pair_geometries, _segment_geometry_id, _signal_query,
    add_traffic_signals, load_signal_cache, load_signal_event_cache, overpass_date_setting,
    save_signal_cache, save_signal_event_cache,
)

RULE_VERSION = "N-rule-v1"
RULE_SOURCE = "osm_rule"
GEOMETRY_VERSION = "repaired-g2"
STOP_LINE, CROSSING, OPPOSING = "stop_line", "crossing", "opposing"
SAME_CONTROL_GAP_M = 35.0     # the valley in node spacing on the six routes
JUNCTION_SPAN_M = 80.0        # widest own-to-opposing stop line pair on the six routes
FAR_SIDE_M = 25.0             # the valley in crossing distance past a junction
JUNCTION_ROADS = 3

MOTOR_HIGHWAYS = (
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "busway", "bus_guideway", "road",
)
BUS_ALLOWED = frozenset({"yes", "designated", "permissive"})
TRUE_VALUES = frozenset({"yes", "1", "true"})

SAMPLE_STEP_M = 10.0
CANDIDATE_RADIUS_M = 30.0
OFFSET_SIGMA_M = 8.0
HEADING_SIGMA_DEG = 35.0
MAX_HEADING_DIFF_DEG = 110.0
TRANSITION_SCALE_M = 6.0
REACH_CUTOFF_M = 150.0
CONTRAFLOW_COST_PER_M = 0.3
GAP_SIGNAL_RADIUS_M = 40.0
BBOX_PAD_M = 150.0
_EARTH_R = 6_371_000.0


def bus_may_use(tags):
    """True when a way carries motor traffic or explicitly admits buses."""
    if tags.get("highway") not in MOTOR_HIGHWAYS:
        return False
    if tags.get("bus") in BUS_ALLOWED or tags.get("psv") in BUS_ALLOWED:
        return True
    return tags.get("motor_vehicle", tags.get("access")) not in {"no", "private"}


def legal_directions(tags):
    """(forward_legal, backward_legal) for a bus on a way, relative to node order."""
    oneway = tags.get("oneway", "yes" if tags.get("junction") == "roundabout" else "no")
    if oneway in TRUE_VALUES:
        forward, backward = True, False
    elif oneway == "-1":
        forward, backward = False, True
    else:
        forward, backward = True, True
    if tags.get("oneway:bus") == "no" or tags.get("oneway:psv") == "no":
        return True, True
    if any(str(tags.get(key, "")).startswith("opposite")
           for key in ("busway", "busway:left", "busway:right", "busway:both")):
        return True, True
    return forward, backward


@dataclass(frozen=True)
class RoadEdge:
    u: int
    v: int
    way_id: int
    forward: bool
    legal: bool
    length_m: float


class RoadGraph:
    """Directed node-to-node edges of the bus-usable OSM roads, in local metres."""

    def __init__(self, elements):
        points = {e["id"]: e for e in elements if e["type"] == "node"}
        self.lat0 = float(np.mean([e["lat"] for e in points.values()]))
        self.lon0 = float(np.mean([e["lon"] for e in points.values()]))
        self.node_tags = {i: e.get("tags", {}) for i, e in points.items()}
        self.node_ll = {i: (e["lat"], e["lon"]) for i, e in points.items()}
        self.node_xy = {i: self.xy(*ll) for i, ll in self.node_ll.items()}
        self.way_tags, self.way_nodes = {}, {}
        self.edges, self.out = {}, defaultdict(list)
        self.neighbours = defaultdict(set)
        for way in elements:
            if way["type"] != "way" or not bus_may_use(way.get("tags", {})):
                continue
            self.way_tags[way["id"]] = way["tags"]
            self.way_nodes[way["id"]] = way["nodes"]
            forward, backward = legal_directions(way["tags"])
            for a, b in zip(way["nodes"][:-1], way["nodes"][1:]):
                if a == b or a not in points or b not in points:
                    continue
                self.neighbours[a].add(b)
                self.neighbours[b].add(a)
                self._add(a, b, way["id"], True, forward)
                self._add(b, a, way["id"], False, backward)
        self._edge_list = list(self.edges.values())
        lines = [LineString([self.node_xy[e.u], self.node_xy[e.v]]) for e in self._edge_list]
        self._tree = STRtree(lines)
        self._reach = {}

    def xy(self, lat, lon):
        return (math.radians(lon - self.lon0) * math.cos(math.radians(self.lat0)) * _EARTH_R,
                math.radians(lat - self.lat0) * _EARTH_R)

    def _add(self, u, v, way_id, forward, legal):
        ux, uy = self.node_xy[u]
        vx, vy = self.node_xy[v]
        edge = RoadEdge(u, v, way_id, forward, legal, math.hypot(vx - ux, vy - uy))
        current = self.edges.get((u, v))
        if current is None or (legal and not current.legal):
            self.edges[(u, v)] = edge
            if current is None:
                self.out[u].append(v)

    def road_degree(self, node):
        """Roads meeting at a node, leaving out driveways and other service roads."""
        return sum(self.way_tags[self.edges[(node, other)].way_id].get("highway") != "service"
                   for other in self.neighbours[node])

    def reach(self, start):
        """Shortest directed distances from a node, up to REACH_CUTOFF_M."""
        cached = self._reach.get(start)
        if cached is not None:
            return cached
        dist, parent, heap = {start: 0.0}, {}, [(0.0, start)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node]:
                continue
            for nxt in self.out[node]:
                edge = self.edges[(node, nxt)]
                cost = edge.length_m * (1.0 if edge.legal else 1.0 + CONTRAFLOW_COST_PER_M)
                nd = d + cost
                if nd <= REACH_CUTOFF_M and nd < dist.get(nxt, math.inf):
                    dist[nxt], parent[nxt] = nd, node
                    heapq.heappush(heap, (nd, nxt))
        self._reach[start] = (dist, parent)
        return self._reach[start]

    def candidates(self, x, y, tx, ty):
        """Directed edges near a sample, as (edge, along_m, emission_cost)."""
        found = []
        for index in self._tree.query(self._box(x, y)):
            edge = self._edge_list[index]
            ux, uy = self.node_xy[edge.u]
            vx, vy = self.node_xy[edge.v]
            dx, dy = vx - ux, vy - uy
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            ex, ey = dx / length, dy / length
            along = max(0.0, min(length, (x - ux) * ex + (y - uy) * ey))
            offset = math.hypot(ux + along * ex - x, uy + along * ey - y)
            if offset > CANDIDATE_RADIUS_M:
                continue
            cos_diff = max(-1.0, min(1.0, ex * tx + ey * ty))
            heading = math.degrees(math.acos(cos_diff))
            if heading > MAX_HEADING_DIFF_DEG:
                continue
            cost = 0.5 * (offset / OFFSET_SIGMA_M) ** 2 + 0.5 * (heading / HEADING_SIGMA_DEG) ** 2
            found.append((edge, along, cost + (0.0 if edge.legal else 2.0)))
        return found

    @staticmethod
    def _box(x, y):
        r = CANDIDATE_RADIUS_M
        return box(x - r, y - r, x + r, y + r)


def _shape_position(shape_points, cum):
    """(lat, lon) at arc position cum, by bisection on the shape's cumulative column."""
    cums = [p[2] for p in shape_points]
    hi = int(np.searchsorted(cums, cum))
    if hi <= 0:
        return shape_points[0][0], shape_points[0][1]
    if hi >= len(shape_points):
        return shape_points[-1][0], shape_points[-1][1]
    (lat0, lon0, c0), (lat1, lon1, c1) = shape_points[hi - 1], shape_points[hi]
    f = 0.0 if c1 <= c0 else (cum - c0) / (c1 - c0)
    return lat0 + f * (lat1 - lat0), lon0 + f * (lon1 - lon0)


@dataclass(frozen=True)
class PathMatch:
    """Ordered road nodes a shape traverses, with their shape arc positions."""

    nodes: tuple          # (node_id, chainage_m), in travel order
    edges: tuple          # (u, v) directed edges, in travel order
    gaps: tuple           # (chainage_from, chainage_to) spans with no road match
    contraflow_m: float   # travelled length against the mapped one-way direction
    mean_offset_m: float
    max_offset_m: float


def match_shape(graph, shape_points, start_m, end_m):
    """
    Viterbi match of a served path onto directed road edges.

    Emission prefers a nearby, aligned edge; transition prefers a road distance
    equal to the distance along the shape. A step no road connects is a gap, which
    restarts the match and is reported, never bridged by assumption.
    """
    count = max(int(math.ceil((end_m - start_m) / SAMPLE_STEP_M)), 1) + 1
    chainages = np.linspace(start_m, end_m, count)
    samples = []
    for c in chainages:
        ax, ay = graph.xy(*_shape_position(shape_points, max(c - 8.0, start_m)))
        bx, by = graph.xy(*_shape_position(shape_points, min(c + 8.0, end_m)))
        norm = math.hypot(bx - ax, by - ay) or 1.0
        x, y = graph.xy(*_shape_position(shape_points, c))
        samples.append((float(c), x, y, (bx - ax) / norm, (by - ay) / norm))

    runs, current, gaps = [], [], []
    previous = None        # [(edge, along, cost, back_index), ...] at the last sample
    for index, (chain, x, y, tx, ty) in enumerate(samples):
        options = graph.candidates(x, y, tx, ty)
        if not options:
            continue
        layer = []
        if previous is None:
            layer = [(edge, along, cost, None) for edge, along, cost in options]
        else:
            step = chain - current[-1][0]
            for edge, along, cost in options:
                best, back = math.inf, None
                for i, (prior, prior_along, prior_cost, _b) in enumerate(previous):
                    moved = _road_distance(graph, prior, prior_along, edge, along)
                    if moved is None:
                        continue
                    total = prior_cost + cost + abs(moved - step) / TRANSITION_SCALE_M
                    if total < best:
                        best, back = total, i
                if back is not None:
                    layer.append((edge, along, best, back))
        if previous is not None and not layer:
            gaps.append((current[-1][0], chain))
            runs.append(current)
            current, previous = [], None
            layer = [(edge, along, cost, None) for edge, along, cost in options]
        current.append((chain, layer, x, y))
        previous = layer
    if current:
        runs.append(current)
    first, last = samples[0][0], samples[-1][0]
    if not runs:
        return PathMatch((), (), ((first, last),), 0.0, math.inf, math.inf)
    if runs[0][0][0] > first:
        gaps.insert(0, (first, runs[0][0][0]))
    if runs[-1][-1][0] < last:
        gaps.append((runs[-1][-1][0], last))

    nodes, edges, offsets = [], [], []
    for run in runs:
        states = _backtrack(run)
        offsets += [_offset(graph, edge, along, x, y) for _c, edge, along, x, y in states]
        edges.append((states[0][1].u, states[0][1].v))
        for (c0, e0, _a0, _x0, _y0), (c1, e1, _a1, _x1, _y1) in zip(states[:-1], states[1:]):
            if e0 == e1:
                continue
            route = _node_route(graph, e0, e1)
            for j, node in enumerate(route):
                nodes.append((node, c0 + (j + 1) / (len(route) + 1) * (c1 - c0)))
            edges += list(zip(route[:-1], route[1:])) + [(e1.u, e1.v)]
    contraflow = sum(graph.edges[hop].length_m for hop in edges if not graph.edges[hop].legal)
    served = [(node, chain) for node, chain in _refined_chainages(graph, shape_points, nodes)
              if start_m <= chain < end_m]
    return PathMatch(tuple(served), tuple(edges), tuple(gaps), contraflow,
                     float(np.mean(offsets)), float(np.max(offsets)))


def _refined_chainages(graph, shape_points, nodes):
    """Replace interpolated node positions with the foot on the nearby shape, in order."""
    cums = np.array([p[2] for p in shape_points])
    xy = np.array([graph.xy(p[0], p[1]) for p in shape_points])
    refined, floor = [], -math.inf
    for node, chain in nodes:
        lo = max(int(np.searchsorted(cums, chain - SAMPLE_STEP_M)) - 1, 0)
        hi = min(int(np.searchsorted(cums, chain + SAMPLE_STEP_M)) + 1, len(cums) - 1)
        px, py = graph.node_xy[node]
        best_d, best_c = math.inf, chain
        for i in range(lo, hi):
            (ax, ay), (bx, by) = xy[i], xy[i + 1]
            dx, dy = bx - ax, by - ay
            denom = dx * dx + dy * dy
            f = 0.0 if denom <= 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
            d = math.hypot(ax + f * dx - px, ay + f * dy - py)
            if d < best_d:
                best_d, best_c = d, cums[i] + f * (cums[i + 1] - cums[i])
        floor = max(floor, float(best_c))
        refined.append((node, floor))
    return refined


def _road_distance(graph, e0, a0, e1, a1):
    """Road distance from a position on one directed edge to one on another."""
    if e0 == e1:
        return a1 - a0 if a1 >= a0 else max(a0 - a1 - 8.0, 0.0) * 4.0
    dist = graph.reach(e0.v)[0].get(e1.u)
    return None if dist is None else (e0.length_m - a0) + dist + a1


def _node_route(graph, e0, e1):
    """Nodes entered between two different edges: e0.v ... e1.u inclusive."""
    parents = graph.reach(e0.v)[1]
    route, node = [e1.u], e1.u
    while node != e0.v:
        node = parents[node]
        route.append(node)
    return route[::-1]


def _backtrack(run):
    chain, layer, x, y = run[-1]
    index = min(range(len(layer)), key=lambda i: layer[i][2])
    states = []
    for step in range(len(run) - 1, -1, -1):
        chain, layer, x, y = run[step]
        edge, along, _cost, back = layer[index]
        states.append((chain, edge, along, x, y))
        index = back
    return states[::-1]


def _offset(graph, edge, along, x, y):
    ux, uy = graph.node_xy[edge.u]
    vx, vy = graph.node_xy[edge.v]
    f = along / edge.length_m if edge.length_m else 0.0
    return math.hypot(ux + f * (vx - ux) - x, uy + f * (vy - uy) - y)


# -----------------------------------------------------------------------------
# The control rules
# -----------------------------------------------------------------------------
def is_signal_tagged(tags):
    """True for the same signal tags the OSM fetch asks for."""
    return any(all(tags.get(key) == value for key, value in query.items())
               for _label, query in OSM_SIGNAL_TAG_QUERIES)


@dataclass(frozen=True)
class Control:
    """One signal feature met on a traversal, in travel order."""

    osm_id: str
    chainage_m: float
    role: str             # STOP_LINE, CROSSING or OPPOSING
    reason: str
    junction: bool        # a junction node with no direction of its own
    past_junction_m: float = math.inf   # distance beyond the last junction on the path


def node_role(graph, previous, node, following):
    """
    What a signal node on the path means for a bus going previous -> node -> following.

    OPPOSING   : a stop line tagged for the other direction. It is never a stop for
                 this bus, but it marks the far side of the control it belongs to.
    CROSSING   : a signalised pedestrian crossing, which holds both directions.
    STOP_LINE  : any other signal, including a junction centre and an untagged
                 mid-block signal, which hold this direction.
    """
    tags = graph.node_tags[node]
    entering = graph.edges.get((previous, node))
    leaving = graph.edges.get((node, following))
    edge = entering or leaving
    if tags.get("highway") != "traffic_signals":
        return CROSSING, "signalised_crossing_holds_both_directions"
    direction = tags.get("traffic_signals:direction", tags.get("direction", ""))
    if direction in ("forward", "backward") and edge is not None:
        if (direction == "forward") == edge.forward:
            return STOP_LINE, "direction_tag_matches_travel"
        return OPPOSING, "direction_tag_opposes_travel"
    if direction == "both":
        return STOP_LINE, "direction_tag_both"
    if edge is not None and edge.legal and not all(legal_directions(graph.way_tags[edge.way_id])):
        return STOP_LINE, "one_way_has_one_direction"
    if graph.road_degree(node) >= JUNCTION_ROADS:
        return STOP_LINE, "junction_centre_holds_all_approaches"
    return STOP_LINE, "untagged_mid_block_signal"


def path_controls(graph, match, crossing_ways=()):
    """
    Ordered controls on a matched traversal.

    A node counts only if the path passes through it, so a signal on a side road
    or the other carriageway never qualifies, however close. crossing_ways adds a
    signalised crossing mapped only as a footway, where it crosses the path.
    """
    sequence = [node for node, _chain in match.nodes]
    junctions = [chain for node, chain in match.nodes
                 if graph.road_degree(node) >= JUNCTION_ROADS]
    controls = []
    for i, (node, chain) in enumerate(match.nodes):
        if not is_signal_tagged(graph.node_tags.get(node, {})):
            continue
        previous = sequence[i - 1] if i else None
        following = sequence[i + 1] if i + 1 < len(sequence) else None
        role, reason = node_role(graph, previous, node, following)
        centre = (graph.road_degree(node) >= JUNCTION_ROADS
                  and reason != "direction_tag_matches_travel")
        controls.append(Control(f"node:{node}", chain, role, reason, centre,
                                _past_junction(junctions, chain)))
    controls += [replace(c, past_junction_m=_past_junction(junctions, c.chainage_m))
                 for c in _crossing_way_controls(graph, match, crossing_ways)]
    return sorted(controls, key=lambda c: (c.chainage_m, c.osm_id))


def _past_junction(junctions, chain):
    """Distance from the last junction strictly behind a position on the path."""
    index = bisect_left(junctions, chain)
    return chain - junctions[index - 1] if index else math.inf


def _crossing_way_controls(graph, match, crossing_ways):
    if len(match.nodes) < 2 or not crossing_ways:
        return []
    points = [graph.node_xy[node] for node, _chain in match.nodes]
    chains = [chain for _node, chain in match.nodes]
    lengths = np.cumsum([0.0] + [math.dist(a, b) for a, b in zip(points[:-1], points[1:])])
    path = LineString(points)
    tree = STRtree([LineString([graph.xy(*p) for p in way["points"]]) for way in crossing_ways])
    found = []
    for index in tree.query(path, predicate="intersects"):
        line = tree.geometries[index]
        hit = line.intersection(path)
        for point in getattr(hit, "geoms", [hit]):
            chain = float(np.interp(path.project(point.representative_point()), lengths, chains))
            found.append(Control(crossing_ways[index]["osm_id"], chain, CROSSING,
                                 "signalised_crossing_way_meets_path", False))
    return found


def stop_points(controls):
    """
    Group a traversal's controls into stop points, one list per point.

    A control joins the current point if it is within SAME_CONTROL_GAP_M of the
    previous member, because nodes that close belong to one signal plan and a bus
    cannot queue between them. An opposing stop line marks the far side of a
    control, so two refinements use it:
      * a stop line after it starts a new point, however close: that is the next
        junction, not this one;
      * a crossing before it stays in the point, however wide the junction, when
        the point has its own stop line within JUNCTION_SPAN_M.
    A dual carriageway has no opposing line to mark the far side, so a crossing
    just past a junction the point controls stays in the point and marks the far
    side itself: the bus has cleared that junction.
    """
    points, closed = [], False
    for index, control in enumerate(controls):
        if control.role == OPPOSING:
            closed = True
            continue
        current = points[-1] if points else None
        joins = far_side = False
        if current is not None:
            near = control.chainage_m - current[-1].chainage_m <= SAME_CONTROL_GAP_M
            if control.role == STOP_LINE:
                joins = near and not closed
            else:
                far_side = not closed and _far_side(control, current)
                joins = near or far_side or _inside_junction(controls, index, current)
        if joins:
            current.append(control)
            closed = closed or far_side
        else:
            points.append([control])
            closed = False
    return points


def _inside_junction(controls, index, point):
    """True if the next non-crossing control closes this point's junction."""
    lines = [c for c in point if c.role == STOP_LINE]
    if not lines:
        return False
    for later in controls[index + 1:]:
        if later.role != CROSSING:
            return (later.role == OPPOSING
                    and later.chainage_m - lines[0].chainage_m <= JUNCTION_SPAN_M)
    return False


def _far_side(control, point):
    """True for a crossing just past a junction reached after this point's last member."""
    return (control.past_junction_m <= FAR_SIDE_M
            and control.past_junction_m < control.chainage_m - point[-1].chainage_m
            and control.chainage_m - point[0].chainage_m <= JUNCTION_SPAN_M)


def stop_position(point):
    """
    Where the bus stops: the first mapped stop line, else the first member.

    A junction-centre node is only used when nothing upstream of it is mapped. No
    offset is applied to it, because the audits found no justified value.
    """
    return next((c for c in point if c.role == STOP_LINE and not c.junction), point[0])


def position_kind(control):
    if control.role == CROSSING:
        return "crossing_position"
    return "junction_centre" if control.junction else "mapped_stop_line"


# -----------------------------------------------------------------------------
# Saved OSM responses
# -----------------------------------------------------------------------------
def road_query(south, west, north, east, snapshot=None):
    return (f"[out:json][timeout:{OVERPASS_TIMEOUT_S}]{overpass_date_setting(snapshot)};"
            f'way["highway"~"^({"|".join(MOTOR_HIGHWAYS)})$"]({south},{west},{north},{east});'
            "(._;>;);out body;")


def saved_elements(query, store, label, fetch=_overpass_elements):
    """Elements for a query, from its saved response if there is one."""
    path = Path(store) / (hashlib.sha256(query.encode()).hexdigest()[:16] + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["elements"]
    elements = fetch(query)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"route": label, "query": query, "elements": elements}),
                    encoding="utf-8")
    return elements


def crossing_ways_from(elements):
    """Mapped signalised crossing ways, in the form path_controls expects."""
    return [{"osm_id": f"way:{e['id']}", "points": [(p["lat"], p["lon"]) for p in e["geometry"]]}
            for e in elements if e["type"] == "way"]


def _shape_bbox(shape_lists, pad_m=BBOX_PAD_M):
    lats = sorted(p[0] for shape in shape_lists for p in shape)
    lons = sorted(p[1] for shape in shape_lists for p in shape)
    dlat = pad_m / 111_320.0
    dlon = pad_m / (111_320.0 * math.cos(math.radians(sum(lats) / len(lats))))
    return lats[0] - dlat, lons[0] - dlon, lats[-1] + dlat, lons[-1] + dlon


# -----------------------------------------------------------------------------
# Inventory and registry
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Pattern:
    route_id: str
    direction_id: str
    shape_id: str
    stop_visits: tuple
    trips: tuple


@dataclass(frozen=True)
class Interval:
    geometry_id: str
    from_id: str
    to_id: str
    shape_id: str
    length_m: float
    polyline: tuple
    start_m: float


def pattern_intervals(rows, stops, shape_points, shape_id):
    """The modelled stop-to-stop intervals of one pattern, with their shape positions."""
    from gtfs_to_segment import _shape_stop_positions

    positions = _shape_stop_positions(rows, stops, shape_points)
    intervals, cursor = [], 0
    for a, b, length_m, poly in _pair_geometries(rows, stops, shape_points=shape_points):
        while rows[cursor] is not a:
            cursor += 1
        f, t = a["stop_id"], b["stop_id"]
        gid = _segment_geometry_id(f, t, shape_id=shape_id, poly=poly)
        intervals.append(Interval(gid, str(f), str(t), str(shape_id), float(length_m),
                                  tuple(poly), float(positions[cursor])))
    return intervals


def interval_events(points, intervals, graph, snapshot, radii):
    """
    One event per stop point, in the runner's cache schema, keyed by interval.

    A stop point belongs to the interval holding its stop position, half-open at
    the stops, so a point is counted once however its members fall.
    """
    starts = [interval.start_m for interval in intervals]
    events = {interval.geometry_id: [] for interval in intervals}
    for point in points:
        control = stop_position(point)
        interval = intervals[bisect_right(starts, control.chainage_m) - 1]
        along = control.chainage_m - interval.start_m
        if not 0.0 <= along < interval.length_m:
            raise ValueError(f"{control.osm_id} falls outside {interval.geometry_id}")
        lat, lon = _interp_on_polyline(list(interval.polyline), along)
        raw_ids = sorted(member.osm_id for member in point)
        node = int(control.osm_id[5:]) if control.osm_id.startswith("node:") else None
        raw_lat, raw_lon = graph.node_ll[node] if node is not None else (lat, lon)
        tags = graph.node_tags[node] if node is not None else {}
        events[interval.geometry_id].append({
            "signal_event_id": _event_id(interval.geometry_id, along, raw_ids, []),
            "segment_geometry_id": interval.geometry_id,
            "from_stop_id": interval.from_id,
            "to_stop_id": interval.to_id,
            "shape_id": interval.shape_id,
            "signal_lat": float(lat),
            "signal_lon": float(lon),
            "distance_along_segment_m": float(along),
            "segment_length_m": interval.length_m,
            "source": RULE_SOURCE,
            "snap_radius_m": radii["snap_radius_m"],
            "relaxed_snap_radius_m": radii["relaxed_snap_radius_m"],
            "cluster_radius_m": radii["cluster_radius_m"],
            "fetched_utc": snapshot,
            "raw_osm_ids": "|".join(raw_ids),
            "cluster_size": len(point),
            "snap_distance_m": _haversine_m(raw_lat, raw_lon, lat, lon),
            "raw_signal_lat": float(raw_lat),
            "raw_signal_lon": float(raw_lon),
            "signal_type": position_kind(control),
            "osm_tags": json.dumps(tags, sort_keys=True),
        })
    return events


def signals_near_gaps(graph, match, shape_points, radius_m=GAP_SIGNAL_RADIUS_M):
    """Signal nodes close to a stretch with no road match: the count there is unknown."""
    found = set()
    tagged = [n for n, tags in graph.node_tags.items() if is_signal_tagged(tags)]
    for start, end in match.gaps:
        chains = np.arange(start, end + SAMPLE_STEP_M, SAMPLE_STEP_M)
        samples = [graph.xy(*_shape_position(shape_points, c)) for c in chains]
        found.update(n for n in tagged
                     if min(math.dist(graph.node_xy[n], s) for s in samples) <= radius_m)
    return sorted(found)


def decision_rows(pattern, controls, points):
    """One audit row per control, naming the stop point it became part of."""
    owner = {id(member): i for i, point in enumerate(points) for member in point}
    stops_at = {id(stop_position(point)) for point in points}
    return [{
        "route_id": pattern.route_id, "direction_id": pattern.direction_id,
        "shape_id": pattern.shape_id, "chainage_m": round(c.chainage_m, 2),
        "osm_id": c.osm_id, "role": c.role, "reason": c.reason, "junction": c.junction,
        "stop_point": owner.get(id(c), ""),
        "decision": ("stop_position" if id(c) in stops_at else
                     "merged_into_stop_point" if id(c) in owner else "not_this_direction"),
    } for c in controls]


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_inventory(gtfs_zip, selectors, service_date, snapshot, out_dir, store_dir,
                    radii, shape_repairs=None, certification=""):
    """
    Build the rule inventory for the selected routes and write it to out_dir.

    Writes the runner's two signal caches, the audit tables and the placement
    registry. A signal close to a stretch with no road match stops the build,
    because the count there cannot be known. N is registered uncertified unless
    certification names its authority.
    """
    from gtfs_to_segment import (build_segments, load_shapes_for_trips, load_small_tables,
                                 resolve_route_id, service_ids_for_date, stop_times_for_trips,
                                 trips_for_route)
    from signal_placement import file_hash, inventory_signature

    routes, trips, stops, calendar, calendar_dates = load_small_tables(gtfs_zip)
    service_ids = service_ids_for_date(calendar, calendar_dates, service_date)
    directions = dict(zip(trips["trip_id"].astype(str), trips["direction_id"].astype(str)))
    route_trips = {}
    for selector in selectors:
        route_id, _name = resolve_route_id(routes, selector)
        route_trips[str(selector)] = (str(route_id),
                                      trips_for_route(trips, route_id, None, service_ids))
    wanted = {trip for _route, ids in route_trips.values() for trip in ids}
    by_trip = stop_times_for_trips(gtfs_zip, wanted)
    shapes, shape_of = load_shapes_for_trips(gtfs_zip, trips, wanted, shape_repairs)
    feed_sha256 = file_hash(gtfs_zip)

    events, summaries, decisions, counts, registered, gap_signals = {}, {}, [], [], [], []
    for selector, (route_id, trip_ids) in route_trips.items():
        patterns = {}
        for trip in trip_ids:
            key = (directions[str(trip)], shape_of[str(trip)],
                   tuple(str(row["stop_id"]) for row in by_trip[trip]))
            patterns.setdefault(key, []).append(trip)
        route_shapes = sorted({key[1] for key in patterns})
        south, west, north, east = _shape_bbox([shapes[shape] for shape in route_shapes])
        graph = RoadGraph(saved_elements(road_query(south, west, north, east, snapshot),
                                         Path(store_dir) / "overpass_roads", selector))
        ways = crossing_ways_from(saved_elements(
            _signal_query(north, south, east, west, snapshot),
            Path(store_dir) / "overpass_signals", selector))
        for (direction, shape_id, visits), pattern_trips in sorted(patterns.items()):
            rows, points_list = by_trip[pattern_trips[0]], shapes[shape_id]
            pattern = Pattern(route_id, direction, shape_id, visits, tuple(pattern_trips))
            intervals = pattern_intervals(rows, stops, points_list, shape_id)
            start_m = intervals[0].start_m
            end_m = intervals[-1].start_m + intervals[-1].length_m
            match = match_shape(graph, points_list, start_m, end_m)
            gap_signals += [(shape_id, node)
                            for node in signals_near_gaps(graph, match, points_list)]
            controls = path_controls(graph, match, ways)
            points = stop_points(controls)
            local = interval_events(points, intervals, graph, snapshot, radii)
            decisions += decision_rows(pattern, controls, points)
            for interval in intervals:
                gid = interval.geometry_id
                if gid in events and [e["raw_osm_ids"] for e in events[gid]] != [
                        e["raw_osm_ids"] for e in local[gid]]:
                    raise ValueError(f"patterns disagree on the signals in {gid}")
                events[gid] = local[gid]
                summaries[gid] = {
                    "segment_geometry_id": gid, "shape_id": shape_id,
                    "from_stop_id": interval.from_id, "to_stop_id": interval.to_id,
                    "n_signals": len(local[gid]), "source": RULE_SOURCE,
                    "length_m": interval.length_m, "fetched_utc": snapshot, **radii,
                }
            kinds = Counter(e["signal_type"] for gid in local for e in local[gid])
            counts.append({
                "route": selector, "route_id": route_id, "direction_id": direction,
                "shape_id": shape_id, "trips": len(pattern_trips), "N": len(points),
                "length_km": round((end_m - start_m) / 1000.0, 3),
                "gaps": len(match.gaps), "gap_m": round(sum(b - a for a, b in match.gaps), 1),
                "contraflow_m": round(match.contraflow_m, 1),
                "mean_offset_m": round(match.mean_offset_m, 2),
                "max_offset_m": round(match.max_offset_m, 2),
                "mapped_stop_line": kinds["mapped_stop_line"],
                "crossing_position": kinds["crossing_position"],
                "junction_centre": kinds["junction_centre"],
            })
            registered.append((pattern, rows, points_list, len(points)))
    if gap_signals:
        raise ValueError(f"Signals sit beside stretches with no road match: {gap_signals}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_signal_cache(out_dir / "traffic_signals.csv", summaries)
    save_signal_event_cache(out_dir / "traffic_signal_events.csv", events)
    _write_csv(out_dir / "control_decisions.csv", decisions)
    _write_csv(out_dir / "pattern_counts.csv", counts)

    # The registry identifies the inventory as the runner reads it back from disk.
    summary_cache = load_signal_cache(out_dir / "traffic_signals.csv")
    event_cache = load_signal_event_cache(out_dir / "traffic_signal_events.csv")
    count_map = {gid: row["n_signals"] for gid, row in summary_cache.items()}
    source_map = {gid: row["source"] for gid, row in summary_cache.items()}
    entries = []
    for pattern, rows, shape_points, n in registered:
        segments = build_segments(rows, stops, shape_points=shape_points)
        add_traffic_signals(segments, rows, stops, count_map=count_map, source_map=source_map,
                            event_map=event_cache, shape_points=shape_points,
                            shape_id=pattern.shape_id, verbose=False)
        boundaries = [0.0]
        for segment in segments:
            boundaries.append(boundaries[-1] + segment.length_m)
        entry = {
            "geometry": {"pattern_id": f"gtfs:{feed_sha256[:16]}:{GEOMETRY_VERSION}:p:{pattern.shape_id}",
                         "version": GEOMETRY_VERSION, "length_m": boundaries[-1],
                         "boundaries_m": boundaries, "accepted": True},
            "count": {"value": n, "certified": bool(certification), "authority": certification},
            "inventory_sha256": inventory_signature(segments),
            "inventory_authority": f"{RULE_VERSION} on OSM {snapshot}",
            "position_compatibility_authority": "",
            "bindings": [{"feed_sha256": feed_sha256, "route_id": pattern.route_id,
                          "direction_id": pattern.direction_id, "shape_id": pattern.shape_id,
                          "stop_visits": list(pattern.stop_visits)}],
            "candidates": {"records": [], "complete": False, "authority": ""},
        }
        if len(pattern.stop_visits) != len(segments) + 1:
            entry["zero_length_interval_authority"] = (
                "co-located consecutive stops merged by iter_valid_stop_pairs")
        entries.append(entry)
    (out_dir / "registry.json").write_text(json.dumps(
        {"schema": "F1-placement-registry-v1", "rule": RULE_VERSION, "patterns": entries},
        indent=2), encoding="utf-8")
    return counts


def main():
    from gtfs_to_segment import get_path, get_section, parse_service_date

    parser = argparse.ArgumentParser(description="Build the rule-based signal inventory.")
    parser.add_argument("--config", help="Path to a model YAML config file.")
    parser.add_argument("--date", required=True, help="GTFS service date, YYYY-MM-DD.")
    parser.add_argument("--routes", help="Comma-separated route_ids; default gtfs.default_routes.")
    parser.add_argument("--certify", default="",
                        help="Authority for certifying N. Omit to register N uncertified.")
    args = parser.parse_args()

    gtfs_cfg = get_section("gtfs", args.config)
    signals_cfg = get_section("traffic_signals", args.config)
    selectors = (args.routes.split(",") if args.routes else gtfs_cfg["default_routes"])
    counts = build_inventory(
        get_path("gtfs_zip", args.config), [str(s) for s in selectors],
        parse_service_date(args.date), get_section("simulation", args.config)["osm_snapshot_utc"],
        get_path("traffic_signals_csv", args.config).parent,
        get_path("osm_responses_dir", args.config),
        {"snap_radius_m": float(signals_cfg.get("snap_radius_m", DEFAULT_SNAP_RADIUS_M)),
         "relaxed_snap_radius_m": float(signals_cfg.get("relaxed_snap_radius_m",
                                                        DEFAULT_RELAXED_SNAP_RADIUS_M)),
         "cluster_radius_m": float(signals_cfg.get("cluster_radius_m",
                                                   DEFAULT_SIGNAL_CLUSTER_RADIUS_M))},
        shape_repairs=gtfs_cfg.get("shape_repairs"), certification=args.certify)
    for row in counts:
        print(f"{row['route']:14s} {row['shape_id']:10s} trips {row['trips']:3d} N {row['N']:3d}")


if __name__ == "__main__":
    main()
