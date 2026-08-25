"""
traffic_signals.py
================================================================================
Count the traffic signals on each stop-to-stop Segment from OpenStreetMap, so the
speed profile can reflect the urban stop-go penalty (see build_speed_profile in
beb_soc_model.py, which splits a segment into sub-links when seg.n_signals > 0).

WHY A CACHE
-----------
Signal counts are STATIC per directed stop pair (from_stop_id -> to_stop_id): the
stop coordinates never move, so the count for a pair does not depend on the date,
the trip, or the time of day. We therefore cache counts in a CSV keyed on the
stop pair. Running the same route on many dates fetches OSM once; trip variants
and other routes that share a corridor reuse the same rows for free.

The count for a pair is a pure function of (pair geometry, OSM signal set, snap
radius), so it is safe to compute each pair independently and store it once.

DATA-DRIVEN vs ASSUMPTION
-------------------------
DATA-DRIVEN : signal locations, from OSM, via OSMnx. The fetch includes both
              vehicle traffic signals and signal-controlled pedestrian crossings
              because Cork suburban stops may be affected by small pedestrian
              lights rather than large junctions.
ASSUMPTION  : snap_radius_m (how near a signal must be to count as "on" the pair)
              and, if OSM is unavailable, a per-km fallback density. Both are
              recorded per row (source column) so a run is auditable.

CACHE CSV COLUMNS
-----------------
    from_stop_id, to_stop_id, n_signals, source, snap_radius_m, length_m, fetched_utc
      source = "osm"      -> counted from OSM signal nodes
      source = "fallback" -> estimated from fallback_per_km * length (OSM missing)

PUBLIC API
----------
    resolve_signal_counts(pairs, coords, ...) -> (count_map, source_map, stats)
        the cache-aware resolver: cache hit -> OSM fetch (once) -> fallback.
    add_traffic_signals(segments, trip_rows, stops, count_map=..., ...)
        post-pass mirroring add_grades_from_dem: writes seg.n_signals (+ source).
    load_signal_cache / save_signal_cache : CSV I/O.
    fetch_traffic_signals_bbox : the only part that needs the network.
================================================================================
"""

import csv
import hashlib
import json
import time
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

DEFAULT_SNAP_RADIUS_M = 30.0
DEFAULT_RELAXED_SNAP_RADIUS_M = 60.0
DEFAULT_SIGNAL_CLUSTER_RADIUS_M = 35.0
DEFAULT_FALLBACK_PER_KM = 2.0        # urban signals/km when OSM is unavailable
_EARTH_R = 6_371_000.0

OSM_SIGNAL_TAG_QUERIES = [
    ("vehicle_signals", {"highway": "traffic_signals"}),
    ("ped_crossing_signals", {"crossing": "traffic_signals"}),
    ("ped_crossing_signal_yes", {"crossing:signals": "yes"}),
    ("ped_crossing_pelican", {"crossing_ref": "pelican"}),
    ("ped_crossing_puffin", {"crossing_ref": "puffin"}),
    ("ped_crossing_toucan", {"crossing_ref": "toucan"}),
]

OSM_SIGNAL_DEDUP_RADIUS_M = 2.0

SIGNAL_CACHE_COLUMNS = [
    "segment_geometry_id", "shape_id",
    "from_stop_id", "to_stop_id", "n_signals", "source",
    "snap_radius_m", "relaxed_snap_radius_m", "cluster_radius_m",
    "length_m", "fetched_utc",
]

SIGNAL_EVENT_CACHE_COLUMNS = [
    "signal_event_id", "segment_geometry_id", "from_stop_id", "to_stop_id",
    "shape_id", "signal_lat", "signal_lon", "distance_along_segment_m",
    "segment_length_m", "source", "snap_radius_m", "relaxed_snap_radius_m",
    "cluster_radius_m", "fetched_utc", "raw_osm_ids", "cluster_size",
    "snap_distance_m", "raw_signal_lat", "raw_signal_lon", "signal_type",
    "osm_tags",
]


def _signal_event_cache_path(summary_path):
    """Default event cache path next to the summary count cache."""
    path = Path(summary_path)
    return path.with_name("traffic_signal_events.csv")


def _signal_lat_lon(signal):
    if isinstance(signal, dict):
        return float(signal["lat"]), float(signal["lon"])
    return float(signal[0]), float(signal[1])


def _signal_record(signal):
    lat, lon = _signal_lat_lon(signal)
    if isinstance(signal, dict):
        return {
            "lat": lat,
            "lon": lon,
            "osm_id": signal.get("osm_id", ""),
            "signal_type": signal.get("signal_type", ""),
            "tags": signal.get("tags") or {},
        }
    return {"lat": lat, "lon": lon, "osm_id": "", "signal_type": "", "tags": {}}


def _segment_geometry_id(from_id, to_id, shape_id=None, poly=None):
    """
    Stable directed geometry identity.

    GTFS shape_id is the smallest existing stable key that distinguishes normal
    shape variants for the same stop pair. When no shape_id is available, we use
    a rounded polyline hash so chord/shape-less geometry variants do not collide.
    """
    f, t = str(from_id), str(to_id)
    shape_id = "" if shape_id is None else str(shape_id)
    if shape_id:
        base = f"shape:{shape_id}"
    else:
        coords = ";".join(f"{float(la):.7f},{float(lo):.7f}" for la, lo in (poly or []))
        digest = hashlib.blake2s(coords.encode("utf-8"), digest_size=8).hexdigest()
        base = f"geom:{digest}"
    return f"{base}|{f}|{t}"


def _pair_fields(pair):
    """Return (geometry_id, from_id, to_id, length_m, shape_id) for old/new pairs."""
    if len(pair) >= 5:
        gid, f, t, length_m, shape_id = pair[:5]
        return str(gid), str(f), str(t), float(length_m), str(shape_id or "")
    f, t, length_m = pair[:3]
    return (str(f), str(t)), str(f), str(t), float(length_m), ""


def _geom_for_pair(geom_map, geometry_id, from_id, to_id):
    if not geom_map:
        return None
    return geom_map.get(geometry_id) or geom_map.get((from_id, to_id))


# -----------------------------------------------------------------------------
# Geometry (local equirectangular metres; same approach as _project_stop_to_shape)
# -----------------------------------------------------------------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return 2 * _EARTH_R * asin(sqrt(a))


def _project_point_to_segment(plat, plon, alat, alon, blat, blon):
    """Perpendicular distance (m) from P to line A-B, and projection parameter t.
       t in [0,1] means the foot of the perpendicular lies between A and B."""
    lat0 = radians(plat)

    def xy(lat, lon):
        return (radians(lon - plon) * cos(lat0) * _EARTH_R,
                radians(lat - plat) * _EARTH_R)

    ax, ay = xy(alat, alon)
    bx, by = xy(blat, blon)
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 0:                       # A == B
        return sqrt(ax * ax + ay * ay), 0.0
    t = -(ax * vx + ay * vy) / denom     # P is at the origin
    tc = max(0.0, min(1.0, t))
    px, py = ax + tc * vx, ay + tc * vy
    return sqrt(px * px + py * py), t


def _clustered_count(positions_m, cluster_radius_m=DEFAULT_SIGNAL_CLUSTER_RADIUS_M):
    """
    Count physical signalised points from raw OSM traffic-signal nodes.

    OSM often maps one real junction as several highway=traffic_signals nodes
    (one per approach, crossing, or lane group). Counting raw nodes therefore
    overstates the number of stop-start opportunities. We first project matching
    nodes onto the segment/path, then merge nearby projected positions.
    """
    if not positions_m:
        return 0
    if cluster_radius_m is None or cluster_radius_m <= 0:
        return len(positions_m)

    clusters = 0
    last = None
    for pos in sorted(float(p) for p in positions_m):
        if last is None or pos - last > cluster_radius_m:
            clusters += 1
        last = pos
    return clusters


def _cluster_positions(positions_m, cluster_radius_m=DEFAULT_SIGNAL_CLUSTER_RADIUS_M):
    """
    Cluster projected positions with the same single-link rule as _clustered_count.

    The representative event position is the median projected distance of the
    cluster members. This is stable, robust to multi-node junction mapping, and
    always lies on the directed route geometry after interpolation.
    """
    if not positions_m:
        return []
    ordered = sorted(float(p) for p in positions_m)
    if cluster_radius_m is None or cluster_radius_m <= 0:
        return [[p] for p in ordered]

    clusters, cur, last = [], [], None
    for pos in ordered:
        if last is None or pos - last <= cluster_radius_m:
            cur.append(pos)
        else:
            clusters.append(cur)
            cur = [pos]
        last = pos
    clusters.append(cur)
    return clusters


def _pair_signal_positions(a_ll, b_ll, signals, snap_radius_m):
    """
    Project matched signal nodes to positions along the directed pair A->B.

    A signal counts if it is within snap_radius_m of the A-B line AND its
    projection t is in [0, 1). The half-open interval assigns a signal sitting
    exactly at a shared stop to the *downstream* pair only (t=0 there, t=1 on the
    upstream pair), so a signal at a stop is never double-counted along a route.
    """
    alat, alon = a_ll
    blat, blon = b_ll
    length_m = _haversine_m(alat, alon, blat, blon)
    positions = []
    for signal in signals:
        slat, slon = _signal_lat_lon(signal)
        d, t = _project_point_to_segment(slat, slon, alat, alon, blat, blon)
        if d <= snap_radius_m and 0.0 <= t < 1.0:
            positions.append(max(t, 0.0) * length_m)
    return positions


def _pair_signal_count(a_ll, b_ll, signals, snap_radius_m,
                       cluster_radius_m=DEFAULT_SIGNAL_CLUSTER_RADIUS_M):
    positions = _pair_signal_positions(a_ll, b_ll, signals, snap_radius_m)
    return _clustered_count(positions, cluster_radius_m)


def _point_to_polyline(plat, plon, poly):
    """
    Nearest-approach of P to a polyline [(lat,lon), ...].

    Returns (min_dist_m, along_m, total_len_m): the smallest perpendicular
    distance to any sub-segment, the arc-length position of that nearest foot
    from the polyline start, and the polyline's total length.
    """
    if len(poly) < 2:
        return float("inf"), 0.0, 0.0
    best_d, best_along, acc = float("inf"), 0.0, 0.0
    for (alat, alon), (blat, blon) in zip(poly[:-1], poly[1:]):
        d, t = _project_point_to_segment(plat, plon, alat, alon, blat, blon)
        seg_len = _haversine_m(alat, alon, blat, blon)
        tc = 0.0 if t < 0 else (1.0 if t > 1 else t)
        if d < best_d:
            best_d, best_along = d, acc + tc * seg_len
        acc += seg_len
    return best_d, best_along, acc


def _polyline_signal_positions(poly, signals, snap_radius_m):
    """
    Project matched signal nodes to positions along a directed sub-path polyline.

    A signal matches if it is within snap_radius_m of the polyline AND the arc
    position of its nearest foot is in [0, total): the
    half-open interval assigns a signal at a shared stop to the downstream
    sub-path only (arc 0 there, arc == total on the upstream one), so it is not
    double-counted along the route. This is the polyline generalisation of the
    straight-line rule; a 2-point polyline reproduces the chord behaviour.
    """
    if len(poly) < 2:
        return 0
    positions = []
    for signal in signals:
        slat, slon = _signal_lat_lon(signal)
        d, along, total = _point_to_polyline(slat, slon, poly)
        if d <= snap_radius_m and total > 0.0 and 0.0 <= along < total:
            positions.append(along)
    return positions


def _polyline_signal_count(poly, signals, snap_radius_m,
                           cluster_radius_m=DEFAULT_SIGNAL_CLUSTER_RADIUS_M):
    positions = _polyline_signal_positions(poly, signals, snap_radius_m)
    return _clustered_count(positions, cluster_radius_m)


def _count_with_relaxed_snap(primary_positions, relaxed_positions,
                             cluster_radius_m):
    """
    Count with a strict first pass and a wider rescue pass only for zero-count
    pairs. This preserves city-centre precision while reducing suburban false
    zeros caused by wide junctions or GTFS/OSM centreline offsets.
    """
    n_primary = _clustered_count(primary_positions, cluster_radius_m)
    if n_primary > 0:
        return n_primary, False
    n_relaxed = _clustered_count(relaxed_positions, cluster_radius_m)
    return n_relaxed, n_relaxed > 0


def _interp_on_polyline(poly, along_m):
    """Route coordinate at `along_m` metres along [(lat, lon), ...]."""
    if not poly:
        return None
    if len(poly) == 1 or along_m <= 0:
        return poly[0]
    acc = 0.0
    for (lat0, lon0), (lat1, lon1) in zip(poly[:-1], poly[1:]):
        seg_len = _haversine_m(lat0, lon0, lat1, lon1)
        if acc + seg_len >= along_m and seg_len > 0:
            f = (along_m - acc) / seg_len
            return (lat0 + f * (lat1 - lat0), lon0 + f * (lon1 - lon0))
        acc += seg_len
    return poly[-1]


def _median(values):
    vals = sorted(float(v) for v in values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _signal_matches_on_polyline(poly, signals, snap_radius_m):
    """
    Matched raw signals with projected along-route distance and snap distance.

    Keeps the existing half-open [0, total) ownership rule at segment boundaries.
    """
    if len(poly) < 2:
        return []
    matches = []
    for signal in signals:
        rec = _signal_record(signal)
        d, along, total = _point_to_polyline(rec["lat"], rec["lon"], poly)
        if d <= snap_radius_m and total > 0.0 and 0.0 <= along < total:
            matches.append({**rec, "along_m": float(along),
                            "snap_distance_m": float(d)})
    return matches


def _event_id(segment_geometry_id, distance_along_m, raw_ids, raw_coords):
    raw = "|".join(sorted(str(x) for x in raw_ids if str(x)))
    if not raw:
        raw = "|".join(f"{lat:.7f},{lon:.7f}" for lat, lon in raw_coords)
    key = f"{segment_geometry_id}|{distance_along_m:.3f}|{raw}"
    return "sig_" + hashlib.blake2s(key.encode("utf-8"), digest_size=10).hexdigest()


def _cluster_signal_events(segment_geometry_id, from_id, to_id, shape_id, poly,
                           signals, source, snap_radius_m,
                           relaxed_snap_radius_m, cluster_radius_m, fetched_utc):
    """
    Convert matched raw OSM points to one event per along-route cluster.

    Cluster representative distance is the median member projection. signal_lat
    and signal_lon are the route-projected coordinates at that median distance;
    raw_signal_lat/lon retain the raw OSM point nearest to the representative.
    """
    matches = _signal_matches_on_polyline(
        poly, signals, relaxed_snap_radius_m if source == "osm_relaxed" else snap_radius_m
    )
    clusters = _cluster_positions([m["along_m"] for m in matches], cluster_radius_m)
    events = []
    total_len = _point_to_polyline(poly[0][0], poly[0][1], poly)[2]
    for cluster in clusters:
        rep_along = _median(cluster)
        members = [m for m in matches if m["along_m"] in set(cluster)]
        members.sort(key=lambda m: (abs(m["along_m"] - rep_along),
                                    m.get("osm_id", ""), m["lat"], m["lon"]))
        rep_raw = members[0]
        route_ll = _interp_on_polyline(poly, rep_along)
        raw_ids = [m.get("osm_id", "") for m in members if m.get("osm_id", "")]
        raw_coords = [(m["lat"], m["lon"]) for m in members]
        event = {
            "signal_event_id": _event_id(segment_geometry_id, rep_along,
                                         raw_ids, raw_coords),
            "segment_geometry_id": segment_geometry_id,
            "from_stop_id": str(from_id),
            "to_stop_id": str(to_id),
            "shape_id": str(shape_id or ""),
            "signal_lat": float(route_ll[0]),
            "signal_lon": float(route_ll[1]),
            "distance_along_segment_m": float(rep_along),
            "segment_length_m": float(total_len),
            "source": source,
            "snap_radius_m": float(snap_radius_m),
            "relaxed_snap_radius_m": float(relaxed_snap_radius_m),
            "cluster_radius_m": float(cluster_radius_m),
            "fetched_utc": fetched_utc,
            "raw_osm_ids": "|".join(sorted(str(x) for x in raw_ids)),
            "cluster_size": len(members),
            "snap_distance_m": float(rep_raw["snap_distance_m"]),
            "raw_signal_lat": float(rep_raw["lat"]),
            "raw_signal_lon": float(rep_raw["lon"]),
            "signal_type": rep_raw.get("signal_type", ""),
            "osm_tags": json.dumps(rep_raw.get("tags") or {}, sort_keys=True),
        }
        if not (0.0 <= event["distance_along_segment_m"] < total_len):
            raise ValueError("signal event position is outside segment bounds")
        events.append(event)
    events.sort(key=lambda e: e["distance_along_segment_m"])
    return events


def _interp_on_shape(shape_points, cum):
    """(lat, lon) at arc position `cum` along a shape [(lat,lon,cum), ...]."""
    if cum <= shape_points[0][2]:
        return (shape_points[0][0], shape_points[0][1])
    if cum >= shape_points[-1][2]:
        return (shape_points[-1][0], shape_points[-1][1])
    for (lat0, lon0, c0), (lat1, lon1, c1) in zip(shape_points[:-1], shape_points[1:]):
        if c0 <= cum <= c1 and c1 > c0:
            f = (cum - c0) / (c1 - c0)
            return (lat0 + f * (lat1 - lat0), lon0 + f * (lon1 - lon0))
    return (shape_points[-1][0], shape_points[-1][1])


def _shape_subpath(shape_points, cum_a, cum_b):
    """The driven-path vertices from arc position cum_a to cum_b, endpoints
       interpolated so the polyline starts and ends at the stops' projections."""
    if not shape_points or cum_b <= cum_a:
        return None
    pts = [_interp_on_shape(shape_points, cum_a)]
    for lat, lon, cum in shape_points:
        if cum_a < cum < cum_b:
            pts.append((lat, lon))
    pts.append(_interp_on_shape(shape_points, cum_b))
    out = [pts[0]]
    for p in pts[1:]:
        if p != out[-1]:
            out.append(p)
    return out if len(out) >= 2 else None


def _bbox_for_pairs(pairs, coords, pad_m, geom_map=None):
    lats, lons = [], []
    for pair in pairs:
        gid, f, t, _l, _shape_id = _pair_fields(pair)
        for sid in (f, t):
            la, lo = coords[sid]
            lats.append(la)
            lons.append(lo)
        if geom_map:                         # include shape sub-path vertices too
            for la, lo in _geom_for_pair(geom_map, gid, f, t) or []:
                lats.append(la)
                lons.append(lo)
    if not lats:
        return None
    dlat = pad_m / 111_320.0
    dlon = pad_m / (111_320.0 * max(cos(radians(sum(lats) / len(lats))), 1e-6))
    return (max(lats) + dlat, min(lats) - dlat,
            max(lons) + dlon, min(lons) - dlon)      # north, south, east, west


# -----------------------------------------------------------------------------
# OSM fetch (the only networked part; version-tolerant across OSMnx 1.x / 2.x)
# -----------------------------------------------------------------------------
def _looks_like_empty_osm_response(exc):
    text = str(exc).lower()
    name = exc.__class__.__name__.lower()
    return (
        "insufficientresponse" in name
        or "no data elements" in text
        or "no matching features" in text
        or "there are no geometries" in text
    )


def _features_from_bbox(ox, north, south, east, west, tags):
    # OSMnx 2.x: features_from_bbox(bbox=(left, bottom, right, top))
    errs = []
    empty_seen = False
    for call in (
        lambda: ox.features_from_bbox(bbox=(west, south, east, north), tags=tags),
        lambda: ox.features_from_bbox(north, south, east, west, tags=tags),   # 1.x
        lambda: ox.geometries_from_bbox(north, south, east, west, tags=tags), # <1.5
    ):
        try:
            return call()
        except Exception as e:                       # try the next signature
            if _looks_like_empty_osm_response(e):
                empty_seen = True
            errs.append(repr(e))
    if empty_seen:
        return None
    raise RuntimeError(" | ".join(errs))


def _points_from_features(gdf, signal_type=""):
    if gdf is None:
        return []
    if getattr(gdf, "empty", False) or not hasattr(gdf, "geometry"):
        return []

    pts = []
    try:
        iterator = gdf.iterrows()
    except AttributeError:
        iterator = enumerate(gdf.geometry)
    for idx, row in iterator:
        geom = row.geometry if hasattr(row, "geometry") else row
        if geom is None:
            continue
        p = geom if getattr(geom, "geom_type", "") == "Point" else geom.centroid
        if getattr(p, "is_empty", False):
            continue
        tags = {}
        if hasattr(row, "items"):
            for key, value in row.items():
                if key == "geometry":
                    continue
                if value is not None and str(value) != "nan":
                    tags[str(key)] = str(value)
        osm_id = ""
        if isinstance(idx, tuple):
            osm_id = ":".join(str(part) for part in idx)
        elif idx is not None:
            osm_id = str(idx)
        pts.append({
            "lat": float(p.y),
            "lon": float(p.x),
            "osm_id": osm_id,
            "signal_type": signal_type,
            "tags": tags,
        })
    return pts


def _dedupe_signal_points(points, tolerance_m=OSM_SIGNAL_DEDUP_RADIUS_M):
    """
    Merge duplicate OSM features returned by overlapping signal tag queries.

    A pedestrian crossing can be mapped with both crossing=traffic_signals and
    crossing_ref=pelican/puffin/toucan. We only want one physical signalised
    point before projecting onto stop-to-stop segments.
    """
    unique = []
    for lat, lon in points:
        if all(_haversine_m(lat, lon, ulat, ulon) > tolerance_m
               for ulat, ulon in unique):
            unique.append((lat, lon))
    return unique


def _dedupe_signal_records(points, tolerance_m=OSM_SIGNAL_DEDUP_RADIUS_M):
    """
    Metadata-preserving twin of _dedupe_signal_points.

    The first record wins inside the tolerance, matching the old coordinate-only
    dedupe behaviour while retaining OSM ids/tags where the fetch exposed them.
    """
    unique = []
    for signal in points:
        rec = _signal_record(signal)
        if all(_haversine_m(rec["lat"], rec["lon"], u["lat"], u["lon"]) > tolerance_m
               for u in unique):
            unique.append(rec)
    return unique


def fetch_traffic_signals_bbox(north, south, east, west, verbose=True):
    """
    Return [(lat, lon), ...] for OSM signalised features in the bbox.

    This includes classic highway=traffic_signals nodes and signal-controlled
    pedestrian crossings such as crossing=traffic_signals, crossing:signals=yes,
    and common UK/Ireland crossing_ref values (pelican, puffin, toucan).

    Returns [] when the bbox genuinely has no signals. Raises RuntimeError when
    OSMnx is missing or the query fails, so the caller can fall back cleanly
    (an empty result and an infrastructure failure must not look the same).
    """
    try:
        import osmnx as ox
    except ImportError as e:
        raise RuntimeError("OSMnx not installed (`pip install osmnx`)") from e

    pts = []
    query_counts = {}
    errs = []
    n_success = 0
    for label, tags in OSM_SIGNAL_TAG_QUERIES:
        try:
            q_pts = _points_from_features(
                _features_from_bbox(ox, north, south, east, west, tags),
                signal_type=label,
            )
            n_success += 1
            query_counts[label] = len(q_pts)
            pts.extend(q_pts)
        except Exception as e:
            errs.append(f"{label}: {e}")
    if n_success == 0:
        raise RuntimeError("OSM signal query failed: " + " | ".join(errs))

    raw_n = len(pts)
    pts = _dedupe_signal_records(pts)
    if verbose:
        nonzero = ", ".join(f"{k}={v}" for k, v in query_counts.items() if v)
        suffix = f" ({nonzero})" if nonzero else ""
        print(f"  signals: fetched {len(pts)} unique OSM signal/crossing features "
              f"in bbox from {raw_n} raw feature(s){suffix}")
    return pts


# -----------------------------------------------------------------------------
# Cache CSV I/O
# -----------------------------------------------------------------------------
def load_signal_cache(path):
    """Return {segment_geometry_id: row_dict}; legacy files keep (from,to) keys."""
    path = Path(path)
    cache = {}
    if not path.exists():
        return cache
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            f, t = str(row["from_stop_id"]), str(row["to_stop_id"])
            gid = row.get("segment_geometry_id") or ""
            key = gid if gid else (f, t)
            if key in cache:
                raise ValueError(f"duplicate signal cache row for stop pair {key}")
            row["segment_geometry_id"] = gid
            row["shape_id"] = row.get("shape_id") or ""
            row["n_signals"] = int(float(row["n_signals"]))
            if row["n_signals"] < 0:
                raise ValueError(f"signal count must be non-negative for stop pair {key}")
            if row.get("source") not in {"osm", "osm_relaxed", "fallback"}:
                raise ValueError(f"invalid signal cache source for stop pair {key}")
            row["snap_radius_m"] = float(row["snap_radius_m"])
            if row["snap_radius_m"] <= 0:
                raise ValueError(f"snap_radius_m must be positive for stop pair {key}")
            row["relaxed_snap_radius_m"] = float(
                row.get("relaxed_snap_radius_m") or 0.0
            )
            row["cluster_radius_m"] = float(row.get("cluster_radius_m") or 0.0)
            row["length_m"] = float(row.get("length_m") or 0.0)
            if row["length_m"] < 0:
                raise ValueError(f"length_m must be non-negative for stop pair {key}")
            cache[key] = row
    return cache


def save_signal_cache(path, cache):
    """Write the merged summary cache back to CSV (one row per geometry)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SIGNAL_CACHE_COLUMNS)
        w.writeheader()
        for key, row in sorted(cache.items(), key=lambda item: str(item[0])):
            f = str(row.get("from_stop_id") or (key[0] if isinstance(key, tuple) else ""))
            t = str(row.get("to_stop_id") or (key[1] if isinstance(key, tuple) else ""))
            gid = row.get("segment_geometry_id") or (
                "" if isinstance(key, tuple) else str(key)
            )
            w.writerow({
                "segment_geometry_id": gid,
                "shape_id": row.get("shape_id", ""),
                "from_stop_id": f, "to_stop_id": t,
                "n_signals": int(row["n_signals"]), "source": row["source"],
                "snap_radius_m": row["snap_radius_m"],
                "relaxed_snap_radius_m": row.get("relaxed_snap_radius_m", 0.0),
                "cluster_radius_m": row.get("cluster_radius_m", 0.0),
                "length_m": round(float(row.get("length_m") or 0.0), 1),
                "fetched_utc": row.get("fetched_utc", ""),
            })
    return path


def load_signal_event_cache(path):
    """Return {segment_geometry_id: [event_row, ...]}. Empty if absent."""
    path = Path(path)
    events = {}
    if not path.exists():
        return events
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            gid = str(row["segment_geometry_id"])
            row["signal_lat"] = float(row["signal_lat"])
            row["signal_lon"] = float(row["signal_lon"])
            row["distance_along_segment_m"] = float(row["distance_along_segment_m"])
            row["segment_length_m"] = float(row["segment_length_m"])
            row["snap_radius_m"] = float(row["snap_radius_m"])
            row["relaxed_snap_radius_m"] = float(row.get("relaxed_snap_radius_m") or 0.0)
            row["cluster_radius_m"] = float(row.get("cluster_radius_m") or 0.0)
            row["cluster_size"] = int(float(row.get("cluster_size") or 1))
            row["snap_distance_m"] = float(row.get("snap_distance_m") or 0.0)
            row["raw_signal_lat"] = float(row.get("raw_signal_lat") or row["signal_lat"])
            row["raw_signal_lon"] = float(row.get("raw_signal_lon") or row["signal_lon"])
            if not (0.0 <= row["distance_along_segment_m"] < row["segment_length_m"]):
                raise ValueError(
                    f"signal event {row['signal_event_id']} is outside segment bounds"
                )
            events.setdefault(gid, []).append(row)
    for rows in events.values():
        rows.sort(key=lambda r: r["distance_along_segment_m"])
    return events


def save_signal_event_cache(path, event_cache):
    """Write one row per clustered, route-projected signal event."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if isinstance(event_cache, dict):
        for group in event_cache.values():
            rows.extend(group)
    else:
        rows = list(event_cache)
    rows.sort(key=lambda r: (
        str(r.get("segment_geometry_id", "")),
        float(r.get("distance_along_segment_m", 0.0)),
        str(r.get("signal_event_id", "")),
    ))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SIGNAL_EVENT_CACHE_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({col: row.get(col, "") for col in SIGNAL_EVENT_CACHE_COLUMNS})
    return path


def _event_consistency_check(summary_cache, event_cache):
    for key, row in summary_cache.items():
        if row.get("source") == "fallback":
            continue
        gid = row.get("segment_geometry_id") or (
            "" if isinstance(key, tuple) else str(key)
        )
        if not gid or gid not in event_cache:
            continue
        n_events = len(event_cache.get(gid, []))
        if int(row["n_signals"]) != n_events:
            raise ValueError(
                f"signal count/event mismatch for {gid}: "
                f"n_signals={row['n_signals']} event_rows={n_events}"
            )


def _raw_equivalent(e1, e2):
    ids1 = {x for x in str(e1.get("raw_osm_ids", "")).split("|") if x}
    ids2 = {x for x in str(e2.get("raw_osm_ids", "")).split("|") if x}
    if ids1 and ids2 and ids1 & ids2:
        return True
    return _haversine_m(
        float(e1.get("raw_signal_lat", e1["signal_lat"])),
        float(e1.get("raw_signal_lon", e1["signal_lon"])),
        float(e2.get("raw_signal_lat", e2["signal_lat"])),
        float(e2.get("raw_signal_lon", e2["signal_lon"])),
    ) <= OSM_SIGNAL_DEDUP_RADIUS_M


def _adjacent_raw_duplicate_count(event_cache):
    rows = []
    for group in event_cache.values():
        rows.extend(group)
    count = 0
    for i, e1 in enumerate(rows):
        for e2 in rows[i + 1:]:
            adjacent = (
                e1.get("to_stop_id") == e2.get("from_stop_id")
                or e2.get("to_stop_id") == e1.get("from_stop_id")
            )
            if adjacent and _raw_equivalent(e1, e2):
                count += 1
    return count


# -----------------------------------------------------------------------------
# Cache-aware resolver
# -----------------------------------------------------------------------------
def resolve_signal_counts(pairs, coords, snap_radius_m=DEFAULT_SNAP_RADIUS_M,
                          relaxed_snap_radius_m=DEFAULT_RELAXED_SNAP_RADIUS_M,
                          cluster_radius_m=DEFAULT_SIGNAL_CLUSTER_RADIUS_M,
                          fallback_per_km=DEFAULT_FALLBACK_PER_KM,
                          fetch_fn=fetch_traffic_signals_bbox,
                          cache=None, cache_path=None, refresh=False,
                          bbox_pad_m=150.0, geom_map=None,
                          event_cache=None, event_cache_path=None,
                          verbose=True):
    """
    Resolve n_signals for every (from_id, to_id, length_m) pair.

    Order of resolution per pair:
      1. cache hit  (same pair, same snap/relaxed/cluster radius, unless refresh=True)
      2. OSM        (one bbox fetch covering all misses, counts computed per pair)
      3. fallback   (fallback_per_km * length_km, only if OSM is unavailable)

    geom_map ({(from_id,to_id): [(lat,lon), ...]}) carries the shape.txt driven
    path for each pair (from route_stop_pairs). When present, signals are snapped
    to that path -- the same geometry used for segment length -- rather than to
    the straight chord between the two stops. Chord is the fallback only when a
    pair has no shape geometry.

    Newly resolved pairs are merged into `cache`; if `cache_path` is given the
    merged cache is written back. Returns (count_map, source_map, stats).
    """
    if cache is None:
        cache = load_signal_cache(cache_path) if cache_path else {}
    if event_cache is None:
        if event_cache_path is None and cache_path is not None:
            event_cache_path = _signal_event_cache_path(cache_path)
        event_cache = load_signal_event_cache(event_cache_path) if event_cache_path else {}

    # de-duplicate pairs, remembering a length for the fallback estimate
    uniq = {}
    for pair in pairs:
        gid, f, t, length_m, shape_id = _pair_fields(pair)
        uniq[gid] = (f, t, float(length_m), shape_id)

    count_map, source_map = {}, {}
    misses = []
    n_hit = 0
    for gid, (f, t, length_m, shape_id) in uniq.items():
        legacy_key = (f, t)
        row_key = gid if gid in cache else legacy_key
        row = None if refresh else cache.get(row_key)
        if (
            row is not None
            and abs(row["snap_radius_m"] - snap_radius_m) < 1e-6
            and abs(row.get("relaxed_snap_radius_m", 0.0) - relaxed_snap_radius_m) < 1e-6
            and abs(row.get("cluster_radius_m", 0.0) - cluster_radius_m) < 1e-6
        ):
            n = int(row["n_signals"])
            if row.get("source") != "fallback" and isinstance(gid, str):
                events = event_cache.get(gid)
                if events is not None:
                    n = len(events)
                    if int(row["n_signals"]) != n:
                        raise ValueError(
                            f"signal count/event mismatch for {gid}: "
                            f"n_signals={row['n_signals']} event_rows={n}"
                        )
            count_map[gid] = n
            source_map[gid] = row["source"]
            if isinstance(gid, tuple):
                count_map[legacy_key] = n
                source_map[legacy_key] = row["source"]
            n_hit += 1
        else:
            misses.append((gid, f, t, length_m, shape_id))

    n_osm = n_osm_relaxed = n_fallback = osm_nodes = 0
    if misses:
        signals = None
        if fetch_fn is not None:
            bbox = _bbox_for_pairs(misses, coords, bbox_pad_m, geom_map=geom_map)
            try:
                signals = fetch_fn(*bbox, verbose=verbose)
                osm_nodes = len(signals)
            except Exception as e:
                if verbose:
                    print(f"  signals: OSM fetch failed ({e}); using fallback "
                          f"{fallback_per_km:.1f}/km for {len(misses)} pair(s)")
                signals = None

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for gid, f, t, length_m, shape_id in misses:
            legacy_key = (f, t)
            if signals is not None:
                poly = _geom_for_pair(geom_map, gid, f, t)
                if poly and len(poly) >= 2:
                    primary_positions = _polyline_signal_positions(
                        poly, signals, snap_radius_m
                    )
                    relaxed_positions = _polyline_signal_positions(
                        poly, signals, relaxed_snap_radius_m
                    )
                else:
                    primary_positions = _pair_signal_positions(
                        coords[f], coords[t], signals, snap_radius_m
                    )
                    relaxed_positions = _pair_signal_positions(
                        coords[f], coords[t], signals, relaxed_snap_radius_m
                    )
                n, used_relaxed = _count_with_relaxed_snap(
                    primary_positions, relaxed_positions, cluster_radius_m
                )
                source = "osm_relaxed" if used_relaxed else "osm"
                if not poly:
                    poly = [coords[f], coords[t]]
                if isinstance(gid, tuple):
                    event_gid = _segment_geometry_id(f, t, shape_id=shape_id, poly=poly)
                else:
                    event_gid = gid
                events = _cluster_signal_events(
                    event_gid, f, t, shape_id, poly, signals, source,
                    snap_radius_m, relaxed_snap_radius_m, cluster_radius_m, now
                )
                n = len(events)
                event_cache[event_gid] = events
                if used_relaxed:
                    n_osm_relaxed += 1
                else:
                    n_osm += 1
            else:
                n = int(round(fallback_per_km * length_m / 1000.0))
                source = "fallback"
                n_fallback += 1
                event_gid = gid
            count_map[gid] = n
            source_map[gid] = source
            if isinstance(gid, tuple):
                count_map[legacy_key] = n
                source_map[legacy_key] = source
            cache_key = legacy_key if isinstance(gid, tuple) else event_gid
            cache[cache_key] = {
                "segment_geometry_id": "" if isinstance(event_gid, tuple) else event_gid,
                "shape_id": shape_id,
                "from_stop_id": f,
                "to_stop_id": t,
                "n_signals": n,
                "source": source,
                "snap_radius_m": float(snap_radius_m),
                "relaxed_snap_radius_m": float(relaxed_snap_radius_m),
                "cluster_radius_m": float(cluster_radius_m),
                "length_m": length_m,
                "fetched_utc": now,
            }

        _event_consistency_check(cache, event_cache)
        if cache_path is not None:
            save_signal_cache(cache_path, cache)
        if event_cache_path is not None:
            save_signal_event_cache(event_cache_path, event_cache)

    current_event_cache = {
        gid: event_cache.get(gid, [])
        for gid in uniq
        if isinstance(gid, str)
    }
    adjacent_dupes = _adjacent_raw_duplicate_count(current_event_cache)
    if verbose and adjacent_dupes:
        print(f"  signals: {adjacent_dupes} adjacent raw-equivalent signal "
              "assignment(s) retained for diagnostics")

    stats = {"n_pairs": len(uniq), "cache_hit": n_hit,
             "osm": n_osm, "osm_relaxed": n_osm_relaxed,
             "fallback": n_fallback,
             "osm_nodes": osm_nodes, "total_signals": sum(count_map.values()),
             "event_rows": sum(len(v) for v in current_event_cache.values()),
             "adjacent_raw_signal_duplicates": adjacent_dupes}
    return count_map, source_map, stats


# -----------------------------------------------------------------------------
# Stop-pair alignment (reuse gtfs pairing; soft import with a local fallback)
# -----------------------------------------------------------------------------
def _stop_pairs(trip_rows, stops, shape_points=None):
    try:
        from gtfs_to_segment import iter_valid_stop_pairs
        return list(iter_valid_stop_pairs(trip_rows, stops, shape_points=shape_points))
    except Exception:
        coord = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
        pairs = []
        for a, b in zip(trip_rows[:-1], trip_rows[1:]):
            sa, sb = coord[a["stop_id"]], coord[b["stop_id"]]
            d = _haversine_m(sa["stop_lat"], sa["stop_lon"],
                             sb["stop_lat"], sb["stop_lon"])
            if d >= 1.0:
                pairs.append((a, b, d))
        return pairs


def _pair_geometries(trip_rows, stops, shape_points=None):
    """
    Like iter_valid_stop_pairs, but also returns the driven-path polyline for
    each pair: yields (from_id, to_id, length_m, polyline). Mirrors gtfs's
    co-located-stop filter exactly so pairs stay aligned with the segments.
    polyline is the shape.txt sub-path between the two stops (same geometry used
    for segment length); it falls back to the [A, B] chord when no shape exists.
    """
    coord = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")

    shape_pos = None
    if shape_points:
        try:
            from gtfs_to_segment import _shape_stop_positions
            shape_pos = _shape_stop_positions(trip_rows, stops, shape_points)
        except Exception:
            shape_pos = None

    out = []
    for i, (a, b) in enumerate(zip(trip_rows[:-1], trip_rows[1:])):
        length_m, poly = None, None
        if shape_pos:
            start_m, end_m = shape_pos[i], shape_pos[i + 1]
            if start_m is not None and end_m is not None and end_m > start_m:
                length_m = end_m - start_m
                poly = _shape_subpath(shape_points, start_m, end_m)
        if length_m is None:
            sa, sb = coord[a["stop_id"]], coord[b["stop_id"]]
            length_m = _haversine_m(sa["stop_lat"], sa["stop_lon"],
                                    sb["stop_lat"], sb["stop_lon"])
        if poly is None:
            sa, sb = coord[a["stop_id"]], coord[b["stop_id"]]
            poly = [(sa["stop_lat"], sa["stop_lon"]), (sb["stop_lat"], sb["stop_lon"])]
        if length_m < 1:
            continue
        out.append((a, b, length_m, poly))
    return out


def route_stop_pairs(trip_rows_iter, stops, shape_points_by_trip=None,
                     shape_id_by_trip=None, include_geometry_id=False):
    """
    Collect the unique directed pairs a route uses across several trips, with the
    shape sub-path for each. trip_rows_iter yields (trip_id, trip_rows). Returns
    (pairs, geom_map): pairs is [(from_id, to_id, length_m), ...] and geom_map is
    {(from_id, to_id): polyline}. OSM is then queried for the whole route at once
    and each pair is counted against its own driven path.

    With include_geometry_id=True, pairs become
    [(segment_geometry_id, from_id, to_id, length_m, shape_id), ...] and geom_map
    is keyed by segment_geometry_id. This is used by the signal event cache so
    the same directed stop pair on different GTFS shape variants does not share
    spatial events.
    """
    pairs, geom_map = {}, {}
    for tid, rows in trip_rows_iter:
        sp = None if shape_points_by_trip is None else shape_points_by_trip.get(tid)
        shape_id = "" if shape_id_by_trip is None else shape_id_by_trip.get(str(tid), "")
        for a, b, length_m, poly in _pair_geometries(rows, stops, shape_points=sp):
            f, t = a["stop_id"], b["stop_id"]
            if include_geometry_id:
                key = _segment_geometry_id(f, t, shape_id=shape_id, poly=poly)
            else:
                key = (f, t)
            if key not in pairs:
                pairs[key] = (f, t, length_m, shape_id)
                geom_map[key] = poly
    if include_geometry_id:
        return [
            (gid, f, t, l, shape_id)
            for gid, (f, t, l, shape_id) in pairs.items()
        ], geom_map
    return [(f, t, l) for _key, (f, t, l, _shape_id) in pairs.items()], geom_map


# -----------------------------------------------------------------------------
# Apply: set Segment.n_signals (+ Segment.signal_source)
# -----------------------------------------------------------------------------
def add_traffic_signals(segments, trip_rows, stops, count_map=None,
                        source_map=None, signals=None,
                        snap_radius_m=DEFAULT_SNAP_RADIUS_M,
                        relaxed_snap_radius_m=DEFAULT_RELAXED_SNAP_RADIUS_M,
                        cluster_radius_m=DEFAULT_SIGNAL_CLUSTER_RADIUS_M,
                        shape_points=None, shape_id=None, verbose=True):
    """
    Write Segment.n_signals (and Segment.signal_source) for one trip.

    Two modes:
      * count_map given  -> pure lookup by segment_geometry_id when present,
                            falling back to the legacy (from_stop_id, to_stop_id)
                            key for old callers/caches.
      * signals given    -> compute per-pair counts directly from OSM points
                            (handy for a one-off manual check of a single trip).
    Mirrors add_grades_from_dem: walks the same iter_valid_stop_pairs so seg i
    lines up with pair i. Returns the segment list (mutated in place).
    """
    pairs = _stop_pairs(trip_rows, stops, shape_points=shape_points)
    if len(pairs) != len(segments):
        raise ValueError(
            f"segment/pair mismatch ({len(segments)} vs {len(pairs)}); "
            "build_segments and add_traffic_signals are out of sync.")
    if not segments:
        return segments

    if count_map is None:
        if signals is None:
            raise ValueError("add_traffic_signals needs count_map or signals.")
        count_map, source_map = {}, {}
        for a, b, _l, poly in _pair_geometries(trip_rows, stops,
                                               shape_points=shape_points):
            f, t = a["stop_id"], b["stop_id"]
            key = _segment_geometry_id(f, t, shape_id=shape_id, poly=poly)
            primary_positions = _polyline_signal_positions(poly, signals, snap_radius_m)
            relaxed_positions = _polyline_signal_positions(
                poly, signals, relaxed_snap_radius_m
            )
            n, used_relaxed = _count_with_relaxed_snap(
                primary_positions, relaxed_positions, cluster_radius_m
            )
            count_map[key] = n
            source_map[key] = "osm_relaxed" if used_relaxed else "osm"

    total = 0
    for seg, (a, b, _l), (_ga, _gb, _gl, poly) in zip(
        segments, pairs, _pair_geometries(trip_rows, stops, shape_points=shape_points)
    ):
        legacy_key = (a["stop_id"], b["stop_id"])
        geom_key = _segment_geometry_id(
            a["stop_id"], b["stop_id"], shape_id=shape_id, poly=poly
        )
        key = geom_key if geom_key in count_map else legacy_key
        seg.n_signals = int(count_map.get(key, 0))
        seg.signal_source = (source_map or {}).get(key, "none")
        total += seg.n_signals

    if verbose:
        with_sig = sum(1 for s in segments if getattr(s, "n_signals", 0) > 0)
        print(f"  signals: {total} on {with_sig}/{len(segments)} segments "
              f"(snap {snap_radius_m:.0f} m)")
    return segments


# -----------------------------------------------------------------------------
# Self-test: geometry + cache round-trip, no network, no GTFS
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, os

    # four stops in a line ~400 m apart -> three directed pairs
    coords = {
        "A": (51.9000, -8.4700), "B": (51.9036, -8.4700),
        "C": (51.9072, -8.4700), "D": (51.9108, -8.4700),
    }
    pairs = [("A", "B", 400.0), ("B", "C", 400.0), ("C", "D", 400.0)]
    signals = [
        (51.9018, -8.4700),   # mid A-B  -> AB
        (51.9036, -8.4700),   # exactly at B -> downstream (BC), not AB
        (51.9054, -8.4700),   # mid B-C  -> BC
        (51.9090, -8.4701),   # mid C-D (8 m off) -> CD
        (51.9090, -8.4730),   # ~200 m off-route -> none
    ]

    # overlapping OSM tag queries should not create duplicate physical signals
    duped = _dedupe_signal_points([
        (51.9018000, -8.4700000),
        (51.9018004, -8.4700004),  # same crossing under another tag
        (51.9054000, -8.4700000),
    ], tolerance_m=2.0)
    assert len(duped) == 2, duped
    print("dedupe: overlapping OSM signal/crossing features -> 2 unique points")

    # no cache yet -> all OSM (pass signals via a stub fetch_fn)
    cache = {}
    cmap, smap, stats = resolve_signal_counts(
        pairs, coords, snap_radius_m=30.0, cache=cache,
        fetch_fn=lambda *b, **k: signals, verbose=False)
    got = [cmap[(f, t)] for f, t, _ in pairs]
    print("counts:", got, "(expect [1, 2, 1])  sources:",
          [smap[(f, t)] for f, t, _ in pairs])
    # signal exactly at stop B is assigned once, to downstream B->C (t=0), not
    # A->B (t=1): B->C has the at-B signal + the mid-BC signal = 2; total 4, so
    # no double-count and the off-route signal is excluded.
    assert got == [1, 2, 1], got
    assert stats["total_signals"] == 4
    assert stats["osm"] == 3 and stats["cache_hit"] == 0

    # Suburban rescue: a wider junction can place the mapped signal node outside
    # the strict 30 m route corridor, while still being part of the bus movement.
    relaxed_cache = {}
    relaxed_signals = [(51.9018, -8.47066)]  # roughly 45 m east of A-B
    cmap_r, smap_r, stats_r = resolve_signal_counts(
        [("A", "B", 400.0)], coords, snap_radius_m=30.0,
        relaxed_snap_radius_m=60.0, cache=relaxed_cache,
        fetch_fn=lambda *b, **k: relaxed_signals, verbose=False)
    assert cmap_r[("A", "B")] == 1
    assert smap_r[("A", "B")] == "osm_relaxed"
    assert stats_r["osm_relaxed"] == 1
    print("relaxed snap: strict 30 m misses, relaxed 60 m -> 1 signal")

    # second call -> all cache hits, fetch_fn must NOT be called
    def _boom(*a, **k):
        raise AssertionError("fetch called on a cache hit")
    cmap2, smap2, stats2 = resolve_signal_counts(
        pairs, coords, snap_radius_m=30.0, cache=cache, fetch_fn=_boom, verbose=False)
    assert stats2["cache_hit"] == 3 and stats2["osm"] == 0
    print("second run: all", stats2["cache_hit"], "pairs from cache (no fetch)")

    # OSM unavailable -> fallback by per-km density
    cmap3, smap3, stats3 = resolve_signal_counts(
        [("X", "Y", 1000.0)], {"X": (51.90, -8.47), "Y": (51.91, -8.47)},
        snap_radius_m=30.0, cache={}, fallback_per_km=2.0,
        fetch_fn=lambda *b, **k: (_ for _ in ()).throw(RuntimeError("no OSM")),
        verbose=False)
    assert smap3[("X", "Y")] == "fallback" and cmap3[("X", "Y")] == 2
    print("fallback: 1000 m @ 2/km ->", cmap3[("X", "Y")], "signals (source fallback)")

    # cache CSV round-trip
    p = os.path.join(tempfile.mkdtemp(), "signals.csv")
    save_signal_cache(p, cache)
    reloaded = load_signal_cache(p)
    assert reloaded.keys() == cache.keys()
    print("cache CSV round-trip OK:", p)
    print("self-test passed")
