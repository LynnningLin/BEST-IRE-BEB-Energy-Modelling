"""
speed_caps.py
================================================================================
Resolve an independent PHYSICAL speed cap for every stop-to-stop segment from
OSM `maxspeed` tags, mirroring the traffic_signals.py workflow (cache CSV keyed
by directed stop pair -> one OSM fetch per route bbox -> per-km fallback).

WHY THIS MODULE EXISTS
----------------------
The speed-profile model needs a physical/legal speed limit that is INDEPENDENT
of the GTFS schedule. Previously seg.v_cruise_ms (derived FROM the GTFS runtime)
was used as the cap, which made schedule-feasibility checks circular:

    GTFS runtime -> estimated cruise speed -> speed cap -> "is the same GTFS
    runtime physically feasible?"

This module supplies Segment.speed_cap_ms from OSM road data instead, so
beb_soc_model can judge feasibility against the actual road, not the schedule.

AGGREGATION: LENGTH-WEIGHTED HARMONIC MEAN
------------------------------------------
A stop-to-stop segment usually traverses several OSM ways with different limits.
The cap is used for (a) minimum feasible traversal time and (b) bounding the
profile peak speed, and (a) is first-order for the model. The length-weighted
harmonic mean is the unique single-value aggregate that preserves the true
minimum cruise time sum(L_i / v_i):

    cap = total_length / sum(L_i / v_i)

Taking min() instead would let a short 30 km/h zone dominate a mostly-50 km/h
segment and re-inflate infeasibility (a milder rerun of the bug being fixed);
an arithmetic mean would overestimate feasibility. The peak-speed error of the
harmonic mean (profile may marginally exceed the slowest sub-zone) is
second-order for energy.

MISSING TAGS
------------
Explicit maxspeed coverage is patchy in Irish OSM, so untagged ways are imputed
from their highway class (HIGHWAY_CLASS_SPEED_KMH) and flagged source="imputed"
when imputed ways dominate a segment. Zone values like "IE:urban" are handled.

CACHE CSV COLUMNS
-----------------
from_stop_id, to_stop_id, speed_cap_ms, speed_cap_kmh, source, coverage_frac,
snap_radius_m, sample_step_m, length_m, fetched_utc

PUBLIC API (mirrors traffic_signals.py)
---------------------------------------
    resolve_speed_caps : cache hit -> OSM fetch (once) -> fallback.
    add_speed_caps     : post-pass writing seg.speed_cap_ms (+ source).
    load_speed_cap_cache / save_speed_cap_cache : CSV I/O.
    fetch_speed_edges_bbox : the only part that needs the network.
================================================================================
"""

import csv
import time
from math import cos, radians
from pathlib import Path

# Reuse the sibling module's geometry + pairing so segments/pairs stay aligned
# with build_segments exactly the way add_traffic_signals does.
from traffic_signals import (
    _bbox_for_pairs,
    _haversine_m,
    _looks_like_empty_osm_response,
    _pair_geometries,
    _stop_pairs,
)

DEFAULT_SPEED_SNAP_RADIUS_M = 25.0
DEFAULT_SAMPLE_STEP_M = 20.0
DEFAULT_MIN_COVERAGE_FRAC = 0.5
DEFAULT_FALLBACK_CAP_KMH = 50.0      # Irish urban default limit
DEFAULT_MAX_CAP_KMH = 90.0
MIN_CAP_KMH = 8.0                    # clamp floor: below this is an OSM error

SPEED_CAP_CACHE_COLUMNS = [
    "from_stop_id", "to_stop_id", "speed_cap_ms", "speed_cap_kmh", "source",
    "coverage_frac", "snap_radius_m", "sample_step_m", "length_m", "fetched_utc",
]

# Deterministic per-class speeds (km/h) for ways without a usable maxspeed tag.
# Chosen for Irish roads; auditable in a methods section, unlike OSMnx's
# observed-mean imputation which varies with the bbox contents.
HIGHWAY_CLASS_SPEED_KMH = {
    "motorway": 120.0, "motorway_link": 60.0,
    "trunk": 100.0, "trunk_link": 50.0,
    "primary": 80.0, "primary_link": 50.0,
    "secondary": 80.0, "secondary_link": 50.0,
    "tertiary": 60.0, "tertiary_link": 50.0,
    "unclassified": 50.0,
    "residential": 50.0,
    "living_street": 15.0,
    "service": 30.0,
    "busway": 50.0,
    "bus_guideway": 50.0,
}
DEFAULT_CLASS_SPEED_KMH = 50.0

# Zone tags occasionally used instead of a numeric value.
ZONE_SPEED_KMH = {
    "ie:urban": 50.0, "ie:rural": 80.0, "ie:motorway": 120.0,
    "gb:nsl_single": 96.0, "gb:nsl_dual": 112.0, "gb:motorway": 112.0,
    "walk": 10.0, "none": 120.0,
}

_MPH_TO_KMH = 1.609344


# -----------------------------------------------------------------------------
# maxspeed parsing
# -----------------------------------------------------------------------------
def _parse_maxspeed_kmh(value):
    """
    One OSM maxspeed value -> km/h float, or None if unusable.
    Handles numerics ("50"), units ("30 mph"), zone refs ("IE:urban"), and
    lists (OSMnx merges parallel ways) by taking the minimum usable entry.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parsed = [_parse_maxspeed_kmh(v) for v in value]
        parsed = [p for p in parsed if p is not None]
        return min(parsed) if parsed else None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if v > 0 else None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in ZONE_SPEED_KMH:
        return ZONE_SPEED_KMH[text]
    unit = 1.0
    if "mph" in text:
        unit = _MPH_TO_KMH
        text = text.replace("mph", "")
    text = text.replace("km/h", "").replace("kmh", "").replace("kph", "").strip()
    try:
        v = float(text) * unit
    except ValueError:
        return None
    return v if v > 0 else None


def _edge_speed_kmh(data):
    """(speed_kmh, tagged: bool) for one OSMnx edge data dict."""
    v = _parse_maxspeed_kmh(data.get("maxspeed"))
    if v is not None:
        return v, True
    hw = data.get("highway")
    if isinstance(hw, (list, tuple, set)):
        speeds = [HIGHWAY_CLASS_SPEED_KMH.get(str(h), DEFAULT_CLASS_SPEED_KMH)
                  for h in hw]
        return (min(speeds) if speeds else DEFAULT_CLASS_SPEED_KMH), False
    return HIGHWAY_CLASS_SPEED_KMH.get(str(hw), DEFAULT_CLASS_SPEED_KMH), False


# -----------------------------------------------------------------------------
# OSM fetch (the only networked part; version-tolerant across OSMnx 1.x / 2.x)
# -----------------------------------------------------------------------------
def _graph_from_bbox(ox, north, south, east, west):
    errs = []
    empty_seen = False
    for call in (
        # OSMnx 2.x: bbox=(left, bottom, right, top)
        lambda: ox.graph_from_bbox(bbox=(west, south, east, north),
                                   network_type="drive", simplify=True,
                                   retain_all=True),
        # OSMnx 1.x positional
        lambda: ox.graph_from_bbox(north, south, east, west,
                                   network_type="drive", simplify=True,
                                   retain_all=True),
    ):
        try:
            return call()
        except Exception as e:
            if _looks_like_empty_osm_response(e):
                empty_seen = True
            errs.append(repr(e))
    if empty_seen:
        return None
    raise RuntimeError(" | ".join(errs))


def fetch_speed_edges_bbox(north, south, east, west, verbose=True):
    """
    Return [(polyline [(lat, lon), ...], speed_kmh, tagged), ...] for drivable
    OSM ways in the bbox. `tagged` is True when the speed comes from an explicit
    maxspeed tag, False when imputed from the highway class.

    Returns [] when the bbox genuinely has no drivable ways. Raises RuntimeError
    when OSMnx is missing or the query fails, so the caller can fall back
    cleanly (an empty result and an infrastructure failure must not look alike).
    """
    try:
        import osmnx as ox
    except ImportError as e:
        raise RuntimeError("OSMnx not installed (`pip install osmnx`)") from e

    G = _graph_from_bbox(ox, north, south, east, west)
    if G is None:
        return []

    edges = []
    n_tagged = 0
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            poly = [(float(y), float(x)) for x, y in geom.coords]
        else:
            nu, nv = G.nodes[u], G.nodes[v]
            poly = [(float(nu["y"]), float(nu["x"])),
                    (float(nv["y"]), float(nv["x"]))]
        if len(poly) < 2:
            continue
        speed_kmh, tagged = _edge_speed_kmh(data)
        n_tagged += int(tagged)
        edges.append((poly, speed_kmh, tagged))

    if verbose:
        print(f"  speed caps: fetched {len(edges)} drivable OSM edge(s) in bbox "
              f"({n_tagged} with explicit maxspeed, "
              f"{len(edges) - n_tagged} imputed by class)")
    return edges


# -----------------------------------------------------------------------------
# Geometry: local-metres grid index over edge sub-segments
# -----------------------------------------------------------------------------
class _EdgeIndex:
    """
    Nearest-edge lookup over many polylines using a uniform grid in a local
    equirectangular projection (metres). Pure Python, no scipy/OSMnx needed at
    query time, and fast enough for route-scale graphs (grid prunes candidates).
    """

    def __init__(self, edges, cell_m=150.0):
        self.cell = float(cell_m)
        self.grid = {}
        self.subsegs = []            # (ax, ay, bx, by, speed_kmh, tagged)
        if not edges:
            self.lat0 = self.lon0 = 0.0
            self._coslat = 1.0
            return
        lat_sum = lon_sum = n = 0
        for poly, _s, _t in edges:
            for lat, lon in poly:
                lat_sum += lat
                lon_sum += lon
                n += 1
        self.lat0, self.lon0 = lat_sum / n, lon_sum / n
        self._coslat = max(cos(radians(self.lat0)), 1e-6)
        for poly, speed_kmh, tagged in edges:
            pts = [self._xy(lat, lon) for lat, lon in poly]
            for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
                idx = len(self.subsegs)
                self.subsegs.append((ax, ay, bx, by, speed_kmh, tagged))
                for key in self._cells_for(ax, ay, bx, by):
                    self.grid.setdefault(key, []).append(idx)

    def _xy(self, lat, lon):
        return (radians(lon - self.lon0) * self._coslat * 6_371_000.0,
                radians(lat - self.lat0) * 6_371_000.0)

    def _cells_for(self, ax, ay, bx, by):
        c = self.cell
        i0, i1 = sorted((int(ax // c), int(bx // c)))
        j0, j1 = sorted((int(ay // c), int(by // c)))
        return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]

    @staticmethod
    def _dist2_point_seg(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        denom = vx * vx + vy * vy
        if denom <= 0:
            dx, dy = px - ax, py - ay
            return dx * dx + dy * dy
        t = ((px - ax) * vx + (py - ay) * vy) / denom
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        dx, dy = px - (ax + t * vx), py - (ay + t * vy)
        return dx * dx + dy * dy

    def nearest_speed(self, lat, lon, radius_m):
        """(speed_kmh, tagged) of the nearest edge within radius_m, or None."""
        if not self.subsegs:
            return None
        px, py = self._xy(lat, lon)
        c = self.cell
        reach = max(int(radius_m // c) + 1, 1)
        i0, j0 = int(px // c), int(py // c)
        best_d2, best = radius_m * radius_m, None
        seen = set()
        for i in range(i0 - reach, i0 + reach + 1):
            for j in range(j0 - reach, j0 + reach + 1):
                for idx in self.grid.get((i, j), ()):
                    if idx in seen:
                        continue
                    seen.add(idx)
                    ax, ay, bx, by, speed, tagged = self.subsegs[idx]
                    d2 = self._dist2_point_seg(px, py, ax, ay, bx, by)
                    if d2 <= best_d2:
                        best_d2, best = d2, (speed, tagged)
        return best


def _sample_polyline(poly, step_m):
    """
    Sample points at the midpoint of consecutive `step_m` intervals along the
    polyline arc length. Returns [(lat, lon, weight_m), ...]; weights sum to the
    polyline length, so equal-ish weights make per-sample speeds directly
    usable in a length-weighted harmonic mean.
    """
    if len(poly) < 2:
        return []
    seg_lens = [
        _haversine_m(a[0], a[1], b[0], b[1])
        for a, b in zip(poly[:-1], poly[1:])
    ]
    total = sum(seg_lens)
    if total <= 0:
        return []
    step = max(float(step_m), 1.0)
    n = max(int(round(total / step)), 1)
    w = total / n
    targets = [(k + 0.5) * w for k in range(n)]

    out, acc, si = [], 0.0, 0
    for tgt in targets:
        while si < len(seg_lens) - 1 and acc + seg_lens[si] < tgt:
            acc += seg_lens[si]
            si += 1
        (alat, alon), (blat, blon) = poly[si], poly[si + 1]
        f = 0.0 if seg_lens[si] <= 0 else min(max((tgt - acc) / seg_lens[si], 0.0), 1.0)
        out.append((alat + f * (blat - alat), alon + f * (blon - alon), w))
    return out


def _harmonic_cap_for_polyline(poly, index, snap_radius_m, sample_step_m):
    """
    (cap_kmh, coverage_frac, tagged_frac) for one stop-pair polyline.

    cap_kmh is the length-weighted harmonic mean of the matched samples --
    total_matched_length / sum(w_i / v_i) -- which preserves the true minimum
    cruise traversal time of the mixed-limit path. coverage_frac is the matched
    share of the polyline length; tagged_frac the explicitly-tagged share of
    the matched length. (None, 0.0, 0.0) when nothing matches.
    """
    samples = _sample_polyline(poly, sample_step_m)
    if not samples:
        return None, 0.0, 0.0
    total_w = sum(w for _la, _lo, w in samples)
    matched_w = tagged_w = inv_sum = 0.0
    for lat, lon, w in samples:
        hit = index.nearest_speed(lat, lon, snap_radius_m)
        if hit is None:
            continue
        speed_kmh, tagged = hit
        if speed_kmh <= 0:
            continue
        matched_w += w
        tagged_w += w * int(bool(tagged))
        inv_sum += w / speed_kmh
    if matched_w <= 0 or inv_sum <= 0:
        return None, 0.0, 0.0
    return (matched_w / inv_sum, matched_w / total_w, tagged_w / matched_w)


# -----------------------------------------------------------------------------
# Cache CSV I/O
# -----------------------------------------------------------------------------
def load_speed_cap_cache(path):
    """Return {(from_id, to_id): row_dict}. Empty dict if the file is absent."""
    path = Path(path)
    cache = {}
    if not path.exists():
        return cache
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (str(row["from_stop_id"]), str(row["to_stop_id"]))
            if key in cache:
                raise ValueError(f"duplicate speed-cap cache row for stop pair {key}")
            row["speed_cap_ms"] = float(row["speed_cap_ms"])
            if row["speed_cap_ms"] <= 0:
                raise ValueError(f"speed_cap_ms must be positive for stop pair {key}")
            row["speed_cap_kmh"] = float(row.get("speed_cap_kmh") or 0.0)
            if row["speed_cap_kmh"] < 0:
                raise ValueError(f"speed_cap_kmh must be non-negative for stop pair {key}")
            if row.get("source") not in {
                "osm",
                "imputed",
                "fallback_low_coverage",
                "fallback",
            }:
                raise ValueError(f"invalid speed-cap cache source for stop pair {key}")
            row["coverage_frac"] = float(row.get("coverage_frac") or 0.0)
            if not (0.0 <= row["coverage_frac"] <= 1.0):
                raise ValueError(f"coverage_frac must be in [0, 1] for stop pair {key}")
            row["snap_radius_m"] = float(row.get("snap_radius_m") or 0.0)
            if row["snap_radius_m"] <= 0:
                raise ValueError(f"snap_radius_m must be positive for stop pair {key}")
            row["sample_step_m"] = float(row.get("sample_step_m") or 0.0)
            if row["sample_step_m"] <= 0:
                raise ValueError(f"sample_step_m must be positive for stop pair {key}")
            row["length_m"] = float(row.get("length_m") or 0.0)
            if row["length_m"] < 0:
                raise ValueError(f"length_m must be non-negative for stop pair {key}")
            cache[key] = row
    return cache


def save_speed_cap_cache(path, cache):
    """Write the merged cache dict back to CSV (one row per stop pair)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SPEED_CAP_CACHE_COLUMNS)
        w.writeheader()
        for (f, t), row in sorted(cache.items()):
            w.writerow({
                "from_stop_id": f, "to_stop_id": t,
                "speed_cap_ms": round(float(row["speed_cap_ms"]), 3),
                "speed_cap_kmh": round(float(row.get("speed_cap_kmh") or 0.0), 1),
                "source": row["source"],
                "coverage_frac": round(float(row.get("coverage_frac") or 0.0), 3),
                "snap_radius_m": row.get("snap_radius_m", 0.0),
                "sample_step_m": row.get("sample_step_m", 0.0),
                "length_m": round(float(row.get("length_m") or 0.0), 1),
                "fetched_utc": row.get("fetched_utc", ""),
            })
    return path


# -----------------------------------------------------------------------------
# Cache-aware resolver (mirrors resolve_signal_counts)
# -----------------------------------------------------------------------------
def resolve_speed_caps(pairs, coords,
                       snap_radius_m=DEFAULT_SPEED_SNAP_RADIUS_M,
                       sample_step_m=DEFAULT_SAMPLE_STEP_M,
                       min_coverage_frac=DEFAULT_MIN_COVERAGE_FRAC,
                       default_cap_kmh=DEFAULT_FALLBACK_CAP_KMH,
                       max_cap_kmh=DEFAULT_MAX_CAP_KMH,
                       fetch_fn=fetch_speed_edges_bbox,
                       cache=None, cache_path=None, refresh=False,
                       bbox_pad_m=200.0, geom_map=None, verbose=True):
    """
    Resolve speed_cap_ms for every (from_id, to_id, length_m) pair.

    Order of resolution per pair:
      1. cache hit  (same pair, same snap/sample params, unless refresh=True)
      2. OSM        (one bbox fetch covering all misses; per-pair harmonic mean
                     of maxspeed along the pair's shape sub-path)
      3. fallback   (default_cap_kmh; also used when matched coverage of a
                     pair's polyline falls below min_coverage_frac)

    Sources: "osm" (explicit maxspeed tags dominate the matched length),
    "imputed" (class-imputed speeds dominate), "fallback_low_coverage",
    "fallback". geom_map carries the shape.txt driven path per pair (same
    geometry as segment lengths); the straight chord is used only without it.

    Returns (cap_map {(f, t): m/s}, source_map, stats).
    """
    if cache is None:
        cache = load_speed_cap_cache(cache_path) if cache_path else {}

    uniq = {}
    for f, t, length_m in pairs:
        uniq[(str(f), str(t))] = float(length_m)

    cap_map, source_map = {}, {}
    misses = []
    n_hit = 0
    for key, length_m in uniq.items():
        row = None if refresh else cache.get(key)
        if (
            row is not None
            and abs(row["snap_radius_m"] - snap_radius_m) < 1e-6
            and abs(row["sample_step_m"] - sample_step_m) < 1e-6
        ):
            cap_map[key] = float(row["speed_cap_ms"])
            source_map[key] = row["source"]
            n_hit += 1
        else:
            misses.append((key[0], key[1], length_m))

    n_osm = n_imputed = n_low_cov = n_fallback = n_edges = 0
    if misses:
        edges = None
        if fetch_fn is not None:
            bbox = _bbox_for_pairs(misses, coords, bbox_pad_m, geom_map=geom_map)
            try:
                edges = fetch_fn(*bbox, verbose=verbose)
                n_edges = len(edges)
            except Exception as e:
                if verbose:
                    print(f"  speed caps: OSM fetch failed ({e}); using fallback "
                          f"{default_cap_kmh:.0f} km/h for {len(misses)} pair(s)")
                edges = None

        index = _EdgeIndex(edges) if edges else None
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for f, t, length_m in misses:
            key = (f, t)
            cap_kmh, coverage, tagged_frac = None, 0.0, 0.0
            if index is not None:
                poly = geom_map.get((f, t)) if geom_map else None
                if not poly or len(poly) < 2:
                    poly = [coords[f], coords[t]]
                cap_kmh, coverage, tagged_frac = _harmonic_cap_for_polyline(
                    poly, index, snap_radius_m, sample_step_m
                )
            if edges is None:
                cap_kmh, source = default_cap_kmh, "fallback"
                n_fallback += 1
            elif cap_kmh is None or coverage < float(min_coverage_frac):
                cap_kmh, source = default_cap_kmh, "fallback_low_coverage"
                n_low_cov += 1
            elif tagged_frac >= 0.5:
                source = "osm"
                n_osm += 1
            else:
                source = "imputed"
                n_imputed += 1

            cap_kmh = min(max(float(cap_kmh), MIN_CAP_KMH), float(max_cap_kmh))
            cap_ms = cap_kmh / 3.6
            cap_map[key] = cap_ms
            source_map[key] = source
            cache[key] = {
                "speed_cap_ms": cap_ms, "speed_cap_kmh": cap_kmh,
                "source": source, "coverage_frac": coverage,
                "snap_radius_m": float(snap_radius_m),
                "sample_step_m": float(sample_step_m),
                "length_m": length_m, "fetched_utc": now,
            }

        if cache_path is not None:
            save_speed_cap_cache(cache_path, cache)

    stats = {"n_pairs": len(uniq), "cache_hit": n_hit,
             "osm": n_osm, "imputed": n_imputed,
             "fallback_low_coverage": n_low_cov, "fallback": n_fallback,
             "osm_edges": n_edges}
    return cap_map, source_map, stats


# -----------------------------------------------------------------------------
# Apply: set Segment.speed_cap_ms (+ Segment.speed_cap_source)
# -----------------------------------------------------------------------------
def add_speed_caps(segments, trip_rows, stops, cap_map=None, source_map=None,
                   shape_points=None, verbose=True):
    """
    Write Segment.speed_cap_ms (and Segment.speed_cap_source) for one trip by
    pure lookup on (from_stop_id, to_stop_id); caps are resolved once per route
    by resolve_speed_caps. Mirrors add_traffic_signals / add_grades_from_dem:
    walks the same iter_valid_stop_pairs so seg i lines up with pair i. Pairs
    absent from cap_map leave the segment untouched (model falls back to the
    config default cap). Returns the segment list (mutated in place).
    """
    if cap_map is None:
        raise ValueError("add_speed_caps needs a cap_map from resolve_speed_caps.")
    pairs = _stop_pairs(trip_rows, stops, shape_points=shape_points)
    if len(pairs) != len(segments):
        raise ValueError(
            f"segment/pair mismatch ({len(segments)} vs {len(pairs)}); "
            "build_segments and add_speed_caps are out of sync.")
    if not segments:
        return segments

    n_set = 0
    for seg, (a, b, _l) in zip(segments, pairs):
        key = (a["stop_id"], b["stop_id"])
        cap = cap_map.get(key)
        if cap is None:
            continue
        seg.speed_cap_ms = float(cap)
        seg.speed_cap_source = (source_map or {}).get(key, "osm")
        n_set += 1

    if verbose and segments:
        caps = [s.speed_cap_ms for s in segments if s.speed_cap_ms is not None]
        if caps:
            print(f"  speed caps: set on {n_set}/{len(segments)} segments "
                  f"(range {min(caps) * 3.6:.0f}-{max(caps) * 3.6:.0f} km/h)")
        else:
            print("  speed caps: none resolved -> config default cap applies.")
    return segments


# -----------------------------------------------------------------------------
# Self-test: parsing + harmonic aggregation + cache round-trip, no network
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import tempfile

    # maxspeed parsing
    assert _parse_maxspeed_kmh("50") == 50.0
    assert abs(_parse_maxspeed_kmh("30 mph") - 48.28) < 0.01
    assert _parse_maxspeed_kmh("IE:urban") == 50.0
    assert _parse_maxspeed_kmh(["50", "30"]) == 30.0
    assert _parse_maxspeed_kmh("signals") is None
    print("maxspeed parsing OK")

    # A straight 1000 m north-south corridor at lat 51.9: 800 m of 50 km/h then
    # 200 m of 30 km/h. Expected harmonic cap = 1 / (0.8/50 + 0.2/30) = 44.12.
    lat0, lon0 = 51.9000, -8.4700
    dlat = 1.0 / 111_320.0  # deg per metre (north-south)
    edges = [
        ([(lat0, lon0), (lat0 + 800 * dlat, lon0)], 50.0, True),
        ([(lat0 + 800 * dlat, lon0), (lat0 + 1000 * dlat, lon0)], 30.0, True),
    ]
    idx = _EdgeIndex(edges)
    poly = [(lat0, lon0), (lat0 + 1000 * dlat, lon0)]
    cap_kmh, coverage, tagged = _harmonic_cap_for_polyline(poly, idx, 25.0, 20.0)
    assert abs(cap_kmh - 44.1176) < 0.15, cap_kmh
    assert coverage > 0.99 and tagged > 0.99
    print(f"harmonic mean: 80% @50 + 20% @30 -> {cap_kmh:.2f} km/h "
          f"(expect 44.12; min would give 30, arithmetic 46)")

    # resolver: OSM path, imputed source, low-coverage fallback, cache behaviour
    coords = {"A": (lat0, lon0), "B": (lat0 + 1000 * dlat, lon0),
              "C": (lat0 + 2000 * dlat, lon0),
              "X": (52.2, -8.47), "Y": (52.21, -8.47)}
    geom_map = {("A", "B"): poly,
                ("B", "C"): [(lat0 + 1000 * dlat, lon0), (lat0 + 2000 * dlat, lon0)],
                ("X", "Y"): [(52.2, -8.47), (52.21, -8.47)]}
    imputed_edges = edges + [
        ([(lat0 + 1000 * dlat, lon0), (lat0 + 2000 * dlat, lon0)], 50.0, False),
    ]
    cache = {}
    cmap, smap, stats = resolve_speed_caps(
        [("A", "B", 1000.0), ("B", "C", 1000.0), ("X", "Y", 1100.0)], coords,
        geom_map=geom_map, cache=cache,
        fetch_fn=lambda *b, **k: imputed_edges, verbose=False)
    assert smap[("A", "B")] == "osm" and abs(cmap[("A", "B")] * 3.6 - 44.12) < 0.2
    assert smap[("B", "C")] == "imputed"
    # X-Y is ~33 km from every edge -> no coverage -> fallback cap
    assert smap[("X", "Y")] == "fallback_low_coverage"
    assert abs(cmap[("X", "Y")] * 3.6 - 50.0) < 1e-6
    print("resolver sources:", [smap[k] for k in cmap], "caps km/h:",
          [round(v * 3.6, 1) for v in cmap.values()])

    # second call -> all cache hits, fetch_fn must NOT be called
    def _boom(*a, **k):
        raise AssertionError("fetch called on a cache hit")
    cmap2, smap2, stats2 = resolve_speed_caps(
        [("A", "B", 1000.0), ("B", "C", 1000.0), ("X", "Y", 1100.0)], coords,
        geom_map=geom_map, cache=cache, fetch_fn=_boom, verbose=False)
    assert stats2["cache_hit"] == 3
    print("second run: all", stats2["cache_hit"], "pairs from cache (no fetch)")

    # OSM unavailable -> plain fallback
    cmap3, smap3, _ = resolve_speed_caps(
        [("A", "B", 1000.0)], coords, geom_map=geom_map, cache={},
        fetch_fn=lambda *b, **k: (_ for _ in ()).throw(RuntimeError("no OSM")),
        verbose=False)
    assert smap3[("A", "B")] == "fallback"
    print("fallback: OSM down ->", round(cmap3[("A", "B")] * 3.6), "km/h default")

    # clamp: a bogus 160 km/h tag is clamped to max_cap_kmh
    fast_edges = [([(lat0, lon0), (lat0 + 1000 * dlat, lon0)], 160.0, True)]
    cmap4, _, _ = resolve_speed_caps(
        [("A", "B", 1000.0)], coords, geom_map=geom_map, cache={},
        max_cap_kmh=90.0, fetch_fn=lambda *b, **k: fast_edges, verbose=False)
    assert abs(cmap4[("A", "B")] * 3.6 - 90.0) < 1e-6
    print("clamp: 160 km/h tag -> 90 km/h max cap")

    # cache CSV round-trip
    p = os.path.join(tempfile.mkdtemp(), "speed_caps.csv")
    save_speed_cap_cache(p, cache)
    reloaded = load_speed_cap_cache(p)
    assert reloaded.keys() == cache.keys()
    assert abs(reloaded[("A", "B")]["speed_cap_ms"]
               - cache[("A", "B")]["speed_cap_ms"]) < 1e-3
    print("cache CSV round-trip OK:", p)
    print("self-test passed")
