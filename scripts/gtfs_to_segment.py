"""
gtfs_to_segments.py
================================================================================
Turn a static GTFS feed into the stop-to-stop `Segment` list consumed by
beb_soc_model.py, for ONE OR MANY trips per route across the day.

ROUTE -> MOTION inputs, from real data:
  * segment length   <- driven distance along shapes.txt between consecutive stops
                        (falls back to stop-to-stop Haversine if shape data is
                         unavailable)
  * dwell time        <- departure - arrival at each stop (stop_times.txt)
  * scheduled run time <- departure at stop A to arrival at stop B
  * grade             <- sampled from SRTM elevations (add_grades_from_dem)
  * passengers        <- modelled by passenger_loading.py from each trip's real
                         start time + an hourly demand profile (optional)

WHICH TRIPS GET PROCESSED
-------------------------
  * service day      <- calendar.txt picks the trips that run on a given weekday
  * start time(s)    <- if given, the nearest trip to each requested HH:MM is
                        used; if NOT given, ALL trips on that service day run.
  * trip start time  <- read from stop_times.txt (departure of first stop). It
                        also feeds passenger_loading so each trip is loaded for
                        the hour it actually runs.

The national feed has a 372 MB stop_times.txt, so it is read in a SINGLE
streaming pass that collects every wanted trip across every requested route at
once (filtering to a Python set, never loading the whole file).

USAGE
-----
    # passenger loading and data paths are configured in configs/model.yaml,
    # so loading applies with no flags -- just name the route(s):
    python3 gtfs_to_segments.py 220
    # every trip on a weekday, all requested routes:
    python3 gtfs_to_segments.py 102 41 --day monday
    # only the trips nearest these clock times:
    python3 gtfs_to_segments.py 102 --start-times 08:00,12:30,17:45
    # optional overrides (a different city / file, or turn loading off):
    python3 gtfs_to_segments.py 102 --demand-city Dublin
    python3 gtfs_to_segments.py 102 --no-demand
from code:
    from gtfs_to_segments import process_routes, load_small_tables
================================================================================
"""

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from dataclasses import asdict, replace
from datetime import date, timedelta
from functools import partial
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

import pandas as pd
import srtm  # SRTM elevation tiles, same source RouteZero uses for grades

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from best_ire_beb.config import get_path, get_section

# Reuse the vehicle model and Segment dataclass from the sibling file.
# MotionParams carries the config-driven motion-profile parameters, including
# the physical speed cap defaults that replaced the GTFS-derived v_cruise cap.
from beb_soc_model import MotionParams, Segment, VehicleParams, simulate_route

# Passenger demand model. Soft import: if the module is absent the pipeline
# still runs, it just leaves the flat `passengers` assumption in place.
try:
    from passenger_loading import HourlyDemandProfile, apply_passenger_loading
    _HAS_LOADING = True
except ImportError:
    HourlyDemandProfile = None
    apply_passenger_loading = None
    _HAS_LOADING = False

try:
    from weather_loading import apply_weather_loading, load_weather_csv
    _HAS_WEATHER = True
except ImportError:
    apply_weather_loading = None
    load_weather_csv = None
    _HAS_WEATHER = False

try:
    from traffic_signals import (add_traffic_signals, resolve_signal_counts,
                                 route_stop_pairs, load_signal_cache,
                                 load_signal_event_cache,
                                 fetch_traffic_signals_bbox,
                                 DEFAULT_SNAP_RADIUS_M,
                                 DEFAULT_RELAXED_SNAP_RADIUS_M,
                                 DEFAULT_SIGNAL_CLUSTER_RADIUS_M,
                                 DEFAULT_FALLBACK_PER_KM)
    _HAS_SIGNALS = True
except ImportError:
    add_traffic_signals = None
    resolve_signal_counts = None
    route_stop_pairs = None
    load_signal_cache = None
    load_signal_event_cache = None
    fetch_traffic_signals_bbox = None
    DEFAULT_SNAP_RADIUS_M = 30.0
    DEFAULT_RELAXED_SNAP_RADIUS_M = 60.0
    DEFAULT_SIGNAL_CLUSTER_RADIUS_M = 35.0
    DEFAULT_FALLBACK_PER_KM = 2.0
    _HAS_SIGNALS = False

# OSM maxspeed -> per-segment physical speed cap. Soft import like the others:
# without it, segments keep speed_cap_ms=None and beb_soc_model falls back to
# motion.default_speed_cap_ms from the config.
try:
    from speed_caps import (add_speed_caps, resolve_speed_caps,
                            load_speed_cap_cache,
                            fetch_speed_edges_bbox,
                            DEFAULT_SPEED_SNAP_RADIUS_M,
                            DEFAULT_SAMPLE_STEP_M,
                            DEFAULT_MIN_COVERAGE_FRAC,
                            DEFAULT_FALLBACK_CAP_KMH,
                            DEFAULT_MAX_CAP_KMH)
    _HAS_SPEED_CAPS = True
except ImportError:
    add_speed_caps = None
    resolve_speed_caps = None
    load_speed_cap_cache = None
    fetch_speed_edges_bbox = None
    DEFAULT_SPEED_SNAP_RADIUS_M = 25.0
    DEFAULT_SAMPLE_STEP_M = 20.0
    DEFAULT_MIN_COVERAGE_FRAC = 0.5
    DEFAULT_FALLBACK_CAP_KMH = 50.0
    DEFAULT_MAX_CAP_KMH = 90.0
    _HAS_SPEED_CAPS = False

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday"]

# Above this a scheduled interval is a length defect, not a fast road.
MAX_PLAUSIBLE_SCHEDULED_KMH = 120.0


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def gtfs_time_to_seconds(t: str) -> int:
    """GTFS times can exceed 24:00:00 (after-midnight trips); handle h>=24."""
    h, m, s = (int(x) for x in str(t).strip().split(":"))
    return h * 3600 + m * 60 + s


def parse_service_date(value):
    if value in (None, ""):
        return None
    return date.fromisoformat(str(value))


def seconds_to_hhmmss(sec: int) -> str:
    """Inverse of gtfs_time_to_seconds; keeps h>=24 for after-midnight trips."""
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    R = 6_371_000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return 2 * R * asin(sqrt(a))


def _to_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


# -----------------------------------------------------------------------------
# Read small tables (routes, trips, stops, calendar)
# -----------------------------------------------------------------------------
def load_small_tables(gtfs_zip: str):
    """routes, trips, stops, calendar, calendar_dates are small enough to read."""
    with zipfile.ZipFile(gtfs_zip) as z:
        names = set(z.namelist())
        required = {"routes.txt", "trips.txt", "stops.txt", "stop_times.txt"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(
                f"GTFS feed is missing required file(s): {', '.join(missing)}"
            )
        routes = pd.read_csv(z.open("routes.txt"), dtype={"route_id": str})
        trips = pd.read_csv(
            z.open("trips.txt"),
            dtype={"trip_id": str, "route_id": str, "shape_id": str,
                   "service_id": str},
        )
        stops = pd.read_csv(z.open("stops.txt"), dtype={"stop_id": str})
        calendar = (pd.read_csv(z.open("calendar.txt"), dtype={"service_id": str})
                    if "calendar.txt" in names else None)
        calendar_dates = (
            pd.read_csv(z.open("calendar_dates.txt"), dtype={"service_id": str})
            if "calendar_dates.txt" in names else None
        )
    return routes, trips, stops, calendar, calendar_dates


def validate_gtfs_feed(gtfs_zip: str):
    """Fast Level 1 GTFS integrity checks for static feed inputs."""
    with zipfile.ZipFile(gtfs_zip) as z:
        names = set(z.namelist())
        required = {"routes.txt", "trips.txt", "stops.txt", "stop_times.txt"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(
                f"GTFS feed is missing required file(s): {', '.join(missing)}"
            )
        routes = pd.read_csv(z.open("routes.txt"), dtype=str)
        trips = pd.read_csv(z.open("trips.txt"), dtype=str)
        stops = pd.read_csv(z.open("stops.txt"), dtype=str)
        stop_times = pd.read_csv(z.open("stop_times.txt"), dtype=str)

    required_columns = {
        "routes.txt": (routes, {"route_id"}),
        "trips.txt": (trips, {"trip_id", "route_id"}),
        "stops.txt": (stops, {"stop_id", "stop_lat", "stop_lon"}),
        "stop_times.txt": (
            stop_times,
            {
                "trip_id",
                "arrival_time",
                "departure_time",
                "stop_id",
                "stop_sequence",
            },
        ),
    }
    for filename, (frame, columns) in required_columns.items():
        missing_cols = sorted(columns - set(frame.columns))
        if missing_cols:
            raise ValueError(
                f"{filename} is missing required column(s): {', '.join(missing_cols)}"
            )

    bad_stop_ids = sorted(set(stop_times["stop_id"]) - set(stops["stop_id"]))
    if bad_stop_ids:
        raise ValueError(f"stop_times references unknown stop_id(s): {bad_stop_ids}")

    bad_trip_ids = sorted(set(stop_times["trip_id"]) - set(trips["trip_id"]))
    if bad_trip_ids:
        raise ValueError(f"stop_times references unknown trip_id(s): {bad_trip_ids}")

    bad_route_ids = sorted(set(trips["route_id"]) - set(routes["route_id"]))
    if bad_route_ids:
        raise ValueError(f"trips references unknown route_id(s): {bad_route_ids}")

    for trip_id, group in stop_times.groupby("trip_id", sort=False):
        seq = pd.to_numeric(group["stop_sequence"], errors="coerce")
        if seq.isna().any():
            raise ValueError(f"trip {trip_id} has non-numeric stop_sequence")
        if seq.duplicated().any() or not seq.is_monotonic_increasing:
            raise ValueError(f"trip {trip_id} stop_sequence values must increase")

        ordered = group.assign(_seq=seq).sort_values("_seq")
        rows = ordered.to_dict("records")
        for a, b in zip(rows[:-1], rows[1:]):
            run_time = (
                gtfs_time_to_seconds(b["arrival_time"])
                - gtfs_time_to_seconds(a["departure_time"])
            )
            if run_time <= 0:
                raise ValueError(f"trip {trip_id} has non-positive travel time")

    return {
        "routes": len(routes),
        "trips": len(trips),
        "stops": len(stops),
        "stop_times": len(stop_times),
    }


# -----------------------------------------------------------------------------
# Route / service / trip selection
# -----------------------------------------------------------------------------
def resolve_route_id(routes, route_selector):
    """
    Resolve an exact route_id or a route short name (e.g. '102') to
    (route_id, route_long_name). Short names are not unique across operators, so
    an ambiguous name is an error rather than a silent first match.
    """
    selector = str(route_selector)
    match = routes[routes["route_id"].astype(str) == selector]
    if match.empty:
        match = routes[routes["route_short_name"].astype(str) == selector]
    if match.empty:
        raise ValueError(f"No route with route_id or short name {selector!r}")
    if len(match) > 1:
        options = "; ".join(
            f"{row.route_id!r} ({row.route_long_name})"
            for row in match.itertuples(index=False)
        )
        raise ValueError(
            f"Route short name {selector!r} is ambiguous; use one route_id: {options}"
        )
    return match.iloc[0]["route_id"], match.iloc[0]["route_long_name"]


def service_ids_for_day(calendar, day="monday"):
    """service_ids that operate on the given weekday, per calendar.txt.
    Returns None (no filter) if calendar.txt is absent."""
    if calendar is None:
        return None
    day = day.lower()
    if day not in calendar.columns:
        raise ValueError(f"calendar.txt has no '{day}' column")
    return set(calendar.loc[calendar[day] == 1, "service_id"].astype(str))


def _service_ids_on_exact_date(calendar, calendar_dates, service_date):
    yyyymmdd = service_date.strftime("%Y%m%d")
    day = service_date.strftime("%A").lower()
    active = set()

    if {"start_date", "end_date", day}.issubset(calendar.columns):
        in_range = (
            (calendar["start_date"].astype(str) <= yyyymmdd)
            & (calendar["end_date"].astype(str) >= yyyymmdd)
            & (calendar[day].astype(str) == "1")
        )
        active = set(calendar.loc[in_range, "service_id"].astype(str))

    if calendar_dates is not None and not calendar_dates.empty:
        exceptions = calendar_dates[calendar_dates["date"].astype(str) == yyyymmdd]
        added = set(
            exceptions.loc[exceptions["exception_type"].astype(str) == "1", "service_id"]
            .astype(str)
        )
        removed = set(
            exceptions.loc[exceptions["exception_type"].astype(str) == "2", "service_id"]
            .astype(str)
        )
        active = (active | added) - removed

    return active


def _gtfs_calendar_years(calendar, calendar_dates):
    years = set()
    if calendar is not None:
        for col in ("start_date", "end_date"):
            if col in calendar.columns:
                values = calendar[col].dropna().astype(str)
                years.update(
                    int(v[:4]) for v in values if len(v) >= 4 and v[:4].isdigit()
                )
    if (
        calendar_dates is not None
        and not calendar_dates.empty
        and "date" in calendar_dates.columns
    ):
        values = calendar_dates["date"].dropna().astype(str)
        years.update(int(v[:4]) for v in values if len(v) >= 4 and v[:4].isdigit())
    return sorted(years)


def _parse_yyyymmdd(value):
    value = str(value)
    if len(value) != 8 or not value.isdigit():
        return None
    return date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}")


def _gtfs_calendar_bounds(calendar, calendar_dates):
    dates = []
    if calendar is not None:
        for col in ("start_date", "end_date"):
            if col in calendar.columns:
                dates.extend(
                    d for d in (_parse_yyyymmdd(v) for v in calendar[col].dropna())
                    if d is not None
                )
    if (
        calendar_dates is not None
        and not calendar_dates.empty
        and "date" in calendar_dates.columns
    ):
        dates.extend(
            d for d in (_parse_yyyymmdd(v) for v in calendar_dates["date"].dropna())
            if d is not None
        )
    if not dates:
        return None, None
    return min(dates), max(dates)


def _real_service_dates(calendar, calendar_dates):
    start, end = _gtfs_calendar_bounds(calendar, calendar_dates)
    if start is None or end is None:
        return []

    days = []
    current = start
    while current <= end:
        if _service_ids_on_exact_date(calendar, calendar_dates, current):
            days.append(current)
        current += timedelta(days=1)
    return days


def _same_day_type(a, b):
    return (a.weekday() < 5) == (b.weekday() < 5)


def _nearest_real_service_date(calendar, calendar_dates, target_date):
    service_dates = _real_service_dates(calendar, calendar_dates)
    if not service_dates:
        return None

    same_weekday = [d for d in service_dates if d.weekday() == target_date.weekday()]
    candidates = same_weekday or [
        d for d in service_dates if _same_day_type(d, target_date)
    ]
    if not candidates:
        candidates = service_dates

    return min(candidates, key=lambda d: (abs((d - target_date).days), d))


def resolve_gtfs_service_date(calendar, calendar_dates, service_date):
    """
    Return the date to use for GTFS service lookup.

    If the requested date's year is outside the GTFS feed years, mirror its
    month/day into a feed year. This keeps scenarios such as 2021-2025 weather
    paired with a 2026-only GTFS schedule by mapping 2025-04-14 -> 2026-04-14.
    """
    if service_date is None or calendar is None:
        return service_date

    years = _gtfs_calendar_years(calendar, calendar_dates)
    if not years or service_date.year in years:
        return service_date

    for year in years:
        try:
            candidate = service_date.replace(year=year)
        except ValueError:
            continue
        print(
            f"GTFS service date {service_date} is outside feed year(s) "
            f"{', '.join(str(y) for y in years)}; using {candidate} for GTFS."
        )
        return candidate

    return service_date


def service_ids_for_date(calendar, calendar_dates, service_date, fallback_to_weekday=True):
    """
    Service IDs active on a concrete date. If the requested date is outside this
    GTFS feed's calendar span, first try the same month/day in a feed year, then
    optionally fall back to the same weekday pattern.
    """
    if calendar is None:
        return None

    lookup_date = resolve_gtfs_service_date(calendar, calendar_dates, service_date)
    active = _service_ids_on_exact_date(calendar, calendar_dates, lookup_date)
    day = lookup_date.strftime("%A").lower()

    if active or not fallback_to_weekday:
        return active

    fallback_date = _nearest_real_service_date(calendar, calendar_dates, lookup_date)
    if fallback_date is not None:
        active = _service_ids_on_exact_date(calendar, calendar_dates, fallback_date)
        print(
            f"GTFS calendar has no exact services on remapped date {lookup_date}; "
            f"using nearest real {fallback_date.strftime('%A').lower()} "
            f"service date {fallback_date} instead."
        )
        return active

    if lookup_date != service_date:
        print(
            f"GTFS calendar has no exact services on remapped date {lookup_date}; "
            f"using broad {day} service pattern instead."
        )
    else:
        print(
            f"GTFS calendar has no exact services on {service_date}; "
            f"using broad {day} service pattern instead."
        )
    return service_ids_for_day(calendar, day)


def trips_for_route(trips, route_id, direction_id=0, service_ids=None):
    """All trip_ids for a route, optionally filtered by direction and service."""
    t = trips[trips["route_id"] == route_id]
    if direction_id is not None and "direction_id" in t.columns:
        d = t[t["direction_id"] == direction_id]
        if not d.empty:
            t = d  # fall back to all directions only if the filter is empty
    if service_ids is not None and "service_id" in t.columns:
        t = t[t["service_id"].astype(str).isin(service_ids)]
    return t["trip_id"].astype(str).tolist()


# -----------------------------------------------------------------------------
# SINGLE streaming pass over the big stop_times.txt for MANY trips
# -----------------------------------------------------------------------------
def stop_times_for_trips(gtfs_zip: str, trip_ids):
    """
    Stream stop_times.txt ONCE and return {trip_id: rows_sorted_by_sequence} for
    every trip in `trip_ids`. One full pass regardless of how many trips are
    requested, so processing 100 trips costs the same scan as processing 1.
    """
    wanted = set(map(str, trip_ids))
    by_trip = {tid: [] for tid in wanted}
    if not wanted:
        return by_trip
    with zipfile.ZipFile(gtfs_zip) as z:
        with z.open("stop_times.txt") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            for r in reader:
                bucket = by_trip.get(r["trip_id"])
                if bucket is not None:
                    bucket.append(r)
    for tid in by_trip:
        by_trip[tid].sort(key=lambda r: int(r["stop_sequence"]))
    return by_trip


def stop_times_for_trip(gtfs_zip: str, trip_id: str):
    """Back-compat single-trip helper (thin wrapper over the batch reader)."""
    return stop_times_for_trips(gtfs_zip, [trip_id])[str(trip_id)]


def _shape_points_with_cumdist(rows):
    points = []
    last = None
    cum_m = 0.0
    for lat, lon, _seq in sorted(rows, key=lambda r: r[2]):
        if last is not None:
            cum_m += haversine_m(last[0], last[1], lat, lon)
        points.append((lat, lon, cum_m))
        last = (lat, lon)
    return points


def _repaired_shape_rows(rows, repairs):
    """
    Drop configured phantom sections: points strictly between each keep_until and
    resume_at shape_pt_sequence. The two named points must coincide, so the path
    stays continuous and a changed feed fails here rather than cutting real road.
    """
    if not repairs:
        return rows
    by_sequence = {seq: (lat, lon) for lat, lon, seq in rows}
    for keep_until, resume_at in repairs:
        start, end = by_sequence.get(keep_until), by_sequence.get(resume_at)
        if start is None or end is None or haversine_m(*start, *end) > 1.0:
            raise ValueError(
                f"Shape repair [{keep_until}, {resume_at}] no longer joins coincident "
                "shape points; review it against this feed."
            )
    return [row for row in rows
            if not any(keep_until < row[2] < resume_at for keep_until, resume_at in repairs)]


def load_shapes_for_trips(gtfs_zip: str, trips, trip_ids, shape_repairs=None):
    """
    Return ({shape_id: [(lat, lon, cumulative_m), ...]}, {trip_id: shape_id}).

    shapes.txt can be large, so this streams the zip member once and keeps only
    the shape_ids used by the candidate trips. shape_repairs maps a shape_id to
    the [keep_until, resume_at] sections to remove (see _repaired_shape_rows).
    """
    wanted_trips = set(map(str, trip_ids))
    empty = {}, {}
    if not wanted_trips or "shape_id" not in trips.columns:
        return empty

    trip_shape = (
        trips.loc[trips["trip_id"].astype(str).isin(wanted_trips),
                  ["trip_id", "shape_id"]]
        .dropna(subset=["shape_id"])
        .astype(str)
    )
    shape_id_by_trip = {
        row.trip_id: row.shape_id
        for row in trip_shape.itertuples(index=False)
        if row.shape_id
    }
    wanted_shapes = set(shape_id_by_trip.values())
    if not wanted_shapes:
        return empty

    with zipfile.ZipFile(gtfs_zip) as z:
        if "shapes.txt" not in set(z.namelist()):
            return {}, shape_id_by_trip
        rows_by_shape = {sid: [] for sid in wanted_shapes}
        with z.open("shapes.txt") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            for r in reader:
                sid = str(r.get("shape_id", ""))
                bucket = rows_by_shape.get(sid)
                if bucket is None:
                    continue
                lat = _to_float(r.get("shape_pt_lat"))
                lon = _to_float(r.get("shape_pt_lon"))
                if lat is None or lon is None:
                    continue
                bucket.append((lat, lon, _to_int(r.get("shape_pt_sequence"))))

    shapes = {
        sid: _shape_points_with_cumdist(
            _repaired_shape_rows(rows, (shape_repairs or {}).get(sid))
        )
        for sid, rows in rows_by_shape.items()
        if len(rows) >= 2
    }
    return shapes, shape_id_by_trip


def trip_start_seconds(rows) -> int:
    """Trip start = departure_time of the first stop (rows sorted by sequence)."""
    return gtfs_time_to_seconds(rows[0]["departure_time"])


def select_trips_by_start(by_trip, start_times=None, tolerance_s=900):
    """
    Choose which trips to process.
      start_times = None        -> ALL trips, ordered by start time.
      start_times = [HH:MM,...]  -> the nearest trip to each requested time,
                                    within `tolerance_s` (default 15 min).
    Returns a list of trip_ids.
    """
    starts = {t: trip_start_seconds(r) for t, r in by_trip.items() if r}
    order = sorted(starts, key=lambda t: starts[t])
    if not start_times:
        return order
    selected = []
    for tgt in start_times:
        tgt = tgt if str(tgt).count(":") == 2 else f"{tgt}:00"
        tsec = gtfs_time_to_seconds(tgt)
        if not order:
            break
        best = min(order, key=lambda t: abs(starts[t] - tsec))
        if abs(starts[best] - tsec) <= tolerance_s:
            if best not in selected:
                selected.append(best)
        else:
            print(f"  no trip within {tolerance_s // 60} min of {tgt}; skipped")
    return selected


# -----------------------------------------------------------------------------
# Build the Segment list (validated)
# -----------------------------------------------------------------------------
def _project_stop_to_shape_m(lat, lon, shape_points, min_cum_m=None):
    """
    Project one stop onto a shape polyline and return cumulative metres.

    The equirectangular projection is local to the stop being snapped, which is
    accurate enough for short GTFS shape segments and avoids extra dependencies.
    """
    if not shape_points or len(shape_points) < 2:
        return None

    earth_r = 6_371_000.0
    lat0 = radians(lat)
    best = None
    for a, b in zip(shape_points[:-1], shape_points[1:]):
        lat_a, lon_a, cum_a = a
        lat_b, lon_b, cum_b = b
        seg_len = cum_b - cum_a
        if seg_len <= 0:
            continue

        ax = radians(lon_a - lon) * cos(lat0) * earth_r
        ay = radians(lat_a - lat) * earth_r
        bx = radians(lon_b - lon) * cos(lat0) * earth_r
        by = radians(lat_b - lat) * earth_r
        vx = bx - ax
        vy = by - ay
        denom = vx * vx + vy * vy
        if denom <= 0:
            continue

        t = max(0.0, min(1.0, -(ax * vx + ay * vy) / denom))
        cum = cum_a + t * seg_len
        if min_cum_m is not None and cum < min_cum_m:
            continue

        px = ax + t * vx
        py = ay + t * vy
        dist2 = px * px + py * py
        if best is None or dist2 < best[0]:
            best = (dist2, cum)

    if best is None and min_cum_m is not None:
        return _project_stop_to_shape_m(lat, lon, shape_points, min_cum_m=None)
    return None if best is None else best[1]


def _shape_stop_positions(trip_rows, stops, shape_points):
    coord = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
    positions = []
    min_cum = None
    for row in trip_rows:
        c = coord[row["stop_id"]]
        pos = _project_stop_to_shape_m(
            c["stop_lat"], c["stop_lon"], shape_points, min_cum_m=min_cum
        )
        positions.append(pos)
        if pos is not None:
            min_cum = pos
    return positions


def _fallback_stop_distance_m(a, b, coord):
    sa, sb = coord[a["stop_id"]], coord[b["stop_id"]]
    return haversine_m(sa["stop_lat"], sa["stop_lon"],
                       sb["stop_lat"], sb["stop_lon"])


def iter_valid_stop_pairs(trip_rows, stops, shape_points=None):
    """
    Yield (row_a, row_b, length_m) for every consecutive stop pair that survives
    the co-located-stop filter (length >= 1 m). build_segments AND
    add_grades_from_dem both consume this, so the i-th segment always matches the
    i-th yielded pair (dropping a duplicate stop never shifts grades/loads).

    When a GTFS shape is available, length_m is the driven distance along the
    shape polyline. Straight-line Haversine is only the fallback.
    """
    coord = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
    shape_pos = None
    if shape_points:
        shape_pos = _shape_stop_positions(trip_rows, stops, shape_points)
    for i, (a, b) in enumerate(zip(trip_rows[:-1], trip_rows[1:])):
        length_m = None
        if shape_pos:
            start_m, end_m = shape_pos[i], shape_pos[i + 1]
            if start_m is not None and end_m is not None and end_m > start_m:
                length_m = end_m - start_m
        if length_m is None:
            length_m = _fallback_stop_distance_m(a, b, coord)
        if length_m < 1:
            continue
        yield a, b, length_m


def build_segments(trip_rows, stops, passengers=20, cruise_factor=1.15,
                   v_min=3.0, v_max=22.0, shape_points=None):
    """Convert ordered stop_times rows into Segment objects (grade=0 for now)."""
    segments = []
    for a, b, length_m in iter_valid_stop_pairs(
        trip_rows, stops, shape_points=shape_points
    ):
        run_time = gtfs_time_to_seconds(b["arrival_time"]) - \
                   gtfs_time_to_seconds(a["departure_time"])
        if run_time > 0:
            v_cruise = min(max((length_m / run_time) * cruise_factor, v_min), v_max)
        else:
            v_cruise = 11.0
        dwell = gtfs_time_to_seconds(b["departure_time"]) - \
                gtfs_time_to_seconds(b["arrival_time"])
        dwell = max(dwell, 0.0)
        segments.append(Segment(length_m=length_m, grade=0.0,
                                 v_cruise_ms=v_cruise, dwell_s=dwell,
                                 passengers=passengers, run_time_s=run_time,
                                 from_stop_departure_time=a["departure_time"],
                                 to_stop_arrival_time=b["arrival_time"],
                                 to_stop_departure_time=b["departure_time"]))
    return segments


def add_grades_from_dem(segments, trip_rows, stops, dem_path=None,
                        elevation_data=None, srtm_cache_dir=None,
                        max_abs_grade=0.15, verbose=True, shape_points=None):
    """
    Fill Segment.grade from SRTM elevations (RouteZero's source):
        grade = (elev_end - elev_start) / segment_length   [rise/run fraction]
    Units, alignment, void and SRTM-noise notes as documented previously.
    """
    pairs = list(iter_valid_stop_pairs(trip_rows, stops, shape_points=shape_points))
    if len(pairs) != len(segments):
        raise ValueError(
            f"segment/pair mismatch ({len(segments)} vs {len(pairs)}); "
            "build_segments and add_grades_from_dem are out of sync."
        )
    if not segments:
        return segments

    if elevation_data is None:
        cache_dir = str(srtm_cache_dir) if srtm_cache_dir else ""
        elevation_data = srtm.get_data(local_cache_dir=cache_dir)

    coord = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
    elev_cache = {}

    def elevation_of(stop_id):
        if stop_id not in elev_cache:
            c = coord[stop_id]
            elev_cache[stop_id] = elevation_data.get_elevation(
                latitude=c["stop_lat"], longitude=c["stop_lon"])
        return elev_cache[stop_id]

    filled = missing = 0
    grades = []
    for seg, (a, b, length_m) in zip(segments, pairs):
        ea, eb = elevation_of(a["stop_id"]), elevation_of(b["stop_id"])
        if ea is None or eb is None or length_m <= 0:
            seg.grade = 0.0
            missing += 1
            continue
        g = (eb - ea) / length_m
        if max_abs_grade is not None:
            g = max(-max_abs_grade, min(max_abs_grade, g))
        seg.grade = g
        filled += 1
        grades.append(g)

    if verbose:
        if grades:
            print(f"  grades: filled {filled}/{len(segments)} from SRTM "
                  f"(range {min(grades) * 100:+.1f}%..{max(grades) * 100:+.1f}%)"
                  + (f"; {missing} flat (no SRTM value)" if missing else ""))
        else:
            print("  grades: SRTM returned no usable elevations -> all flat.")
    return segments


# -----------------------------------------------------------------------------
# One trip -> segments (build + grades + optional passenger loading)
#  >>> THIS is the single point where passenger_loading.py is invoked. <<<
# -----------------------------------------------------------------------------
def segments_for_trip(trip_rows, stops, passengers=20, elevation_data=None,
                      srtm_cache_dir=None, demand_profile=None,
                      crush_capacity=70, loading_kwargs=None,
                      weather_series=None, service_date=None,
                      weather_kwargs=None, shape_points=None,
                      shape_id=None,
                      signal_count_map=None, signal_source_map=None,
                      signal_event_map=None,
                      speed_cap_map=None, speed_cap_source_map=None,
                      verbose=False):
    """
    Full per-trip build. If `demand_profile` is given (a HourlyDemandProfile),
    passenger_loading.apply_passenger_loading() overwrites Segment.passengers
    using THIS trip's real start time -> hour -> demand factor. Otherwise the
    flat `passengers` assumption stands.

    If `signal_count_map` is given ({(from_stop_id, to_stop_id): n_signals},
    resolved once per route by resolve_signal_counts), add_traffic_signals()
    writes Segment.n_signals so the speed profile reflects stop-go behaviour.
    """
    segs = build_segments(
        trip_rows, stops, passengers=passengers, shape_points=shape_points
    )
    trip_id = str(trip_rows[0].get("trip_id", "")) if trip_rows else ""
    service_date_text = service_date.isoformat() if service_date else ""
    for idx, seg in enumerate(segs):
        seg.trip_id = trip_id
        seg.service_date = service_date_text
        seg.segment_index = idx
    segs = add_grades_from_dem(segs, trip_rows, stops,
                               elevation_data=elevation_data,
                               srtm_cache_dir=srtm_cache_dir, verbose=verbose,
                               shape_points=shape_points)
    if signal_count_map is not None:
        if not _HAS_SIGNALS:
            raise RuntimeError("signal_count_map given but traffic_signals.py "
                               "could not be imported.")
        segs = add_traffic_signals(segs, trip_rows, stops,
                                   count_map=signal_count_map,
                                   source_map=signal_source_map,
                                   event_map=signal_event_map,
                                   shape_points=shape_points, shape_id=shape_id,
                                   verbose=verbose)
    if speed_cap_map is not None:
        if not _HAS_SPEED_CAPS:
            raise RuntimeError("speed_cap_map given but speed_caps.py "
                               "could not be imported.")
        segs = add_speed_caps(segs, trip_rows, stops, cap_map=speed_cap_map,
                              source_map=speed_cap_source_map,
                              shape_points=shape_points, verbose=verbose)
    if demand_profile is not None:
        if not _HAS_LOADING:
            raise RuntimeError("demand_profile given but passenger_loading.py "
                               "could not be imported.")
        segs = apply_passenger_loading(segs, trip_rows, demand_profile,
                                       crush_capacity=crush_capacity,
                                       verbose=verbose, **(loading_kwargs or {}))
    if weather_series is not None:
        if not _HAS_WEATHER:
            raise RuntimeError("weather_series given but weather_loading.py "
                               "could not be imported.")
        if service_date is None:
            raise ValueError("service_date is required when weather loading is enabled")
        segs = apply_weather_loading(
            segs, trip_rows, weather_series, service_date=service_date,
            verbose=verbose, **(weather_kwargs or {})
        )
    return segs


# -----------------------------------------------------------------------------
# Results dataframes
# -----------------------------------------------------------------------------
def implausible_intervals(segments):
    """Indices of intervals whose driven length cannot fit their scheduled time."""
    return [i for i, seg in enumerate(segments)
            if seg.run_time_s and seg.run_time_s > 0
            and seg.length_m / seg.run_time_s * 3.6 > MAX_PLAUSIBLE_SCHEDULED_KMH]


def trip_results_dataframe(segs, trip_info, vehicle_params=None, soc0_pct=100.0,
                           motion_params=None):
    """Simulate one trip and prepend its route/trip metadata to every row."""
    if vehicle_params is None:
        vehicle_params = VehicleParams()
    df = simulate_route(segs, vehicle_params, soc0_pct=soc0_pct,
                        motion_params=motion_params)
    for col, value in reversed(list(trip_info.items())):
        if col in df.columns:
            if not df[col].eq(value).all():
                raise ValueError(f"Conflicting trip/result metadata: {col}")
            continue
        df.insert(0, col, value)
    return df


def _trip_meta_maps(trips):
    trip_meta = trips.set_index("trip_id")
    return {
        "direction_id": trip_meta["direction_id"].to_dict()
        if "direction_id" in trip_meta.columns else {},
        "service_id": trip_meta["service_id"].to_dict()
        if "service_id" in trip_meta.columns else {},
        "block_id": trip_meta["block_id"].to_dict()
        if "block_id" in trip_meta.columns else {},
    }


def _duty_groups(chosen, route_trips, service_by_trip, block_by_trip,
                 simulation_level):
    chosen = sorted(chosen, key=lambda tid: trip_start_seconds(route_trips[tid]))
    if simulation_level == "trip":
        return [
            ((service_by_trip.get(tid, ""), f"trip_{tid}"), [tid])
            for tid in chosen
        ]

    groups = {}
    for tid in chosen:
        service_id = str(service_by_trip.get(tid, ""))
        block_id = str(block_by_trip.get(tid, "") or f"trip_{tid}")
        groups.setdefault((service_id, block_id), []).append(tid)
    return sorted(
        groups.items(),
        key=lambda item: trip_start_seconds(route_trips[item[1][0]]),
    )


def safe_route_name(route_short_name) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(route_short_name).strip())
    return cleaned.strip("._") or "route"


def route_output_filename(route_short_name, service_date=None) -> str:
    route_name = safe_route_name(route_short_name)
    if service_date is None:
        return f"route_{route_name}_trips.csv"
    date_suffix = service_date.isoformat().replace("-", "_")
    return f"route_{route_name}_trips_{date_suffix}.csv"


def route_signal_event_output_filename(route_short_name, service_date=None) -> str:
    route_name = safe_route_name(route_short_name)
    if service_date is None:
        return f"route_{route_name}_signal_events.csv"
    date_suffix = service_date.isoformat().replace("-", "_")
    return f"route_{route_name}_signal_events_{date_suffix}.csv"


def _print_route_summary(rsn, df, n_trips):
    energy_col = "net_battery_energy_kWh"
    if energy_col not in df.columns:
        energy_col = "energy_kWh"
    g = (df.groupby(["trip_id", "trip_start_time"])[energy_col].sum()
         .reset_index().sort_values("trip_start_time"))
    print(f"\nRoute {rsn}: {n_trips} trip(s) processed")
    if not g.empty:
        trip_energy = g[energy_col]
        print(f"  net battery/trip  min {trip_energy.min():.1f} | "
              f"mean {trip_energy.mean():.1f} | max {trip_energy.max():.1f} kWh")
        busiest = g.loc[trip_energy.idxmax()]
        print(f"  busiest trip {busiest.trip_start_time} -> "
              f"{busiest[energy_col]:.1f} kWh")
    if {"duty_id", "duty_trip_index", "trip_end_soc_%"}.issubset(df.columns):
        duty = (
            df.sort_values(["duty_id", "duty_trip_index", "segment"])
            .groupby("duty_id")
            .tail(1)
        )
        if not duty.empty:
            print(f"  duties        {len(duty)} | final SoC min "
                  f"{duty['trip_end_soc_%'].min():.1f}% | mean "
                  f"{duty['trip_end_soc_%'].mean():.1f}%")

    # Motion diagnostics from beb_soc_model.py. These columns are present in the
    # refined model output and make traffic-signal runs auditable from the route
    # console output as well as from the CSV.
    diag_cols = {
        "n_effective_signal_stops",
        "signal_wait_s",
        "signal_wait_requested_s",
        "signal_wait_reduced_s",
        "schedule_delay_s",
        "schedule_infeasible",
        "actual_profile_time_s",
        "scheduled_run_time_s",
    }
    if diag_cols.issubset(df.columns):
        n_sig = int(df.get("n_signals", 0).sum()) if "n_signals" in df.columns else 0
        n_eff = int(df["n_effective_signal_stops"].sum())
        wait_h = df["signal_wait_s"].sum() / 3600.0
        requested_h = df["signal_wait_requested_s"].sum() / 3600.0
        reduced_h = df["signal_wait_reduced_s"].sum() / 3600.0
        delay_h = df["schedule_delay_s"].sum() / 3600.0
        infeasible = int(df["schedule_infeasible"].astype(bool).sum())
        scheduled_h = df["scheduled_run_time_s"].sum() / 3600.0
        actual_h = df["actual_profile_time_s"].sum() / 3600.0
        print(f"  signals diag  raw {n_sig} | effective stops {n_eff} | "
              f"modelled wait {wait_h:.2f} h "
              f"(requested {requested_h:.2f}, reduced {reduced_h:.2f})")
        print(f"  time diag     scheduled {scheduled_h:.2f} h | profile "
              f"{actual_h:.2f} h | delay {delay_h:.2f} h | "
              f"infeasible segments {infeasible}/{len(df)}")
    if {"speed_cap_ms", "speed_cap_source"}.issubset(df.columns):
        caps = df["speed_cap_ms"].astype(float)
        srcs = df["speed_cap_source"].astype(str).value_counts().to_dict()
        src_txt = ", ".join(f"{k} {v}" for k, v in sorted(srcs.items()))
        print(f"  speed caps    {caps.min() * 3.6:.0f}-{caps.max() * 3.6:.0f} "
              f"km/h (median {caps.median() * 3.6:.0f}) | sources: {src_txt}")


# -----------------------------------------------------------------------------
# Top-level: process MANY trips across MANY routes (one streaming pass)
# -----------------------------------------------------------------------------
def process_routes(gtfs_zip, route_short_names, output_dir, tables=None,
                   direction_id=0, day="monday", service_id=None,
                   service_date=None,
                   all_services=False, start_times=None, tolerance_s=900,
                   passengers=20, vehicle_params=None, elevation_data=None,
                   srtm_cache_dir=None, demand_profile=None, crush_capacity=70,
                   loading_kwargs=None, weather_series=None, weather_kwargs=None,
                   signals_enabled=False, signals_cache_path=None,
                   signal_events_cache_path=None,
                   signal_event_diagnostics_enabled=False,
                   signals_snap_m=DEFAULT_SNAP_RADIUS_M,
                   signals_relaxed_snap_m=DEFAULT_RELAXED_SNAP_RADIUS_M,
                   signals_cluster_m=DEFAULT_SIGNAL_CLUSTER_RADIUS_M,
                   signals_fallback_per_km=DEFAULT_FALLBACK_PER_KM,
                   signals_refresh=False, signals_frozen=False,
                   motion_params=None,
                   speed_caps_enabled=False, speed_caps_cache_path=None,
                   speed_caps_snap_m=DEFAULT_SPEED_SNAP_RADIUS_M,
                   speed_caps_sample_step_m=DEFAULT_SAMPLE_STEP_M,
                   speed_caps_min_coverage=DEFAULT_MIN_COVERAGE_FRAC,
                   speed_caps_default_kmh=DEFAULT_FALLBACK_CAP_KMH,
                   speed_caps_max_kmh=DEFAULT_MAX_CAP_KMH,
                   speed_caps_refresh=False,
                   simulation_level="block", placement_registry_path=None,
                   run_records=None, input_source_paths=(), osm_snapshot=None,
                   shape_repairs=None):
    """
    For each route: pick trips (by service day, then by start time or ALL),
    build/simulate each, and save one CSV per route with every trip stacked
    (columns include trip_id + trip_start_time). Results go in a subfolder of
    output_dir named traffic_signals_off or traffic_signals_on_<signal_fidelity>.
    Returns the saved paths.
    """
    if tables is None:
        tables = load_small_tables(gtfs_zip)
    if len(tables) == 4:
        routes, trips, stops, calendar = tables
        calendar_dates = None
    else:
        routes, trips, stops, calendar, calendar_dates = tables
    if vehicle_params is None:
        vehicle_params = VehicleParams()
    if motion_params is None:
        motion_params = MotionParams()
    from signal_placement import (
        F1_VARIANTS, PlacementRegistry, PlacementUnavailable, file_hash,
        scientific_hash, source_inventory, context_evidence,
    )
    registered_f1 = motion_params.signal_fidelity in F1_VARIANTS
    registry = PlacementRegistry(placement_registry_path) if registered_f1 or run_records is not None else None
    feed_sha256 = file_hash(gtfs_zip) if registry is not None else None

    # 1) which service_ids count as "the day"
    if all_services or calendar is None:
        service_ids = None
        if calendar is None and not all_services:
            print("calendar.txt not found -> processing ALL services (may mix "
                  "weekday/weekend trips).")
    elif service_id is not None:
        service_ids = {str(service_id)}
    elif service_date is not None:
        service_ids = service_ids_for_date(calendar, calendar_dates, service_date)
    else:
        service_ids = service_ids_for_day(calendar, day)

    simulation_level = str(simulation_level or "block").lower()
    if simulation_level not in {"block", "trip"}:
        raise ValueError("simulation_level must be 'block' or 'trip'")
    route_direction_id = None if simulation_level == "block" else direction_id
    if simulation_level == "block" and direction_id is not None:
        print("Block-level simulation: using all directions so duties are complete.")

    # 2) candidate trips per route, and the union to scan for
    meta = _trip_meta_maps(trips)
    dir_by_trip = meta["direction_id"]
    service_by_trip = meta["service_id"]
    block_by_trip = meta["block_id"]
    plan, union = {}, set()
    for rsn in route_short_names:
        route_id, long_name = resolve_route_id(routes, rsn)
        short_name = routes.loc[routes["route_id"] == route_id, "route_short_name"].iloc[0]
        tids = trips_for_route(trips, route_id, route_direction_id, service_ids)
        plan[rsn] = (route_id, long_name, tids, str(short_name))
        union.update(tids)

    # 3) ONE streaming pass over stop_times.txt for every wanted trip
    print(f"Scanning stop_times.txt once for {len(union)} candidate trip(s)...")
    by_trip_all = stop_times_for_trips(gtfs_zip, union)
    shapes_by_id, shape_id_by_trip = load_shapes_for_trips(
        gtfs_zip, trips, union, shape_repairs
    )
    if shapes_by_id:
        print(f"Loaded {len(shapes_by_id)} GTFS shape(s) for driven distances.")
    else:
        print("GTFS shapes unavailable for selected trips -> using stop-to-stop "
              "Haversine distances.")

    # Signals: load the shared cache once (accumulates across routes) and the
    # stop coordinate lookup used to resolve any pairs not already cached.
    signal_cache = None
    signal_event_cache = None
    stop_coords = None
    if signals_enabled and _HAS_SIGNALS:
        signal_cache = (load_signal_cache(signals_cache_path)
                        if signals_cache_path else {})
        signal_event_cache = (load_signal_event_cache(signal_events_cache_path)
                              if signal_events_cache_path else {})

    # Speed caps use the same directed-stop-pair keying and shape geometry as
    # the signal cache, resolved once per route and reused across dates.
    speed_cap_cache = None
    if speed_caps_enabled and _HAS_SPEED_CAPS:
        speed_cap_cache = (load_speed_cap_cache(speed_caps_cache_path)
                           if speed_caps_cache_path else {})

    if (signals_enabled and _HAS_SIGNALS) or (speed_caps_enabled
                                              and _HAS_SPEED_CAPS):
        _c = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
        stop_coords = {sid: (v["stop_lat"], v["stop_lon"]) for sid, v in _c.items()}

    # 4) per route: select + build + simulate + save
    signal_folder = (
        f"traffic_signals_on_{motion_params.signal_fidelity}"
        if signals_enabled and _HAS_SIGNALS else "traffic_signals_off"
    )
    output_dir = Path(output_dir) / signal_folder
    if registered_f1:
        output_dir = output_dir / f"seed_{motion_params.signal_location_seed}_p{motion_params.signal_placement_index}_b{motion_params.signal_behaviour_index}_seed{motion_params.behaviour_seed()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if registered_f1 and placement_registry_path:
        target = output_dir / "source_registry.json"
        if Path(placement_registry_path).resolve() != target.resolve():
            target.write_bytes(Path(placement_registry_path).read_bytes())
    saved = []
    for rsn, (route_id, long_name, tids, short_name) in plan.items():
        route_trips = {t: by_trip_all.get(t, []) for t in tids}
        route_trips = {t: r for t, r in route_trips.items() if r}
        chosen = select_trips_by_start(route_trips, start_times, tolerance_s)
        frames = []
        signal_event_diag_rows = []

        # Resolve traffic-signal counts and OSM speed caps ONCE per route
        # (both static per directed stop pair): cache hit -> OSM fetch (once,
        # route bbox) -> fallback. They share the same shape sub-path geometry.
        route_signal_count_map = route_signal_source_map = None
        route_speed_cap_map = route_speed_cap_source_map = None
        want_signals = signals_enabled and _HAS_SIGNALS and chosen
        want_caps = speed_caps_enabled and _HAS_SPEED_CAPS and chosen
        route_pairs = route_geom = None
        signal_route_pairs = signal_route_geom = None
        if want_caps:
            shp_by_trip = {
                tid: shapes_by_id.get(shape_id_by_trip.get(str(tid), ""))
                for tid in chosen
            }
            route_pairs, route_geom = route_stop_pairs(
                ((tid, route_trips[tid]) for tid in chosen), stops, shp_by_trip
            )
        if want_signals:
            shp_by_trip = {
                tid: shapes_by_id.get(shape_id_by_trip.get(str(tid), ""))
                for tid in chosen
            }
            shp_id_by_trip = {
                str(tid): shape_id_by_trip.get(str(tid), "")
                for tid in chosen
            }
            signal_route_pairs, signal_route_geom = route_stop_pairs(
                ((tid, route_trips[tid]) for tid in chosen), stops, shp_by_trip,
                shape_id_by_trip=shp_id_by_trip, include_geometry_id=True
            )
        if want_signals:
            (route_signal_count_map, route_signal_source_map,
             sstats) = resolve_signal_counts(
                signal_route_pairs, stop_coords, snap_radius_m=signals_snap_m,
                relaxed_snap_radius_m=signals_relaxed_snap_m,
                cluster_radius_m=signals_cluster_m,
                fallback_per_km=signals_fallback_per_km,
                fetch_fn=partial(fetch_traffic_signals_bbox, snapshot=osm_snapshot),
                geom_map=signal_route_geom,
                cache=signal_cache, cache_path=signals_cache_path,
                event_cache=signal_event_cache,
                event_cache_path=signal_events_cache_path,
                refresh=signals_refresh, verbose=True, frozen=signals_frozen)
            if run_records is not None and sstats["fallback"]:
                raise ValueError(f"Matched comparison refuses estimated signals on route "
                                 f"{rsn}: the OSM fetch failed for {sstats['fallback']} pair(s).")
            print(f"  signals: route {rsn} -> {sstats['n_pairs']} stop-pair(s) "
                  f"[cache {sstats['cache_hit']}, OSM {sstats['osm']}, "
                  f"relaxed {sstats.get('osm_relaxed', 0)}, "
                  f"fallback {sstats['fallback']}] "
                  f"-> {sstats['total_signals']} signals "
                  f"({sstats.get('event_rows', 0)} event rows)")
        if want_caps:
            (route_speed_cap_map, route_speed_cap_source_map,
             cstats) = resolve_speed_caps(
                route_pairs, stop_coords, snap_radius_m=speed_caps_snap_m,
                sample_step_m=speed_caps_sample_step_m,
                min_coverage_frac=speed_caps_min_coverage,
                default_cap_kmh=speed_caps_default_kmh,
                max_cap_kmh=speed_caps_max_kmh, geom_map=route_geom,
                fetch_fn=partial(fetch_speed_edges_bbox, snapshot=osm_snapshot),
                cache=speed_cap_cache, cache_path=speed_caps_cache_path,
                refresh=speed_caps_refresh, verbose=True)
            if run_records is not None and cstats["fallback"]:
                raise ValueError(f"Matched comparison refuses default speed caps on route "
                                 f"{rsn}: the OSM fetch failed for {cstats['fallback']} pair(s).")
            print(f"  speed caps: route {rsn} -> {cstats['n_pairs']} "
                  f"stop-pair(s) [cache {cstats['cache_hit']}, "
                  f"OSM {cstats['osm']}, imputed {cstats['imputed']}, "
                  f"low-coverage {cstats['fallback_low_coverage']}, "
                  f"fallback {cstats['fallback']}]")

        groups = _duty_groups(
            chosen, route_trips, service_by_trip, block_by_trip, simulation_level
        )
        for duty_index, ((duty_service_id, duty_block_id), duty_tids) in enumerate(
            groups
        ):
            duty_id = f"{duty_service_id}:{duty_block_id}"
            duty_start_time = seconds_to_hhmmss(
                trip_start_seconds(route_trips[duty_tids[0]])
            )
            duty_soc = 100.0
            duty_blocker = None
            for duty_trip_index, tid in enumerate(duty_tids):
                rows = route_trips[tid]
                trip_start_soc = duty_soc
                shape_points = shapes_by_id.get(shape_id_by_trip.get(str(tid), ""))
                trip_shape_id = shape_id_by_trip.get(str(tid), "")
                segs = segments_for_trip(
                    rows, stops, passengers=passengers,
                    elevation_data=elevation_data, srtm_cache_dir=srtm_cache_dir,
                    demand_profile=demand_profile, crush_capacity=crush_capacity,
                    loading_kwargs=loading_kwargs,
                    weather_series=weather_series, service_date=service_date,
                    weather_kwargs=weather_kwargs,
                    shape_points=shape_points,
                    shape_id=trip_shape_id,
                    signal_count_map=route_signal_count_map,
                    signal_source_map=route_signal_source_map,
                    signal_event_map=signal_event_cache,
                    speed_cap_map=route_speed_cap_map,
                    speed_cap_source_map=route_speed_cap_source_map,
                )
                implausible = implausible_intervals(segs)
                if implausible:
                    if run_records is not None:
                        raise ValueError(f"Matched comparison refuses trip {tid}: interval(s) "
                                         f"{implausible} are scheduled above "
                                         f"{MAX_PLAUSIBLE_SCHEDULED_KMH:.0f} km/h.")
                    print(f"  WARNING: trip {tid} interval(s) {implausible} are scheduled above "
                          f"{MAX_PLAUSIBLE_SCHEDULED_KMH:.0f} km/h; check its shape.")
                info = {
                    "route_short_name": short_name,
                    "route_long_name": long_name,
                    "route_id": route_id,
                    "service_id": service_by_trip.get(tid, ""),
                    "block_id": block_by_trip.get(tid, ""),
                    "duty_id": duty_id,
                    "duty_index": duty_index,
                    "duty_trip_index": duty_trip_index,
                    "duty_trip_count": len(duty_tids),
                    "duty_start_time": duty_start_time,
                    "direction_id": dir_by_trip.get(tid, ""),
                    "trip_id": tid,
                    "simulation_date": service_date.isoformat()
                    if service_date else "",
                    "trip_start_time": seconds_to_hhmmss(trip_start_seconds(rows)),
                    "trip_start_soc_%": round(trip_start_soc, 2),
                }
                context = None
                record = None
                blocker = None
                if registry is not None:
                    binding = {"feed_sha256": feed_sha256, "route_id": str(route_id),
                        "direction_id": str(dir_by_trip.get(tid, "")), "shape_id": str(trip_shape_id),
                        "stop_visits": [str(row["stop_id"]) for row in rows]}
                    pattern_id = registry.pattern_id(binding)
                    if registered_f1:
                        info["pattern_id"] = pattern_id
                    physical = [{k: v for k, v in asdict(seg).items() if k not in
                        {"n_signals", "signal_source", "signal_events", "signal_event_diagnostics"}} for seg in segs]
                    non_signal_motion = {k: v for k, v in asdict(motion_params).items() if k not in
                        {"signal_fidelity", "signal_location_seed", "signal_placement_index",
                         "signal_behaviour_index", "signal_behaviour_seeds", "signal_random_seed"}}
                    record = {"feed_sha256": feed_sha256, "route_id": str(route_id),
                        "pattern_id": pattern_id, "trip_id": str(tid), "service_date": service_date.isoformat() if service_date else "",
                        "fidelity": motion_params.signal_fidelity,
                        "placement_index": motion_params.signal_placement_index if registered_f1 else None,
                        "behaviour_index": None if motion_params.signal_fidelity == "F0_none" else motion_params.signal_behaviour_index,
                        "analysis_mode": simulation_level, "scenario_id": "normal-GTFS-runner",
                        "timing_rule": motion_params.signal_time_policy,
                        "B1_stratum": f"{motion_params.gtfs_stop_policy}:q={motion_params.q_stop}:g={motion_params.gtfs_stop_random_seed}",
                        "non_signal_input_signature": scientific_hash({"segments": physical,
                            "vehicle": asdict(vehicle_params), "motion": non_signal_motion,
                            "simulation_level": simulation_level, "duty_id": duty_id,
                            "duty_initial_soc_pct": 100.0}),
                        "source_signature": scientific_hash([asdict(seg) for seg in segs]),
                        "registry_sha256": registry.sha256,
                        "motion": asdict(motion_params), "vehicle": asdict(vehicle_params),
                        "trip_start_soc_pct": trip_start_soc, "event_ledger": []}
                    if registered_f1:
                        try:
                            if not signals_enabled or not _HAS_SIGNALS:
                                raise PlacementUnavailable("BLOCKED_INVENTORY", "F1-U/J require signal extraction enabled")
                            context = registry.bind(segs, **binding)
                            for seg in segs:
                                seg.signal_placement_context = context
                            name = "pattern_" + scientific_hash(context.geometry.pattern_id)[:16] + ".json"
                            (output_dir / name).write_text(json.dumps(context_evidence(context, motion_params),
                                sort_keys=True, indent=2), encoding="utf-8")
                        except PlacementUnavailable as exc:
                            blocker = exc
                if registered_f1 and duty_blocker is not None:
                    blocker = PlacementUnavailable("BLOCKED_PRIOR_DUTY", duty_blocker)
                try:
                    if blocker:
                        raise blocker
                    frame = trip_results_dataframe(
                        segs, info, vehicle_params, soc0_pct=trip_start_soc,
                        motion_params=motion_params,
                    )
                except PlacementUnavailable as exc:
                    if not registered_f1:
                        raise
                    if duty_blocker is None:
                        duty_blocker = str(exc)
                    details = {**exc.diagnostics, "placement_status": exc.status, "reason": exc.reason}
                    frame = pd.DataFrame([{**info, "signal_fidelity": motion_params.signal_fidelity, **details}])
                    frames.append(frame)
                    if run_records is not None:
                        record.update(status=exc.status, blocker=exc.reason, result=None, readiness=details)
                        record["run_id"] = scientific_hash(record)
                        run_records.append(record)
                    print(f"  trip {tid}: {exc}")
                    continue
                if run_records is not None:
                    raw = frame.attrs["unrounded_energy"]
                    ledger = [dict(row) for seg in segs for row in seg.signal_event_diagnostics or []]
                    if motion_params.signal_fidelity == "F2_spatial":
                        try:
                            _, ledger = source_inventory(segs, motion_params)
                        except PlacementUnavailable:
                            pass
                    record.update(status="complete", blocker="", result={
                        "net_kWh": sum(row["net_kWh"] for row in raw),
                        "distance_m": sum(seg.length_m for seg in segs),
                        "components": raw, "soc_end_unrounded_pct": frame.attrs["soc_end_unrounded_pct"]},
                        event_ledger=ledger,
                        readiness={"registered_inputs": context is not None})
                    record["run_id"] = scientific_hash(record)
                    run_records.append(record)
                if signal_event_diagnostics_enabled or registered_f1:
                    for seg_index, seg in enumerate(segs):
                        for row in getattr(seg, "signal_event_diagnostics", None) or []:
                            signal_event_diag_rows.append({
                                **info,
                                "segment": seg_index,
                                **row,
                            })
                duty_soc = float(frame["SoC_end_%"].iloc[-1])
                frame["trip_end_soc_%"] = round(duty_soc, 2)
                frames.append(frame)

        if not frames:
            print(f"Route {rsn}: no trips selected.")
            continue
        df = pd.concat(frames, ignore_index=True)
        out_path = output_dir / route_output_filename(rsn, service_date)
        df.to_csv(out_path, index=False)
        if registered_f1 or (signal_event_diagnostics_enabled and signal_event_diag_rows):
            diag_path = output_dir / route_signal_event_output_filename(rsn, service_date)
            columns = None if signal_event_diag_rows else ["trip_id", "pattern_id", "source_ordinal",
                "synthetic_ordinal", "source_encounter_id", "candidate_id", "receiving_interval",
                "stop_probability", "random_value", "stop_decision", "red_wait_s_requested"]
            pd.DataFrame(signal_event_diag_rows, columns=columns).to_csv(diag_path, index=False)
            print(f"  saved signal diagnostics: {diag_path}")
        if "net_battery_energy_kWh" in df:
            summary = df[df["net_battery_energy_kWh"].notna()]
            _print_route_summary(rsn, summary, summary["trip_id"].nunique())
        print(f"  saved: {out_path}")
        saved.append(out_path)
    return saved


# -----------------------------------------------------------------------------
# Optional: load an hourly demand profile from CSV
# -----------------------------------------------------------------------------
# Two shapes are supported. Both keep ALL CSV/pandas I/O on this side; the model
# (HourlyDemandProfile) stays pure in passenger_loading.py.
#
#   LONG / multi-city (the bus_hourly_average_*_long.csv file):
#       columns: city, hour, avg_hourly_flow_percent[, years_available]
#       -> use load_city_demand_profile(path, city). Preferred: tidy, carries
#          per-city provenance (years_available), scales to new cities without
#          a schema change.
#   GENERIC single profile:
#       columns: hour, <percent|pct|share|fraction|flow>
#       -> use load_demand_profile_csv(path).
# -----------------------------------------------------------------------------
LONG_DEMAND_COLS = {"city", "hour", "avg_hourly_flow_percent"}


def available_cities(path):
    """City names present in a long-format demand CSV ([] if not that shape)."""
    df = pd.read_csv(path)
    if "city" not in df.columns:
        return []
    return sorted(df["city"].astype(str).unique())


def load_city_demand_profile(path, city):
    """
    Build a HourlyDemandProfile for one city from the long-format hourly CSV
    (city, hour, avg_hourly_flow_percent). City match is case-insensitive.
    """
    if not _HAS_LOADING:
        raise RuntimeError("passenger_loading.py is required for demand profiles.")
    df = pd.read_csv(path)
    if not LONG_DEMAND_COLS.issubset(df.columns):
        raise ValueError(f"{path} is not the long-format demand CSV "
                         f"(need columns {sorted(LONG_DEMAND_COLS)}).")
    sub = df[df["city"].astype(str).str.lower() == str(city).lower()]
    if sub.empty:
        opts = ", ".join(sorted(df["city"].astype(str).unique()))
        raise ValueError(f"city {city!r} not found in {path}. Available: {opts}")
    sub = sub.sort_values("hour")
    if sub["hour"].nunique() != 24:
        print(f"  warning: {city} has {sub['hour'].nunique()} hours, not 24.")
    yrs = sub["years_available"].iloc[0] if "years_available" in sub.columns else "?"
    print(f"  demand profile: {city} (avg over {yrs} years, 2013-2023)")
    mapping = {int(h): float(v)
               for h, v in zip(sub["hour"], sub["avg_hourly_flow_percent"])}
    return HourlyDemandProfile.from_percent(mapping)


def load_demand_profile_csv(path, city=None):
    """
    Load a demand profile from CSV. If the file is long/multi-city, a `city`
    must be given (use available_cities(path) to list them); otherwise the file
    is treated as a single generic hour/percent profile.
    """
    if not _HAS_LOADING:
        raise RuntimeError("passenger_loading.py is required for --demand-csv.")
    df = pd.read_csv(path)
    if "city" in df.columns:
        if not city:
            raise ValueError(
                f"{path} is multi-city; pass a city. "
                f"Available: {', '.join(available_cities(path))}")
        return load_city_demand_profile(path, city)
    cols = {c.lower(): c for c in df.columns}
    hcol = cols.get("hour")
    vcol = next((cols[k] for k in ("percent", "pct", "share", "fraction", "flow")
                 if k in cols), None)
    if hcol is None or vcol is None:
        raise ValueError("demand CSV needs an 'hour' column and a "
                         "percent/share/flow column.")
    if df[hcol].nunique() != 24:
        print(f"  warning: demand profile has {df[hcol].nunique()} hours, not 24.")
    mapping = {int(h): float(v) for h, v in zip(df[hcol], df[vcol])}
    return HourlyDemandProfile.from_percent(mapping)


def find_demand_csv(preferred=None,
                    filename="bus_hourly_average_2013_2023_long.csv"):
    """
    Return the first existing demand CSV, or None. The single rigid path was the
    bug: PROJECT_ROOT = parents[1] of the script, so DEFAULT_DEMAND_CSV often
    pointed somewhere the file wasn't, loading silently fell back to flat 20.
    This searches the obvious places instead: the preferred path, then the
    script dir, the project root, and the current dir -- each also under data/
    and data/raw/.
    """
    candidates = []
    if preferred:
        candidates.append(Path(preferred))
        filename = Path(preferred).name  # honour a custom filename in the search
    script_dir = Path(__file__).resolve().parent
    for root in (script_dir, PROJECT_ROOT, Path.cwd()):
        for sub in (Path("."), Path("data"), Path("data") / "raw"):
            candidates.append(root / sub / filename)
    seen = set()
    for c in candidates:
        c = c.expanduser()
        if str(c) in seen:
            continue
        seen.add(str(c))
        if c.is_file():
            return c
    return None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_route_list(route_args, routes_csv):
    routes = []
    for value in route_args:
        routes.extend(part.strip() for part in value.split(","))
    if routes_csv:
        routes.extend(part.strip() for part in routes_csv.split(","))
    return [r for r in routes if r]


def parse_args():
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", help="Path to a model YAML config file.")
    config_args, _ = base.parse_known_args()
    config_path = config_args.config
    simulation_cfg = get_section("simulation", config_path)
    gtfs_cfg = get_section("gtfs", config_path)
    loading_cfg = get_section("passenger_loading", config_path)
    weather_cfg = get_section("weather", config_path)
    signals_cfg = get_section("traffic_signals", config_path)
    speed_caps_cfg = get_section("speed_caps", config_path)

    default_gtfs = get_path("gtfs_zip", config_path)
    default_output_dir = get_path("gtfs_output_dir", config_path)
    default_srtm_cache = get_path("srtm_cache_dir", config_path)
    default_demand_csv = get_path("passenger_loading_csv", config_path)
    default_weather_csv = get_path("weather_csv", config_path)
    default_signals_csv = get_path("traffic_signals_csv", config_path)
    default_signal_events_csv = get_path("traffic_signal_events_csv", config_path)
    default_speed_caps_csv = get_path("speed_caps_csv", config_path)

    p = argparse.ArgumentParser(
        description="GTFS route short names -> per-trip BEB segment CSVs.",
        parents=[base])
    p.add_argument("route_short_names", nargs="*",
                   help="Route short names or exact route_ids, e.g. 102 41 15 "
                        "(commas allowed). Ambiguous short names are rejected.")
    p.add_argument("--routes", help="Comma-separated route short names or route_ids.")
    p.add_argument("--gtfs", default=str(default_gtfs))
    p.add_argument("--output-dir", default=str(default_output_dir),
                   help="Parent output directory; route CSVs are grouped in a "
                        "traffic_signals_off or traffic_signals_on_<fidelity> subfolder.")
    p.add_argument("--signal-fidelity", choices=["F0_none", "F1-U", "F1-J", "F2_spatial", "F1_frequency"],
                   help="Override motion.signal_fidelity; F1_frequency is historical trip-keyed F1.")
    p.add_argument("--solver-mode", choices=["segment_legacy", "trip_continuous"])
    p.add_argument("--placement-index", type=int, help="Registered infrastructure realisation index.")
    p.add_argument("--behaviour-index", type=int, help="Registered signal-behaviour realisation index.")
    p.add_argument("--signal-placement-registry", default=str(get_path("signal_placement_registry", config_path) or ""),
                   help="Versioned accepted pattern/count/inventory and candidate evidence JSON.")
    p.add_argument("--compare-fidelities", action="store_true",
                   help="Run matched F0/F1-U/F1-J/F2 cells and retain unavailable cases; requires trip_continuous.")
    p.add_argument("--direction-id", type=int,
                   default=int(gtfs_cfg.get("direction_id", 0)))
    p.add_argument("--any-direction", action="store_true",
                   help="Use trips from all directions.")
    p.add_argument("--simulation-level", choices=["block", "trip"],
                   default=gtfs_cfg.get("simulation_level", "block"),
                   help="block chains trips by service_id+block_id; trip resets "
                        "SoC at each trip.")
    # which day / service
    p.add_argument("--day", default=gtfs_cfg.get("day", "monday"), choices=WEEKDAYS,
                   help="Weekday whose services to use (via calendar.txt).")
    p.add_argument("--service-id", help="Force a specific service_id.")
    p.add_argument("--all-services", action="store_true",
                   help="Ignore the calendar filter (process every service).")
    p.add_argument("--date", default=simulation_cfg.get("date", ""),
                   help="YYYY-MM-DD scenario date. Used for weather lookup and "
                        "GTFS service selection when possible.")
    p.add_argument("--osm-snapshot", default=simulation_cfg.get("osm_snapshot_utc"),
                   help="UTC timestamp pinned on OSM signal/speed-cap fetches, e.g. "
                        "2026-08-26T16:04:56Z. Only affects rows actually fetched.")
    # which trips
    p.add_argument("--start-times",
                   help="Comma-separated HH:MM[:SS]; nearest trip to each is "
                        "used. Omit to process ALL trips on the service day.")
    p.add_argument("--tolerance-min", type=int,
                   default=int(gtfs_cfg.get("tolerance_min", 15)),
                   help="Max minutes between a requested and an actual start.")
    # vehicle / loading
    p.add_argument("--passengers", type=int,
                   default=int(gtfs_cfg.get("flat_passengers", 20)),
                   help="Flat fallback load used when no demand profile is set.")
    p.add_argument("--demand-csv", default=str(default_demand_csv or ""),
                   help="Override the passenger_loading_csv set in the config. "
                        "Long/multi-city needs --demand-city.")
    p.add_argument("--demand-city", default=loading_cfg.get("demand_city", "Cork"),
                   help="Override passenger_loading.demand_city (e.g. Cork, Dublin, "
                        "Galway, Limerick, Waterford).")
    p.add_argument("--no-demand", action="store_true",
                   help="Ignore the demand profile; use the flat --passengers load.")
    p.add_argument("--weather-csv", default=str(default_weather_csv or ""),
                   help="Override the weather_csv set in the config.")
    p.add_argument("--no-weather", action="store_true",
                   default=not bool(weather_cfg.get("enabled", True)),
                   help="Ignore historical weather/HVAC loading.")
    # traffic signals
    signal_toggle = p.add_mutually_exclusive_group()
    signal_toggle.add_argument("--signals", dest="signals_enabled",
                               action="store_true",
                               help="Enable OSM traffic-signal stop-go loading.")
    signal_toggle.add_argument("--no-signals", dest="signals_enabled",
                               action="store_false",
                               help="Ignore OSM traffic-signal stop-go loading.")
    p.set_defaults(signals_enabled=bool(signals_cfg.get("enabled", True)))
    p.add_argument("--signals-csv", default=str(default_signals_csv or ""),
                   help="Traffic-signal count cache CSV (keyed by stop pair). "
                        "Fetched once and reused across dates/routes.")
    p.add_argument("--signal-events-csv",
                   default=str(default_signal_events_csv or ""),
                   help="Event-level traffic-signal cache CSV, one row per "
                        "clustered route-relevant signal.")
    p.add_argument("--signals-snap-m", type=float,
                   default=float(signals_cfg.get("snap_radius_m",
                                                 DEFAULT_SNAP_RADIUS_M)),
                   help="Max distance (m) for a signal to count on a stop pair.")
    p.add_argument("--signals-relaxed-snap-m", type=float,
                   default=float(signals_cfg.get("relaxed_snap_radius_m",
                                                 DEFAULT_RELAXED_SNAP_RADIUS_M)),
                   help="Second-pass snap distance used only when strict count is zero.")
    p.add_argument("--signals-cluster-m", type=float,
                   default=float(signals_cfg.get("cluster_radius_m",
                                                 DEFAULT_SIGNAL_CLUSTER_RADIUS_M)),
                   help="Merge OSM signal nodes closer than this along the route.")
    p.add_argument("--signals-fallback-per-km", type=float,
                   default=float(signals_cfg.get("fallback_per_km",
                                                 DEFAULT_FALLBACK_PER_KM)),
                   help="Signals/km assumed when OSM is unavailable (fallback).")
    p.add_argument("--refresh-signals", action="store_true",
                   default=bool(signals_cfg.get("refresh", False)),
                   help="Ignore cached signal counts and re-fetch from OSM.")
    p.add_argument("--signal-event-diagnostics", action="store_true",
                   default=bool(signals_cfg.get("write_event_diagnostics", False)),
                   help="Write optional per-signal stop/pass diagnostics CSV.")
    # OSM speed caps (independent physical speed limit per segment)
    caps_toggle = p.add_mutually_exclusive_group()
    caps_toggle.add_argument("--speed-caps", dest="speed_caps_enabled",
                             action="store_true",
                             help="Enable OSM maxspeed -> per-segment speed cap.")
    caps_toggle.add_argument("--no-speed-caps", dest="speed_caps_enabled",
                             action="store_false",
                             help="Skip OSM speed caps; use the config default "
                                  "cap (motion.default_speed_cap_ms).")
    p.set_defaults(speed_caps_enabled=bool(speed_caps_cfg.get("enabled", True)))
    p.add_argument("--speed-caps-csv", default=str(default_speed_caps_csv or ""),
                   help="Speed-cap cache CSV (keyed by stop pair). Fetched once "
                        "and reused across dates/routes.")
    p.add_argument("--speed-caps-snap-m", type=float,
                   default=float(speed_caps_cfg.get(
                       "snap_radius_m", DEFAULT_SPEED_SNAP_RADIUS_M)),
                   help="Max distance (m) from a route sample to an OSM way.")
    p.add_argument("--speed-caps-sample-step-m", type=float,
                   default=float(speed_caps_cfg.get(
                       "sample_step_m", DEFAULT_SAMPLE_STEP_M)),
                   help="Sample spacing (m) along the shape sub-path; the cap "
                        "is the length-weighted harmonic mean of matches.")
    p.add_argument("--speed-caps-min-coverage", type=float,
                   default=float(speed_caps_cfg.get(
                       "min_coverage_frac", DEFAULT_MIN_COVERAGE_FRAC)),
                   help="Matched-length fraction below which the default cap "
                        "is used instead.")
    p.add_argument("--speed-caps-default-kmh", type=float,
                   default=float(speed_caps_cfg.get(
                       "default_cap_kmh", DEFAULT_FALLBACK_CAP_KMH)),
                   help="Fallback cap (km/h) when OSM is unavailable/uncovered.")
    p.add_argument("--speed-caps-max-kmh", type=float,
                   default=float(speed_caps_cfg.get(
                       "max_cap_kmh", DEFAULT_MAX_CAP_KMH)),
                   help="Sanity clamp (km/h) on resolved caps.")
    p.add_argument("--refresh-speed-caps", action="store_true",
                   default=bool(speed_caps_cfg.get("refresh", False)),
                   help="Ignore cached speed caps and re-fetch from OSM.")
    p.add_argument("--crush-capacity", type=int,
                   default=int(loading_cfg.get("crush_capacity", 70)),
                   help="Override passenger_loading.crush_capacity.")
    p.add_argument("--board-frac", type=float,
                   default=float(loading_cfg.get("board_frac", 0.2)))
    p.add_argument("--alight-frac", type=float,
                   default=float(loading_cfg.get("alight_frac", 0.2)))
    p.add_argument("--floor-frac", type=float,
                   default=float(loading_cfg.get("floor_frac", 0.0)))
    p.add_argument("--hour-mode", default=loading_cfg.get("hour_mode", "midpoint"),
                   choices=["midpoint", "start"])
    p.add_argument("--load-shape", default=loading_cfg.get("load_shape", "beta"),
                   choices=["beta", "trapezoid", "triangular", "flat"],
                   help="Spatial load profile along the route.")
    p.add_argument("--board-pos", type=float,
                   default=float(loading_cfg.get("board_pos", 0.25)),
                   help="beta shape: where boardings concentrate (0=start,1=end).")
    p.add_argument("--alight-pos", type=float,
                   default=float(loading_cfg.get("alight_pos", 0.75)),
                   help="beta shape: where alightings concentrate (raise for a "
                        "route that fills toward a city-centre terminus).")
    p.add_argument("--concentration", type=float,
                   default=float(loading_cfg.get("concentration", 6.0)),
                   help="beta shape: peak sharpness (higher = tighter peak).")
    p.add_argument("--srtm-cache-dir", default=str(default_srtm_cache))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gtfs_cfg = get_section("gtfs", args.config)
    loading_cfg = get_section("passenger_loading", args.config)
    weather_cfg = get_section("weather", args.config)
    route_short_names = parse_route_list(args.route_short_names, args.routes)
    if not route_short_names:
        route_short_names = [str(r) for r in gtfs_cfg.get("default_routes", ["208"])]

    start_times = ([s.strip() for s in args.start_times.split(",") if s.strip()]
                   if args.start_times else None)
    direction_id = None if args.any_direction else args.direction_id
    service_date = parse_service_date(args.date)
    if service_date is not None:
        args.day = service_date.strftime("%A").lower()

    # Passenger loading is driven by configs/model.yaml, so it works with no CLI
    # flags. --no-demand or a missing file -> flat load.
    demand_profile = None
    if args.no_demand or not loading_cfg.get("enabled", True):
        print(f"Passenger loading: OFF (--no-demand) -> flat "
              f"{args.passengers} pax/segment.")
    elif not args.demand_csv:
        print(f"Passenger loading: OFF (passenger_loading_csv is empty) -> flat "
              f"{args.passengers} pax/segment.")
        print("  To enable: set paths.passenger_loading_csv in configs/model.yaml,")
        print("  or pass --demand-csv <path>.")
    elif args.demand_csv:
        found = find_demand_csv(args.demand_csv)
        if found is not None:
            demand_profile = load_demand_profile_csv(str(found), args.demand_city)
            print(f"Passenger loading: ON  (file={found}, "
                  f"city={args.demand_city}, crush_capacity={args.crush_capacity})")
        else:
            print("=" * 72)
            print("WARNING: demand CSV not found -> passenger loading is OFF and "
                  "every")
            print(f"         segment gets a flat {args.passengers} pax. Looked for "
                  f"'{Path(args.demand_csv).name}'")
            print("         near the script, the project root and the current dir "
                  "(also their")
            print("         data/ and data/raw/ subfolders). Drop the CSV in one of "
                  "those, or")
            print("         edit configs/model.yaml / pass --demand-csv <path>.")
            print("=" * 72)
    loading_kwargs = {"shape": args.load_shape,
                      "board_pos": args.board_pos,
                      "alight_pos": args.alight_pos,
                      "concentration": args.concentration,
                      "board_frac": args.board_frac,
                      "alight_frac": args.alight_frac,
                      "floor_frac": args.floor_frac,
                      "hour_mode": args.hour_mode}

    weather_series = None
    weather_kwargs = {"config_path": args.config}
    if args.no_weather or not weather_cfg.get("enabled", True):
        print("Weather loading: OFF -> using vehicle auxiliary load.")
    elif not args.weather_csv:
        print("Weather loading: OFF (weather_csv is empty).")
    else:
        if not _HAS_WEATHER:
            raise RuntimeError("weather_loading.py could not be imported.")
        if service_date is None:
            raise ValueError("simulation.date or --date is required for weather loading")
        weather_series = load_weather_csv(
            args.weather_csv, config_path=args.config, verbose=True
        )
        print(f"Weather loading: ON  (date={service_date}, file={args.weather_csv})")

    # Traffic signals are controlled by traffic_signals.enabled in the config,
    # with --signals / --no-signals available as one-off CLI overrides. Static
    # per stop pair, so the count cache is fetched once and reused across dates
    # and routes.
    signals_enabled = _HAS_SIGNALS and args.signals_enabled
    signals_cache_path = args.signals_csv or None
    signal_events_cache_path = args.signal_events_csv or None
    if not _HAS_SIGNALS:
        print("Traffic signals: OFF (traffic_signals.py could not be imported).")
    elif not args.signals_enabled:
        print("Traffic signals: OFF (traffic_signals.enabled=false / --no-signals).")
    elif not signals_cache_path:
        print("Traffic signals: OFF (no --signals-csv / traffic_signals_csv set).")
        signals_enabled = False
    else:
        mode = "refresh -> re-fetch" if args.refresh_signals else "cache-first"
        print(f"Traffic signals: ON  (cache={signals_cache_path}, "
              f"events={signal_events_cache_path or 'none'}, "
              f"snap={args.signals_snap_m:.0f} m, "
              f"relaxed={args.signals_relaxed_snap_m:.0f} m, "
              f"cluster={args.signals_cluster_m:.0f} m, "
              f"fallback={args.signals_fallback_per_km:.1f}/km, {mode})")

    # OSM speed caps: config-driven with --speed-caps / --no-speed-caps as
    # one-off overrides. Static per stop pair -> fetched once, cached, reused.
    motion_params = MotionParams.from_config(args.config)
    overrides = {key: value for key, value in {
        "signal_fidelity": args.signal_fidelity, "solver_mode": args.solver_mode,
        "signal_placement_index": args.placement_index,
        "signal_behaviour_index": args.behaviour_index,
    }.items() if value is not None}
    motion_params = replace(motion_params, **overrides)
    speed_caps_enabled = _HAS_SPEED_CAPS and args.speed_caps_enabled
    speed_caps_cache_path = args.speed_caps_csv or None
    if not _HAS_SPEED_CAPS:
        print("Speed caps: OFF (speed_caps.py could not be imported) -> "
              f"config default cap {motion_params.default_speed_cap_ms * 3.6:.0f} "
              "km/h on every segment.")
    elif not args.speed_caps_enabled:
        print("Speed caps: OFF (speed_caps.enabled=false / --no-speed-caps) -> "
              f"config default cap {motion_params.default_speed_cap_ms * 3.6:.0f} "
              "km/h on every segment.")
    elif not speed_caps_cache_path:
        print("Speed caps: OFF (no --speed-caps-csv / paths.speed_caps_csv set) "
              f"-> config default cap "
              f"{motion_params.default_speed_cap_ms * 3.6:.0f} km/h.")
        speed_caps_enabled = False
    else:
        mode = "refresh -> re-fetch" if args.refresh_speed_caps else "cache-first"
        print(f"Speed caps: ON  (cache={speed_caps_cache_path}, "
              f"snap={args.speed_caps_snap_m:.0f} m, "
              f"step={args.speed_caps_sample_step_m:.0f} m, "
              f"fallback={args.speed_caps_default_kmh:.0f} km/h, "
              f"clamp={args.speed_caps_max_kmh:.0f} km/h, {mode})")
    print(f"Motion: accel={motion_params.accel_ms2:.1f} m/s2, "
          f"decel={motion_params.decel_ms2:.1f} m/s2, "
          f"default cap={motion_params.default_speed_cap_ms * 3.6:.0f} km/h, "
          f"policy={motion_params.signal_time_policy}")

    tables = load_small_tables(Path(args.gtfs))
    elevation_data = srtm.get_data(local_cache_dir=str(args.srtm_cache_dir))
    vehicle_params = VehicleParams.from_config(args.config)

    runner = process_routes
    if args.compare_fidelities:
        from fidelity_comparison import process_fidelity_comparison
        runner = process_fidelity_comparison
    saved = runner(
        Path(args.gtfs), route_short_names, Path(args.output_dir), tables=tables,
        direction_id=direction_id, day=args.day, service_id=args.service_id,
        service_date=service_date,
        all_services=args.all_services, start_times=start_times,
        tolerance_s=args.tolerance_min * 60, passengers=args.passengers,
        vehicle_params=vehicle_params, elevation_data=elevation_data,
        srtm_cache_dir=args.srtm_cache_dir,
        demand_profile=demand_profile, crush_capacity=args.crush_capacity,
        loading_kwargs=loading_kwargs,
        weather_series=weather_series, weather_kwargs=weather_kwargs,
        signals_enabled=signals_enabled, signals_cache_path=signals_cache_path,
        signal_events_cache_path=signal_events_cache_path,
        signal_event_diagnostics_enabled=args.signal_event_diagnostics,
        signals_snap_m=args.signals_snap_m,
        signals_relaxed_snap_m=args.signals_relaxed_snap_m,
        signals_cluster_m=args.signals_cluster_m,
        signals_fallback_per_km=args.signals_fallback_per_km,
        signals_refresh=args.refresh_signals,
        signals_frozen=bool(get_section("traffic_signals", args.config).get("inventory_frozen")),
        motion_params=motion_params,
        speed_caps_enabled=speed_caps_enabled,
        speed_caps_cache_path=speed_caps_cache_path,
        speed_caps_snap_m=args.speed_caps_snap_m,
        speed_caps_sample_step_m=args.speed_caps_sample_step_m,
        speed_caps_min_coverage=args.speed_caps_min_coverage,
        speed_caps_default_kmh=args.speed_caps_default_kmh,
        speed_caps_max_kmh=args.speed_caps_max_kmh,
        speed_caps_refresh=args.refresh_speed_caps,
        simulation_level=args.simulation_level,
        osm_snapshot=args.osm_snapshot,
        shape_repairs=gtfs_cfg.get("shape_repairs"),
        placement_registry_path=args.signal_placement_registry or None,
        input_source_paths=[args.config or PROJECT_ROOT / "configs/model.yaml", args.gtfs,
            args.weather_csv if weather_series is not None else None,
            found if demand_profile is not None else None],
    )
    print(f"\nProcessed {len(route_short_names)} route(s) -> {len(saved)} file(s).")
