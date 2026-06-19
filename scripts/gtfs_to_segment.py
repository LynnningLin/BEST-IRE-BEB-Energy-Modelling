"""
gtfs_to_segments.py
================================================================================
Turn a static GTFS feed into the stop-to-stop `Segment` list consumed by
beb_soc_model.py, for ONE OR MANY trips per route across the day.

ROUTE -> MOTION inputs, from real data:
  * segment length   <- distance between consecutive stops (stops.txt)
  * dwell time        <- departure - arrival at each stop (stop_times.txt)
  * cruise speed      <- scheduled run time between stops -> average speed
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
import re
import sys
import zipfile
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
from beb_soc_model import Segment, VehicleParams, simulate_route

# Passenger demand model. Soft import: if the module is absent the pipeline
# still runs, it just leaves the flat `passengers` assumption in place.
try:
    from passenger_loading import HourlyDemandProfile, apply_passenger_loading
    _HAS_LOADING = True
except ImportError:
    HourlyDemandProfile = None
    apply_passenger_loading = None
    _HAS_LOADING = False

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday"]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def gtfs_time_to_seconds(t: str) -> int:
    """GTFS times can exceed 24:00:00 (after-midnight trips); handle h>=24."""
    h, m, s = (int(x) for x in str(t).strip().split(":"))
    return h * 3600 + m * 60 + s


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


# -----------------------------------------------------------------------------
# Read small tables (routes, trips, stops, calendar)
# -----------------------------------------------------------------------------
def load_small_tables(gtfs_zip: str):
    """routes, trips, stops, calendar are small enough to read fully."""
    with zipfile.ZipFile(gtfs_zip) as z:
        names = set(z.namelist())
        routes = pd.read_csv(z.open("routes.txt"), dtype={"route_id": str})
        trips = pd.read_csv(
            z.open("trips.txt"),
            dtype={"trip_id": str, "route_id": str, "shape_id": str,
                   "service_id": str},
        )
        stops = pd.read_csv(z.open("stops.txt"), dtype={"stop_id": str})
        calendar = (pd.read_csv(z.open("calendar.txt"), dtype={"service_id": str})
                    if "calendar.txt" in names else None)
    return routes, trips, stops, calendar


# -----------------------------------------------------------------------------
# Route / service / trip selection
# -----------------------------------------------------------------------------
def resolve_route_id(routes, route_short_name):
    """Resolve a human route name (e.g. '102') to (route_id, route_long_name)."""
    match = routes[routes["route_short_name"].astype(str) == str(route_short_name)]
    if match.empty:
        raise ValueError(f"No route with short name {route_short_name!r}")
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
# Build the Segment list (unchanged, validated)
# -----------------------------------------------------------------------------
def iter_valid_stop_pairs(trip_rows, stops):
    """
    Yield (row_a, row_b, length_m) for every consecutive stop pair that survives
    the co-located-stop filter (length >= 1 m). build_segments AND
    add_grades_from_dem both consume this, so the i-th segment always matches the
    i-th yielded pair (dropping a duplicate stop never shifts grades/loads).
    """
    coord = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
    for a, b in zip(trip_rows[:-1], trip_rows[1:]):
        sa, sb = coord[a["stop_id"]], coord[b["stop_id"]]
        length_m = haversine_m(sa["stop_lat"], sa["stop_lon"],
                               sb["stop_lat"], sb["stop_lon"])
        if length_m < 1:
            continue
        yield a, b, length_m


def build_segments(trip_rows, stops, passengers=20, cruise_factor=1.15,
                   v_min=3.0, v_max=22.0):
    """Convert ordered stop_times rows into Segment objects (grade=0 for now)."""
    segments = []
    for a, b, length_m in iter_valid_stop_pairs(trip_rows, stops):
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
                                 passengers=passengers))
    return segments


def add_grades_from_dem(segments, trip_rows, stops, dem_path=None,
                        elevation_data=None, srtm_cache_dir=None,
                        max_abs_grade=0.15, verbose=True):
    """
    Fill Segment.grade from SRTM elevations (RouteZero's source):
        grade = (elev_end - elev_start) / segment_length   [rise/run fraction]
    Units, alignment, void and SRTM-noise notes as documented previously.
    """
    pairs = list(iter_valid_stop_pairs(trip_rows, stops))
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
                      crush_capacity=70, loading_kwargs=None, verbose=False):
    """
    Full per-trip build. If `demand_profile` is given (a HourlyDemandProfile),
    passenger_loading.apply_passenger_loading() overwrites Segment.passengers
    using THIS trip's real start time -> hour -> demand factor. Otherwise the
    flat `passengers` assumption stands.
    """
    segs = build_segments(trip_rows, stops, passengers=passengers)
    segs = add_grades_from_dem(segs, trip_rows, stops,
                               elevation_data=elevation_data,
                               srtm_cache_dir=srtm_cache_dir, verbose=verbose)
    if demand_profile is not None:
        if not _HAS_LOADING:
            raise RuntimeError("demand_profile given but passenger_loading.py "
                               "could not be imported.")
        segs = apply_passenger_loading(segs, trip_rows, demand_profile,
                                       crush_capacity=crush_capacity,
                                       verbose=verbose, **(loading_kwargs or {}))
    return segs


# -----------------------------------------------------------------------------
# Results dataframes
# -----------------------------------------------------------------------------
def trip_results_dataframe(segs, trip_info, vehicle_params=None, soc0_pct=100.0):
    """Simulate one trip and prepend its route/trip metadata to every row."""
    if vehicle_params is None:
        vehicle_params = VehicleParams()
    df = simulate_route(segs, vehicle_params, soc0_pct=soc0_pct)
    for col, value in reversed(list(trip_info.items())):
        df.insert(0, col, value)
    return df


def safe_route_name(route_short_name) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(route_short_name).strip())
    return cleaned.strip("._") or "route"


def _print_route_summary(rsn, df, n_trips):
    g = (df.groupby(["trip_id", "trip_start_time"])["energy_kWh"].sum()
         .reset_index().sort_values("trip_start_time"))
    print(f"\nRoute {rsn}: {n_trips} trip(s) processed")
    if not g.empty:
        print(f"  energy/trip  min {g.energy_kWh.min():.1f} | "
              f"mean {g.energy_kWh.mean():.1f} | max {g.energy_kWh.max():.1f} kWh")
        busiest = g.loc[g.energy_kWh.idxmax()]
        print(f"  busiest trip {busiest.trip_start_time} -> "
              f"{busiest.energy_kWh:.1f} kWh")


# -----------------------------------------------------------------------------
# Top-level: process MANY trips across MANY routes (one streaming pass)
# -----------------------------------------------------------------------------
def process_routes(gtfs_zip, route_short_names, output_dir, tables=None,
                   direction_id=0, day="monday", service_id=None,
                   all_services=False, start_times=None, tolerance_s=900,
                   passengers=20, vehicle_params=None, elevation_data=None,
                   srtm_cache_dir=None, demand_profile=None, crush_capacity=70,
                   loading_kwargs=None):
    """
    For each route: pick trips (by service day, then by start time or ALL),
    build/simulate each, and save one CSV per route with every trip stacked
    (columns include trip_id + trip_start_time). Returns the saved paths.
    """
    if tables is None:
        tables = load_small_tables(gtfs_zip)
    routes, trips, stops, calendar = tables
    if vehicle_params is None:
        vehicle_params = VehicleParams()

    # 1) which service_ids count as "the day"
    if all_services or calendar is None:
        service_ids = None
        if calendar is None and not all_services:
            print("calendar.txt not found -> processing ALL services (may mix "
                  "weekday/weekend trips).")
    elif service_id is not None:
        service_ids = {str(service_id)}
    else:
        service_ids = service_ids_for_day(calendar, day)

    # 2) candidate trips per route, and the union to scan for
    dir_by_trip = (trips.set_index("trip_id")["direction_id"].to_dict()
                   if "direction_id" in trips.columns else {})
    plan, union = {}, set()
    for rsn in route_short_names:
        route_id, long_name = resolve_route_id(routes, rsn)
        tids = trips_for_route(trips, route_id, direction_id, service_ids)
        plan[rsn] = (route_id, long_name, tids)
        union.update(tids)

    # 3) ONE streaming pass over stop_times.txt for every wanted trip
    print(f"Scanning stop_times.txt once for {len(union)} candidate trip(s)...")
    by_trip_all = stop_times_for_trips(gtfs_zip, union)

    # 4) per route: select + build + simulate + save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for rsn, (route_id, long_name, tids) in plan.items():
        route_trips = {t: by_trip_all.get(t, []) for t in tids}
        route_trips = {t: r for t, r in route_trips.items() if r}
        chosen = select_trips_by_start(route_trips, start_times, tolerance_s)
        frames = []
        for tid in chosen:
            rows = route_trips[tid]
            segs = segments_for_trip(
                rows, stops, passengers=passengers,
                elevation_data=elevation_data, srtm_cache_dir=srtm_cache_dir,
                demand_profile=demand_profile, crush_capacity=crush_capacity,
                loading_kwargs=loading_kwargs,
            )
            info = {
                "route_short_name": rsn,
                "route_long_name": long_name,
                "route_id": route_id,
                "direction_id": dir_by_trip.get(tid, ""),
                "trip_id": tid,
                "trip_start_time": seconds_to_hhmmss(trip_start_seconds(rows)),
            }
            frames.append(trip_results_dataframe(segs, info, vehicle_params))

        if not frames:
            print(f"Route {rsn}: no trips selected.")
            continue
        df = pd.concat(frames, ignore_index=True)
        out_path = output_dir / f"route_{safe_route_name(rsn)}_trips.csv"
        df.to_csv(out_path, index=False)
        _print_route_summary(rsn, df, len(chosen))
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
    gtfs_cfg = get_section("gtfs", config_path)
    loading_cfg = get_section("passenger_loading", config_path)

    default_gtfs = get_path("gtfs_zip", config_path)
    default_output_dir = get_path("gtfs_output_dir", config_path)
    default_srtm_cache = get_path("srtm_cache_dir", config_path)
    default_demand_csv = get_path("passenger_loading_csv", config_path)

    p = argparse.ArgumentParser(
        description="GTFS route short names -> per-trip BEB segment CSVs.",
        parents=[base])
    p.add_argument("route_short_names", nargs="*",
                   help="e.g. 102 41 15 (commas allowed).")
    p.add_argument("--routes", help="Comma-separated route short names.")
    p.add_argument("--gtfs", default=str(default_gtfs))
    p.add_argument("--output-dir", default=str(default_output_dir))
    p.add_argument("--direction-id", type=int,
                   default=int(gtfs_cfg.get("direction_id", 0)))
    p.add_argument("--any-direction", action="store_true",
                   help="Use trips from all directions.")
    # which day / service
    p.add_argument("--day", default=gtfs_cfg.get("day", "monday"), choices=WEEKDAYS,
                   help="Weekday whose services to use (via calendar.txt).")
    p.add_argument("--service-id", help="Force a specific service_id.")
    p.add_argument("--all-services", action="store_true",
                   help="Ignore the calendar filter (process every service).")
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
    route_short_names = parse_route_list(args.route_short_names, args.routes)
    if not route_short_names:
        route_short_names = [str(r) for r in gtfs_cfg.get("default_routes", ["208"])]

    start_times = ([s.strip() for s in args.start_times.split(",") if s.strip()]
                   if args.start_times else None)
    direction_id = None if args.any_direction else args.direction_id

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

    tables = load_small_tables(Path(args.gtfs))
    elevation_data = srtm.get_data(local_cache_dir=str(args.srtm_cache_dir))
    vehicle_params = VehicleParams.from_config(args.config)

    saved = process_routes(
        Path(args.gtfs), route_short_names, Path(args.output_dir), tables=tables,
        direction_id=direction_id, day=args.day, service_id=args.service_id,
        all_services=args.all_services, start_times=start_times,
        tolerance_s=args.tolerance_min * 60, passengers=args.passengers,
        vehicle_params=vehicle_params, elevation_data=elevation_data,
        srtm_cache_dir=args.srtm_cache_dir,
        demand_profile=demand_profile, crush_capacity=args.crush_capacity,
        loading_kwargs=loading_kwargs,
    )
    print(f"\nProcessed {len(route_short_names)} route(s) -> {len(saved)} file(s).")
