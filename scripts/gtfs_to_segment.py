"""
gtfs_to_segments.py
================================================================================
Turn a static GTFS feed into the stop-to-stop `Segment` list consumed by
beb_soc_model.py.

It does the ROUTE -> MOTION inputs job from real data:
  * segment length   <- distance between consecutive stops (stops.txt)
  * dwell time        <- departure - arrival at each stop (stop_times.txt)
  * cruise speed      <- scheduled run time between stops -> average speed
  * grade             <- sampled from SRTM elevations (see add_grades_from_dem).
  * passengers        <- NOT in GTFS. Set by assumption; replace with APC later.

Designed for a LARGE national feed (the Irish/NTA feed has a 372 MB
stop_times.txt and 299 MB shapes.txt), so it never loads those whole files:
small tables are read normally; stop_times is *streamed* from the zip and
filtered to a single trip.

USAGE
-----
    python3 scripts/gtfs_to_segment.py 102 41 15
    python3 scripts/gtfs_to_segment.py --routes 102,41,15
or, from your own code:
    from gtfs_to_segment import segments_for_route
    segs = segments_for_route("path/to/GTFS.zip", route_short_name="102")
================================================================================
"""

import argparse
import csv
import io
import re
import zipfile
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

import pandas as pd
import srtm  # SRTM elevation tiles, same source RouteZero uses for grades

# Reuse the vehicle model and Segment dataclass from the sibling file.
from beb_soc_model import Segment, VehicleParams, simulate_route


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GTFS = PROJECT_ROOT / "data" / "raw" / "GTFS_Realtime.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Where SRTM .hgt tiles are cached on disk so they download only once.
DEFAULT_SRTM_CACHE = PROJECT_ROOT / "data" / "srtm_cache"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def gtfs_time_to_seconds(t: str) -> int:
    """GTFS times can exceed 24:00:00 (after-midnight trips); handle h>=24."""
    h, m, s = (int(x) for x in t.strip().split(":"))
    return h * 3600 + m * 60 + s


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    R = 6_371_000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * R * asin(sqrt(a))


# -----------------------------------------------------------------------------
# Read small tables
# -----------------------------------------------------------------------------
def load_small_tables(gtfs_zip: str):
    """routes, trips, stops are small enough to read fully."""
    with zipfile.ZipFile(gtfs_zip) as z:
        routes = pd.read_csv(z.open("routes.txt"))
        trips = pd.read_csv(z.open("trips.txt"),
                            dtype={"trip_id": str, "route_id": str, "shape_id": str})
        stops = pd.read_csv(z.open("stops.txt"),
                            dtype={"stop_id": str})
    return routes, trips, stops


def choose_trip(routes, trips, route_short_name, direction_id=0):
    """Resolve a human route name (e.g. '102') to one representative trip_id."""
    match = routes[routes["route_short_name"].astype(str) == str(route_short_name)]
    if match.empty:
        raise ValueError(f"No route with short name {route_short_name!r}")
    route_id = match.iloc[0]["route_id"]
    rtrips = trips[trips["route_id"] == route_id]
    if direction_id is not None and "direction_id" in rtrips.columns:
        d = rtrips[rtrips["direction_id"] == direction_id]
        if not d.empty:
            rtrips = d
    trip_id = rtrips.iloc[0]["trip_id"]   # first trip in this direction
    return route_id, trip_id, match.iloc[0]["route_long_name"]


def safe_route_name(route_short_name) -> str:
    """Return a filesystem-safe route label for output filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(route_short_name).strip())
    return cleaned.strip("._") or "route"


# -----------------------------------------------------------------------------
# Stream the big stop_times.txt, keeping only one trip
# -----------------------------------------------------------------------------
def stop_times_for_trip(gtfs_zip: str, trip_id: str, assume_grouped=True):
    """
    Stream stop_times.txt from the zip and return the rows for one trip, sorted
    by stop_sequence. Avoids loading the whole 372 MB file into memory.
    `assume_grouped` lets us stop early once the trip's contiguous block ends.
    """
    rows = []
    started = False
    with zipfile.ZipFile(gtfs_zip) as z:
        with z.open("stop_times.txt") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            for r in reader:
                if r["trip_id"] == trip_id:
                    rows.append(r)
                    started = True
                elif started and assume_grouped:
                    break  # trip's block finished; stop scanning
    rows.sort(key=lambda r: int(r["stop_sequence"]))
    return rows


# -----------------------------------------------------------------------------
# Build the Segment list
# -----------------------------------------------------------------------------
def iter_valid_stop_pairs(trip_rows, stops):
    """
    Yield (row_a, row_b, length_m) for every consecutive stop pair that survives
    the co-located-stop filter (length >= 1 m).

    build_segments AND add_grades_from_dem both consume this, so the i-th
    segment always corresponds to the i-th yielded pair. Without this shared
    source of truth the two passes drift apart whenever a duplicate stop is
    dropped, and grades get attached to the wrong segments.
    """
    coord = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
    for a, b in zip(trip_rows[:-1], trip_rows[1:]):
        sa, sb = coord[a["stop_id"]], coord[b["stop_id"]]
        length_m = haversine_m(sa["stop_lat"], sa["stop_lon"],
                               sb["stop_lat"], sb["stop_lon"])
        if length_m < 1:
            continue  # duplicate / co-located stop
        yield a, b, length_m


def build_segments(trip_rows, stops, passengers=20, cruise_factor=1.15,
                   v_min=3.0, v_max=22.0):
    """
    Convert ordered stop_times rows into a list of Segment objects.
    cruise_factor scales the schedule-implied *average* speed up toward a
    free-flow *cruise* speed (the trapezoidal profile spends time accelerating).
    Grade is left at 0.0 here and filled later by add_grades_from_dem().
    """
    segments = []
    for a, b, length_m in iter_valid_stop_pairs(trip_rows, stops):
        run_time = gtfs_time_to_seconds(b["arrival_time"]) - \
                   gtfs_time_to_seconds(a["departure_time"])
        if run_time > 0:
            v_cruise = min(max((length_m / run_time) * cruise_factor, v_min), v_max)
        else:
            v_cruise = 11.0  # fallback if schedule has no usable time here

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
    Fill each Segment.grade from SRTM elevations, the same data source RouteZero
    uses (`srtm.get_data().get_elevation(lat, lon)`).

        grade = (elev_end_stop - elev_start_stop) / segment_length   [rise/run]

    UNITS: grade is a dimensionless fraction (rise/run), matching this file's
    original docstring. RouteZero reports *percent* (this value * 100). If
    beb_soc_model expects percent or degrees instead, convert here -- getting
    the unit wrong rescales the gravitational term by 100x.

    Elevation is sampled once per unique stop (cached). Segments are aligned to
    the SAME pairs build_segments used, via iter_valid_stop_pairs, so a dropped
    co-located stop never shifts grades onto the wrong segment.

    SRTM gaps: get_elevation() returns None over water or where a tile is
    missing/unreachable. Such segments are left flat (grade 0.0) and counted, so
    a network/coverage problem is visible instead of silently corrupting energy.

    Note vs RouteZero: it computes grade over a whole TRIP (long baseline, so
    90 m SRTM noise averages out). Here grade is per stop-to-stop SEGMENT, which
    can be a couple hundred metres, so SRTM noise is proportionally larger --
    hence the max_abs_grade clamp. Pass max_abs_grade=None to disable it.
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

    filled = 0
    missing = 0
    grades = []
    for seg, (a, b, length_m) in zip(segments, pairs):
        ea = elevation_of(a["stop_id"])
        eb = elevation_of(b["stop_id"])
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
            print(f"  grades: filled {filled}/{len(segments)} segments from SRTM "
                  f"(range {min(grades) * 100:+.1f}%..{max(grades) * 100:+.1f}%, "
                  f"mean {sum(grades) / len(grades) * 100:+.2f}%)"
                  + (f"; {missing} left flat (no SRTM value)" if missing else ""))
        else:
            print("  grades: SRTM returned no usable elevations -> all flat. "
                  "Check network access / tile coverage for this area.")
    return segments


# -----------------------------------------------------------------------------
# Convenience: name -> segments in one call
# -----------------------------------------------------------------------------
def segments_for_route(gtfs_zip, route_short_name, direction_id=0,
                       passengers=20, tables=None, elevation_data=None,
                       srtm_cache_dir=None):
    if tables is None:
        routes, trips, stops = load_small_tables(gtfs_zip)
    else:
        routes, trips, stops = tables
    route_id, trip_id, long_name = choose_trip(routes, trips,
                                               route_short_name, direction_id)
    trip_rows = stop_times_for_trip(gtfs_zip, trip_id)
    segs = build_segments(trip_rows, stops, passengers=passengers)
    segs = add_grades_from_dem(segs, trip_rows, stops,
                               elevation_data=elevation_data,
                               srtm_cache_dir=srtm_cache_dir)
    print(f"Route {route_short_name}  ({long_name})")
    print(f"  trip_id           : {trip_id}")
    print(f"  stops in trip     : {len(trip_rows)}")
    print(f"  usable segments   : {len(segs)}")
    return segs, {
        "route_short_name": route_short_name,
        "route_long_name": long_name,
        "route_id": route_id,
        "trip_id": trip_id,
        "stops_in_trip": len(trip_rows),
    }


def route_results_dataframe(segs, route_info, vehicle_params=None, soc0_pct=100.0):
    """Simulate one route and attach route metadata to every output row."""
    if vehicle_params is None:
        vehicle_params = VehicleParams()
    df = simulate_route(segs, vehicle_params, soc0_pct=soc0_pct)
    for col, value in reversed(route_info.items()):
        df.insert(0, col, value)
    return df


def process_route(gtfs_zip, route_short_name, output_dir, direction_id=0,
                  passengers=20, tables=None, vehicle_params=None,
                  elevation_data=None, srtm_cache_dir=None):
    """Build, simulate, and save one route to its own CSV file."""
    segs, route_info = segments_for_route(
        gtfs_zip,
        route_short_name,
        direction_id=direction_id,
        passengers=passengers,
        tables=tables,
        elevation_data=elevation_data,
        srtm_cache_dir=srtm_cache_dir,
    )
    df = route_results_dataframe(segs, route_info, vehicle_params=vehicle_params)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"route_{safe_route_name(route_short_name)}_segments.csv"
    df.to_csv(output_path, index=False)

    total_E = df["energy_kWh"].sum()
    total_km = df["cum_dist_km"].iloc[-1]
    print("\n--- Route summary (grades from SRTM) ---")
    print(f"Total distance      : {total_km:.2f} km")
    print(f"Total energy        : {total_E:.2f} kWh")
    print(f"Average consumption : {total_E / total_km:.3f} kWh/km")
    print(f"SoC at end          : {df['SoC_end_%'].iloc[-1]:.1f} %")
    print(f"Saved: {output_path}")
    return output_path


def parse_route_list(route_args, routes_csv):
    """Combine positional and comma-separated route arguments."""
    routes = []
    for value in route_args:
        routes.extend(part.strip() for part in value.split(","))
    if routes_csv:
        routes.extend(part.strip() for part in routes_csv.split(","))
    return [route for route in routes if route]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert one or more GTFS route short names to BEB segment CSVs."
    )
    parser.add_argument(
        "route_short_names",
        nargs="*",
        help="Route short names to process, e.g. 102 41 15. Commas are also allowed.",
    )
    parser.add_argument(
        "--routes",
        help="Comma-separated route short names, e.g. 102,41,15.",
    )
    parser.add_argument(
        "--gtfs",
        default=str(DEFAULT_GTFS),
        help=f"Path to the static GTFS zip. Default: {DEFAULT_GTFS}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for per-route CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--direction-id",
        type=int,
        default=0,
        help="GTFS direction_id to prefer. Default: 0.",
    )
    parser.add_argument(
        "--any-direction",
        action="store_true",
        help="Use the first available trip regardless of direction_id.",
    )
    parser.add_argument(
        "--passengers",
        type=int,
        default=20,
        help="Assumed passengers per segment. Default: 20.",
    )
    parser.add_argument(
        "--srtm-cache-dir",
        default=str(DEFAULT_SRTM_CACHE),
        help=f"Directory to cache SRTM tiles. Default: {DEFAULT_SRTM_CACHE}",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    route_short_names = parse_route_list(args.route_short_names, args.routes)
    if not route_short_names:
        route_short_names = ["208"]  # Keep the old demo behavior.

    direction_id = None if args.any_direction else args.direction_id
    gtfs_zip = Path(args.gtfs)
    output_dir = Path(args.output_dir)
    tables = load_small_tables(gtfs_zip)
    vehicle_params = VehicleParams()

    # Build the SRTM reader once and reuse it for every route (tiles are cached
    # on disk in srtm_cache_dir, so repeated runs don't re-download).
    elevation_data = srtm.get_data(local_cache_dir=str(args.srtm_cache_dir))

    saved_paths = []
    for route_short_name in route_short_names:
        saved_paths.append(
            process_route(
                gtfs_zip,
                route_short_name,
                output_dir,
                direction_id=direction_id,
                passengers=args.passengers,
                tables=tables,
                vehicle_params=vehicle_params,
                elevation_data=elevation_data,
                srtm_cache_dir=args.srtm_cache_dir,
            )
        )

    print(f"\nProcessed {len(saved_paths)} route(s).")