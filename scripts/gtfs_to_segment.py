"""
gtfs_to_segments.py
================================================================================
Turn a static GTFS feed into the stop-to-stop `Segment` list consumed by
beb_soc_model.py.

It does the ROUTE -> MOTION inputs job from real data:
  * segment length   <- distance between consecutive stops (stops.txt)
  * dwell time        <- departure - arrival at each stop (stop_times.txt)
  * cruise speed      <- scheduled run time between stops -> average speed
  * grade             <- NOT in GTFS. Left at 0.0; see add_grades_from_dem().
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

# Reuse the vehicle model and Segment dataclass from the sibling file.
from beb_soc_model import Segment, VehicleParams, simulate_route


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GTFS = PROJECT_ROOT / "data" / "raw" / "GTFS_Realtime.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


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
def build_segments(trip_rows, stops, passengers=20, cruise_factor=1.15,
                   v_min=3.0, v_max=22.0):
    """
    Convert ordered stop_times rows into a list of Segment objects.
    cruise_factor scales the schedule-implied *average* speed up toward a
    free-flow *cruise* speed (the trapezoidal profile spends time accelerating).
    """
    coord = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
    segments = []
    for a, b in zip(trip_rows[:-1], trip_rows[1:]):
        sa, sb = coord[a["stop_id"]], coord[b["stop_id"]]
        length_m = haversine_m(sa["stop_lat"], sa["stop_lon"],
                               sb["stop_lat"], sb["stop_lon"])
        if length_m < 1:
            continue  # duplicate / co-located stop

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


def add_grades_from_dem(segments, trip_rows, stops, dem_path=None):
    """
    STUB / hook. GTFS has no elevation, so grade defaults to 0.0 above.
    To add real grade you need a Digital Elevation Model (e.g. Copernicus
    GLO-30 or SRTM). The logic would be:
        1. for each stop, sample DEM elevation at (lat, lon)  [rasterio]
        2. grade = (elev_next - elev_this) / segment_length
        3. write seg.grade = grade
    Left unimplemented because no DEM was provided. Flat terrain (grade=0)
    until you wire this up.
    """
    return segments


# ----------------------------------------------------------------------------- 
# Convenience: name -> segments in one call
# -----------------------------------------------------------------------------
def segments_for_route(gtfs_zip, route_short_name, direction_id=0,
                       passengers=20, tables=None):
    if tables is None:
        routes, trips, stops = load_small_tables(gtfs_zip)
    else:
        routes, trips, stops = tables
    route_id, trip_id, long_name = choose_trip(routes, trips,
                                               route_short_name, direction_id)
    trip_rows = stop_times_for_trip(gtfs_zip, trip_id)
    segs = build_segments(trip_rows, stops, passengers=passengers)
    segs = add_grades_from_dem(segs, trip_rows, stops)  # no-op until DEM added
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
                  passengers=20, tables=None, vehicle_params=None):
    """Build, simulate, and save one route to its own CSV file."""
    segs, route_info = segments_for_route(
        gtfs_zip,
        route_short_name,
        direction_id=direction_id,
        passengers=passengers,
        tables=tables,
    )
    df = route_results_dataframe(segs, route_info, vehicle_params=vehicle_params)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"route_{safe_route_name(route_short_name)}_segments.csv"
    df.to_csv(output_path, index=False)

    total_E = df["energy_kWh"].sum()
    total_km = df["cum_dist_km"].iloc[-1]
    print("\n--- Route summary (flat grade until DEM added) ---")
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
            )
        )

    print(f"\nProcessed {len(saved_paths)} route(s).")
