"""Versioned C06 input adapters. Historical acquisition and lookup paths are untouched."""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/publication_c06_v1.json"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["demand_profile_enabled"] or config["weather_loading_enabled"]:
        raise ValueError("Neutral primary configuration requires fixed load and auxiliaries")
    for key in ("passengers", "in_service_aux_power_kW", "served_dwell_s", "dt_s"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid {key}")
    if config["dt_s"] == 0 or not 0 <= config["initial_soc_fraction"] <= 1:
        raise ValueError("Invalid numerical or initial battery setting")
    expected_mass = (config["vehicle"]["curb_mass_kg"]
                     + config["passengers"] * config["vehicle"]["passenger_mass_kg"])
    if not math.isclose(config["total_vehicle_mass_kg"], expected_mass):
        raise ValueError("Declared mass disagrees with nominal load")
    if config["vehicle"]["aux_power_kW"] != config["in_service_aux_power_kW"]:
        raise ValueError("Vehicle and controlled auxiliary power disagree")
    return config


def neutral_segments(segments, config):
    """Copy physical inputs; never invoke demand, weather, region or B1 lookups."""
    result = []
    for original in segments:
        seg = copy.deepcopy(original)
        seg.passengers = config["passengers"]
        seg.aux_power_kW = config["in_service_aux_power_kW"]
        seg.controlled_temperature_c = config["temperature_c"]
        result.append(seg)
    return result


def service_seconds(value):
    h, m, s = (int(x) for x in value.split(":"))
    if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError("Invalid extended GTFS time")
    return h * 3600 + m * 60 + s


def service_time_utc(service_date, seconds, agency_timezone="Europe/London"):
    """GTFS service origin is local noon minus 12 hours, including DST days.

    Add extended service seconds on the UTC timeline, avoiding ambiguous local
    01:xx wall times. This adapter is for the new weather sensitivity path.
    """
    day = date.fromisoformat(service_date) if isinstance(service_date, str) else service_date
    noon = datetime.combine(day, time(12), ZoneInfo(agency_timezone))
    return noon.astimezone(timezone.utc) - timedelta(hours=12) + timedelta(seconds=seconds)


def weather_for_trip_utc(weather, service_date, trip_rows, mode="midpoint",
                         agency_timezone="Europe/London"):
    """UTC-aware lookup retaining the selected row, offset and extended day.

    The supplied retained weather series must represent UTC; a naive index is
    explicitly interpreted as the archived UTC index, never as agency local.
    """
    import pandas as pd
    first = service_seconds(trip_rows[0]["departure_time"])
    last = service_seconds(trip_rows[-1]["arrival_time"])
    if mode not in ("start", "midpoint"):
        raise ValueError("Unknown weather reference mode")
    query = pd.Timestamp(service_time_utc(
        service_date, first if mode == "start" else (first + last) / 2, agency_timezone))
    frame = weather.df.copy()
    frame.index = (frame.index.tz_localize("UTC") if frame.index.tz is None
                   else frame.index.tz_convert("UTC"))
    frame = frame.sort_index()
    if frame.empty or frame.index.has_duplicates:
        raise ValueError("Weather rows must be nonempty and uniquely timestamped")
    index = frame.index.get_indexer([query], method="nearest")[0]
    selected = frame.index[index]
    return {"query_utc": query.isoformat(), "selected_row_utc": selected.isoformat(),
            "offset_s": (selected - query).total_seconds(),
            "conditions": frame.iloc[index].to_dict()}


def endpoint_grade(elevation_a_m, elevation_b_m, distance_m):
    """Frozen rounded-HGT endpoint convention; no missing-to-flat fallback."""
    if (distance_m <= 0 or not all(math.isfinite(v) for v in
                                  (elevation_a_m, elevation_b_m, distance_m))):
        raise ValueError("Missing/invalid elevation or distance blocks grade input")
    raw = (elevation_b_m - elevation_a_m) / distance_m
    return {"grade": max(-0.15, min(0.15, raw)), "raw_grade": raw,
            "clipped": abs(raw) > 0.15, "smoothing": "none",
            "source_status": "endpoint approximation; DEM edition/datum unverified"}


def read_hgt_rounded(path, latitude, longitude):
    """Read a local tile only, matching C03/C05's nearest (round) indexing."""
    path = Path(path)
    match = re.fullmatch(r"([NS])(\d+)([EW])(\d+)\.hgt", path.name, re.I)
    if not match:
        raise ValueError("HGT filename must declare its tile origin")
    ns, lat, ew, lon = match.groups()
    south = int(lat) * (1 if ns.upper() == "N" else -1)
    west = int(lon) * (1 if ew.upper() == "E" else -1)
    size = math.isqrt(path.stat().st_size // 2)
    if size < 2 or size * size * 2 != path.stat().st_size:
        raise ValueError("Invalid HGT dimensions")
    if not (south <= latitude <= south + 1 and west <= longitude <= west + 1):
        raise ValueError("Coordinate outside declared HGT tile")
    row = round((south + 1 - latitude) * (size - 1))
    col = round((longitude - west) * (size - 1))
    with path.open("rb") as stream:
        stream.seek(2 * (row * size + col))
        value = int.from_bytes(stream.read(2), "big", signed=True)
    if value == -32768:
        raise ValueError("DEM void blocks grade input")
    return float(value)


def conservative_cap(tags, direction, registration_accepted=False):
    """Explicit minimum-of-possible-numeric-limits assumption, not legal proof.

    Conditional syntax is deliberately bounded: numeric km/h or mph clauses.
    Unsupported symbolic conditions/limits and unresolved road matches block.
    Taking every possible conditional numeric minimum avoids weather/time
    switching between paired cases; no fallback cap is invented.
    """
    if not registration_accepted or direction not in ("forward", "backward"):
        raise ValueError("Directed road registration is unresolved")

    def numeric(value):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh)?\s*", str(value))
        if not match or float(match[1]) <= 0:
            raise ValueError("Unsupported/missing numeric speed limit")
        return float(match[1]) * (0.44704 if match[2] == "mph" else 1 / 3.6)

    base_key = f"maxspeed:{direction}" if f"maxspeed:{direction}" in tags else "maxspeed"
    caps = [numeric(tags.get(base_key, ""))]
    for key in ("maxspeed:conditional", f"maxspeed:{direction}:conditional"):
        if key in tags:
            for clause in tags[key].split(";"):
                if "@" not in clause or not clause.split("@", 1)[1].strip():
                    raise ValueError("Unsupported conditional limit")
                caps.append(numeric(clause.split("@", 1)[0]))
    return {"cap_ms": min(caps), "status": "assumed_conservative_numeric_minimum",
            "tags": dict(tags), "direction": direction, "legal_bound_verified": False}
