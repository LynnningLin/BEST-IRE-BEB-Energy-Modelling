import csv
import io
import zipfile
from datetime import date
from math import degrees

import pandas as pd
import pytest

from beb_soc_model import MotionParams, VehicleParams
from gtfs_to_segment import load_small_tables
from passenger_loading import HourlyDemandProfile
from weather_loading import WeatherSeries


EARTH_R = 6_371_000.0
STEP_DEG = degrees(500.0 / EARTH_R)


def _csv_text(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


@pytest.fixture
def integration_gtfs_zip(tmp_path):
    stops = [
        {"stop_id": "s1", "stop_name": "A", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "sdup", "stop_name": "A duplicate", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s2", "stop_name": "B", "stop_lat": STEP_DEG, "stop_lon": 0.0},
        {"stop_id": "s3", "stop_name": "C", "stop_lat": 2 * STEP_DEG, "stop_lon": 0.0},
    ]
    tables = {
        "routes.txt": [
            {"route_id": "r1", "route_short_name": "208", "route_long_name": "Synthetic 208"}
        ],
        "trips.txt": [
            {
                "route_id": "r1",
                "service_id": "svc",
                "trip_id": "t1",
                "trip_headsign": "Outbound",
                "direction_id": "0",
                "block_id": "block_a",
                "shape_id": "shape_fwd",
            },
            {
                "route_id": "r1",
                "service_id": "svc",
                "trip_id": "t2",
                "trip_headsign": "Inbound",
                "direction_id": "1",
                "block_id": "block_a",
                "shape_id": "shape_rev",
            },
        ],
        "stops.txt": stops,
        "stop_times.txt": [
            {
                "trip_id": "t1",
                "arrival_time": "08:00:00",
                "departure_time": "08:00:00",
                "stop_id": "s1",
                "stop_sequence": "1",
            },
            {
                "trip_id": "t1",
                "arrival_time": "08:00:10",
                "departure_time": "08:00:10",
                "stop_id": "sdup",
                "stop_sequence": "2",
            },
            {
                "trip_id": "t1",
                "arrival_time": "08:02:00",
                "departure_time": "08:02:20",
                "stop_id": "s2",
                "stop_sequence": "3",
            },
            {
                "trip_id": "t1",
                "arrival_time": "08:04:00",
                "departure_time": "08:04:15",
                "stop_id": "s3",
                "stop_sequence": "4",
            },
            {
                "trip_id": "t2",
                "arrival_time": "09:00:00",
                "departure_time": "09:00:00",
                "stop_id": "s3",
                "stop_sequence": "1",
            },
            {
                "trip_id": "t2",
                "arrival_time": "09:02:00",
                "departure_time": "09:02:20",
                "stop_id": "s2",
                "stop_sequence": "2",
            },
            {
                "trip_id": "t2",
                "arrival_time": "09:04:00",
                "departure_time": "09:04:15",
                "stop_id": "s1",
                "stop_sequence": "3",
            },
        ],
        "calendar.txt": [
            {
                "service_id": "svc",
                "monday": "1",
                "tuesday": "1",
                "wednesday": "1",
                "thursday": "1",
                "friday": "1",
                "saturday": "0",
                "sunday": "0",
                "start_date": "20250101",
                "end_date": "20251231",
            }
        ],
        "calendar_dates.txt": [
            {"service_id": "svc", "date": "20250709", "exception_type": "1"}
        ],
        "shapes.txt": [
            {"shape_id": "shape_fwd", "shape_pt_lat": 0.0, "shape_pt_lon": 0.0, "shape_pt_sequence": "1"},
            {"shape_id": "shape_fwd", "shape_pt_lat": STEP_DEG, "shape_pt_lon": 0.0, "shape_pt_sequence": "2"},
            {"shape_id": "shape_fwd", "shape_pt_lat": 2 * STEP_DEG, "shape_pt_lon": 0.0, "shape_pt_sequence": "3"},
            {"shape_id": "shape_rev", "shape_pt_lat": 2 * STEP_DEG, "shape_pt_lon": 0.0, "shape_pt_sequence": "1"},
            {"shape_id": "shape_rev", "shape_pt_lat": STEP_DEG, "shape_pt_lon": 0.0, "shape_pt_sequence": "2"},
            {"shape_id": "shape_rev", "shape_pt_lat": 0.0, "shape_pt_lon": 0.0, "shape_pt_sequence": "3"},
        ],
    }
    path = tmp_path / "integration_gtfs.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, rows in tables.items():
            zf.writestr(name, _csv_text(rows))
    return path


@pytest.fixture
def integration_tables(integration_gtfs_zip):
    return load_small_tables(integration_gtfs_zip)


class ElevationByCoordinate:
    def __init__(self, mapping):
        self.mapping = {
            (round(float(lat), 12), round(float(lon), 12)): elev
            for (lat, lon), elev in mapping.items()
        }

    def get_elevation(self, latitude, longitude):
        return self.mapping.get((round(float(latitude), 12), round(float(longitude), 12)))


@pytest.fixture
def full_elevation():
    return ElevationByCoordinate(
        {
            (0.0, 0.0): 100.0,
            (STEP_DEG, 0.0): 110.0,
            (2 * STEP_DEG, 0.0): 105.0,
        }
    )


@pytest.fixture
def missing_last_elevation():
    return ElevationByCoordinate({(0.0, 0.0): 100.0, (STEP_DEG, 0.0): 110.0})


@pytest.fixture
def integration_motion():
    return MotionParams(
        accel_ms2=1.0,
        decel_ms2=1.0,
        dt_s=0.25,
        default_speed_cap_ms=12.0,
        max_speed_cap_ms=20.0,
        stop_prob=1.0,
        red_wait_s=10.0,
        signal_time_policy="preserve_schedule",
        max_signal_wait_share=1.0,
        use_hourly_signal_stop_probability=False,
        stop_prob_by_hour={},
    )


@pytest.fixture
def integration_vehicle():
    return VehicleParams(
        curb_mass_kg=10_000.0,
        passenger_mass_kg=75.0,
        frontal_area_m2=8.0,
        drag_coeff=0.6,
        roll_coeff=0.01,
        rot_inertia_factor=1.0,
        eta_driveline=0.9,
        eta_motor=0.9,
        regen_fraction=0.5,
        regen_power_cap_kW=100.0,
        regen_min_speed_ms=0.0,
        aux_power_kW=5.0,
        battery_usable_kWh=400.0,
        air_density=1.2,
        g=9.81,
    )


@pytest.fixture
def demand_profile():
    return HourlyDemandProfile.from_percent({8: 100.0, 9: 50.0})


@pytest.fixture
def weather_series():
    df = pd.DataFrame(
        {"temp": [0.0, 20.0, 30.0], "rh": [0.80, 0.50, 0.90], "solar": [0.0, 0.0, 0.0]},
        index=pd.to_datetime(["2025-07-09 08:00:00", "2025-07-09 09:00:00", "2025-07-09 10:00:00"]),
    )
    return WeatherSeries(df)


@pytest.fixture
def service_date():
    return date(2025, 7, 9)
