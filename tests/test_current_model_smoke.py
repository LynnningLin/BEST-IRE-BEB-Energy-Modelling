import sys
import types
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault(
    "srtm", types.SimpleNamespace(get_data=lambda *args, **kwargs: None)
)

from beb_soc_model import (  # noqa: E402
    Segment,
    VehicleParams,
    build_speed_profile,
    segment_energy_kWh,
    simulate_route,
)
from gtfs_to_segment import _shape_points_with_cumdist, segments_for_trip  # noqa: E402
from passenger_loading import HourlyDemandProfile  # noqa: E402
from weather_loading import WeatherSeries  # noqa: E402
from best_ire_beb.config import get_section  # noqa: E402


class DummyElevation:
    def __init__(self, elevations_by_lat):
        self.elevations_by_lat = elevations_by_lat

    def get_elevation(self, latitude, longitude):
        return self.elevations_by_lat[round(float(latitude), 4)]


def _trip_rows():
    return [
        {
            "trip_id": "smoke_trip",
            "stop_id": "s0",
            "arrival_time": "08:00:00",
            "departure_time": "08:00:00",
            "stop_sequence": "1",
        },
        {
            "trip_id": "smoke_trip",
            "stop_id": "s1",
            "arrival_time": "08:06:00",
            "departure_time": "08:06:30",
            "stop_sequence": "2",
        },
        {
            "trip_id": "smoke_trip",
            "stop_id": "s2",
            "arrival_time": "08:15:00",
            "departure_time": "08:15:20",
            "stop_sequence": "3",
        },
        {
            "trip_id": "smoke_trip",
            "stop_id": "s3",
            "arrival_time": "08:26:00",
            "departure_time": "08:26:25",
            "stop_sequence": "4",
        },
    ]


def _stops():
    return pd.DataFrame(
        [
            {"stop_id": "s0", "stop_lat": 51.9000, "stop_lon": -8.4700},
            {"stop_id": "s1", "stop_lat": 51.9020, "stop_lon": -8.4680},
            {"stop_id": "s2", "stop_lat": 51.9040, "stop_lon": -8.4660},
            {"stop_id": "s3", "stop_lat": 51.9060, "stop_lon": -8.4640},
        ]
    )


def _shape_points():
    return _shape_points_with_cumdist(
        [
            (51.9000, -8.4700, 1),
            (51.9010, -8.4700, 2),
            (51.9020, -8.4680, 3),
            (51.9030, -8.4680, 4),
            (51.9040, -8.4660, 5),
            (51.9050, -8.4660, 6),
            (51.9060, -8.4640, 7),
        ]
    )


def _weather_series():
    df = pd.DataFrame(
        {
            "temp": [27.0, 27.5, 28.0],
            "rh": [0.70, 0.75, 0.80],
            "solar": [450.0, 500.0, 520.0],
        },
        index=pd.to_datetime(
            [
                "2025-07-09 08:00:00",
                "2025-07-09 08:30:00",
                "2025-07-09 09:00:00",
            ]
        ),
    )
    return WeatherSeries(df)


def test_current_model_smoke_exercises_gtfs_loading_weather_traffic_and_physics():
    vehicle = VehicleParams.from_config(CONFIG_PATH)
    passenger_cfg = get_section("passenger_loading", CONFIG_PATH)
    weather_cfg = get_section("weather", CONFIG_PATH)

    demand_profile = HourlyDemandProfile.from_percent(
        {hour: (10.0 if hour == 8 else 1.0) for hour in range(24)}
    )
    signal_counts = {("s0", "s1"): 1, ("s1", "s2"): 0, ("s2", "s3"): 2}
    signal_sources = {pair: "smoke_cache" for pair in signal_counts}
    elevations = DummyElevation(
        {
            51.9000: 10.0,
            51.9020: 20.0,
            51.9040: 12.0,
            51.9060: 18.0,
        }
    )

    segments = segments_for_trip(
        _trip_rows(),
        _stops(),
        passengers=5,
        elevation_data=elevations,
        demand_profile=demand_profile,
        crush_capacity=int(passenger_cfg["crush_capacity"]),
        loading_kwargs={
            "shape": passenger_cfg["load_shape"],
            "board_pos": float(passenger_cfg["board_pos"]),
            "alight_pos": float(passenger_cfg["alight_pos"]),
            "concentration": float(passenger_cfg["concentration"]),
            "board_frac": float(passenger_cfg["board_frac"]),
            "alight_frac": float(passenger_cfg["alight_frac"]),
            "floor_frac": float(passenger_cfg["floor_frac"]),
            "hour_mode": passenger_cfg["hour_mode"],
        },
        weather_series=_weather_series(),
        service_date=date.fromisoformat(str(get_section("simulation", CONFIG_PATH)["date"])),
        weather_kwargs={"config_path": CONFIG_PATH},
        shape_points=_shape_points(),
        signal_count_map=signal_counts,
        signal_source_map=signal_sources,
        verbose=False,
    )

    assert len(segments) == 3
    assert all(seg.length_m > 0 for seg in segments)
    assert any(seg.grade > 0 for seg in segments)
    assert any(seg.grade < 0 for seg in segments)

    assert [seg.n_signals for seg in segments] == [1, 0, 2]
    assert [seg.signal_source for seg in segments] == ["smoke_cache"] * 3

    passenger_loads = [seg.passengers for seg in segments]
    assert max(passenger_loads) > 40
    assert len(set(passenger_loads)) > 1

    base_aux_kW = float(weather_cfg["hvac"]["base_aux_kW"])
    assert all(seg.aux_power_kW >= base_aux_kW for seg in segments)
    assert any(seg.aux_power_kW > base_aux_kW for seg in segments)

    no_signal = Segment(
        length_m=segments[0].length_m,
        grade=segments[0].grade,
        v_cruise_ms=segments[0].v_cruise_ms,
        dwell_s=segments[0].dwell_s,
        passengers=segments[0].passengers,
        run_time_s=segments[0].run_time_s,
        n_signals=0,
    )
    with_signal = Segment(
        length_m=segments[0].length_m,
        grade=segments[0].grade,
        v_cruise_ms=segments[0].v_cruise_ms,
        dwell_s=segments[0].dwell_s,
        passengers=segments[0].passengers,
        run_time_s=segments[0].run_time_s,
        n_signals=2,
    )
    _, no_signal_diag = build_speed_profile(
        no_signal, stop_prob=1.0, return_diagnostics=True
    )
    _, with_signal_diag = build_speed_profile(
        with_signal, stop_prob=1.0, return_diagnostics=True
    )
    assert no_signal_diag["n_motion_sublinks"] == 1
    assert with_signal_diag["n_motion_sublinks"] > no_signal_diag["n_motion_sublinks"]
    assert segment_energy_kWh(with_signal, vehicle) > segment_energy_kWh(no_signal, vehicle)

    empty_bus = Segment(length_m=500.0, passengers=0, run_time_s=80.0)
    loaded_bus = Segment(length_m=500.0, passengers=50, run_time_s=80.0)
    empty_bus.aux_power_kW = 0.0
    loaded_bus.aux_power_kW = 0.0
    assert segment_energy_kWh(loaded_bus, vehicle) > segment_energy_kWh(empty_bus, vehicle)

    results = simulate_route(segments, vehicle, soc0_pct=100.0)
    assert len(results) == len(segments)
    assert results["net_battery_energy_kWh"].sum() > 0.0
    assert results["SoC_end_%"].iloc[-1] < results["SoC_start_%"].iloc[0]
    assert set(
        [
            "passengers",
            "n_signals",
            "net_battery_energy_kWh",
            "gross_consumed_kWh",
            "regen_recovered_kWh",
            "aux_energy_kWh",
            "SoC_end_%",
        ]
    ).issubset(results.columns)
