"""Project configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "model.yaml"
CONFIG_ENV_VAR = "BEB_CONFIG"

_BUILTIN_DEFAULTS: dict[str, Any] = {
    "paths": {
        "data_dir": "data",
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "gtfs_zip": "data/raw/GTFS_Realtime.zip",
        "passenger_loading_csv": (
            "data/processed/bus_hourly_loading_profile_2013_2023_long.csv"
        ),
        "weather_csv": "data/processed/cork_weather_hourly_model_input_2021_2025.csv",
        "gtfs_output_dir": "data/processed",
        "srtm_cache_dir": "data/srtm_cache",
        "traffic_signals_csv": "data/processed/traffic_signals.csv",
        "synthetic_segment_results_csv": "data/processed/beb_segment_results.csv",
        "synthetic_soc_trace_png": "reports/figures/beb_soc_trace.png",
    },
    "simulation": {
        "date": "2025-01-06",
    },
    "vehicle": {
        "curb_mass_kg": 14_000.0,
        "passenger_mass_kg": 70.0,
        "frontal_area_m2": 10.8,
        "drag_coeff": 0.70,
        "roll_coeff": 0.0085,
        "rot_inertia_factor": 1.05,
        "eta_driveline": 0.92,
        "eta_motor": 0.90,
        "regen_fraction": 0.60,
        "regen_power_cap_kW": 150.0,
        "regen_min_speed_ms": 2.0,
        "aux_power_kW": 7.0,
        "battery_usable_kWh": 410.0,
        "air_density": 1.225,
        "g": 9.81,
    },
    "gtfs": {
        "default_routes": ["208"],
        "simulation_level": "block",
        "direction_id": 0,
        "day": "monday",
        "tolerance_min": 15,
        "flat_passengers": 20,
    },
    "passenger_loading": {
        "demand_city": "Cork",
        "crush_capacity": 85,
        "enabled": True,
        "load_shape": "beta",
        "board_pos": 0.25,
        "alight_pos": 0.75,
        "concentration": 6.0,
        "board_frac": 0.2,
        "alight_frac": 0.2,
        "floor_frac": 0.0,
        "hour_mode": "midpoint",
    },
    "weather": {
        "enabled": True,
        "climate_control": {
            "heat_below_c": 10.0,
            "cool_above_c": 20.0,
            "heating_months": [11, 12, 1, 2, 3],
        },
        "hvac": {
            "base_aux_kW": 3.0,
            "cabin_loss_W_per_K": 500.0,
            "solar_aperture_m2": 6.0,
            "pax_sensible_W": 90.0,
            "use_passenger_gain": True,
            "heater_type": "heat_pump",
            "heat_cop_at_0c": 2.2,
            "heat_cop_slope": 0.06,
            "heat_cop_min": 1.0,
            "heat_cop_max": 3.5,
            "cool_cop": 2.5,
            "latent_load_at_full_rh": 0.30,
            "hvac_max_kW": 30.0,
            "model": "thermal",
            "hour_mode": "midpoint",
            "k_heat_kW_per_K": 0.40,
            "k_cool_kW_per_K": 0.35,
        },
    },
    "traffic_signals": {
        "enabled": True,
        "snap_radius_m": 30.0,
        "cluster_radius_m": 35.0,
        "fallback_per_km": 2.0,
        "refresh": False,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config_path(config_path: str | Path | None = None) -> Path:
    selected = config_path or os.getenv(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
    return Path(selected).expanduser()


@lru_cache(maxsize=None)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML project config, merged over stable built-in defaults."""
    path = _config_path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return dict(_BUILTIN_DEFAULTS)
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return _deep_merge(_BUILTIN_DEFAULTS, loaded)


def get_section(name: str, config_path: str | Path | None = None) -> dict[str, Any]:
    section = load_config(config_path).get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"config section {name!r} must be a mapping.")
    return section


def project_path(value: str | Path | None) -> Path | None:
    """Resolve a config path relative to PROJECT_ROOT unless already absolute."""
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_path(
    name: str,
    config_path: str | Path | None = None,
    default: str | Path | None = None,
) -> Path | None:
    value = get_section("paths", config_path).get(name, default)
    return project_path(value)


def vehicle_params(config_path: str | Path | None = None) -> dict[str, float]:
    return {k: float(v) for k, v in get_section("vehicle", config_path).items()}


DATA_DIR = get_path("data_dir")
RAW_DATA_DIR = get_path("raw_data_dir")
PROCESSED_DATA_DIR = get_path("processed_data_dir")
