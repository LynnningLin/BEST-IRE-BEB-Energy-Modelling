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
        "speed_caps_csv": "data/processed/speed_caps.csv",
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
    # Motion-profile parameters for beb_soc_model.build_speed_profile().
    # These separate the PHYSICAL speed cap from the GTFS-derived target cruise
    # speed: the cap bounds what the vehicle may do; GTFS only shapes the target.
    "motion": {
        "accel_ms2": 1.0,
        "decel_ms2": 1.2,
        "dt_s": 0.5,
        # Cap used when a segment has no seg.speed_cap_ms (e.g. speed_caps
        # disabled or OSM unresolved). 13.9 m/s = 50 km/h Irish urban default.
        "default_speed_cap_ms": 13.9,
        # Sanity clamp on any per-segment cap (OSM errors, unit mix-ups).
        # 25.0 m/s = 90 km/h.
        "max_speed_cap_ms": 25.0,
        "stop_prob": 0.5,
        "red_wait_s": 15.0,
        "signal_time_policy": "preserve_schedule",
        "max_signal_wait_share": 0.35,
        "use_hourly_signal_stop_probability": True,
        # Optional {hour: probability} mapping; None -> module default table.
        "stop_prob_by_hour": None,
    },
    # OSM maxspeed resolution for per-segment physical speed caps
    # (scripts/speed_caps.py; mirrors the traffic_signals cache workflow).
    "speed_caps": {
        "enabled": True,
        "snap_radius_m": 25.0,
        "sample_step_m": 20.0,
        "min_coverage_frac": 0.5,
        "default_cap_kmh": 50.0,
        "max_cap_kmh": 90.0,
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
    params = {k: float(v) for k, v in get_section("vehicle", config_path).items()}
    _validate_vehicle_params(params)
    return params


def motion_params(config_path: str | Path | None = None) -> dict[str, Any]:
    """Motion-profile parameters (mixed types: floats, str policy, dict table)."""
    return dict(get_section("motion", config_path))


def speed_cap_params(config_path: str | Path | None = None) -> dict[str, Any]:
    """OSM maxspeed resolution parameters for scripts/speed_caps.py."""
    return dict(get_section("speed_caps", config_path))


DATA_DIR = get_path("data_dir")
RAW_DATA_DIR = get_path("raw_data_dir")
PROCESSED_DATA_DIR = get_path("processed_data_dir")


def _require_keys(section_name: str, section: dict[str, Any], keys: set[str]) -> None:
    missing = sorted(keys - set(section))
    if missing:
        raise ValueError(
            f"config section {section_name!r} is missing required key(s): "
            f"{', '.join(missing)}"
        )


def _require_positive(name: str, value: Any) -> None:
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_non_negative(name: str, value: Any) -> None:
    if float(value) < 0:
        raise ValueError(f"{name} must be non-negative.")


def _require_interval(name: str, value: Any, low: float, high: float) -> None:
    numeric = float(value)
    if not (low <= numeric <= high):
        raise ValueError(f"{name} must lie within [{low}, {high}].")


def _require_open_closed_interval(
    name: str, value: Any, low: float, high: float
) -> None:
    numeric = float(value)
    if not (low < numeric <= high):
        raise ValueError(f"{name} must lie within ({low}, {high}].")


def _validate_vehicle_params(params: dict[str, float]) -> None:
    _require_positive("vehicle.curb_mass_kg", params["curb_mass_kg"])
    _require_positive("vehicle.battery_usable_kWh", params["battery_usable_kWh"])
    _require_non_negative("vehicle.aux_power_kW", params["aux_power_kW"])
    _require_open_closed_interval(
        "vehicle.eta_driveline", params["eta_driveline"], 0.0, 1.0
    )
    _require_open_closed_interval("vehicle.eta_motor", params["eta_motor"], 0.0, 1.0)
    _require_interval("vehicle.regen_fraction", params["regen_fraction"], 0.0, 1.0)


def validate_config(
    config_path: str | Path | None = None,
    *,
    require_existing_paths: bool = False,
) -> dict[str, Any]:
    """Validate Level 1 config structure and physical bounds.

    The returned config is the same defaults-merged mapping produced by
    ``load_config``. Path existence checks are opt-in because output paths are
    often created later, while CI tests usually use synthetic files.
    """
    cfg = load_config(config_path)
    required = {
        "simulation": {"date"},
        "paths": {
            "data_dir",
            "raw_data_dir",
            "processed_data_dir",
            "gtfs_zip",
            "passenger_loading_csv",
            "weather_csv",
            "traffic_signals_csv",
            "speed_caps_csv",
        },
        "vehicle": {
            "curb_mass_kg",
            "eta_driveline",
            "eta_motor",
            "regen_fraction",
            "aux_power_kW",
            "battery_usable_kWh",
        },
        "gtfs": {"default_routes", "simulation_level", "flat_passengers"},
        "passenger_loading": {"demand_city", "crush_capacity", "enabled"},
        "traffic_signals": {"enabled", "snap_radius_m", "fallback_per_km"},
        "motion": {
            "accel_ms2",
            "decel_ms2",
            "dt_s",
            "default_speed_cap_ms",
            "max_speed_cap_ms",
            "stop_prob",
            "red_wait_s",
            "max_signal_wait_share",
        },
        "speed_caps": {
            "enabled",
            "snap_radius_m",
            "sample_step_m",
            "min_coverage_frac",
            "default_cap_kmh",
            "max_cap_kmh",
        },
        "weather": {"enabled", "climate_control", "hvac"},
    }
    for section_name, keys in required.items():
        section = cfg.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"config section {section_name!r} must be a mapping.")
        _require_keys(section_name, section, keys)

    paths = cfg["paths"]
    path_keys = (
        "gtfs_zip",
        "passenger_loading_csv",
        "weather_csv",
        "traffic_signals_csv",
        "speed_caps_csv",
    )
    for key in path_keys:
        resolved = project_path(paths[key])
        if require_existing_paths and (resolved is None or not resolved.exists()):
            raise FileNotFoundError(f"configured path paths.{key} does not exist: {resolved}")

    _validate_vehicle_params({k: float(v) for k, v in cfg["vehicle"].items()})

    motion = cfg["motion"]
    _require_positive("motion.accel_ms2", motion["accel_ms2"])
    _require_positive("motion.decel_ms2", motion["decel_ms2"])
    _require_positive("motion.dt_s", motion["dt_s"])
    _require_positive("motion.default_speed_cap_ms", motion["default_speed_cap_ms"])
    _require_positive("motion.max_speed_cap_ms", motion["max_speed_cap_ms"])
    _require_interval("motion.stop_prob", motion["stop_prob"], 0.0, 1.0)
    _require_non_negative("motion.red_wait_s", motion["red_wait_s"])
    _require_interval(
        "motion.max_signal_wait_share", motion["max_signal_wait_share"], 0.0, 1.0
    )
    table = motion.get("stop_prob_by_hour")
    if table:
        for hour, value in dict(table).items():
            _require_interval(f"motion.stop_prob_by_hour[{hour}]", value, 0.0, 1.0)

    loading = cfg["passenger_loading"]
    _require_positive("passenger_loading.crush_capacity", loading["crush_capacity"])

    climate = cfg["weather"]["climate_control"]
    _require_keys("weather.climate_control", climate, {"heat_below_c", "cool_above_c"})
    if float(climate["heat_below_c"]) >= float(climate["cool_above_c"]):
        raise ValueError("weather heat_below_c must be below cool_above_c.")
    hvac = cfg["weather"]["hvac"]
    _require_keys("weather.hvac", hvac, {"hvac_max_kW"})
    _require_non_negative("weather.hvac.hvac_max_kW", hvac["hvac_max_kW"])

    return cfg
