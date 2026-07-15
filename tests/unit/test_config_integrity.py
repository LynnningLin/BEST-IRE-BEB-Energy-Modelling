from pathlib import Path

import pytest

from best_ire_beb.config import PROJECT_ROOT, get_path, load_config, validate_config
from beb_soc_model import (
    MotionParams,
    Segment,
    VehicleParams,
    _signal_stop_prob_for_segment,
    build_speed_profile,
)


REQUIRED_SECTIONS = {
    "simulation",
    "paths",
    "vehicle",
    "gtfs",
    "passenger_loading",
    "traffic_signals",
    "motion",
    "speed_caps",
    "weather",
}


def _write_yaml(tmp_path, text):
    path = tmp_path / "model.yaml"
    path.write_text(text, encoding="utf-8")
    load_config.cache_clear()
    return path


def test_model_yaml_has_required_sections_and_keys():
    cfg = validate_config()

    assert REQUIRED_SECTIONS.issubset(cfg), "model.yaml lacks a required top-level section"
    assert cfg["weather"]["climate_control"]["heat_below_c"] < cfg["weather"][
        "climate_control"
    ]["cool_above_c"], "heating threshold must be below cooling threshold"


def test_configured_input_paths_exist_for_checked_in_model_config():
    validate_config(require_existing_paths=True)

    for key in (
        "gtfs_zip",
        "passenger_loading_csv",
        "weather_csv",
        "traffic_signals_csv",
        "speed_caps_csv",
    ):
        resolved = get_path(key)
        assert resolved is not None and resolved.is_absolute(), f"paths.{key} is not absolute"
        assert resolved.exists(), f"configured input path is missing: paths.{key}={resolved}"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("vehicle:\n  curb_mass_kg: 0\n", "curb_mass_kg"),
        ("vehicle:\n  battery_usable_kWh: -1\n", "battery_usable_kWh"),
        ("vehicle:\n  aux_power_kW: -0.1\n", "aux_power_kW"),
        ("vehicle:\n  eta_motor: 1.2\n", "eta_motor"),
        ("vehicle:\n  eta_driveline: 0\n", "eta_driveline"),
        ("vehicle:\n  regen_fraction: -0.01\n", "regen_fraction"),
        ("passenger_loading:\n  crush_capacity: 0\n", "crush_capacity"),
        (
            "weather:\n  climate_control:\n    heat_below_c: 28\n    cool_above_c: 26\n",
            "heat_below_c",
        ),
        ("weather:\n  hvac:\n    hvac_max_kW: -1\n", "hvac_max_kW"),
    ],
)
def test_invalid_config_values_are_rejected(tmp_path, override, message):
    path = _write_yaml(tmp_path, override)

    with pytest.raises(ValueError, match=message):
        validate_config(path)


@pytest.mark.parametrize(
    "override",
    [
        "motion:\n  accel_ms2: 0\n",
        "motion:\n  decel_ms2: -1\n",
        "motion:\n  dt_s: 0\n",
        "motion:\n  default_speed_cap_ms: 0\n",
        "motion:\n  max_speed_cap_ms: -1\n",
    ],
)
def test_motion_params_reject_invalid_physical_values(tmp_path, override):
    path = _write_yaml(tmp_path, override)

    with pytest.raises(ValueError):
        MotionParams.from_config(path)


def test_vehicle_params_from_config_rejects_invalid_vehicle_values(tmp_path):
    path = _write_yaml(tmp_path, "vehicle:\n  battery_usable_kWh: 0\n")

    with pytest.raises(ValueError, match="battery_usable_kWh"):
        VehicleParams.from_config(path)


def test_speed_cap_default_is_clamped_when_max_is_lower():
    params = MotionParams(default_speed_cap_ms=12.0, max_speed_cap_ms=8.0)

    cap, source = params.resolve_cap(Segment(length_m=100.0))

    assert cap == 8.0, "default speed cap should clamp to motion.max_speed_cap_ms"
    assert source == "config_default"


def test_signal_probability_is_clamped_to_valid_probability():
    prob, hour, source = _signal_stop_prob_for_segment(
        Segment(length_m=100.0, from_stop_departure_time="08:00:00"),
        default_prob=1.7,
        stop_prob_by_hour={8: -0.2},
    )

    assert prob == 0.0, "hourly signal probability should clamp into [0, 1]"
    assert hour == 8
    assert source == "hourly"


def test_negative_red_light_wait_is_clamped_in_profile_diagnostics():
    seg = Segment(
        length_m=100.0,
        v_cruise_ms=8.0,
        run_time_s=60.0,
        n_signals=1,
        from_stop_departure_time="08:00:00",
    )

    _, diag = build_speed_profile(
        seg,
        red_wait_s=-10.0,
        stop_prob=1.0,
        stop_prob_by_hour={},
        return_diagnostics=True,
    )

    assert diag["signal_wait_s"] == 0.0, "negative red-light wait must not add time"


def test_project_root_assumption_matches_repository():
    assert (PROJECT_ROOT / "configs" / "model.yaml").exists()
    assert Path(get_path("data_dir")) == PROJECT_ROOT / "data"
