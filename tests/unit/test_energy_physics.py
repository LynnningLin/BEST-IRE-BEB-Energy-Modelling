import numpy as np
import pytest

import beb_soc_model as model
from beb_soc_model import MotionParams, Segment, VehicleParams, segment_energy_breakdown_kWh


def _vehicle(**overrides):
    defaults = {
        "curb_mass_kg": 10_000.0,
        "passenger_mass_kg": 75.0,
        "frontal_area_m2": 8.0,
        "drag_coeff": 0.6,
        "roll_coeff": 0.01,
        "rot_inertia_factor": 1.0,
        "eta_driveline": 0.9,
        "eta_motor": 0.9,
        "regen_fraction": 0.5,
        "regen_power_cap_kW": 100.0,
        "regen_min_speed_ms": 0.0,
        "aux_power_kW": 5.0,
        "battery_usable_kWh": 400.0,
        "air_density": 1.2,
        "g": 9.81,
    }
    defaults.update(overrides)
    return VehicleParams(**defaults)


def _motion():
    return MotionParams(
        accel_ms2=1.0,
        decel_ms2=1.0,
        dt_s=0.25,
        default_speed_cap_ms=20.0,
        max_speed_cap_ms=30.0,
        stop_prob=0.0,
        red_wait_s=0.0,
        signal_time_policy="preserve_schedule",
        max_signal_wait_share=0.0,
        use_hourly_signal_stop_probability=False,
        stop_prob_by_hour={},
    )


def test_flat_constant_speed_energy_matches_analytical_reference(monkeypatch):
    p = _vehicle()
    seg = Segment(length_m=1000.0, grade=0.0, passengers=0, dwell_s=0.0)
    v = 10.0
    duration = 100.0

    def constant_profile(_seg, motion_params=None, return_diagnostics=False, **kwargs):
        profile = (
            np.array([0.0]),
            np.array([v]),
            np.array([0.0]),
            np.array([duration]),
        )
        diag = {"actual_profile_time_s": duration}
        return (profile, diag) if return_diagnostics else profile

    monkeypatch.setattr(model, "build_speed_profile", constant_profile)

    energy = segment_energy_breakdown_kWh(seg, p)

    m = p.curb_mass_kg
    f_roll = p.roll_coeff * m * p.g
    f_aero = 0.5 * p.air_density * p.drag_coeff * p.frontal_area_m2 * v**2
    wheel_power = (f_roll + f_aero) * v
    expected_gross = wheel_power / (p.eta_driveline * p.eta_motor) * duration / 3.6e6
    expected_aux = p.aux_power_kW * duration / 3600.0

    assert energy.gross_consumed_kWh == pytest.approx(expected_gross)
    assert energy.aux_energy_kWh == pytest.approx(expected_aux)
    assert energy.net_battery_energy_kWh == pytest.approx(expected_gross + expected_aux)


def test_energy_identity_holds_for_sampled_profile():
    p = _vehicle()
    seg = Segment(length_m=300.0, grade=0.01, v_cruise_ms=12.0, passengers=10)

    energy = segment_energy_breakdown_kWh(seg, p, motion_params=_motion())

    assert energy.net_battery_energy_kWh == pytest.approx(
        energy.gross_consumed_kWh + energy.aux_energy_kWh - energy.regen_recovered_kWh
    )


def test_directional_energy_relationships():
    mp = _motion()
    base = Segment(length_m=300.0, grade=0.0, v_cruise_ms=12.0, passengers=0)
    flat = segment_energy_breakdown_kWh(base, _vehicle(), motion_params=mp).net_battery_energy_kWh
    heavy = segment_energy_breakdown_kWh(
        Segment(length_m=300.0, grade=0.0, v_cruise_ms=12.0, passengers=50),
        _vehicle(),
        motion_params=mp,
    ).net_battery_energy_kWh
    uphill = segment_energy_breakdown_kWh(
        Segment(length_m=300.0, grade=0.04, v_cruise_ms=12.0, passengers=0),
        _vehicle(),
        motion_params=mp,
    ).net_battery_energy_kWh
    downhill = segment_energy_breakdown_kWh(
        Segment(length_m=300.0, grade=-0.04, v_cruise_ms=12.0, passengers=0),
        _vehicle(),
        motion_params=mp,
    ).net_battery_energy_kWh

    assert heavy > flat
    assert uphill > flat
    assert downhill < flat


def test_auxiliary_power_increases_energy_linearly_with_time(monkeypatch):
    seg = Segment(length_m=1000.0, dwell_s=20.0)
    duration = 100.0

    def idle_like_profile(_seg, motion_params=None, return_diagnostics=False, **kwargs):
        profile = (np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([duration]))
        diag = {"actual_profile_time_s": duration}
        return (profile, diag) if return_diagnostics else profile

    monkeypatch.setattr(model, "build_speed_profile", idle_like_profile)

    low = segment_energy_breakdown_kWh(seg, _vehicle(aux_power_kW=2.0))
    high = segment_energy_breakdown_kWh(seg, _vehicle(aux_power_kW=7.0))

    assert high.net_battery_energy_kWh - low.net_battery_energy_kWh == pytest.approx(
        (7.0 - 2.0) * (duration + seg.dwell_s) / 3600.0
    )


def test_regen_cap_and_regen_fraction_relationships():
    mp = _motion()
    seg = Segment(length_m=300.0, grade=-0.08, v_cruise_ms=12.0)

    no_regen = segment_energy_breakdown_kWh(
        seg, _vehicle(regen_fraction=0.0), soc_start_pct=50.0, motion_params=mp
    )
    regen = segment_energy_breakdown_kWh(
        seg, _vehicle(regen_fraction=0.8, regen_power_cap_kW=100.0),
        soc_start_pct=50.0,
        motion_params=mp,
    )
    capped = segment_energy_breakdown_kWh(
        seg, _vehicle(regen_fraction=0.8, regen_power_cap_kW=1.0),
        soc_start_pct=50.0,
        motion_params=mp,
    )

    assert no_regen.net_battery_energy_kWh >= regen.net_battery_energy_kWh
    assert capped.regen_recovered_kWh <= regen.regen_recovered_kWh


def test_drag_rolling_and_determinism():
    mp = _motion()
    seg = Segment(length_m=300.0, grade=0.0, v_cruise_ms=12.0)

    base = segment_energy_breakdown_kWh(seg, _vehicle(), motion_params=mp)
    drag = segment_energy_breakdown_kWh(seg, _vehicle(drag_coeff=0.9), motion_params=mp)
    roll = segment_energy_breakdown_kWh(seg, _vehicle(roll_coeff=0.02), motion_params=mp)
    repeat = segment_energy_breakdown_kWh(seg, _vehicle(), motion_params=mp)

    assert drag.net_battery_energy_kWh > base.net_battery_energy_kWh
    assert roll.net_battery_energy_kWh > base.net_battery_energy_kWh
    assert repeat.net_battery_energy_kWh == pytest.approx(base.net_battery_energy_kWh)
