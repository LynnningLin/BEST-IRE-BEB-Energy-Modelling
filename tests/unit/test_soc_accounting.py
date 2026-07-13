import pytest

import beb_soc_model as model
from beb_soc_model import Segment, SegmentEnergy, VehicleParams, simulate_route


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


def _fixed_energy(monkeypatch, energies):
    values = iter(energies)

    def fake_breakdown(seg, p, soc_start_pct=None, motion_params=None):
        value = next(values)
        return SegmentEnergy(
            net_battery_energy_kWh=value,
            gross_consumed_kWh=max(value, 0.0),
            regen_recovered_kWh=max(-value, 0.0),
            aux_energy_kWh=0.0,
            motion_diagnostics={"actual_profile_time_s": 0.0},
        )

    monkeypatch.setattr(model, "segment_energy_breakdown_kWh", fake_breakdown)


def test_single_segment_soc_matches_analytical_result(monkeypatch):
    _fixed_energy(monkeypatch, [20.0])

    df = simulate_route([Segment(length_m=1000.0)], _vehicle(), soc0_pct=90.0)

    assert df["SoC_end_%"].iloc[-1] == pytest.approx(85.0)


def test_sequential_segments_total_soc_change_matches_total_net_energy(monkeypatch):
    _fixed_energy(monkeypatch, [20.0, 10.0, -5.0])

    df = simulate_route(
        [Segment(length_m=1000.0), Segment(length_m=500.0), Segment(length_m=500.0)],
        _vehicle(),
        soc0_pct=90.0,
    )

    total_energy = df["net_battery_energy_kWh"].sum()
    expected_end = 90.0 - total_energy / 400.0 * 100.0
    assert df["SoC_end_%"].iloc[-1] == pytest.approx(expected_end)
    assert df["SoC_start_%"].iloc[1] == pytest.approx(df["SoC_end_%"].iloc[0])
    assert df["SoC_start_%"].iloc[2] == pytest.approx(df["SoC_end_%"].iloc[1])


def test_regeneration_cannot_raise_soc_above_100(monkeypatch):
    _fixed_energy(monkeypatch, [-20.0])

    df = simulate_route([Segment(length_m=1000.0)], _vehicle(), soc0_pct=98.0)

    assert df["SoC_end_%"].iloc[-1] == 100.0


def test_behaviour_near_zero_soc_is_linear_not_hidden():
    df = simulate_route(
        [Segment(length_m=300.0, v_cruise_ms=8.0)],
        _vehicle(battery_usable_kWh=1.0),
        soc0_pct=1.0,
    )

    assert df["SoC_end_%"].iloc[-1] < df["SoC_start_%"].iloc[0]


def test_trip_level_reset_and_duty_level_continuity(monkeypatch):
    _fixed_energy(monkeypatch, [20.0, 20.0])
    p = _vehicle()

    trip_a = simulate_route([Segment(length_m=1000.0)], p, soc0_pct=90.0)
    trip_b_reset = simulate_route([Segment(length_m=1000.0)], p, soc0_pct=90.0)

    assert trip_a["SoC_start_%"].iloc[0] == 90.0
    assert trip_b_reset["SoC_start_%"].iloc[0] == 90.0
    assert trip_a["SoC_end_%"].iloc[-1] == pytest.approx(trip_b_reset["SoC_end_%"].iloc[-1])

    _fixed_energy(monkeypatch, [20.0, 20.0])
    duty = simulate_route([Segment(length_m=1000.0), Segment(length_m=1000.0)], p, soc0_pct=90.0)

    assert duty["SoC_start_%"].iloc[1] == pytest.approx(duty["SoC_end_%"].iloc[0])
    assert duty["SoC_end_%"].iloc[-1] == pytest.approx(80.0)
