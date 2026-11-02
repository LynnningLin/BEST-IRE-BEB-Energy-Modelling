from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import beb_soc_model as model
from beb_soc_model import MotionParams, Segment, VehicleParams
from trip_motion import build_trip_events, solve_motion_leg, solve_trip_motion
from signal_motion_cases import CASES, build_validation_segment, motion_params_for_case


def spatial_segment(length=600.0, positions=(0.0, 80.0, 350.0), runtime=100.0):
    return Segment(length, run_time_s=runtime, dwell_s=12.3, trip_id="trip",
                   segment_index=0, service_date="2025-07-09",
                   segment_geometry_id="shape:a:b", from_stop_departure_time="25:00:00",
                   n_signals=len(positions), signal_events=[
                       {"signal_event_id": f"signal:{i}", "distance_along_segment_m": x}
                       for i, x in enumerate(positions)])


def assert_profiles_equal(actual, expected):
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_phase4_profiles_and_energy_match(case):
    seg = build_validation_segment(case)
    mp = motion_params_for_case(case)
    old_profile, old_diag = model.build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
    trip = solve_trip_motion([seg], mp)
    assert_profiles_equal(trip.segment_profiles[0], old_profile)
    for key, expected in old_diag.items():
        actual = trip.segment_diagnostics[0][key]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected, abs=1e-10)
        else:
            assert actual == expected
    old = model.segment_energy_breakdown_kWh(seg, VehicleParams(), 99.99, mp)
    new = model.segment_energy_breakdown_kWh(
        seg, VehicleParams(), 99.99, mp,
        profile_and_diagnostics=(trip.segment_profiles[0], trip.segment_diagnostics[0]))
    for key in ("net_battery_energy_kWh", "gross_consumed_kWh", "regen_recovered_kWh", "aux_energy_kWh"):
        assert getattr(new, key) == pytest.approx(getattr(old, key), abs=1e-12)


@pytest.mark.parametrize("policy", ["preserve_schedule", "add_delay"])
@pytest.mark.parametrize("runtime", [None, 0.0, 8.0, 65.0, 180.0])
@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("spatial", [False, True])
def test_b0_equivalence_across_timing_and_signal_modes(policy, runtime, probability, spatial):
    seg = spatial_segment(runtime=runtime)
    if not spatial:
        seg.signal_events = None
    mp = MotionParams(signal_time_policy=policy, stop_prob=probability,
                      use_hourly_signal_stop_probability=False, dt_s=0.7, red_wait_s=0.1)
    profile, diag = model.build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
    trip = solve_trip_motion([seg], mp)
    assert_profiles_equal(trip.segment_profiles[0], profile)
    for name in ("schedule_delay_s", "actual_moving_time_s", "signal_wait_applied_s",
                 "minimum_feasible_moving_time_s"):
        assert trip.segment_diagnostics[0][name] == pytest.approx(diag[name], abs=1e-10)
    assert seg.signal_event_diagnostics == diag["signal_event_diagnostics"]


def test_trip_timeline_has_exact_positions_continuity_and_single_dwell_accounting():
    segments = [spatial_segment(runtime=15.0), spatial_segment(400.0, (125.0,), 160.0)]
    segments[1].segment_index = 1
    segments[1].aux_power_kW = 11.0
    segments[1].grade = -0.06
    segments[1].passengers = 70
    mp = MotionParams(stop_prob=1, use_hourly_signal_stop_probability=False)
    trip = solve_trip_motion(segments, mp)
    assert trip.length_m == 1000.0
    gtfs = [e for e in trip.events if e.kind == "gtfs_stop"]
    assert [e.distance_m for e in gtfs] == [0.0, 600.0, 1000.0]
    assert all(e.required_speed_ms == e.actual_speed_ms == 0.0 for e in gtfs)
    assert all(e.departure_time_s - e.arrival_time_s == pytest.approx(e.dwell_s) for e in gtfs)
    assert np.all(np.diff(trip.profile[0]) > 0)
    assert np.all(np.diff(trip.distance_m) >= -1e-9)
    for a, b in zip(trip.spans, trip.spans[1:]):
        assert a.end_time_s == pytest.approx(b.start_time_s)
        assert a.end_distance_m == pytest.approx(b.start_distance_m)
    for span in trip.spans:
        if span.leg:
            x, v = span.leg.position_speed(span.leg.duration_s)
            assert x == span.end_distance_m
            assert v == 0
    assert trip.duration_s == pytest.approx(sum(
        d["actual_segment_time_s"] + s.dwell_s for d, s in zip(trip.segment_diagnostics, segments)))
    assert trip.segment_diagnostics[-1]["cumulative_schedule_delay_s"] == pytest.approx(
        sum(d["schedule_delay_s"] for d in trip.segment_diagnostics))
    # The later feasible budget does not erase earlier B0 delay.
    assert trip.segment_diagnostics[1]["schedule_delay_s"] == pytest.approx(0)
    assert trip.segment_diagnostics[1]["cumulative_schedule_delay_s"] > 0
    old = model.simulate_route(deepcopy(segments), VehicleParams(), 99.9, mp)
    new = model.simulate_route(deepcopy(segments), VehicleParams(), 99.9,
                               replace(mp, solver_mode="trip_continuous"))
    pd.testing.assert_frame_equal(new[old.columns], old)
    assert set(new.solver_mode) == {"trip_continuous"}


def test_pass_events_are_timestamped_without_forcing_stops_or_resampling():
    seg = spatial_segment(700.0, (125.0, 525.0), 100.0)
    trip = solve_trip_motion([seg], MotionParams(stop_prob=0, use_hourly_signal_stop_probability=False))
    assert sum(s.kind == "motion" for s in trip.spans) == 1
    leg = trip.spans[0].leg
    for event in trip.events:
        if event.kind == "signal":
            assert event.required_speed_ms is None
            assert event.actual_speed_ms > 0
            assert event.arrival_time_s == event.departure_time_s
            assert float(leg.position_speed(event.arrival_time_s)[0]) == pytest.approx(event.distance_m)


def test_continuous_dispatch_does_not_call_legacy_profile_builder(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Continuous solver called the legacy segment builder")
    monkeypatch.setattr(model, "build_speed_profile", forbidden)
    monkeypatch.setattr(model, "_single_segment_profile", forbidden)
    result = model.simulate_route([spatial_segment()], VehicleParams(), motion_params=MotionParams(
        solver_mode="trip_continuous"))
    assert len(result) == 1


def test_default_and_configuration_modes(tmp_path):
    from best_ire_beb.config import validate_config
    assert MotionParams().solver_mode == "segment_legacy"
    path = tmp_path / "motion.yaml"
    path.write_text("motion:\n  solver_mode: trip_continuous\n", encoding="utf-8")
    assert MotionParams.from_config(path).solver_mode == "trip_continuous"
    assert validate_config(path)["motion"]["solver_mode"] == "trip_continuous"
    with pytest.raises(ValueError, match="solver_mode"):
        MotionParams(solver_mode="typo")
    bad = tmp_path / "bad.yaml"
    bad.write_text("motion:\n  solver_mode: typo\n", encoding="utf-8")
    with pytest.raises(ValueError, match="solver_mode"):
        validate_config(bad)


def test_boundary_primitive_supports_nonzero_speeds_but_gtfs_remains_b0():
    leg = solve_motion_leg(10.0, 310.0, 4.0, 7.0, 12.0, 15.0, 1.0, 1.2, 35.0)
    assert leg.duration_s == pytest.approx(35.0)
    assert leg.position_speed(0.0) == (10.0, 4.0)
    assert leg.position_speed(35.0) == (310.0, 7.0)
    # Analytic ramp/cruise distances must sum without endpoint clamping.
    integrated = ((4 + leg.peak_speed_ms)*leg.t_acc_s/2 + leg.peak_speed_ms*leg.t_cruise_s
                  + (7 + leg.peak_speed_ms)*leg.t_dec_s/2)
    assert integrated == pytest.approx(300)
    events, _ = build_trip_events([spatial_segment()])
    assert all(e.required_speed_ms == 0 for e in events if e.kind == "gtfs_stop")
    with pytest.raises(ValueError, match="Insufficient distance"):
        solve_motion_leg(0, 1, 0, 10, 12, 15, 1, 1.2)
    with pytest.raises(ValueError, match="speed-valley"):
        solve_motion_leg(0, 300, 10, 10, 12, 15, 1, 1.2, 100)


def test_empty_invalid_and_missing_schedule():
    assert solve_trip_motion([]).duration_s == 0
    assert model.simulate_route([], VehicleParams(), motion_params=MotionParams(solver_mode="trip_continuous")).empty
    for seg in (Segment(0), Segment(-1), Segment(float("nan")), Segment(100, dwell_s=-1)):
        with pytest.raises(ValueError):
            solve_trip_motion([seg])
    with pytest.raises(ValueError, match="one trip"):
        solve_trip_motion([Segment(100, trip_id="a"), Segment(100, trip_id="b")])
    trip = solve_trip_motion([Segment(100, run_time_s=None), Segment(200, run_time_s=80)])
    assert trip.segment_diagnostics[-1]["cumulative_schedule_delay_s"] is None


def test_sub_machine_precision_tail_samples_do_not_duplicate_trip_timestamps():
    segments = [Segment(300, run_time_s=100.00000000000001, dwell_s=0),
                Segment(400, run_time_s=100.00000000000001, dwell_s=0)]
    trip = solve_trip_motion(segments)
    assert np.all(np.diff(trip.profile[0]) > 0)
    assert sum(trip.profile[3]) == pytest.approx(trip.duration_s, abs=1e-10)
    for seg, profile in zip(segments, trip.segment_profiles):
        assert_profiles_equal(profile, model.build_speed_profile(seg))


