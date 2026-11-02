import numpy as np
import pytest

import beb_soc_model as model
from beb_soc_model import (
    MotionParams,
    Segment,
    _freeflow_duration,
    build_speed_profile,
)


def _mp(**overrides):
    defaults = {
        "accel_ms2": 1.0,
        "decel_ms2": 1.0,
        "dt_s": 0.25,
        "default_speed_cap_ms": 10.0,
        "max_speed_cap_ms": 30.0,
        "stop_prob": 1.0,
        "red_wait_s": 10.0,
        "signal_time_policy": "preserve_schedule",
        "max_signal_wait_share": 1.0,
        "use_hourly_signal_stop_probability": False,
        "stop_prob_by_hour": {},
    }
    defaults.update(overrides)
    return MotionParams(**defaults)


def _integrated_distance(profile):
    _t, v, a, step = profile
    return float(np.sum(v * step + 0.5 * a * step**2))


def _event(event_id, along_m):
    return {
        "signal_event_id": event_id,
        "segment_geometry_id": "shape:shape1|a|b",
        "from_stop_id": "a",
        "to_stop_id": "b",
        "shape_id": "shape1",
        "distance_along_segment_m": float(along_m),
        "segment_length_m": 800.0,
        "signal_lat": 51.0,
        "signal_lon": -8.0,
        "source": "osm",
    }


def _spatial_segment(events, trip_id="trip_a"):
    return Segment(
        length_m=800.0,
        v_cruise_ms=12.0,
        run_time_s=160.0,
        n_signals=len(events),
        segment_geometry_id="shape:shape1|a|b",
        shape_id="shape1",
        trip_id=trip_id,
        service_date="2025-07-09",
        segment_index=0,
        from_stop_departure_time="08:00:00",
        signal_events=events,
    )


def assert_profile_valid(profile, diag, seg, mp, distance_tol=1.5):
    t, v, a, step = profile
    assert len(t) == len(v) == len(a) == len(step)
    assert len(t) > 0
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(step))
    assert np.all(step > 0)
    assert np.min(v) >= -1e-9
    assert np.max(v) <= diag["speed_cap_ms"] + 1e-9
    assert np.max(a) <= mp.accel_ms2 + 1e-9
    assert np.min(a) >= -mp.decel_ms2 - 1e-9
    assert np.sum(step) == pytest.approx(diag["actual_profile_time_s"], abs=1e-9)
    assert _integrated_distance(profile) == pytest.approx(seg.length_m, abs=distance_tol)


@pytest.mark.parametrize(
    ("name", "seg", "expect_infeasible"),
    [
        ("feasible", Segment(length_m=100.0, v_cruise_ms=8.0, run_time_s=25.0), False),
        ("infeasible", Segment(length_m=100.0, v_cruise_ms=8.0, run_time_s=18.0), True),
        ("triangular", Segment(length_m=20.0, v_cruise_ms=10.0, run_time_s=None), False),
        ("trapezoidal", Segment(length_m=200.0, v_cruise_ms=10.0, run_time_s=None), False),
    ],
)
def test_basic_motion_profile_scenarios(name, seg, expect_infeasible):
    mp = _mp()

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert diag["schedule_infeasible"] is expect_infeasible
    if name == "triangular":
        assert not np.any(profile[2] == 0.0), "short segment should have no cruise phase"
    if name == "trapezoidal":
        assert np.any(profile[2] == 0.0), "long segment should include a cruise phase"


def test_schedule_boundary_above_and_below_feasibility():
    mp = _mp()
    min_t = _freeflow_duration(100.0, 10.0, 1.0, 1.0)

    exact = Segment(length_m=100.0, v_cruise_ms=10.0, run_time_s=min_t)
    above = Segment(length_m=100.0, v_cruise_ms=10.0, run_time_s=min_t + 2.0)
    below = Segment(length_m=100.0, v_cruise_ms=10.0, run_time_s=min_t - 0.1)

    for seg, expected_delay in [(exact, 0.0), (above, 0.0)]:
        profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
        assert_profile_valid(profile, diag, seg, mp)
        assert diag["schedule_delay_s"] == pytest.approx(expected_delay, abs=0.01)
        assert diag["schedule_infeasible"] is False

    profile, diag = build_speed_profile(below, motion_params=mp, return_diagnostics=True)
    assert_profile_valid(profile, diag, below, mp)
    assert diag["schedule_infeasible"] is True
    assert diag["schedule_delay_s"] > 0.0


def test_target_cruise_is_not_used_as_physical_cap_for_schedule_fit():
    mp = _mp(default_speed_cap_ms=20.0)
    seg = Segment(length_m=100.0, v_cruise_ms=3.0, run_time_s=25.0)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert np.max(profile[1]) > seg.v_cruise_ms
    assert diag["schedule_infeasible"] is False


def test_physical_cap_below_needed_speed_marks_schedule_infeasible():
    mp = _mp(default_speed_cap_ms=5.0)
    seg = Segment(length_m=100.0, v_cruise_ms=20.0, run_time_s=20.0)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert np.max(profile[1]) <= 5.0 + 1e-9
    assert diag["schedule_infeasible"] is True


@pytest.mark.parametrize(
    ("n_signals", "stop_prob", "expected_stops"),
    [(0, 1.0, 0), (1, 0.0, 0), (1, 1.0, 1), (3, 1.0, 3)],
)
def test_signal_stop_counts_in_profile_diagnostics(n_signals, stop_prob, expected_stops):
    mp = _mp(stop_prob=stop_prob, red_wait_s=5.0)
    seg = Segment(
        length_m=150.0,
        v_cruise_ms=10.0,
        run_time_s=60.0,
        n_signals=n_signals,
        from_stop_departure_time="25:00:00",
    )

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert diag["n_effective_signal_stops"] == expected_stops
    assert diag["signal_hour"] == 1


def test_preserve_schedule_retains_full_wait_when_it_fits():
    run_time = 80.0
    red_wait = 10.0
    mp = _mp(red_wait_s=red_wait, stop_prob=1.0)
    seg = Segment(length_m=100.0, v_cruise_ms=10.0, run_time_s=run_time, n_signals=1)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert diag["scheduled_time_s"] == pytest.approx(run_time)
    assert diag["signal_wait_requested_s"] == pytest.approx(red_wait)
    assert diag["signal_wait_applied_s"] == pytest.approx(red_wait)
    assert diag["signal_wait_s"] == pytest.approx(red_wait)
    assert diag["signal_wait_reduced_s"] == pytest.approx(0.0)
    assert diag["moving_time_budget_s"] == pytest.approx(run_time - red_wait)
    assert diag["actual_moving_time_s"] == pytest.approx(run_time - red_wait)
    assert diag["actual_segment_time_s"] == pytest.approx(run_time)
    assert diag["actual_profile_time_s"] == pytest.approx(run_time)
    assert diag["schedule_delay_s"] == pytest.approx(0.0)
    assert diag["schedule_infeasible"] is False


def test_preserve_schedule_exact_moving_budget_boundary():
    min_motion = _freeflow_duration(50.0, 10.0, 1.0, 1.0) * 2.0
    red_wait = 20.0
    run_time = min_motion + red_wait
    mp = _mp(red_wait_s=red_wait, stop_prob=1.0)
    seg = Segment(length_m=100.0, v_cruise_ms=10.0, run_time_s=run_time, n_signals=1)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert diag["minimum_feasible_moving_time_s"] == pytest.approx(min_motion)
    assert diag["min_feasible_motion_time_s"] == pytest.approx(min_motion)
    assert diag["moving_time_budget_s"] == pytest.approx(min_motion)
    assert diag["signal_wait_applied_s"] == pytest.approx(red_wait)
    assert diag["actual_moving_time_s"] == pytest.approx(min_motion, abs=0.01)
    assert diag["actual_segment_time_s"] == pytest.approx(run_time, abs=0.01)
    assert diag["schedule_delay_s"] == pytest.approx(0.0, abs=0.01)
    assert diag["schedule_infeasible"] is False


def test_preserve_schedule_retains_full_wait_and_reports_delay_when_budget_is_too_tight():
    mp = _mp(red_wait_s=20.0, stop_prob=1.0)
    seg = Segment(length_m=100.0, v_cruise_ms=10.0, run_time_s=40.0, n_signals=1)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert diag["signal_wait_requested_s"] == pytest.approx(20.0)
    assert diag["signal_wait_applied_s"] == pytest.approx(20.0)
    assert diag["signal_wait_s"] == pytest.approx(20.0)
    assert diag["signal_wait_reduced_s"] == pytest.approx(0.0)
    assert diag["moving_time_budget_s"] == pytest.approx(20.0)
    assert diag["actual_moving_time_s"] == pytest.approx(
        diag["minimum_feasible_moving_time_s"], abs=0.01
    )
    assert diag["actual_segment_time_s"] > seg.run_time_s
    assert diag["schedule_delay_s"] == pytest.approx(
        diag["actual_segment_time_s"] - seg.run_time_s
    )
    assert diag["schedule_infeasible"] is True


def test_preserve_schedule_retains_wait_when_wait_exceeds_schedule():
    mp = _mp(red_wait_s=45.0, stop_prob=1.0)
    seg = Segment(length_m=100.0, v_cruise_ms=10.0, run_time_s=40.0, n_signals=1)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert diag["signal_wait_applied_s"] == pytest.approx(45.0)
    assert diag["moving_time_budget_s"] == pytest.approx(-5.0)
    assert diag["actual_segment_time_s"] == pytest.approx(
        diag["actual_moving_time_s"] + 45.0
    )
    assert diag["schedule_delay_s"] > 0.0
    assert diag["schedule_infeasible"] is True


def test_add_delay_policy_missing_runtime_after_midnight_and_segment_cap_override():
    mp = _mp(signal_time_policy="add_delay", stop_prob=1.0, red_wait_s=7.0)
    seg = Segment(
        length_m=100.0,
        v_cruise_ms=20.0,
        run_time_s=None,
        n_signals=1,
        from_stop_departure_time="25:15:00",
        speed_cap_ms=6.0,
        speed_cap_source="test",
    )

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert diag["signal_time_policy"] == "add_delay"
    assert diag["signal_wait_s"] == pytest.approx(7.0)
    assert diag["signal_wait_applied_s"] == pytest.approx(7.0)
    assert diag["signal_hour"] == 1
    assert diag["speed_cap_ms"] == 6.0
    assert diag["speed_cap_source"] == "test"


def test_invalid_per_segment_speed_cap_falls_back_to_config_default():
    mp = _mp(default_speed_cap_ms=9.0)
    seg = Segment(length_m=100.0, v_cruise_ms=20.0, run_time_s=None, speed_cap_ms=-1.0)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert diag["speed_cap_ms"] == 9.0
    assert diag["speed_cap_source"] == "config_default"


def test_spatial_signal_events_use_actual_realised_stop_positions(monkeypatch):
    events = [_event("s1", 100.0), _event("s2", 170.0),
              _event("s3", 610.0), _event("s4", 700.0)]
    seg = _spatial_segment(events)
    mp = _mp(stop_prob=0.5, red_wait_s=0.0)
    randoms = {"s1": 0.1, "s2": 0.2, "s3": 0.9, "s4": 0.3}
    monkeypatch.setattr(
        model,
        "_signal_encounter_random_value",
        lambda _seg, event_id, _seed: randoms[event_id],
    )

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert diag["signal_event_mode"] == "spatial"
    assert diag["n_effective_signal_stops"] == 3
    assert diag["realised_stop_positions_m"] == [100.0, 170.0, 700.0]
    assert diag["motion_sub_lengths_m"] == pytest.approx([100.0, 70.0, 530.0, 100.0])
    assert [d["stop_decision"] for d in diag["signal_event_diagnostics"]] == [
        True, True, False, True
    ]
    assert_profile_valid(profile, diag, seg, mp)


def test_pass_spatial_events_do_not_split_motion_profile():
    events = [_event("s1", 100.0), _event("s2", 170.0)]
    seg = _spatial_segment(events)
    mp = _mp(stop_prob=0.0)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert diag["n_effective_signal_stops"] == 0
    assert diag["n_motion_sublinks"] == 1
    assert diag["realised_stop_positions_m"] == []
    assert diag["motion_sub_lengths_m"] == pytest.approx([800.0])
    assert all(not d["stop_decision"] for d in diag["signal_event_diagnostics"])
    assert_profile_valid(profile, diag, seg, mp)


def test_stop_probability_one_stops_at_every_spatial_event():
    events = [_event("s1", 100.0), _event("s2", 170.0),
              _event("s3", 610.0), _event("s4", 700.0)]
    seg = _spatial_segment(events)
    mp = _mp(stop_prob=1.0, red_wait_s=0.0)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert diag["n_effective_signal_stops"] == 4
    assert diag["realised_stop_positions_m"] == [100.0, 170.0, 610.0, 700.0]
    assert diag["motion_sub_lengths_m"] == pytest.approx([100.0, 70.0, 440.0, 90.0, 100.0])
    assert_profile_valid(profile, diag, seg, mp)


def test_event_encounter_seed_and_trip_context_are_reproducible_and_distinct():
    event = _event("same_signal", 100.0)
    seg_a = _spatial_segment([event], trip_id="trip_a")
    seg_b = _spatial_segment([event], trip_id="trip_b")
    mp_a = _mp(stop_prob=0.5, signal_random_seed=11)
    mp_a_again = _mp(stop_prob=0.5, signal_random_seed=11)
    mp_seed_b = _mp(stop_prob=0.5, signal_random_seed=12)

    _, diag_a = build_speed_profile(seg_a, motion_params=mp_a, return_diagnostics=True)
    _, diag_a_again = build_speed_profile(
        seg_a, motion_params=mp_a_again, return_diagnostics=True
    )
    _, diag_seed_b = build_speed_profile(
        seg_a, motion_params=mp_seed_b, return_diagnostics=True
    )
    _, diag_trip_b = build_speed_profile(seg_b, motion_params=mp_a, return_diagnostics=True)

    r_a = diag_a["signal_event_diagnostics"][0]["random_value"]
    assert diag_a_again["signal_event_diagnostics"][0]["random_value"] == pytest.approx(r_a)
    assert diag_seed_b["signal_event_diagnostics"][0]["random_value"] != pytest.approx(r_a)
    assert diag_trip_b["signal_event_diagnostics"][0]["random_value"] != pytest.approx(r_a)


def test_hourly_probability_still_applies_to_spatial_events():
    seg = _spatial_segment([_event("s1", 100.0)])
    mp = _mp(stop_prob=0.0, red_wait_s=0.0,
             use_hourly_signal_stop_probability=True, stop_prob_by_hour={8: 1.0})

    _profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert diag["signal_stop_prob_source"] == "hourly"
    assert diag["signal_stop_probability"] == pytest.approx(1.0)
    assert diag["n_effective_signal_stops"] == 1


def test_closely_spaced_spatial_events_keep_valid_short_leg_physics():
    events = [_event("s1", 100.0), _event("s2", 105.0)]
    seg = _spatial_segment(events)
    mp = _mp(stop_prob=1.0, red_wait_s=0.0, default_speed_cap_ms=15.0)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert diag["motion_sub_lengths_m"] == pytest.approx([100.0, 5.0, 695.0])
    assert_profile_valid(profile, diag, seg, mp)


def test_fallback_count_only_segments_keep_legacy_equal_spacing():
    seg = Segment(length_m=800.0, v_cruise_ms=12.0, run_time_s=160.0, n_signals=3)
    mp = _mp(stop_prob=1.0, red_wait_s=0.0)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert diag["signal_event_mode"] == "legacy_count"
    assert diag["n_motion_sublinks"] == 4
    assert diag["motion_sub_lengths_m"] == pytest.approx([200.0, 200.0, 200.0, 200.0])
    assert_profile_valid(profile, diag, seg, mp)
