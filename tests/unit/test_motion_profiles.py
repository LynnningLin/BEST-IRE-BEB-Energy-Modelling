import numpy as np
import pytest

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


@pytest.mark.parametrize(
    ("run_time", "red_wait", "expected_wait_relation"),
        [
            (80.0, 10.0, "all"),
            (47.0, 20.0, "partial"),
            (_freeflow_duration(50.0, 10.0, 1.0, 1.0) * 2.0, 20.0, "none"),
        ],
    )
def test_preserve_schedule_wait_fit_cases(run_time, red_wait, expected_wait_relation):
    mp = _mp(red_wait_s=red_wait, stop_prob=1.0)
    seg = Segment(length_m=100.0, v_cruise_ms=10.0, run_time_s=run_time, n_signals=1)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    if expected_wait_relation == "all":
        assert diag["signal_wait_s"] == pytest.approx(diag["signal_wait_requested_s"])
    elif expected_wait_relation == "partial":
        assert 0.0 < diag["signal_wait_s"] < diag["signal_wait_requested_s"]
    else:
        assert diag["signal_wait_s"] == 0.0


def test_signal_motion_infeasible_even_with_zero_wait():
    mp = _mp(red_wait_s=20.0, stop_prob=1.0)
    seg = Segment(length_m=100.0, v_cruise_ms=10.0, run_time_s=20.0, n_signals=1)

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert_profile_valid(profile, diag, seg, mp)
    assert diag["signal_wait_s"] == 0.0
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
