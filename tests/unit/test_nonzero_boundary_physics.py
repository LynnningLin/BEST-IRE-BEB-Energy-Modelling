import numpy as np
import pytest

from continuous_interval import solve_interval
from trip_motion import solve_motion_leg


CASES = [("C0", 0, 0, 60), ("C1", 0, 8, 60), ("C2", 8, 0, 60),
         ("C3", 8, 8, 35), ("C4", 8, 8, 60), ("C5", 8, 8, None),
         ("C6", 8, 8, 10)]


@pytest.mark.parametrize("name,v0,v1,budget", CASES)
def test_c0_c6_physics(name, v0, v1, budget):
    minimum = solve_interval(300, v0, v1, 13.9).minimum_time_s
    if name == "C5":
        budget = minimum
    result = solve_interval(300, v0, v1, 13.9, scheduled_time_s=budget)
    assert result.distance_integral_m == pytest.approx(300, abs=1e-8)
    assert result.duration_s == pytest.approx(max(budget, minimum), abs=1e-8)
    assert result.schedule_delay_s == pytest.approx(max(minimum-budget, 0), abs=1e-8)
    assert result.position_speed(0)[1] == pytest.approx(v0)
    assert result.position_speed(result.duration_s)[1] == pytest.approx(v1, abs=1e-9)
    assert result.position_speed(result.duration_s)[0] == pytest.approx(300, abs=1e-8)
    for left, right in zip(result.phases, result.phases[1:]):
        assert left.initial_speed_ms + left.acceleration_ms2*left.duration_s == pytest.approx(right.initial_speed_ms)
    for phase in result.phases:
        assert -1.2 <= phase.acceleration_ms2 <= 1
        assert phase.duration_s > 0
    _, speeds = result.position_speed(np.linspace(0, result.duration_s, 5001))
    assert speeds.max() <= 13.9 + 1e-9
    assert speeds.min() >= 0
    if v0 > 0 and v1 > 0:
        assert speeds.min() > 0
    if name == "C4":
        assert result.shape == "valley"
        assert result.minimum_speed_ms < 8
    if name == "C0":
        old = solve_motion_leg(0, 300, 0, 0, 13.9, 13.9, 1, 1.2, 60)
        for actual, expected in zip(result.sample(0.5), old.sample(0.5)):
            np.testing.assert_array_equal(actual, expected)


def test_monotone_solution_and_finite_maximum_nonstop_time():
    monotone = solve_interval(300, 8, 2, 13.9, scheduled_time_s=60)
    assert monotone.shape == "monotone"
    assert monotone.distance_integral_m == pytest.approx(300)
    impossible = solve_interval(5, 8, 8, 13.9, scheduled_time_s=60)
    assert impossible.timing_status == "too_long_for_nonstop_boundaries"
    assert impossible.duration_s == pytest.approx(impossible.minimum_time_s)
    assert impossible.minimum_speed_ms > 0
    with pytest.raises(ValueError, match="Insufficient distance"):
        solve_interval(1, 0, 8, 13.9)


def test_inverse_distance_and_kinetic_integral():
    result = solve_interval(300, 8, 5, 13.9, scheduled_time_s=90)
    for distance in np.linspace(0, 300, 21):
        t = result.time_at_distance(distance)
        assert result.position_speed(t)[0] == pytest.approx(distance, abs=1e-8)
    kinetic_per_mass = sum(p.acceleration_ms2*p.distance_m for p in result.phases)
    assert kinetic_per_mass == pytest.approx((5**2-8**2)/2, abs=1e-8)


def test_pure_braking_distance_roundoff_keeps_minimum_time_feasible():
    length=14.097841123198927
    initial=5.816770469571382
    minimum=solve_interval(length,initial,0,13.9)
    result=solve_interval(length,initial,0,13.9,scheduled_time_s=minimum.minimum_time_s)
    assert result.timing_status=="feasible"
    assert result.duration_s==pytest.approx(minimum.minimum_time_s,abs=1e-10)
    assert result.distance_integral_m==pytest.approx(length,abs=1e-8)


def test_zero_speed_valley_is_not_used_to_consume_a_nonstop_budget():
    length=8**2/(2*1.2)+8**2/2
    duration=8/1.2+8
    result=solve_interval(length,8,8,13.9,scheduled_time_s=duration)
    assert result.timing_status=="too_long_for_nonstop_boundaries"
    assert result.minimum_speed_ms>0


def test_deterministic_sweep_preserves_feasible_motion():
    rng = np.random.default_rng(707)
    for _ in range(300):
        cap = rng.uniform(8, 20)
        v0, v1 = rng.uniform(0.1, cap, 2)
        length = rng.uniform(100, 1000)
        shortest = solve_interval(length, v0, v1, cap)
        largest = min(shortest.minimum_time_s*5, shortest.maximum_nonstop_time_s)
        budget = rng.uniform(shortest.minimum_time_s, largest)
        result = solve_interval(length, v0, v1, cap, scheduled_time_s=budget)
        assert result.distance_integral_m == pytest.approx(length, abs=1e-7)
        assert result.duration_s == pytest.approx(budget, abs=1e-7)
        assert result.minimum_speed_ms > 0
