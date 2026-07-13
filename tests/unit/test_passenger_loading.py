from copy import deepcopy

import pytest

from beb_soc_model import Segment
from passenger_loading import HourlyDemandProfile, apply_passenger_loading


def _segments(n=5):
    return [Segment(length_m=100.0) for _ in range(n)]


def _trip(hour=8):
    return [
        {"departure_time": f"{hour:02d}:00:00", "arrival_time": f"{hour:02d}:00:00"},
        {"departure_time": f"{hour:02d}:30:00", "arrival_time": f"{hour:02d}:30:00"},
    ]


def test_hourly_profile_normalisation_peak_and_zero_hour():
    profile = HourlyDemandProfile.from_percent({0: 0, 7: 20, 8: 40, 9: 40})

    assert sum(profile.hourly_fraction.values()) == pytest.approx(1.0)
    assert profile.peak_hour in {8, 9}
    assert profile.temporal_factor(profile.peak_hour) == 1.0
    assert profile.temporal_factor(0) == 0.0
    assert profile.temporal_factor(23) == 0.0


def test_flat_loading_is_equal_non_negative_and_bounded_by_crush_capacity():
    profile = HourlyDemandProfile.from_percent({8: 1})
    segments = apply_passenger_loading(
        _segments(4), _trip(8), profile, crush_capacity=80, shape="flat", verbose=False
    )

    loads = [s.passengers for s in segments]
    assert loads == [80, 80, 80, 80]
    assert min(loads) >= 0
    assert max(loads) <= 80


def test_trapezoid_loading_rises_plateaus_and_falls():
    profile = HourlyDemandProfile.from_percent({8: 1})
    segments = apply_passenger_loading(
        _segments(6),
        _trip(8),
        profile,
        crush_capacity=60,
        shape="trapezoid",
        board_frac=0.3,
        alight_frac=0.3,
        round_to_int=False,
        verbose=False,
    )
    loads = [s.passengers for s in segments]

    assert loads[0] < loads[1] <= loads[2]
    assert loads[2] == pytest.approx(loads[3])
    assert loads[4] > loads[5]


def test_beta_loading_starts_low_peaks_and_falls():
    profile = HourlyDemandProfile.from_percent({8: 1})
    segments = apply_passenger_loading(
        _segments(8),
        _trip(8),
        profile,
        crush_capacity=80,
        shape="beta",
        round_to_int=False,
        verbose=False,
    )
    loads = [s.passengers for s in segments]

    assert loads[0] < max(loads)
    assert loads[-1] < max(loads)
    assert all(load >= 0 for load in loads)


def test_crush_capacity_and_hourly_demand_scale_monotonically():
    low_profile = HourlyDemandProfile.from_percent({8: 1, 9: 2})
    high_profile = HourlyDemandProfile.from_percent({8: 2, 9: 2})

    base = apply_passenger_loading(
        _segments(3), _trip(8), low_profile, crush_capacity=50, shape="flat", verbose=False
    )
    larger_bus = apply_passenger_loading(
        _segments(3), _trip(8), low_profile, crush_capacity=100, shape="flat", verbose=False
    )
    higher_demand = apply_passenger_loading(
        _segments(3), _trip(8), high_profile, crush_capacity=50, shape="flat", verbose=False
    )

    assert larger_bus[0].passengers == 2 * base[0].passengers
    assert higher_demand[0].passengers >= base[0].passengers


def test_passenger_loading_is_deterministic_and_invalid_shape_raises():
    profile = HourlyDemandProfile.from_percent({8: 1})
    first = apply_passenger_loading(
        _segments(5), _trip(8), profile, shape="beta", verbose=False
    )
    second = apply_passenger_loading(
        deepcopy(_segments(5)), _trip(8), profile, shape="beta", verbose=False
    )

    assert [s.passengers for s in first] == [s.passengers for s in second]
    with pytest.raises(ValueError, match="unknown shape"):
        apply_passenger_loading(_segments(2), _trip(8), profile, shape="banana")
