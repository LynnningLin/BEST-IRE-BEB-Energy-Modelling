import pytest

from gtfs_to_segment import load_demand_profile_csv


def _hourly_rows(value_col="percent", values=None):
    values = values or {hour: hour + 1 for hour in range(24)}
    return [{"hour": hour, value_col: values[hour]} for hour in range(24)]


def test_valid_24_hour_profile_loads_correctly(write_csv):
    path = write_csv("demand.csv", _hourly_rows())

    profile = load_demand_profile_csv(path)

    assert len(profile.hourly_fraction) == 24
    assert sum(profile.hourly_fraction.values()) == pytest.approx(1.0)
    assert profile.peak_hour == 23


def test_profile_with_missing_hours_is_warned_about(write_csv, capsys):
    path = write_csv("demand.csv", _hourly_rows()[:23])

    profile = load_demand_profile_csv(path)

    captured = capsys.readouterr()
    assert "23 hours, not 24" in captured.out
    assert len(profile.hourly_fraction) == 23


def test_profile_with_zero_total_demand_is_rejected(write_csv):
    path = write_csv("demand.csv", _hourly_rows(values={hour: 0 for hour in range(24)}))

    with pytest.raises(ValueError, match="sums to zero"):
        load_demand_profile_csv(path)


def test_negative_demand_values_are_rejected(write_csv):
    values = {hour: 1 for hour in range(24)}
    values[7] = -1
    path = write_csv("demand.csv", _hourly_rows(values=values))

    with pytest.raises(ValueError, match="negative demand"):
        load_demand_profile_csv(path)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("percent", {hour: 100 / 24 for hour in range(24)}),
        ("flow", {hour: 10 for hour in range(24)}),
        ("fraction", {hour: 1 / 24 for hour in range(24)}),
    ],
)
def test_percentage_count_and_fraction_inputs_normalise(write_csv, column, values):
    path = write_csv("demand.csv", _hourly_rows(value_col=column, values=values))

    profile = load_demand_profile_csv(path)

    assert sum(profile.hourly_fraction.values()) == pytest.approx(1.0)
    assert profile.temporal_factor(0) == pytest.approx(1.0)


def test_multi_city_demand_file_requires_city_selection(write_csv):
    path = write_csv(
        "demand.csv",
        [
            {"city": "Cork", "hour": h, "avg_hourly_flow_percent": 1}
            for h in range(24)
        ],
    )

    with pytest.raises(ValueError, match="multi-city; pass a city"):
        load_demand_profile_csv(path)


def test_multi_city_demand_file_requires_valid_city(write_csv):
    path = write_csv(
        "demand.csv",
        [
            {"city": "Cork", "hour": h, "avg_hourly_flow_percent": 1}
            for h in range(24)
        ],
    )

    with pytest.raises(ValueError, match="city 'Dublin' not found"):
        load_demand_profile_csv(path, city="Dublin")


def test_multi_city_demand_file_loads_selected_city(write_csv):
    path = write_csv(
        "demand.csv",
        [
            {"city": "Cork", "hour": h, "avg_hourly_flow_percent": h + 1}
            for h in range(24)
        ]
        + [
            {"city": "Dublin", "hour": h, "avg_hourly_flow_percent": 1}
            for h in range(24)
        ],
    )

    profile = load_demand_profile_csv(path, city="Cork")

    assert profile.peak_hour == 23
    assert sum(profile.hourly_fraction.values()) == pytest.approx(1.0)
