from datetime import datetime

import pytest

from weather_loading import load_weather_csv


def test_weather_loader_detects_timestamp_and_normalises_columns(write_csv):
    path = write_csv(
        "weather.csv",
        [
            {
                "time": "2025-07-09 09:00:00",
                "temp": "17.5",
                "rhum": "85",
                "solar": "-12",
            }
        ],
    )

    series = load_weather_csv(path, verbose=False)
    row = series.df.iloc[0]

    assert row["temp"] == 17.5, "temperature should load as numeric Celsius"
    assert row["rh"] == 0.85, "percentage humidity should convert to a fraction"
    assert row["solar"] == 0.0, "negative solar radiation should be clipped to zero"


def test_weather_loader_assembles_timestamp_from_date_parts(write_csv):
    path = write_csv(
        "weather.csv",
        [{"year": "2025", "month": "7", "day": "9", "hour": "8", "temp": "16"}],
    )

    series = load_weather_csv(path, verbose=False)

    assert series.df.index[0] == datetime(2025, 7, 9, 8)


@pytest.mark.parametrize(
    "rows",
    [
        [{"time": "2025-07-09 09:00:00", "rhum": "85"}],
        [{"time": "2025-07-09 09:00:00", "temp": "not-numeric"}],
    ],
)
def test_missing_required_temperature_data_cause_clear_error(write_csv, rows):
    path = write_csv("weather.csv", rows)

    with pytest.raises(ValueError, match="temperature|valid timestamp/temperature"):
        load_weather_csv(path, verbose=False)


@pytest.mark.parametrize("humidity", ["-1", "101"])
def test_relative_humidity_must_be_valid_after_conversion(write_csv, humidity):
    path = write_csv(
        "weather.csv",
        [{"time": "2025-07-09 09:00:00", "temp": "17", "rhum": humidity}],
    )

    with pytest.raises(ValueError, match="relative humidity"):
        load_weather_csv(path, verbose=False)


def test_duplicate_weather_timestamps_are_rejected(write_csv):
    path = write_csv(
        "weather.csv",
        [
            {"time": "2025-07-09 09:00:00", "temp": "17"},
            {"time": "2025-07-09 09:00:00", "temp": "18"},
        ],
    )

    with pytest.raises(ValueError, match="timestamps must be unique"):
        load_weather_csv(path, verbose=False)


def test_weather_rows_are_sorted_chronologically(write_csv):
    path = write_csv(
        "weather.csv",
        [
            {"time": "2025-07-09 10:00:00", "temp": "20"},
            {"time": "2025-07-09 08:00:00", "temp": "16"},
        ],
    )

    series = load_weather_csv(path, verbose=False)

    assert list(series.df["temp"]) == [16, 20], "weather rows should sort by timestamp"


def test_nearest_timestamp_lookup_returns_expected_row(write_csv):
    path = write_csv(
        "weather.csv",
        [
            {"time": "2025-07-09 08:00:00", "temp": "16", "rhum": "80"},
            {"time": "2025-07-09 10:00:00", "temp": "20", "rhum": "60"},
        ],
    )
    series = load_weather_csv(path, verbose=False)

    weather = series.at(datetime(2025, 7, 9, 9, 40))

    assert weather.air_temp_c == 20.0, "nearest timestamp lookup chose wrong row"
    assert weather.relative_humidity == 0.60
