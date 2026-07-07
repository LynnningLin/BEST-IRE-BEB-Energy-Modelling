from best_ire_beb.config import (
    PROJECT_ROOT,
    get_path,
    get_section,
    load_config,
    vehicle_params,
)


def test_default_model_config_is_loaded() -> None:
    assert get_path("gtfs_zip") == PROJECT_ROOT / "data" / "raw" / "GTFS_Realtime.zip"
    signals_csv = PROJECT_ROOT / "data" / "processed" / "traffic_signals.csv"
    assert get_path("traffic_signals_csv") == signals_csv
    weather_csv = PROJECT_ROOT / "data" / "processed"
    weather_csv = weather_csv / "cork_weather_hourly_model_input_2021_2025_with_solar.csv"
    assert get_path("weather_csv") == weather_csv
    assert get_section("passenger_loading")["demand_city"] == "Cork"
    signals = get_section("traffic_signals")
    assert signals["enabled"] is True
    assert signals["snap_radius_m"] == 30.0
    climate = get_section("weather")["climate_control"]
    assert climate["heat_below_c"] == 18.0
    assert climate["heating_months"] == [11, 12, 1, 2, 3]
    assert vehicle_params()["battery_usable_kWh"] == 410.0


def test_custom_model_config_overrides_defaults() -> None:
    config_path = PROJECT_ROOT / ".test_model_config.yaml"
    config_path.unlink(missing_ok=True)
    load_config.cache_clear()
    try:
        config_path.write_text(
            "\n".join(
                [
                    "paths:",
                    "  gtfs_zip: custom/feed.zip",
                    "  weather_csv: custom/weather.csv",
                    "vehicle:",
                    "  battery_usable_kWh: 500",
                    "weather:",
                    "  climate_control:",
                    "    cool_above_c: 24",
                    "traffic_signals:",
                    "  enabled: false",
                    "  snap_radius_m: 45",
                ]
            ),
            encoding="utf-8",
        )

        assert get_path("gtfs_zip", config_path) == PROJECT_ROOT / "custom" / "feed.zip"
        custom_weather_csv = PROJECT_ROOT / "custom" / "weather.csv"
        assert get_path("weather_csv", config_path) == custom_weather_csv
        climate = get_section("weather", config_path)["climate_control"]
        assert climate["cool_above_c"] == 24
        assert climate["heat_below_c"] == 10.0
        assert climate["heating_months"] == [11, 12, 1, 2, 3]
        signals = get_section("traffic_signals", config_path)
        assert signals["enabled"] is False
        assert signals["snap_radius_m"] == 45
        assert signals["fallback_per_km"] == 2.0
        params = vehicle_params(config_path)
        assert params["battery_usable_kWh"] == 500.0
        assert params["curb_mass_kg"] == 14_000.0
    finally:
        config_path.unlink(missing_ok=True)
        load_config.cache_clear()
