import pandas as pd
import pytest

from gtfs_to_segment import process_routes


SIGNAL_ROWS = [
    {
        "from_stop_id": "sdup",
        "to_stop_id": "s2",
        "n_signals": "1",
        "source": "osm",
        "snap_radius_m": "30",
        "relaxed_snap_radius_m": "60",
        "cluster_radius_m": "35",
        "length_m": "500",
        "fetched_utc": "2026-01-01T00:00:00Z",
    },
    {
        "from_stop_id": "s2",
        "to_stop_id": "s3",
        "n_signals": "0",
        "source": "osm",
        "snap_radius_m": "30",
        "relaxed_snap_radius_m": "60",
        "cluster_radius_m": "35",
        "length_m": "500",
        "fetched_utc": "2026-01-01T00:00:00Z",
    },
]


SPEED_ROWS = [
    {
        "from_stop_id": "sdup",
        "to_stop_id": "s2",
        "speed_cap_ms": "12",
        "speed_cap_kmh": "43.2",
        "source": "osm",
        "coverage_frac": "1",
        "snap_radius_m": "25",
        "sample_step_m": "20",
        "length_m": "500",
        "fetched_utc": "2026-01-01T00:00:00Z",
    },
    {
        "from_stop_id": "s2",
        "to_stop_id": "s3",
        "speed_cap_ms": "8",
        "speed_cap_kmh": "28.8",
        "source": "osm",
        "coverage_frac": "1",
        "snap_radius_m": "25",
        "sample_step_m": "20",
        "length_m": "500",
        "fetched_utc": "2026-01-01T00:00:00Z",
    },
]


def test_complete_trip_pipeline_writes_auditable_result_csv(
    tmp_path,
    write_csv,
    integration_gtfs_zip,
    integration_tables,
    full_elevation,
    demand_profile,
    weather_series,
    service_date,
    integration_vehicle,
    integration_motion,
):
    signals_csv = write_csv("signals.csv", SIGNAL_ROWS)
    speed_csv = write_csv("speed_caps.csv", SPEED_ROWS)

    saved = process_routes(
        integration_gtfs_zip,
        ["208"],
        tmp_path / "out",
        tables=integration_tables,
        direction_id=0,
        service_date=service_date,
        demand_profile=demand_profile,
        crush_capacity=80,
        loading_kwargs={"shape": "flat", "hour_mode": "start"},
        weather_series=weather_series,
        weather_kwargs={"hour_mode": "start"},
        signals_enabled=True,
        signals_cache_path=signals_csv,
        speed_caps_enabled=True,
        speed_caps_cache_path=speed_csv,
        vehicle_params=integration_vehicle,
        elevation_data=full_elevation,
        motion_params=integration_motion,
        simulation_level="trip",
    )

    assert len(saved) == 1
    df = pd.read_csv(saved[0])

    required_cols = {
        "route_short_name",
        "route_id",
        "trip_id",
        "segment",
        "length_m",
        "run_time_s",
        "dwell_s",
        "passengers",
        "n_signals",
        "signal_source",
        "n_effective_signal_stops",
        "signal_wait_s",
        "speed_cap_ms",
        "speed_cap_source",
        "signal_stop_prob",
        "actual_profile_time_s",
        "schedule_delay_s",
        "net_battery_energy_kWh",
        "gross_consumed_kWh",
        "regen_recovered_kWh",
        "aux_energy_kWh",
        "SoC_start_%",
        "SoC_end_%",
        "trip_end_soc_%",
    }
    critical = [
        "segment",
        "length_m",
        "net_battery_energy_kWh",
        "gross_consumed_kWh",
        "aux_energy_kWh",
        "SoC_start_%",
        "SoC_end_%",
    ]

    assert len(df) == 2
    assert not any(col.startswith("Unnamed") for col in df.columns)
    assert required_cols.issubset(df.columns)
    assert df[critical].notna().all().all()
    assert list(df["segment"]) == [0, 1]
    assert list(df["trip_id"].unique()) == ["t1"]
    assert list(df["speed_cap_source"]) == ["osm", "osm"]
    assert list(df["signal_source"]) == ["osm", "osm"]
    assert list(df["passengers"]) == [80, 80]

    identity = df["gross_consumed_kWh"] + df["aux_energy_kWh"] - df["regen_recovered_kWh"]
    assert df["net_battery_energy_kWh"].to_numpy() == pytest.approx(identity.to_numpy())
    assert df.loc[1, "SoC_start_%"] == pytest.approx(df.loc[0, "SoC_end_%"])

    total_energy = df["net_battery_energy_kWh"].sum()
    expected_final_soc = 100.0 - total_energy / integration_vehicle.battery_usable_kWh * 100.0
    assert df["trip_end_soc_%"].iloc[-1] == pytest.approx(df["SoC_end_%"].iloc[-1])
    assert df["SoC_end_%"].iloc[-1] == pytest.approx(expected_final_soc, abs=0.02)

    reread = pd.read_csv(saved[0])
    assert len(reread) == len(df)
    assert pd.api.types.is_numeric_dtype(reread["net_battery_energy_kWh"])
