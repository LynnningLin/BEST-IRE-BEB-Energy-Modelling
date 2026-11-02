import pandas as pd
import pytest

import gtfs_to_segment
from gtfs_to_segment import process_routes


def _run_pipeline(tmp_path, integration_gtfs_zip, integration_tables, full_elevation,
                  integration_vehicle, integration_motion, simulation_level,
                  direction_id=0):
    out_dir = tmp_path / simulation_level
    saved = process_routes(
        integration_gtfs_zip,
        ["208"],
        out_dir,
        tables=integration_tables,
        direction_id=direction_id,
        service_date=None,
        all_services=True,
        vehicle_params=integration_vehicle,
        elevation_data=full_elevation,
        motion_params=integration_motion,
        signals_enabled=False,
        speed_caps_enabled=False,
        simulation_level=simulation_level,
    )
    assert len(saved) == 1
    assert saved[0].parent == out_dir / "traffic_signals_off"
    return pd.read_csv(saved[0])


def test_block_pipeline_keeps_chronological_duty_soc_continuity(
    tmp_path,
    integration_gtfs_zip,
    integration_tables,
    full_elevation,
    integration_vehicle,
    integration_motion,
):
    df = _run_pipeline(
        tmp_path,
        integration_gtfs_zip,
        integration_tables,
        full_elevation,
        integration_vehicle,
        integration_motion,
        simulation_level="block",
    )

    trips = df.groupby("trip_id", sort=False).agg(
        start=("trip_start_time", "first"),
        start_soc=("trip_start_soc_%", "first"),
        end_soc=("trip_end_soc_%", "last"),
        direction=("direction_id", "first"),
        duty_index=("duty_trip_index", "first"),
        energy=("net_battery_energy_kWh", "sum"),
    )

    assert list(trips.index) == ["t1", "t2"]
    assert list(trips["start"]) == ["08:00:00", "09:00:00"]
    assert set(trips["direction"]) == {0, 1}
    assert df["duty_id"].nunique() == 1
    assert df["duty_id"].iloc[0] == "svc:block_a"
    assert list(trips["duty_index"]) == [0, 1]
    assert trips.loc["t2", "start_soc"] == pytest.approx(trips.loc["t1", "end_soc"])
    assert df["net_battery_energy_kWh"].sum() == pytest.approx(trips["energy"].sum())
    assert df["SoC_start_%"].iloc[2] == pytest.approx(df["SoC_end_%"].iloc[1])


def test_trip_pipeline_resets_soc_while_block_pipeline_does_not(
    tmp_path,
    integration_gtfs_zip,
    integration_tables,
    full_elevation,
    integration_vehicle,
    integration_motion,
):
    trip_df = _run_pipeline(
        tmp_path,
        integration_gtfs_zip,
        integration_tables,
        full_elevation,
        integration_vehicle,
        integration_motion,
        simulation_level="trip",
        direction_id=None,
    )
    block_df = _run_pipeline(
        tmp_path,
        integration_gtfs_zip,
        integration_tables,
        full_elevation,
        integration_vehicle,
        integration_motion,
        simulation_level="block",
    )

    trip_starts = trip_df.groupby("trip_id", sort=False)["trip_start_soc_%"].first()
    block_starts = block_df.groupby("trip_id", sort=False)["trip_start_soc_%"].first()

    assert list(trip_starts.index) == ["t1", "t2"]
    assert trip_starts.to_list() == [100.0, 100.0]
    assert list(block_starts.index) == ["t1", "t2"]
    assert block_starts.loc["t1"] == 100.0
    assert block_starts.loc["t2"] < 100.0


def test_exact_route_id_keeps_the_real_short_name_in_results(
    tmp_path, integration_gtfs_zip, integration_tables, full_elevation
):
    saved = process_routes(
        integration_gtfs_zip, ["r1"], tmp_path, tables=integration_tables,
        all_services=True, elevation_data=full_elevation, simulation_level="trip",
        direction_id=None,
    )

    results = pd.read_csv(saved[0], dtype={"route_short_name": str})
    assert saved[0].name.startswith("route_r1_")
    assert set(results["route_short_name"]) == {"208"}
    assert set(results["route_id"]) == {"r1"}


@pytest.mark.parametrize("failing", ["signals", "speed_caps"])
def test_matched_runs_refuse_fallback_inputs(
    tmp_path, integration_gtfs_zip, integration_tables, full_elevation, monkeypatch, failing
):
    def offline(*args, **kwargs):
        raise RuntimeError("offline")

    def no_signals(*args, **kwargs):
        return []

    monkeypatch.setattr(
        gtfs_to_segment,
        "fetch_traffic_signals_bbox",
        offline if failing == "signals" else no_signals,
    )
    monkeypatch.setattr(gtfs_to_segment, "fetch_speed_edges_bbox", offline)

    with pytest.raises(ValueError, match="Matched comparison refuses"):
        process_routes(
            integration_gtfs_zip, ["208"], tmp_path, tables=integration_tables,
            all_services=True, elevation_data=full_elevation, simulation_level="trip",
            direction_id=None, run_records=[], signals_enabled=True,
            speed_caps_enabled=True,
        )


def test_matched_runs_refuse_impossible_scheduled_speeds(
    tmp_path, integration_gtfs_zip, integration_tables, full_elevation, monkeypatch
):
    monkeypatch.setattr(gtfs_to_segment, "MAX_PLAUSIBLE_SCHEDULED_KMH", 0.1)

    with pytest.raises(ValueError, match="Matched comparison refuses trip"):
        process_routes(
            integration_gtfs_zip, ["208"], tmp_path, tables=integration_tables,
            all_services=True, elevation_data=full_elevation, simulation_level="trip",
            direction_id=None, run_records=[],
        )


def test_other_runs_only_warn_about_impossible_scheduled_speeds(
    tmp_path, integration_gtfs_zip, integration_tables, full_elevation, monkeypatch, capsys
):
    monkeypatch.setattr(gtfs_to_segment, "MAX_PLAUSIBLE_SCHEDULED_KMH", 0.1)

    process_routes(
        integration_gtfs_zip, ["208"], tmp_path, tables=integration_tables,
        all_services=True, elevation_data=full_elevation, simulation_level="trip",
        direction_id=None,
    )

    assert "check its shape" in capsys.readouterr().out
