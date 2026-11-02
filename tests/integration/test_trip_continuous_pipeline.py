from dataclasses import replace

import pandas as pd
import pytest

from gtfs_to_segment import process_routes


@pytest.mark.parametrize("level", ["trip", "block"])
def test_continuous_pipeline_preserves_results_and_duty_soc(
    tmp_path, integration_gtfs_zip, integration_tables, full_elevation,
    integration_vehicle, integration_motion, demand_profile, weather_series, service_date, level,
):
    results = []
    for mode in ("segment_legacy", "trip_continuous"):
        paths = process_routes(
            integration_gtfs_zip, ["208"], tmp_path / mode,
            tables=integration_tables, direction_id=None, all_services=True, service_date=service_date,
            vehicle_params=integration_vehicle, elevation_data=full_elevation,
            demand_profile=demand_profile, crush_capacity=80,
            weather_series=weather_series,
            motion_params=replace(integration_motion, solver_mode=mode),
            signals_enabled=False, speed_caps_enabled=False, simulation_level=level,
        )
        results.append(pd.read_csv(paths[0]))
    old, new = results
    pd.testing.assert_frame_equal(new[old.columns], old, check_exact=True)
    assert new.solver_mode.eq("trip_continuous").all()
    assert new.initial_speed_ms.eq(0).all()
    assert new.terminal_speed_ms.eq(0).all()
    for _, trip in new.groupby("trip_id", sort=False):
        assert trip.trip_start_time_s.iloc[0] == 0
        assert trip.trip_arrival_time_s.is_monotonic_increasing
    starts = new.groupby("trip_id", sort=False)["trip_start_soc_%"].first()
    if level == "trip":
        assert starts.eq(100).all()
    else:
        assert starts.iloc[1] < 100
