from dataclasses import replace

import pandas as pd
import pytest

from beb_soc_model import MotionParams
from gtfs_to_segment import process_routes


def test_configured_b1_runs_through_gtfs_pipeline_with_positive_boundary_speed(
    tmp_path,minimal_gtfs_zip,full_elevation,integration_vehicle,service_date,
):
    gtfs=minimal_gtfs_zip({"stop_times.txt":[
        {"trip_id":"t1","arrival_time":"08:00:00","departure_time":"08:00:00","stop_id":"s1","stop_sequence":"1"},
        {"trip_id":"t1","arrival_time":"08:01:00","departure_time":"08:01:00","stop_id":"s2","stop_sequence":"2"},
        {"trip_id":"t1","arrival_time":"08:02:00","departure_time":"08:02:00","stop_id":"s3","stop_sequence":"3"}]})
    config=tmp_path/"b1.yaml"
    config.write_text("motion:\n  solver_mode: trip_continuous\n  gtfs_stop_policy: conditional_structural\n  q_stop: 0.0\n",encoding="utf-8")
    mp=MotionParams.from_config(config)
    outputs=[]
    for policy in ("mandatory","conditional_structural"):
        paths=process_routes(gtfs,["208"],tmp_path/policy,service_date=service_date,
                             vehicle_params=integration_vehicle,elevation_data=full_elevation,
                             signals_enabled=False,speed_caps_enabled=False,
                             motion_params=replace(mp,gtfs_stop_policy=policy),simulation_level="trip")
        outputs.append(pd.read_csv(paths[0]))
    b0,b1=outputs
    assert b1.loc[0,"terminal_speed_ms"]==pytest.approx(b1.loc[1,"initial_speed_ms"])
    assert b1.loc[0,"terminal_speed_ms"]>0
    assert not b1.loc[0,"gtfs_to_stop_selected"]
    assert b1.actual_segment_time_s.tolist()==pytest.approx(b0.actual_segment_time_s.tolist())
    assert b1.net_battery_energy_kWh.to_numpy()==pytest.approx(
        (b1.gross_consumed_kWh+b1.aux_energy_kWh-b1.regen_recovered_kWh).to_numpy(),abs=.002)
