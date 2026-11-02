from dataclasses import replace

import pandas as pd

from beb_soc_model import MotionParams
import gtfs_to_segment as gtfs


def test_f1_flows_through_trip_pipeline_and_diagnostic_export(
    tmp_path,minimal_gtfs_zip,full_elevation,integration_vehicle,service_date,monkeypatch,
):
    original = gtfs.segments_for_trip

    def with_inventory(*args,**kwargs):
        segments = original(*args,**kwargs)
        for i,seg in enumerate(segments):
            seg.signal_events = [{"signal_event_id":f"cached-{i}",
                                  "distance_along_segment_m":seg.length_m*.25}]
            seg.n_signals = 1
            seg.signal_source = "osm"
        return segments

    monkeypatch.setattr(gtfs,"segments_for_trip",with_inventory)
    mp = MotionParams(solver_mode="trip_continuous",signal_fidelity="F1_frequency")
    profiles,events = [],[]
    source = minimal_gtfs_zip()
    for fidelity in ("F1_frequency","F2_spatial"):
        directory = tmp_path/fidelity
        paths = gtfs.process_routes(source,["208"],directory,service_date=service_date,
                                    vehicle_params=integration_vehicle,elevation_data=full_elevation,
                                    signals_enabled=False,speed_caps_enabled=False,simulation_level="trip",
                                    signal_event_diagnostics_enabled=True,
                                    motion_params=replace(mp,signal_fidelity=fidelity))
        profiles.append(pd.read_csv(paths[0]))
        events.append(pd.read_csv(next(paths[0].parent.glob("*signal_events*.csv"))))
    f1,f2 = profiles
    assert f1.n_signals.sum()==f2.n_signals.sum()==2
    assert f1.n_signal_stops.sum()==f2.n_signal_stops.sum()
    assert f1.signal_fidelity.eq("F1_frequency").all()
    for key in ("stop_probability","random_value","stop_decision","red_wait_s_requested"):
        assert events[0][key].tolist()==events[1][key].tolist()
    assert events[0].paired_F2_event_id.tolist()==events[1].signal_event_id.tolist()
