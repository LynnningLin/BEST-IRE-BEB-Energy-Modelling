from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from beb_soc_model import MotionParams, Segment, VehicleParams, build_speed_profile, simulate_route
from conditional_gtfs_motion import select_gtfs_stops
from trip_motion import solve_trip_motion


def trip():
    return [Segment(300,run_time_s=60,dwell_s=0,trip_id="trip",service_date="2025-07-09",
                    segment_index=i,from_stop_departure_time=f"08:0{i}:00",
                    to_stop_arrival_time=f"08:0{i+1}:00",to_stop_departure_time=f"08:0{i+1}:00")
            for i in range(3)]


def params(**kwargs):
    return MotionParams(solver_mode="trip_continuous",gtfs_stop_policy="conditional_structural",
                        q_stop=kwargs.pop("q_stop",0.0),**kwargs)


def test_eligibility_is_conservative_and_does_not_infer_physical_stops_from_service_restrictions():
    segments=trip()
    segments[0].pickup_type=1
    decisions=select_gtfs_stops(segments,0,7)
    assert [d["stop_decision"] for d in decisions]==[True,False,False,True]
    segments[0].dwell_s=20
    assert select_gtfs_stops(segments,0,7)[1]["reason"]=="positive_scheduled_dwell"
    segments[1].to_stop_compulsory=True
    assert select_gtfs_stops(segments,0,7)[2]["reason"]=="explicit_terminal_or_hold"


def test_q1_and_mandatory_retain_b0_profiles_outputs_and_signals_exactly():
    segments=trip()
    segments[1].n_signals=1
    segments[1].signal_events=[{"signal_event_id":"x","distance_along_segment_m":150}]
    base=MotionParams(solver_mode="trip_continuous")
    old=solve_trip_motion(deepcopy(segments),base)
    new=solve_trip_motion(deepcopy(segments),params(q_stop=1))
    assert old.segment_diagnostics==new.segment_diagnostics
    for expected,actual in zip(old.profile,new.profile):
        np.testing.assert_array_equal(expected,actual)
    pd.testing.assert_frame_equal(simulate_route(deepcopy(segments),VehicleParams(),motion_params=base),
                                  simulate_route(deepcopy(segments),VehicleParams(),motion_params=params(q_stop=1)))


def test_pass_speeds_are_shared_and_local_schedules_are_preserved():
    result=solve_trip_motion(trip(),params())
    gtfs=[e for e in result.events if e.kind=="gtfs_stop"]
    assert [e.actual_speed_ms for e in gtfs]==pytest.approx([0,5,5,0])
    assert [e.arrival_time_s for e in gtfs]==pytest.approx([0,60,120,180])
    for left,right in zip(result.intervals,result.intervals[1:]):
        assert left.end is right.start
        assert left.diagnostics["terminal_speed_ms"]==right.diagnostics["initial_speed_ms"]>0
    assert not any(s.kind=="gtfs_dwell" for s in result.spans)
    assert np.all(np.diff(result.profile[0])>0)
    assert sum(result.profile[3])==pytest.approx(180)
    assert sum(d["kinetic_integral_per_mass"] for d in result.segment_diagnostics)==pytest.approx(0,abs=1e-8)
    for diag in result.segment_diagnostics:
        assert diag["analytic_distance_m"]==pytest.approx(300)
        assert diag["schedule_delay_s"]==pytest.approx(0)


def test_independent_nested_gtfs_random_draws_and_paired_signal_context():
    segments=trip()*8
    for i,s in enumerate(segments):
        s=deepcopy(s)
        s.segment_index=i
        s.from_stop_departure_time=None
        s.to_stop_departure_time=None
        segments[i]=s
    f2=deepcopy(segments)
    for s in f2:
        s.segment_geometry_id="only-populated-when-signals-on"
        s.n_signals=1
        s.signal_events=[{"signal_event_id":"light","distance_along_segment_m":150}]
    for q in (0,.25,.5,.75,1):
        assert select_gtfs_stops(segments,q,7)==select_gtfs_stops(f2,q,7)
    draws=select_gtfs_stops(f2,.5,7)
    assert draws==select_gtfs_stops(f2,.5,7)
    assert [d["random_value"] for d in draws]!=[d["random_value"] for d in select_gtfs_stops(f2,.5,8)]
    lower=select_gtfs_stops(f2,.25,7)
    higher=select_gtfs_stops(f2,.75,7)
    assert all(not a["stop_decision"] or b["stop_decision"] for a,b in zip(lower,higher))
    for seed in (7,8):
        mp=params(gtfs_stop_random_seed=seed)
        result=solve_trip_motion(deepcopy(f2),mp)
        for seg,diag in zip(f2,result.segment_diagnostics):
            _,reference=build_speed_profile(seg,motion_params=mp,return_diagnostics=True)
            assert diag["signal_event_diagnostics"]==reference["signal_event_diagnostics"]


@pytest.mark.parametrize("signal_distance",[0.0,0.01,50.0,299.9])
def test_signal_stops_survive_passed_gtfs_stop_and_remain_spatially_unchanged(signal_distance):
    segments=trip()
    segments[1].n_signals=1
    segments[1].signal_events=[{"signal_event_id":"x","distance_along_segment_m":signal_distance}]
    mp=params(stop_prob=1,use_hourly_signal_stop_probability=False)
    result=solve_trip_motion(segments,mp)
    signal=next(e for e in result.events if e.kind=="signal")
    assert signal.distance_m==300+signal_distance
    assert signal.actual_speed_ms==0
    assert signal.departure_time_s-signal.arrival_time_s==pytest.approx(15)
    assert not result.gtfs_stop_decisions[1]["stop_decision"]
    if signal_distance==0:
        assert result.gtfs_stop_decisions[1]["signal_coincident_zero"]
    else:
        assert result.gtfs_stop_decisions[1]["boundary_speed_ms"]>0
    for d in result.segment_diagnostics:
        assert d["actual_segment_time_s"]>=d["scheduled_time_s"]-1e-8


def test_dwell_auxiliaries_and_delay_are_not_transferred_between_intervals():
    segments=trip()
    segments[0].run_time_s=5
    segments[1].dwell_s=25
    segments[2].run_time_s=150
    p=VehicleParams()
    result=solve_trip_motion(segments,params())
    assert result.segment_diagnostics[0]["schedule_delay_s"]>0
    assert result.segment_diagnostics[2]["actual_segment_time_s"]==pytest.approx(150)
    frame=simulate_route(segments,p,motion_params=params())
    expected_time=sum(d["actual_segment_time_s"] for d in result.segment_diagnostics)+25
    assert result.duration_s==pytest.approx(expected_time)
    assert frame.aux_energy_kWh.sum()==pytest.approx(p.aux_power_kW*expected_time/3600,abs=.002)


def test_invalid_configuration_does_not_silently_enable_b1():
    assert MotionParams().gtfs_stop_policy=="mandatory"
    assert MotionParams().q_stop==1
    with pytest.raises(ValueError,match="requires trip_continuous"):
        MotionParams(gtfs_stop_policy="conditional_structural")
    with pytest.raises(ValueError,match="q_stop"):
        params(q_stop=-.1)
    with pytest.raises(ValueError,match="seed"):
        params(gtfs_stop_random_seed=1.5)
