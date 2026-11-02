from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from beb_soc_model import MotionParams, Segment, VehicleParams, segment_energy_breakdown_kWh, simulate_route
from conditional_gtfs_motion import select_gtfs_stops
from signal_fidelity import prepare_signal_fidelity, synthetic_locations
from trip_motion import solve_trip_motion


def toy_trip():
    return [Segment(500, dwell_s=0, run_time_s=100, trip_id="toy", service_date="2025-07-09",
                    segment_index=i, segment_geometry_id=f"g{i}",
                    from_stop_departure_time=f"{8+i:02d}:00:00",
                    to_stop_arrival_time=f"{9+i:02d}:00:00",
                    to_stop_departure_time=f"{9+i:02d}:00:00",
                    speed_cap_ms=10+i, grade=.01*i, passengers=10+5*i,
                    n_signals=2, signal_events=[
                        {"signal_event_id":f"s{i}-{j}", "distance_along_segment_m":x,
                         "signal_lat":51., "signal_lon":-8.}
                        for j,x in enumerate((100,300))]) for i in range(2)]


def params(**kwargs):
    return MotionParams(solver_mode="trip_continuous", signal_fidelity="F1_frequency", **kwargs)


def signals(result):
    return [e for e in result.events if e.kind == "signal"]


def exposure(result):
    return [tuple(e.signal_decision[k] for k in
                  ("stop_probability", "random_value", "stop_decision", "red_wait_s_requested"))
            for e in signals(result)]


@pytest.mark.parametrize("q", [0., 1.])
def test_matched_exposure_physics_and_energy(q):
    segments = toy_trip()
    mp = params(gtfs_stop_policy="conditional_structural", q_stop=q)
    results = {f:solve_trip_motion(segments, replace(mp, signal_fidelity=f))
               for f in ("F0_none", "F1_frequency", "F2_spatial")}
    assert [len(signals(r)) for r in results.values()] == [0,4,4]
    assert exposure(results["F1_frequency"]) == exposure(results["F2_spatial"])
    result = results["F1_frequency"]
    assert result.length_m == 1000
    assert all(a.distance_m <= b.distance_m for a,b in zip(result.events,result.events[1:]))
    for left,right in zip(result.spans,result.spans[1:]):
        assert left.end_time_s == pytest.approx(right.start_time_s)
        assert left.end_distance_m == pytest.approx(right.start_distance_m)
        assert (left.leg.terminal_speed_ms if left.leg else 0) == pytest.approx(
            right.leg.initial_speed_ms if right.leg else 0)
    for interval, profile, diag in zip(result.intervals,result.segment_profiles,result.segment_diagnostics):
        assert np.max(profile[1]) <= interval.segment.speed_cap_ms+1e-8
        assert np.min(profile[2]) >= -mp.decel_ms2-1e-8
        assert np.max(profile[2]) <= mp.accel_ms2+1e-8
        assert diag["actual_segment_time_s"] == pytest.approx(100+diag["schedule_delay_s"])
        assert diag["signal_wait_applied_s"] == diag["signal_wait_requested_s"]
        energy = segment_energy_breakdown_kWh(interval.segment, VehicleParams(), 50, mp,
                                             profile_and_diagnostics=(profile,diag))
        assert energy.net_battery_energy_kWh == pytest.approx(
            energy.gross_consumed_kWh + energy.aux_energy_kWh - energy.regen_recovered_kWh)


def test_location_seed_is_independent_and_sources_not_mutated():
    source = toy_trip()
    before = deepcopy(source)
    a = solve_trip_motion(source, params(signal_location_seed=1))
    b = solve_trip_motion(source, params(signal_location_seed=1))
    c = solve_trip_motion(source, params(signal_location_seed=2))
    assert [e.distance_m for e in signals(a)] == [e.distance_m for e in signals(b)]
    assert [e.distance_m for e in signals(a)] != [e.distance_m for e in signals(c)]
    assert exposure(a) == exposure(b) == exposure(c)
    assert source == before
    assert select_gtfs_stops(source,.5,10) == select_gtfs_stops(
        [i.segment for i in c.intervals],.5,10)
    for old,interval in zip(source,a.intervals):
        for key in ("grade", "passengers", "run_time_s", "speed_cap_ms", "length_m"):
            assert getattr(old,key) == getattr(interval.segment,key)


def test_true_positions_and_coordinates_do_not_influence_f1():
    source = toy_trip()
    a = solve_trip_motion(source,params())
    for seg in source:
        for event,x in zip(seg.signal_events,(10.,499.)):
            event.update(distance_along_segment_m=x,signal_lat=-99.,signal_lon=100.)
    b = solve_trip_motion(source,params())
    for x,y in zip(a.profile,b.profile):
        np.testing.assert_array_equal(x,y)
    assert all(e.signal_decision["signal_lat"] is None for e in signals(a))
    assert synthetic_locations(4,1000,"toy","date",1) == synthetic_locations(4,1000,"toy","date",1)


def test_hour_context_is_inherited_when_events_move(monkeypatch):
    monkeypatch.setattr("signal_fidelity.synthetic_locations",lambda *args:[501.,502.,503.,504.])
    result = solve_trip_motion(toy_trip(),params(stop_prob_by_hour={8:0.,9:1.}))
    assert [e.segment_index for e in signals(result)] == [1]*4
    assert [e.signal_decision["stop_probability"] for e in signals(result)] == [0.,0.,1.,1.]
    assert [e.required_speed_ms for e in signals(result)] == [None,None,0.,0.]


@pytest.mark.parametrize("q", [0., 1.])
def test_coincident_and_boundary_events_keep_all_waits(monkeypatch,q):
    monkeypatch.setattr("signal_fidelity.synthetic_locations",lambda *args:[0.,500.,500.,1000.])
    result = solve_trip_motion(toy_trip(),params(stop_prob=1.,use_hourly_signal_stop_probability=False,
                                               gtfs_stop_policy="conditional_structural",q_stop=q))
    assert [e.distance_m for e in signals(result)] == [0.,500.,500.,1000.]
    assert sum(d["signal_wait_applied_s"] for d in result.segment_diagnostics) == 60.
    assert sum(s.end_time_s-s.start_time_s for s in result.spans if s.kind=="signal_wait") == pytest.approx(60.)


def test_extreme_positions_change_energy_not_exposure(monkeypatch):
    energies, exposures = [],[]
    for positions in ([250.,251.,252.,253.],[100.,350.,600.,850.],[.001,499.999,500.001,999.999]):
        monkeypatch.setattr("signal_fidelity.synthetic_locations",lambda *args:positions)
        mp = params(stop_prob=1.,use_hourly_signal_stop_probability=False)
        result = solve_trip_motion(toy_trip(),mp)
        exposures.append(exposure(result))
        energies.append(sum(segment_energy_breakdown_kWh(i.segment,VehicleParams(),50,mp,
                            profile_and_diagnostics=(p,d)).net_battery_energy_kWh
                            for i,p,d in zip(result.intervals,result.segment_profiles,result.segment_diagnostics)))
    assert exposures[0] == exposures[1] == exposures[2]
    assert max(energies)-min(energies) > .01


def test_fallback_rejected_and_default_unchanged():
    with pytest.raises(ValueError,match="complete F2"):
        prepare_signal_fidelity([Segment(100,n_signals=1)],params())
    with pytest.raises(ValueError,match="complete F2"):
        prepare_signal_fidelity([Segment(100)],params())
    assert prepare_signal_fidelity([Segment(100,signal_events=[])],params())[0].n_signals == 0
    assert MotionParams().signal_fidelity == "F2_spatial"
    with pytest.raises(ValueError,match="trip_continuous"):
        MotionParams(signal_fidelity="F1_frequency")
    with pytest.raises(ValueError,match="signal_location_seed"):
        params(signal_location_seed=1.5)


def test_simulate_route_exports_synthetic_diagnostics():
    source = toy_trip()
    rows = simulate_route(source,VehicleParams(),motion_params=params())
    assert rows.n_signals.sum() == 4
    assert rows.signal_source.eq("F1_frequency").all()
    assert sum(len(s.signal_event_diagnostics) for s in source) == 4
    assert all(s.signal_events[0]["signal_event_id"].startswith("s") for s in source)


def test_config_dispatch(tmp_path):
    from best_ire_beb.config import validate_config
    path = tmp_path/"f1.yaml"
    path.write_text("motion:\n  solver_mode: trip_continuous\n  signal_fidelity: F1_frequency\n  signal_location_seed: 12\n",encoding="utf-8")
    assert MotionParams.from_config(path).signal_location_seed == 12
    validate_config(path)


@pytest.mark.parametrize("mode",["segment_legacy","trip_continuous"])
def test_f0_route_clears_previous_event_diagnostics_without_changing_inventory(mode):
    source = toy_trip()
    simulate_route(source,VehicleParams(),motion_params=MotionParams(solver_mode=mode))
    assert any(s.signal_event_diagnostics for s in source)
    result = simulate_route(source,VehicleParams(),motion_params=MotionParams(solver_mode=mode,signal_fidelity="F0_none"))
    assert result.n_signals.sum() == 0
    assert all(s.signal_event_diagnostics == [] for s in source)
    assert sum(len(s.signal_events) for s in source) == 4
