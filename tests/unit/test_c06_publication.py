"""Independently calculable C06 contracts; no corrected pilot or ensembles."""

import copy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from beb_soc_model import Segment, VehicleParams
from publication_contract import (fixed_weight_summary, group_key, input_signatures,
                                  matched_contrasts, raw_cell)
from publication_inputs import (conservative_cap, endpoint_grade, load_config,
                                neutral_segments, read_hgt_rounded, service_seconds,
                                service_time_utc, weather_for_trip_utc)
from publication_timing import (Anchor, PlacedSignal, Road, SignalTuple, State, Visit,
                                bounded_allocation, desired_speed, independent_start,
                                plan_span, run_constructed_gap, run_controlled_trip, run_span, run_trip,
                                stable_uniform, supplied_placement)


def source(ordinal=0, wait=15, stopped=True):
    return SignalTuple(ordinal, f"encounter-{ordinal}", "08:00 scheduled source", .5,
                       .1 if stopped else .9, stopped, wait, ("fixture", ordinal))


def worked_plan(T=150, rule="stretch", hold=True):
    # At 10 m/s with a=d=1, 900 m requires 10+80+10=100 seconds.
    road = Road(0, 900, 10, 10, 0, "controlled fixture", True)
    stops = [Visit("origin", 0, 0, 0),
             Visit("destination", 900, T-20, T, reliable_passenger_dwell_s=20,
                   authorised_hold=hold)]
    return plan_span([road], stops, [PlacedSignal(0, source())],
                     Anchor(0, 0, "departure"), Anchor(900, T, "departure"),
                     rule=rule, accel_ms2=1, decel_ms2=1)


def execute(plan, incoming=0, stored=369):
    vehicle = VehicleParams()
    return run_span(plan, State(plan.start.scheduled_s+incoming, stored, 410), vehicle)


@pytest.mark.parametrize("rule,movement,holding", [("stretch",115,0),("desired_holding",100,15)])
def test_feasible_worked_ledger(rule, movement, holding):
    state, r = execute(worked_plan(rule=rule))
    assert r["elapsed_s"] == pytest.approx(150, abs=1e-6)
    assert r["movement_s"] == pytest.approx(movement, abs=1e-6)
    assert r["passenger_dwell_s"] == 20
    assert r["signal_wait_s"] == 15
    assert r["other_holding_s"] == pytest.approx(holding)
    assert r["auxiliary_kWh"] == pytest.approx(.125, abs=1e-12)
    assert r["outgoing_lateness_s"] == 0
    assert state.stored_kWh == pytest.approx(369-r["net_kWh"], abs=1e-10)
    assert r["visits"][-1]["arrival_s"] == pytest.approx(movement+15)
    assert r["visits"][-1]["departure_s"] == pytest.approx(150)


@pytest.mark.parametrize("rule", ["stretch","desired_holding"])
def test_infeasible_keeps_full_wait_dwell(rule):
    _, r = execute(worked_plan(T=120,rule=rule))
    assert r["elapsed_s"] == 135
    assert r["outgoing_lateness_s"] == 15
    assert r["signal_wait_s"] == 15 and r["passenger_dwell_s"] == 20
    assert r["auxiliary_kWh"] == pytest.approx(.1125)
    assert "minimum_motion_dwell_wait_exceeds_anchor" in r["flags"]


@pytest.mark.parametrize("incoming,elapsed,late", [(15,45,0),(40,30,10)])
def test_gap_actual_time_energy(incoming, elapsed, late):
    end, r = run_constructed_gap(State(150+incoming,369,410),150,60,30,VehicleParams(),
                                 aux_power_kW=3,operating_state="powered_fixture_layover")
    assert r["elapsed_s"] == elapsed
    assert r["outgoing_lateness_s"] == late
    assert end.clock_s == 210+late
    assert end.stored_kWh == pytest.approx(369-3*elapsed/3600)


def test_lateness_removes_only_hold_not_nominal_movement():
    for rule, expected in [("stretch",160),("desired_holding",150)]:
        plan = worked_plan(rule=rule)
        end, r = execute(plan,incoming=10)
        assert end.clock_s == pytest.approx(expected)
        assert r["movement_s"] == pytest.approx(115 if rule=="stretch" else 100)
        assert r["recovered_holding_s"] == (0 if rule=="stretch" else 10)
        assert r["auxiliary_kWh"] == pytest.approx(3*r["elapsed_s"]/3600)


def test_two_spans_and_gap_carry_clock_and_full_precision_energy():
    vehicle=VehicleParams()
    first, a=execute(worked_plan(T=120))
    second, gap=run_constructed_gap(first,120,60,30,vehicle,aux_power_kW=3,
                                     operating_state="constructed powered gap")
    plan=worked_plan()
    plan.start=replace(plan.start,scheduled_s=180)
    plan.end=replace(plan.end,scheduled_s=330)
    final, b=run_span(plan,second,vehicle)
    assert gap["elapsed_s"]==45
    assert b["initial_state"]==asdict(second)
    assert final.stored_kWh==pytest.approx(369-a["net_kWh"]-gap["auxiliary_kWh"]-b["net_kWh"])


def test_bounded_length_weighted_allocation_and_limits():
    values,residual=bounded_allocation([100,300],[10,20],[15,100],80)
    assert values==pytest.approx([15,65]); assert residual==0
    values,residual=bounded_allocation([100,300],[10,20],[15,25],80)
    assert values==[15,25]; assert residual==40
    assert bounded_allocation([],[],[],10)==([],10)


@pytest.mark.parametrize("kind,expected_dwell", [("arrival",0),("departure",20)])
def test_endpoint_dwell_ownership(kind,expected_dwell):
    plan=worked_plan()
    p=plan_span(plan.roads,plan.visits,[],plan.start,replace(plan.end,kind=kind),
                accel_ms2=1,decel_ms2=1)
    _,r=execute(p)
    assert r["passenger_dwell_s"]==expected_dwell


def test_arrival_to_departure_owns_dwell_and_hold():
    v=Visit("station",0,0,30,reliable_passenger_dwell_s=20,authorised_hold=True)
    p=plan_span([], [v], [], Anchor(0,0,"arrival"),Anchor(0,30,"departure"))
    _,r=execute(p)
    assert r["passenger_dwell_s"]==20 and r["other_holding_s"]==10


def test_no_authorised_hold_incompatible_departure_is_unavailable():
    v=Visit("station",0,0,30,reliable_passenger_dwell_s=20)
    p=plan_span([], [v], [], Anchor(0,0,"arrival"),Anchor(0,30,"departure"))
    _,r=execute(p)
    assert r["elapsed_s"]==20 and r["failure"]
    assert "incompatible_spare_time_no_authorised_hold" in r["flags"]


def test_alternative_fallback_and_legal_speed_increase():
    p=worked_plan(rule="desired_holding",hold=False)
    assert "desired_holding_fallback_stretch_no_authorised_hold" in p.flags
    r=Road(0,900,10,5,0,"fixture",True)
    base=worked_plan()
    p=plan_span([r],base.visits,base.signals,base.start,base.end,
                 rule="desired_holding",accel_ms2=1,decel_ms2=1)
    _,result=execute(p)
    assert result["movement_s"]==pytest.approx(115,abs=1e-6)
    assert all(period.profile.peak_speed_ms<=10 for period in p.periods if period.profile)


@pytest.mark.parametrize("served",[True,False])
def test_served_passed_continuous_boundary_local_cap_and_wait(served):
    roads=[Road(0,100,10,8,0,"fixture A",True),Road(100,300,6,6,.03,"fixture B",True)]
    visits=[Visit("a",0,0,0),Visit("b",100,30,30,served=served),Visit("c",300,100,100)]
    p=plan_span(roads,visits,[],Anchor(0,0,"departure"),Anchor(300,100,"arrival"))
    _,r=execute(p)
    assert r["passenger_dwell_s"]==(20 if served else 0)
    assert p.boundary_speeds[100]==(0 if served else 6)
    for period in p.periods:
        if period.profile:
            assert period.profile.peak_speed_ms<=roads[period.road_index].cap_ms+1e-9
            assert period.profile.distance_integral_m==pytest.approx(period.end_m-period.position_m,abs=1e-6)
    # A passed GTFS stop coincident with a stopped signal still has full signal wait.
    p=plan_span(roads,visits,[PlacedSignal(100,source())],p.start,p.end)
    _,r=execute(p)
    assert r["signal_wait_s"]==15
    assert p.boundary_speeds[100]==0
    assert r["passenger_dwell_s"]==(20 if served else 0)
    assert next(x for x in p.periods if x.kind=="signal_wait").road_index==1


def test_positive_recorded_dwell_compulsory_but_not_automatic_duration_or_hold():
    base=worked_plan()
    v=replace(base.visits[-1],recorded_dwell_s=570,reliable_passenger_dwell_s=None,
               authorised_hold=False)
    p=plan_span(base.roads,[base.visits[0],v],[],base.start,base.end)
    assert sum(x.duration_s for x in p.periods if x.kind=="passenger_dwell")==20
    assert all(x.kind!="other_holding" for x in p.periods)
    with pytest.raises(ValueError,match="cannot be passed"):
        replace(v,served=False).passenger_dwell(20)


@pytest.mark.parametrize("positions",[[],[0],[900],[100,100.001],[100,100],[0,100,900]])
def test_zero_multiple_close_boundary_signals_no_loss_or_overlap(positions):
    base=worked_plan(T=400)
    sources=[source(i) for i in range(len(positions))]
    signals=supplied_placement(sources,positions)
    p=plan_span(base.roads,base.visits,signals,base.start,base.end,
                 include_terminal_signals=True)
    _,r=execute(p)
    assert r["signal_wait_s"]==15*len(positions)
    assert len(r["signal_tuples"])==len(positions)
    assert r["elapsed_s"]==pytest.approx(sum(r[k] for k in
        ("movement_s","signal_wait_s","passenger_dwell_s","other_holding_s")),abs=1e-6)


def test_source_tuples_immutable_ordinal_and_order_independent():
    inventory=[source(0),source(1,stopped=False)]
    a=supplied_placement(inventory,[10,20]); b=supplied_placement(inventory,[500,600])
    assert [x.source for x in a]==[x.source for x in b]
    keys=[("trip-b",0),("trip-a",1),("trip-a",0)]
    first={k:stable_uniform("signal-behaviour-v1",20260701,*k) for k in keys}
    second={k:stable_uniform("signal-behaviour-v1",20260701,*k) for k in reversed(keys)}
    assert first==second
    assert stable_uniform("B1-stop-v1",20260702,"a",0)!=first[("trip-a",0)]
    with pytest.raises(ValueError): supplied_placement(inventory,[10])
    with pytest.raises(ValueError): supplied_placement(inventory,[20,10])


def test_block_unaccepted_roads_incoherent_anchors_and_early_incoming():
    base=worked_plan()
    with pytest.raises(ValueError,match="physical input"):
        plan_span([replace(base.roads[0],accepted=False)],base.visits,[],base.start,base.end)
    with pytest.raises(ValueError,match="anchor span"):
        plan_span(base.roads,base.visits,[],base.start,replace(base.end,scheduled_s=0))
    with pytest.raises(ValueError,match="Early incoming"):
        run_span(base,State(-1,369,410),VehicleParams())


def test_neutral_inputs_ignore_weather_route_hour_b1_and_preserve_originals():
    config=load_config()
    inputs=[Segment(100,grade=.03,passengers=0),Segment(200,grade=-.02,passengers=85)]
    inputs[0].aux_power_kW=20
    changed=neutral_segments(inputs,config)
    assert [s.passengers for s in changed]==[20,20]
    assert [s.aux_power_kW for s in changed]==[3,3]
    assert [s.controlled_temperature_c for s in changed]==[22,22]
    assert inputs[0].passengers==0 and inputs[0].aux_power_kW==20
    assert [s.grade for s in changed]==[.03,-.02]
    assert config["total_vehicle_mass_kg"]==15400


@pytest.mark.parametrize("day,value,utc",[("2026-07-09","24:03:00","2026-07-09T23:03:00+00:00"),
    ("2026-01-09","25:03:00","2026-01-10T01:03:00+00:00"),
    ("2026-03-29","00:00:00","2026-03-28T23:00:00+00:00"),
    ("2026-10-25","00:00:00","2026-10-25T00:00:00+00:00")])
def test_extended_day_and_dst_service_origin(day,value,utc):
    assert service_time_utc(day,service_seconds(value)).isoformat()==utc


def test_weather_corrected_row_identity_without_changing_historical_reader():
    class Weather:
        df=pd.DataFrame({"temp":[11,22]},index=pd.to_datetime(["2026-07-09 23:00","2026-07-10 00:00"]))
    r=weather_for_trip_utc(Weather(),"2026-07-09",
                          [{"departure_time":"24:03:00","arrival_time":"24:03:00"}])
    assert r["selected_row_utc"]=="2026-07-09T23:00:00+00:00"
    assert r["conditions"]["temp"]==11 and r["offset_s"]==-180


def test_dem_reader_grade_missing_not_flat_and_conservative_directional_caps(tmp_path):
    p=tmp_path/"N51W009.hgt"
    np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=">i2").tofile(p)
    assert read_hgt_rounded(p,51.5,-8.5)==5
    assert endpoint_grade(5,10,100)["grade"]==.05
    assert endpoint_grade(0,100,100)["clipped"]
    with pytest.raises(ValueError): endpoint_grade(float('nan'),5,100)
    tags={"maxspeed":"50","maxspeed:forward":"40",
          "maxspeed:conditional":"30 @ (Mo-Fr 08:00-18:00)"}
    cap=conservative_cap(tags,"forward",True)
    assert cap["cap_ms"]==pytest.approx(30/3.6) and not cap["legal_bound_verified"]
    assert conservative_cap({"maxspeed":"30 mph"},"backward",True)["cap_ms"]==pytest.approx(13.4112)
    with pytest.raises(ValueError): conservative_cap(tags,"forward",False)
    with pytest.raises(ValueError): conservative_cap({"maxspeed":"signals"},"forward",True)


def example_cell(trip="t",fidelity="F0",p=None,b=None,energy=1,distance=1000):
    group=dict(feed_sha256="fixture-feed",service_date="2026-07-09",route_id="route",
               pattern_id="pattern",trip_id=trip,analysis_mode="standalone_trip",
               scenario_id="controlled-fixture",timing_rule="stretch",B1_stratum="B0")
    inputs={**{k:group[k] for k in ("feed_sha256","service_date","route_id","pattern_id","trip_id")},
            "scheduled_visits":[0,100],"physical_roads":[distance],"stop_decisions":[True,True],
            "scenario":{"load":20,"power":3},"initial_state":{"stored_kWh":369},
            "numerical":{"dt_s":.5},"source_hashes":{"code":"fixture"},"runtime":"fixture",
            "signal_representation":{"fidelity":fidelity,"placement_index":p,"behaviour_index":b}}
    result={"net_kWh":energy,"traction_kWh":energy,"auxiliary_kWh":0,"accepted_regen_kWh":0,
            "distance_m":distance,"elapsed_s":100,"movement_s":100,"passenger_dwell_s":0,
            "signal_wait_s":0,"other_holding_s":0,"initial_state":{"stored_kWh":369},
            "final_state":{"stored_kWh":369-energy},"incoming_lateness_s":0,
            "outgoing_lateness_s":0,"flags":[],"visits":[],"failure":False}
    return raw_cell(group,fidelity,result,inputs,placement_index=p,behaviour_index=b),inputs


def test_matching_before_weighting_missing_j_and_reference_reuse():
    cells=[]; expected=[]
    for trip,base,dist in [("short",1,1000),("long",4,4000)]:
        rows=[example_cell(trip,"F0",energy=base,distance=dist)[0],
              example_cell(trip,"F2",b=0,energy=base+3,distance=dist)[0]]
        for p in [0,1]:
            rows.append(example_cell(trip,"F1-U",p=p,b=0,energy=base+1,distance=dist)[0])
            for rule in ["U","J"]:
                expected.append({**{k:rows[0][k] for k in (
                    "feed_sha256","service_date","route_id","pattern_id","trip_id","analysis_mode",
                    "scenario_id","timing_rule","B1_stratum")},"placement_rule":rule,
                    "placement_index":p,"behaviour_index":0})
        cells+=rows
    result=matched_contrasts(list(reversed(cells)),expected)
    complete=[r for r in result if r["status"]=="complete"]
    assert len(complete)==4 and len(result)==8
    assert all(r["exposure_kWh"]+r["location_kWh"]==r["total_kWh"] for r in complete)
    subset=[r for r in complete if r["placement_index"]==0]
    summary=fixed_weight_summary(subset,{group_key(r):1 for r in subset})
    assert summary["F2_kWh_per_km"]==pytest.approx(11/5)
    assert summary["exposure_pct"]==40  # 2 / 5, not the mean of 100% and 25%.
    assert len({r["F2_ref"] for r in complete})==2
    j=[r for r in result if r["placement_rule"]=="J" and r["placement_index"]==0]
    assert fixed_weight_summary(j,{group_key(r):1 for r in j})["status"]=="incomplete"


def test_non_signal_hash_detects_hidden_physics_changes_and_guard_denominators():
    f0,inputs=example_cell(energy=0)
    f1,_=example_cell(fidelity="F1-U",p=0,b=0,energy=1)
    f2,_=example_cell(fidelity="F2",b=0,energy=2)
    expected={**{k:f0[k] for k in (
        "feed_sha256","service_date","route_id","pattern_id","trip_id","analysis_mode",
        "scenario_id","timing_rule","B1_stratum")},"placement_rule":"U","placement_index":0,"behaviour_index":0}
    r=matched_contrasts([f0,f1,f2],[expected])[0]
    assert r["total_pct"] is None and r["exposure_pct"] is None
    changed=copy.deepcopy(inputs); changed["scenario"]["load"]=21
    assert input_signatures(changed)["non_signal_input_signature"]!=f0["non_signal_input_signature"]
    f1["non_signal_input_signature"]="unequal"
    with pytest.raises(ValueError,match="unequal non-signal"):
        matched_contrasts([f0,f1,f2],[expected])
    with pytest.raises(ValueError,match="Duplicate raw"):
        matched_contrasts([f0,f0],[expected])


def test_independent_starts_and_headroom_at_full_precision():
    vehicle=VehicleParams()
    assert independent_start(200,vehicle).stored_kWh==369
    plan=worked_plan()
    state,r=execute(plan,stored=410)
    assert state.stored_kWh<=410
    assert state.stored_kWh==pytest.approx(410-r["net_kWh"],abs=1e-10)
    assert r["accepted_regen_kWh"]>=0


def test_desired_speed_registered_rule():
    assert desired_speed(100,10,20)==pytest.approx(11.5)
    assert desired_speed(100,None,10)==10
    assert desired_speed(100,1000,20)==3


def test_whole_trip_two_anchors_no_dwell_duplication_and_independent_modes():
    vehicle=VehicleParams()
    a=Anchor(0,0,"departure"); b=Anchor(100,60,"departure"); c=Anchor(200,120,"arrival")
    v0=Visit("a",0,0,0); v1=Visit("b",100,40,60,authorised_hold=True); v2=Visit("c",200,120,120)
    first=plan_span([Road(0,100,10,8,0,"fixture",True)],[v0,v1],[],a,b)
    second=plan_span([Road(100,200,10,8,.01,"fixture",True)],[v1,v2],[],b,c)
    end,r=run_trip([first,second],vehicle)
    assert r["elapsed_s"]==pytest.approx(120)
    assert r["passenger_dwell_s"]==20
    assert r["initial_state"]["stored_kWh"]==369
    assert r["auxiliary_kWh"]==pytest.approx(.1)
    # Start the corresponding continuous trip from an explicit prior state.
    _,continuous=run_trip([first,second],vehicle,analysis_mode="duty_continuous",
                           state=State(15,300,410))
    assert continuous["initial_state"]["stored_kWh"]==300
    assert continuous["incoming_lateness_s"]==15
    assert continuous["outgoing_lateness_s"]==pytest.approx(15)
    with pytest.raises(ValueError,match="initialise independently"):
        run_trip([first,second],vehicle,state=end)


def test_explicit_holding_visit_cannot_be_passed():
    with pytest.raises(ValueError,match="cannot be passed"):
        Visit("hold",100,10,20,served=False,authorised_hold=True).passenger_dwell(20)


def test_completed_late_cells_retained_and_execution_failures_visible():
    cell,inputs=example_cell()
    group={k:cell[k] for k in ("feed_sha256","service_date","route_id","pattern_id","trip_id",
                              "analysis_mode","scenario_id","timing_rule","B1_stratum")}
    late={**cell["result"],"outgoing_lateness_s":15,"flags":["minimum_motion_dwell_wait_exceeds_anchor"]}
    assert raw_cell(group,"F0",late,inputs)["status"]=="complete"
    failed=raw_cell(group,"F0",{},inputs,status="failed",blocker="controlled execution error")
    assert failed["blocker"]=="controlled execution error"
    with pytest.raises(ValueError,match="Failed execution"):
        raw_cell(group,"F0",{**late,"failure":True},inputs)


def test_full_battery_accepts_regeneration_offsetting_same_sample_auxiliary_draw():
    from continuous_interval import solve_interval
    from publication_timing import Period, _charge_period
    # Pure braking 10->2 m/s over 48 m in 8 s. At every sampled speed the
    # recoverable braking power exceeds 3 kW; an exactly full pack can accept
    # precisely the concurrent auxiliary draw and remains full.
    profile=solve_interval(48,10,2,10,1,1,scheduled_time_s=8)
    period=Period("movement",8,0,0,profile=profile,end_m=48)
    state,energy=_charge_period(period,Road(0,48,10,10,0,"fixture",True),
                                 State(0,410,410),VehicleParams(),20,3,.5)
    assert state.stored_kWh==pytest.approx(410,abs=1e-12)
    assert energy["accepted_regen_kWh"]==pytest.approx(3*8/3600,abs=1e-12)
    assert energy["net_kWh"]==pytest.approx(0,abs=1e-12)


def test_cell_rejects_state_hash_and_battery_energy_mismatches():
    cell,inputs=example_cell()
    group={k:cell[k] for k in ("feed_sha256","service_date","route_id","pattern_id","trip_id",
                              "analysis_mode","scenario_id","timing_rule","B1_stratum")}
    changed=copy.deepcopy(cell["result"])
    changed["initial_state"]["stored_kWh"]=300
    with pytest.raises(ValueError,match="Actual initial state"):
        raw_cell(group,"F0",changed,inputs)
    changed=copy.deepcopy(cell["result"])
    changed["final_state"]["stored_kWh"]+=.1
    with pytest.raises(ValueError,match="battery state"):
        raw_cell(group,"F0",changed,inputs)


def test_holding_is_at_latest_authorised_served_visit_inside_span():
    roads=[Road(0,900,10,8,0,"fixture",True)]
    visits=[Visit("origin",0,0,0,authorised_hold=True),
            Visit("early-hold",300,80,100,authorised_hold=True),
            Visit("late-hold",600,180,200,authorised_hold=True),
            Visit("final",900,400,500,authorised_hold=True)]
    plan=plan_span(roads,visits,[],Anchor(0,0,"departure"),Anchor(900,400,"arrival"),
                    rule="desired_holding")
    holds=[p for p in plan.periods if p.kind=="other_holding"]
    assert len(holds)==1 and holds[0].visit_id=="late-hold"
    assert sum(p.duration_s for p in plan.periods if p.kind=="passenger_dwell")==40
    # Neither preparation before first departure nor dwell after final arrival
    # is owned, even though those endpoint visits have explicit hold permission.
    _,result=execute(plan)
    assert result["elapsed_s"]==pytest.approx(400,abs=1e-6)


def test_configured_scalars_drive_controlled_runner():
    config=load_config()
    config["passengers"]=30
    config["in_service_aux_power_kW"]=4
    config["vehicle"]["aux_power_kW"]=4
    config["total_vehicle_mass_kg"]=16100
    roads=[Road(0,900,10,8,0,"fixture",True)]
    visits=[Visit("a",0,0,0),Visit("b",900,150,150)]
    plan=plan_span(roads,visits,[],Anchor(0,0,"departure"),Anchor(900,150,"arrival"))
    _,r=run_controlled_trip([plan],config)
    assert r["auxiliary_kWh"]==pytest.approx(4*150/3600)
    assert r["initial_state"]["stored_kWh"]==369
