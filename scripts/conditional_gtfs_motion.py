"""B1 structural stop sensitivity; frozen signal encounters and local budgets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from beb_soc_model import _concat_profile_parts, _idle_profile, _profile_duration, _stable_uniform01
from continuous_interval import solve_interval
from trip_motion import TimelineSpan, TripMotionProfile, build_trip_events


def select_gtfs_stops(segments, q_stop, seed):
    """Stable nested structural draws, independent of all signal-side fields."""
    if not 0 <= q_stop <= 1:
        raise ValueError("q_stop must be in [0, 1]")
    if not segments:
        return []
    decisions = []
    for j in range(len(segments) + 1):
        source = segments[max(j - 1, 0)]
        reason = "eligible_zero_dwell"
        if j == 0:
            reason = "first_stop"
        elif j == len(segments):
            reason = "last_stop"
        elif source.dwell_s > 0:
            reason = "positive_scheduled_dwell"
        elif getattr(source, "to_stop_compulsory", False):
            reason = "explicit_terminal_or_hold"
        elif any(s.run_time_s is None or s.run_time_s <= 0 for s in (source, segments[j])):
            reason = "missing_local_timing"
        elif (source.to_stop_departure_time and segments[j].from_stop_departure_time
              and source.to_stop_departure_time != segments[j].from_stop_departure_time):
            reason = "discontinuous_timing_boundary"
        eligible = reason == "eligible_zero_dwell"
        # Geometry IDs can be absent in F0; never use signal-populated context.
        random_value = _stable_uniform01("gtfs_stop_structural_v1", seed,
                                         source.service_date, source.trip_id, j,
                                         source.segment_index, source.to_stop_arrival_time,
                                         source.to_stop_departure_time)
        decisions.append({"event_id":f"gtfs:{j}", "stop_index":j, "eligible":eligible,
                          "reason":reason, "random_value":random_value,
                          "stop_decision":not eligible or random_value < q_stop,
                          "q_stop":q_stop, "gtfs_stop_random_seed":seed})
    return decisions


@dataclass(frozen=True)
class PlacedInterval:
    start_distance_m: float
    end_distance_m: float
    motion: object

    @property
    def initial_speed_ms(self):
        return self.motion.initial_speed_ms

    @property
    def terminal_speed_ms(self):
        return self.motion.terminal_speed_ms

    @property
    def peak_speed_ms(self):
        return self.motion.peak_speed_ms

    @property
    def duration_s(self):
        return self.motion.duration_s

    def position_speed(self, time_s):
        x, v = self.motion.position_speed(time_s)
        return x + self.start_distance_m, v


@dataclass
class ConditionalTripProfile(TripMotionProfile):
    gtfs_stop_decisions: list | None = None


def _boundary_speeds(events, intervals, decisions, mp):
    gtfs = [e for e in events if e.kind == "gtfs_stop"]
    for j, (event, decision) in enumerate(zip(gtfs, decisions)):
        if decision["stop_decision"]:
            event.required_speed_ms = 0.0
        else:
            adjacent = [intervals[j-1].segment, intervals[j].segment]
            event.required_speed_ms = min(
                *(mp.resolve_cap(s)[0] for s in adjacent),
                *(s.length_m/s.run_time_s for s in adjacent))
            assert event.dwell_s == 0.0
        event.actual_speed_ms = event.required_speed_ms
    nodes = [e for e in events if e.required_speed_ms is not None]
    speeds = [float(e.required_speed_ms) for e in nodes]
    # Forward/backward reachability gives one shared speed at every boundary.
    for i in range(1,len(nodes)):
        length = max(nodes[i].distance_m-nodes[i-1].distance_m,0.0)
        speeds[i] = min(speeds[i],math.sqrt(speeds[i-1]**2+2*mp.accel_ms2*length))
    for i in range(len(nodes)-2,-1,-1):
        length = max(nodes[i+1].distance_m-nodes[i].distance_m,0.0)
        speeds[i] = min(speeds[i],math.sqrt(speeds[i+1]**2+2*mp.decel_ms2*length))
    for event,speed in zip(nodes,speeds):
        event.required_speed_ms = event.actual_speed_ms = speed
    for event, decision in zip(gtfs,decisions):
        decision["boundary_speed_ms"] = event.required_speed_ms
        decision["signal_coincident_zero"] = not decision["stop_decision"] and event.required_speed_ms == 0.0
        if decision["signal_coincident_zero"]:
            assert any(e.kind != "gtfs_stop" and e.required_speed_ms == 0 and
                       abs(e.distance_m-event.distance_m) < 1e-9 for e in nodes)


def _allocate_local_times(lengths, minima, maxima, budget):
    if budget is None:
        return [None]*len(lengths)
    if budget <= sum(minima) + 1e-9:
        return list(minima)
    assigned = list(minima)
    remaining = budget - sum(assigned)
    # Slack can move between signal legs inside this one GTFS interval only.
    for _ in range(len(lengths)+1):
        active = [i for i in range(len(lengths)) if lengths[i] > 1e-9 and maxima[i]-assigned[i] > 1e-9]
        if remaining < 1e-8:
            break
        if not active:
            raise ValueError("Local budget exceeds all physically valid nonstop leg durations")
        weight = sum(lengths[i] for i in active)
        increments = {i:min(remaining*lengths[i]/weight,maxima[i]-assigned[i]) for i in active}
        for i,extra in increments.items():
            assigned[i] += extra
        remaining -= sum(increments.values())
    if remaining > 1e-7:
        raise ValueError("Could not preserve local scheduled moving time")
    return assigned


def solve_conditional_trip(segments, mp):
    segments = list(segments)
    events, intervals = build_trip_events(segments,mp)
    decisions = select_gtfs_stops(segments,mp.q_stop,mp.gtfs_stop_random_seed)
    if not intervals:
        empty = np.array([],dtype=float)
        return ConditionalTripProfile([],[],[],(empty,empty,empty,empty),empty,
                                      np.array([],dtype=int),np.array([],dtype="U12"),gtfs_stop_decisions=[])
    _boundary_speeds(events,intervals,decisions,mp)
    spans, trip_parts, distances, owners, kinds = [],[],[],[],[]
    segment_profiles, diagnostics = [],[]
    clock, scheduled_elapsed = 0.0,0.0

    def append(kind,index,profile,x,leg=None):
        nonlocal clock
        duration = _profile_duration(profile)
        if duration <= 0:
            return
        spans.append(TimelineSpan(kind,index,clock,clock+duration,float(x[0]),
                                  leg.end_distance_m if leg else float(x[0]),leg))
        trip_parts.append(profile)
        distances.append(x)
        owners.append(np.full(len(x),index,dtype=int))
        kinds.append(np.full(len(x),kind,dtype="U12"))
        clock += duration

    for interval in intervals:
        i, seg, diag = interval.segment_index, interval.segment, interval.diagnostics
        start_time = clock
        interval.start.departure_time_s = clock
        signals = [e for e in interval.signals if e.required_speed_ms is not None]
        bounds = [interval.start,*signals,interval.end]
        lengths = interval.leg_lengths_m
        minima, maxima = [],[]
        for length,origin,destination in zip(lengths,bounds,bounds[1:]):
            if length <= 1e-9:
                if abs(origin.required_speed_ms-destination.required_speed_ms)>1e-8:
                    raise ValueError("Coincident event constraints have incompatible speeds")
                minima.append(0.0)
                maxima.append(0.0)
                continue
            fastest = solve_interval(length,origin.required_speed_ms,destination.required_speed_ms,
                                      diag["speed_cap_ms"],mp.accel_ms2,mp.decel_ms2)
            minima.append(fastest.minimum_time_s)
            maxima.append(fastest.maximum_nonstop_time_s)
        minimum = sum(minima)
        budget = diag["moving_time_budget_s"]
        budgets = _allocate_local_times(lengths,minima,maxima,budget)
        interval.leg_budgets_s = budgets
        interval.fastest = budget is not None and budget < minimum-1e-9
        parts, moving, valley_count, maximum_speed = [],0.0,0,0.0
        exact_distance, kinetic_per_mass = 0.0,0.0
        for k,(length,rt,origin,destination) in enumerate(zip(lengths,budgets,bounds,bounds[1:])):
            destination.arrival_time_s = clock
            if length > 1e-9:
                motion = solve_interval(length,origin.required_speed_ms,destination.required_speed_ms,
                                        diag["speed_cap_ms"],mp.accel_ms2,mp.decel_ms2,rt,
                                        diag["target_cruise_ms"])
                if motion.timing_status == "too_long_for_nonstop_boundaries":
                    raise ValueError(f"B1 cannot pass a timing boundary early: trip={seg.trip_id}, "
                                     f"segment={i}, leg={k}, length={length}, "
                                     f"speeds={origin.required_speed_ms},{destination.required_speed_ms}, "
                                     f"budget={rt}, min={motion.minimum_time_s}, "
                                     f"max={motion.maximum_nonstop_time_s}")
                leg = PlacedInterval(origin.distance_m,destination.distance_m,motion)
                profile = motion.sample(mp.dt_s)
                for event in interval.signals:
                    if event.required_speed_ms is None and origin.distance_m <= event.distance_m <= destination.distance_m:
                        t = motion.time_at_distance(event.distance_m-origin.distance_m)
                        event.arrival_time_s = event.departure_time_s = clock+t
                        event.actual_speed_ms = float(motion.position_speed(t)[1])
                append("motion",i,profile,leg.position_speed(profile[0])[0],leg)
                parts.append(profile)
                moving += _profile_duration(profile)
                exact_distance += motion.distance_integral_m
                kinetic_per_mass += sum(p.acceleration_ms2*p.distance_m for p in motion.phases)
                valley_count += motion.shape == "valley"
                maximum_speed = max(maximum_speed,motion.peak_speed_ms)
                destination.arrival_time_s = clock
            destination.departure_time_s = clock
            destination.actual_speed_ms = destination.required_speed_ms
            if k < len(bounds)-2 and destination.dwell_s > 0:
                wait = _idle_profile(destination.dwell_s,mp.dt_s)
                append("signal_wait",i,wait,np.full(len(wait[0]),destination.distance_m))
                parts.append(wait)
                destination.departure_time_s = clock
        profile = _concat_profile_parts(parts)
        actual = _profile_duration(profile)
        scheduled = diag["scheduled_time_s"]
        delay = max(actual-scheduled,0.0) if scheduled is not None else 0.0
        if scheduled is not None and actual < scheduled-1e-7:
            raise ValueError("B1 passed a local scheduled boundary early")
        if abs(exact_distance-seg.length_m)>1e-6:
            raise ValueError("B1 interval failed distance conservation")
        expected_kinetic = (interval.end.actual_speed_ms**2-interval.start.actual_speed_ms**2)/2
        if abs(kinetic_per_mass-expected_kinetic)>1e-6:
            raise ValueError("Kinetic energy was lost or duplicated at a boundary")
        if seg.dwell_s>0:
            assert decisions[i+1]["stop_decision"]
            dwell = _idle_profile(seg.dwell_s,mp.dt_s)
            append("gtfs_dwell",i,dwell,np.full(len(dwell[0]),interval.end.distance_m))
        interval.end.departure_time_s = clock
        if scheduled_elapsed is not None:
            scheduled_elapsed = scheduled_elapsed+scheduled+seg.dwell_s if scheduled is not None else None
        diag.update(actual_moving_time_s=moving,actual_segment_time_s=actual,
                    moving_profile_time_s=moving,actual_profile_time_s=actual,
                    minimum_feasible_moving_time_s=minimum,min_feasible_motion_time_s=minimum,
                    schedule_delay_s=delay,schedule_infeasible=delay>1e-6,
                    solver_mode="trip_continuous",gtfs_stop_policy="B1_structural",
                    trip_start_time_s=start_time,trip_arrival_time_s=interval.end.arrival_time_s,
                    trip_departure_time_s=clock,
                    cumulative_schedule_delay_s=max(clock-scheduled_elapsed,0.0) if scheduled_elapsed is not None else None,
                    initial_speed_ms=interval.start.actual_speed_ms,terminal_speed_ms=interval.end.actual_speed_ms,
                    q_stop=mp.q_stop,gtfs_stop_random_seed=mp.gtfs_stop_random_seed,
                    gtfs_from_stop_selected=decisions[i]["stop_decision"],
                    gtfs_to_stop_selected=decisions[i+1]["stop_decision"],
                    gtfs_to_stop_eligible=decisions[i+1]["eligible"],
                    gtfs_to_stop_reason=decisions[i+1]["reason"],
                    signal_coincident_zero=decisions[i+1]["signal_coincident_zero"],
                    n_speed_valley_legs=valley_count,maximum_speed_ms=maximum_speed,
                    analytic_distance_m=exact_distance,kinetic_integral_per_mass=kinetic_per_mass)
        segment_profiles.append(profile)
        diagnostics.append(diag)
    profile = _concat_profile_parts(trip_parts)
    x,owner,kind = np.concatenate(distances),np.concatenate(owners),np.concatenate(kinds)
    starts = np.r_[0,np.flatnonzero(np.diff(profile[0])>0)+1]
    ends = np.r_[starts[1:]-1,len(profile[0])-1]
    collapsed = len(profile[0])-len(starts)
    profile = (profile[0][ends],profile[1][ends],profile[2][ends],np.add.reduceat(profile[3],starts))
    return ConditionalTripProfile(events,intervals,spans,profile,x[ends],owner[ends],kind[ends],
                                   segment_profiles,diagnostics,collapsed,decisions)
