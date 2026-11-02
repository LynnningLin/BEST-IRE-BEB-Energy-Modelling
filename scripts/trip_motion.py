"""Experimental trip event solver. Phase 7B fixes every GTFS boundary to zero.

Events and kinematic legs own the trajectory; GTFS intervals own budgets and
physical attributes. B0 keeps historical interval budgets and sample quadrature.
No call to the legacy segment profile builder is made by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from beb_soc_model import (
    MotionParams, Segment, _allocate_spatial_leg_times, _concat_profile_parts,
    _freeflow_duration, _freeflow_phases, _idle_profile, _profile_duration,
    _sample_profile, _signal_events_for_segment, _signal_stop_count,
    _signal_stop_prob_for_segment, _spatial_signal_decisions,
    _sub_lengths_from_stop_positions,
)


@dataclass
class TripEvent:
    event_id: str
    kind: str
    distance_m: float
    segment_index: int
    required_speed_ms: Optional[float]
    dwell_s: float = 0.0
    scheduled_arrival: Optional[str] = None
    scheduled_departure: Optional[str] = None
    signal_decision: Optional[dict] = None
    arrival_time_s: float = 0.0
    departure_time_s: float = 0.0
    actual_speed_ms: float = 0.0


@dataclass(frozen=True)
class MotionLeg:
    """Analytic motion between speed constraints, independent of GTFS stops."""

    start_distance_m: float
    end_distance_m: float
    initial_speed_ms: float
    terminal_speed_ms: float
    peak_speed_ms: float
    accel_ms2: float
    decel_ms2: float
    t_acc_s: float
    t_cruise_s: float
    t_dec_s: float

    @property
    def duration_s(self):
        return self.t_acc_s + self.t_cruise_s + self.t_dec_s

    def position_speed(self, time_s):
        """Exact position and speed, including the terminal endpoint."""
        t = np.clip(np.asarray(time_s, dtype=float), 0.0, self.duration_s)
        ta = np.minimum(t, self.t_acc_s)
        tc = np.clip(t - self.t_acc_s, 0.0, self.t_cruise_s)
        td = np.clip(t - self.t_acc_s - self.t_cruise_s, 0.0, self.t_dec_s)
        x = (self.initial_speed_ms * ta + 0.5 * self.accel_ms2 * ta**2
             + self.peak_speed_ms * tc
             + self.peak_speed_ms * td - 0.5 * self.decel_ms2 * td**2)
        v = np.where(t < self.t_acc_s,
                     self.initial_speed_ms + self.accel_ms2 * ta,
                     np.where(t < self.t_acc_s + self.t_cruise_s,
                              self.peak_speed_ms,
                              self.peak_speed_ms - self.decel_ms2 * td))
        x = np.where(t >= self.duration_s, self.end_distance_m - self.start_distance_m, x)
        v = np.where(t >= self.duration_s, self.terminal_speed_ms, v)
        return x + self.start_distance_m, np.maximum(v, 0.0)

    def time_at_distance(self, distance_m):
        """Monotone inverse used to timestamp pass events without resampling."""
        lo, hi = 0.0, self.duration_s
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(self.position_speed(mid)[0]) < distance_m:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def sample(self, dt_s):
        if self.initial_speed_ms == 0.0 and self.terminal_speed_ms == 0.0:
            return _sample_profile(self.t_acc_s, self.t_cruise_s, self.t_dec_s,
                                   self.peak_speed_ms, self.accel_ms2,
                                   self.decel_ms2, dt_s)
        t = np.arange(0.0, self.duration_s, dt_s)
        step = np.minimum(dt_s, self.duration_s - t)
        _, v = self.position_speed(t)
        a = np.where(t < self.t_acc_s, self.accel_ms2,
                     np.where(t < self.t_acc_s + self.t_cruise_s, 0.0, -self.decel_ms2))
        return t, v, a, step


def solve_motion_leg(start_m, end_m, initial_speed_ms, terminal_speed_ms,
                     target_ms, cap_ms, accel_ms2, decel_ms2, budget_s=None):
    """Solve a capped accelerate/cruise/brake leg with explicit boundary speeds.

    The trip planner supplies zero boundary speeds in B0. Nonzero primitives are
    available for later planners. A long nonzero-boundary budget requiring a
    speed valley is rejected rather than silently inventing a stop or a hold.
    """
    length = float(end_m - start_m)
    v0, v1 = float(initial_speed_ms), float(terminal_speed_ms)
    if not all(np.isfinite(x) for x in (start_m, end_m, v0, v1, target_ms,
                                       cap_ms, accel_ms2, decel_ms2)):
        raise ValueError("Motion leg parameters must be finite.")
    if length < 0 or min(v0, v1) < 0 or min(cap_ms, accel_ms2, decel_ms2) <= 0:
        raise ValueError("Invalid motion leg bounds or dynamics.")
    if max(v0, v1) > cap_ms + 1e-9:
        raise ValueError("Boundary speed exceeds the physical cap.")
    a, d = accel_ms2, decel_ms2
    minimum_distance = ((v1**2 - v0**2) / (2 * a) if v1 >= v0
                        else (v0**2 - v1**2) / (2 * d))
    if length < minimum_distance - 1e-9:
        raise ValueError("Insufficient distance to reach the terminal speed.")

    # Preserve the exact B0 arithmetic, including the feasibility tolerance.
    if v0 == 0.0 and v1 == 0.0:
        min_t = _freeflow_duration(length, cap_ms, a, d)
        if budget_s is not None and budget_s > 0:
            if budget_s < min_t - 1e-9:
                ta, tc, td, peak = _freeflow_phases(length, cap_ms, a, d)
            else:
                curve = 0.5 * (1.0 / a + 1.0 / d)
                disc = max(budget_s**2 - 4.0 * curve * length, 0.0)
                peak = (budget_s - np.sqrt(disc)) / (2.0 * curve)
                if peak > cap_ms + 1e-9:
                    ta, tc, td, peak = _freeflow_phases(length, cap_ms, a, d)
                else:
                    ta, td = peak / a, peak / d
                    tc = max(budget_s - ta - td, 0.0)
        else:
            ta, tc, td, peak = _freeflow_phases(
                length, min(max(target_ms, 1e-6), cap_ms), a, d)
    else:
        curve = 0.5 * (1.0 / a + 1.0 / d)
        c = length + v0**2 / (2 * a) + v1**2 / (2 * d)
        fastest_peak = min(cap_ms, float(np.sqrt(c / curve)))

        def phases(peak):
            ta, td = max((peak - v0) / a, 0.0), max((peak - v1) / d, 0.0)
            ramp_distance = (v0 + peak) * ta / 2 + (v1 + peak) * td / 2
            return ta, max((length - ramp_distance) / peak, 0.0), td

        fastest = phases(fastest_peak)
        peak = min(max(target_ms, v0, v1), fastest_peak)
        if budget_s is not None and budget_s > 0:
            peak = fastest_peak
            if budget_s >= sum(fastest) - 1e-9:
                b = budget_s + v0 / a + v1 / d
                peak = (b - np.sqrt(max(b*b - 4*curve*c, 0.0))) / (2*curve)
                if peak < max(v0, v1) - 1e-9:
                    raise ValueError("Nonzero-boundary budget requires a speed-valley planner.")
        ta, tc, td = phases(peak)
    return MotionLeg(start_m, end_m, v0, v1, float(peak), a, d,
                     float(ta), float(tc), float(td))


@dataclass
class ControlInterval:
    segment_index: int
    segment: Segment
    start: TripEvent
    end: TripEvent
    signals: list[TripEvent]
    diagnostics: dict
    leg_lengths_m: list[float]
    leg_budgets_s: list[Optional[float]]
    fastest: bool


@dataclass
class TimelineSpan:
    kind: str
    segment_index: int
    start_time_s: float
    end_time_s: float
    start_distance_m: float
    end_distance_m: float
    leg: Optional[MotionLeg] = None


@dataclass
class TripMotionProfile:
    events: list[TripEvent]
    intervals: list[ControlInterval]
    spans: list[TimelineSpan]
    profile: tuple
    distance_m: np.ndarray
    segment_index: np.ndarray
    sample_kind: np.ndarray
    segment_profiles: list[tuple] = field(default_factory=list)
    segment_diagnostics: list[dict] = field(default_factory=list)
    collapsed_sample_count: int = 0

    @property
    def duration_s(self):
        return self.spans[-1].end_time_s if self.spans else 0.0

    @property
    def length_m(self):
        return self.intervals[-1].end.distance_m if self.intervals else 0.0


def _plan_interval(i, seg, start, end, mp):
    scheduled = float(seg.run_time_s) if seg.run_time_s is not None and seg.run_time_s > 0 else None
    cap, cap_source = mp.resolve_cap(seg)
    target = min(max(float(seg.v_cruise_ms), 1e-6), cap)
    prob, hour, prob_source = _signal_stop_prob_for_segment(
        seg, mp.stop_prob, mp.hour_table_or_none())
    cached = _signal_events_for_segment(seg)
    n_signals = max(int(seg.n_signals or 0), 0)
    if cached and len(cached) != n_signals:
        raise ValueError("Segment.n_signals does not match associated signal events")
    if mp.signal_fidelity in {"F1_frequency", "F1-U", "F1-J"}:
        decisions = seg._paired_signal_decisions
        prob = float(np.mean([d["stop_probability"] for d in decisions])) if decisions else 0.0
        hour, prob_source = None, "paired_F2_events"
    else:
        decisions = (_spatial_signal_decisions(seg, cached, prob, mp.behaviour_seed(),
                                              mp.red_wait_s) if cached else [])
    positions = [d["distance_along_segment_m"] for d in decisions if d["stop_decision"]]
    stops = len(positions) if cached else _signal_stop_count(seg, n_signals, prob)
    signals = [TripEvent(
        f"signal:{i}:{j}:{d['signal_event_id']}", "signal",
        start.distance_m + d["distance_along_segment_m"], i,
        0.0 if d["stop_decision"] else None, d["red_wait_s_requested"],
        signal_decision=d,
    ) for j, d in enumerate(decisions)]
    if not cached:
        signals = [TripEvent(f"fallback:{i}:{j}", "fallback_signal_stop",
                             start.distance_m + seg.length_m * (j + 1) / (stops + 1),
                             i, 0.0, max(mp.red_wait_s, 0.0)) for j in range(stops)]

    if mp.signal_fidelity in {"F1_frequency", "F1-U", "F1-J"}:
        boundaries = [0.0, *positions, float(seg.length_m)]
        lengths = [b - a for a, b in zip(boundaries, boundaries[1:])]
    else:
        lengths = ([float(seg.length_m)] if not stops else
                   _sub_lengths_from_stop_positions(seg.length_m, positions) if cached else
                   [float(seg.length_m) / (stops + 1)] * (stops + 1))
    if cached and stops:
        minimum = sum(_freeflow_duration(l, cap, mp.accel_ms2, mp.decel_ms2)
                      if l > 1e-9 else 0.0 for l in lengths)
    elif stops:
        minimum = _freeflow_duration(lengths[0], cap, mp.accel_ms2, mp.decel_ms2) * len(lengths)
    else:
        minimum = _freeflow_duration(seg.length_m, cap, mp.accel_ms2, mp.decel_ms2)
    wait = stops * max(mp.red_wait_s, 0.0)
    if mp.signal_fidelity in {"F1-U", "F1-J"}:
        wait = sum(d["red_wait_s_requested"] for d in decisions)
    policy = str(mp.signal_time_policy or "preserve_schedule")
    budget = (scheduled - wait if policy == "preserve_schedule" else scheduled) if scheduled is not None else None
    fastest = False
    if not stops:
        budgets = [scheduled]
    elif scheduled is not None and policy == "preserve_schedule":
        fastest = budget < minimum - 1e-9
        budgets = ([None] * len(lengths) if fastest else
                   _allocate_spatial_leg_times(lengths, budget, cap, mp.accel_ms2, mp.decel_ms2)
                   if cached else [budget / len(lengths)] * len(lengths))
    elif cached:
        budgets = _allocate_spatial_leg_times(lengths, scheduled, cap, mp.accel_ms2, mp.decel_ms2)
        fastest = scheduled is not None and any(t is None for t in budgets)
    else:
        budgets = [scheduled / len(lengths) if scheduled is not None else None] * len(lengths)

    diag = {
        "n_signals": n_signals, "n_effective_signal_stops": stops, "n_signal_stops": stops,
        "signal_stop_probability": float(prob), "segment_geometry_id": seg.segment_geometry_id,
        "signal_hour": hour, "signal_stop_prob": float(prob), "signal_stop_prob_default": float(mp.stop_prob),
        "signal_stop_prob_source": prob_source, "signal_random_seed": mp.behaviour_seed(),
        "signal_event_mode": "spatial" if cached else "legacy_count",
        "signal_event_diagnostics": decisions, "realised_stop_positions_m": positions,
        "red_wait_s_assumed": float(mp.red_wait_s), "signal_time_policy": policy,
        "scheduled_time_s": scheduled, "scheduled_run_time_s": scheduled,
        "signal_wait_requested_s": wait, "signal_wait_applied_s": wait,
        "signal_wait_s": wait, "signal_wait_reduced_s": 0.0,
        "moving_time_budget_s": budget, "minimum_feasible_moving_time_s": minimum,
        "min_feasible_motion_time_s": minimum, "target_cruise_ms": target,
        "speed_cap_ms": cap, "speed_cap_source": cap_source,
        "n_motion_sublinks": len(lengths), "motion_sub_lengths_m": lengths,
        "schedule_infeasible": False,
    }
    if stops:
        diag["max_signal_wait_share"] = float(mp.max_signal_wait_share)
    if mp.signal_fidelity in {"F1_frequency", "F1-U", "F1-J"}:
        diag["signal_event_mode"] = ("F1_frequency_stochastic" if mp.signal_fidelity == "F1_frequency" else mp.signal_fidelity)
        diag["signal_location_seed"] = mp.signal_location_seed
    seg.signal_event_diagnostics = decisions
    return ControlInterval(i, seg, start, end, signals, diag, lengths, budgets, fastest)


def build_trip_events(segments, motion_params=None):
    """Build the full ordered B0 event graph before solving any motion."""
    mp = motion_params or MotionParams()
    segments = list(segments)
    if not segments:
        return [], []
    trip_ids = {s.trip_id for s in segments if s.trip_id is not None}
    if len(trip_ids) > 1:
        raise ValueError("The continuous solver accepts one trip at a time.")
    for seg in segments:
        if not np.isfinite(seg.length_m) or seg.length_m <= 0:
            raise ValueError("Trip segments must have positive finite lengths.")
        if not np.isfinite(seg.dwell_s) or seg.dwell_s < 0:
            raise ValueError("Trip dwell must be finite and nonnegative.")
        if seg.run_time_s is not None and not np.isfinite(seg.run_time_s):
            raise ValueError("Trip run time must be finite or None.")
    if mp.signal_fidelity != "F2_spatial":
        from signal_fidelity import prepare_signal_fidelity
        segments = prepare_signal_fidelity(segments, mp)
    start = TripEvent("gtfs:0", "gtfs_stop", 0.0, 0, 0.0,
                      scheduled_departure=segments[0].from_stop_departure_time)
    events, intervals = [start], []
    for i, seg in enumerate(segments):
        end = TripEvent(f"gtfs:{i+1}", "gtfs_stop", start.distance_m + seg.length_m,
                        i, 0.0, seg.dwell_s, seg.to_stop_arrival_time,
                        seg.to_stop_departure_time)
        interval = _plan_interval(i, seg, start, end, mp)
        events.extend(interval.signals)
        events.append(end)
        intervals.append(interval)
        start = end
    return events, intervals


def solve_trip_motion(segments, motion_params=None):
    """Solve B0 events on one distance/time axis and retain segment ownership.

    Time zero is first-stop departure. Full destination dwell and red waits are
    represented as stationary spans. B0 does not redistribute later scheduled
    slack to recover earlier delay; its actual clock carries the resulting time.
    """
    mp = motion_params or MotionParams()
    if mp.gtfs_stop_policy == "conditional_structural" and mp.q_stop < 1.0:
        from conditional_gtfs_motion import solve_conditional_trip
        return solve_conditional_trip(segments, mp)
    events, intervals = build_trip_events(segments, mp)
    spans, trip_parts, distances, owners, kinds = [], [], [], [], []
    segment_profiles, diagnostics = [], []
    clock = 0.0
    scheduled_elapsed = 0.0

    def append_span(kind, index, profile, x, leg=None):
        nonlocal clock
        duration = _profile_duration(profile)
        if duration <= 0:
            return
        spans.append(TimelineSpan(kind, index, clock, clock + duration,
                                  float(x[0]) if len(x) else 0.0,
                                  leg.end_distance_m if leg else float(x[0]), leg))
        trip_parts.append(profile)
        distances.append(x)
        owners.append(np.full(len(profile[0]), index, dtype=int))
        kinds.append(np.full(len(profile[0]), kind, dtype="U12"))
        clock += duration

    for interval in intervals:
        i, seg, diag = interval.segment_index, interval.segment, interval.diagnostics
        start_time = clock
        interval.start.departure_time_s = clock
        realised = [e for e in interval.signals if e.required_speed_ms == 0.0]
        boundaries = [interval.start, *realised, interval.end]
        parts, moving = [], 0.0
        for k, (length, budget) in enumerate(zip(interval.leg_lengths_m, interval.leg_budgets_s)):
            origin, destination = boundaries[k:k+2]
            destination.arrival_time_s = clock
            if length > 1e-9:
                target = diag["speed_cap_ms"] if interval.fastest and budget is None else diag["target_cruise_ms"]
                # Use the local length for B0 floating-point equivalence; the
                # analytic object is then placed on the cumulative trip axis.
                local = solve_motion_leg(0.0, length, origin.required_speed_ms,
                                         destination.required_speed_ms, target,
                                         diag["speed_cap_ms"], mp.accel_ms2, mp.decel_ms2, budget)
                leg = MotionLeg(origin.distance_m, destination.distance_m,
                                local.initial_speed_ms, local.terminal_speed_ms,
                                local.peak_speed_ms, local.accel_ms2, local.decel_ms2,
                                local.t_acc_s, local.t_cruise_s, local.t_dec_s)
                profile = leg.sample(mp.dt_s)
                for event in interval.signals:
                    if event.required_speed_ms is None and origin.distance_m <= event.distance_m <= destination.distance_m:
                        encounter_time = leg.time_at_distance(event.distance_m)
                        event.arrival_time_s = event.departure_time_s = clock + encounter_time
                        event.actual_speed_ms = float(leg.position_speed(encounter_time)[1])
                append_span("motion", i, profile, leg.position_speed(profile[0])[0], leg)
                parts.append(profile)
                moving += _profile_duration(profile)
                destination.arrival_time_s = clock
            destination.actual_speed_ms = 0.0
            destination.departure_time_s = clock
            if k < len(boundaries) - 2 and destination.dwell_s > 0:
                profile = _idle_profile(destination.dwell_s, mp.dt_s)
                append_span("signal_wait", i, profile,
                            np.full(len(profile[0]), destination.distance_m))
                parts.append(profile)
                destination.departure_time_s = clock
        profile = _concat_profile_parts(parts)
        actual = _profile_duration(profile)
        scheduled = diag["scheduled_time_s"]
        delay = max(actual - scheduled, 0.0) if scheduled is not None else 0.0
        diag.update(actual_moving_time_s=moving, actual_segment_time_s=actual,
                    moving_profile_time_s=moving, actual_profile_time_s=actual,
                    schedule_delay_s=delay, schedule_infeasible=delay > 1e-6)
        if seg.dwell_s > 0:
            dwell = _idle_profile(seg.dwell_s, mp.dt_s)
            append_span("gtfs_dwell", i, dwell, np.full(len(dwell[0]), interval.end.distance_m))
        interval.end.departure_time_s = clock
        if scheduled_elapsed is not None:
            scheduled_elapsed = (scheduled_elapsed + scheduled + seg.dwell_s
                                 if scheduled is not None else None)
        diag.update(solver_mode="trip_continuous", gtfs_stop_policy="B0_compulsory",
                    trip_start_time_s=start_time, trip_arrival_time_s=interval.end.arrival_time_s,
                    trip_departure_time_s=clock,
                    cumulative_schedule_delay_s=(max(clock - scheduled_elapsed, 0.0)
                                                  if scheduled_elapsed is not None else None),
                    initial_speed_ms=interval.start.actual_speed_ms,
                    terminal_speed_ms=interval.end.actual_speed_ms)
        segment_profiles.append(profile)
        diagnostics.append(diag)

    empty = np.array([], dtype=float)
    profile = _concat_profile_parts(trip_parts)
    x = np.concatenate(distances) if distances else empty
    owner = np.concatenate(owners) if owners else np.array([], dtype=int)
    kind = np.concatenate(kinds) if kinds else np.array([], dtype="U12")
    collapsed = 0
    if len(profile[0]):
        # A legacy final step can be shorter than one ULP of the trip clock.
        # Merge coincident global samples, keeping the right-hand boundary
        # state and total duration. Original local energy samples stay intact.
        starts = np.r_[0, np.flatnonzero(np.diff(profile[0]) > 0) + 1]
        ends = np.r_[starts[1:] - 1, len(profile[0]) - 1]
        collapsed = len(profile[0]) - len(starts)
        profile = (profile[0][ends], profile[1][ends], profile[2][ends],
                   np.add.reduceat(profile[3], starts))
        x, owner, kind = x[ends], owner[ends], kind[ends]
    return TripMotionProfile(events, intervals, spans, profile, x, owner, kind,
                             segment_profiles, diagnostics, collapsed)
