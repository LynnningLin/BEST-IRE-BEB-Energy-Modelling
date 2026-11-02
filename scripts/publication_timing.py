"""C06 anchor-span ledger and actual-clock adapter over verified motion/physics.

No acquisition, placement generators, ensemble runner or duty builder lives here.
Callers supply versioned local roads, stop decisions, anchors and paired events.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

from beb_soc_model import Segment, VehicleParams, segment_energy_breakdown_kWh
from continuous_interval import solve_interval


TOL = 1e-6


def stable_uniform(namespace, master_seed, *typed_fields):
    """C02 scalar primitive for supplied-event fixtures; full pairing is C08."""
    fields = [namespace, master_seed, *typed_fields]
    if any(type(v) not in (str, int) for v in fields):
        raise ValueError("Stable key fields must be explicit strings or integers")
    data = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(data, digest_size=8).digest(), "big") / 2**64


@dataclass(frozen=True)
class SignalTuple:
    source_ordinal: int
    encounter_id: str
    nominal_context: str
    probability: float
    uniform: float
    stopped: bool
    conditional_wait_s: float
    event_key: tuple

    def __post_init__(self):
        if (self.source_ordinal < 0 or not self.encounter_id or not self.event_key
                or not 0 <= self.probability <= 1 or not 0 <= self.uniform < 1
                or not math.isfinite(self.conditional_wait_s) or self.conditional_wait_s < 0
                or self.stopped != (self.uniform < self.probability)):
            raise ValueError("Invalid supplied source-context signal tuple")

    @property
    def wait_s(self):
        return self.conditional_wait_s if self.stopped else 0.0


@dataclass(frozen=True)
class PlacedSignal:
    position_m: float
    source: SignalTuple


def supplied_placement(source_inventory, sorted_positions_m):
    """Consume ordinal placements supplied by C07/C08; never generate/redraw."""
    if len(source_inventory) != len(sorted_positions_m):
        raise ValueError("Potential encounter count changed")
    if sorted(sorted_positions_m) != list(sorted_positions_m):
        raise ValueError("Supplied positions must be sorted")
    if [s.source_ordinal for s in source_inventory] != list(range(len(source_inventory))):
        raise ValueError("Source inventory must have contiguous canonical ordinals")
    if len({s.encounter_id for s in source_inventory}) != len(source_inventory):
        raise ValueError("Encounter identities must be unique, including repeated visits")
    return tuple(PlacedSignal(float(x), s) for s, x in zip(source_inventory, sorted_positions_m))


@dataclass(frozen=True)
class Road:
    start_m: float
    end_m: float
    cap_ms: float
    desired_ms: float
    grade: float
    source: str
    accepted: bool = False


@dataclass(frozen=True)
class Visit:
    visit_id: str
    position_m: float
    scheduled_arrival_s: float
    scheduled_departure_s: float
    served: bool = True
    recorded_dwell_s: float = 0.0
    reliable_passenger_dwell_s: float | None = None
    authorised_hold: bool = False
    compulsory: bool = False

    def passenger_dwell(self, assumed_s):
        if (self.compulsory or self.authorised_hold or self.recorded_dwell_s > 0) and not self.served:
            raise ValueError("A compulsory/positive-recorded-dwell stop cannot be passed")
        value = assumed_s if self.reliable_passenger_dwell_s is None else self.reliable_passenger_dwell_s
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid passenger dwell")
        return value if self.served else 0.0


@dataclass(frozen=True)
class Anchor:
    position_m: float
    scheduled_s: float
    kind: str
    evidence: str = "labelled terminal timetable fallback; intermediate times soft"

    def __post_init__(self):
        if self.kind not in ("arrival", "departure"):
            raise ValueError("Anchor kind must be arrival or departure")


@dataclass
class Period:
    kind: str
    duration_s: float
    position_m: float
    road_index: int
    visit_id: str | None = None
    profile: object = None
    recoverable: bool = False
    end_m: float | None = None


@dataclass
class SpanPlan:
    start: Anchor
    end: Anchor
    periods: list[Period]
    roads: tuple[Road, ...]
    visits: tuple[Visit, ...]
    signals: tuple[PlacedSignal, ...]
    movement_minimum_s: float
    movement_maximum_s: float
    rule: str
    flags: list[str] = field(default_factory=list)
    boundary_speeds: dict = field(default_factory=dict)


@dataclass(frozen=True)
class State:
    clock_s: float
    stored_kWh: float
    capacity_kWh: float


def independent_start(scheduled_departure_s, vehicle, initial_soc_fraction=0.9):
    if not 0 <= initial_soc_fraction <= 1:
        raise ValueError("Invalid initial state")
    return State(scheduled_departure_s, vehicle.battery_usable_kWh * initial_soc_fraction,
                 vehicle.battery_usable_kWh)


def bounded_allocation(lengths, minima, maxima, budget):
    """Exact length-weighted water filling; return residual instead of inventing hold."""
    if len(lengths) != len(minima) or len(minima) != len(maxima):
        raise ValueError("Allocation dimensions differ")
    if any(l <= 0 or m < 0 or u < m for l, m, u in zip(lengths, minima, maxima)):
        raise ValueError("Invalid nonstop bounds")
    assigned = list(minima)
    remaining = max(budget - sum(minima), 0.0)
    for _ in range(len(lengths) + 1):
        active = [i for i in range(len(lengths)) if maxima[i] - assigned[i] > 1e-10]
        if not active or remaining < 1e-10:
            break
        total_length = sum(lengths[i] for i in active)
        additions = {i: min(remaining * lengths[i] / total_length,
                            maxima[i] - assigned[i]) for i in active}
        for i, extra in additions.items():
            assigned[i] += extra
        remaining = max(remaining - sum(additions.values()), 0.0)
    return assigned, remaining


def desired_speed(length_m, scheduled_runtime_s, cap_ms):
    target = (length_m / scheduled_runtime_s * 1.15
              if scheduled_runtime_s is not None and scheduled_runtime_s > 0 else 11.0)
    return min(cap_ms, min(22.0, max(3.0, target)))


def _owns_visit(visit, start, end):
    return ((visit.position_m > start.position_m or start.kind == "arrival")
            and (visit.position_m < end.position_m or end.kind == "departure"))


def plan_span(roads, visits, signals, start, end, *, rule="stretch", assumed_dwell_s=20.0,
              accel_ms2=1.0, decel_ms2=1.2, include_terminal_signals=False):
    """Freeze a nominal plan before consuming incoming lateness.

    Supplied signals belong to [start,end), plus final-trip endpoint if requested.
    Road boundaries and passed GTFS stops share reachable continuous speeds;
    served stops and stopped signals force zero. Dwell/waits remain sequential.
    """
    roads, visits, signals = tuple(roads), tuple(visits), tuple(signals)
    if rule not in ("stretch", "desired_holding"):
        raise ValueError("Unregistered timing rule")
    if end.scheduled_s <= start.scheduled_s or end.position_m < start.position_m:
        raise ValueError("Incompatible/nonpositive anchor span")
    if accel_ms2 <= 0 or decel_ms2 <= 0:
        raise ValueError("Invalid dynamics")
    if (len({v.visit_id for v in visits}) != len(visits)
            or len({v.position_m for v in visits}) != len(visits)):
        raise ValueError("Duplicate visit identity/chainage needs explicit decomposition")
    for v in visits:
        v.passenger_dwell(assumed_dwell_s)
    if end.position_m > start.position_m:
        if (not roads or roads[0].start_m != start.position_m or roads[-1].end_m != end.position_m
                or any(a.end_m != b.start_m for a, b in zip(roads, roads[1:]))):
            raise ValueError("Complete contiguous local roads required for this span")
    for r in roads:
        if (not r.accepted or not r.source or r.end_m <= r.start_m
                or not all(math.isfinite(x) for x in (r.start_m, r.end_m, r.cap_ms, r.desired_ms, r.grade))
                or min(r.cap_ms, r.desired_ms) <= 0 or r.desired_ms > r.cap_ms):
            raise ValueError("Missing/unaccepted local physical input; no grade/cap fallback")
    for anchor in (start, end):
        if not any(v.position_m == anchor.position_m and v.served for v in visits):
            raise ValueError("An enforced anchor must identify a served visit")
    if any(v.position_m < start.position_m or v.position_m > end.position_m for v in visits):
        raise ValueError("Visit outside anchor span")
    if len({s.source.encounter_id for s in signals}) != len(signals):
        raise ValueError("Duplicated supplied encounter")
    if len({s.source.source_ordinal for s in signals}) != len(signals):
        raise ValueError("Duplicated source ordinal")
    if any(not math.isfinite(s.position_m) or not
           (start.position_m <= s.position_m < end.position_m
            or include_terminal_signals and s.position_m == end.position_m) for s in signals):
        raise ValueError("Signal outside its half-open anchor ownership")
    road_ends = [r.end_m for r in roads]

    def road_index(position):
        return min(bisect.bisect_right(road_ends, position), len(roads) - 1) if roads else -1

    positions = sorted({start.position_m, end.position_m,
                        *(r.start_m for r in roads), *(r.end_m for r in roads),
                        *(v.position_m for v in visits),
                        *(s.position_m for s in signals if s.source.stopped)})
    speeds = []
    for x in positions:
        zero = (x in (start.position_m, end.position_m)
                or any(v.position_m == x and v.served for v in visits)
                or any(s.position_m == x and s.source.stopped for s in signals))
        adjacent = [r for r in roads if r.start_m <= x <= r.end_m]
        speeds.append(0.0 if zero else min(min(r.cap_ms, r.desired_ms) for r in adjacent))
    for i in range(1, len(positions)):
        speeds[i] = min(speeds[i], math.sqrt(speeds[i - 1]**2
                                           + 2 * accel_ms2 * (positions[i] - positions[i - 1])))
    for i in range(len(positions) - 2, -1, -1):
        speeds[i] = min(speeds[i], math.sqrt(speeds[i + 1]**2
                                           + 2 * decel_ms2 * (positions[i + 1] - positions[i])))
    specs, minima, maxima, lengths = [], [], [], []
    for i, (a, b) in enumerate(zip(positions, positions[1:])):
        owner = road_index((a + b) / 2)
        r = roads[owner]
        spec = (b - a, speeds[i], speeds[i + 1], r.cap_ms, accel_ms2, decel_ms2)
        fastest = solve_interval(*spec, target_ms=r.cap_ms)
        specs.append((a, b, owner, spec))
        lengths.append(b - a)
        minima.append(fastest.minimum_time_s)
        maxima.append(fastest.maximum_nonstop_time_s)
    owned_visits = [v for v in visits if _owns_visit(v, start, end)]
    holds = [v for v in owned_visits if v.served and v.authorised_hold]
    hold_visit = max(holds, key=lambda v: v.position_m) if holds else None
    dwell = sum(v.passenger_dwell(assumed_dwell_s) for v in owned_visits)
    waits = sum(s.source.wait_s for s in signals)
    budget = end.scheduled_s - start.scheduled_s - dwell - waits
    flags = []
    if any(v.recorded_dwell_s > 0 and v.reliable_passenger_dwell_s is None for v in visits):
        flags.append("positive_recorded_interval_decomposition_assumed_no_hold_permission_inferred")
    use_stretch = rule == "stretch"
    if rule == "desired_holding" and hold_visit is None:
        use_stretch = True
        flags.append("desired_holding_fallback_stretch_no_authorised_hold")
    if budget < sum(minima) - TOL:
        times, residual = list(minima), 0.0
        flags.append("minimum_motion_dwell_wait_exceeds_anchor")
    elif use_stretch:
        times, residual = bounded_allocation(lengths, minima, maxima, budget)
    else:
        def desired_profiles(lam):
            return [solve_interval(*spec, target_ms=roads[owner].desired_ms + lam *
                                   (roads[owner].cap_ms - roads[owner].desired_ms))
                    for _, _, owner, spec in specs]
        profiles = desired_profiles(0.0)
        if sum(p.duration_s for p in profiles) > budget:
            low, high = 0.0, 1.0
            for _ in range(65):
                mid = (low + high) / 2
                if sum(p.duration_s for p in desired_profiles(mid)) > budget:
                    low = mid
                else:
                    high = mid
            profiles = desired_profiles(high)
        times = [p.duration_s for p in profiles]
        residual = max(budget - sum(times), 0.0)
    if residual > TOL and hold_visit is None:
        flags.append("incompatible_spare_time_no_authorised_hold")
    periods = []
    for i, x in enumerate(positions):
        owner = road_index(x)
        for signal in sorted((s for s in signals if s.position_m == x),
                             key=lambda s: s.source.source_ordinal):
            if signal.source.wait_s:
                periods.append(Period("signal_wait", signal.source.wait_s, x, owner))
        for visit in (v for v in owned_visits if v.position_m == x):
            duration = visit.passenger_dwell(assumed_dwell_s)
            if duration:
                periods.append(Period("passenger_dwell", duration, x, owner, visit.visit_id))
            if visit == hold_visit and residual > 0:
                periods.append(Period("other_holding", residual, x, owner,
                                      visit.visit_id, recoverable=True))
        if i < len(specs):
            a, b, owner, spec = specs[i]
            profile = solve_interval(*spec, scheduled_time_s=times[i])
            if (abs(profile.distance_integral_m - (b - a)) > 1e-6
                    or abs(profile.duration_s - times[i]) > TOL
                    or profile.timing_status != "feasible"):
                raise ValueError("Motion solver did not honour bounded allocation")
            periods.append(Period("movement", profile.duration_s, a, owner,
                                  profile=profile, end_m=b))
    return SpanPlan(start, end, periods, roads, visits, signals, sum(minima), sum(maxima),
                    rule, flags, dict(zip(positions, speeds)))


def _charge_period(period, road, state, vehicle, passengers, aux_power_kW, dt_s):
    """Reuse battery-side force/conversion primitive, then apply exact headroom.

    Request potential recovery without the legacy SoC gate; enforce the C02
    min(R, capacity-stored+draw) rule at every numerical sample at full precision.
    """
    if period.kind == "movement":
        _, velocity, acceleration, steps = period.profile.sample(dt_s)
    else:
        velocity, acceleration, steps = [0.0], [0.0], [period.duration_s]
    seg = Segment(1.0, grade=road.grade if road else 0.0,
                  passengers=passengers, dwell_s=0.0)
    seg.aux_power_kW = aux_power_kW
    stored, traction, aux, regen = state.stored_kWh, 0.0, 0.0, 0.0
    for v, a, step in zip(velocity, acceleration, steps):
        part = segment_energy_breakdown_kWh(seg, vehicle, soc_start_pct=None,
                    profile_and_diagnostics=(([0.0], [v], [a], [step]), {}))
        draw = part.gross_consumed_kWh + part.aux_energy_kWh
        accepted = min(part.regen_recovered_kWh, max(state.capacity_kWh - stored, 0.0) + draw)
        stored += accepted - draw
        traction += part.gross_consumed_kWh
        aux += part.aux_energy_kWh
        regen += accepted
    return State(state.clock_s + period.duration_s, stored, state.capacity_kWh), {
        "traction_kWh": traction, "auxiliary_kWh": aux, "accepted_regen_kWh": regen,
        "net_kWh": traction + aux - regen}


def run_span(plan, state, vehicle, *, passengers=20, aux_power_kW=3.0, dt_s=0.5):
    """Consume a frozen nominal plan, recovering lateness from holding only."""
    if (not all(math.isfinite(x) for x in (dt_s, aux_power_kW, passengers, state.clock_s,
                                          state.stored_kWh, state.capacity_kWh))
            or dt_s <= 0 or aux_power_kW < 0 or passengers < 0 or state.capacity_kWh <= 0
            or state.stored_kWh > state.capacity_kWh + 1e-9
            or not math.isclose(state.capacity_kWh, vehicle.battery_usable_kWh)):
        raise ValueError("Invalid numerical, load or battery state")
    incoming = state.clock_s - plan.start.scheduled_s
    if incoming < -TOL:
        raise ValueError("Early incoming clock requires an explicitly represented permitted hold")
    incoming = max(incoming, 0.0)
    recovery_left = incoming
    initial = state
    ledger = {k: 0.0 for k in ("movement", "passenger_dwell", "signal_wait", "other_holding")}
    energy = {k: 0.0 for k in ("traction_kWh", "auxiliary_kWh", "accepted_regen_kWh", "net_kWh")}
    actual_periods, visits = [], []
    for period in plan.periods:
        duration = period.duration_s
        if period.recoverable:
            removed = min(recovery_left, duration)
            recovery_left -= removed
            duration -= removed
        actual = Period(**{**period.__dict__, "duration_s": duration})
        begin = state.clock_s
        state, delta = _charge_period(actual,
            plan.roads[period.road_index] if plan.roads else None,
            state, vehicle, passengers, aux_power_kW, dt_s)
        ledger[period.kind] += duration
        for k in energy:
            energy[k] += delta[k]
        actual_periods.append({"kind": period.kind, "start_s": begin, "end_s": state.clock_s,
                               "position_m": period.position_m, "end_m": period.end_m,
                               "road_index": period.road_index, "visit_id": period.visit_id})
    elapsed = state.clock_s - initial.clock_s
    if abs(elapsed - sum(ledger.values())) > TOL:
        raise AssertionError("Actual clock and ledger do not reconcile")
    if abs(energy["auxiliary_kWh"] - aux_power_kW * elapsed / 3600) > 1e-9:
        raise AssertionError("Constant-power auxiliary energy does not reconcile")
    for visit in plan.visits:
        at = [p for p in actual_periods if p["position_m"] == visit.position_m]
        incoming_motion = [p for p in actual_periods if p["kind"] == "movement"
                           and p["end_m"] == visit.position_m]
        # Arrival follows a coincident control wait, before passenger dwell/hold.
        signal_ends = [p["end_s"] for p in at if p["kind"] == "signal_wait"]
        arrival = (max(signal_ends) if signal_ends else incoming_motion[-1]["end_s"]
                   if incoming_motion else initial.clock_s)
        owned = [p["end_s"] for p in at if p["kind"] in ("passenger_dwell", "other_holding")]
        departure = max(owned) if owned else arrival
        visits.append({"visit_id": visit.visit_id, "served": visit.served,
                       "speed_ms": plan.boundary_speeds.get(visit.position_m),
                       "arrival_s": arrival, "departure_s": departure,
                       "arrival_lateness_s": max(arrival - visit.scheduled_arrival_s, 0.0),
                       "departure_lateness_s": max(departure - visit.scheduled_departure_s, 0.0),
                       "arrival_represented": not (visit.position_m == plan.start.position_m
                                                   and plan.start.kind == "departure"),
                       "departure_represented": not (visit.position_m == plan.end.position_m
                                                     and plan.end.kind == "arrival")})
    flags = list(plan.flags)
    early_departure = plan.end.kind == "departure" and state.clock_s < plan.end.scheduled_s - TOL
    if early_departure:
        flags.append("enforced_departure_unavailable_no_authorised_hold")
    if state.stored_kWh < 0:
        flags.append("battery_below_zero_diagnostic_trace")
    return state, {"nominal_duration_s": plan.end.scheduled_s - plan.start.scheduled_s,
                   "elapsed_s": elapsed, **{k + "_s": v for k, v in ledger.items()}, **energy,
                   "distance_m": sum(p.profile.length_m for p in plan.periods if p.profile),
                   "incoming_lateness_s": incoming,
                   "outgoing_lateness_s": max(state.clock_s - plan.end.scheduled_s, 0.0),
                   "recovered_holding_s": incoming - recovery_left,
                   "initial_state": asdict(initial), "final_state": asdict(state),
                   "flags": flags, "failure": early_departure,
                   "visits": visits, "periods": actual_periods,
                   "signal_tuples": [asdict(s) for s in plan.signals]}


def run_constructed_gap(state, scheduled_start_s, gap_s, necessary_s, vehicle,
                        *, aux_power_kW, operating_state):
    """Constructed stationary Q + authorised layover only; real duties are C16."""
    if (min(gap_s, necessary_s, aux_power_kW) < 0 or not operating_state
            or state.clock_s < scheduled_start_s - TOL):
        raise ValueError("Gap requires a declared state and non-early incoming clock")
    incoming = max(state.clock_s - scheduled_start_s, 0.0)
    planned_hold = max(gap_s - necessary_s, 0.0)
    actual_hold = max(planned_hold - incoming, 0.0)
    elapsed = necessary_s + actual_hold
    energy = aux_power_kW * elapsed / 3600
    result = State(state.clock_s + elapsed, state.stored_kWh - energy, state.capacity_kWh)
    return result, {"analysis_mode": "duty_continuous_constructed_fixture",
                    "operating_state": operating_state, "gap_s": gap_s,
                    "necessary_stationary_time_s": necessary_s,
                    "actual_layover_s": actual_hold, "elapsed_s": elapsed,
                    "movement_s": 0.0, "passenger_dwell_s": 0.0, "signal_wait_s": 0.0,
                    "other_holding_s": elapsed, "auxiliary_kWh": energy,
                    "outgoing_lateness_s": max(result.clock_s - scheduled_start_s - gap_s, 0.0),
                    "initial_state": asdict(state), "final_state": asdict(result)}


def run_trip(plans, vehicle, *, analysis_mode="standalone_trip", state=None,
             initial_soc_fraction=0.9, passengers=20, aux_power_kW=3.0, dt_s=0.5):
    """Trip-level ledger; explicit continuous state is never reset implicitly."""
    plans = list(plans)
    if not plans or plans[0].start.kind != "departure" or plans[-1].end.kind != "arrival":
        raise ValueError("Trip reporting must span first departure to final arrival")
    if analysis_mode == "standalone_trip":
        if state is not None:
            raise ValueError("Standalone trips must initialise independently")
        state = independent_start(plans[0].start.scheduled_s, vehicle, initial_soc_fraction)
    elif analysis_mode != "duty_continuous" or state is None:
        raise ValueError("Continuous trips require an explicit carried state")
    initial = state
    spans = []
    for i, plan in enumerate(plans):
        if i and (plan.start != plans[i - 1].end):
            raise ValueError("Anchor spans must join exactly without hidden gaps")
        state, result = run_span(plan, state, vehicle, passengers=passengers,
                                aux_power_kW=aux_power_kW, dt_s=dt_s)
        spans.append(result)
        if result["failure"]:
            break
    sums = {key: sum(r[key] for r in spans) for key in (
        "elapsed_s", "movement_s", "passenger_dwell_s", "signal_wait_s", "other_holding_s",
        "distance_m", "traction_kWh", "auxiliary_kWh", "accepted_regen_kWh", "net_kWh")}
    return state, {**sums, "analysis_mode": analysis_mode, "initial_state": asdict(initial),
                   "final_state": asdict(state), "incoming_lateness_s": spans[0]["incoming_lateness_s"],
                   "outgoing_lateness_s": spans[-1]["outgoing_lateness_s"],
                   "flags": sorted({flag for r in spans for flag in r["flags"]}),
                   "failure": any(r["failure"] for r in spans),
                   "visits": [v for r in spans for v in r["visits"]], "spans": spans}


def run_controlled_trip(plans, config, *, state=None):
    """Resolve all controlled scalar inputs from one versioned configuration."""
    plans = list(plans)
    if config["demand_profile_enabled"] or config["weather_loading_enabled"]:
        raise ValueError("Controlled trip requires fixed load and auxiliary configuration")
    if config["B_structure"] == "B0" and any(not v.served for p in plans for v in p.visits):
        raise ValueError("B0 requires every GTFS stop served")
    for plan in plans:
        if plan.rule != config["timing_rule"]:
            raise ValueError("Timing scenario differs from registered configuration")
        for period in plan.periods:
            if period.profile and any(
                phase.acceleration_ms2 > config["accel_ms2"] + 1e-9
                or phase.acceleration_ms2 < -config["decel_ms2"] - 1e-9
                for phase in period.profile.phases
            ):
                raise ValueError("Planned dynamics exceed configured bounds")
    return run_trip(plans, VehicleParams(**config["vehicle"]),
        analysis_mode=config["analysis_mode"], state=state,
        initial_soc_fraction=config["initial_soc_fraction"], passengers=config["passengers"],
        aux_power_kW=config["in_service_aux_power_kW"], dt_s=config["dt_s"])
