"""Synthetic signal cases and physics helpers shared by regression tests."""

from __future__ import annotations


from dataclasses import dataclass


import numpy as np


import pandas as pd


from beb_soc_model import (
    MotionParams,
    Segment,
    VehicleParams,
    build_speed_profile,
    segment_energy_breakdown_kWh,
)

LENGTH_M = 1000.0


SCHEDULED_TIME_S = 120.0


DT_S = 0.25


SPEED_CAP_MS = 20.0


ACCEL_MS2 = 1.5


DECEL_MS2 = 1.5


MOTION_PARAMS = MotionParams(
    accel_ms2=ACCEL_MS2,
    decel_ms2=DECEL_MS2,
    dt_s=DT_S,
    default_speed_cap_ms=SPEED_CAP_MS,
    max_speed_cap_ms=25.0,
    stop_prob=1.0,
    red_wait_s=15.0,
    signal_time_policy="preserve_schedule",
    max_signal_wait_share=1.0,
    use_hourly_signal_stop_probability=False,
    stop_prob_by_hour={},
)


VEHICLE_PARAMS = VehicleParams(
    curb_mass_kg=10_000.0,
    passenger_mass_kg=75.0,
    frontal_area_m2=8.0,
    drag_coeff=0.6,
    roll_coeff=0.01,
    rot_inertia_factor=1.0,
    eta_driveline=0.9,
    eta_motor=0.9,
    regen_fraction=0.5,
    regen_power_cap_kW=500.0,
    regen_min_speed_ms=0.0,
    aux_power_kW=6.0,
    battery_usable_kWh=400.0,
    air_density=1.2,
    g=9.81,
)


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    label: str
    stop_positions_m: tuple[float, ...]
    red_wait_s: float = 15.0
    scheduled_time_s: float = SCHEDULED_TIME_S


CASES = [
    ValidationCase("V0", "No signal stop", ()),
    ValidationCase("V1", "One central stop", (500.0,)),
    ValidationCase("V2", "Two separated stops", (250.0, 750.0)),
    ValidationCase("V3", "Two closely spaced stops", (450.0, 500.0)),
    ValidationCase("V4", "Higher burden, feasible", (250.0, 500.0, 750.0), 5.0),
    ValidationCase("V5", "Schedule infeasible", (200.0, 400.0, 600.0, 800.0)),
]


def signal_event(event_id: str, distance_m: float, length_m: float = LENGTH_M) -> dict:
    return {
        "signal_event_id": event_id,
        "segment_geometry_id": "synthetic|a|b",
        "from_stop_id": "a",
        "to_stop_id": "b",
        "shape_id": "synthetic_shape",
        "distance_along_segment_m": float(distance_m),
        "segment_length_m": float(length_m),
        "signal_lat": 51.0,
        "signal_lon": -8.0,
        "source": "synthetic_validation",
    }


def build_validation_segment(case: ValidationCase) -> Segment:
    events = [
        signal_event(f"{case.case_id}_signal_{i}", pos)
        for i, pos in enumerate(case.stop_positions_m, start=1)
    ]
    return Segment(
        length_m=LENGTH_M,
        grade=0.0,
        v_cruise_ms=12.0,
        dwell_s=0.0,
        passengers=20,
        run_time_s=case.scheduled_time_s,
        from_stop_departure_time="08:00:00",
        to_stop_arrival_time="08:02:00",
        n_signals=len(events),
        signal_source="synthetic_validation",
        segment_geometry_id="synthetic|a|b",
        shape_id="synthetic_shape",
        trip_id=f"trip_{case.case_id}",
        service_date="2025-07-09",
        segment_index=0,
        signal_events=events,
    )


def motion_params_for_case(case: ValidationCase) -> MotionParams:
    return MotionParams(
        accel_ms2=MOTION_PARAMS.accel_ms2,
        decel_ms2=MOTION_PARAMS.decel_ms2,
        dt_s=MOTION_PARAMS.dt_s,
        default_speed_cap_ms=MOTION_PARAMS.default_speed_cap_ms,
        max_speed_cap_ms=MOTION_PARAMS.max_speed_cap_ms,
        stop_prob=1.0,
        red_wait_s=case.red_wait_s,
        signal_time_policy="preserve_schedule",
        max_signal_wait_share=1.0,
        use_hourly_signal_stop_probability=False,
        stop_prob_by_hour={},
    )


def integrated_distance_m(profile) -> float:
    _t, v, a, step = profile
    return float(np.sum(v * step + 0.5 * a * step**2))


def cumulative_distance_before_steps(profile) -> np.ndarray:
    _t, v, a, step = profile
    step_distance = v * step + 0.5 * a * step**2
    return np.r_[0.0, np.cumsum(step_distance)[:-1]]


def idle_stop_positions_m(profile, min_idle_s: float = 1.0) -> list[float]:
    _t, v, a, step = profile
    if len(step) == 0:
        return []
    before = cumulative_distance_before_steps(profile)
    idle = (np.abs(v) < 1e-9) & (np.abs(a) < 1e-9)
    positions = []
    i = 0
    while i < len(step):
        if not idle[i]:
            i += 1
            continue
        j = i
        idle_duration = 0.0
        while j < len(step) and idle[j]:
            idle_duration += float(step[j])
            j += 1
        if idle_duration >= min_idle_s:
            positions.append(float(before[i]))
        i = j
    return positions


def braking_mechanical_kWh(profile, seg: Segment, p: VehicleParams) -> float:
    _t, v, a, step = profile
    mass_kg = p.curb_mass_kg + seg.passengers * p.passenger_mass_kg
    theta = np.arctan(seg.grade)
    braking_j = 0.0
    for vi, ai, dti in zip(v, a, step):
        moving = vi > 0.01
        f_roll = p.roll_coeff * mass_kg * p.g * np.cos(theta) * moving
        f_aero = 0.5 * p.air_density * p.drag_coeff * p.frontal_area_m2 * vi**2
        f_grade = mass_kg * p.g * np.sin(theta) * moving
        f_inert = mass_kg * p.rot_inertia_factor * ai
        p_wheel = (f_roll + f_aero + f_grade + f_inert) * vi
        if p_wheel < 0.0:
            braking_j += -p_wheel * dti
    return braking_j / 3.6e6


def required_average_moving_speed_ms(diag: dict) -> float | None:
    budget = diag.get("moving_time_budget_s")
    if budget is None or budget <= 0:
        return None
    return LENGTH_M / float(budget)


def run_case(case: ValidationCase) -> dict:
    seg = build_validation_segment(case)
    mp = motion_params_for_case(case)
    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
    energy = segment_energy_breakdown_kWh(
        seg, VEHICLE_PARAMS, soc_start_pct=50.0, motion_params=mp
    )
    _t, v, a, _step = profile
    expected_feasible = (
        diag["minimum_feasible_moving_time_s"] <= diag["moving_time_budget_s"] + 1e-9
        if diag["moving_time_budget_s"] is not None
        else False
    )
    return {
        "case": case.case_id,
        "label": case.label,
        "stops": int(diag["n_signal_stops"]),
        "stop_positions": ",".join(f"{p:.0f}" for p in case.stop_positions_m) or "-",
        "wait_s": float(diag["signal_wait_applied_s"]),
        "move_budget_s": diag["moving_time_budget_s"],
        "min_move_s": float(diag["minimum_feasible_moving_time_s"]),
        "actual_move_s": float(diag["actual_moving_time_s"]),
        "actual_segment_s": float(diag["actual_segment_time_s"]),
        "max_speed_ms": float(np.max(v)) if len(v) else 0.0,
        "max_accel_ms2": float(np.max(a)) if len(a) else 0.0,
        "max_decel_ms2": float(-np.min(a)) if len(a) else 0.0,
        "required_avg_move_speed_ms": required_average_moving_speed_ms(diag),
        "delay_s": float(diag["schedule_delay_s"]),
        "traction_kWh": float(energy.gross_consumed_kWh),
        "regen_kWh": float(energy.regen_recovered_kWh),
        "aux_hvac_kWh": float(energy.aux_energy_kWh),
        "net_kWh": float(energy.net_battery_energy_kWh),
        "integrated_distance_m": integrated_distance_m(profile),
        "idle_stop_positions_m": idle_stop_positions_m(profile),
        "braking_mechanical_kWh": braking_mechanical_kWh(profile, seg, VEHICLE_PARAMS),
        "schedule_feasible_by_budget": expected_feasible,
        "signal_event_mode": diag["signal_event_mode"],
        "profile": profile,
        "diag": diag,
    }


def transition_sweep(max_stops: int = 6) -> pd.DataFrame:
    rows = []
    for n_stops in range(max_stops + 1):
        spacing = LENGTH_M / (n_stops + 1)
        positions = tuple(spacing * i for i in range(1, n_stops + 1))
        row = run_case(
            ValidationCase(
                f"S{n_stops}",
                f"{n_stops} evenly spaced stops",
                positions,
                15.0,
                SCHEDULED_TIME_S,
            )
        )
        rows.append(
            {
                "n_signal_stops": row["stops"],
                "signal_wait_s": row["wait_s"],
                "required_avg_move_speed_ms": row["required_avg_move_speed_ms"],
                "maximum_speed_ms": row["max_speed_ms"],
                "actual_moving_time_s": row["actual_move_s"],
                "actual_segment_time_s": row["actual_segment_s"],
                "schedule_delay_s": row["delay_s"],
            }
        )
    return pd.DataFrame(rows)


def spatial_comparison_cases() -> list[ValidationCase]:
    return [
        ValidationCase("A", "Evenly separated", (333.333, 666.667)),
        ValidationCase("B", "Clustered", (450.0, 500.0)),
        ValidationCase("C", "Near ends", (50.0, 950.0)),
    ]
