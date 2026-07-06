"""
beb_soc_model.py
================================================================================
Segment-level (stop-to-stop) Battery Electric Bus energy & State-of-Charge model.

WHAT THIS IS
------------
A *quasi-static, backward-facing* longitudinal vehicle model. "Backward-facing"
means we assume the bus follows a known speed profile and we work *backwards* to
the power the battery must supply -- no driver/controller model needed, so it is
fast, deterministic, and easy to run thousands of times inside an optimisation.

It does the two jobs we talked about:
  1. ROUTE -> MOTION : turn each stop-to-stop segment (length, grade, dwell) into
                       a speed-vs-time profile.
  2. MOTION -> ENERGY -> SoC : turn that motion into a battery power demand, then
                       deplete the battery and track State of Charge.

The physics is just a force balance at the wheels:

    F_traction = F_rolling + F_aero + F_grade + F_inertia

    F_rolling  = Crr * m * g * cos(theta)        (tyres)
    F_aero     = 0.5 * rho * Cd * A * v^2         (air drag)
    F_grade    = m * g * sin(theta)               (the hill)
    F_inertia  = m * lambda * a                    (speeding up / slowing down)

Wheel power P = F_traction * v. We then convert to battery power through the
drivetrain/motor efficiency (when accelerating) or recover part of it through
regenerative braking (when slowing down), add a constant auxiliary/HVAC load,
and integrate to get energy. SoC is energy-based Coulomb counting:

    SoC(t) = SoC0 - (cumulative battery energy / usable capacity) * 100

References for the approach (for your lit review):
  - Kunith et al. (2017), segment-level energy feeding charger-placement MILP.
  - NREL FASTSim, a validated backward-facing vehicle energy model.

HOW TO PLUG IN REAL DATA
------------------------
Replace make_synthetic_route() with a function that returns a list of Segment
objects built from your GTFS feed (segment length, scheduled speed, dwell) and a
DEM (average grade per segment). Nothing else needs to change. Swapping synthetic
-> real data is just swapping the input list.
================================================================================
"""

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; we save figures to file
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from best_ire_beb.config import get_path, vehicle_params


# ----------------------------------------------------------------------------- 
# 1. PARAMETERS
# -----------------------------------------------------------------------------
_VEHICLE_DEFAULTS = vehicle_params()


@dataclass
class VehicleParams:
    curb_mass_kg: float = _VEHICLE_DEFAULTS["curb_mass_kg"]
    passenger_mass_kg: float = _VEHICLE_DEFAULTS["passenger_mass_kg"]
    frontal_area_m2: float = _VEHICLE_DEFAULTS["frontal_area_m2"]
    drag_coeff: float = _VEHICLE_DEFAULTS["drag_coeff"]
    roll_coeff: float = _VEHICLE_DEFAULTS["roll_coeff"]
    rot_inertia_factor: float = _VEHICLE_DEFAULTS["rot_inertia_factor"]
    eta_driveline: float = _VEHICLE_DEFAULTS["eta_driveline"]
    eta_motor: float = _VEHICLE_DEFAULTS["eta_motor"]
    regen_fraction: float = _VEHICLE_DEFAULTS["regen_fraction"]
    regen_power_cap_kW: float = _VEHICLE_DEFAULTS["regen_power_cap_kW"]
    regen_min_speed_ms: float = _VEHICLE_DEFAULTS["regen_min_speed_ms"]
    aux_power_kW: float = _VEHICLE_DEFAULTS["aux_power_kW"]
    battery_usable_kWh: float = _VEHICLE_DEFAULTS["battery_usable_kWh"]
    air_density: float = _VEHICLE_DEFAULTS["air_density"]
    g: float = _VEHICLE_DEFAULTS["g"]

    @classmethod
    def from_config(cls, config_path=None):
        return cls(**vehicle_params(config_path))


@dataclass
class Segment:
    """One stop-to-stop link."""
    length_m: float          # distance to the next stop
    grade: float = 0.0       # rise/run as a fraction (0.03 = +3% uphill)
    v_cruise_ms: float = 11.0  # free-flow cruising speed (m/s). 11 m/s ~ 40 km/h
    dwell_s: float = 20.0    # time stopped at the *end* stop (doors open)
    passengers: int = 20     # average occupancy on this segment
    run_time_s: Optional[float] = None  # scheduled driving time stop-to-stop
    from_stop_departure_time: Optional[str] = None
    to_stop_arrival_time: Optional[str] = None
    to_stop_departure_time: Optional[str] = None
    n_signals: int = 0       # OSM traffic signals on this segment (traffic_signals.py)
    signal_source: Optional[str] = None  # "osm" | "fallback" | "none"


@dataclass
class SegmentEnergy:
    """Battery-side energy accounting for one segment, in kWh."""
    net_battery_energy_kWh: float
    gross_consumed_kWh: float
    regen_recovered_kWh: float
    aux_energy_kWh: float
    motion_diagnostics: Optional[dict] = None


# ----------------------------------------------------------------------------- 
# 2. ROUTE -> MOTION : build a speed profile for one segment
# -----------------------------------------------------------------------------
# Default stop-injection behaviour when a segment carries traffic signals.
#
# Important modelling choice:
# GTFS run_time_s is a scheduled stop-to-stop time. It normally already contains
# average delays from junctions, pedestrian crossings, and normal traffic.
# Therefore, when run_time_s exists, red-light waiting should be treated as an
# internal part of that scheduled time, not as extra time added on top. Otherwise
# the signal mode mainly increases auxiliary/HVAC energy rather than traction.
DEFAULT_STOP_PROB = 0.5       # probability that one signal forces a full stop
DEFAULT_RED_WAIT_S = 20.0     # seconds of idle wait per actual signal stop
DEFAULT_SIGNAL_TIME_POLICY = "preserve_schedule"  # or "add_delay"
DEFAULT_MAX_SIGNAL_WAIT_SHARE = 0.35              # cap idle share of run_time_s


def _sample_profile(t_acc, t_cruise, t_dec, v_peak, a_eff, d_eff, dt):
    """Sample one accel -> cruise -> decel phase set into (t, v, a, step_s)."""
    total_t = max(float(t_acc + t_cruise + t_dec), 0.0)
    if total_t <= 0.0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=float), np.array([], dtype=float))
    t = np.arange(0, total_t, dt)
    step_s = np.minimum(dt, total_t - t)
    v = np.zeros_like(t)
    a = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < t_acc:                       # accelerating
            a[i] = a_eff
            v[i] = a_eff * ti
        elif ti < t_acc + t_cruise:          # cruising
            a[i] = 0.0
            v[i] = v_peak
        else:                                # decelerating
            a[i] = -d_eff
            v[i] = v_peak - d_eff * (ti - t_acc - t_cruise)
    v = np.clip(v, 0.0, None)
    return t, v, a, step_s


def _freeflow_phases(length_m, v_c, a_accel, a_decel):
    """Return accel/cruise/decel phases for the fastest feasible profile at v_c."""
    length_m = max(float(length_m), 0.0)
    v_c = max(float(v_c), 1e-6)
    d_acc = v_c**2 / (2 * a_accel)
    d_dec = v_c**2 / (2 * a_decel)
    if d_acc + d_dec <= length_m:
        t_acc = v_c / a_accel
        t_cruise = (length_m - d_acc - d_dec) / v_c
        t_dec = v_c / a_decel
        v_peak = v_c
    else:                                    # too short to reach v_c: triangular
        v_peak = np.sqrt(2 * length_m * a_accel * a_decel / (a_accel + a_decel))
        t_acc = v_peak / a_accel
        t_cruise = 0.0
        t_dec = v_peak / a_decel
    return t_acc, t_cruise, t_dec, v_peak


def _freeflow_duration(length_m, v_c, a_accel, a_decel):
    """Minimum feasible duration over length_m under accel/decel and speed cap."""
    t_acc, t_cruise, t_dec, _v = _freeflow_phases(
        length_m, v_c, a_accel, a_decel
    )
    return t_acc + t_cruise + t_dec


def _freeflow_profile(length_m, v_c, a_accel, a_decel, dt):
    """Free-flow accel -> (cruise at v_c) -> decel over length_m."""
    t_acc, t_cruise, t_dec, v_peak = _freeflow_phases(
        length_m, v_c, a_accel, a_decel
    )
    return _sample_profile(t_acc, t_cruise, t_dec, v_peak, a_accel, a_decel, dt)


def _profile_duration(profile):
    """Duration represented by a sampled profile tuple."""
    if profile is None or len(profile) < 4 or len(profile[3]) == 0:
        return 0.0
    return float(np.sum(profile[3]))


def _single_segment_profile(length_m, run_time_s, v_cruise_ms,
                            a_accel=1.0, a_decel=1.2, dt=0.5, v_cap=None):
    """
    One accelerate -> (cruise) -> decelerate-to-stop profile over length_m.

    If run_time_s is feasible, the profile duration equals run_time_s. If a
    speed cap is supplied and the requested time is physically infeasible, the
    fastest feasible capped profile is returned instead. The caller records the
    resulting delay in diagnostics.
    """
    if a_accel <= 0 or a_decel <= 0 or dt <= 0:
        raise ValueError("a_accel, a_decel, and dt must be positive.")

    v_limit = v_cruise_ms if v_cap is None else min(float(v_cruise_ms), float(v_cap))
    v_limit = max(float(v_limit), 1e-6)

    if run_time_s is not None and run_time_s > 0:
        total_t = float(run_time_s)
        min_t = _freeflow_duration(length_m, v_limit, a_accel, a_decel)

        # With a cap, never invent speeds/accelerations that are faster than the
        # feasible minimum. Return the minimum-time profile and let diagnostics
        # record schedule delay.
        if v_cap is not None and total_t < min_t - 1e-9:
            return _freeflow_profile(length_m, v_limit, a_accel, a_decel, dt)

        inv_accel_sum = (1.0 / a_accel) + (1.0 / a_decel)
        triangular_t = np.sqrt(2.0 * length_m * inv_accel_sum)
        if total_t >= triangular_t:
            curve = 0.5 * inv_accel_sum
            disc = max(total_t**2 - 4.0 * curve * length_m, 0.0)
            v_peak = (total_t - np.sqrt(disc)) / (2.0 * curve)
            a_eff, d_eff = a_accel, a_decel
            t_acc = v_peak / a_eff
            t_dec = v_peak / d_eff
            t_cruise = max(total_t - t_acc - t_dec, 0.0)
        else:
            # Legacy fallback when no speed cap is enforced: this compresses the
            # profile by using higher effective acceleration/deceleration.
            scale = 2.0 * length_m * inv_accel_sum / total_t**2
            a_eff = a_accel * scale
            d_eff = a_decel * scale
            v_peak = 2.0 * length_m / total_t
            t_acc = v_peak / a_eff
            t_dec = v_peak / d_eff
            t_cruise = 0.0

        if v_cap is not None and v_peak > v_limit + 1e-9:
            return _freeflow_profile(length_m, v_limit, a_accel, a_decel, dt)
        return _sample_profile(t_acc, t_cruise, t_dec, v_peak, a_eff, d_eff, dt)

    return _freeflow_profile(length_m, v_limit, a_accel, a_decel, dt)


def _stable_uniform01(*parts) -> float:
    """Repeatable pseudo-random number in [0, 1) from segment attributes."""
    key = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _signal_stop_count(seg: Segment, n_signals: int, stop_prob: float) -> int:
    """
    Number of actual full stops caused by signalised points.

    The previous half-up rounding rule made every one-signal segment stop when
    stop_prob=0.5. That is biased for suburban/pedestrian-actuated signals. This
    deterministic Bernoulli draw keeps the model reproducible while preserving
    the intended probability across many segments.
    """
    n_signals = max(int(n_signals or 0), 0)
    stop_prob = max(0.0, min(float(stop_prob), 1.0))
    if n_signals == 0 or stop_prob <= 0.0:
        return 0
    if stop_prob >= 1.0:
        return n_signals

    # Use available segment attributes to create a stable segment-specific key.
    # The loop represents each signal as one Bernoulli opportunity.
    base_key = (
        getattr(seg, "from_stop_departure_time", None),
        getattr(seg, "to_stop_arrival_time", None),
        round(float(getattr(seg, "length_m", 0.0) or 0.0), 1),
        n_signals,
    )
    return sum(
        1 for j in range(n_signals)
        if _stable_uniform01(*base_key, j) < stop_prob
    )


def _idle_profile(duration_s, dt):
    """Zero-speed profile block for red-light waiting."""
    duration_s = max(float(duration_s), 0.0)
    if duration_s <= 0.0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=float), np.array([], dtype=float))
    n_idle = max(int(np.ceil(duration_s / dt)), 1)
    idle_steps = np.full(n_idle, dt)
    idle_steps[-1] = max(duration_s - dt * (n_idle - 1), 1e-9)
    return (
        np.r_[0.0, np.cumsum(idle_steps[:-1])],
        np.zeros(n_idle),
        np.zeros(n_idle),
        idle_steps,
    )


def _concat_profile_parts(parts):
    """Concatenate profile parts, adding each part's local time offset."""
    ts, vs, as_, ss = [], [], [], []
    t_off = 0.0
    for t, v, a, step in parts:
        if len(step) == 0:
            continue
        ts.append(t + t_off)
        vs.append(v)
        as_.append(a)
        ss.append(step)
        t_off += float(np.sum(step))
    if not ss:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    return np.concatenate(ts), np.concatenate(vs), np.concatenate(as_), np.concatenate(ss)


def build_speed_profile(seg: Segment, a_accel=1.0, a_decel=1.2, dt=0.5,
                        stop_prob=DEFAULT_STOP_PROB, red_wait_s=DEFAULT_RED_WAIT_S,
                        signal_time_policy=DEFAULT_SIGNAL_TIME_POLICY,
                        max_signal_wait_share=DEFAULT_MAX_SIGNAL_WAIT_SHARE,
                        return_diagnostics=False):
    """
    Return arrays (t, v, a, step_s) for the segment motion profile.

    With traffic signals, the segment is split into extra stop-start sub-links.
    In preserve_schedule mode, GTFS run_time_s remains the target total duration.
    Red-light waiting is treated as internal schedule slack. If the assumed wait
    would make the capped profile infeasible, the model reduces the *modelled*
    signal wait first, instead of adding an extra auxiliary-only idle block on
    top of the GTFS time. Only truly infeasible capped profiles create delay.
    """
    if a_accel <= 0 or a_decel <= 0 or dt <= 0:
        raise ValueError("a_accel, a_decel, and dt must be positive.")

    run_time_s = getattr(seg, "run_time_s", None)
    scheduled = float(run_time_s) if run_time_s is not None and run_time_s > 0 else None
    n_signals = max(int(getattr(seg, "n_signals", 0) or 0), 0)
    n_stops = _signal_stop_count(seg, n_signals, stop_prob)
    policy = str(signal_time_policy or "preserve_schedule")

    diag = {
        "n_signals": n_signals,
        "n_effective_signal_stops": n_stops,
        "signal_stop_prob": float(stop_prob),
        "red_wait_s_assumed": float(red_wait_s),
        "signal_time_policy": policy,
        "scheduled_run_time_s": scheduled,
        "signal_wait_requested_s": 0.0,
        "signal_wait_s": 0.0,
        "signal_wait_reduced_s": 0.0,
        "moving_profile_time_s": 0.0,
        "actual_profile_time_s": 0.0,
        "schedule_delay_s": 0.0,
        "schedule_infeasible": False,
        "speed_cap_ms": None,
        "min_feasible_motion_time_s": None,
        "n_motion_sublinks": 1,
    }

    if n_stops <= 0:
        profile = _single_segment_profile(seg.length_m, scheduled,
                                          seg.v_cruise_ms, a_accel, a_decel,
                                          dt, v_cap=None)
        actual = _profile_duration(profile)
        diag["moving_profile_time_s"] = actual
        diag["actual_profile_time_s"] = actual
        if scheduled is not None:
            diag["schedule_delay_s"] = max(actual - scheduled, 0.0)
            diag["schedule_infeasible"] = diag["schedule_delay_s"] > 1e-6
        return (profile, diag) if return_diagnostics else profile

    n_sub = n_stops + 1
    sub_len = float(seg.length_m) / n_sub
    v_cap = max(float(seg.v_cruise_ms), 1e-6)
    min_sub_time = _freeflow_duration(sub_len, v_cap, a_accel, a_decel)
    min_motion_total = min_sub_time * n_sub
    requested_wait = n_stops * max(float(red_wait_s), 0.0)
    if scheduled is not None:
        requested_wait = min(
            requested_wait,
            max(float(max_signal_wait_share), 0.0) * scheduled,
        )

    diag.update({
        "signal_wait_requested_s": requested_wait,
        "speed_cap_ms": v_cap,
        "min_feasible_motion_time_s": min_motion_total,
        "n_motion_sublinks": n_sub,
    })

    if scheduled is not None and policy == "preserve_schedule":
        if scheduled >= min_motion_total - 1e-9:
            # Preserve the GTFS duration. Red-light wait is only the part of the
            # schedule slack that can physically fit under the speed cap.
            max_wait_that_fits = max(scheduled - min_motion_total, 0.0)
            modeled_wait = min(requested_wait, max_wait_that_fits)
            moving_total = scheduled - modeled_wait
            sub_rt = moving_total / n_sub
            idle_each_s = modeled_wait / n_stops if n_stops > 0 else 0.0
            diag["signal_wait_s"] = modeled_wait
            diag["signal_wait_reduced_s"] = max(requested_wait - modeled_wait, 0.0)
        else:
            # Even zero signal waiting cannot fit the scheduled time with the
            # extra stop-start cycles and speed cap. Use the fastest feasible
            # split profile and expose the true schedule delay.
            sub_rt = None
            idle_each_s = 0.0
            diag["signal_wait_s"] = 0.0
            diag["signal_wait_reduced_s"] = requested_wait
            diag["schedule_infeasible"] = True
    else:
        # Explicit delay-adding mode, mainly for scenario/sensitivity analysis.
        sub_rt = (scheduled / n_sub) if scheduled is not None else None
        idle_each_s = max(float(red_wait_s), 0.0)
        diag["signal_wait_s"] = idle_each_s * n_stops

    parts = []
    moving_time = 0.0
    for k in range(n_sub):
        prof = _single_segment_profile(
            sub_len, sub_rt, seg.v_cruise_ms, a_accel, a_decel, dt,
            v_cap=v_cap
        )
        parts.append(prof)
        moving_time += _profile_duration(prof)
        if k < n_sub - 1 and idle_each_s > 0.0:
            parts.append(_idle_profile(idle_each_s, dt))

    profile = _concat_profile_parts(parts)
    actual = _profile_duration(profile)
    diag["moving_profile_time_s"] = moving_time
    diag["actual_profile_time_s"] = actual
    if scheduled is not None:
        diag["schedule_delay_s"] = max(actual - scheduled, 0.0)
        diag["schedule_infeasible"] = (
            bool(diag["schedule_infeasible"]) or diag["schedule_delay_s"] > 1e-6
        )

    return (profile, diag) if return_diagnostics else profile


# ----------------------------------------------------------------------------- 
# 3. MOTION -> ENERGY : battery energy over one segment
# -----------------------------------------------------------------------------
def segment_energy_breakdown_kWh(seg: Segment, p: VehicleParams, soc_start_pct=None):
    """
    Return battery-side energy accounting for one segment.

    net_battery_energy_kWh is the value that changes SoC. It can be negative on
    downhill or braking-heavy segments because regenerative braking is counted.
    gross_consumed_kWh is traction battery draw before regen and excludes aux,
    so: net = gross_consumed + aux_energy - regen_recovered.
    """
    m = p.curb_mass_kg + seg.passengers * p.passenger_mass_kg
    theta = np.arctan(seg.grade)
    # Use the per-segment aux written by apply_weather_loading (base + HVAC) when
    # present; otherwise fall back to the vehicle's constant aux. Without this the
    # weather/HVAC module has no effect -- segment_energy_kWh would always read
    # the flat p.aux_power_kW and discard seg.aux_power_kW.
    seg_aux = getattr(seg, "aux_power_kW", None)
    aux_kW = seg_aux if seg_aux is not None else p.aux_power_kW
    aux_W = aux_kW * 1000.0

    (t, v, a, step_s), motion_diag = build_speed_profile(
        seg, return_diagnostics=True
    )

    net_joules = 0.0
    gross_joules = 0.0
    regen_joules = 0.0
    aux_joules = 0.0
    soc_pct = None if soc_start_pct is None else min(float(soc_start_pct), 100.0)
    drive_eff = p.eta_driveline * p.eta_motor
    regen_power_cap_W = max(p.regen_power_cap_kW, 0.0) * 1000.0
    for vi, ai, dti in zip(v, a, step_s):
        moving = vi > 0.01
        F_roll  = p.roll_coeff * m * p.g * np.cos(theta) * moving
        F_aero  = 0.5 * p.air_density * p.drag_coeff * p.frontal_area_m2 * vi**2
        F_grade = m * p.g * np.sin(theta) * moving
        F_inert = m * p.rot_inertia_factor * ai
        F_trac = F_roll + F_aero + F_grade + F_inert

        P_wheel = F_trac * vi  # mechanical power at the wheels (W)
        aux_step_joules = aux_W * dti
        aux_joules += aux_step_joules

        if P_wheel >= 0:
            # Drawing power: divide by efficiencies (losses make it cost more).
            gross_step_joules = (P_wheel / drive_eff) * dti
            regen_step_joules = 0.0
            net_step_joules = gross_step_joules + aux_step_joules
        else:
            regen_power_W = 0.0
            if vi >= p.regen_min_speed_ms and (soc_pct is None or soc_pct < 100.0):
                # Braking: recover a bounded fraction back into the battery.
                recoverable_W = -P_wheel * drive_eff * p.regen_fraction
                regen_power_W = min(recoverable_W, regen_power_cap_W)
            gross_step_joules = 0.0
            regen_step_joules = regen_power_W * dti
            net_step_joules = aux_step_joules - regen_step_joules

        if soc_pct is not None and net_step_joules < 0:
            charge_room_joules = (
                max(100.0 - soc_pct, 0.0) / 100.0 * p.battery_usable_kWh * 3.6e6
            )
            clipped_net_joules = max(net_step_joules, -charge_room_joules)
            if clipped_net_joules != net_step_joules:
                regen_step_joules = (
                    gross_step_joules + aux_step_joules - clipped_net_joules
                )
                net_step_joules = clipped_net_joules

        gross_joules += gross_step_joules
        regen_joules += regen_step_joules
        net_joules += net_step_joules
        if soc_pct is not None:
            soc_pct -= net_step_joules / 3.6e6 / p.battery_usable_kWh * 100.0
            soc_pct = min(soc_pct, 100.0)

    # Dwell at the stop: bus stationary, only auxiliary load draws power.
    dwell_aux_joules = aux_W * seg.dwell_s
    aux_joules += dwell_aux_joules
    net_joules += dwell_aux_joules

    if motion_diag is not None:
        motion_diag = dict(motion_diag)
        motion_diag["aux_power_kW"] = float(aux_kW)
        motion_diag["dwell_aux_time_s"] = float(seg.dwell_s)
        motion_diag["aux_total_time_s"] = (
            float(motion_diag.get("actual_profile_time_s") or 0.0) + float(seg.dwell_s)
        )

    return SegmentEnergy(
        net_battery_energy_kWh=net_joules / 3.6e6,
        gross_consumed_kWh=gross_joules / 3.6e6,
        regen_recovered_kWh=regen_joules / 3.6e6,
        aux_energy_kWh=aux_joules / 3.6e6,
        motion_diagnostics=motion_diag,
    )


def segment_energy_kWh(seg: Segment, p: VehicleParams, soc_start_pct=None):
    """Back-compatible alias returning net battery energy, in kWh."""
    return segment_energy_breakdown_kWh(
        seg, p, soc_start_pct=soc_start_pct
    ).net_battery_energy_kWh


# ----------------------------------------------------------------------------- 
# 4. Simulate a whole route and track SoC
# -----------------------------------------------------------------------------
def simulate_route(segments, p: VehicleParams, soc0_pct=100.0):
    rows = []
    soc = soc0_pct
    cum_dist_km = 0.0
    for i, seg in enumerate(segments):
        energy = segment_energy_breakdown_kWh(seg, p, soc_start_pct=soc)
        net_E = energy.net_battery_energy_kWh
        soc_before = soc
        soc -= net_E / p.battery_usable_kWh * 100.0
        soc = min(soc, 100.0)  # a real BMS caps charging at 100%
        cum_dist_km += seg.length_m / 1000.0
        dist_km = seg.length_m / 1000.0
        rows.append({
            "segment": i,
            "from_stop_departure_time": getattr(seg, "from_stop_departure_time", None),
            "to_stop_arrival_time": getattr(seg, "to_stop_arrival_time", None),
            "to_stop_departure_time": getattr(seg, "to_stop_departure_time", None),
            "run_time_s": round(seg.run_time_s, 1) if seg.run_time_s is not None else None,
            "dwell_s": round(seg.dwell_s, 1),
            "length_m": round(seg.length_m, 1),
            "grade_%": round(seg.grade * 100, 2),
            "passengers": seg.passengers,
            "n_signals": getattr(seg, "n_signals", 0),
            "signal_source": getattr(seg, "signal_source", None),
            "n_effective_signal_stops": int((energy.motion_diagnostics or {}).get(
                "n_effective_signal_stops", 0
            )),
            "signal_wait_s": round((energy.motion_diagnostics or {}).get(
                "signal_wait_s", 0.0
            ), 3),
            "signal_wait_requested_s": round((energy.motion_diagnostics or {}).get(
                "signal_wait_requested_s", 0.0
            ), 3),
            "signal_wait_reduced_s": round((energy.motion_diagnostics or {}).get(
                "signal_wait_reduced_s", 0.0
            ), 3),
            "actual_profile_time_s": round((energy.motion_diagnostics or {}).get(
                "actual_profile_time_s", 0.0
            ), 3),
            "moving_profile_time_s": round((energy.motion_diagnostics or {}).get(
                "moving_profile_time_s", 0.0
            ), 3),
            "scheduled_run_time_s": round((energy.motion_diagnostics or {}).get(
                "scheduled_run_time_s", 0.0
            ) or 0.0, 3),
            "schedule_delay_s": round((energy.motion_diagnostics or {}).get(
                "schedule_delay_s", 0.0
            ), 3),
            "schedule_infeasible": bool((energy.motion_diagnostics or {}).get(
                "schedule_infeasible", False
            )),
            "signal_time_policy": (energy.motion_diagnostics or {}).get(
                "signal_time_policy"
            ),
            "speed_cap_ms": round((energy.motion_diagnostics or {}).get(
                "speed_cap_ms", 0.0
            ) or 0.0, 3),
            "min_feasible_motion_time_s": round((energy.motion_diagnostics or {}).get(
                "min_feasible_motion_time_s", 0.0
            ) or 0.0, 3),
            "aux_power_kW": round((energy.motion_diagnostics or {}).get(
                "aux_power_kW", 0.0
            ), 3),
            "aux_total_time_s": round((energy.motion_diagnostics or {}).get(
                "aux_total_time_s", 0.0
            ), 3),
            "net_battery_energy_kWh": round(net_E, 3),
            "gross_consumed_kWh": round(energy.gross_consumed_kWh, 3),
            "regen_recovered_kWh": round(energy.regen_recovered_kWh, 3),
            "aux_energy_kWh": round(energy.aux_energy_kWh, 3),
            "net_battery_kWh_per_km": round(net_E / dist_km, 3),
            "gross_consumed_kWh_per_km": round(
                energy.gross_consumed_kWh / dist_km, 3
            ),
            "cum_dist_km": round(cum_dist_km, 3),
            "SoC_start_%": round(soc_before, 2),
            "SoC_end_%": round(soc, 2),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 
# 5. SYNTHETIC ROUTE  (replace this with your GTFS + DEM loader)
# -----------------------------------------------------------------------------
def make_synthetic_route(n_segments=40, seed=42):
    """
    Build a plausible urban route with deliberate spatial heterogeneity, so the
    downstream charger-placement optimisation has something non-trivial to solve.
    """
    rng = np.random.default_rng(seed)
    segments = []
    for _ in range(n_segments):
        length = rng.uniform(300, 650)            # urban stop spacing (m)
        grade = rng.normal(0.0, 0.02)             # mostly flat, some hills (+-)
        grade = float(np.clip(grade, -0.06, 0.06))
        v_cruise = rng.uniform(8.5, 13.5)         # 30-49 km/h free-flow
        dwell = rng.uniform(12, 30)               # seconds at the stop
        pax = int(rng.integers(8, 45))            # occupancy
        segments.append(Segment(length, grade, v_cruise, dwell, pax))
    return segments


# ----------------------------------------------------------------------------- 
# 6. RUN
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run the synthetic BEB SoC example.")
    p.add_argument("--config", help="Path to a model YAML config file.")
    p.add_argument("--results-csv", help="Override synthetic results CSV path.")
    p.add_argument("--trace-png", help="Override synthetic SoC trace PNG path.")
    return p.parse_args()


def main():
    args = parse_args()
    p = VehicleParams.from_config(args.config)
    segments = make_synthetic_route()
    df = simulate_route(segments, p, soc0_pct=100.0)

    total_net_E = df["net_battery_energy_kWh"].sum()
    total_gross_E = df["gross_consumed_kWh"].sum()
    total_regen_E = df["regen_recovered_kWh"].sum()
    total_aux_E = df["aux_energy_kWh"].sum()
    total_km = df["cum_dist_km"].iloc[-1]
    print(df.to_string(index=False))
    print("\n--- Route summary ---")
    print(f"Segments              : {len(df)}")
    print(f"Total distance        : {total_km:.2f} km")
    print(f"Net battery energy    : {total_net_E:.2f} kWh")
    print(f"Gross traction draw   : {total_gross_E:.2f} kWh")
    print(f"Regen recovered       : {total_regen_E:.2f} kWh")
    print(f"Aux energy            : {total_aux_E:.2f} kWh")
    print(f"Average net intensity : {total_net_E / total_km:.3f} kWh/km")
    print(f"SoC at end of route   : {df['SoC_end_%'].iloc[-1]:.1f} %")

    results_csv = Path(args.results_csv) if args.results_csv else get_path(
        "synthetic_segment_results_csv", args.config
    )
    trace_png = Path(args.trace_png) if args.trace_png else get_path(
        "synthetic_soc_trace_png", args.config
    )
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    trace_png.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv, index=False)

    # Plot SoC and per-segment net battery energy against distance.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(df["cum_dist_km"], df["SoC_end_%"], marker="o", ms=3)
    ax1.set_ylabel("State of Charge (%)")
    ax1.set_title("BEB State of Charge along the route")
    ax1.grid(True, alpha=0.3)

    ax2.bar(df["cum_dist_km"], df["net_battery_kWh_per_km"], width=0.2)
    ax2.set_ylabel("Net battery energy (kWh/km)")
    ax2.set_xlabel("Cumulative distance (km)")
    ax2.set_title("Per-segment net battery energy")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(trace_png, dpi=130)
    print(f"\nSaved: {results_csv}  and  {trace_png}")


if __name__ == "__main__":
    main()