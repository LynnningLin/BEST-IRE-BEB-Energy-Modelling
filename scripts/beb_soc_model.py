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
import sys
from dataclasses import dataclass
from pathlib import Path

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


# ----------------------------------------------------------------------------- 
# 2. ROUTE -> MOTION : build a speed profile for one segment
# -----------------------------------------------------------------------------
def build_speed_profile(seg: Segment, a_accel=1.0, a_decel=1.2, dt=0.5):
    """
    Return arrays (t, v, a) for the *driving* part of the segment, using a simple
    trapezoidal profile: accelerate -> (cruise) -> decelerate to a stop.
    Falls back to a triangular profile on segments too short to reach v_cruise.
    a_accel, a_decel in m/s^2 ; dt = time step in seconds.
    """
    v_c = seg.v_cruise_ms
    d_acc = v_c**2 / (2 * a_accel)          # distance to reach cruise speed
    d_dec = v_c**2 / (2 * a_decel)          # distance to brake from cruise

    if d_acc + d_dec <= seg.length_m:
        # Trapezoidal: there is room to cruise.
        d_cruise = seg.length_m - d_acc - d_dec
        t_acc = v_c / a_accel
        t_cruise = d_cruise / v_c
        t_dec = v_c / a_decel
        v_peak = v_c
    else:
        # Triangular: segment too short, never reach cruise speed.
        v_peak = np.sqrt(2 * seg.length_m * a_accel * a_decel / (a_accel + a_decel))
        t_acc = v_peak / a_accel
        t_cruise = 0.0
        t_dec = v_peak / a_decel

    total_t = t_acc + t_cruise + t_dec
    t = np.arange(0, total_t, dt)
    v = np.zeros_like(t)
    a = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < t_acc:                       # accelerating
            a[i] = a_accel
            v[i] = a_accel * ti
        elif ti < t_acc + t_cruise:          # cruising
            a[i] = 0.0
            v[i] = v_peak
        else:                                # decelerating
            a[i] = -a_decel
            v[i] = v_peak - a_decel * (ti - t_acc - t_cruise)

    v = np.clip(v, 0.0, None)
    return t, v, a, dt


# ----------------------------------------------------------------------------- 
# 3. MOTION -> ENERGY : battery energy used over one segment
# -----------------------------------------------------------------------------
def segment_energy_kWh(seg: Segment, p: VehicleParams, soc_start_pct=None):
    """Return battery energy (kWh) for one segment, including dwell aux load."""
    m = p.curb_mass_kg + seg.passengers * p.passenger_mass_kg
    theta = np.arctan(seg.grade)
    # Use the per-segment aux written by apply_weather_loading (base + HVAC) when
    # present; otherwise fall back to the vehicle's constant aux. Without this the
    # weather/HVAC module has no effect -- segment_energy_kWh would always read
    # the flat p.aux_power_kW and discard seg.aux_power_kW.
    seg_aux = getattr(seg, "aux_power_kW", None)
    aux_kW = seg_aux if seg_aux is not None else p.aux_power_kW
    aux_W = aux_kW * 1000.0

    t, v, a, dt = build_speed_profile(seg)

    E_joules = 0.0
    soc_pct = None if soc_start_pct is None else min(float(soc_start_pct), 100.0)
    drive_eff = p.eta_driveline * p.eta_motor
    regen_power_cap_W = max(p.regen_power_cap_kW, 0.0) * 1000.0
    for vi, ai in zip(v, a):
        moving = vi > 0.01
        F_roll  = p.roll_coeff * m * p.g * np.cos(theta) * moving
        F_aero  = 0.5 * p.air_density * p.drag_coeff * p.frontal_area_m2 * vi**2
        F_grade = m * p.g * np.sin(theta) * moving
        F_inert = m * p.rot_inertia_factor * ai
        F_trac = F_roll + F_aero + F_grade + F_inert

        P_wheel = F_trac * vi  # mechanical power at the wheels (W)

        if P_wheel >= 0:
            # Drawing power: divide by efficiencies (losses make it cost more).
            P_batt = P_wheel / drive_eff + aux_W
        else:
            regen_power_W = 0.0
            if vi >= p.regen_min_speed_ms and (soc_pct is None or soc_pct < 100.0):
                # Braking: recover a bounded fraction back into the battery.
                recoverable_W = -P_wheel * drive_eff * p.regen_fraction
                regen_power_W = min(recoverable_W, regen_power_cap_W)
            P_batt = aux_W - regen_power_W

        E_step_joules = P_batt * dt
        if soc_pct is not None and E_step_joules < 0:
            charge_room_joules = (
                max(100.0 - soc_pct, 0.0) / 100.0 * p.battery_usable_kWh * 3.6e6
            )
            E_step_joules = max(E_step_joules, -charge_room_joules)

        E_joules += E_step_joules
        if soc_pct is not None:
            soc_pct -= E_step_joules / 3.6e6 / p.battery_usable_kWh * 100.0
            soc_pct = min(soc_pct, 100.0)

    # Dwell at the stop: bus stationary, only auxiliary load draws power.
    E_joules += aux_W * seg.dwell_s

    return E_joules / 3.6e6  # joules -> kWh


# ----------------------------------------------------------------------------- 
# 4. Simulate a whole route and track SoC
# -----------------------------------------------------------------------------
def simulate_route(segments, p: VehicleParams, soc0_pct=100.0):
    rows = []
    soc = soc0_pct
    cum_dist_km = 0.0
    for i, seg in enumerate(segments):
        E = segment_energy_kWh(seg, p, soc_start_pct=soc)
        soc_before = soc
        soc -= E / p.battery_usable_kWh * 100.0
        soc = min(soc, 100.0)  # a real BMS caps charging at 100%
        cum_dist_km += seg.length_m / 1000.0
        rows.append({
            "segment": i,
            "length_m": round(seg.length_m, 1),
            "grade_%": round(seg.grade * 100, 2),
            "passengers": seg.passengers,
            "energy_kWh": round(E, 3),
            "kWh_per_km": round(E / (seg.length_m / 1000.0), 3),
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

    total_E = df["energy_kWh"].sum()
    total_km = df["cum_dist_km"].iloc[-1]
    print(df.to_string(index=False))
    print("\n--- Route summary ---")
    print(f"Segments              : {len(df)}")
    print(f"Total distance        : {total_km:.2f} km")
    print(f"Total energy          : {total_E:.2f} kWh")
    print(f"Average consumption   : {total_E / total_km:.3f} kWh/km")
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

    # Plot SoC and per-segment consumption against distance.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(df["cum_dist_km"], df["SoC_end_%"], marker="o", ms=3)
    ax1.set_ylabel("State of Charge (%)")
    ax1.set_title("BEB State of Charge along the route")
    ax1.grid(True, alpha=0.3)

    ax2.bar(df["cum_dist_km"], df["kWh_per_km"], width=0.2)
    ax2.set_ylabel("Consumption (kWh/km)")
    ax2.set_xlabel("Cumulative distance (km)")
    ax2.set_title("Per-segment energy consumption")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(trace_png, dpi=130)
    print(f"\nSaved: {results_csv}  and  {trace_png}")


if __name__ == "__main__":
    main()
