"""
sanity_motion.py
================================================================================
Sanity checks for the speed-cap refactor of beb_soc_model.build_speed_profile().

Run:  PYTHONPATH=src:scripts python scripts/sanity_motion.py

Covers (per the refactor task list):
  1. MotionParams imports from beb_soc_model and reads the config.
  2. Feasible no-signal segment preserves the GTFS scheduled runtime exactly.
  3. Infeasible no-signal segment returns the fastest capped profile, logs the
     delay, and NEVER exceeds the speed cap or the configured accel/decel
     (the old acceleration-inflating fallback is gone).
  4. Signal segment reduces the modelled signal wait BEFORE creating delay;
     only physically impossible schedules produce schedule_delay_s.
  5. Per-segment OSM caps (seg.speed_cap_ms) override the config default and
     the source is reported; v_cruise_ms no longer acts as the physical cap.
  6. Back-compat: legacy call signatures and simulate_route() without
     motion_params still work, and the CSV columns are a superset of the old
     ones (nothing existing was renamed or dropped).
================================================================================
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# --- 1. import check ---------------------------------------------------------
from beb_soc_model import (  # noqa: E402
    MotionParams, Segment, VehicleParams, build_speed_profile,
    segment_energy_breakdown_kWh, simulate_route, _freeflow_duration,
)

EPS_V = 1e-6      # speed tolerance (m/s)
EPS_T = 0.55      # time tolerance (s): one dt sampling step


def check(label, cond, detail=""):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(label + " :: " + detail)


def profile_stats(profile):
    t, v, a, step = profile
    return {
        "duration": float(np.sum(step)),
        "distance": float(np.sum(v * step)),
        "v_max": float(np.max(v)) if len(v) else 0.0,
        "a_max": float(np.max(np.abs(a))) if len(a) else 0.0,
    }


mp = MotionParams(accel_ms2=1.0, decel_ms2=1.2, dt_s=0.5,
                  default_speed_cap_ms=13.9, max_speed_cap_ms=25.0,
                  red_wait_s=15.0, max_signal_wait_share=0.35,
                  use_hourly_signal_stop_probability=False, stop_prob=1.0)
CAP = mp.default_speed_cap_ms

print("1) MotionParams import + config round-trip")
check("MotionParams imported from beb_soc_model", True)
mp_cfg = MotionParams.from_config()
check("from_config() builds", isinstance(mp_cfg, MotionParams),
      f"default cap {mp_cfg.default_speed_cap_ms} m/s")
check("hourly table resolves", len(mp_cfg.hour_table_or_none()) == 24)

# --- 2. feasible no-signal segment -------------------------------------------
print("2) feasible no-signal segment preserves the schedule")
seg = Segment(length_m=500.0, run_time_s=75.0, v_cruise_ms=8.0, dwell_s=0)
prof, d = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
s = profile_stats(prof)
check("duration == scheduled", abs(s["duration"] - 75.0) < EPS_T,
      f"{s['duration']:.2f}s vs 75s")
check("distance covered", abs(s["distance"] - 500.0) < 5.0,
      f"{s['distance']:.1f} m")
check("no delay flagged", d["schedule_delay_s"] < 1e-6
      and not d["schedule_infeasible"])
check("cap fields populated", d["speed_cap_ms"] == CAP
      and d["speed_cap_source"] == "config_default"
      and d["target_cruise_ms"] == 8.0)
check("v_peak within cap", s["v_max"] <= CAP + EPS_V, f"{s['v_max']:.2f} m/s")

# The KEY behavioural change: a schedule faster than the GTFS-derived target
# cruise but feasible under the PHYSICAL cap is no longer punished. Old code
# capped at v_cruise_ms and would have flagged this infeasible.
print("   circularity removed: schedule feasible under cap but above v_cruise")
seg = Segment(length_m=500.0, run_time_s=55.0, v_cruise_ms=8.0, dwell_s=0)
# 55 s needs v_peak ~ 11.2 m/s > v_cruise 8, but < cap 13.9 (min_t is 48.7 s).
prof, d = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
s = profile_stats(prof)
check("schedule preserved despite v_cruise < needed peak",
      abs(s["duration"] - 55.0) < EPS_T and not d["schedule_infeasible"],
      f"{s['duration']:.2f}s, v_max {s['v_max']:.2f} <= cap {CAP}")
check("peak above old (wrong) cap, below real cap",
      8.0 + 0.5 < s["v_max"] <= CAP + EPS_V, f"v_max {s['v_max']:.2f}")

# --- 3. infeasible no-signal segment ------------------------------------------
print("3) infeasible no-signal segment: fastest capped profile + logged delay")
seg = Segment(length_m=500.0, run_time_s=20.0, v_cruise_ms=22.0, dwell_s=0)
min_t = _freeflow_duration(500.0, CAP, mp.accel_ms2, mp.decel_ms2)
prof, d = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
s = profile_stats(prof)
check("profile is the feasible minimum, not the schedule",
      abs(s["duration"] - min_t) < EPS_T,
      f"{s['duration']:.2f}s vs min {min_t:.2f}s (schedule 20s)")
check("delay logged", abs(d["schedule_delay_s"] - (s["duration"] - 20.0)) < EPS_T
      and d["schedule_infeasible"])
check("min_feasible_motion_time_s reported",
      abs(d["min_feasible_motion_time_s"] - min_t) < 1e-6)
check("speed cap NEVER exceeded", s["v_max"] <= CAP + EPS_V,
      f"v_max {s['v_max']:.2f} <= {CAP}")
check("accel/decel NEVER inflated",
      s["a_max"] <= max(mp.accel_ms2, mp.decel_ms2) + EPS_V,
      f"a_max {s['a_max']:.2f} <= {max(mp.accel_ms2, mp.decel_ms2)}")

# Regression for the removed legacy fallback: schedule shorter than even the
# triangular profile used to trigger a_eff/d_eff inflation (a_max >> config).
print("   legacy accel-inflating fallback is gone")
seg = Segment(length_m=400.0, run_time_s=10.0, v_cruise_ms=22.0, dwell_s=0)
prof, d = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
s = profile_stats(prof)
check("dynamics respected even for absurd schedules",
      s["a_max"] <= max(mp.accel_ms2, mp.decel_ms2) + EPS_V
      and s["v_max"] <= CAP + EPS_V and d["schedule_infeasible"],
      f"a_max {s['a_max']:.2f}, v_max {s['v_max']:.2f}, "
      f"delay {d['schedule_delay_s']:.1f}s")

# --- 4. signal segments: wait squeezed before delay ---------------------------
print("4) signal segment reduces modelled wait before creating delay")


def signal_seg(run_time_s, n_signals=2):
    return Segment(length_m=600.0, run_time_s=run_time_s, v_cruise_ms=8.0,
                   dwell_s=0, n_signals=n_signals,
                   from_stop_departure_time="08:00:00",
                   to_stop_arrival_time="08:03:00")


# stop_prob=1.0 and hourly table off -> both signals force stops, wait
# requested = 2 * 15 = 30 s, capped by share 0.35 * scheduled.
# (a) generous schedule: full (share-capped) wait fits inside it
seg = signal_seg(run_time_s=120.0)
prof, d = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
s = profile_stats(prof)
check("generous schedule: full requested wait modelled",
      abs(d["signal_wait_s"] - d["signal_wait_requested_s"]) < 1e-6
      and d["signal_wait_reduced_s"] < 1e-6,
      f"wait {d['signal_wait_s']:.1f}s of {d['signal_wait_requested_s']:.1f}s")
check("total still == schedule (wait is internal slack)",
      abs(s["duration"] - 120.0) < 3 * EPS_T, f"{s['duration']:.2f}s")
check("no delay", d["schedule_delay_s"] < 1e-6)

# (b) tight schedule: motion fits but not the full wait -> wait reduced FIRST
min_motion = d["min_feasible_motion_time_s"]  # same geometry/cap/n_sub
tight = min_motion + 8.0                      # room for only ~8 s of waiting
seg = signal_seg(run_time_s=tight)
prof, d = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
s = profile_stats(prof)
check("tight schedule: wait reduced, not delayed",
      d["signal_wait_reduced_s"] > 1e-6 and d["schedule_delay_s"] < 3 * EPS_T
      and not d["schedule_infeasible"],
      f"wait {d['signal_wait_s']:.1f}s (reduced by "
      f"{d['signal_wait_reduced_s']:.1f}s), duration {s['duration']:.1f}s "
      f"vs schedule {tight:.1f}s")
check("wait equals exactly the available slack",
      abs(d["signal_wait_s"] - 8.0) < 1e-6, f"{d['signal_wait_s']:.2f}s")

# (c) impossible schedule: even zero wait cannot fit -> wait 0, delay logged
seg = signal_seg(run_time_s=min_motion - 10.0)
prof, d = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
s = profile_stats(prof)
check("impossible schedule: wait zeroed before any delay",
      d["signal_wait_s"] < 1e-6
      and abs(d["signal_wait_reduced_s"] - d["signal_wait_requested_s"]) < 1e-6)
check("delay = true physical excess", d["schedule_infeasible"]
      and abs(d["schedule_delay_s"] - (s["duration"] - seg.run_time_s)) < EPS_T,
      f"delay {d['schedule_delay_s']:.1f}s")
check("cap respected across all sub-links", s["v_max"] <= CAP + EPS_V)

# --- 5. per-segment OSM cap overrides the default -----------------------------
print("5) seg.speed_cap_ms (OSM) overrides config default; v_cruise is not a cap")
seg = Segment(length_m=500.0, run_time_s=40.0, v_cruise_ms=22.0, dwell_s=0,
              speed_cap_ms=8.33, speed_cap_source="osm")   # 30 km/h zone
prof, d = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
s = profile_stats(prof)
check("OSM cap used", d["speed_cap_ms"] == 8.33
      and d["speed_cap_source"] == "osm")
check("target cruise clamped by cap", d["target_cruise_ms"] == 8.33)
check("30 km/h zone enforced", s["v_max"] <= 8.33 + EPS_V,
      f"v_max {s['v_max']:.2f}")
check("infeasible 40s schedule under 30 km/h detected",
      d["schedule_infeasible"] and d["schedule_delay_s"] > 5.0,
      f"delay {d['schedule_delay_s']:.1f}s")

seg = Segment(length_m=500.0, run_time_s=60.0, v_cruise_ms=10.0, dwell_s=0,
              speed_cap_ms=200.0, speed_cap_source="osm")  # bogus OSM value
prof, d = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
check("bogus per-segment cap clamped to max_speed_cap_ms",
      d["speed_cap_ms"] == mp.max_speed_cap_ms, f"{d['speed_cap_ms']} m/s")

# --- 6. back-compat ------------------------------------------------------------
print("6) backward compatibility")
seg = Segment(length_m=500.0, run_time_s=75.0, v_cruise_ms=8.0)
prof_legacy = build_speed_profile(seg, a_accel=1.0, a_decel=1.2, dt=0.5)
check("legacy positional-style kwargs call works",
      profile_stats(prof_legacy)["duration"] > 0)

vp = VehicleParams()
e = segment_energy_breakdown_kWh(seg, vp, soc_start_pct=90.0)
check("segment_energy_breakdown_kWh without motion_params",
      e.net_battery_energy_kWh > 0)

segs = [Segment(length_m=400.0 + 50 * i, run_time_s=60.0, v_cruise_ms=9.0,
                dwell_s=15, n_signals=i % 2,
                from_stop_departure_time=f"08:{i:02d}:00")
        for i in range(4)]
df_old_style = simulate_route(segs, vp)                       # no motion_params
df_new_style = simulate_route(segs, vp, motion_params=mp_cfg)
OLD_COLUMNS = {
    "segment", "from_stop_departure_time", "to_stop_arrival_time",
    "to_stop_departure_time", "run_time_s", "dwell_s", "length_m", "grade_%",
    "passengers", "signal_hour", "signal_stop_prob", "signal_stop_prob_default",
    "signal_stop_prob_source", "n_effective_signal_stops", "signal_wait_s",
    "signal_wait_requested_s", "signal_wait_reduced_s", "actual_profile_time_s",
    "moving_profile_time_s", "scheduled_run_time_s", "schedule_delay_s",
    "schedule_infeasible", "signal_time_policy", "speed_cap_ms",
    "min_feasible_motion_time_s", "aux_power_kW", "aux_total_time_s",
    "net_battery_energy_kWh", "gross_consumed_kWh", "regen_recovered_kWh",
    "aux_energy_kWh", "net_battery_kWh_per_km", "gross_consumed_kWh_per_km",
    "cum_dist_km", "SoC_start_%", "SoC_end_%",
}
check("all pre-refactor CSV columns still present",
      OLD_COLUMNS.issubset(df_old_style.columns),
      f"missing: {OLD_COLUMNS - set(df_old_style.columns)}")
check("new diagnostics columns added",
      {"target_cruise_ms", "speed_cap_source"}.issubset(df_new_style.columns))
check("energy accounting closes (net = gross + aux - regen)",
      bool(np.allclose(
          df_new_style["net_battery_energy_kWh"],
          df_new_style["gross_consumed_kWh"]
          + df_new_style["aux_energy_kWh"]
          - df_new_style["regen_recovered_kWh"], atol=2e-3)))

print("\nall sanity checks passed")
