"""Bounded nonzero-boundary motion for Phase 7C, separate from frozen B0."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from trip_motion import solve_motion_leg


@dataclass(frozen=True)
class MotionPhase:
    duration_s: float
    initial_speed_ms: float
    acceleration_ms2: float

    @property
    def distance_m(self):
        return (self.initial_speed_ms * self.duration_s
                + 0.5 * self.acceleration_ms2 * self.duration_s**2)


@dataclass(frozen=True)
class ContinuousInterval:
    length_m: float
    initial_speed_ms: float
    terminal_speed_ms: float
    phases: tuple[MotionPhase, ...]
    minimum_time_s: float
    maximum_nonstop_time_s: float
    scheduled_time_s: float | None
    timing_status: str
    shape: str
    legacy_leg: object = None

    @property
    def duration_s(self):
        return sum(p.duration_s for p in self.phases)

    @property
    def distance_integral_m(self):
        return sum(p.distance_m for p in self.phases)

    @property
    def schedule_delay_s(self):
        return max(self.duration_s - self.scheduled_time_s, 0.0) if self.scheduled_time_s is not None else 0.0

    @property
    def peak_speed_ms(self):
        return max([self.initial_speed_ms, self.terminal_speed_ms]
                   + [p.initial_speed_ms for p in self.phases])

    @property
    def minimum_speed_ms(self):
        return min([self.initial_speed_ms, self.terminal_speed_ms]
                   + [p.initial_speed_ms for p in self.phases])

    def position_speed(self, time_s):
        times = np.clip(np.asarray(time_s, dtype=float), 0.0, self.duration_s)
        x = np.zeros_like(times)
        v = np.full_like(times, self.initial_speed_ms)
        elapsed = 0.0
        for phase in self.phases:
            local = np.clip(times - elapsed, 0.0, phase.duration_s)
            x += phase.initial_speed_ms * local + 0.5 * phase.acceleration_ms2 * local**2
            v = np.where(times >= elapsed,
                         phase.initial_speed_ms + phase.acceleration_ms2 * local, v)
            elapsed += phase.duration_s
        return x, np.maximum(v, 0.0)

    def time_at_distance(self, distance_m):
        elapsed, distance = 0.0, 0.0
        for phase in self.phases:
            if distance_m <= distance + phase.distance_m + 1e-9:
                dx = max(0.0, min(distance_m - distance, phase.distance_m))
                if abs(phase.acceleration_ms2) < 1e-14:
                    local = dx / phase.initial_speed_ms if phase.initial_speed_ms > 0 else 0.0
                else:
                    end_v = math.sqrt(max(phase.initial_speed_ms**2 + 2 * phase.acceleration_ms2 * dx, 0.0))
                    local = 2 * dx / (phase.initial_speed_ms + end_v) if dx > 0 else 0.0
                return elapsed + min(local, phase.duration_s)
            elapsed += phase.duration_s
            distance += phase.distance_m
        return self.duration_s

    def sample(self, dt_s):
        """Retain the reference left-endpoint quadrature; endpoints are events."""
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if self.legacy_leg is not None:
            return self.legacy_leg.sample(dt_s)
        t = np.arange(0.0, self.duration_s, dt_s)
        step = np.minimum(dt_s, self.duration_s - t)
        _, v = self.position_speed(t)
        a = np.zeros_like(t)
        elapsed = 0.0
        for phase in self.phases:
            a[t >= elapsed] = phase.acceleration_ms2
            elapsed += phase.duration_s
        return t, v, a, step


def _phases(v0, v1, plateau, length, accel, decel, total_time=None):
    first_a = accel if plateau >= v0 else -decel
    last_a = accel if v1 >= plateau else -decel
    first_t = abs((plateau - v0) / first_a)
    last_t = abs((v1 - plateau) / last_a)
    ramps_distance = (v0 + plateau) * first_t / 2 + (plateau + v1) * last_t / 2
    if total_time is None:
        cruise_t = max((length - ramps_distance) / plateau, 0.0) if plateau > 0 else 0.0
    else:
        cruise_t = max(total_time - first_t - last_t, 0.0)
    return tuple(p for p in (MotionPhase(first_t, v0, first_a),
                             MotionPhase(cruise_t, plateau, 0.0),
                             MotionPhase(last_t, plateau, last_a)) if p.duration_s > 0)


def solve_interval(length_m, initial_speed_ms, terminal_speed_ms, cap_ms,
                   accel_ms2=1.0, decel_ms2=1.2, scheduled_time_s=None,
                   target_ms=None):
    """Solve a ramp/cruise/ramp trajectory with peak, monotone or valley speed.

    A prescribed time is feasible iff the distance lies between the integrated
    lower and upper speed envelopes. A scalar plateau-speed root then gives a
    continuous solution at the configured acceleration/braking limits. Long
    budgets never create a stationary cruise. If only an early fastest solution
    exists, timing_status exposes that incompatibility for the trip planner.
    """
    length, v0, v1 = float(length_m), float(initial_speed_ms), float(terminal_speed_ms)
    a, d, cap = float(accel_ms2), float(decel_ms2), float(cap_ms)
    if not all(math.isfinite(x) for x in (length, v0, v1, a, d, cap)):
        raise ValueError("Interval parameters must be finite")
    if length <= 0 or min(a, d, cap) <= 0 or min(v0, v1) < 0:
        raise ValueError("Invalid interval distance, speed or dynamics")
    if max(v0, v1) > cap + 1e-9:
        raise ValueError("Boundary speed exceeds cap")
    if scheduled_time_s is not None and not math.isfinite(scheduled_time_s):
        raise ValueError("Scheduled time must be finite or None")
    scheduled = float(scheduled_time_s) if scheduled_time_s is not None else None
    minimum_distance = ((v1*v1 - v0*v0) / (2*a) if v1 >= v0 else (v0*v0 - v1*v1) / (2*d))
    if length < minimum_distance - 1e-8:
        raise ValueError("Insufficient distance for required boundary speeds")
    curve = 0.5 * (1/a + 1/d)
    peak = min(cap, math.sqrt((length + v0*v0/(2*a) + v1*v1/(2*d)) / curve))
    fastest = _phases(v0, v1, peak, length, a, d)
    minimum_time = sum(p.duration_s for p in fastest)
    stopping_distance = v0*v0/(2*d) + v1*v1/(2*a)
    maximum_time = math.inf
    distance_tolerance = 1e-10 * max(length, 1.0)
    if abs(length - stopping_distance) <= distance_tolerance:
        # Avoid taking a square root of cancellation noise at pure braking or
        # acceleration distance, which otherwise makes max time smaller than min.
        maximum_time = max(minimum_time, v0/d + v1/a)
    elif length < stopping_distance:
        valley = math.sqrt(max((stopping_distance - length) / curve, 0.0))
        maximum_time = max(minimum_time, max((v0 - valley)/d, 0.0) + max((v1 - valley)/a, 0.0))
    if scheduled is not None and scheduled < minimum_time - 1e-9:
        phases, status, shape = fastest, "too_short", "peak"
    elif scheduled is not None and scheduled > maximum_time + 1e-9:
        phases, status, shape = fastest, "too_long_for_nonstop_boundaries", "peak"
    elif scheduled is None:
        plateau = min(max(float(target_ms if target_ms is not None else cap), v0, v1, 1e-6), peak)
        phases, status, shape = _phases(v0, v1, plateau, length, a, d), "unscheduled", "peak"
    else:
        total = max(scheduled, minimum_time)
        low = max(0.0, (v0/d + v1/a - total)/(1/d + 1/a))
        high = min(cap, (total + v0/a + v1/d)/(1/a + 1/d))

        def distance_error(plateau):
            return sum(p.distance_m for p in _phases(v0, v1, plateau, length, a, d, total)) - length

        if abs(distance_error(high)) < 1e-9:
            plateau = high
        elif abs(distance_error(low)) < 1e-9 and (low > 0 or distance_error(low) >= 0):
            plateau = low
        else:
            # Distance is monotone in plateau speed on the reachable bracket.
            for _ in range(60):
                midpoint = (low + high) / 2
                if distance_error(midpoint) < 0:
                    low = midpoint
                else:
                    high = midpoint
            plateau = (low + high) / 2
        phases = _phases(v0, v1, plateau, length, a, d, total)
        if ((min(v0,v1)>0 and plateau<=1e-12)
                or any(p.initial_speed_ms == 0 and p.acceleration_ms2 == 0 and p.duration_s > 1e-9 for p in phases)):
            phases, status, shape = fastest, "too_long_for_nonstop_boundaries", "peak"
        else:
            status = "feasible"
            shape = "valley" if plateau < min(v0, v1) - 1e-9 else "peak" if plateau >= max(v0, v1) else "monotone"
    legacy = None
    if v0 == v1 == 0:
        legacy = solve_motion_leg(0.0, length, 0.0, 0.0,
                                  cap if target_ms is None else target_ms, cap, a, d,
                                  scheduled if scheduled is None or scheduled > 0 else None)
        # A nonpositive explicit time is infeasible, so use fastest capped motion.
        if scheduled is not None and scheduled <= 0:
            legacy = solve_motion_leg(0.0, length, 0.0, 0.0, cap, cap, a, d)
        phases = tuple(p for p in (MotionPhase(legacy.t_acc_s, 0.0, a),
                                   MotionPhase(legacy.t_cruise_s, legacy.peak_speed_ms, 0.0),
                                   MotionPhase(legacy.t_dec_s, legacy.peak_speed_ms, -d)) if p.duration_s > 0)
    return ContinuousInterval(length, v0, v1, phases, minimum_time, maximum_time,
                              scheduled, status, shape, legacy)
