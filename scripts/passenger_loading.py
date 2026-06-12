"""
passenger_loading.py
================================================================================
Turn an hourly passenger-flow profile (% of daily flow per hour) into per-segment
on-board occupancy for the BEB energy model.

Occupancy is modelled as the product of two layers:

    occupancy(segment) = peak_occupancy(hour) * load_shape(position on route)

  * peak_occupancy(hour)  -- DATA-DRIVEN. Scales with your hourly % profile,
        anchored so the peak hour's busiest segment == crush_capacity
        ("fully packed"). This avoids needing an absolute daily ridership total.

  * load_shape(position)  -- ASSUMPTION. A trapezoid along the route: load builds
        from the origin, plateaus across the busy middle, and empties toward the
        terminus. Replace this with a real boarding/alighting profile once APC or
        stop-level OD data is available; the temporal layer stays unchanged.

Designed to be applied as a post-pass over the Segment list, exactly like
add_grades_from_dem(): build_segments() -> add_grades_from_dem() ->
apply_passenger_loading().
================================================================================
"""

from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Local copy of the GTFS time parser so this module stands alone.
# (Mirrors gtfs_to_segments.gtfs_time_to_seconds; GTFS times can exceed 24:00.)
# -----------------------------------------------------------------------------
def _time_to_seconds(t: str) -> int:
    h, m, s = (int(x) for x in str(t).strip().split(":"))
    return h * 3600 + m * 60 + s


# -----------------------------------------------------------------------------
# Layer 1: temporal demand from the hourly % profile
# -----------------------------------------------------------------------------
@dataclass
class HourlyDemandProfile:
    """
    Fraction of daily passenger flow per clock hour (keys 0..23).
    Values are normalised internally, so they can be given as percentages,
    counts, or fractions -- only their relative size matters.
    """
    hourly_fraction: dict

    @classmethod
    def from_percent(cls, mapping):
        """mapping: {hour:int -> percent:float}, e.g. {7:6.2, 8:8.9, 16:8.95}."""
        total = float(sum(mapping.values()))
        if total <= 0:
            raise ValueError("hourly profile sums to zero")
        return cls({int(h): float(v) / total for h, v in mapping.items()})

    @property
    def peak_hour(self) -> int:
        return max(self.hourly_fraction, key=self.hourly_fraction.get)

    def temporal_factor(self, hour: int) -> float:
        """
        Demand in `hour` relative to the peak hour, in [0, 1].
        peak hour -> 1.0; an hour with half the peak's share -> 0.5.
        """
        peak = self.hourly_fraction[self.peak_hour]
        return self.hourly_fraction.get(int(hour), 0.0) / peak if peak > 0 else 0.0


def trip_reference_hour(trip_rows, mode="midpoint") -> int:
    """
    The clock hour used to look up demand for a whole trip.
      'start'    -> hour the trip departs its first stop
      'midpoint' -> hour at the temporal middle of the trip (default; better for
                    long trips that straddle the peak)
    """
    t0 = _time_to_seconds(trip_rows[0]["departure_time"])
    t1 = _time_to_seconds(trip_rows[-1]["arrival_time"])
    sec = t0 if mode == "start" else (t0 + t1) // 2
    return int((sec // 3600) % 24)


# -----------------------------------------------------------------------------
# Layer 2: spatial load shape along the route
# -----------------------------------------------------------------------------
def trapezoid_shape(frac, board_frac=0.2, alight_frac=0.2, floor_frac=0.0):
    """
    Multiplier in [floor_frac, 1] for position `frac` in [0, 1] along the route.
        rises 0->1 over the first `board_frac` of the route,
        holds at 1 across the middle,
        falls 1->0 over the last `alight_frac`.
    Presets:
        trapezoid  : board=alight=0.2  (mean ~0.8x peak)
        triangular : board=alight=0.5  (peak at midpoint, mean ~0.5x peak)
        flat       : board=alight=0.0  (every segment at peak)
    `floor_frac` lifts the empty ends to a baseline load (e.g. 0.1 for routes
    that never fully clear).
    """
    if board_frac > 0 and frac < board_frac:
        m = frac / board_frac
    elif alight_frac > 0 and frac > 1.0 - alight_frac:
        m = (1.0 - frac) / alight_frac
    else:
        m = 1.0
    return floor_frac + (1.0 - floor_frac) * m


# -----------------------------------------------------------------------------
# Apply: set Segment.passengers
# -----------------------------------------------------------------------------
def apply_passenger_loading(segments, trip_rows, profile, crush_capacity=70,
                            board_frac=0.2, alight_frac=0.2, floor_frac=0.0,
                            hour_mode="midpoint", round_to_int=True,
                            verbose=True):
    """
    Overwrite each Segment.passengers with a modelled on-board occupancy.

    crush_capacity : "fully packed" peak occupancy for YOUR vehicle (seated +
                     standing). This is a vehicle property -- set it to match the
                     BEB you simulate (a 12 m single-deck is ~70-80; a Dublin
                     double-deck is ~90-100+). It anchors the peak-hour maximum.
    board/alight/floor_frac : load-shape parameters (see trapezoid_shape).
    hour_mode      : 'midpoint' or 'start' (see trip_reference_hour).

    Returns the same segment list (mutated in place).
    """
    if not segments:
        return segments

    hour = trip_reference_hour(trip_rows, hour_mode)
    tf = profile.temporal_factor(hour)
    peak_occ = crush_capacity * tf

    lengths = [s.length_m for s in segments]
    total = sum(lengths)
    cum = 0.0
    loads = []
    for seg, L in zip(segments, lengths):
        mid_frac = (cum + L / 2.0) / total if total > 0 else 0.5
        cum += L
        occ = peak_occ * trapezoid_shape(mid_frac, board_frac, alight_frac,
                                         floor_frac)
        seg.passengers = int(round(occ)) if round_to_int else occ
        loads.append(seg.passengers)

    if verbose:
        mean = sum(loads) / len(loads)
        print(f"  loading: trip hour {hour:02d}:00, demand factor {tf:.2f} -> "
              f"peak {peak_occ:.0f} pax (mean {mean:.0f}, max {max(loads)}) "
              f"across {len(loads)} segments")
    return segments