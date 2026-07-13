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
        negatives = {h: v for h, v in mapping.items() if float(v) < 0.0}
        if negatives:
            bad_hours = ", ".join(str(h) for h in sorted(negatives))
            raise ValueError(f"hourly profile has negative demand at hour(s): {bad_hours}")
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


def _beta_weights(positions, mode, concentration):
    """
    Beta-density weights (summing to 1) over `positions` in [0, 1], peaked near
    `mode`. Higher `concentration` => tighter peak. Used to spread boardings and
    alightings along the route without any external stats dependency.
    """
    c = max(float(concentration), 2.0001)          # >2 keeps an interior peak
    a = 1.0 + (c - 2.0) * mode
    b = 1.0 + (c - 2.0) * (1.0 - mode)
    w = []
    for x in positions:
        x = min(max(float(x), 1e-6), 1.0 - 1e-6)
        w.append(x ** (a - 1.0) * (1.0 - x) ** (b - 1.0))
    s = sum(w)
    return [wi / s for wi in w] if s > 0 else [1.0 / len(w)] * len(w)


def occupancy_from_boarding_alighting(stop_fracs, board_pos=0.25, alight_pos=0.75,
                                      concentration=6.0, floor_frac=0.0):
    """
    Realistic load shape, built the way occupancy actually arises:

        occupancy(segment) = (everyone boarded so far) - (everyone alighted so far)

    Boardings are spread along the route with a density peaked near `board_pos`
    (front-loaded), alightings near `alight_pos` (back-loaded). The running
    difference is naturally ~empty at both ends and fullest where the boarding
    lead over alighting is largest -- the "busy middle". Move board_pos /
    alight_pos to place that busy zone (e.g. alight_pos high for a radial route
    that fills toward a city-centre terminus). This is exactly the structure of
    APC ons/offs data, so measured counts can replace the two Beta densities
    later with no change downstream.

    stop_fracs : cumulative distance fraction at each STOP, 0.0..1.0
                 (length = n_segments + 1).
    Returns one multiplier in [floor_frac, 1] per SEGMENT (normalised so the
    busiest segment == 1).
    """
    n_seg = len(stop_fracs) - 1
    if n_seg <= 0:
        return []
    on_w = _beta_weights(stop_fracs[:-1], board_pos, concentration)   # stops 0..N-1
    off_w = _beta_weights(stop_fracs[1:], alight_pos, concentration)  # stops 1..N

    occ = []
    cum_on = cum_off = 0.0
    for j in range(n_seg):
        cum_on += on_w[j]
        if j >= 1:
            cum_off += off_w[j - 1]
        occ.append(max(cum_on - cum_off, 0.0))   # clamp tiny negatives

    mx = max(occ) if occ else 0.0
    if mx <= 0:
        return [0.0] * n_seg
    return [floor_frac + (1.0 - floor_frac) * (o / mx) for o in occ]


def _stop_fractions(lengths):
    """Cumulative distance fraction at each stop boundary (0..1), from segment
    lengths. len(result) = len(lengths) + 1."""
    total = float(sum(lengths))
    fracs = [0.0]
    run = 0.0
    for L in lengths:
        run += L
        fracs.append(run / total if total > 0 else 0.0)
    return fracs


# -----------------------------------------------------------------------------
# Apply: set Segment.passengers
# -----------------------------------------------------------------------------
def apply_passenger_loading(segments, trip_rows, profile, crush_capacity=70,
                            shape="beta", board_pos=0.25, alight_pos=0.75,
                            concentration=6.0, board_frac=0.2, alight_frac=0.2,
                            floor_frac=0.0, hour_mode="midpoint",
                            round_to_int=True, verbose=True):
    """
    Overwrite each Segment.passengers with a modelled on-board occupancy:

        occupancy = crush_capacity * temporal_factor(hour) * shape(position)

    crush_capacity : "fully packed" peak occupancy for YOUR vehicle (seated +
                     standing). Anchors the peak-hour, busiest-segment maximum.
    shape          : spatial load profile along the route:
        'beta'      -> realistic boarding/alighting curve (default; low at ends,
                       full in the middle). Controlled by board_pos, alight_pos,
                       concentration -- see occupancy_from_boarding_alighting.
        'trapezoid' -> piecewise-linear ramp/plateau/ramp (board_frac/alight_frac)
        'triangular'-> board_frac=alight_frac=0.5
        'flat'      -> every segment at the peak level
    hour_mode      : 'midpoint' or 'start' (see trip_reference_hour).

    Returns the same segment list (mutated in place).
    """
    if not segments:
        return segments

    hour = trip_reference_hour(trip_rows, hour_mode)
    tf = profile.temporal_factor(hour)
    peak_occ = crush_capacity * tf

    lengths = [s.length_m for s in segments]

    if shape in ("beta", "realistic", "boarding_alighting"):
        stop_fracs = _stop_fractions(lengths)
        mult = occupancy_from_boarding_alighting(
            stop_fracs, board_pos=board_pos, alight_pos=alight_pos,
            concentration=concentration, floor_frac=floor_frac)
    elif shape in ("trapezoid", "triangular", "flat"):
        if shape == "triangular":
            board_frac = alight_frac = 0.5
        elif shape == "flat":
            board_frac = alight_frac = 0.0
        total = float(sum(lengths))
        mult, cum = [], 0.0
        for L in lengths:
            mid = (cum + L / 2.0) / total if total > 0 else 0.5
            cum += L
            mult.append(trapezoid_shape(mid, board_frac, alight_frac, floor_frac))
    else:
        raise ValueError(f"unknown shape {shape!r}")

    loads = []
    for seg, m in zip(segments, mult):
        occ = peak_occ * m
        seg.passengers = int(round(occ)) if round_to_int else occ
        loads.append(seg.passengers)

    if verbose:
        mean = sum(loads) / len(loads)
        peak_at = (mult.index(max(mult)) + 0.5) / len(mult) if mult else 0.0
        print(f"  loading: hour {hour:02d}:00, factor {tf:.2f} -> peak "
              f"{peak_occ:.0f} pax (mean {mean:.0f}, max {max(loads)}); "
              f"shape={shape}, busiest at ~{peak_at * 100:.0f}% of route")
    return segments
