"""C07 geometry-only infrastructure draws; no signal-behaviour randomisation.

Scientific keys deliberately have no trip, F2 inventory, timetable or physics inputs.
The C06 scalar primitive and supplied-event adapter remain unchanged.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math

from publication_timing import stable_uniform, supplied_placement


ALGORITHM = "C02-json-blake2s64-be-v1"
KEY_VERSION = "C07-keys-v1"
INFRA_SEED = 20260801


def scientific_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class Geometry:
    pattern_id: str
    version: str
    length_m: float
    boundaries_m: tuple[float, ...]
    accepted: bool = False

    def __post_init__(self):
        object.__setattr__(self, "boundaries_m", tuple(self.boundaries_m))
        if not self.pattern_id or not self.version:
            raise ValueError("Canonical pattern and geometry version are required")
        if not math.isfinite(self.length_m) or self.length_m <= 0:
            raise ValueError("Geometry length must be finite and positive")
        b = self.boundaries_m
        if (len(b) < 2 or any(not math.isfinite(x) for x in b)
                or b[0] != 0 or b[-1] != self.length_m):
            raise ValueError("Explicit complete served-path interval boundaries required")
        if any(y <= x for x, y in zip(b, b[1:])):
            raise ValueError("Repeated/zero-length stop intervals require explicit C06 registration")

    def receiving_interval(self, position_m):
        if not math.isfinite(position_m) or not 0 <= position_m <= self.length_m:
            raise ValueError("Position outside served path; clipping is forbidden")
        return min(bisect_right(self.boundaries_m, position_m) - 1,
                   len(self.boundaries_m) - 2)


@dataclass(frozen=True)
class EncounterCount:
    value: int | None
    certified: bool
    authority: str

    def __post_init__(self):
        if self.value is not None:
            integer(self.value, "N")
        if self.certified and (self.value is None or not self.authority):
            raise ValueError("Certification requires N and its authority")


@dataclass(frozen=True)
class Draw:
    identity: str
    key: tuple
    uniform: float
    position_m: float | None
    selected: bool
    sorted_ordinal: int | None
    receiving_interval: int | None


@dataclass(frozen=True)
class Layout:
    rule: str
    status: str
    reasons: tuple[str, ...]
    pattern_id: str
    geometry_version: str
    placement_index: int
    seed: int
    candidate_hash: str | None
    draws: tuple[Draw, ...] = ()
    algorithm: str = ALGORITHM
    key_version: str = KEY_VERSION

    @property
    def positions_m(self):
        return tuple(d.position_m for d in sorted(
            (d for d in self.draws if d.selected), key=lambda d: d.sorted_ordinal))

    @property
    def layout_id(self):
        # Includes realised layout/status; never used as a random key.
        from dataclasses import asdict
        return scientific_hash(asdict(self))


def generate_layout(rule, geometry, count, placement_index, population=None, seed=INFRA_SEED):
    """Draw U/J once per canonical pattern/p. Blocked results never conceal empties.

    ``population`` is a publication_candidates.Population. Its completeness is an
    explicit caller certification, separate from the number of observed candidates.
    """
    if rule not in {"U", "J"}:
        raise ValueError("Unknown placement rule")
    integer(placement_index, "placement index")
    integer(seed, "infrastructure seed")
    ch = population.sampling_hash if population is not None and rule == "J" else None

    def result(status, reasons=(), draws=()):
        return Layout(rule, status, tuple(reasons), geometry.pattern_id, geometry.version,
                      placement_index, seed, ch, tuple(draws))

    if not count.certified:
        return result("BLOCKED_UNKNOWN_N", ("complete potential encounter N not certified",))
    if count.value == 0:
        return result("EMPTY_N_ZERO")
    if not geometry.accepted:
        return result("BLOCKED_GEOMETRY", ("directed served-path geometry not accepted",))
    n = count.value
    if rule == "U":
        values = []
        for ordinal in range(n):
            key = ("infra-U-v1", seed, geometry.pattern_id, geometry.version,
                   placement_index, ordinal)
            u = stable_uniform(*key)
            # C02's binary64 conversion can round the largest integer to 1.0.
            # Preserve the half-open domain at that representation edge only.
            x = min(u * geometry.length_m, math.nextafter(geometry.length_m, 0.0))
            values.append((str(ordinal), key, u, x))
        ordered = sorted(values, key=lambda v: (v[3], int(v[0])))
        draws = [Draw(identity, key, u, x, True, j, geometry.receiving_interval(x))
                 for j, (identity, key, u, x) in enumerate(ordered)]
        return result("READY", draws=draws)
    if population is None:
        return result("BLOCKED_INCOMPLETE_CANDIDATES", ("candidate universe absent",))
    if population.pattern_id != geometry.pattern_id:
        raise ValueError("Candidate population belongs to a different canonical pattern")
    if not population.complete or population.unresolved_eligibility:
        return result("BLOCKED_INCOMPLETE_CANDIDATES", ("candidate universe not certified complete",))
    candidates = population.eligible
    if len(candidates) < n:
        return result("UNAVAILABLE_K_LT_N", (f"complete K={len(candidates)} < certified N={n}",))
    if any(c.position_status != "READY" for c in candidates):
        return result("BLOCKED_POSITIONS", ("eligible candidates retain unresolved/invalid positions",))
    ranks = []
    for c in candidates:
        # Validate every eligible position, including candidates not selected later.
        geometry.receiving_interval(c.position_m)
        key = ("infra-J-v1", seed, geometry.pattern_id, geometry.version,
               placement_index, ch, c.candidate_id)
        ranks.append((stable_uniform(*key), c.candidate_id, key, c.position_m))
    ranks.sort(key=lambda v: (v[0], v[1]))
    selected = {v[1] for v in ranks[:n]}
    ordinals = {v[1]: j for j, v in enumerate(sorted(
        ranks[:n], key=lambda v: (v[3], v[1])))}
    draws = [Draw(cid, key, u, x, cid in selected, ordinals.get(cid),
                  geometry.receiving_interval(x)) for u, cid, key, x in ranks]
    return result("READY", draws=draws)


def pair_supplied_inventory(layout, supplied_inventory, *, full_inventory_accepted=False):
    """C08 handoff only: attach already supplied immutable C06 tuples ordinally.

    This function never constructs probability, behaviour uniforms, outcomes or waits.
    An unresolved inventory has no fabricated substitute identifiers.
    """
    if layout.status not in {"READY", "EMPTY_N_ZERO"}:
        raise ValueError("Cannot pair a blocked/unavailable layout")
    if not full_inventory_accepted or supplied_inventory is None:
        raise ValueError("Full ordered potential inventory unresolved")
    return supplied_placement(tuple(supplied_inventory), layout.positions_m)
