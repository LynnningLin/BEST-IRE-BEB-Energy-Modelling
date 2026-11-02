"""C07 label-independent candidate registry, positions and diagnostic coverage.

The constructor consumes explicit directed structural evidence, never F2 fields.
Raw source hashes belong to provenance and are not scientific sampling keys.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math

from publication_placement import scientific_hash


POSITION_RULE = "C07-position-v1-independent-approach"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    pattern_id: str
    structural_id: str
    physical_id: str
    movement: tuple[str, str]
    occurrence: int
    layer: str
    kind: str
    eligibility: str
    reason: str
    raw_projection_m: float | None
    position_m: float | None
    offset_low_m: float | None
    offset_m: float | None
    offset_high_m: float | None
    position_low_m: float | None
    position_high_m: float | None
    position_status: str
    position_evidence: str
    raw_interval: int | None
    receiving_interval: int | None
    ordering_changed: bool
    source_records: tuple[str, ...]
    dedup_evidence: str


@dataclass(frozen=True)
class Population:
    pattern_id: str
    candidates: tuple[Candidate, ...]
    complete: bool
    completeness_authority: str

    def __post_init__(self):
        object.__setattr__(self, "candidates", tuple(self.candidates))
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != len(ids) or any(c.pattern_id != self.pattern_id for c in self.candidates):
            raise ValueError("Candidate IDs must be unique within their canonical pattern")
        if self.complete and (not self.completeness_authority or self.unresolved_eligibility):
            raise ValueError("Completeness cannot conceal unresolved eligibility")

    @property
    def eligible(self):
        return tuple(c for c in self.candidates if c.eligibility == "ACCEPTED")

    @property
    def unresolved_eligibility(self):
        return sum(c.eligibility == "UNRESOLVED" for c in self.candidates)

    @property
    def sampling_hash(self):
        # Only the eligible structural/position universe affects the ranks.
        # Raw tag bytes, review row IDs and source order are provenance-only.
        fields = ("candidate_id", "physical_id", "movement", "occurrence", "layer", "kind",
                  "raw_projection_m", "position_m", "offset_low_m", "offset_m", "offset_high_m",
                  "position_low_m", "position_high_m", "position_status")
        return scientific_hash({"rule": POSITION_RULE, "candidates": [
            {k: asdict(c)[k] for k in fields}
            for c in sorted(self.eligible, key=lambda c: c.candidate_id)]})

    def diagnostics(self, count):
        k = len(self.eligible)
        certified_k = self.complete and not self.unresolved_eligibility
        return {"observed_eligible": k, "K": k if certified_k else None,
                "candidate_population_complete": certified_k,
                "unresolved_eligibility": self.unresolved_eligibility,
                "unresolved_positions": sum(c.position_status != "READY" for c in self.eligible),
                "N_over_K": count.value / k if count.certified and certified_k and k else None}


def _eligibility(r):
    if r.get("service") in {"driveway", "parking_aisle"}:
        return "REJECTED", "excluded_" + r["service"]
    if r.get("kind") == "way_split":
        return "REJECTED", "simple_way_split"
    if r.get("kind") not in {"junction", "crossing"}:
        return "REJECTED", "no_independent_junction_or_crossing_evidence"
    if r.get("connected") is False or r.get("same_grade") is False:
        return "REJECTED", "grade_separated_or_disconnected"
    if r.get("traversed") is False or r.get("directed_legal") is False:
        return "REJECTED", "not_a_legal_traversed_movement"
    if r.get("road_class") == "service" and r.get("service") not in {
            "alley", "facility_access", "access_road", "emergency_access"}:
        return "UNRESOLVED", "service_road_classification_requires_review"
    required = ("connected", "same_grade", "traversed", "directed_legal", "restrictions_checked")
    if any(r.get(k) is not True for k in required):
        return "UNRESOLVED", "unproved_directed_connectivity_access_or_restrictions"
    return "ACCEPTED", "independent_connected_traversed_" + r["kind"]


def _position(r, geometry):
    raw = r.get("raw_projection_m")
    if raw is not None and not math.isfinite(raw):
        raise ValueError("Nonfinite raw projection")
    proposed = low = high = off = ol = oh = None
    evidence = r.get("position_evidence", "")
    if r.get("position_kind") == "independent_mapped_approach" and evidence:
        proposed = r.get("mapped_approach_m")
        uncertainty = r.get("position_uncertainty_m")
        if proposed is not None and uncertainty is not None:
            if not math.isfinite(uncertainty) or uncertainty < 0:
                raise ValueError("Invalid position uncertainty")
            low, high = proposed - uncertainty, proposed + uncertainty
    elif r.get("position_kind") == "centre" and evidence:
        ol, off, oh = (r.get(k) for k in ("offset_low_m", "offset_m", "offset_high_m"))
        if raw is not None and all(x is not None for x in (ol, off, oh)):
            if not all(math.isfinite(x) for x in (ol, off, oh)) or not 0 <= ol <= off <= oh:
                raise ValueError("Offset needs justified finite nonnegative low/base/high values")
            proposed, low, high = raw - off, raw - oh, raw - ol
    status = "BLOCKED_POSITIONS"
    owner = None
    if proposed is not None:
        if not math.isfinite(proposed):
            raise ValueError("Nonfinite proposed approach position")
        if not 0 <= proposed <= geometry.length_m:
            status = "OUT_OF_PATH"
        elif low is not None and high is not None:
            status = "READY"
            owner = geometry.receiving_interval(proposed)
    raw_owner = (geometry.receiving_interval(raw)
                 if raw is not None and 0 <= raw <= geometry.length_m else None)
    return raw, proposed, ol, off, oh, low, high, status, evidence, raw_owner, owner


def build_population(geometry, records, *, complete=False, completeness_authority=""):
    """Build from explicit evidence; ignore signal tags/F2/N/energy even if supplied.

    A shared ``equivalence_key`` consolidates records only with explicit dedup
    evidence. Otherwise distinct structural IDs survive at coincident positions.
    Conflicting representations are rejected for review, never chosen by row order.
    """
    groups = defaultdict(list)
    for original in records:
        r = dict(original)
        required = ("record_id", "structural_id", "physical_id", "incoming", "outgoing")
        if any(not isinstance(r.get(k), str) or not r[k] for k in required):
            raise ValueError("Stable source/physical/directed movement IDs required")
        occurrence = r.get("occurrence")
        if type(occurrence) is not int or occurrence < 1:
            raise ValueError("Traversal occurrence must be a positive integer")
        eq = r.get("equivalence_key")
        if eq and not r.get("dedup_evidence"):
            raise ValueError("Deduplication requires demonstrated same-control evidence")
        structural = eq or r["structural_id"]
        identity = (geometry.pattern_id, structural, r["physical_id"], r["incoming"],
                    r["outgoing"], occurrence, str(r.get("layer", "0")))
        groups[identity].append(r)
    candidates = []
    for identity, rows in sorted(groups.items()):
        evaluations = [(_eligibility(r), _position(r, geometry)) for r in rows]
        if any(v != evaluations[0] for v in evaluations[1:]):
            raise ValueError("Conflicting duplicate structural evidence requires review")
        if len(rows) > 1 and not all(r.get("dedup_evidence") for r in rows):
            raise ValueError("Repeated structural identity requires explicit duplicate evidence")
        r = rows[0]
        eligibility, position = evaluations[0]
        cid = "cand:" + scientific_hash(identity)[:24]
        candidates.append(Candidate(cid, geometry.pattern_id, identity[1], identity[2],
                                    (identity[3], identity[4]), identity[5], identity[6],
                                    r.get("kind", "unknown"), *eligibility, *position,
                                    False, tuple(sorted(r["record_id"] for r in rows)),
                                    " | ".join(sorted({r.get("dedup_evidence", "") for r in rows}))))
    # Flag every eligible order inversion against the raw geometric order.
    from dataclasses import replace
    resolved = [c for c in candidates if c.eligibility == "ACCEPTED"
                and c.position_status == "READY" and c.raw_projection_m is not None]
    raw_order = sorted(resolved, key=lambda c: (c.raw_projection_m, c.candidate_id))
    inverted = set()
    for i, c in enumerate(raw_order):
        for d in raw_order[i+1:]:
            if c.position_m > d.position_m:
                inverted.update((c.candidate_id, d.candidate_id))
    return Population(geometry.pattern_id, tuple(replace(c, ordering_changed=c.candidate_id in inverted)
                                                 for c in candidates), complete, completeness_authority)


def audited_coverage(population, f2_records, *, inventory_complete=False, tolerance_m=2.0):
    """Multiplicity-aware maximum matching, post-construction; no sampling feedback.

    Records need independently reconciled physical/movement/occurrence/layer IDs,
    registered positions and uncertainty. Unsupported identities remain unresolved.
    """
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Invalid registration tolerance")
    records = sorted((dict(r) for r in f2_records), key=lambda r: r["encounter_id"])
    if len({r["encounter_id"] for r in records}) != len(records):
        raise ValueError("Duplicate F2 encounter identity")
    edges, unresolved = {}, set()
    for r in records:
        eid = r["encounter_id"]
        keys = ("physical_id", "incoming", "outgoing", "occurrence", "layer",
                "position_m", "uncertainty_m")
        if any(r.get(k) is None for k in keys) or not r.get("identity_accepted", False):
            unresolved.add(eid)
            edges[eid] = []
            continue
        x, u = r["position_m"], r["uncertainty_m"]
        if not math.isfinite(x) or not math.isfinite(u) or u < 0:
            raise ValueError("Invalid F2 position/uncertainty")
        options = []
        for c in population.eligible:
            identity = (c.physical_id == r["physical_id"] and c.movement == (r["incoming"], r["outgoing"])
                        and c.occurrence == r["occurrence"] and c.layer == str(r["layer"]))
            if not identity:
                continue
            if c.position_status != "READY":
                unresolved.add(eid)
            elif c.position_low_m - tolerance_m <= x + u and x - u <= c.position_high_m + tolerance_m:
                options.append(c.candidate_id)
        edges[eid] = sorted(options)
    assigned = {}

    def augment(eid, seen):
        for cid in edges[eid]:
            if cid in seen:
                continue
            seen.add(cid)
            if cid not in assigned or augment(assigned[cid], seen):
                assigned[cid] = eid
                return True
        return False

    for eid in edges:
        augment(eid, set())
    matched = set(assigned.values())
    physical_by_encounter = {r["encounter_id"]: r.get("physical_id") for r in records}
    return {"scope": "complete_F2_inventory" if inventory_complete else "supported_subset_only",
            "candidate_population_complete": population.complete,
            "matched_encounters": len(matched), "denominator_encounters": len(records),
            "matched_physical_sites": len({physical_by_encounter[eid] for eid in matched}),
            "denominator_physical_sites": len({r.get("physical_id") for r in records if r.get("physical_id")}),
            "unresolved_encounters": sorted(unresolved - matched),
            "unmatched_encounters": sorted(set(edges) - matched - unresolved),
            "matches": sorted((eid, cid) for cid, eid in assigned.items()),
            "complete_recall_certified": inventory_complete and population.complete and not unresolved}
