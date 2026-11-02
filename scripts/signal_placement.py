"""C07 placement and read-only working-F2 tuple adapter for the normal runner.

An explicit registry binds accepted geometry/count/inventory to actual inputs.
Cache counts and provisional shapes never certify themselves. No network access.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from publication_candidates import audited_coverage, build_population
from publication_placement import (
    EncounterCount,
    Geometry,
    generate_layout,
    pair_supplied_inventory,
    scientific_hash,
)
from publication_timing import SignalTuple

F1_VARIANTS = frozenset({"F1-U", "F1-J"})
READY = frozenset({"READY", "EMPTY_N_ZERO"})


class PlacementUnavailable(ValueError):
    def __init__(self, status, reason, diagnostics=None):
        self.status = status
        self.reason = reason
        self.diagnostics = diagnostics or {}
        super().__init__(f"{status}: {reason}")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_signature(segments):
    """Full potential inventory identity, not a placement random key."""
    from beb_soc_model import _signal_events_for_segment

    rows = []
    for i, seg in enumerate(segments):
        if seg.signal_events is None or seg.signal_source == "fallback":
            raise PlacementUnavailable(
                "BLOCKED_INVENTORY",
                "complete F2 event lists required, including explicit empty lists",
            )
        events = _signal_events_for_segment(seg)
        if len(events) != int(seg.n_signals or 0):
            raise PlacementUnavailable(
                "BLOCKED_INVENTORY", "F2 count and full potential inventory disagree"
            )
        ids = [e.get("signal_event_id") for e in events]
        if any(not identity for identity in ids) or len(set(ids)) != len(ids):
            raise PlacementUnavailable(
                "BLOCKED_INVENTORY",
                "encounter IDs must be unique within each traversal interval",
            )
        rows.append(
            {
                "interval": i,
                "length_m": seg.length_m,
                "geometry_id": seg.segment_geometry_id,
                "events": sorted(
                    events,
                    key=lambda e: (e["distance_along_segment_m"], e["signal_event_id"]),
                ),
            }
        )
    return scientific_hash(rows)


@dataclass(frozen=True)
class PlacementContext:
    geometry: Geometry
    count: EncounterCount
    population: object = None
    inventory_sha256: str | None = None
    inventory_authority: str = ""
    coverage: object = None
    registry_sha256: str | None = None
    position_compatibility_authority: str = ""
    candidate_error: str = ""

    def check_inventory(self, segments):
        if not self.count.certified:
            raise PlacementUnavailable(
                "BLOCKED_UNKNOWN_N", "complete accepted potential N is not certified"
            )
        signature = inventory_signature(segments)
        if not self.inventory_authority or self.inventory_sha256 != signature:
            raise PlacementUnavailable(
                "BLOCKED_INVENTORY",
                "source F2 inventory is not the registered accepted version",
            )
        if sum(s.n_signals for s in segments) != self.count.value:
            raise PlacementUnavailable(
                "BLOCKED_INVENTORY",
                "registered N differs from the complete potential inventory",
            )
        boundaries = [0.0]
        for segment in segments:
            boundaries.append(boundaries[-1] + segment.length_m)
        if len(boundaries) != len(self.geometry.boundaries_m) or any(
            not math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
            for a, b in zip(boundaries, self.geometry.boundaries_m)
        ):
            raise PlacementUnavailable(
                "BLOCKED_GEOMETRY",
                "served-path intervals differ from registered geometry; no clipping or rescaling",
            )


_LAYOUT_CACHE = {}


def cached_layout(rule, geometry, count, placement_index, population, seed):
    # Cache the scientific layout/readiness inputs, not evidence prose or labels.
    key = (
        rule,
        geometry,
        count.value,
        count.certified,
        placement_index,
        seed,
        (
            (
                population.sampling_hash,
                population.complete,
                population.unresolved_eligibility,
            )
            if rule == "J" and population is not None
            else None
        ),
    )
    if key not in _LAYOUT_CACHE:
        if len(_LAYOUT_CACHE) >= 4096:
            _LAYOUT_CACHE.pop(next(iter(_LAYOUT_CACHE)))
        _LAYOUT_CACHE[key] = generate_layout(
            rule,
            geometry,
            count,
            placement_index,
            population if rule == "J" else None,
            seed,
        )
    return _LAYOUT_CACHE[key]


cached_layout.cache_clear = _LAYOUT_CACHE.clear


def source_inventory(segments, mp):
    """Replay working F2 before relocation; retain every pass and source context."""
    inventory_signature(segments)
    from beb_soc_model import (
        _signal_events_for_segment,
        _signal_stop_prob_for_segment,
        _spatial_signal_decisions,
    )

    ordered, offset = [], 0.0
    for i, seg in enumerate(segments):
        p, hour, source = _signal_stop_prob_for_segment(
            seg, mp.stop_prob, mp.hour_table_or_none()
        )
        decisions = _spatial_signal_decisions(
            seg, _signal_events_for_segment(seg), p, mp.behaviour_seed(), mp.red_wait_s
        )
        for decision in decisions:
            identity = decision["signal_event_id"]
            x = offset + decision["distance_along_segment_m"]
            ordered.append(
                (
                    x,
                    identity,
                    i,
                    {
                        **decision,
                        "paired_F2_event_id": identity,
                        "paired_F2_segment": i,
                        "paired_signal_hour": hour,
                        "paired_probability_source": source,
                        "source_scheduled_context": seg.from_stop_departure_time,
                        "source_segment_geometry_id": seg.segment_geometry_id,
                        "source_random_key": (
                            mp.behaviour_seed(),
                            seg.service_date,
                            seg.trip_id,
                            seg.segment_geometry_id,
                            seg.segment_index,
                            seg.from_stop_departure_time,
                            identity,
                        ),
                        "behaviour_key_algorithm": "working-F2-pipe-blake2s64-v1",
                        "source_position_m": x,
                        "source_encounter_id": f"{i}:{identity}",
                        "conditional_wait_s": max(float(mp.red_wait_s), 0.0),
                    },
                )
            )
        offset += seg.length_m
    ordered.sort(key=lambda row: row[:3])
    tuples, metadata = [], []
    for ordinal, (_, identity, i, event) in enumerate(ordered):
        event["source_ordinal"] = ordinal
        tuples.append(
            SignalTuple(
                ordinal,
                event["source_encounter_id"],
                event["source_scheduled_context"] or "missing_scheduled_context",
                event["stop_probability"],
                event["random_value"],
                event["stop_decision"],
                event["conditional_wait_s"],
                event["source_random_key"],
            )
        )
        metadata.append(event)
    return tuple(tuples), metadata


def prepared_events(segments, mp):
    context = getattr(segments[0], "signal_placement_context", None)
    if context is None:
        raise PlacementUnavailable(
            "BLOCKED_UNKNOWN_N", "no accepted canonical-pattern registry attached"
        )
    context.check_inventory(segments)
    layout = cached_layout(
        mp.signal_fidelity[-1],
        context.geometry,
        context.count,
        mp.signal_placement_index,
        context.population,
        mp.signal_location_seed,
    )
    diagnostics = {
        "pattern_id": context.geometry.pattern_id,
        "geometry_version": context.geometry.version,
        "N": context.count.value,
        "layout_id": layout.layout_id,
        "placement_status": layout.status,
        "placement_rule": layout.rule,
        "placement_index": mp.signal_placement_index,
        "behaviour_index": mp.signal_behaviour_index,
        "candidate_hash": layout.candidate_hash,
        "registry_sha256": context.registry_sha256,
        "inventory_sha256": context.inventory_sha256,
        "algorithm": layout.algorithm,
        "key_version": layout.key_version,
        "candidate_error": context.candidate_error,
        **(
            context.population.diagnostics(context.count)
            if context.population
            else {"K": None, "N_over_K": None, "candidate_population_complete": False}
        ),
    }
    if layout.status not in READY:
        raise PlacementUnavailable(
            layout.status,
            "; ".join((*layout.reasons, context.candidate_error)).strip("; "),
            diagnostics,
        )
    if (
        layout.rule == "J"
        and context.count.value
        and not context.position_compatibility_authority
    ):
        raise PlacementUnavailable(
            "BLOCKED_POSITION_CONVENTION",
            "F1-J/F2 directed approach convention compatibility is not registered",
            diagnostics,
        )
    sources, metadata = source_inventory(segments, mp)
    placed = pair_supplied_inventory(layout, sources, full_inventory_accepted=True)
    draws = sorted(
        (d for d in layout.draws if d.selected), key=lambda d: d.sorted_ordinal
    )
    events = []
    for draw, position, source in zip(draws, placed, metadata):
        owner = context.geometry.receiving_interval(position.position_m)
        events.append(
            {
                **source,
                **diagnostics,
                "signal_event_id": f"{mp.signal_fidelity}:{layout.layout_id}:{draw.sorted_ordinal}",
                "source": mp.signal_fidelity,
                "signal_lat": None,
                "signal_lon": None,
                "signal_location_seed": mp.signal_location_seed,
                "signal_ordinal": draw.sorted_ordinal,
                "synthetic_ordinal": draw.sorted_ordinal,
                "candidate_id": draw.identity if layout.rule == "J" else None,
                "cumulative_distance_m": position.position_m,
                "assigned_segment": owner,
                "receiving_interval": owner,
                "distance_along_segment_m": position.position_m
                - context.geometry.boundaries_m[owner],
                "red_wait_s_requested": position.source.wait_s,
                "segment_geometry_id": segments[owner].segment_geometry_id,
            }
        )
    return events, diagnostics


class PlacementRegistry:
    """Versioned local evidence; source binding is separate from scientific keys."""

    def __init__(self, path=None):
        self.sha256 = file_hash(path) if path else None
        data = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        if data and data.get("schema") != "F1-placement-registry-v1":
            raise ValueError("Expected F1-placement-registry-v1")
        self.patterns = data.get("patterns", [])
        identities = [r["geometry"]["pattern_id"] for r in self.patterns]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate canonical pattern registration")

    def pattern_id(self, binding):
        matches = [r for r in self.patterns if binding in r.get("bindings", [])]
        if len(matches) > 1:
            raise ValueError("Source traversal has multiple canonical registrations")
        return (
            matches[0]["geometry"]["pattern_id"]
            if matches
            else "unregistered:" + scientific_hash(binding)[:24]
        )

    def bind(
        self, segments, *, feed_sha256, route_id, direction_id, shape_id, stop_visits
    ):
        binding = {
            "feed_sha256": feed_sha256,
            "route_id": str(route_id),
            "direction_id": str(direction_id),
            "shape_id": str(shape_id),
            "stop_visits": [str(s) for s in stop_visits],
        }
        matches = [r for r in self.patterns if binding in r.get("bindings", [])]
        if len(matches) > 1:
            raise ValueError("Source traversal has multiple canonical registrations")
        if not matches:
            raise PlacementUnavailable(
                "BLOCKED_UNKNOWN_N",
                "no accepted registration for this feed/direction/shape/ordered stop traversal",
            )
        row = matches[0]
        if len(stop_visits) != len(segments) + 1 and not row.get(
            "zero_length_interval_authority"
        ):
            raise PlacementUnavailable(
                "BLOCKED_ZERO_LENGTH_INTERVALS",
                "filtered/repeated zero-length visits require explicit source-interval registration",
            )
        geometry = Geometry(**row["geometry"])
        count = EncounterCount(**row["count"])
        candidate = row.get("candidates", {})
        candidate_error = ""
        try:
            population = build_population(
                geometry,
                candidate.get("records", []),
                complete=candidate.get("complete", False),
                completeness_authority=candidate.get("authority", ""),
            )
        except ValueError as exc:
            # A J evidence defect does not invalidate a certified U layout.
            population, candidate_error = None, str(exc)
        try:
            coverage = (
                audited_coverage(
                    population,
                    row.get("f2_coverage_records", []),
                    inventory_complete=row.get("coverage_inventory_complete", False),
                )
                if population is not None
                else {
                    "status": "unresolved_candidate_registry",
                    "reason": candidate_error,
                }
            )
        except (ValueError, KeyError) as exc:
            # Coverage is diagnostic and must not control selection.
            coverage = {"status": "unresolved_coverage", "reason": str(exc)}
        context = PlacementContext(
            geometry,
            count,
            population,
            row.get("inventory_sha256"),
            row.get("inventory_authority", ""),
            coverage,
            self.sha256,
            row.get("position_compatibility_authority", ""),
            candidate_error,
        )
        context.check_inventory(segments)
        return context


def context_evidence(context, mp):
    """Retain rejected/unresolved records even when no J layout can be executed."""
    layout = cached_layout(
        mp.signal_fidelity[-1],
        context.geometry,
        context.count,
        mp.signal_placement_index,
        context.population,
        mp.signal_location_seed,
    )
    return {
        "geometry": asdict(context.geometry),
        "count": asdict(context.count),
        "layout": asdict(layout),
        "registry_sha256": context.registry_sha256,
        "candidate_registry": (
            [asdict(c) for c in context.population.candidates]
            if context.population
            else []
        ),
        "candidate_diagnostics": (
            context.population.diagnostics(context.count)
            if context.population
            else {
                "K": None,
                "N_over_K": None,
                "candidate_error": context.candidate_error,
            }
        ),
        "candidate_record_counts": {
            kind: (
                sum(c.eligibility == kind for c in context.population.candidates)
                if context.population
                else None
            )
            for kind in ("ACCEPTED", "REJECTED", "UNRESOLVED")
        },
        "coverage": context.coverage,
        "position_compatibility_authority": context.position_compatibility_authority,
    }
