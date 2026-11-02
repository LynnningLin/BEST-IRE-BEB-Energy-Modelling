"""C07 structural, reproducibility and C06-interface risks; no energy ensemble."""
from dataclasses import asdict, replace
import copy
import math

import pytest

import publication_placement as pp
from publication_candidates import Population, audited_coverage, build_population
from publication_placement import EncounterCount, Geometry, generate_layout, pair_supplied_inventory
from publication_timing import Anchor, Road, SignalTuple, Visit, plan_span


def geom(**changes):
    return replace(Geometry("fixture-loop", "g1", 100, (0, 50, 100), True), **changes)


def count(n=2):
    return EncounterCount(n, True, "constructed complete fixture inventory")


def record(identity="a", x=30, **changes):
    r = dict(record_id="row:" + identity, structural_id=identity, physical_id="node:" + identity,
             incoming="edge:in", outgoing="edge:out", occurrence=1, layer="0", kind="junction",
             connected=True, same_grade=True, traversed=True, directed_legal=True,
             restrictions_checked=True, road_class="residential", raw_projection_m=x,
             position_kind="centre", offset_low_m=2., offset_m=5., offset_high_m=8.,
             position_evidence="constructed independent approach geometry; not a real offset")
    r.update(changes)
    return r


def pop(rows=None, complete=True):
    return build_population(geom(), rows if rows is not None else [record(), record("b", 70)],
                            complete=complete, completeness_authority="fixture exhaustive topology" if complete else "")


def test_independent_saved_c02_key_vectors():
    # Independently calculated compact UTF8 bytes / 8-byte big-endian digests.
    # U ordinals 0/1: 3108b1c389b406c2 / fabd6fde9ff27e69.
    layout = generate_layout("U", geom(), count(), 0)
    assert [d.uniform for d in layout.draws] == [3533269361059169986 / 2**64,
                                                18067720282037190249 / 2**64]
    assert layout.positions_m == (100 * (3533269361059169986 / 2**64),
                                  100 * (18067720282037190249 / 2**64))
    assert layout.draws[0].key == ("infra-U-v1", 20260801, "fixture-loop", "g1", 0, 0)
    assert pp.stable_uniform("infra-J-v1", 20260801, "p", "g", 0, "hash", "c") == 14885980218633677702 / 2**64


@pytest.mark.parametrize("n", [0, 1, 2, 12])
def test_u_exact_count_domain_and_reproducibility(n):
    a = generate_layout("U", geom(), count(n), 1)
    assert len(a.positions_m) == n
    assert a.positions_m == tuple(sorted(a.positions_m))
    assert all(0 <= x < 100 for x in a.positions_m)
    assert a == generate_layout("U", geom(), count(n), 1)
    assert a.status == ("EMPTY_N_ZERO" if n == 0 else "READY")


@pytest.mark.parametrize("rule", ["U", "J"])
def test_certified_zero_needs_no_geometry_or_draws(monkeypatch, rule):
    monkeypatch.setattr(pp, "stable_uniform", lambda *args: pytest.fail("zero must not draw"))
    r = generate_layout(rule, geom(accepted=False), count(0), 0)
    assert r.status == "EMPTY_N_ZERO" and r.draws == ()


@pytest.mark.parametrize("bad", [-1, 1.1, True, "2"])
def test_invalid_counts(bad):
    with pytest.raises(ValueError):
        EncounterCount(bad, True, "bad")


@pytest.mark.parametrize("bad", [-1, 1.1, True, "2"])
def test_invalid_indices(bad):
    with pytest.raises(ValueError):
        generate_layout("U", geom(), count(), bad)


@pytest.mark.parametrize("change", [dict(length_m=0), dict(length_m=math.inf),
                                   dict(boundaries_m=(0, 50, 50, 100)),
                                   dict(boundaries_m=(0, 100, 50)),
                                   dict(boundaries_m=(0, math.nan, 100)), dict(pattern_id="")])
def test_geometry_validation(change):
    with pytest.raises(ValueError):
        geom(**change)


def test_unknown_n_is_not_zero_and_geometry_is_separate():
    assert generate_layout("U", geom(), EncounterCount(None, False, "unknown"), 0).status == "BLOCKED_UNKNOWN_N"
    assert generate_layout("J", geom(), EncounterCount(0, False, "legacy zero"), 0).status == "BLOCKED_UNKNOWN_N"
    assert generate_layout("U", geom(accepted=False), count(), 0).status == "BLOCKED_GEOMETRY"
    # No caps, F2 coordinates or energy inputs occur in the U interface.
    assert generate_layout("U", geom(), count(), 0).status == "READY"


def test_uniform_close_tied_and_half_open_representation_edge(monkeypatch):
    monkeypatch.setattr(pp, "stable_uniform", lambda *args: .5)
    a = generate_layout("U", geom(), count(3), 0)
    assert a.positions_m == (50, 50, 50)
    assert [d.identity for d in a.draws] == ["0", "1", "2"]
    assert all(d.receiving_interval == 1 for d in a.draws)
    monkeypatch.setattr(pp, "stable_uniform", lambda *args: 1.)
    assert generate_layout("U", geom(), count(1), 0).positions_m == (math.nextafter(100., 0.),)


@pytest.mark.parametrize("n,k,status", [(0,0,"EMPTY_N_ZERO"),(1,0,"UNAVAILABLE_K_LT_N"),
                                       (2,1,"UNAVAILABLE_K_LT_N"),(2,2,"READY"),(1,2,"READY")])
def test_j_edges_and_without_replacement(n, k, status):
    population = pop([record(str(i), 20+i*10) for i in range(k)])
    a = generate_layout("J", geom(), count(n), 0, population)
    assert a.status == status
    if status == "READY":
        selected = [d.identity for d in a.draws if d.selected]
        assert len(set(selected)) == len(selected) == n
        assert len(a.draws) == k


def test_incomplete_k_never_proves_shortage_and_positions_do_not_shrink_k():
    assert generate_layout("J", geom(), count(5), 0, pop([record()], False)).status == "BLOCKED_INCOMPLETE_CANDIDATES"
    p = pop([record(offset_m=None)])
    assert p.diagnostics(count(1))["K"] == 1
    assert generate_layout("J", geom(), count(1), 0, p).status == "BLOCKED_POSITIONS"
    assert generate_layout("J", geom(), count(2), 0, p).status == "UNAVAILABLE_K_LT_N"
    assert pop([]).diagnostics(count(0))["N_over_K"] is None
    assert pop([record()], False).diagnostics(count(1))["N_over_K"] is None


@pytest.mark.parametrize("changes,eligibility,reason", [
    ({}, "ACCEPTED", "independent_connected"),
    ({"kind":"crossing"}, "ACCEPTED", "independent_connected"),
    ({"road_class":"service","service":"alley"}, "ACCEPTED", "independent_connected"),
    ({"road_class":"service","service":"facility_access"}, "ACCEPTED", "independent_connected"),
    ({"road_class":"service","service":"driveway"}, "REJECTED", "excluded_driveway"),
    ({"road_class":"service","service":"parking_aisle"}, "REJECTED", "excluded_parking"),
    ({"road_class":"service"}, "UNRESOLVED", "classification"),
    ({"kind":"way_split"}, "REJECTED", "simple_way_split"),
    ({"kind":"signal_node"}, "REJECTED", "independent_junction"),
    ({"connected":False}, "REJECTED", "disconnected"),
    ({"same_grade":False}, "REJECTED", "grade_separated"),
    ({"directed_legal":False}, "REJECTED", "legal_traversed"),
    ({"traversed":False}, "REJECTED", "legal_traversed"),
    ({"restrictions_checked":False}, "UNRESOLVED", "unproved"),
    ({"connected":None}, "UNRESOLVED", "unproved"),
])
def test_structural_eligibility(changes, eligibility, reason):
    c = pop([record(**changes)], False).candidates[0]
    assert c.eligibility == eligibility and reason in c.reason


def test_proven_dedup_distinct_coincident_and_loop_occurrences():
    rows = [record("a", 55, equivalence_key="control", dedup_evidence="same node/movement/control"),
            record("alias",55,physical_id="node:a",equivalence_key="control",dedup_evidence="same node/movement/control"),
            record("distinct",55), record("return",80,physical_id="node:a",occurrence=2)]
    p = pop(rows)
    assert len(p.eligible) == 3
    assert any(len(c.source_records) == 2 for c in p.eligible)
    assert generate_layout("J", geom(), count(3), 0, p).positions_m == (50,50,75)
    with pytest.raises(ValueError, match="Deduplication"):
        pop([record(equivalence_key="unsupported")])
    with pytest.raises(ValueError, match="duplicate"):
        pop([record(),record()])


def test_label_flip_f2_positions_row_order_and_irrelevant_input_invariance():
    original = [record(), record("b",70)]
    altered = copy.deepcopy(original)[::-1]
    for r in altered:
        r.update(signal_tags={"highway":"traffic_signals","direction":"backward"},
                 F2_position_m=99, N=1000, energy_kWh=-20, trip_id="other", departure="23:00",
                 probability=1, behaviour_uniform=0, cap=2, temperature=50)
    a,b = pop(original),pop(altered)
    assert a == b and a.sampling_hash == b.sampling_hash
    assert pp.scientific_hash(original) != pp.scientific_hash(altered)  # permitted provenance change
    for p in [0,1]:
        assert generate_layout("J", geom(), count(1), p, a) == generate_layout("J", geom(), count(1), p, b)


def test_stable_rank_ties_and_direction_independence(monkeypatch):
    population = pop([record("b",55),record("a",55)])
    monkeypatch.setattr(pp,"stable_uniform",lambda *args:.25)
    a = generate_layout("J",geom(),count(1),0,population)
    assert next(d.identity for d in a.draws if d.selected) == min(c.candidate_id for c in population.eligible)
    assert generate_layout("J",geom(),count(2),0,population).positions_m == (50,50)
    reverse = pop([record(incoming="edge:out",outgoing="edge:in")])
    assert reverse.eligible[0].candidate_id != pop([record()]).eligible[0].candidate_id


def test_layout_reuse_processing_order_batching_and_distinct_pattern_keys():
    a = {trip: generate_layout("U",geom(),count(),1) for trip in ["t2","t1","t3"]}
    b = {trip: generate_layout("U",geom(),count(),1) for batch in [["t1"],["t3","t2"]] for trip in batch}
    assert a == b and len({v.layout_id for v in a.values()}) == 1
    other = generate_layout("U",geom(pattern_id="opposite-pattern"),count(),1)
    assert a["t1"].positions_m != other.positions_m


def test_offsets_boundary_ownership_no_clipping_and_order_inversion():
    p = pop([record("a",52),record("b",54,offset_low_m=9,offset_m=10,offset_high_m=11)])
    assert all(c.ordering_changed for c in p.eligible)
    c = next(c for c in p.eligible if c.structural_id == "a")
    assert c.raw_interval == 1 and c.receiving_interval == 0 and c.position_m == 47
    out = pop([record(x=2)])
    assert out.eligible[0].position_m == -3 and out.eligible[0].position_status == "OUT_OF_PATH"
    assert generate_layout("J",geom(),count(1),0,out).status == "BLOCKED_POSITIONS"
    assert [geom().receiving_interval(x) for x in [0,49.999,50,100]] == [0,0,1,1]
    for x in [-1,101,math.nan]:
        with pytest.raises(ValueError):geom().receiving_interval(x)


def test_no_implicit_zero_offset_or_unsupported_stopline():
    c = pop([record(offset_m=None)]).eligible[0]
    assert c.position_m is None and c.position_status == "BLOCKED_POSITIONS"
    c = pop([record(position_evidence="")]).eligible[0]
    assert c.position_m is None
    assert pop([record()]).eligible[0].position_status == "READY"  # synthetic unsignalised approach allowed


def test_immutable_inputs_outputs_and_candidate_id_validation():
    rows = [record()]; before = copy.deepcopy(rows); p = pop(rows)
    assert rows == before
    rows[0]["raw_projection_m"] = 99
    assert p.eligible[0].raw_projection_m == 30
    with pytest.raises(Exception):p.eligible[0].position_m=4
    with pytest.raises(ValueError):Population(geom().pattern_id,(p.eligible[0],p.eligible[0]),True,"bad")
    with pytest.raises(ValueError):pop([record(occurrence=0)])
    with pytest.raises(ValueError):pop([record(offset_low_m=-1)])
    with pytest.raises(ValueError):pop([record(connected=None)],True)


def test_c08_roundtrip_all_potential_events_including_pass_and_coincident_waits():
    rows = [record(str(i),55) for i in range(3)]
    layout = generate_layout("J",geom(),count(3),0,pop(rows))
    inventory = tuple(SignalTuple(i,f"source:{i}","08h nominal",.5,.9 if i==1 else .1,
                                  i!=1,15+i,("supplied",i)) for i in range(3))
    placed = pair_supplied_inventory(layout,inventory,full_inventory_accepted=True)
    assert len(placed) == 3 and all(p.source is s for p,s in zip(placed,inventory))
    assert [p.position_m for p in placed] == [50,50,50]
    assert sum(p.source.wait_s for p in placed) == 32
    plan = plan_span([Road(0,100,10,10,0,"fixture",True)],
                     [Visit("origin",0,0,0),Visit("terminal",100,200,200)], placed,
                     Anchor(0,0,"departure"),Anchor(100,200,"arrival"))
    assert sum(p.duration_s for p in plan.periods if p.kind=="signal_wait") == 32
    assert len(plan.signals) == 3
    with pytest.raises(ValueError,match="unresolved"):
        pair_supplied_inventory(layout,None)
    with pytest.raises(ValueError):
        pair_supplied_inventory(layout,inventory[:2],full_inventory_accepted=True)


def test_coverage_multiplicity_identity_layer_occurrence_and_uncertainty():
    p = pop([record()])
    base = dict(encounter_id="f2:1",physical_id="node:a",incoming="edge:in",outgoing="edge:out",
                occurrence=1,layer="0",position_m=25,uncertainty_m=0,identity_accepted=True)
    a = audited_coverage(p,[base,dict(base,encounter_id="f2:2")])
    assert a["matched_encounters"] == 1 and a["denominator_encounters"] == 2
    assert a["scope"] == "supported_subset_only"
    for change in [dict(layer="1"),dict(occurrence=2),dict(physical_id="nearby-different"),dict(position_m=80)]:
        assert audited_coverage(p,[dict(base,**change)])["matched_encounters"] == 0
    a = audited_coverage(p,[dict(base,identity_accepted=False)])
    assert a["unresolved_encounters"] == ["f2:1"]
    before = p.sampling_hash
    audited_coverage(p,[dict(base,position_m=99)])
    assert p.sampling_hash == before


def test_independent_mapped_position_terminal_and_missing_uncertainty():
    r=record(position_kind="independent_mapped_approach",mapped_approach_m=100.,
             position_uncertainty_m=1.,position_evidence="independent mapped geometry")
    p=pop([r]);layout=generate_layout("J",geom(),count(1),0,p)
    assert layout.positions_m==(100.,) and layout.draws[0].receiving_interval==1
    assert pop([dict(r,position_uncertainty_m=None)]).eligible[0].position_status=="BLOCKED_POSITIONS"
    with pytest.raises(ValueError):pop([dict(r,position_uncertainty_m=-1)])


def test_key_types_and_invalid_certification():
    with pytest.raises(ValueError):pp.stable_uniform("infra-U-v1",20260801,"p",1.)
    with pytest.raises(ValueError):EncounterCount(None,True,"invented")
    with pytest.raises(ValueError):EncounterCount(1,True,"")
    with pytest.raises(ValueError):generate_layout("Q",geom(),count(),0)
    with pytest.raises(ValueError):generate_layout("J",geom(pattern_id="other"),count(),0,pop())
