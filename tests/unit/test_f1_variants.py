"""Integrated C07 layouts and existing-F2 behaviour, with explicit fixture authority."""

from copy import deepcopy
import random

import numpy as np
import pandas as pd
import pytest

from beb_soc_model import MotionParams, VehicleParams, simulate_route
from publication_candidates import build_population
from publication_placement import EncounterCount, Geometry
from signal_placement import (
    PlacementContext,
    PlacementUnavailable,
    cached_layout,
    inventory_signature,
    source_inventory,
)
from test_c07_placement import record
from test_signal_fidelity import toy_trip
from trip_motion import solve_trip_motion


def accepted(segments, records=None, *, complete=True, count=None, geometry=None):
    geometry = geometry or Geometry(
        "fixture-pattern", "geometry-v1", 1000.0, (0.0, 500.0, 1000.0), True
    )
    count = count or EncounterCount(
        sum(s.n_signals for s in segments), True, "constructed complete fixture"
    )
    population = build_population(
        geometry,
        (
            records
            if records is not None
            else [
                record(str(i), x)
                for i, x in enumerate((100.0, 260.0, 500.0, 700.0, 900.0))
            ]
        ),
        complete=complete,
        completeness_authority="exhaustive constructed topology" if complete else "",
    )
    context = PlacementContext(
        geometry,
        count,
        population,
        inventory_signature(segments),
        "constructed accepted inventory",
        position_compatibility_authority="compatible constructed directed approach convention",
    )
    for segment in segments:
        segment.signal_placement_context = context
    return context


def parameters(fidelity="F1-U", **kwargs):
    return MotionParams(
        solver_mode="trip_continuous", signal_fidelity=fidelity, **kwargs
    )


def event_rows(trip):
    return [e.signal_decision for e in trip.events if e.kind == "signal"]


@pytest.mark.parametrize("fidelity", ["F1-U", "F1-J"])
def test_layout_shared_by_trips_and_behaviour_realisations(fidelity):
    source = toy_trip()
    accepted(source)
    first = solve_trip_motion(source, parameters(fidelity))
    other = deepcopy(source)
    for segment in other:
        segment.trip_id = "another-departure"
        segment.from_stop_departure_time = "27:00:00"
    # Inventory certification excludes trip/timetable, as do infrastructure keys.
    second = solve_trip_motion(
        other,
        parameters(fidelity, signal_behaviour_index=1, signal_behaviour_seeds={1: 23}),
    )
    assert [r["cumulative_distance_m"] for r in event_rows(first)] == [
        r["cumulative_distance_m"] for r in event_rows(second)
    ]
    assert [r["random_value"] for r in event_rows(first)] != [
        r["random_value"] for r in event_rows(second)
    ]
    assert len(event_rows(first)) == 4
    assert {r["layout_id"] for r in event_rows(first)} == {
        r["layout_id"] for r in event_rows(second)
    }


@pytest.mark.parametrize("fidelity", ["F1-U", "F1-J"])
@pytest.mark.parametrize("q", [0.0, 1.0])
def test_all_source_tuples_preserved_with_b1_and_receiving_physics(fidelity, q):
    source = toy_trip()
    accepted(source)
    before = deepcopy(source)
    mp = parameters(
        fidelity,
        stop_prob_by_hour={8: 0.0, 9: 1.0},
        gtfs_stop_policy="conditional_structural",
        q_stop=q,
    )
    _, expected = source_inventory(source, mp)
    result = solve_trip_motion(source, mp)
    rows = event_rows(result)
    assert len(rows) == 4
    for actual, reference in zip(rows, expected):
        for key in (
            "source_ordinal",
            "source_encounter_id",
            "source_scheduled_context",
            "source_segment_geometry_id",
            "source_random_key",
            "behaviour_key_algorithm",
            "paired_F2_event_id",
            "paired_signal_hour",
            "paired_probability_source",
            "stop_probability",
            "random_value",
            "stop_decision",
            "conditional_wait_s",
            "red_wait_s_requested",
        ):
            assert actual[key] == reference[key]
        receiver = actual["receiving_interval"]
        assert result.intervals[receiver].segment.grade == before[receiver].grade
        assert (
            result.intervals[receiver].segment.speed_cap_ms
            == before[receiver].speed_cap_ms
        )
    assert sum(row["stop_decision"] for row in rows) == 2
    assert sum(row["red_wait_s_requested"] for row in rows) == 30.0
    assert sum(d["signal_wait_applied_s"] for d in result.segment_diagnostics) == 30.0
    assert [s.signal_events for s in source] == [s.signal_events for s in before]


def test_uniform_across_entire_path_ties_boundary_ownership_and_close_controls(
    monkeypatch,
):
    import publication_placement as placement

    source = toy_trip()
    accepted(source)
    cached_layout.cache_clear()
    monkeypatch.setattr(placement, "stable_uniform", lambda *args: 0.5)
    result = solve_trip_motion(
        source, parameters(stop_prob=1.0, use_hourly_signal_stop_probability=False)
    )
    assert [d["n_signals"] for d in result.segment_diagnostics] == [0, 4]
    assert [r["receiving_interval"] for r in event_rows(result)] == [1] * 4
    assert sum(d["signal_wait_applied_s"] for d in result.segment_diagnostics) == 60.0
    assert len({r["signal_event_id"] for r in event_rows(result)}) == 4
    cached_layout.cache_clear()


@pytest.mark.parametrize(
    "state,status",
    [
        ("short", "UNAVAILABLE_K_LT_N"),
        ("incomplete", "BLOCKED_INCOMPLETE_CANDIDATES"),
        ("positions", "BLOCKED_POSITIONS"),
        ("unknown", "BLOCKED_UNKNOWN_N"),
        ("geometry", "BLOCKED_GEOMETRY"),
    ],
)
def test_j_readiness_never_silently_falls_back(state, status):
    source = toy_trip()
    rows = [
        record(str(i), x) for i, x in enumerate((100.0, 260.0, 500.0, 700.0, 900.0))
    ]
    if state == "short":
        rows = rows[:2]
    if state == "positions":
        rows[0].pop("position_evidence")
    geometry = Geometry(
        "fixture-pattern",
        "geometry-v1",
        1000.0,
        (0.0, 500.0, 1000.0),
        state != "geometry",
    )
    count = EncounterCount(None, False, "unknown") if state == "unknown" else None
    accepted(
        source, rows, complete=state != "incomplete", count=count, geometry=geometry
    )
    with pytest.raises(PlacementUnavailable) as exc:
        solve_trip_motion(source, parameters("F1-J"))
    assert exc.value.status == status
    # Independent F0/F2 software execution remains available.
    for fidelity in ("F0_none", "F2_spatial"):
        assert (
            len(
                simulate_route(
                    deepcopy(source),
                    VehicleParams(),
                    motion_params=parameters(fidelity),
                )
            )
            == 2
        )


@pytest.mark.parametrize("fidelity", ["F1-U", "F1-J"])
def test_certified_zero_is_empty_without_any_random_draws(monkeypatch, fidelity):
    import publication_placement as placement

    source = toy_trip()
    for segment in source:
        segment.signal_events, segment.n_signals = [], 0
    accepted(source, [])
    cached_layout.cache_clear()
    monkeypatch.setattr(
        placement, "stable_uniform", lambda *args: pytest.fail("N=0 must not draw")
    )
    result = simulate_route(source, VehicleParams(), motion_params=parameters(fidelity))
    assert result.n_signals.sum() == 0
    assert result.placement_status.eq("EMPTY_N_ZERO").all()


def test_f2_positions_and_signal_labels_cannot_change_new_layouts():
    source = toy_trip()
    first = accepted(source)
    for rule in ("U", "J"):
        a = cached_layout(
            rule, first.geometry, first.count, 0, first.population, 20260801
        )
        changed = deepcopy(source)
        for segment in changed:
            for event, x in zip(segment.signal_events, (2.0, 498.0)):
                event.update(distance_along_segment_m=x, signal_lat=0.0, signal_lon=0.0)
        records = [
            record(str(i), x, traffic_signals=False, F2_member=False)
            for i, x in enumerate((100.0, 260.0, 500.0, 700.0, 900.0))
        ]
        second = accepted(changed, list(reversed(records)))
        b = cached_layout(
            rule, second.geometry, second.count, 0, second.population, 20260801
        )
        assert first.population.sampling_hash == second.population.sampling_hash
        assert a == b


def test_call_order_preserves_f0_f2_values_inputs_and_global_random_state():
    source = toy_trip()
    accepted(source)
    baseline = {
        f: simulate_route(
            deepcopy(source), VehicleParams(), motion_params=parameters(f)
        )
        for f in ("F0_none", "F2_spatial")
    }
    state, numpy_state = random.getstate(), np.random.get_state()
    for fidelity in ("F1-J", "F1-U", "F1-J"):
        simulate_route(
            deepcopy(source), VehicleParams(), motion_params=parameters(fidelity)
        )
    assert random.getstate() == state
    after_state = np.random.get_state()
    assert numpy_state[0] == after_state[0]
    np.testing.assert_array_equal(numpy_state[1], after_state[1])
    assert numpy_state[2:] == after_state[2:]
    for fidelity, expected in baseline.items():
        pd.testing.assert_frame_equal(
            simulate_route(
                deepcopy(source), VehicleParams(), motion_params=parameters(fidelity)
            ),
            expected,
            check_exact=True,
        )


def test_unknown_and_changed_inventory_are_not_accepted_zero():
    source = toy_trip()
    with pytest.raises(PlacementUnavailable, match="BLOCKED_UNKNOWN_N"):
        solve_trip_motion(source, parameters())
    accepted(source)
    source[0].signal_events[0]["distance_along_segment_m"] += 1
    with pytest.raises(PlacementUnavailable, match="BLOCKED_INVENTORY"):
        solve_trip_motion(source, parameters())


def test_approach_convention_and_canonical_boundaries_require_registration():
    from dataclasses import replace

    source = toy_trip()
    context = accepted(source)
    for segment in source:
        segment.signal_placement_context = replace(
            context, position_compatibility_authority=""
        )
    with pytest.raises(PlacementUnavailable, match="BLOCKED_POSITION_CONVENTION"):
        solve_trip_motion(source, parameters("F1-J"))
    # U does not use a junction approach convention.
    assert len(event_rows(solve_trip_motion(source, parameters("F1-U")))) == 4
    source[0].length_m += 1
    with pytest.raises(
        PlacementUnavailable, match="BLOCKED_INVENTORY|BLOCKED_GEOMETRY"
    ):
        solve_trip_motion(source, parameters())


def test_config_selectors_and_behaviour_registration(tmp_path):
    from best_ire_beb.config import validate_config

    path = tmp_path / "variants.yaml"
    path.write_text(
        "motion:\n  solver_mode: trip_continuous\n  signal_fidelity: F1-J\n  signal_placement_index: 1\n  signal_behaviour_index: 2\n  signal_behaviour_seeds: {2: 41}\n",
        encoding="utf-8",
    )
    validate_config(path)
    mp = MotionParams.from_config(path)
    assert mp.signal_fidelity == "F1-J" and mp.signal_placement_index == 1
    assert mp.behaviour_seed() == 41
    with pytest.raises(ValueError, match="explicit|registration"):
        parameters(signal_behaviour_index=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        parameters(signal_placement_index=True)
