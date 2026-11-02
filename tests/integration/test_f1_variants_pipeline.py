"""Normal GTFS runner, accepted fixture registry and matched-cell exports."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pandas as pd
import pytest
import yaml

from beb_soc_model import MotionParams
from best_ire_beb.config import load_config
from fidelity_comparison import process_fidelity_comparison
import gtfs_to_segment as gtfs
from signal_placement import file_hash, inventory_signature


@pytest.fixture
def fixture_registry(
    tmp_path, integration_gtfs_zip, integration_tables, full_elevation, monkeypatch
):
    original = gtfs.segments_for_trip

    def with_inventory(*args, **kwargs):
        segments = original(*args, **kwargs)
        for i, segment in enumerate(segments):
            segment.segment_geometry_id = f"{kwargs.get('shape_id')}:{i}"
            segment.signal_events = [
                {
                    "signal_event_id": f"cached-{i}",
                    "distance_along_segment_m": segment.length_m * 0.25,
                    "source": "fixture",
                }
            ]
            segment.n_signals, segment.signal_source = 1, "osm"
        return segments

    monkeypatch.setattr(gtfs, "segments_for_trip", with_inventory)
    stats = dict(n_pairs=0, cache_hit=0, osm=0, fallback=0, total_signals=0)
    monkeypatch.setattr(
        gtfs, "resolve_signal_counts", lambda *args, **kwargs: ({}, {}, stats)
    )
    trips, stops = integration_tables[1:3]
    by_trip = gtfs.stop_times_for_trips(integration_gtfs_zip, trips.trip_id)
    shapes, shape_by_trip = gtfs.load_shapes_for_trips(
        integration_gtfs_zip, trips, set(by_trip)
    )
    patterns = []
    for trip in trips.to_dict(orient="records"):
        tid, shape_id = trip["trip_id"], shape_by_trip[trip["trip_id"]]
        segments = with_inventory(
            by_trip[tid],
            stops,
            elevation_data=full_elevation,
            shape_points=shapes[shape_id],
            shape_id=shape_id,
        )
        boundaries = [0.0]
        for segment in segments:
            boundaries.append(boundaries[-1] + segment.length_m)
        length = boundaries[-1]
        rows = []
        for i, fraction in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
            rows.append(
                dict(
                    record_id=f"row:{i}",
                    structural_id=str(i),
                    physical_id=f"node:{i}",
                    incoming="in",
                    outgoing="out",
                    occurrence=1,
                    layer="0",
                    kind="junction",
                    connected=True,
                    same_grade=True,
                    traversed=True,
                    directed_legal=True,
                    restrictions_checked=True,
                    road_class="residential",
                    raw_projection_m=fraction * length,
                    position_kind="independent_mapped_approach",
                    mapped_approach_m=fraction * length,
                    position_uncertainty_m=0.0,
                    position_evidence="constructed independent fixture geometry",
                )
            )
        patterns.append(
            dict(
                geometry=dict(
                    pattern_id=f"fixture:{tid}",
                    version="g1",
                    length_m=length,
                    boundaries_m=boundaries,
                    accepted=True,
                ),
                count=dict(
                    value=len(segments),
                    certified=True,
                    authority="constructed complete fixture",
                ),
                inventory_sha256=inventory_signature(segments),
                inventory_authority="constructed inventory",
                position_compatibility_authority="compatible constructed directed mapped approaches",
                zero_length_interval_authority="constructed colocated alias visit registration; existing positive intervals retained",
                bindings=[
                    dict(
                        feed_sha256=file_hash(integration_gtfs_zip),
                        route_id=str(trip["route_id"]),
                        direction_id=str(trip["direction_id"]),
                        shape_id=shape_id,
                        stop_visits=[row["stop_id"] for row in by_trip[tid]],
                    )
                ],
                candidates=dict(
                    records=rows,
                    complete=True,
                    authority="exhaustive constructed topology",
                ),
            )
        )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(dict(schema="F1-placement-registry-v1", patterns=patterns)),
        encoding="utf-8",
    )
    return registry, patterns


def comparison_options(
    integration_tables, full_elevation, integration_vehicle, registry
):
    return dict(
        tables=integration_tables,
        all_services=True,
        direction_id=None,
        simulation_level="trip",
        elevation_data=full_elevation,
        vehicle_params=integration_vehicle,
        signals_enabled=True,
        speed_caps_enabled=False,
        placement_registry_path=registry,
        signal_event_diagnostics_enabled=True,
        motion_params=MotionParams(
            solver_mode="trip_continuous", signal_fidelity="F2_spatial"
        ),
    )


@pytest.mark.parametrize("short_j", [False, True])
def test_matched_runner_retains_every_trip_and_unavailable_j(
    tmp_path,
    fixture_registry,
    integration_gtfs_zip,
    integration_tables,
    full_elevation,
    integration_vehicle,
    short_j,
):
    registry, patterns = fixture_registry
    if short_j:
        for pattern in patterns:
            pattern["candidates"]["records"] = []
        registry.write_text(
            json.dumps(dict(schema="F1-placement-registry-v1", patterns=patterns)),
            encoding="utf-8",
        )
    options = comparison_options(
        integration_tables, full_elevation, integration_vehicle, registry
    )
    paths = process_fidelity_comparison(
        integration_gtfs_zip, ["208"], tmp_path / "results", **options
    )
    assert len(paths) == 4
    manifest = next((tmp_path / "results").glob("matched/*/run_manifest.json"))
    cells = json.loads(manifest.with_name("raw_cells.json").read_text())
    contrasts = json.loads(manifest.with_name("matched_contrasts.json").read_text())
    assert len(cells) == 8 and len(contrasts) == 4
    assert all(
        len([r for r in cells if r["fidelity"] == f]) == 2
        for f in ("F0", "F1-U", "F1-J", "F2")
    )
    for contrast in contrasts:
        if short_j and contrast["placement_rule"] == "J":
            assert contrast["status"] == "incomplete"
            assert "K=0" in contrast["blocker"]
            continue
        assert contrast["status"] == "complete"
        assert contrast["exposure_kWh"] + contrast["location_kWh"] == pytest.approx(
            contrast["total_kWh"], abs=1e-12
        )
    for trip_id in ("t1", "t2"):
        same_trip = [
            r
            for r in cells
            if r["trip_id"] == trip_id
            and r["fidelity"] != "F0"
            and r["status"] == "complete"
        ]
        tuples = [
            [
                (
                    event["source_encounter_id"],
                    event["stop_probability"],
                    event["random_value"],
                    event["stop_decision"],
                    event["red_wait_s_requested"],
                )
                for event in r["event_ledger"]
            ]
            for r in same_trip
        ]
        assert all(row == tuples[0] for row in tuples)
    repeated = process_fidelity_comparison(
        integration_gtfs_zip, ["208"], tmp_path / "results", **options
    )
    assert repeated == paths
    assert json.loads(manifest.with_name("raw_cells.json").read_text()) == cells
    assert json.loads(manifest.read_text())["accepted_real_route_evidence"] is False


def test_missing_registry_exports_status_for_all_trips(
    tmp_path,
    fixture_registry,
    integration_gtfs_zip,
    integration_tables,
    full_elevation,
    integration_vehicle,
):
    options = comparison_options(
        integration_tables, full_elevation, integration_vehicle, None
    )
    options["motion_params"] = MotionParams(
        solver_mode="trip_continuous", signal_fidelity="F1-J"
    )
    paths = gtfs.process_routes(
        integration_gtfs_zip, ["208"], tmp_path / "missing", **options
    )
    frame = pd.read_csv(paths[0])
    assert len(frame) == 2
    assert frame.placement_status.eq("BLOCKED_UNKNOWN_N").all()
    assert "net_battery_energy_kWh" not in frame


def test_invalid_j_evidence_retains_u_and_baselines(
    tmp_path,
    fixture_registry,
    integration_gtfs_zip,
    integration_tables,
    full_elevation,
    integration_vehicle,
):
    registry, patterns = fixture_registry
    # Completeness cannot certify an unclassified service-road movement.
    patterns[0]["candidates"]["records"][0]["road_class"] = "service"
    registry.write_text(
        json.dumps(dict(schema="F1-placement-registry-v1", patterns=patterns)),
        encoding="utf-8",
    )
    options = comparison_options(
        integration_tables, full_elevation, integration_vehicle, registry
    )
    process_fidelity_comparison(
        integration_gtfs_zip, ["208"], tmp_path / "bad_j", **options
    )
    raw = next((tmp_path / "bad_j").glob("matched/*/raw_cells.json"))
    cells = json.loads(raw.read_text())
    assert all(
        row["status"] == "complete"
        for row in cells
        if row["fidelity"] in ("F0", "F1-U", "F2")
    )
    blocked = [row for row in cells if row["status"] != "complete"]
    assert len(blocked) == 1
    assert blocked[0]["fidelity"] == "F1-J"
    assert blocked[0]["status"] == "BLOCKED_INCOMPLETE_CANDIDATES"


def test_config_cli_to_f1_export(
    tmp_path, minimal_gtfs_zip, full_elevation, monkeypatch
):
    # No registry: execution must export explicit blocked identities, not run legacy F1.
    config = deepcopy(load_config())
    config["paths"].update(
        gtfs_zip=str(minimal_gtfs_zip()),
        gtfs_output_dir=str(tmp_path / "cli"),
        traffic_signals_csv=str(tmp_path / "signals.csv"),
    )
    config["gtfs"]["default_routes"] = ["208"]
    for section in ("passenger_loading", "weather", "speed_caps"):
        config[section]["enabled"] = False
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    import srtm
    import traffic_signals

    monkeypatch.setattr(srtm, "get_data", lambda **kwargs: full_elevation)
    stats = dict(n_pairs=0, cache_hit=0, osm=0, fallback=0, total_signals=0)
    monkeypatch.setattr(
        traffic_signals,
        "resolve_signal_counts",
        lambda *args, **kwargs: ({}, {}, stats),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gtfs_to_segment.py",
            "--config",
            str(config_path),
            "--solver-mode",
            "trip_continuous",
            "--signal-fidelity",
            "F1-U",
            "--placement-index",
            "1",
        ],
    )
    runpy.run_path(str(Path(gtfs.__file__)), run_name="__main__")
    paths = list((tmp_path / "cli").rglob("route_208_trips*.csv"))
    assert len(paths) == 1 and "traffic_signals_on_F1-U" in str(paths[0])
    assert "_p1_" in str(paths[0])
    assert pd.read_csv(paths[0]).placement_status.eq("BLOCKED_UNKNOWN_N").all()


@pytest.mark.parametrize("comparison", [False, True])
def test_accepted_fixture_cli_exports_new_variants(
    tmp_path,
    fixture_registry,
    integration_gtfs_zip,
    full_elevation,
    monkeypatch,
    comparison,
):
    registry, _ = fixture_registry
    config = deepcopy(load_config())
    config["paths"].update(
        gtfs_zip=str(integration_gtfs_zip),
        gtfs_output_dir=str(tmp_path / "accepted_cli"),
        traffic_signals_csv=str(tmp_path / "signals.csv"),
        traffic_signal_events_csv=str(tmp_path / "events.csv"),
    )
    config["gtfs"]["default_routes"] = ["208"]
    for section in ("passenger_loading", "weather", "speed_caps"):
        config[section]["enabled"] = False
    config_path = tmp_path / "accepted.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    import srtm
    import traffic_signals

    def attach(segments, *args, **kwargs):
        for i, segment in enumerate(segments):
            segment.segment_geometry_id = f"{kwargs.get('shape_id')}:{i}"
            segment.signal_events = [
                {
                    "signal_event_id": f"cached-{i}",
                    "distance_along_segment_m": segment.length_m * 0.25,
                    "source": "fixture",
                }
            ]
            segment.n_signals, segment.signal_source = 1, "osm"
        return segments

    stats = dict(n_pairs=0, cache_hit=0, osm=0, fallback=0, total_signals=0)
    monkeypatch.setattr(srtm, "get_data", lambda **kwargs: full_elevation)
    monkeypatch.setattr(traffic_signals, "add_traffic_signals", attach)
    monkeypatch.setattr(
        traffic_signals,
        "resolve_signal_counts",
        lambda *args, **kwargs: ({}, {}, stats),
    )
    arguments = [
        "gtfs_to_segment.py",
        "--config",
        str(config_path),
        "--solver-mode",
        "trip_continuous",
        "--signal-fidelity",
        "F1-J",
        "--signal-placement-registry",
        str(registry),
        "--simulation-level",
        "trip",
    ]
    if comparison:
        arguments.append("--compare-fidelities")
    monkeypatch.setattr(sys, "argv", arguments)
    runpy.run_path(str(Path(gtfs.__file__)), run_name="__main__")
    paths = list((tmp_path / "accepted_cli").rglob("route_208_trips*.csv"))
    assert len(paths) == (4 if comparison else 1)
    new_variants = [path for path in paths if "traffic_signals_on_F1-" in str(path)]
    assert len(new_variants) == (2 if comparison else 1)
    for path in new_variants:
        frame = pd.read_csv(path)
        assert frame.placement_status.eq("READY").all()
        assert frame.net_battery_energy_kWh.notna().all()
        assert len(list(path.parent.glob("*signal_events*.csv"))) == 1
