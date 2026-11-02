"""Matched normal-runner cells; no ensemble estimator or source refresh."""

from dataclasses import asdict, replace
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import pandas as pd

from publication_contract import GROUP_FIELDS, matched_contrasts
from signal_placement import file_hash, scientific_hash


def process_fidelity_comparison(gtfs_zip, route_short_names, output_dir, **kwargs):
    from gtfs_to_segment import process_routes
    from beb_soc_model import MotionParams

    mp = kwargs.get("motion_params") or MotionParams()
    if mp.solver_mode != "trip_continuous":
        raise ValueError("Matched comparison requires --solver-mode trip_continuous")
    if not kwargs.get("signals_enabled"):
        raise ValueError("Matched F1-U/J/F2 comparison requires --signals")
    if kwargs.get("signals_refresh") or kwargs.get("speed_caps_refresh"):
        raise ValueError(
            "Matched comparison forbids source refresh; prepare caches separately"
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, relative_paths = [], []
    # Bound all temporary output/copy paths to the caller's output directory.
    with tempfile.TemporaryDirectory(
        prefix=".fidelity-comparison-", dir=output_dir
    ) as temporary:
        temporary = Path(temporary).resolve()
        if temporary.parent != output_dir:
            raise ValueError("Unexpected temporary output path")
        # Each invariant F0/F2 runs once. Each trajectory keeps its own battery state.
        for fidelity in ("F2_spatial", "F0_none", "F1-U", "F1-J"):
            options = {
                **kwargs,
                "motion_params": replace(mp, signal_fidelity=fidelity),
                "run_records": records,
            }
            paths = process_routes(gtfs_zip, route_short_names, temporary, **options)
            relative_paths.extend(path.relative_to(temporary) for path in paths)
        cells = []
        for record in records:
            cells.append(
                {
                    **record,
                    "fidelity": {"F0_none": "F0", "F2_spatial": "F2"}.get(
                        record["fidelity"], record["fidelity"]
                    ),
                }
            )
        groups = [
            {
                **{k: record[k] for k in GROUP_FIELDS},
                "placement_rule": rule,
                "placement_index": mp.signal_placement_index,
                "behaviour_index": mp.signal_behaviour_index,
            }
            for record in records
            if record["fidelity"] == "F0_none"
            for rule in ("U", "J")
        ]
        root = Path(__file__).resolve().parents[1]
        sources = [
            *root.joinpath("scripts").glob("*.py"),
            *root.joinpath("src/best_ire_beb").glob("*.py"),
        ]
        inputs = [gtfs_zip, *kwargs.get("input_source_paths", ())]
        inputs += [
            kwargs.get(key)
            for key in (
                "signals_cache_path",
                "signal_events_cache_path",
                "speed_caps_cache_path",
                "placement_registry_path",
            )
        ]
        if kwargs.get("srtm_cache_dir"):
            inputs += list(Path(kwargs["srtm_cache_dir"]).glob("*.hgt"))
        source_files = {
            str(Path(path).resolve()): file_hash(path)
            for path in inputs
            if path and Path(path).is_file()
        }
        identity = {
            "schema": "F1-U-J-matched-v1",
            "source_hashes": {
                path.relative_to(root).as_posix(): file_hash(path)
                for path in sorted(sources)
            },
            "python": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "pyyaml")
            },
            "motion": asdict(mp),
            "routes": list(route_short_names),
            "input_source_hashes": source_files,
            "cells": [
                {
                    k: row[k]
                    for k in (
                        "feed_sha256",
                        "route_id",
                        "pattern_id",
                        "trip_id",
                        "service_date",
                        "fidelity",
                        "placement_index",
                        "behaviour_index",
                        "analysis_mode",
                        "non_signal_input_signature",
                        "source_signature",
                        "registry_sha256",
                    )
                }
                for row in records
            ],
        }
        run_id = scientific_hash(identity)
        for cell in cells:
            cell["run_id"] = scientific_hash(
                {"execution_id": run_id, "cell_id": cell["run_id"]}
            )
            cell["execution_id"] = run_id
        contrasts = matched_contrasts(cells, groups)
        # The full identity stays in every cell/manifest. A checked short directory
        # keeps date-stamped route/event exports usable with Windows MAX_PATH.
        destination = output_dir / "matched" / run_id[:24]
        if not destination.is_relative_to(output_dir):
            raise ValueError("Comparison output escaped its parent")
        destination.mkdir(parents=True, exist_ok=True)
        existing_manifest = destination / "run_manifest.json"
        if (
            existing_manifest.exists()
            and json.loads(existing_manifest.read_text(encoding="utf-8"))["run_id"]
            != run_id
        ):
            raise ValueError("Comparison output identity prefix collision")
        (temporary / "raw_cells.json").write_text(
            json.dumps(cells, indent=2, sort_keys=True), encoding="utf-8"
        )
        (temporary / "matched_contrasts.json").write_text(
            json.dumps(contrasts, indent=2, sort_keys=True), encoding="utf-8"
        )
        pd.DataFrame(contrasts).to_csv(temporary / "matched_contrasts.csv", index=False)
        (temporary / "run_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    **identity,
                    "expected_trip_cells": len(records),
                    "matched_contrast_cells": len(groups),
                    "accepted_real_route_evidence": False,
                    "interpretation": "Working normal-runner software comparison; real input/protocol acceptance is separate.",
                    "behaviour_contract": "working F2 keys replayed unchanged; explicit seed registration for nonzero b",
                    "replication": "F0 and F2 are shared references, not independent p replicates",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        for source in temporary.rglob("*"):
            if source.is_file():
                target = destination / source.relative_to(temporary)
                target.parent.mkdir(parents=True, exist_ok=True)
                # CopyFile2 (used by copy2) can fail at MAX_PATH on Windows.
                # Export bytes only; filesystem timestamps are not result inputs.
                source_name, target_name = str(source), str(target)
                if os.name == "nt":
                    source_name = "\\\\?\\" + source_name
                    target_name = "\\\\?\\" + target_name
                with (
                    open(source_name, "rb") as input_stream,
                    open(target_name, "wb") as output_stream,
                ):
                    shutil.copyfileobj(input_stream, output_stream)
        print(f"Matched comparison: {destination / 'matched_contrasts.csv'}")
        return [destination / path for path in relative_paths]
