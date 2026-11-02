"""C06 raw trip cells, explicit expected joins and contrasts before aggregation.

References to invariant F0/F2 are reused observations. No variance estimator or
ensemble execution is implemented here (C08/C12/C19).
"""

from __future__ import annotations

import hashlib
import json
import math


GROUP_FIELDS = ("feed_sha256", "service_date", "route_id", "pattern_id", "trip_id",
                "analysis_mode", "scenario_id", "timing_rule", "B1_stratum")
REQUIRED_INPUTS = {"feed_sha256", "service_date", "route_id", "pattern_id", "trip_id",
                   "scheduled_visits", "physical_roads", "stop_decisions", "scenario",
                   "initial_state", "numerical", "source_hashes", "runtime",
                   "signal_representation"}


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def input_signatures(inputs):
    missing = REQUIRED_INPUTS - inputs.keys()
    if missing:
        raise ValueError(f"Incomplete simulation identity: {sorted(missing)}")
    # Exclude exactly the declared signal representation object. Unrecognised
    # physical/config fields remain protected rather than disappearing silently.
    non_signal = {k: v for k, v in inputs.items() if k != "signal_representation"}
    return {"non_signal_input_signature": digest(non_signal), "input_hash": digest(inputs)}


def raw_cell(group, fidelity, result, inputs, *, placement_index=None,
             behaviour_index=None, status="complete", blocker=""):
    if fidelity not in ("F0", "F1-U", "F1-J", "F2"):
        raise ValueError("Unknown fidelity")
    if set(GROUP_FIELDS) - group.keys():
        raise ValueError("Incomplete trip/scenario group key")
    if fidelity in ("F0", "F2") and placement_index is not None:
        raise ValueError("Invariant reference cannot carry an infrastructure observation index")
    if fidelity.startswith("F1") and (placement_index is None or behaviour_index is None):
        raise ValueError("F1 cells require both applicable realisation indices")
    if fidelity == "F2" and behaviour_index is None:
        raise ValueError("F2 requires its unique behaviour index")
    for field in ("feed_sha256", "service_date", "route_id", "pattern_id", "trip_id"):
        if group[field] != inputs[field]:
            raise ValueError(f"Cell identity differs from input identity: {field}")
    if status not in ("complete", "failed", "blocked", "unavailable"):
        raise ValueError("Unknown cell execution status")
    if status != "complete" and not blocker:
        raise ValueError("Missing/failed cells need a visible blocker")
    if status == "complete":
        fields = ("traction_kWh", "auxiliary_kWh", "accepted_regen_kWh", "net_kWh",
                  "distance_m", "elapsed_s", "movement_s", "passenger_dwell_s",
                  "signal_wait_s", "other_holding_s", "initial_state", "final_state",
                  "incoming_lateness_s", "outgoing_lateness_s", "flags", "visits", "failure")
        if set(fields) - result.keys():
            raise ValueError("Incomplete energy/timing/state result")
        if result["failure"]:
            raise ValueError("Failed execution cannot be exported as complete")
        numbers = [result[k] for k in fields[:10]]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("Nonfinite completed cell")
        if abs(result["elapsed_s"] - sum(result[k] for k in
                ("movement_s", "passenger_dwell_s", "signal_wait_s", "other_holding_s"))) > 1e-6:
            raise ValueError("Trip ledger identity failed")
        if abs(result["net_kWh"] - result["traction_kWh"] - result["auxiliary_kWh"]
               + result["accepted_regen_kWh"]) > 1e-9:
            raise ValueError("Trip energy identity failed")
        if result["initial_state"] != inputs["initial_state"]:
            raise ValueError("Actual initial state differs from the hashed input state")
        if abs(result["initial_state"]["stored_kWh"] - result["final_state"]["stored_kWh"]
               - result["net_kWh"]) > 1e-8:
            raise ValueError("Full-precision battery state and net energy differ")
    signatures = input_signatures(inputs)
    identity = {**{k: group[k] for k in GROUP_FIELDS}, "fidelity": fidelity,
                "placement_index": placement_index, "behaviour_index": behaviour_index,
                **signatures}
    return {**identity, "run_id": digest(identity), "configuration_hash": digest(inputs["scenario"]),
            "status": status, "blocker": blocker, "result": result}


def group_key(row):
    return tuple(row[k] for k in GROUP_FIELDS)


def _percent(numerator, denominator):
    return 100 * numerator / denominator if denominator > 1e-9 else None


def matched_contrasts(raw_cells, expected_groups):
    """Join every requested U/J,p,b key; keep missing cases and J unavailability.

    expected_groups contain GROUP_FIELDS plus placement_rule, placement_index,
    behaviour_index. F0 with no b is reused only with an equal nonsignal signature;
    F2 is indexed by b. Duplicate cells/expected keys are errors.
    """
    index = {}
    for row in raw_cells:
        key = (group_key(row), row["fidelity"], row["placement_index"], row["behaviour_index"])
        if key in index:
            raise ValueError("Duplicate raw cell identity")
        index[key] = row
    allowed = set()
    for group in expected_groups:
        gk = group_key(group)
        p, b, rule = group["placement_index"], group["behaviour_index"], group["placement_rule"]
        allowed.update(((gk, "F0", None, None), (gk, "F0", None, b),
                        (gk, "F2", None, b), (gk, "F1-" + rule, p, b)))
    if set(index) - allowed:
        raise ValueError("Raw cells outside the declared expected cohort")
    output, seen = [], set()
    for group in expected_groups:
        gk = group_key(group)
        p, b, rule = group["placement_index"], group["behaviour_index"], group["placement_rule"]
        if rule not in ("U", "J") or (gk, rule, p, b) in seen:
            raise ValueError("Invalid/duplicate expected matched key")
        seen.add((gk, rule, p, b))
        invariant_f0 = index.get((gk, "F0", None, None))
        indexed_f0 = index.get((gk, "F0", None, b))
        if invariant_f0 is not None and indexed_f0 is not None:
            raise ValueError("Ambiguous invariant and behaviour-specific F0 references")
        f0 = indexed_f0 or invariant_f0
        f1 = index.get((gk, "F1-" + rule, p, b))
        f2 = index.get((gk, "F2", None, b))
        if len({cell["non_signal_input_signature"] for cell in (f0, f1, f2) if cell}) > 1:
            raise ValueError("Matched cases have unequal non-signal inputs")
        row = {**group, "F0_ref": f0["run_id"] if f0 else None,
               "F1_ref": f1["run_id"] if f1 else None, "F2_ref": f2["run_id"] if f2 else None}
        missing = [name for name, cell in (("F0", f0), ("F1-" + rule, f1), ("F2", f2))
                   if cell is None or cell["status"] != "complete"]
        if missing:
            row.update(status="incomplete", missing_cells=missing,
                       blocker="; ".join(cell["blocker"] for cell in (f0, f1, f2)
                                         if cell and cell["status"] != "complete"))
        else:
            if len({cell["non_signal_input_signature"] for cell in (f0, f1, f2)}) != 1:
                raise ValueError("Matched cases have unequal non-signal inputs")
            energies = [cell["result"]["net_kWh"] for cell in (f0, f1, f2)]
            distances = [cell["result"]["distance_m"] for cell in (f0, f1, f2)]
            if max(distances) - min(distances) > 1e-6:
                raise ValueError("Matched trip distances differ")
            e0, e1, e2 = energies
            x, y, z = e1 - e0, e2 - e1, e2 - e0
            tolerance = max(1e-9, 1e-12 * sum(abs(v) for v in energies))
            if abs(x + y - z) > tolerance:
                raise AssertionError("Exposure + location != total before aggregation")
            row.update(status="complete", E_F0_kWh=e0, E_F1_kWh=e1, E_F2_kWh=e2,
                       distance_km=distances[0] / 1000, exposure_kWh=x,
                       location_kWh=y, total_kWh=z, exposure_pct=_percent(x, e0),
                       location_pct=_percent(y, e1), total_pct=_percent(z, e0),
                       non_signal_input_signature=f0["non_signal_input_signature"])
        output.append(row)
    return output


def fixed_weight_summary(contrasts, trip_weights):
    """One declared cohort and one rule,p,b cell, checked before weighting.

    Route/day/ensemble callers must supply full expected trip weights and retain
    each rule/stratum separately. This returns no survivor mean for incomplete keys.
    """
    if not contrasts or len({(r["placement_rule"], r["placement_index"], r["behaviour_index"],
                             r["analysis_mode"], r["scenario_id"], r["timing_rule"], r["B1_stratum"])
                            for r in contrasts}) != 1:
        raise ValueError("Aggregate must contain one fixed matched realisation/stratum")
    rows = {group_key(r): r for r in contrasts}
    if len(rows) != len(contrasts) or set(rows) != set(trip_weights):
        raise ValueError("Aggregate keys do not equal the declared full cohort")
    if any(not math.isfinite(w) or w <= 0 for w in trip_weights.values()):
        raise ValueError("Weights must be predefined positive finite values")
    missing = [k for k, r in rows.items() if r["status"] != "complete"]
    if missing:
        return {"status": "incomplete", "expected_trips": len(rows),
                "complete_trips": len(rows) - len(missing), "missing_trip_keys": missing}
    total_weight = sum(trip_weights.values())
    sums = {name: sum(trip_weights[k] * r[name] for k, r in rows.items()) for name in
            ("E_F0_kWh", "E_F1_kWh", "E_F2_kWh", "exposure_kWh", "location_kWh", "total_kWh", "distance_km")}
    return {"status": "complete", "expected_trips": len(rows), "total_weight": total_weight,
            "weighted_totals": sums,
            "weighted_means": {k: v / total_weight for k, v in sums.items()},
            "F0_kWh_per_km": sums["E_F0_kWh"] / sums["distance_km"] if sums["distance_km"] > 0 else None,
            "F1_kWh_per_km": sums["E_F1_kWh"] / sums["distance_km"] if sums["distance_km"] > 0 else None,
            "F2_kWh_per_km": sums["E_F2_kWh"] / sums["distance_km"] if sums["distance_km"] > 0 else None,
            "exposure_pct": _percent(sums["exposure_kWh"], sums["E_F0_kWh"]),
            "location_pct": _percent(sums["location_kWh"], sums["E_F1_kWh"]),
            "total_pct": _percent(sums["total_kWh"], sums["E_F0_kWh"])}
