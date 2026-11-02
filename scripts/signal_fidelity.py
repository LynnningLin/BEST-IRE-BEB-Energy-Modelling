"""Signal-information ablation, independent of the original F2 model.

Only exposure pairing uses F2 positions, to establish ordinal order. Placement
accepts count, length and independent identity/seed, never source coordinates.
"""

from bisect import bisect_right
from copy import copy
import math

from beb_soc_model import (
    _signal_events_for_segment, _signal_stop_prob_for_segment,
    _spatial_signal_decisions, _stable_uniform01,
)


def synthetic_locations(count, length_m, trip_id, service_date, location_seed):
    """Ordered uniform locations conditional on N; no physical exclusion zones."""
    if count < 0 or not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("Location generation requires nonnegative count and positive trip length.")
    return sorted(length_m * _stable_uniform01(
        "F1_frequency_locations_v1", location_seed, service_date, trip_id, ordinal
    ) for ordinal in range(count))


def paired_exposure(segments, mp):
    """Frozen F2 draws in spatial ordinal order, never rerandomised in F1."""
    inventory = []
    for i, seg in enumerate(segments):
        if seg.signal_events is None or seg.signal_source == "fallback":
            raise ValueError("F1 requires a complete F2 event inventory, including explicit empty lists.")
        events = _signal_events_for_segment(seg)
        if len(events) != int(seg.n_signals or 0):
            raise ValueError("F1 requires a complete F2 event inventory; count fallback is not F1.")
        probability, hour, source = _signal_stop_prob_for_segment(
            seg, mp.stop_prob, mp.hour_table_or_none())
        for decision in _spatial_signal_decisions(
                seg, events, probability, mp.behaviour_seed(), mp.red_wait_s):
            inventory.append({
                "paired_F2_event_id": decision["signal_event_id"],
                "paired_F2_segment": i, "paired_signal_hour": hour,
                "paired_probability_source": source,
                **{k: decision[k] for k in ("stop_probability", "random_value",
                                          "stop_decision", "red_wait_s_requested")},
            })
    return inventory


def prepare_signal_fidelity(segments, mp):
    """Copy ablated segments, preserving all physical and GTFS inputs.

    Half-open intervals assign boundaries downstream, except the final endpoint.
    Coincident synthetic events remain separate opportunities with separate waits.
    """
    segments = list(segments)
    if mp.signal_fidelity == "F2_spatial" or not segments:
        return segments
    result = [copy(seg) for seg in segments]
    for seg in result:
        seg.signal_events = []
        seg.signal_event_diagnostics = []
        seg.n_signals = 0
        seg.signal_source = mp.signal_fidelity
    if mp.signal_fidelity == "F0_none":
        return result
    if any(not math.isfinite(s.length_m) or s.length_m <= 0 for s in segments):
        raise ValueError("F1 requires positive finite segment lengths.")
    if mp.signal_fidelity in {"F1-U", "F1-J"}:
        from signal_placement import prepared_events
        events, diagnostics = prepared_events(segments, mp)
        for seg in result:
            seg._paired_signal_decisions = []
            seg._placement_diagnostics = diagnostics
        for event in events:
            seg = result[event["receiving_interval"]]
            seg.signal_events.append(event)
            seg._paired_signal_decisions.append(event)
            seg.n_signals += 1
        return result
    inventory = paired_exposure(segments, mp)
    boundaries = [0.0]
    for seg in segments:
        boundaries.append(boundaries[-1] + seg.length_m)
    locations = synthetic_locations(len(inventory), boundaries[-1], segments[0].trip_id,
                                    segments[0].service_date, mp.signal_location_seed)
    for seg in result:
        seg._paired_signal_decisions = []
    for ordinal, (position, exposure) in enumerate(zip(locations, inventory)):
        i = min(bisect_right(boundaries, position) - 1, len(result) - 1)
        seg = result[i]
        event = {
            **exposure, "trip_id": seg.trip_id,
            "signal_event_id": f"F1:{seg.trip_id}:{mp.signal_location_seed}:{ordinal}",
            "signal_location_seed": mp.signal_location_seed, "signal_ordinal": ordinal,
            "cumulative_distance_m": position, "assigned_segment": i,
            "distance_along_segment_m": position - boundaries[i],
            "segment_geometry_id": seg.segment_geometry_id,
            "signal_lat": None, "signal_lon": None, "source": "F1_frequency_stochastic",
        }
        seg.signal_events.append(event)
        seg._paired_signal_decisions.append(event)
        seg.n_signals += 1
    return result
