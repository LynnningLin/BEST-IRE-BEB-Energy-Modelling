import numpy as np
import pytest

from beb_soc_model import MotionParams, Segment, build_speed_profile
from signal_motion_cases import (
    ACCEL_MS2,
    CASES,
    DECEL_MS2,
    LENGTH_M,
    MOTION_PARAMS,
    SCHEDULED_TIME_S,
    SPEED_CAP_MS,
    VEHICLE_PARAMS,
    ValidationCase,
    braking_mechanical_kWh,
    build_validation_segment,
    idle_stop_positions_m,
    integrated_distance_m,
    run_case,
    signal_event,
    spatial_comparison_cases,
    transition_sweep,
)


def assert_physical_constraints(row):
    assert row["max_speed_ms"] <= SPEED_CAP_MS + 1e-9
    assert row["max_accel_ms2"] <= ACCEL_MS2 + 1e-9
    assert row["max_decel_ms2"] <= DECEL_MS2 + 1e-9


def test_v0_to_v5_controlled_cases_match_timing_physics_and_energy_rules():
    for case in CASES:
        row = run_case(case)
        diag = row["diag"]

        assert sum(diag["motion_sub_lengths_m"]) == pytest.approx(LENGTH_M)
        assert row["integrated_distance_m"] == pytest.approx(LENGTH_M, abs=1.5)
        assert_physical_constraints(row)
        assert row["wait_s"] == pytest.approx(diag["signal_wait_requested_s"])
        assert row["wait_s"] == pytest.approx(diag["signal_wait_applied_s"])
        assert diag["signal_wait_reduced_s"] == pytest.approx(0.0)
        assert row["regen_kWh"] <= row["braking_mechanical_kWh"] + 1e-9
        assert row["net_kWh"] == pytest.approx(
            row["traction_kWh"] + row["aux_hvac_kWh"] - row["regen_kWh"]
        )

        if case.stop_positions_m:
            assert row["signal_event_mode"] == "spatial"
            assert row["idle_stop_positions_m"] == pytest.approx(
                list(case.stop_positions_m), abs=1.5
            )
        else:
            assert row["stops"] == 0
            assert row["wait_s"] == pytest.approx(0.0)
            assert row["idle_stop_positions_m"] == []

        if row["schedule_feasible_by_budget"]:
            assert row["actual_move_s"] + row["wait_s"] == pytest.approx(
                SCHEDULED_TIME_S, abs=0.01
            )
            assert row["actual_segment_s"] == pytest.approx(SCHEDULED_TIME_S, abs=0.01)
            assert row["delay_s"] == pytest.approx(0.0, abs=0.01)
        else:
            assert row["actual_move_s"] == pytest.approx(row["min_move_s"], abs=0.01)
            assert row["actual_segment_s"] == pytest.approx(
                row["min_move_s"] + row["wait_s"], abs=0.01
            )
            assert row["delay_s"] == pytest.approx(
                row["actual_segment_s"] - SCHEDULED_TIME_S, abs=0.01
            )


def test_pass_spatial_events_do_not_split_or_wait():
    case = ValidationCase("P", "Pass events", (250.0, 750.0))
    seg = build_validation_segment(case)
    mp = MotionParams(
        accel_ms2=MOTION_PARAMS.accel_ms2,
        decel_ms2=MOTION_PARAMS.decel_ms2,
        dt_s=MOTION_PARAMS.dt_s,
        default_speed_cap_ms=MOTION_PARAMS.default_speed_cap_ms,
        max_speed_cap_ms=MOTION_PARAMS.max_speed_cap_ms,
        stop_prob=0.0,
        red_wait_s=15.0,
        signal_time_policy="preserve_schedule",
        max_signal_wait_share=1.0,
        use_hourly_signal_stop_probability=False,
        stop_prob_by_hour={},
    )

    profile, diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)

    assert diag["signal_event_mode"] == "spatial"
    assert diag["n_signal_stops"] == 0
    assert diag["n_motion_sublinks"] == 1
    assert diag["motion_sub_lengths_m"] == pytest.approx([LENGTH_M])
    assert diag["signal_wait_applied_s"] == pytest.approx(0.0)
    assert idle_stop_positions_m(profile) == []
    assert integrated_distance_m(profile) == pytest.approx(LENGTH_M, abs=1.5)


def test_closely_spaced_case_preserves_short_triangular_leg():
    row = run_case(next(case for case in CASES if case.case_id == "V3"))
    diag = row["diag"]

    assert diag["motion_sub_lengths_m"] == pytest.approx([450.0, 50.0, 500.0])
    assert diag["minimum_feasible_moving_time_s"] == pytest.approx(
        sum(
            [
                35.833333333333336,
                2.0 * np.sqrt(50.0 / ACCEL_MS2),
                38.333333333333336,
            ]
        )
    )
    assert_physical_constraints(row)


def test_transition_sweep_moves_from_schedule_recovery_to_delay_monotonically():
    sweep = transition_sweep(max_stops=6)

    assert list(sweep["n_signal_stops"]) == list(range(7))
    assert sweep["signal_wait_s"].is_monotonic_increasing
    assert sweep["required_avg_move_speed_ms"].dropna().is_monotonic_increasing
    assert sweep["maximum_speed_ms"].max() <= SPEED_CAP_MS + 1e-9
    feasible = sweep[sweep["schedule_delay_s"] <= 1e-9]
    assert feasible["maximum_speed_ms"].is_monotonic_increasing
    assert sweep.loc[sweep["n_signal_stops"] >= 2, "maximum_speed_ms"].max() == pytest.approx(
        SPEED_CAP_MS
    )
    assert sweep["schedule_delay_s"].is_monotonic_increasing
    assert list(sweep.loc[sweep["schedule_delay_s"] <= 1e-9, "n_signal_stops"]) == [
        0,
        1,
        2,
    ]
    assert list(sweep.loc[sweep["schedule_delay_s"] > 1e-9, "n_signal_stops"]) == [
        3,
        4,
        5,
        6,
    ]


def test_spatial_placement_changes_trajectory_and_energy_at_fixed_count_and_wait():
    rows = [run_case(case) for case in spatial_comparison_cases()]

    assert {row["stops"] for row in rows} == {2}
    assert {row["wait_s"] for row in rows} == {30.0}
    for row in rows:
        assert row["actual_segment_s"] == pytest.approx(SCHEDULED_TIME_S)

    max_speeds = {round(row["max_speed_ms"], 6) for row in rows}
    traction = {round(row["traction_kWh"], 8) for row in rows}
    regen = {round(row["regen_kWh"], 8) for row in rows}
    net = {round(row["net_kWh"], 8) for row in rows}

    assert len(max_speeds) > 1
    assert len(traction) > 1
    assert len(regen) > 1
    assert len(net) > 1


def test_aux_hvac_energy_tracks_total_elapsed_time_not_signal_count_alone():
    no_stop = run_case(next(case for case in CASES if case.case_id == "V0"))
    feasible_signal = run_case(next(case for case in CASES if case.case_id == "V2"))
    delayed_signal = run_case(next(case for case in CASES if case.case_id == "V5"))

    assert no_stop["actual_segment_s"] == pytest.approx(SCHEDULED_TIME_S)
    assert feasible_signal["actual_segment_s"] == pytest.approx(SCHEDULED_TIME_S)
    assert feasible_signal["wait_s"] > 0.0
    assert feasible_signal["aux_hvac_kWh"] == pytest.approx(no_stop["aux_hvac_kWh"])

    assert delayed_signal["actual_segment_s"] > SCHEDULED_TIME_S
    assert delayed_signal["aux_hvac_kWh"] > no_stop["aux_hvac_kWh"]


def test_spatial_and_fallback_paths_remain_distinguishable():
    spatial = build_validation_segment(ValidationCase("SP", "Spatial", (400.0, 600.0)))
    fallback = Segment(
        length_m=LENGTH_M,
        v_cruise_ms=12.0,
        run_time_s=SCHEDULED_TIME_S,
        n_signals=2,
        from_stop_departure_time="08:00:00",
    )

    spatial_profile, spatial_diag = build_speed_profile(
        spatial, motion_params=MOTION_PARAMS, return_diagnostics=True
    )
    fallback_profile, fallback_diag = build_speed_profile(
        fallback, motion_params=MOTION_PARAMS, return_diagnostics=True
    )

    assert spatial_diag["signal_event_mode"] == "spatial"
    assert spatial_diag["motion_sub_lengths_m"] == pytest.approx([400.0, 200.0, 400.0])
    assert idle_stop_positions_m(spatial_profile) == pytest.approx([400.0, 600.0], abs=1.5)

    assert fallback_diag["signal_event_mode"] == "legacy_count"
    assert fallback_diag["motion_sub_lengths_m"] == pytest.approx(
        [LENGTH_M / 3.0] * 3
    )
    assert idle_stop_positions_m(fallback_profile) == pytest.approx(
        [LENGTH_M / 3.0, 2.0 * LENGTH_M / 3.0], abs=1.5
    )


def test_event_level_randomness_boundaries_reproducibility_and_context():
    events = [signal_event(f"r{i}", i * 45.0 + 25.0) for i in range(20)]
    base = Segment(
        length_m=LENGTH_M,
        v_cruise_ms=12.0,
        run_time_s=180.0,
        n_signals=len(events),
        segment_geometry_id="synthetic|a|b",
        trip_id="trip_a",
        service_date="2025-07-09",
        segment_index=0,
        from_stop_departure_time="08:00:00",
        signal_events=events,
    )

    p0 = MotionParams(**{**MOTION_PARAMS.__dict__, "stop_prob": 0.0})
    p1 = MotionParams(**{**MOTION_PARAMS.__dict__, "stop_prob": 1.0})
    pa = MotionParams(**{**MOTION_PARAMS.__dict__, "stop_prob": 0.5, "signal_random_seed": 11})
    pb = MotionParams(**{**MOTION_PARAMS.__dict__, "stop_prob": 0.5, "signal_random_seed": 12})
    trip_b = Segment(**{**base.__dict__, "trip_id": "trip_b"})

    _, d0 = build_speed_profile(base, motion_params=p0, return_diagnostics=True)
    _, d1 = build_speed_profile(base, motion_params=p1, return_diagnostics=True)
    _, da = build_speed_profile(base, motion_params=pa, return_diagnostics=True)
    _, da_again = build_speed_profile(base, motion_params=pa, return_diagnostics=True)
    _, db = build_speed_profile(base, motion_params=pb, return_diagnostics=True)
    _, d_trip_b = build_speed_profile(trip_b, motion_params=pa, return_diagnostics=True)

    assert d0["n_signal_stops"] == 0
    assert d1["n_signal_stops"] == len(events)
    assert [x["stop_decision"] for x in da["signal_event_diagnostics"]] == [
        x["stop_decision"] for x in da_again["signal_event_diagnostics"]
    ]
    assert [x["random_value"] for x in da["signal_event_diagnostics"]] != [
        x["random_value"] for x in db["signal_event_diagnostics"]
    ]
    assert [x["random_value"] for x in da["signal_event_diagnostics"]] != [
        x["random_value"] for x in d_trip_b["signal_event_diagnostics"]
    ]


def test_regen_never_exceeds_available_braking_energy_for_controlled_cases():
    for case in CASES:
        seg = build_validation_segment(case)
        mp = MotionParams(**{**MOTION_PARAMS.__dict__, "red_wait_s": case.red_wait_s})
        profile, _diag = build_speed_profile(seg, motion_params=mp, return_diagnostics=True)
        energy = run_case(case)

        assert energy["regen_kWh"] <= braking_mechanical_kWh(
            profile, seg, VEHICLE_PARAMS
        ) + 1e-9
