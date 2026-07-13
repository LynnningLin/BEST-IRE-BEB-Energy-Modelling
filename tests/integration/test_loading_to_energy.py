from copy import deepcopy

from beb_soc_model import segment_energy_breakdown_kWh
from gtfs_to_segment import build_segments, load_shapes_for_trips, stop_times_for_trip
from passenger_loading import apply_passenger_loading


def _stable_inputs(segments):
    return [(s.length_m, s.run_time_s, s.dwell_s, s.grade, s.v_cruise_ms) for s in segments]


def test_passenger_loading_changes_mass_and_energy_monotonically(
    integration_gtfs_zip,
    integration_tables,
    demand_profile,
    integration_vehicle,
    integration_motion,
):
    _routes, trips, stops, _calendar, _calendar_dates = integration_tables
    rows = stop_times_for_trip(integration_gtfs_zip, "t1")
    shapes, shape_by_trip = load_shapes_for_trips(integration_gtfs_zip, trips, ["t1"])
    base_segments = build_segments(rows, stops, shape_points=shapes[shape_by_trip["t1"]])

    zero = deepcopy(base_segments)
    medium = deepcopy(base_segments)
    crush = deepcopy(base_segments)
    for seg in zero:
        seg.passengers = 0
    apply_passenger_loading(
        medium,
        [{"departure_time": "09:00:00", "arrival_time": "09:30:00"}],
        demand_profile,
        crush_capacity=80,
        shape="flat",
        verbose=False,
    )
    apply_passenger_loading(
        crush,
        [{"departure_time": "08:00:00", "arrival_time": "08:30:00"}],
        demand_profile,
        crush_capacity=80,
        shape="flat",
        verbose=False,
    )

    assert _stable_inputs(zero) == _stable_inputs(medium) == _stable_inputs(crush)
    assert [s.passengers for s in zero] == [0, 0]
    assert [s.passengers for s in medium] == [40, 40]
    assert [s.passengers for s in crush] == [80, 80]

    mass = lambda seg: integration_vehicle.curb_mass_kg + seg.passengers * integration_vehicle.passenger_mass_kg
    assert mass(zero[0]) < mass(medium[0]) < mass(crush[0])

    zero_e = sum(
        segment_energy_breakdown_kWh(s, integration_vehicle, motion_params=integration_motion).net_battery_energy_kWh
        for s in zero
    )
    medium_e = sum(
        segment_energy_breakdown_kWh(s, integration_vehicle, motion_params=integration_motion).net_battery_energy_kWh
        for s in medium
    )
    crush_e = sum(
        segment_energy_breakdown_kWh(s, integration_vehicle, motion_params=integration_motion).net_battery_energy_kWh
        for s in crush
    )

    assert zero_e <= medium_e <= crush_e
