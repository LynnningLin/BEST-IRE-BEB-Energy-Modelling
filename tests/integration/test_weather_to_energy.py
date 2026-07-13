from copy import deepcopy
from datetime import date

import pytest

from beb_soc_model import segment_energy_breakdown_kWh
from gtfs_to_segment import build_segments, load_shapes_for_trips, stop_times_for_trip
from weather_loading import ClimateControlPlug, HVACParams, WeatherConditions, apply_weather_loading


def _weather_hp():
    return HVACParams(
        plug=ClimateControlPlug(heat_below_c=10.0, cool_above_c=26.0, heating_months=(1, 2, 3)),
        base_aux_kW=3.0,
        cabin_loss_W_per_K=700.0,
        solar_aperture_m2=0.0,
        pax_sensible_W=90.0,
        use_passenger_gain=True,
        heater_type="resistive",
        cool_cop=2.5,
        latent_load_at_full_rh=0.30,
        hvac_max_kW=30.0,
    )


def _case_energy(segments, wx, when, hp, vehicle, motion):
    segs = deepcopy(segments)
    apply_weather_loading(
        segs,
        [{"departure_time": "08:00:00", "arrival_time": "08:30:00"}],
        wx,
        hp=hp,
        service_date=when,
        model="thermal",
        verbose=False,
    )
    energies = [segment_energy_breakdown_kWh(s, vehicle, motion_params=motion) for s in segs]
    return segs, energies


def test_weather_auxiliary_load_overrides_vehicle_default_and_changes_only_aux_energy(
    integration_gtfs_zip,
    integration_tables,
    integration_vehicle,
    integration_motion,
):
    _routes, trips, stops, _calendar, _calendar_dates = integration_tables
    rows = stop_times_for_trip(integration_gtfs_zip, "t1")
    shapes, shape_by_trip = load_shapes_for_trips(integration_gtfs_zip, trips, ["t1"])
    segments = build_segments(rows, stops, passengers=20, shape_points=shapes[shape_by_trip["t1"]])
    hp = _weather_hp()

    cold, cold_e = _case_energy(
        segments, WeatherConditions(0.0, 0.80), date(2025, 1, 9), hp,
        integration_vehicle, integration_motion
    )
    dead, dead_e = _case_energy(
        segments, WeatherConditions(20.0, 0.50), date(2025, 7, 9), hp,
        integration_vehicle, integration_motion
    )
    hot_dry, dry_e = _case_energy(
        segments, WeatherConditions(30.0, 0.40), date(2025, 7, 9), hp,
        integration_vehicle, integration_motion
    )
    hot_humid, humid_e = _case_energy(
        segments, WeatherConditions(30.0, 0.95), date(2025, 7, 9), hp,
        integration_vehicle, integration_motion
    )

    assert all(seg.aux_power_kW is not None for seg in cold + dead + hot_dry + hot_humid)
    assert all(seg.aux_power_kW != integration_vehicle.aux_power_kW for seg in dead)

    sum_aux = lambda energies: sum(e.aux_energy_kWh for e in energies)
    sum_net = lambda energies: sum(e.net_battery_energy_kWh for e in energies)
    sum_gross = lambda energies: sum(e.gross_consumed_kWh for e in energies)

    assert sum_aux(cold_e) > sum_aux(dead_e)
    assert sum_aux(dry_e) > sum_aux(dead_e)
    assert sum_aux(humid_e) >= sum_aux(dry_e)
    assert sum_gross(cold_e) == pytest.approx(sum_gross(dead_e))
    assert sum_gross(dry_e) == pytest.approx(sum_gross(dead_e))
    assert sum_net(cold_e) > sum_net(dead_e)
    assert sum_net(humid_e) >= sum_net(dry_e)
