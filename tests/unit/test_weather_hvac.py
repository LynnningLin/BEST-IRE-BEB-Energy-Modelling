from datetime import date, datetime

import pytest

from weather_loading import (
    ClimateControlPlug,
    HVACParams,
    WeatherConditions,
    apply_weather_loading,
    decide_mode,
    hvac_kW_piecewise,
    hvac_kW_thermal,
)


class AuxSegment:
    def __init__(self, passengers=0):
        self.passengers = passengers
        self.aux_power_kW = None


def _hp(**overrides):
    plug = overrides.pop(
        "plug",
        ClimateControlPlug(
            heat_below_c=10.0, cool_above_c=26.0, heating_months=(1, 2, 3)
        ),
    )
    defaults = {
        "base_aux_kW": 3.0,
        "cabin_loss_W_per_K": 500.0,
        "solar_aperture_m2": 0.0,
        "pax_sensible_W": 90.0,
        "use_passenger_gain": True,
        "heater_type": "resistive",
        "cool_cop": 2.5,
        "latent_load_at_full_rh": 0.30,
        "hvac_max_kW": 30.0,
    }
    defaults.update(overrides)
    return HVACParams(plug=plug, **defaults)


@pytest.mark.parametrize(
    ("temp", "rh", "when", "mode"),
    [
        (0.0, 0.80, datetime(2025, 1, 1), "heat"),
        (15.0, 0.80, datetime(2025, 1, 1), "off"),
        (27.0, 0.40, datetime(2025, 7, 1), "cool"),
        (27.0, 0.95, datetime(2025, 7, 1), "cool"),
    ],
)
def test_climate_mode_table(temp, rh, when, mode):
    hp = _hp()

    assert decide_mode(temp, hp.plug, when) == mode


def test_heating_disabled_outside_heating_months_and_deadband_zero():
    hp = _hp()

    assert decide_mode(0.0, hp.plug, datetime(2025, 7, 1)) == "off"
    assert hvac_kW_thermal(WeatherConditions(15.0, 0.8), 20, hp) == 0.0


def test_humidity_and_passengers_change_thermal_loads_predictably():
    hp = _hp()
    dry = hvac_kW_thermal(
        WeatherConditions(27.0, relative_humidity=0.40), passengers=10, hp=hp
    )
    humid = hvac_kW_thermal(
        WeatherConditions(27.0, relative_humidity=0.95), passengers=10, hp=hp
    )
    cold_empty = hvac_kW_thermal(
        WeatherConditions(0.0, relative_humidity=0.80), passengers=0, hp=hp,
        when=datetime(2025, 1, 1)
    )
    cold_full = hvac_kW_thermal(
        WeatherConditions(0.0, relative_humidity=0.80), passengers=40, hp=hp,
        when=datetime(2025, 1, 1)
    )
    hot_empty = hvac_kW_thermal(WeatherConditions(30.0, 0.80), passengers=0, hp=hp)
    hot_full = hvac_kW_thermal(WeatherConditions(30.0, 0.80), passengers=40, hp=hp)

    assert humid > dry
    assert cold_full < cold_empty
    assert hot_full > hot_empty
    assert min(dry, humid, cold_full, hot_full) >= 0.0


def test_hvac_power_is_capped_and_aux_equals_base_plus_hvac():
    hp = _hp(hvac_max_kW=1.25, cabin_loss_W_per_K=10_000.0)
    wx = WeatherConditions(35.0, relative_humidity=1.0)

    hvac = hvac_kW_thermal(wx, passengers=100, hp=hp)
    segments = apply_weather_loading(
        [AuxSegment(passengers=100)],
        [{"departure_time": "08:00:00", "arrival_time": "08:30:00"}],
        wx,
        hp=hp,
        service_date=date(2025, 7, 1),
        model="thermal",
        verbose=False,
    )

    assert hvac == pytest.approx(1.25)
    assert segments[0].aux_power_kW == pytest.approx(hp.base_aux_kW + hvac)


def test_piecewise_mode_has_expected_linear_heating_and_cooling():
    hp = _hp()

    heat = hvac_kW_piecewise(
        WeatherConditions(0.0, 0.8), hp, k_heat_kW_per_K=0.4,
        when=datetime(2025, 1, 1)
    )
    cool = hvac_kW_piecewise(
        WeatherConditions(30.0, 0.8), hp, k_cool_kW_per_K=0.5,
        when=datetime(2025, 7, 1)
    )

    assert heat == pytest.approx(4.0)
    assert cool == pytest.approx(2.0)
