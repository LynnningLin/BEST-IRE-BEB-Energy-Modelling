import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from weather_loading import (  # noqa: E402
    ClimateControlPlug,
    HVACParams,
    WeatherConditions,
    apply_weather_loading,
    decide_mode,
    hvac_kW_piecewise,
)


@dataclass
class DummySegment:
    passengers: int = 0
    aux_power_kW: float = 0.0


def test_heating_only_engages_in_configured_heating_months() -> None:
    plug = ClimateControlPlug(heat_below_c=10.0, cool_above_c=20.0)

    assert decide_mode(5.0, plug, datetime(2025, 1, 6)) == "heat"
    assert decide_mode(5.0, plug, datetime(2025, 4, 14)) == "off"
    assert decide_mode(21.0, plug, datetime(2025, 1, 6)) == "cool"


def test_piecewise_heating_is_zero_on_cold_april_day() -> None:
    plug = ClimateControlPlug(heat_below_c=10.0, cool_above_c=20.0)
    hp = HVACParams(plug=plug, hvac_max_kW=30.0)
    weather = WeatherConditions(air_temp_c=5.0)

    jan_kw = hvac_kW_piecewise(weather, hp, when=datetime(2025, 1, 6))
    apr_kw = hvac_kW_piecewise(weather, hp, when=datetime(2025, 4, 14))

    assert jan_kw > 0.0
    assert apr_kw == 0.0


def test_apply_weather_loading_uses_service_date_for_direct_conditions() -> None:
    plug = ClimateControlPlug(heat_below_c=10.0, cool_above_c=20.0)
    hp = HVACParams(plug=plug, base_aux_kW=3.0, hvac_max_kW=30.0)
    trip_rows = [{"departure_time": "08:00:00", "arrival_time": "08:30:00"}]
    weather = WeatherConditions(air_temp_c=5.0)

    apr_segments = apply_weather_loading(
        [DummySegment()],
        trip_rows,
        weather,
        hp=hp,
        service_date=date(2025, 4, 14),
        model="piecewise",
        verbose=False,
    )
    jan_segments = apply_weather_loading(
        [DummySegment()],
        trip_rows,
        weather,
        hp=hp,
        service_date=date(2025, 1, 6),
        model="piecewise",
        verbose=False,
    )

    assert apr_segments[0].aux_power_kW == hp.base_aux_kW
    assert jan_segments[0].aux_power_kW > hp.base_aux_kW
