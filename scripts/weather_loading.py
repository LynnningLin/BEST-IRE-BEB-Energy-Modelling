"""
weather_loading.py
================================================================================
Weather -> HVAC auxiliary load for the BEB longitudinal-dynamics energy model,
plus a loader for the project's hourly weather CSV (temperature + humidity).

WHAT THIS MODULE DOES
---------------------
1. Reads the weather CSV (columns: time, year, month, day, temp, rhum, solar) into a
   timestamp-indexed WeatherSeries, converting humidity from % to a 0..1 fraction.
2. Resolves the conditions any trip experienced (service date + GTFS clock time).
3. Turns those conditions into a per-segment auxiliary power:

       aux(segment) = base_aux_kW + hvac_kW(weather, occupancy)

   where hvac_kW uses BOTH air temperature and relative humidity (humidity acts
   through the latent / dehumidification term on the cooling branch).

A CUSTOMISABLE "PLUG" (see ClimateControlPlug below) decides WHEN heating or
cooling switches on, e.g. heat below 10 C, cool above 20 C, off in between.

Applies as a post-pass over the Segment list, after apply_passenger_loading():
    build_segments() -> add_grades_from_dem() -> apply_passenger_loading()
                     -> apply_weather_loading()        <-- this module, LAST
Requires Segment to carry an `aux_power_kW` field (two-line patch to
beb_soc_model.py: add `aux_power_kW: float = None` to the Segment dataclass and
read it in segment_energy_kWh).

NOTE: this file now contains the CSV loading that previously lived in
weather_series.py, so weather_series.py is redundant and can be deleted.
================================================================================
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from best_ire_beb.config import get_path, get_section

WEATHER_CSV_PATH = get_path("weather_csv")
_WEATHER_DEFAULTS = get_section("weather")
_CLIMATE_DEFAULTS = _WEATHER_DEFAULTS["climate_control"]
_HVAC_DEFAULTS = _WEATHER_DEFAULTS["hvac"]
DEFAULT_HEATING_MONTHS = (11, 12, 1, 2, 3)


# -----------------------------------------------------------------------------
# GTFS time helpers (kept local so the module stands alone)
# -----------------------------------------------------------------------------
def _time_to_seconds(t: str) -> int:
    """GTFS time -> seconds since service-day midnight (may exceed 86400)."""
    h, m, s = (int(x) for x in str(t).strip().split(":"))
    return h * 3600 + m * 60 + s


def trip_reference_hour(trip_rows, mode="midpoint") -> int:
    t0 = _time_to_seconds(trip_rows[0]["departure_time"])
    t1 = _time_to_seconds(trip_rows[-1]["arrival_time"])
    sec = t0 if mode == "start" else (t0 + t1) // 2
    return int((sec // 3600) % 24)


# -----------------------------------------------------------------------------
# Weather inputs  (DATA-DRIVEN -- comes from the CSV)
# -----------------------------------------------------------------------------
@dataclass
class WeatherConditions:
    """Ambient conditions for one trip (or one clock hour)."""
    air_temp_c: float                  # dry-bulb air temperature (deg C)   <- CSV 'temp'
    relative_humidity: float = 0.80    # 0..1 fraction                      <- CSV 'rhum'/100
    solar_W_m2: float = 0.0            # global horizontal irradiance (W/m^2)  <- CSV 'solar'
    is_raining: bool = False           # not in this CSV; hook for future Crr coupling
    observed_at: Optional[datetime] = None


# #############################################################################
# ##  THE CLIMATE-CONTROL PLUG                                               ##
# ##  Decides WHEN heating / cooling engage. Customise these two numbers.    ##
# ##  Example from the brief: heat_below_c=10, cool_above_c=20               ##
# ##    ambient < heat_below_c, Nov-Mar -> HEATING on                        ##
# ##    ambient > cool_above_c  -> COOLING on                                ##
# ##    in between              -> OFF (comfort dead-band, fans only)        ##
# ##  Each threshold also serves as the cabin target it conditions toward,   ##
# ##  so the HVAC load falls smoothly to zero at the threshold (no jump).    ##
# #############################################################################
@dataclass
class ClimateControlPlug:
    heat_below_c: float = float(_CLIMATE_DEFAULTS["heat_below_c"])
    cool_above_c: float = float(_CLIMATE_DEFAULTS["cool_above_c"])
    heating_months: tuple = DEFAULT_HEATING_MONTHS

    @classmethod
    def from_config(cls, config_path=None):
        cfg = get_section("weather", config_path).get("climate_control", {})
        return cls(
            heat_below_c=float(cfg.get("heat_below_c", cls.heat_below_c)),
            cool_above_c=float(cfg.get("cool_above_c", cls.cool_above_c)),
            heating_months=_normalise_months(
                cfg.get("heating_months", DEFAULT_HEATING_MONTHS)
            ),
        )

    def heating_allowed(self, when=None) -> bool:
        month = _month_from_when(when)
        return month is None or month in self.heating_months


def _normalise_months(value) -> tuple:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(int(month) for month in value)


def _month_from_when(when) -> Optional[int]:
    if when is None:
        return None
    return int(when.month)


def decide_mode(temp_c: float, plug: ClimateControlPlug, when=None) -> str:
    """Return 'heat', 'cool', or 'off' for an ambient temperature. THE PLUG."""
    if temp_c < plug.heat_below_c and plug.heating_allowed(when):
        return "heat"
    if temp_c > plug.cool_above_c:
        return "cool"
    return "off"


# -----------------------------------------------------------------------------
# HVAC assumptions  (ASSUMPTION -- calibrate / cite; placeholders for a 12 m BEB)
# -----------------------------------------------------------------------------
@dataclass
class HVACParams:
    plug: ClimateControlPlug = field(default_factory=ClimateControlPlug)

    # non-thermal hotel load (lights, doors, fans)
    base_aux_kW: float = float(_HVAC_DEFAULTS["base_aux_kW"])

    # cabin steady-state heat balance
    # lumped UA: shell + glazing + ventilation + doors
    cabin_loss_W_per_K: float = float(_HVAC_DEFAULTS["cabin_loss_W_per_K"])
    # effective glazing*SHGC (m^2); 0 W gain w/o solar data
    solar_aperture_m2: float = float(_HVAC_DEFAULTS["solar_aperture_m2"])
    pax_sensible_W: float = float(_HVAC_DEFAULTS["pax_sensible_W"])
    use_passenger_gain: bool = bool(_HVAC_DEFAULTS["use_passenger_gain"])

    # thermal -> electrical (COP)
    heater_type: str = str(_HVAC_DEFAULTS["heater_type"])
    heat_cop_at_0c: float = float(_HVAC_DEFAULTS["heat_cop_at_0c"])
    heat_cop_slope: float = float(_HVAC_DEFAULTS["heat_cop_slope"])
    heat_cop_min: float = float(_HVAC_DEFAULTS["heat_cop_min"])
    heat_cop_max: float = float(_HVAC_DEFAULTS["heat_cop_max"])
    cool_cop: float = float(_HVAC_DEFAULTS["cool_cop"])
    # HUMIDITY enters here: extra cooling fraction for dehumidification at RH=1.0,
    # scaled linearly by relative humidity. This is the latent-load term.
    latent_load_at_full_rh: float = float(_HVAC_DEFAULTS["latent_load_at_full_rh"])

    hvac_max_kW: float = float(_HVAC_DEFAULTS["hvac_max_kW"])

    @classmethod
    def from_config(cls, config_path=None, plug=None):
        cfg = get_section("weather", config_path).get("hvac", {})
        plug = plug or ClimateControlPlug.from_config(config_path)
        fields = {
            "base_aux_kW",
            "cabin_loss_W_per_K",
            "solar_aperture_m2",
            "pax_sensible_W",
            "use_passenger_gain",
            "heater_type",
            "heat_cop_at_0c",
            "heat_cop_slope",
            "heat_cop_min",
            "heat_cop_max",
            "cool_cop",
            "latent_load_at_full_rh",
            "hvac_max_kW",
        }
        return cls(plug=plug, **{k: v for k, v in cfg.items() if k in fields})


def _heating_cop(t_amb_c: float, hp: HVACParams) -> float:
    if hp.heater_type == "resistive":
        return 1.0
    cop = hp.heat_cop_at_0c + hp.heat_cop_slope * t_amb_c
    return float(min(max(cop, hp.heat_cop_min), hp.heat_cop_max))


# -----------------------------------------------------------------------------
# HVAC model 1: physical cabin heat balance (DEFAULT) -- uses temp AND humidity
# -----------------------------------------------------------------------------
def hvac_kW_thermal(weather: WeatherConditions, passengers: int, hp: HVACParams,
                    when=None) -> float:
    mode = decide_mode(weather.air_temp_c, hp.plug, when or weather.observed_at)
    if mode == "off":
        return 0.0

    t = weather.air_temp_c
    solar_W = hp.solar_aperture_m2 * max(weather.solar_W_m2, 0.0)
    pax_W = (hp.pax_sensible_W * max(passengers, 0)) if hp.use_passenger_gain else 0.0

    if mode == "heat":
        dT = hp.plug.heat_below_c - t
        q_loss = hp.cabin_loss_W_per_K * dT
        q_thermal = max(q_loss - solar_W - pax_W, 0.0)     # gains reduce heating
        p_elec_W = q_thermal / _heating_cop(t, hp)
    else:  # cool  -- humidity matters here
        dT = t - hp.plug.cool_above_c
        q_loss = hp.cabin_loss_W_per_K * dT
        q_sensible = q_loss + solar_W + pax_W              # gains add to cooling
        rh = max(min(weather.relative_humidity, 1.0), 0.0)
        latent = 1.0 + hp.latent_load_at_full_rh * rh      # dehumidification load
        q_thermal = q_sensible * latent
        p_elec_W = q_thermal / max(hp.cool_cop, 1e-3)

    return min(p_elec_W / 1000.0, hp.hvac_max_kW)


# -----------------------------------------------------------------------------
# HVAC model 2: empirical V-shape (literature workhorse; quick to calibrate)
# -----------------------------------------------------------------------------
def hvac_kW_piecewise(weather: WeatherConditions, hp: HVACParams,
                      k_heat_kW_per_K: float = 0.40,
                      k_cool_kW_per_K: float = 0.35,
                      when=None) -> float:
    mode = decide_mode(weather.air_temp_c, hp.plug, when or weather.observed_at)
    if mode == "heat":
        p = k_heat_kW_per_K * (hp.plug.heat_below_c - weather.air_temp_c)
    elif mode == "cool":
        p = k_cool_kW_per_K * (weather.air_temp_c - hp.plug.cool_above_c)
    else:
        p = 0.0
    return min(p, hp.hvac_max_kW)


# -----------------------------------------------------------------------------
# Weather CSV  ->  WeatherSeries   (handles this project's file shape)
# -----------------------------------------------------------------------------
# Column-name candidates, lower-cased, so light header changes still load.
_TIME_COLS  = ["time", "datetime", "timestamp", "date_time"]
_TEMP_COLS  = ["temp", "temperature", "air_temp", "air temperature", "t2m", "temp_c"]
_RH_COLS    = ["rhum", "relative_humidity", "humidity", "rh", "relative humidity"]
_SOLAR_COLS = ["solar", "shortwave_radiation", "ghi", "glorad", "radiation", "solar_w_m2"]


def _pick(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


class WeatherSeries:
    """Timestamp-indexed weather (temp + humidity + solar) with nearest-row lookup."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_index()          # DatetimeIndex; cols: temp, rh, [solar]

    def at(self, when: datetime) -> WeatherConditions:
        ts = pd.Timestamp(when)
        row = self.df.iloc[self.df.index.get_indexer([ts], method="nearest")[0]]
        rh = float(row["rh"]) if "rh" in self.df.columns and pd.notna(row.get("rh")) else 0.80
        solar = float(row["solar"]) if "solar" in self.df.columns and pd.notna(row.get("solar")) else 0.0
        return WeatherConditions(air_temp_c=float(row["temp"]), relative_humidity=rh,
                                 solar_W_m2=max(solar, 0.0),
                                 observed_at=ts.to_pydatetime())

    def trip_datetime(self, service_date: date, trip_rows, mode="midpoint") -> datetime:
        t0 = _time_to_seconds(trip_rows[0]["departure_time"])
        t1 = _time_to_seconds(trip_rows[-1]["arrival_time"])
        sec = t0 if mode == "start" else (t0 + t1) // 2
        base = datetime.combine(service_date, time(0, 0))
        return base + timedelta(seconds=sec)

    def weather_for_trip(self, service_date: date, trip_rows, mode="midpoint") -> WeatherConditions:
        return self.at(self.trip_datetime(service_date, trip_rows, mode))

    # -- design days / diagnostics ------------------------------------------
    def daily_mean_temp(self) -> pd.Series:
        return self.df["temp"].resample("D").mean()

    def design_days(self):
        """(coldest, median, hottest) dates by daily-mean temperature."""
        dmt = self.daily_mean_temp().dropna()
        return dmt.idxmin().date(), dmt.sub(dmt.median()).abs().idxmin().date(), dmt.idxmax().date()

    def mode_breakdown(self, plug: ClimateControlPlug) -> dict:
        """Fraction of hours the plug puts in heat / cool / off (sanity check)."""
        m = pd.Series(
            [decide_mode(row["temp"], plug, idx) for idx, row in self.df.iterrows()]
        )
        frac = m.value_counts(normalize=True)
        return {k: round(100 * float(frac.get(k, 0.0)), 1) for k in ("heat", "cool", "off")}


def load_weather_csv(path=None, time_col=None, temp_col=None,
                     rh_col=None, solar_col=None, rh_is_percent=True,
                     verbose=True, config_path=None) -> WeatherSeries:
    """
    Load the weather CSV from `path` or paths.weather_csv in the project config.

    Expects this project's shape -- time, year, month, day, temp, rhum, solar --
    but autodetects column names and falls back to building the timestamp from
    year/month/day(/hour) if a single 'time' column is absent. Humidity given as
    a percentage (rhum 0..100) is converted to a 0..1 fraction; solar is read in
    W/m^2 and clipped at 0. A solar column is optional -- if absent, solar gain
    defaults to 0 and the model runs exactly as before.
    """
    path = Path(path) if path else get_path("weather_csv", config_path)
    raw = pd.read_csv(path)
    cols = list(raw.columns)

    xcol = temp_col or _pick(cols, _TEMP_COLS)
    if xcol is None:
        raise ValueError(f"no temperature column found in {cols}")
    rcol = rh_col or _pick(cols, _RH_COLS)
    scol = solar_col or _pick(cols, _SOLAR_COLS)

    tcol = time_col or _pick(cols, _TIME_COLS)
    if tcol is not None:
        ts = pd.to_datetime(raw[tcol], errors="coerce")
    else:  # assemble from parts (year/month/day[/hour])
        lc = {c.lower(): c for c in cols}
        parts = {"year": lc.get("year"), "month": lc.get("month"), "day": lc.get("day")}
        if not all(parts.values()):
            raise ValueError(f"no 'time' column and cannot assemble one from {cols}")
        frame = {k: raw[v] for k, v in parts.items()}
        frame["hour"] = raw[lc["hour"]] if "hour" in lc else 0
        ts = pd.to_datetime(pd.DataFrame(frame), errors="coerce")

    out = pd.DataFrame({"temp": pd.to_numeric(raw[xcol], errors="coerce")})
    if rcol is not None:
        rh = pd.to_numeric(raw[rcol], errors="coerce")
        out["rh"] = (rh / 100.0) if rh_is_percent else rh
    if scol is not None:
        out["solar"] = pd.to_numeric(raw[scol], errors="coerce").clip(lower=0.0)
    out.index = ts
    out = out[out.index.notna() & out["temp"].notna()]
    if out.empty:
        raise ValueError(f"{path} contains no valid timestamp/temperature rows")
    if "rh" in out.columns:
        bad_rh = out["rh"].dropna()
        if ((bad_rh < 0.0) | (bad_rh > 1.0)).any():
            raise ValueError("relative humidity must be within [0, 1] after conversion")
    if out.index.has_duplicates:
        raise ValueError("weather timestamps must be unique")

    if verbose:
        span = f"{out.index.min()} -> {out.index.max()}"
        hum = f"humidity '{rcol}' (%->frac)" if rcol else "NO humidity"
        sol = f"solar '{scol}' (W/m^2)" if scol else "NO solar (defaults 0)"
        print(f"  weather CSV: {len(out)} rows, {span}; temp '{xcol}', {hum}, {sol}")
    return WeatherSeries(out)


# -----------------------------------------------------------------------------
# Apply: set Segment.aux_power_kW from weather
# -----------------------------------------------------------------------------
def apply_weather_loading(segments, trip_rows, weather, hp: HVACParams = None,
                          service_date: date = None, model=None,
                          hour_mode=None, k_heat_kW_per_K=None,
                          k_cool_kW_per_K=None, verbose=True,
                          config_path=None):
    """
    Write per-segment aux power onto each Segment, reflecting weather.

        seg.aux_power_kW = base_aux_kW + hvac_kW(weather, seg.passengers)

    weather : a WeatherConditions (one condition for the whole trip), OR a
              WeatherSeries (then `service_date` is required -- the trip is
              resolved to its real timestamp and looked up in the CSV).
    model   : 'thermal' (physical, uses temp+humidity) or 'piecewise'.
    """
    cfg = get_section("weather", config_path).get("hvac", {})
    if hp is None:
        hp = HVACParams.from_config(config_path)
    model = model or cfg.get("model", "thermal")
    hour_mode = hour_mode or cfg.get("hour_mode", "midpoint")
    if k_heat_kW_per_K is None:
        k_heat_kW_per_K = float(cfg.get("k_heat_kW_per_K", 0.40))
    if k_cool_kW_per_K is None:
        k_cool_kW_per_K = float(cfg.get("k_cool_kW_per_K", 0.35))
    if not segments:
        return segments

    if isinstance(weather, WeatherSeries):
        if service_date is None:
            raise ValueError("service_date is required when weather is a WeatherSeries")
        weather_time = weather.trip_datetime(service_date, trip_rows, hour_mode)
        wx = weather.at(weather_time)
    else:
        wx = weather
        weather_time = getattr(wx, "observed_at", None)
        if weather_time is None and service_date is not None:
            weather_time = datetime.combine(service_date, time(0, 0))

    aux_list = []
    for seg in segments:
        if model == "thermal":
            hvac = hvac_kW_thermal(wx, getattr(seg, "passengers", 0), hp,
                                   when=weather_time)
        elif model == "piecewise":
            hvac = hvac_kW_piecewise(wx, hp, k_heat_kW_per_K, k_cool_kW_per_K,
                                     when=weather_time)
        else:
            raise ValueError(f"unknown model {model!r}")
        seg.aux_power_kW = hp.base_aux_kW + hvac
        aux_list.append(seg.aux_power_kW)

    if verbose:
        mode = decide_mode(wx.air_temp_c, hp.plug, weather_time)
        mean_aux = sum(aux_list) / len(aux_list)
        when = f" on {service_date}" if service_date else ""
        print(f"  weather{when}: {wx.air_temp_c:+.1f} C, RH {wx.relative_humidity*100:.0f}% "
              f"-> {mode}; aux {mean_aux:.1f} kW (base {hp.base_aux_kW:.1f} + "
              f"HVAC {mean_aux - hp.base_aux_kW:.1f}); model={model}")
    return segments


def parse_args():
    p = argparse.ArgumentParser(description="Exercise the weather/HVAC loader.")
    p.add_argument("--config", help="Path to a model YAML config file.")
    p.add_argument("--weather-csv", help="Override paths.weather_csv.")
    p.add_argument("--date", help="Override simulation.date (YYYY-MM-DD).")
    return p.parse_args()


if __name__ == "__main__":
    # --- exercise against the real CSV --------------------------------------
    args = parse_args()
    plug = ClimateControlPlug.from_config(args.config)
    hp = HVACParams.from_config(args.config, plug=plug)
    simulation_cfg = get_section("simulation", args.config)
    scenario_date = date.fromisoformat(args.date or str(simulation_cfg["date"]))

    series = load_weather_csv(args.weather_csv, config_path=args.config)
    print(
        f"  plug (heat<{plug.heat_below_c:g} in months {plug.heating_months} / "
        f"cool>{plug.cool_above_c:g}) "
        f"hour split: {series.mode_breakdown(plug)}"
    )
    cold, med, hot = series.design_days()
    print(f"  design days -> cold {cold}, median {med}, hot {hot}")

    trip_rows = [{"departure_time": "08:00:00", "arrival_time": "08:45:00"}]
    wx = series.weather_for_trip(scenario_date, trip_rows)
    kw = hvac_kW_thermal(wx, passengers=20, hp=hp)
    print(f"  scenario {scenario_date} 08:00 -> {wx.air_temp_c:+.1f} C, "
          f"RH {wx.relative_humidity*100:.0f}%  =>  HVAC {kw:.1f} kW "
          f"({decide_mode(wx.air_temp_c, plug, wx.observed_at)})")
    for label, d in [("cold", cold), ("median", med), ("hot", hot)]:
        wx = series.weather_for_trip(d, trip_rows)
        kw = hvac_kW_thermal(wx, passengers=20, hp=hp)
        print(f"  {label:6s} {d} 08:00 -> {wx.air_temp_c:+.1f} C, "
              f"RH {wx.relative_humidity*100:.0f}%  =>  HVAC {kw:.1f} kW "
              f"({decide_mode(wx.air_temp_c, plug, wx.observed_at)})")

    # humidity effect, holding temperature fixed (only bites when cooling is ON)
    print("\n  humidity effect at fixed temperature (thermal model, 20 pax):")
    for t in (6.0, 25.0):
        dry = hvac_kW_thermal(WeatherConditions(t, relative_humidity=0.40), 20, hp)
        wet = hvac_kW_thermal(WeatherConditions(t, relative_humidity=0.95), 20, hp)
        print(f"    {t:+.0f} C: RH 40% -> {dry:.2f} kW | RH 95% -> {wet:.2f} kW "
              f"(delta {wet - dry:+.2f})")

