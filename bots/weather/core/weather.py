from __future__ import annotations

import httpx
import structlog
from cachetools import TTLCache

from pm_bot.models.config import CACHE_TTL, CITY_COORDS
from pm_bot.models.market import ForecastResult

log = structlog.get_logger()

OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
ENSEMBLE_BASE = "https://ensemble-api.open-meteo.com/v1/ensemble"

_forecast_cache: TTLCache[str, ForecastResult] = TTLCache(maxsize=128, ttl=CACHE_TTL["forecast"])

# Open-Meteo GFS ensemble member key pattern
_MEMBER_KEYS = [f"temperature_2m_max_member{i:02d}" for i in range(1, 31)]


async def fetch_forecast(
    client: httpx.AsyncClient,
    city: str,
    date: str = "",
    model: str = "gfs_seamless",
) -> ForecastResult | None:
    coords = CITY_COORDS.get(city)
    if not coords:
        log.warning("unknown_city", city=city)
        return None

    lat, lon = coords
    key = f"{city}:{model}:high"
    if key in _forecast_cache:
        return _forecast_cache[key]

    params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "forecast_days": 3,
        "timezone": "auto",
    }

    try:
        params_model = {**params, "models": model}
        resp = await client.get(f"{OPEN_METEO_BASE}/forecast", params=params_model)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        log.error("weather_api_error", city=city, error=str(e))
        return None

    daily = data.get("daily", {})
    temps = daily.get("temperature_2m_max", [])
    main_temp = float(temps[0]) if temps and isinstance(temps[0], (int, float)) else 0.0

    members: list[float] = []
    try:
        params_ens = {**params, "models": model}
        resp = await client.get(ENSEMBLE_BASE, params=params_ens)
        resp.raise_for_status()
        ens_data = resp.json()
        ens_daily = ens_data.get("daily", {})
        for mk in _MEMBER_KEYS:
            member_data = ens_daily.get(mk, [])
            if member_data:
                v = member_data[0]
                if isinstance(v, (int, float)):
                    members.append(float(v))
    except httpx.HTTPError as e:
        log.warning("ensemble_fetch_failed", city=city, error=str(e))

    result = ForecastResult(
        city=city,
        date=date,
        model=model,
        temp_high_c=main_temp,
        measure_type="high",
        members=members,
    )

    _forecast_cache[key] = result
    return result


_TAIL_BOUND = 999.0


def bucket_probability_numpy(
    forecast: ForecastResult,
    temp_low_c: float,
    temp_high_c: float,
    temp_unit: str = "C",
) -> float:
    import numpy as np

    if forecast.members:
        arr = np.array(forecast.members)
        if temp_unit == "F":
            arr_f = arr * 1.8 + 32.0
            truncated_f = np.floor(arr_f)
            bucket_low_f = temp_low_c * 1.8 + 32.0
            bucket_high_f = temp_high_c * 1.8 + 32.0
            if temp_high_c >= _TAIL_BOUND:
                count = float(np.sum(truncated_f >= bucket_low_f))
            elif temp_low_c <= -_TAIL_BOUND:
                count = float(np.sum(truncated_f <= bucket_high_f))
            else:
                count = float(np.sum((truncated_f >= bucket_low_f) & (truncated_f <= bucket_high_f)))
        else:
            truncated = np.floor(arr)
            if temp_high_c >= _TAIL_BOUND:
                count = float(np.sum(truncated >= temp_low_c))
            elif temp_low_c <= -_TAIL_BOUND:
                count = float(np.sum(truncated <= temp_high_c))
            else:
                count = float(np.sum(truncated == temp_low_c))
        return count / len(forecast.members)

    mean = forecast.temp_high_c
    std = forecast.std if forecast.std > 0.5 else 2.5
    from math import erf, sqrt

    if temp_high_c >= _TAIL_BOUND:
        z = (temp_low_c - mean) / std
        p = 0.5 * (1.0 - erf(z / sqrt(2)))
    elif temp_low_c <= -_TAIL_BOUND:
        z = (temp_high_c - mean) / std
        p = 0.5 * (1.0 + erf(z / sqrt(2)))
    elif temp_unit == "F":
        z_low = (temp_low_c - mean) / std
        z_high = (temp_high_c + 1.0 / 1.8 - mean) / std
        p = 0.5 * (erf(z_high / sqrt(2)) - erf(z_low / sqrt(2)))
    else:
        z_low = (temp_low_c - mean) / std
        z_high = (temp_low_c + 1.0 - mean) / std
        p = 0.5 * (erf(z_high / sqrt(2)) - erf(z_low / sqrt(2)))
    return max(0.0, min(1.0, p))


# ── Strategy-layer helpers (were referenced by strategies/emos_edge.py
#    but never implemented; added 2026-10-18) ──

_calibrator_cache: dict[str, object] = {}

# Last METAR observation per city (°C). Populated by the METAR poller;
# strategies read it for the final-hours boost. Empty = no boost.
_last_metar_obs: dict[str, float] = {}


def _get_emos_calibrator(city: str):
    """Return a trained EMOSCalibrator for the city (cached).

    Loads coefficients from data/emos_coeffs/emos_all.json, falling back
    to per-city files, then to defaults.
    """
    if city in _calibrator_cache:
        return _calibrator_cache[city]
    from pm_bot.core.emos import EMOSCalibrator

    coeffs: dict = {}
    try:
        from pathlib import Path

        for cand in (
            Path("data/emos_coeffs/emos_all.json"),
            Path(__file__).resolve().parents[3] / "data" / "emos_coeffs" / "emos_all.json",
        ):
            if cand.exists():
                import json

                all_coeffs = json.loads(cand.read_text())
                coeffs = all_coeffs.get(city, {})
                break
        if not coeffs:
            per_city = Path(__file__).resolve().parents[3] / "data" / "emos_coeffs" / f"{city.lower().replace(' ', '_')}.json"
            if per_city.exists():
                import json

                data = json.loads(per_city.read_text())
                coeffs = data.get("coeffs", data)
    except Exception:
        coeffs = {}
    cal = EMOSCalibrator(coeffs=coeffs or None, city=city)
    _calibrator_cache[city] = cal
    return cal


def bucket_probability(
    forecast: ForecastResult,
    temp_low_c: float,
    temp_high_c: float,
    temp_unit: str = "C",
    use_emos: bool = True,
) -> float:
    """Calibrated bucket probability for a forecast.

    Uses the city's trained EMOS calibrator when available and requested;
    falls back to the raw ensemble member distribution otherwise.
    """
    members = forecast.members or []
    if use_emos and members:
        try:
            cal = _get_emos_calibrator(forecast.city)
            return cal.calibrate_probability(members, temp_low_c, temp_high_c, temp_unit)
        except Exception:
            pass
    if members:
        return bucket_probability_numpy(forecast, temp_low_c, temp_high_c, temp_unit)
    return 0.0


def _hours_until_resolution(date: str) -> float:
    """Hours from now until end of the given date (YYYY-MM-DD).

    Weather markets resolve on the day's observed high, so the effective
    deadline is local end-of-day for the market date.
    """
    from datetime import datetime

    try:
        day = datetime.strptime(date[:10], "%Y-%m-%d")
        # End of that day, local time
        deadline = day.replace(hour=23, minute=59, second=59)
        delta = (deadline - datetime.now()).total_seconds() / 3600.0
        return max(0.0, delta)
    except Exception:
        return 24.0
