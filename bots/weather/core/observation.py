from __future__ import annotations

import math
import re
import structlog
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import httpx

if TYPE_CHECKING:
    from pm_bot.models.market import Recommendation

log = structlog.get_logger()

CITY_ICAO: dict[str, str] = {
    "New York": "KLGA",
    "London": "EGLC",  # settlement station per gamma resolutionSource (was EGLL)
    "Denver": "KDEN",
    "Helsinki": "EFHK",
    "Paris": "LFPB",
    "Tokyo": "RJTT",
    "Chicago": "KORD",
    "Dallas": "KDAL",
    "San Francisco": "KSFO",
    "Austin": "KAUS",
    "Seoul": "RKSI",
    "Hong Kong": "VHHH",
    "Warsaw": "EPWA",
    "Lagos": "DNMM",
    "Taipei": "RCTP",
    "Miami": "KMIA",
}

CITY_TZ: dict[str, str] = {
    "New York": "America/New_York",
    "London": "Europe/London",
    "Denver": "America/Denver",
    "Helsinki": "Europe/Helsinki",
    "Paris": "Europe/Paris",
    "Tokyo": "Asia/Tokyo",
    "Chicago": "America/Chicago",
    "Dallas": "America/Chicago",
    "San Francisco": "America/Los_Angeles",
    "Austin": "America/Chicago",
    "Seoul": "Asia/Seoul",
    "Hong Kong": "Asia/Hong_Kong",
    "Warsaw": "Europe/Warsaw",
    "Lagos": "Africa/Lagos",
    "Taipei": "Asia/Taipei",
    "Miami": "America/New_York",
}

AWC_URL = "https://aviationweather.gov/api/data/metar"
CUTOFF_HOUR = 17
SPIKE_THRESHOLD_C = 3.0


@dataclass
class ObservedTemp:
    city: str
    observed_c: float
    obs_time_utc: datetime
    local_time: datetime
    is_past_cutoff: bool
    anomaly_detected: bool = False


ObservedHigh = ObservedTemp


async def fetch_metar_obs(
    client: httpx.AsyncClient,
    icao: str,
) -> Any:
    try:
        resp = await client.get(
            AWC_URL,
            params={"ids": icao, "format": "json", "taf": "false"},
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        latest = data[0]
        return latest
    except Exception as e:
        log.warning("metar_fetch_failed", icao=icao, error=str(e))
        return None


async def fetch_previous_metar(
    client: httpx.AsyncClient,
    icao: str,
) -> Any:
    try:
        resp = await client.get(
            AWC_URL,
            params={"ids": icao, "format": "json", "taf": "false"},
        )
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2:
            return None
        return data[1]
    except Exception:
        return None


async def fetch_observation(
    client: httpx.AsyncClient,
    city: str,
) -> ObservedTemp | None:
    icao = CITY_ICAO.get(city)
    if not icao:
        log.debug("no_icao_for_city", city=city)
        return None

    tz_name = CITY_TZ.get(city, "UTC")
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)

    metar = await fetch_metar_obs(client, icao)
    if not metar:
        return None

    temp_c = metar.get("temp")
    if temp_c is None:
        return None

    try:
        temp_c = float(temp_c)
    except (ValueError, TypeError):
        return None

    obs_time_str = metar.get("obsTime", "")
    try:
        obs_time_utc = datetime.fromisoformat(obs_time_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        obs_time_utc = datetime.now(timezone.utc)

    is_past_cutoff = now_local.hour >= CUTOFF_HOUR

    anomaly = False
    prev = await fetch_previous_metar(client, icao)
    if prev and prev.get("temp") is not None:
        try:
            prev_temp = float(prev["temp"])
            delta = abs(temp_c - prev_temp)
            if delta >= SPIKE_THRESHOLD_C:
                anomaly = True
                log.warning(
                    "temp_spike_anomaly",
                    city=city,
                    icao=icao,
                    current=temp_c,
                    previous=prev_temp,
                    delta=delta,
                )
        except (ValueError, TypeError):
            pass

    return ObservedTemp(
        city=city,
        observed_c=temp_c,
        obs_time_utc=obs_time_utc,
        local_time=now_local,
        is_past_cutoff=is_past_cutoff,
        anomaly_detected=anomaly,
    )


async def fetch_observed_high(
    client: httpx.AsyncClient,
    city: str,
) -> ObservedHigh | None:
    return await fetch_observation(client, city)


async def fetch_day_high(
    client: httpx.AsyncClient,
    city: str,
    min_local_hour: int = 15,
) -> ObservedTemp | None:
    """Running max of today's METAR temps at the settlement station.

    Returns None before min_local_hour (day still developing) or when the
    station has no usable reports today. observed_c = max temp so far.
    """
    icao = CITY_ICAO.get(city)
    if not icao:
        return None
    tz = ZoneInfo(CITY_TZ.get(city, "UTC"))
    now_local = datetime.now(tz)
    if now_local.hour < min_local_hour:
        return None
    try:
        resp = await client.get(
            AWC_URL,
            params={"ids": icao, "format": "json", "taf": "false", "hours": "30"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    best: float | None = None
    best_time: datetime | None = None
    for m in data if isinstance(data, list) else []:
        t = m.get("temp")
        if t is None:
            continue
        try:
            tv = float(t)
            ot = datetime.fromisoformat(str(m.get("obsTime", "")).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ot.astimezone(tz).date() != now_local.date():
            continue
        if best is None or tv > best:
            best, best_time = tv, ot
    if best is None or best_time is None:
        return None
    return ObservedTemp(
        city=city,
        observed_c=best,
        obs_time_utc=best_time,
        local_time=now_local,
        is_past_cutoff=now_local.hour >= CUTOFF_HOUR,
    )


def _obs_dt(m: dict) -> datetime | None:
    """Parse a METAR report timestamp (ISO string or epoch, either field)."""
    for key in ("reportTime", "obsTime"):
        v = m.get(key)
        if v is None:
            continue
        try:
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(v, tz=timezone.utc)
            return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except (ValueError, TypeError, OSError):
            continue
    return None


async def fetch_day_high_for_date(
    client: httpx.AsyncClient,
    city: str,
    day: date,
) -> float | None:
    """Final METAR day high (C) for a PAST local date at the settlement station.

    Uses the AWC `date` parameter to page historical reports; returns the
    max temp among reports whose local date == day, or None.
    """
    icao = CITY_ICAO.get(city)
    if not icao:
        return None
    tz = ZoneInfo(CITY_TZ.get(city, "UTC"))
    end_local = datetime(day.year, day.month, day.day, 23, 59, tzinfo=tz)
    date_param = end_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        resp = await client.get(
            AWC_URL,
            params={"ids": icao, "format": "json", "taf": "false", "hours": "48", "date": date_param},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    best: float | None = None
    for m in data if isinstance(data, list) else []:
        t = m.get("temp")
        if t is None:
            continue
        ot = _obs_dt(m)
        if ot is None:
            continue
        try:
            tv = float(t)
        except (ValueError, TypeError):
            continue
        if ot.astimezone(tz).date() != day:
            continue
        if best is None or tv > best:
            best = tv
    return best


def resolve_icao_from_description(description: str) -> str | None:
    pattern = r"wunderground\.com/[^/]+/[^/]+/[^/]+/[^/]+/([A-Z]{4})"
    m = re.search(pattern, description, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def get_icao(city: str, description: str | None = None) -> str | None:
    if description:
        dynamic = resolve_icao_from_description(description)
        if dynamic:
            if dynamic != CITY_ICAO.get(city):
                log.info("icao_dynamic_override", city=city, static=CITY_ICAO.get(city), dynamic=dynamic)
            return dynamic
    return CITY_ICAO.get(city)


def should_filter_bucket(
    bucket_temp_low_c: float,
    obs: ObservedTemp,
) -> bool:
    if not obs.is_past_cutoff:
        return False
    if bucket_temp_low_c == float("-inf"):
        return False
    floor_obs = math.floor(obs.observed_c)
    return bucket_temp_low_c < floor_obs


def filter_recommendations(
    recs: list[Recommendation],
    obs: ObservedTemp | None,
) -> list[Recommendation]:
    if not obs or not obs.is_past_cutoff:
        return recs

    filtered = []
    for r in recs:
        if r.direction == "YES" and should_filter_bucket(r.bucket.temp_low_c, obs):
            log.debug(
                "filtering_impossible_yes",
                strategy=r.strategy,
                city=r.city,
                bucket_low_c=r.bucket.temp_low_c,
                observed_c=obs.observed_c,
            )
            continue
        if r.direction == "NO" and not should_filter_bucket(r.bucket.temp_low_c, obs):
            floor_obs = math.floor(obs.observed_c)
            if r.bucket.temp_low_c == floor_obs:
                log.debug(
                    "filtering_confirmed_no",
                    strategy=r.strategy,
                    city=r.city,
                    bucket_low_c=r.bucket.temp_low_c,
                    floor_obs=floor_obs,
                )
                continue
        filtered.append(r)
    return filtered
