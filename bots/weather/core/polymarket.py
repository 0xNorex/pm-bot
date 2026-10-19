from __future__ import annotations

import re

import httpx
import structlog
from cachetools import TTLCache

from pm_bot.models.config import CACHE_TTL
from pm_bot.models.market import TemperatureBucket, WeatherEvent
from pm_bot.core.parser import parse_bucket

log = structlog.get_logger()

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

_cache: dict[str, TTLCache] = {}


def _get_cache(name: str) -> TTLCache:
    if name not in _cache:
        _cache[name] = TTLCache(maxsize=256, ttl=CACHE_TTL.get(name, 300))
    return _cache[name]


async def _get(client: httpx.AsyncClient, url: str, cache_name: str, **params) -> dict | list:  # type: ignore[type-arg]
    cache = _get_cache(cache_name)
    key = f"{url}?{sorted(params.items())}"
    if key in cache:
        return cache[key]  # type: ignore[no-any-return]
    log.debug("api_request", url=url, params=params)
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    cache[key] = data
    return data  # type: ignore[no-any-return]


async def fetch_weather_events(client: httpx.AsyncClient, include_closed: bool = False) -> list[WeatherEvent]:
    # Discover via public-search, one query per city for today's event.
    # The old blind pagination over /events assumed 500 results/page, but the
    # gamma API caps pages at 100 — so it only ever scanned the first 100 open
    # events (top-volume markets) and never reached the weather series, meaning
    # every daemon scan silently found nothing and booked no signals.
    from datetime import date as _date

    from pm_bot.core.city_selector import WEATHER_SERIES

    today = _date.today()
    date_str = f"{today.strftime('%B')} {today.day}"  # e.g. "October 19"

    events: list[WeatherEvent] = []
    skipped = 0
    for city in WEATHER_SERIES:
        query = f"Highest temperature in {city} on {date_str}"
        try:
            data = await _get(client, f"{GAMMA_BASE}/public-search", "weather_search", q=query)
        except httpx.HTTPError:
            continue
        evs = data.get("events", []) if isinstance(data, dict) else []
        for ev in evs:
            if not isinstance(ev, dict):
                continue
            if not include_closed and str(ev.get("endDate", "") or "")[:10] < today.isoformat():
                continue
            event = _parse_event(ev)
            if event and event.city == city and (include_closed or event.date == date_str):
                events.append(event)
            elif _is_weather_event(ev):
                skipped += 1

    if skipped:
        log.info("events_skipped_no_buckets", count=skipped)

    return events


def _is_weather_event(ev: dict) -> bool:
    title = ev.get("title", "").lower()
    return "temperature" in title or "highest temp" in title or "high temp" in title


def _parse_event(ev: dict) -> WeatherEvent | None:
    title = ev.get("title", "")
    title_lower = title.lower()

    if not ("temperature" in title_lower or "highest temp" in title_lower):
        # Also check if any market has temperature buckets
        has_temp_bucket = False
        for m in ev.get("markets", []):
            if isinstance(m, dict):
                q = m.get("question", "")
                if "°f" in q.lower() or "°c" in q.lower() or "°F" in q or "°C" in q:
                    has_temp_bucket = True
                    break
        if not has_temp_bucket:
            return None

    city, date = _extract_city_date(title)
    if not city:
        return None

    description = ev.get("description", "") or ""
    airport = _extract_airport(description)

    buckets: list[TemperatureBucket] = []
    for m in ev.get("markets", []):
        if not isinstance(m, dict):
            continue
        question = m.get("question", "")
        mid = str(m.get("id", m.get("condition_id", "")))
        yes_price, no_price = _parse_prices(m)
        volume = float(m.get("volume", 0) or 0)
        bucket = parse_bucket(question, mid, yes_price, no_price, volume)
        if bucket:
            buckets.append(bucket)

    if not buckets:
        return None

    def sort_key(b: TemperatureBucket) -> float:
        if b.is_low_tail:
            return float("-inf")
        return b.temp_low_c

    buckets.sort(key=sort_key)

    return WeatherEvent(
        event_id=str(ev.get("id", "")),
        title=title,
        slug=ev.get("slug", ""),
        city=city,
        date=date,
        measure_type="high",
        airport_code=airport,
        buckets=buckets,
    )


def _parse_prices(market: dict) -> tuple[float, float]:
    outcomes = market.get("outcomes", "")
    prices = market.get("outcomePrices", "")
    yes_price = 0.0
    no_price = 0.0

    if isinstance(outcomes, str) and isinstance(prices, str):
        outcome_list = [o.strip().strip('"') for o in outcomes.strip("[]").split(",")]
        price_list = [p.strip().strip('"') for p in prices.strip("[]").split(",")]
        if len(outcome_list) == len(price_list):
            for o, p in zip(outcome_list, price_list):
                try:
                    val = float(p)
                except ValueError:
                    continue
                if o.upper() == "YES":
                    yes_price = val
                elif o.upper() == "NO":
                    no_price = val

    return yes_price, no_price


_CITY_PATTERNS: list[str] = [
    "New York",
    "NYC",
    "Los Angeles",
    "LA",
    "Chicago",
    "Miami",
    "Dallas",
    "San Francisco",
    "Atlanta",
    "London",
    "Paris",
    "Hong Kong",
    "Seoul",
    "Tokyo",
    "Shanghai",
    "Buenos Aires",
    "Jeddah",
    "Ankara",
    "Lagos",
    "São Paulo",
    "Sao Paulo",
    "Angeles",
]

# Map shorter names to canonical form
_CITY_CANONICAL: dict[str, str] = {
    "NYC": "New York",
    "LA": "Los Angeles",
    "New York's Central Park": "New York",
}


def _extract_city_date(title: str) -> tuple[str | None, str]:
    city = None
    # Longest-first + word boundaries: avoids "LA" matching inside "Dallas",
    # and lets "Los Angeles" win over bare "LA".
    for c in sorted(_CITY_PATTERNS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(c)}\b", title, re.IGNORECASE):
            city = _CITY_CANONICAL.get(c, c)
            break

    # Try "on January 22" or "on Jan 22" format
    date_match = re.search(r"on\s+(\w+\s+\d{1,2})", title, re.I)
    date = date_match.group(1) if date_match else ""

    # Also try "Month Day, Year" format
    if not date:
        date_match = re.search(r"(\w+\s+\d{1,2},?\s+\d{4})", title)
        date = date_match.group(1) if date_match else ""

    return city, date


def _extract_airport(description: str) -> str | None:
    # Pattern: "LaGuardia Airport Station (KLGA)" or "/KLGA" in wunderground URL
    m = re.search(r"(?:Station|station|airport|Airport)\s*\(([A-Z]{4})\)", description)
    if m:
        return m.group(1)
    # Pattern: wunderground URL ending in /KLGA
    m = re.search(r"wunderground\.com.*?/([A-Z]{4})", description)
    if m:
        return m.group(1)
    # Fallback: any 4-letter uppercase code near "recorded at"
    m = re.search(r"recorded at.*?([A-Z]{4})", description)
    if m:
        return m.group(1)
    return None
