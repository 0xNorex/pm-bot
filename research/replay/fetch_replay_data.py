#!/usr/bin/env python3
"""Replay Polymarket daily high-temperature tail buckets.

For each city/day in the requested 30-day window this script:

1. Finds the daily "highest temperature" event through Gamma public-search.
2. Picks the highest temperature bucket (the upper tail).
3. Finds the last trade at or before (event endDate - 24 h) through the
   Polymarket Data API, normalized to the YES leg so it is an implied
   probability for the tail bucket.
4. Resolves the winning bucket from Gamma's resolved outcome prices and
   records whether the tail bucket won.
5. Pulls the observed daily maximum from the Open-Meteo archive as a
   cross-check (the market itself resolves on airport station data).

Output: /home/hatch/workspace/pm-bot/research/replay/tail_bias_data.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - dependency check only
    print("requests is required. Install it with: pip install requests", file=sys.stderr)
    raise SystemExit(1)


GAMMA_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
DATA_TRADES_URL = "https://data-api.polymarket.com/trades"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CITIES: dict[str, tuple[float, float]] = {
    "New York": (40.7772, -73.8726),
    "Chicago": (41.9742, -87.9073),
    "Miami": (25.7953, -80.2902),
    "London": (51.5048, 0.0495),
    "Tokyo": (35.5522, 139.7796),
}
CITY_ORDER = {city: index for index, city in enumerate(CITIES)}

# The requested replay window: the last 30 complete days as of 2026-10-18.
START_DATE = date(2026, 9, 18)
END_DATE = date(2026, 10, 17)
OUTPUT_CSV = Path("/home/hatch/workspace/pm-bot/research/replay/tail_bias_data.csv")

CSV_COLUMNS = (
    "city",
    "date",
    "tail_question",
    "tail_threshold_f",
    "price_24h_before",
    "won",
    "observed_max_c",
    "n_markets",
)

MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
MONTH_NUMBERS = {name: index for index, name in enumerate(MONTHS, start=1)}

REQUEST_INTERVAL_S = 0.3
REQUEST_TIMEOUT_S = 30
TRADE_PAGE_LIMIT = 200
MAX_TRADE_PAGES = 100

# Matches "72°F", "between 62-63°F", "62-63F", "19°C", "-5--3°C", etc.
TEMP_RE = re.compile(
    r"(?P<low>-?\d{1,3})\s*(?:[-\u2013\u2014]\s*(?P<high>-?\d{1,3}))?"
    r"\s*(?:\u00b0\s*)?(?P<unit>[CF])\b",
    re.IGNORECASE,
)
SLUG_DATE_RE = re.compile(r"-on-([a-z]+)-(\d{1,2})-(\d{4})$", re.IGNORECASE)


@dataclass
class Row:
    city: str
    date: str
    tail_question: str
    tail_threshold_f: float
    price_24h_before: float | None
    won: int
    observed_max_c: float | None
    n_markets: int
    price_fallback: bool = False
    price_note: str = ""


class RateLimiter:
    """Ensure at least `interval` seconds between request starts."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._last_start = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delay = self.interval - (now - self._last_start)
        if delay > 0:
            time.sleep(delay)
        self._last_start = time.monotonic()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "pm-bot-tail-replay/1.0 (research)",
            "Accept": "application/json",
        }
    )
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    limiter: RateLimiter,
    *,
    label: str,
    attempts: int = 3,
) -> Any:
    """GET JSON with rate limiting, retries, and graceful failure."""
    for attempt in range(1, attempts + 1):
        limiter.wait()
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            if attempt == attempts:
                log(f"[WARN] {label}: request failed after {attempts} attempts: {exc}")
                return None
            time.sleep(1.5 * attempt)
            continue

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                if attempt == attempts:
                    log(f"[WARN] {label}: invalid JSON: {exc}")
                    return None
                time.sleep(1.0 * attempt)
                continue

        if response.status_code in (400, 401, 403, 404):
            log(f"[WARN] {label}: HTTP {response.status_code}: {response.text[:200]}")
            return None

        if response.status_code == 429 or 500 <= response.status_code < 600:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 1.5 * attempt
            except (TypeError, ValueError):
                delay = 1.5 * attempt
            delay = min(delay, 30.0)
            if attempt == attempts:
                log(f"[WARN] {label}: HTTP {response.status_code} after {attempts} attempts")
                return None
            log(f"[WARN] {label}: HTTP {response.status_code}; retrying in {delay:.1f}s")
            time.sleep(delay)
            continue

        log(f"[WARN] {label}: unexpected HTTP {response.status_code}: {response.text[:200]}")
        return None

    return None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_list_field(value: Any) -> list[Any]:
    """Parse Gamma list fields, which are usually JSON strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return parsed

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip().strip('"').strip("'") for part in text.split(",") if part.strip()]


def parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def event_date(event: dict[str, Any]) -> date | None:
    """Return the local market date encoded in the event slug or endDate."""
    slug = str(event.get("slug") or "").strip().lower()
    match = SLUG_DATE_RE.search(slug)
    if match:
        month = MONTH_NUMBERS.get(match.group(1).lower())
        if month:
            try:
                return date(int(match.group(3)), month, int(match.group(2)))
            except ValueError:
                pass

    end_dt = parse_iso_datetime(event.get("endDate"))
    return end_dt.date() if end_dt else None


def parse_temp_threshold_f(question: str) -> float | None:
    """Return the highest temperature bound in the question, in Fahrenheit.

    Celsius questions (London/Tokyo) are converted so tail_threshold_f is
    comparable across cities.  For ranges the upper bound is used; for
    "X or higher" the single published number is the tail threshold.
    """
    best: float | None = None
    for match in TEMP_RE.finditer(question):
        raw = match.group("high") or match.group("low")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if match.group("unit").upper() == "C":
            value = value * 9.0 / 5.0 + 32.0
        if best is None or value > best:
            best = value
    return best


def pick_tail_market(markets: list[dict[str, Any]]) -> tuple[float, int, str] | None:
    """Pick the market with the highest temperature bound."""
    best: tuple[float, int, str] | None = None
    for index, market in enumerate(markets):
        question = str(market.get("question") or "")
        threshold = parse_temp_threshold_f(question)
        if threshold is None:
            continue
        if best is None or threshold > best[0]:
            best = (threshold, index, question)
    return best


def market_yes_price(market: dict[str, Any]) -> float | None:
    """Read the YES resolution price from a Gamma market."""
    outcomes = parse_list_field(market.get("outcomes"))
    prices = parse_list_field(market.get("outcomePrices"))
    if not prices:
        return None

    yes_index = 0
    for index, outcome in enumerate(outcomes):
        if str(outcome).strip().lower() == "yes":
            yes_index = index
            break
    if yes_index >= len(prices):
        yes_index = 0
    return as_float(prices[yes_index])


def resolve_winner(markets: list[dict[str, Any]]) -> tuple[int | None, str]:
    """Identify the winning bucket.

    Resolved outcomePrices are the primary signal because they are the
    settlement record.  Gamma's lastTradePrice is used as a fallback (and to
    disambiguate) as requested by the replay spec; on older events it can be
    stale and report 0.99+ for several buckets at once.
    """
    resolved_candidates: list[tuple[float, int]] = []
    last_candidates: list[tuple[float, int]] = []

    for index, market in enumerate(markets):
        yes_price = market_yes_price(market)
        if yes_price is not None and yes_price >= 0.99:
            resolved_candidates.append((yes_price, index))

        last_price = as_float(market.get("lastTradePrice"))
        if last_price is not None and last_price >= 0.99:
            last_candidates.append((last_price, index))

    if len(resolved_candidates) == 1:
        return resolved_candidates[0][1], "outcomePrices"

    if len(resolved_candidates) > 1:
        resolved_candidates.sort(reverse=True)
        if resolved_candidates[0][0] > resolved_candidates[1][0] + 1e-9:
            return resolved_candidates[0][1], "outcomePrices-max"

    if len(last_candidates) == 1:
        return last_candidates[0][1], "lastTradePrice"

    if len(last_candidates) > 1:
        last_candidates.sort(reverse=True)
        if last_candidates[0][0] > last_candidates[1][0] + 1e-9:
            return last_candidates[0][1], "lastTradePrice-max"

    return None, "unresolved"


def trade_yes_price(trade: dict[str, Any], yes_token_id: str, no_token_id: str) -> float | None:
    """Convert a trade price to the YES-leg implied probability.

    Data API prices are quoted for the traded outcome.  A NO trade at 0.999 is
    a YES price of 0.001, so raw prices cannot be used without normalizing.
    """
    price = as_float(trade.get("price"))
    if price is None:
        return None

    asset = str(trade.get("asset") or "").strip()
    if yes_token_id and asset == yes_token_id:
        value = price
    elif no_token_id and asset == no_token_id:
        value = 1.0 - price
    else:
        outcome_index = trade.get("outcomeIndex")
        try:
            outcome_index = int(outcome_index)
        except (TypeError, ValueError):
            outcome_index = None

        if outcome_index == 0:
            value = price
        elif outcome_index == 1:
            value = 1.0 - price
        else:
            outcome = str(trade.get("outcome") or "").strip().lower()
            if outcome == "no":
                value = 1.0 - price
            else:
                # "yes" or an unknown shape: assume the price is for YES.
                value = price

    return min(1.0, max(0.0, value))


def fetch_tail_trade_price(
    session: requests.Session,
    limiter: RateLimiter,
    market: dict[str, Any],
    cutoff_ts: float,
    label: str,
) -> tuple[float | None, bool, str]:
    """Return (YES price, used_earliest_fallback, note)."""
    condition_id = str(market.get("conditionId") or "").strip()
    if not condition_id:
        return None, False, "missing conditionId"

    token_ids = parse_list_field(market.get("clobTokenIds"))
    yes_token_id = str(token_ids[0]).strip() if len(token_ids) > 0 else ""
    no_token_id = str(token_ids[1]).strip() if len(token_ids) > 1 else ""

    best: tuple[float, float] | None = None  # (timestamp, YES price)
    earliest: tuple[float, float] | None = None
    offset = 0
    pages = 0
    request_failed = False
    truncated = False

    while pages < MAX_TRADE_PAGES:
        payload = get_json(
            session,
            DATA_TRADES_URL,
            {"market": condition_id, "limit": TRADE_PAGE_LIMIT, "offset": offset},
            limiter,
            label=f"trades {label} offset={offset}",
        )
        pages += 1

        if payload is None:
            request_failed = True
            break
        if not isinstance(payload, list):
            log(f"[WARN] trades {label}: unexpected response shape: {str(payload)[:200]}")
            request_failed = True
            break
        if not payload:
            break

        for trade in payload:
            if not isinstance(trade, dict):
                continue

            timestamp = as_float(trade.get("timestamp"))
            if timestamp is None:
                continue
            if timestamp > 1e12:  # tolerate millisecond timestamps
                timestamp /= 1000.0

            price = trade_yes_price(trade, yes_token_id, no_token_id)
            if price is None:
                continue

            if earliest is None or timestamp < earliest[0]:
                earliest = (timestamp, price)

            if timestamp <= cutoff_ts and (best is None or timestamp > best[0]):
                best = (timestamp, price)

        # The Data API returns newest trades first.  Once this page contains a
        # trade at/before the cutoff, the newest such trade has been found.
        if best is not None:
            break

        if len(payload) < TRADE_PAGE_LIMIT:
            break

        offset += TRADE_PAGE_LIMIT
    else:
        truncated = True

    if best is not None:
        when = datetime.fromtimestamp(best[0], timezone.utc).isoformat()
        return best[1], False, f"last trade <= cutoff at {when}"

    if earliest is not None:
        when = datetime.fromtimestamp(earliest[0], timezone.utc).isoformat()
        reasons = []
        if request_failed:
            reasons.append("request failed before cutoff")
        if truncated:
            reasons.append("pagination limit")
        if not reasons:
            reasons.append("no trade at/before cutoff")
        return earliest[1], True, f"FALLBACK earliest trade at {when} ({'; '.join(reasons)})"

    return None, False, "no trades"


def fetch_observed_max_c(
    session: requests.Session,
    limiter: RateLimiter,
    latitude: float,
    longitude: float,
    day: date,
    label: str,
) -> float | None:
    payload = get_json(
        session,
        ARCHIVE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "daily": "temperature_2m_max",
            # Local-day max, matching how the temperature market is dated.
            "timezone": "auto",
        },
        limiter,
        label=f"archive {label}",
    )
    if not isinstance(payload, dict):
        return None

    daily = payload.get("daily")
    if not isinstance(daily, dict):
        return None

    values = daily.get("temperature_2m_max")
    if isinstance(values, list) and values:
        return as_float(values[0])
    return None


def search_events(
    session: requests.Session,
    limiter: RateLimiter,
    query: str,
) -> list[dict[str, Any]]:
    payload = get_json(
        session,
        GAMMA_SEARCH_URL,
        {"q": query},
        limiter,
        label=f"search {query!r}",
    )
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    if isinstance(payload, dict):
        events = payload.get("events")
        if isinstance(events, list):
            return [event for event in events if isinstance(event, dict)]
    return []


def find_event_for_day(
    session: requests.Session,
    limiter: RateLimiter,
    city: str,
    day: date,
) -> tuple[dict[str, Any] | None, str | None]:
    city_slug = city.lower().replace(" ", "-")
    month = MONTHS[day.month - 1]
    queries = (
        f"highest-temperature-in-{city_slug}-on-{month}-{day.day}",
        f"highest-temperature-in-{city_slug}-on-{day.year}-{month}-{day.day}",
    )

    smallest_match: tuple[dict[str, Any], str] | None = None
    for query in queries:
        events = search_events(session, limiter, query)
        matches = [event for event in events if event_date(event) == day]
        if not matches:
            log(f"[INFO] no search hit for {city} {day} via {query!r}")
            continue

        def rank(event: dict[str, Any]) -> tuple[int, int]:
            slug = str(event.get("slug") or "")
            year_bonus = 1 if slug.endswith(f"-{day.year}") else 0
            markets = event.get("markets")
            market_count = len(markets) if isinstance(markets, list) else 0
            return (year_bonus, market_count)

        matches.sort(key=rank, reverse=True)
        event = matches[0]
        markets = event.get("markets")
        if isinstance(markets, list) and len(markets) >= 5:
            return event, query

        # Keep the small event as a last resort, but try the year-qualified
        # query in case search returned a truncated market list.
        if smallest_match is None:
            smallest_match = (event, query)

    if smallest_match is not None:
        return smallest_match
    return None, None


def process_day(
    session: requests.Session,
    limiter: RateLimiter,
    city: str,
    latitude: float,
    longitude: float,
    day: date,
) -> tuple[Row | None, str | None]:
    event, query = find_event_for_day(session, limiter, city, day)
    if event is None:
        log(f"[SKIP] {city} {day}: no event found")
        return None, "no_event"

    if event.get("closed") is False:
        log(f"[WARN] {city} {day}: event {event.get('id')} is not closed")

    raw_markets = event.get("markets")
    if not isinstance(raw_markets, list):
        log(f"[SKIP] {city} {day}: event has no markets list")
        return None, "no_markets"

    markets = [market for market in raw_markets if isinstance(market, dict)]
    n_markets = len(markets)
    if n_markets < 5:
        log(f"[SKIP] {city} {day}: only {n_markets} markets")
        return None, "too_few_markets"

    tail = pick_tail_market(markets)
    if tail is None:
        log(f"[SKIP] {city} {day}: no parseable temperature buckets")
        return None, "no_tail"

    threshold_f, tail_index, tail_question = tail
    tail_market = markets[tail_index]

    end_dt = parse_iso_datetime(event.get("endDate"))
    if end_dt is None:
        for market in markets:
            end_dt = parse_iso_datetime(market.get("endDate"))
            if end_dt is not None:
                break
    if end_dt is None:
        log(f"[SKIP] {city} {day}: no endDate")
        return None, "no_end_date"

    cutoff_ts = (end_dt - timedelta(hours=24)).timestamp()

    winner_index, resolution_source = resolve_winner(markets)
    if winner_index is None:
        log(f"[SKIP] {city} {day}: could not resolve winner")
        return None, "unresolved"
    if resolution_source != "outcomePrices":
        log(f"[INFO] {city} {day}: winner resolved from {resolution_source}")

    won = int(winner_index == tail_index)

    price, used_fallback, price_note = fetch_tail_trade_price(
        session,
        limiter,
        tail_market,
        cutoff_ts,
        f"{city} {day} ({query})",
    )
    if price is None:
        log(f"[WARN] {city} {day}: no usable trade price ({price_note})")
    elif used_fallback:
        log(f"[WARN] {city} {day}: {price_note} -> price={price:.4f}")

    observed_max_c = fetch_observed_max_c(
        session,
        limiter,
        latitude,
        longitude,
        day,
        f"{city} {day}",
    )
    if observed_max_c is None:
        log(f"[WARN] {city} {day}: no Open-Meteo archive value")

    price_text = format_number(price, 6) if price is not None else "NA"
    observed_text = format_number(observed_max_c, 2) if observed_max_c is not None else "NA"
    threshold_text = format_number(threshold_f, 1)
    suffix = f" [{price_note}]" if used_fallback else ""
    log(
        f"[ROW] {city} {day}: n={n_markets} tail={threshold_text}F "
        f"price={price_text} won={won} obs={observed_text}C{suffix}"
    )

    return (
        Row(
            city=city,
            date=day.isoformat(),
            tail_question=tail_question,
            tail_threshold_f=threshold_f,
            price_24h_before=price,
            won=won,
            observed_max_c=observed_max_c,
            n_markets=n_markets,
            price_fallback=used_fallback,
            price_note=price_note,
        ),
        None,
    )


def date_range(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def format_number(value: float | None, decimals: int) -> str:
    if value is None:
        return ""
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(
                (
                    row.city,
                    row.date,
                    row.tail_question,
                    format_number(row.tail_threshold_f, 1),
                    format_number(row.price_24h_before, 6),
                    row.won,
                    format_number(row.observed_max_c, 2),
                    row.n_markets,
                )
            )


def print_summary(rows: list[Row], stats: Counter[str], path: Path) -> None:
    total = len(rows)
    prices = [row.price_24h_before for row in rows if row.price_24h_before is not None]
    wins = sum(row.won for row in rows)
    avg_price = sum(prices) / len(prices) if prices else None
    win_rate = wins / total if total else None

    print()
    print(f"Rows collected: {total}")
    if avg_price is None:
        print("Avg implied prob (mean price_24h_before): n/a")
    else:
        print(
            f"Avg implied prob (mean price_24h_before): {avg_price:.4f} "
            f"(n={len(prices)})"
        )
    if win_rate is None:
        print("Realized win rate (mean won): n/a")
    else:
        print(f"Realized win rate (mean won): {win_rate:.4f} ({wins}/{total})")
    print(f"CSV: {path}")

    skip_reasons = (
        "no_event",
        "no_markets",
        "too_few_markets",
        "no_tail",
        "no_end_date",
        "unresolved",
        "error",
    )
    skipped = [f"{reason}={stats[reason]}" for reason in skip_reasons if stats[reason]]
    if skipped:
        print("Skipped: " + ", ".join(skipped))

    quality = []
    if stats["fallback_price"]:
        quality.append(f"earliest-trade fallbacks={stats['fallback_price']}")
    if stats["no_price"]:
        quality.append(f"no trade price={stats['no_price']}")
    if stats["no_archive"]:
        quality.append(f"no archive value={stats['no_archive']}")
    if quality:
        print("Data quality: " + ", ".join(quality))


def main() -> int:
    session = make_session()
    limiter = RateLimiter(REQUEST_INTERVAL_S)
    days = date_range(START_DATE, END_DATE)
    rows: list[Row] = []
    stats: Counter[str] = Counter()

    log(
        f"[INFO] Replaying {len(CITIES)} cities x {len(days)} days "
        f"({START_DATE} to {END_DATE})"
    )

    for city, (latitude, longitude) in CITIES.items():
        log(f"[INFO] City: {city}")
        for day in days:
            try:
                row, reason = process_day(
                    session,
                    limiter,
                    city,
                    latitude,
                    longitude,
                    day,
                )
            except Exception as exc:  # keep the replay going on a bad market
                log(f"[ERROR] {city} {day}: {type(exc).__name__}: {exc}")
                stats["error"] += 1
                continue

            if row is None:
                stats[reason or "unknown"] += 1
                continue

            rows.append(row)
            if row.price_fallback:
                stats["fallback_price"] += 1
            if row.price_24h_before is None:
                stats["no_price"] += 1
            if row.observed_max_c is None:
                stats["no_archive"] += 1

    rows.sort(key=lambda row: (CITY_ORDER[row.city], row.date))

    try:
        write_csv(rows, OUTPUT_CSV)
    except OSError as exc:
        log(f"[ERROR] could not write {OUTPUT_CSV}: {exc}")
        return 1

    print_summary(rows, stats, OUTPUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
