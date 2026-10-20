#!/usr/bin/env python3
"""Honest S1 dataset: point-in-time forecasts + real market prices + settlement.

Forecast discipline (the fix vs the contaminated previous-runs replay):
  - Single Runs API: one specific published run per model, all initialized
    on D-1 and published before day D anywhere:
      GFS   run D-1 06:00Z (published ~D-1 10Z)
      ICON  run D-1 06:00Z
      ECMWF run D-1 00:00Z (published ~D-1 07:36Z)
  - mu = mean of the models' daily-max for local day D (from hourly),
    spread = std across models. No information from day D or later.
Market data: same method as fetch_s1_data.py (gamma event, winner from
outcomePrices, YES price = last trade at/before resolution-24h, NO trades
flipped). Buckets stored in NATIVE unit (no F/C conversion hacks).

Output: research/replay/honest_data.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

CITIES = {
    "New York": {"slug": "nyc", "lat": 40.7772, "lon": -73.8726},
    "Chicago": {"slug": "chicago", "lat": 41.9742, "lon": -87.9073},
    "London": {"slug": "london", "lat": 51.5048, "lon": 0.0495},
    "Dallas": {"slug": "dallas", "lat": 32.8471, "lon": -96.8518},
    "San Francisco": {"slug": "san-francisco", "lat": 37.6213, "lon": -122.3790},
    "Tokyo": {"slug": "tokyo", "lat": 35.5522, "lon": 139.7796},
    "Paris": {"slug": "paris", "lat": 48.9694, "lon": 2.4414},
    "Seoul": {"slug": "seoul", "lat": 37.4602, "lon": 126.4407},
}
MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]
SR = "https://single-runs-api.open-meteo.com/v1/forecast"
RUNS = [("gfs_seamless", "06:00"), ("icon_seamless", "06:00"), ("ecmwf_ifs", "00:00")]
CITY_TZ = {
    "New York": "America/New_York", "Chicago": "America/Chicago",
    "London": "Europe/London", "Dallas": "America/Chicago",
    "San Francisco": "America/Los_Angeles", "Tokyo": "Asia/Tokyo",
    "Paris": "Europe/Paris", "Seoul": "Asia/Seoul",
}


def get_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "pm-bot-honest/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_bucket(q: str):
    """-> (lo, hi, unit) in native unit; None = open end."""
    for unit in ("F", "C"):
        m = re.search(rf"between\s+(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)\s*°?{unit}\b", q, re.I)
        if m: return (float(m.group(1)), float(m.group(2)), unit)
        m = re.search(rf"(-?\d+(?:\.\d+)?)\s*°?{unit}\s+or\s+(?:higher|above|more)", q, re.I)
        if m: return (float(m.group(1)), None, unit)
        m = re.search(rf"(-?\d+(?:\.\d+)?)\s*°?{unit}\s+or\s+(?:lower|below|less)", q, re.I)
        if m: return (None, float(m.group(1)), unit)
        m = re.search(rf"[≥>]\s*(-?\d+(?:\.\d+)?)\s*°?{unit}\b", q)
        if m: return (float(m.group(1)), None, unit)
        m = re.search(rf"[≤<]\s*(-?\d+(?:\.\d+)?)\s*°?{unit}\b", q)
        if m: return (None, float(m.group(1)), unit)
        m = re.search(rf"be\s+(-?\d+(?:\.\d+)?)\s*°?{unit}(?!\s*or)", q, re.I)
        if m:
            v = float(m.group(1)); return (v, v, unit)
    return None


def find_event(slug: str, date_str: str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    month = MONTHS[d.month - 1]
    for q in (f"highest-temperature-in-{slug}-on-{month}-{d.day}-{d.year}",
              f"highest-temperature-in-{slug}-on-{month}-{d.day}"):
        try:
            res = get_json(f"https://gamma-api.polymarket.com/public-search?q={q}")
            for e in res.get("events", []):
                if str(d.day) in e.get("ticker", "") and month in e.get("ticker", ""):
                    return e
            # No unchecked fallback: returning search[0] could silently
            # grab an adjacent day's event (wrong winner/prices).
        except Exception:
            pass
        time.sleep(0.3)
    return None


def yes_price_before(condition_id: str, cutoff_ts: float, token_ids: list):
    try:
        trades = get_json(f"https://data-api.polymarket.com/trades?market={condition_id}&limit=200")
    except Exception:
        return None
    yes_tid = token_ids[0] if token_ids else ""
    no_tid = token_ids[1] if len(token_ids) > 1 else ""
    for t in trades:  # newest first
        if t.get("timestamp", 0) > cutoff_ts:
            continue
        p = float(t["price"])
        asset = str(t.get("asset", ""))
        if asset == no_tid:
            p = 1.0 - p
        elif asset != yes_tid and str(t.get("outcome", "")).lower() == "no":
            p = 1.0 - p
        return max(0.001, min(0.999, p))
    return None


def honest_forecast(lat: float, lon: float, date_str: str):
    """(mu, spread, n_models) from D-1 single runs; (None,None,0) on failure."""
    d = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    maxes = []
    for model, hhmm in RUNS:
        run = f"{d.strftime('%Y-%m-%d')}T{hhmm}"
        try:
            j = get_json(f"{SR}?latitude={lat}&longitude={lon}&hourly=temperature_2m"
                         f"&models={model}&run={run}&forecast_days=3&timezone=auto")
            times = j["hourly"]["time"]
            temps = j["hourly"]["temperature_2m"]
            day = [tt for t, tt in zip(times, temps) if tt is not None and t.startswith(date_str)]
            if day:
                maxes.append(max(day))
        except Exception:
            pass
        time.sleep(0.15)
    if len(maxes) < 2:
        return None, None, len(maxes)
    mu = sum(maxes) / len(maxes)
    var = sum((x - mu) ** 2 for x in maxes) / (len(maxes) - 1)
    return mu, var ** 0.5, len(maxes)


def main():
    start_s = sys.argv[1] if len(sys.argv) > 1 else "2026-08-20"
    end_s = sys.argv[2] if len(sys.argv) > 2 else "2026-10-19"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "/home/hatch/workspace/pm-bot/research/replay/honest_data.csv"
    start_d = datetime.strptime(start_s, "%Y-%m-%d")
    end_d = datetime.strptime(end_s, "%Y-%m-%d")
    pairs = []
    d = start_d
    while d <= end_d:
        for city in CITIES:
            pairs.append((city, d.strftime("%Y-%m-%d")))
        d += timedelta(days=1)
    print(f"fetching {len(pairs)} city-days {start_s}..{end_s}", flush=True)

    rows = []
    misses = 0
    for i, (city, date_str) in enumerate(pairs):
        cfg = CITIES[city]
        ev = find_event(cfg["slug"], date_str)
        if not ev:
            misses += 1
            continue
        try:
            res_ts = datetime.fromisoformat(ev["endDate"].replace("Z", "+00:00")).timestamp()
        except Exception:
            misses += 1
            continue
        mu, spread, nmod = honest_forecast(cfg["lat"], cfg["lon"], date_str)
        if mu is None:
            print(f"{city} {date_str}: forecast failed ({nmod} models)", flush=True)
            misses += 1
            continue
        markets = ev.get("markets", [])
        winner_idx = None
        for k, m in enumerate(markets):
            try:
                ops = m.get("outcomePrices")
                ops = json.loads(ops) if isinstance(ops, str) else ops
                if ops and float(ops[0]) >= 0.99:
                    winner_idx = k
                    break
            except Exception:
                pass
        if winner_idx is None:
            misses += 1
            continue  # unresolved: excluded, never guessed
        cutoff = res_ts - 24 * 3600
        for k, m in enumerate(markets):
            b = parse_bucket(m.get("question", ""))
            if not b:
                continue
            lo, hi, unit = b
            tids = m.get("clobTokenIds", [])
            if isinstance(tids, str):
                try:
                    tids = json.loads(tids)
                except Exception:
                    tids = []
            y24 = yes_price_before(m.get("conditionId", ""), cutoff, tids) if m.get("conditionId") else None
            time.sleep(0.12)
            y3 = None
            if date_str >= "2026-09-20":
                y3 = yes_price_before(m.get("conditionId", ""), res_ts - 3 * 3600, tids) if m.get("conditionId") else None
                time.sleep(0.12)
            y_s4 = None
            if date_str >= "2026-09-20":
                # S4 decision anchor: last trade at/before 17:30 city-local
                # on day D (S4 starts 16:00 local; 17:30 is a fill proxy).
                tz = ZoneInfo(CITY_TZ[city])
                s4_cutoff = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    hour=17, minute=30, tzinfo=tz).timestamp()
                y_s4 = yes_price_before(m.get("conditionId", ""), s4_cutoff, tids) if m.get("conditionId") else None
                time.sleep(0.12)
            rows.append({
                "city": city, "date": date_str, "question": m.get("question", "")[:70],
                "lo": "" if lo is None else lo, "hi": "" if hi is None else hi, "unit": unit,
                "yes_24h": "" if y24 is None else round(y24, 4),
                "yes_3h": "" if y3 is None else round(y3, 4),
                "yes_s4": "" if y_s4 is None else round(y_s4, 4),
                "won": 1 if k == winner_idx else 0,
                "fcst_mu": round(mu, 3), "fcst_spread": round(spread, 3), "n_models": nmod,
            })
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(pairs)} city-days, {len(rows)} buckets", flush=True)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["city", "date", "question", "lo", "hi", "unit",
                                          "yes_24h", "yes_3h", "yes_s4", "won", "fcst_mu", "fcst_spread", "n_models"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}: {len(rows)} buckets, {misses} city-days missed", flush=True)


if __name__ == "__main__":
    main()
