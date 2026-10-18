#!/usr/bin/env python3
"""S1 replay data: all buckets per event (not just tail).

For each (city, date) in tail_bias_data.csv:
  - Re-find the event, get ALL markets
  - For each bucket: YES price at T-24h and T-6h (from trades)
  - Model P(YES) from previous-runs forecast + empirical error dist
  - Winner from outcomePrices

Output: research/replay/s1_data.csv
Columns: city, date, question, threshold_f, yes_24h, yes_6h, p_model, won
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from math import erf, sqrt

CITIES = {
    "New York": {"slug": "nyc", "lat": 40.7772, "lon": -73.8726},
    "Chicago": {"slug": "chicago", "lat": 41.9742, "lon": -87.9073},
    "Miami": {"slug": "miami", "lat": 25.7953, "lon": -80.2902},
    "London": {"slug": "london", "lat": 51.5048, "lon": 0.0495},
    "Tokyo": {"slug": "tokyo", "lat": 35.5522, "lon": 139.7796},
}
MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]

# Empirical 24h GFS error (n=150): bias -0.54C, sigma 0.90C
BIAS_C = -0.54
SIGMA_C = 0.90


def get_json(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "pm-bot-s1/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def bucket_p(fcst_c: float, lo_f: float | None, hi_f: float | None) -> float:
    """P(bucket) under N(fcst+bias, sigma). lo/hi in Fahrenheit, None = open."""
    mu = fcst_c + BIAS_C
    s = SIGMA_C
    def to_c(f): return (f - 32) * 5 / 9
    if hi_f is None:  # open top: >= lo
        return 1.0 - norm_cdf((to_c(lo_f) - mu) / s)
    if lo_f is None:  # open bottom: <= hi
        return norm_cdf((to_c(hi_f) - mu) / s)
    return norm_cdf((to_c(hi_f) - mu) / s) - norm_cdf((to_c(lo_f) - mu) / s)


def parse_bucket(question: str) -> tuple[float | None, float | None] | None:
    """Parse (lo_f, hi_f) from question. None = open end."""
    # "between 62-63°F" -> (62, 63)
    m = re.search(r"between\s+(-?\d+)-(-?\d+)\s*°?F", question, re.I)
    if m: return (float(m.group(1)), float(m.group(2)))
    # "62°F or higher" / "≥62°F" -> (62, None)
    m = re.search(r"(-?\d+)\s*°?F\s+or\s+higher", question, re.I)
    if m: return (float(m.group(1)), None)
    m = re.search(r"[≥>=]\s*(-?\d+)\s*°?F", question)
    if m: return (float(m.group(1)), None)
    # "be 75°F" (single) -> treat as (74, 75)? Actually single-value buckets
    # on Polymarket are usually "X°F" meaning exactly X; use (X-1, X)
    m = re.search(r"be\s+(-?\d+)\s*°?F(?!\s*or)", question, re.I)
    if m:
        v = float(m.group(1)); return (v - 1, v)
    # Celsius (London/Tokyo): convert
    m = re.search(r"between\s+(-?\d+)-(-?\d+)\s*°?C", question, re.I)
    if m:
        lo_c, hi_c = float(m.group(1)), float(m.group(2))
        return (lo_c * 9 / 5 + 32, hi_c * 9 / 5 + 32)
    m = re.search(r"(-?\d+)\s*°?C\s+or\s+higher", question, re.I)
    if m:
        return (float(m.group(1)) * 9 / 5 + 32, None)
    return None


def find_event(city_slug: str, date_str: str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    month = MONTHS[d.month - 1]
    for q in (f"highest-temperature-in-{city_slug}-on-{month}-{d.day}-{d.year}",
              f"highest-temperature-in-{city_slug}-on-{month}-{d.day}"):
        try:
            res = get_json(f"https://gamma-api.polymarket.com/public-search?q={q}")
            for e in res.get("events", []):
                if str(d.day) in e.get("ticker", "") and month in e.get("ticker", ""):
                    return e
            if res.get("events"): return res["events"][0]
        except Exception: pass
        time.sleep(0.3)
    return None


def yes_price_before(condition_id: str, cutoff_ts: float, token_ids: list) -> float | None:
    """Last YES-normalized trade price at/before cutoff."""
    try:
        trades = get_json(f"https://data-api.polymarket.com/trades?market={condition_id}&limit=200")
    except Exception:
        return None
    yes_tid = token_ids[0] if len(token_ids) > 0 else ""
    no_tid = token_ids[1] if len(token_ids) > 1 else ""
    for t in trades:  # newest first
        ts = t.get("timestamp", 0)
        if ts > cutoff_ts: continue
        p = float(t["price"])
        asset = str(t.get("asset", ""))
        if asset == no_tid: p = 1.0 - p
        elif asset != yes_tid:
            # fallback by outcome field
            if str(t.get("outcome", "")).lower() == "no": p = 1.0 - p
        return max(0.001, min(0.999, p))
    return None


def prev_fcst(lat: float, lon: float, date_str: str) -> float | None:
    try:
        d = get_json(f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                     f"&daily=temperature_2m_max&models=gfs_seamless&past_days=35&timezone=auto")
        i = d["daily"]["time"].index(date_str)
        v = d["daily"]["temperature_2m_max"][i]
        return float(v) if v is not None else None
    except Exception:
        return None


def main():
    import sys
    # usage: fetch_s1_data.py [start_date] [end_date] (YYYY-MM-DD)
    # defaults: 2026-07-20 to 2026-10-17 (90 days, covers summer)
    start_s = sys.argv[1] if len(sys.argv) > 1 else "2026-07-20"
    end_s = sys.argv[2] if len(sys.argv) > 2 else "2026-10-17"
    out_path = "/home/hatch/workspace/pm-bot/research/replay/s1_data_90d.csv"

    start_d = datetime.strptime(start_s, "%Y-%m-%d")
    end_d = datetime.strptime(end_s, "%Y-%m-%d")
    pairs = []
    d = start_d
    while d <= end_d:
        for city in CITIES:
            pairs.append((city, d.strftime("%Y-%m-%d")))
        d += timedelta(days=1)
    print(f"fetching {len(pairs)} city-days ({start_s} to {end_s})...", flush=True)

    rows = []
    for city, date_str in pairs:
        cfg = CITIES[city]
        print(f"{city} {date_str}...", flush=True)
        ev = find_event(cfg["slug"], date_str)
        if not ev: continue
        try:
            res_ts = datetime.fromisoformat(ev["endDate"].replace("Z", "+00:00")).timestamp()
        except Exception: continue
        cutoff_24h = res_ts - 24 * 3600
        cutoff_6h = res_ts - 6 * 3600

        fcst = prev_fcst(cfg["lat"], cfg["lon"], date_str)
        if fcst is None:
            print(f"  no forecast", flush=True); continue
        time.sleep(0.2)

        markets = ev.get("markets", [])
        # winner
        winner_idx = None
        for i, m in enumerate(markets):
            try:
                ops = json.loads(m.get("outcomePrices", "[]")) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
                if ops and float(ops[0]) >= 0.99:
                    winner_idx = i; break
            except Exception: pass

        for i, m in enumerate(markets):
            q = m.get("question", "")
            bounds = parse_bucket(q)
            if not bounds: continue
            lo_f, hi_f = bounds
            tids = m.get("clobTokenIds", [])
            if isinstance(tids, str):
                try: tids = json.loads(tids)
                except Exception: tids = []
            cid = m.get("conditionId", "")

            y24 = yes_price_before(cid, cutoff_24h, tids) if cid else None
            time.sleep(0.25)
            y6 = yes_price_before(cid, cutoff_6h, tids) if cid else None
            time.sleep(0.25)

            pm = bucket_p(fcst, lo_f, hi_f)
            thresh = hi_f if hi_f else (lo_f if lo_f else 0)
            rows.append({
                "city": city, "date": date_str,
                "question": q[:60], "lo_f": lo_f or "", "hi_f": hi_f or "",
                "yes_24h": round(y24, 4) if y24 else "",
                "yes_6h": round(y6, 4) if y6 else "",
                "p_model": round(pm, 4),
                "won": 1 if i == winner_idx else 0,
            })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["city","date","question","lo_f","hi_f",
                                          "yes_24h","yes_6h","p_model","won"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} buckets)")


if __name__ == "__main__":
    main()
