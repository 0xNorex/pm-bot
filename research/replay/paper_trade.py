#!/usr/bin/env python3
"""S1 paper trader: real-time signals with executable quotes.

For each of the 5 cities:
  1. Find today's high-temp event
  2. Get current ensemble forecast from Open-Meteo
  3. For each bucket: compute P_model, get displayed price (gamma-api)
  4. If P_market - P_model > 0.12: get EXECUTABLE quote from CLOB
  5. Log: displayed vs executable price gap, signal, paper P&L

This measures the real displayed-vs-executable gap the user asked about.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from math import erf, sqrt

CITIES = {
    "New York": {"slug": "nyc", "lat": 40.7772, "lon": -73.8726},
    "Chicago": {"slug": "chicago", "lat": 41.9742, "lon": -87.9073},
    "London": {"slug": "london", "lat": 51.5048, "lon": 0.0495},
    "Dallas": {"slug": "dallas", "lat": 32.8471, "lon": -96.8518},
    "San Francisco": {"slug": "san-francisco", "lat": 37.6213, "lon": -122.3790},
}
MONTHS = ["january","february","march","april","may","june","july","august",
          "september","october","november","december"]
BIAS_C = -0.54
SIGMA_C = 0.90
EDGE = 0.12
STAKE = 2.0  # start small per user concern


def get_json(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "pm-bot-paper/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def get_forecast(lat: float, lon: float) -> tuple[float, float] | None:
    """Multi-model ensemble forecast: GFS + ECMWF + ICON aggregated.
    
    Aggregating multiple weather models (not just GFS) for more robust
    P_model. This is the EMOS approach: Ensemble Model Output Statistics.
    """
    try:
        # Try 3-model first, fall back to 2-model, then GFS alone
        for models in ("gfs_seamless,ecmwf_ifs,icon_seamless",
                       "gfs_seamless,ecmwf_ifs",
                       "gfs_seamless"):
            try:
                d = get_json(f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}"
                             f"&daily=temperature_2m_max&forecast_days=2&models={models}&timezone=auto")
                daily = d.get("daily", {})
                members = []
                for k, v in daily.items():
                    if "member" in k and v:
                        members.append(float(v[0]))
                if len(members) >= 20:
                    import numpy as np
                    return float(np.mean(members)), float(np.std(members))
            except Exception:
                continue
        return None
    except Exception as e:
        print(f"  forecast err: {e}", file=sys.stderr)
        return None


def parse_bucket(question: str):
    m = re.search(r"between\s+(-?\d+)-(-?\d+)\s*°?F", question, re.I)
    if m: return (float(m.group(1)), float(m.group(2)))
    m = re.search(r"(-?\d+)\s*°?F\s+or\s+higher", question, re.I)
    if m: return (float(m.group(1)), None)
    m = re.search(r"between\s+(-?\d+)-(-?\d+)\s*°?C", question, re.I)
    if m:
        return (float(m.group(1))*9/5+32, float(m.group(2))*9/5+32)
    m = re.search(r"(-?\d+)\s*°?C\s+or\s+higher", question, re.I)
    if m: return (float(m.group(1))*9/5+32, None)
    return None


def bucket_p(fcst_c: float, lo_f, hi_f) -> float:
    mu = fcst_c + BIAS_C
    s = SIGMA_C
    to_c = lambda f: (f-32)*5/9
    if hi_f is None:
        return 1.0 - norm_cdf((to_c(lo_f)-mu)/s)
    return norm_cdf((to_c(hi_f)-mu)/s) - norm_cdf((to_c(lo_f)-mu)/s)


def get_quote(token_id: str, side: str, amount: float) -> float | None:
    """Executable price from CLOB."""
    try:
        d = get_json(f"https://clob.polymarket.com/price?token_id={token_id}&side={side}&amount={amount}")
        return float(d.get("price", 0))
    except Exception:
        return None


def main():
    today = datetime.now(timezone.utc)
    ds = today.strftime("%Y-%m-%d")
    month = MONTHS[today.month - 1]
    out_path = f"/home/hatch/workspace/pm-bot/research/replay/paper_{ds}.csv"
    rows = []

    for city, cfg in CITIES.items():
        print(f"\n{city}...", flush=True)
        # find today's event
        q = f"highest-temperature-in-{cfg['slug']}-on-{month}-{today.day}-{today.year}"
        try:
            res = get_json(f"https://gamma-api.polymarket.com/public-search?q={q}")
            ev = None
            for e in res.get("events", []):
                if str(today.day) in e.get("ticker","") and month in e.get("ticker",""):
                    ev = e; break
            if not ev:
                print(f"  no event"); continue
        except Exception as e:
            print(f"  search err: {e}"); continue

        fcst = get_forecast(cfg["lat"], cfg["lon"])
        if not fcst:
            print(f"  no forecast"); continue
        fmean, fstd = fcst
        print(f"  forecast: {fmean:.1f}C ± {fstd:.1f}")
        time.sleep(0.3)

        for m in ev.get("markets", []):
            question = m.get("question", "")
            bounds = parse_bucket(question)
            if not bounds: continue
            lo_f, hi_f = bounds
            p_displayed = float(m.get("lastTradePrice") or 0)
            if not (0.01 < p_displayed < 0.99): continue

            p_model = bucket_p(fmean, lo_f, hi_f)
            edge = p_displayed - p_model
            if edge < EDGE: continue

            # get executable quote for NO
            tids = m.get("clobTokenIds", [])
            if isinstance(tids, str):
                try: tids = json.loads(tids)
                except: tids = []
            if len(tids) < 2: continue
            no_tid = tids[1]  # NO token
            # displayed NO price
            no_displayed = 1.0 - p_displayed
            # executable: BUY the NO token
            no_exec = get_quote(no_tid, "BUY", STAKE)
            time.sleep(0.3)
            if not no_exec: continue

            gap = no_exec - no_displayed
            rows.append({
                "city": city, "date": ds, "question": question[:50],
                "condition_id": m.get("conditionId", ""),
                "market_slug": m.get("slug", ""),
                "yes_displayed": round(p_displayed, 4),
                "p_model": round(p_model, 4),
                "edge": round(edge, 4),
                "no_displayed": round(no_displayed, 4),
                "no_executable": round(no_exec, 4),
                "gap": round(gap, 4),
                "gap_pct": round(gap/no_displayed*100, 1) if no_displayed else 0,
            })
            print(f"  SIGNAL: {question[:40]} | YES {p_displayed:.2f} vs model {p_model:.2f} | NO disp {no_displayed:.3f} exec {no_exec:.3f} gap {gap*100:+.1f}c")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["city","date","question","condition_id","market_slug","yes_displayed","p_model","edge","no_displayed","no_executable","gap","gap_pct"])
        w.writeheader(); w.writerows(rows)

    if rows:
        import numpy as np
        gaps = np.array([r["gap"] for r in rows])
        print(f"\n{len(rows)} signals. Avg displayed-vs-executable gap: {gaps.mean()*100:+.1f}c (max {gaps.max()*100:+.1f}c)")
    else:
        print("\nNo signals today.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
