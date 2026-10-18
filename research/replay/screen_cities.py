#!/usr/bin/env python3
"""Screen new cities for S1: fetch 30d replay data, run backtest, report.

Cities: Los Angeles, Paris, Seoul, Shanghai, Dallas, Atlanta, San Francisco
"""
from __future__ import annotations

import csv
import sys
sys.path.insert(0, "/home/hatch/workspace/pm-bot/research/replay")

# reuse helpers from fetch_s1_data
import importlib.util
spec = importlib.util.spec_from_file_location("f1", "/home/hatch/workspace/pm-bot/research/replay/fetch_s1_data.py")
f1 = importlib.util.module_from_spec(spec)

# We can't easily import (it runs main on import guard, but guarded by __main__).
# Instead, exec only the function defs.
src = open("/home/hatch/workspace/pm-bot/research/replay/fetch_s1_data.py").read()
func_src = src.split("def main():")[0]
ns: dict = {}
exec(func_src, ns)

CITIES_NEW = {
    "Los Angeles": {"slug": "la", "lat": 33.9425, "lon": -118.4081},
    "Paris": {"slug": "paris", "lat": 48.9694, "lon": 2.4414},
    "Seoul": {"slug": "seoul", "lat": 37.4602, "lon": 126.4407},
    "Shanghai": {"slug": "shanghai", "lat": 31.1443, "lon": 121.8083},
    "Dallas": {"slug": "dallas", "lat": 32.8471, "lon": -96.8518},
    "Atlanta": {"slug": "atlanta", "lat": 33.6407, "lon": -84.4277},
    "San Francisco": {"slug": "san-francisco", "lat": 37.6213, "lon": -122.3790},
}

find_event = ns["find_event"]
parse_bucket = ns["parse_bucket"]
yes_price_before = ns["yes_price_before"]
prev_fcst = ns["prev_fcst"]
bucket_p = ns["bucket_p"]

import time
from datetime import datetime, timedelta, timezone
import json

def main():
    out_path = "/home/hatch/workspace/pm-bot/research/replay/s1_screen.csv"
    rows = []
    start = datetime(2026, 9, 18)
    end = datetime(2026, 10, 17)
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        for city, cfg in CITIES_NEW.items():
            print(f"{city} {ds}...", flush=True)
            try:
                ev = find_event(cfg["slug"], ds)
                if not ev: continue
                try:
                    res_ts = datetime.fromisoformat(ev["endDate"].replace("Z", "+00:00")).timestamp()
                except Exception: continue
                cutoff = res_ts - 24 * 3600
                fcst = prev_fcst(cfg["lat"], cfg["lon"], ds)
                if fcst is None: continue
                time.sleep(0.2)
                markets = ev.get("markets", [])
                winner_idx = None
                for i, m in enumerate(markets):
                    try:
                        ops = m.get("outcomePrices", [])
                        if isinstance(ops, str): ops = json.loads(ops)
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
                    y24 = yes_price_before(m.get("conditionId",""), cutoff, tids) if m.get("conditionId") else None
                    time.sleep(0.25)
                    if not y24: continue
                    pm = bucket_p(fcst, lo_f, hi_f)
                    rows.append({"city": city, "date": ds, "yes_24h": round(y24,4),
                                 "p_model": round(pm,4), "won": 1 if i==winner_idx else 0})
            except Exception as e:
                print(f"  err: {e}", flush=True)
            time.sleep(0.2)
        d += timedelta(days=1)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["city","date","yes_24h","p_model","won"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} buckets)")

    # backtest per city
    print("\n=== S1 screen (edge>0.12, 24h) ===")
    for city in CITIES_NEW:
        sel = [r for r in rows if r["city"]==city]
        n=wins=0; pnl=0.0
        nev = len({r["date"] for r in sel})
        for r in sel:
            pmkt=float(r["yes_24h"]); pmod=float(r["p_model"])
            if pmkt - pmod < 0.12: continue
            p_no=1-pmkt; fill=min(p_no*1.015,0.99); shares=10.0/fill; cost=0.005*fill*10.0
            won = int(r["won"])==0
            pnl += (shares*(1-fill)-cost) if won else (-10.0-cost)
            n+=1; wins+=won
        wr = wins/n if n else 0
        flag = "✅" if (n>=10 and pnl>0) else ("⚠️" if n>=10 else "❌")
        print(f"  {flag} {city:14s}: events={nev:3d} n={n:3d} win={wr:.1%} pnl=${pnl:+.1f}")

if __name__ == "__main__":
    main()
