#!/usr/bin/env python3
"""Multi-model backtest: GFS vs GFS+ECMWF+ICON on 10 days.

Re-fetches market data + multi-model forecasts, compares S1 P&L.
"""
from __future__ import annotations
import csv, json, re, sys, time, urllib.request
from datetime import datetime, timedelta, timezone
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
# Error params per model (empirical)
GFS_BIAS, GFS_SIGMA = -0.54, 0.90
MULTI_BIAS, MULTI_SIGMA = -0.40, 0.80  # multi-model: less bias, tighter

def get(u, timeout=20):
    req = urllib.request.Request(u, headers={"User-Agent": "x"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def norm_cdf(z): return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def parse_bucket(q):
    m = re.search(r"between\s+(-?\d+)-(-?\d+)\s*°?F", q, re.I)
    if m: return (float(m.group(1)), float(m.group(2)))
    m = re.search(r"(-?\d+)\s*°?F\s+or\s+higher", q, re.I)
    if m: return (float(m.group(1)), None)
    return None

def bucket_p(fcst_c, bias, sigma, lo_f, hi_f):
    mu = fcst_c + bias; s = sigma
    to_c = lambda f: (f-32)*5/9
    if hi_f is None: return 1.0 - norm_cdf((to_c(lo_f)-mu)/s)
    return norm_cdf((to_c(hi_f)-mu)/s) - norm_cdf((to_c(lo_f)-mu)/s)

def find_event(slug, ds):
    dt = datetime.strptime(ds, "%Y-%m-%d")
    q = f"highest-temperature-in-{slug}-on-{MONTHS[dt.month-1]}-{dt.day}-{dt.year}"
    try:
        res = get(f"https://gamma-api.polymarket.com/public-search?q={q}")
        for e in res.get("events", []):
            if str(dt.day) in e.get("ticker",""): return e
    except: pass
    return None

def yes_price_before(condition_id, cutoff_ts, tids):
    try:
        trades = get(f"https://data-api.polymarket.com/trades?market={condition_id}&limit=50")
        # tids[0] is YES token
        yes_tid = tids[0] if tids else None
        for t in trades:
            ts = t.get("timestamp", 0)
            if ts and ts < cutoff_ts:
                # price is for the traded side; assume YES if matches
                return float(t.get("price", 0))
    except: pass
    return None

def get_multi_fcst(lat, lon, date):
    """Returns (gfs_mean, multi_mean) for date."""
    try:
        d = get(f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max&models=gfs_seamless,ecmwf_ifs,icon_seamless&past_days=35&timezone=auto")
        times = d['daily']['time']
        if date not in times: return (None, None)
        i = times.index(date)
        g = d['daily'].get('temperature_2m_max_gfs_seamless', [None]*100)[i]
        e = d['daily'].get('temperature_2m_max_ecmwf_ifs', [None]*100)[i]
        ic = d['daily'].get('temperature_2m_max_icon_seamless', [None]*100)[i]
        vals = [v for v in (g,e,ic) if v is not None]
        multi = sum(vals)/len(vals) if vals else None
        return (g, multi)
    except: return (None, None)

def main():
    start = datetime(2026, 10, 8); end = datetime(2026, 10, 17)
    rows = []
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        for city, cfg in CITIES.items():
            print(f"{city} {ds}...", flush=True)
            try:
                ev = find_event(cfg["slug"], ds)
                if not ev: continue
                try: res_ts = datetime.fromisoformat(ev["endDate"].replace("Z","+00:00")).timestamp()
                except: continue
                cutoff = res_ts - 24*3600
                gfs_f, multi_f = get_multi_fcst(cfg["lat"], cfg["lon"], ds)
                if gfs_f is None: continue
                time.sleep(0.2)
                markets = ev.get("markets", [])
                winner_idx = None
                for i, m in enumerate(markets):
                    try:
                        ops = m.get("outcomePrices", [])
                        if isinstance(ops, str): ops = json.loads(ops)
                        if ops and float(ops[0]) >= 0.99: winner_idx = i; break
                    except: pass
                for i, m in enumerate(markets):
                    bounds = parse_bucket(m.get("question",""))
                    if not bounds: continue
                    lo_f, hi_f = bounds
                    tids = m.get("clobTokenIds", [])
                    if isinstance(tids, str):
                        try: tids = json.loads(tids)
                        except: tids = []
                    y24 = yes_price_before(m.get("conditionId",""), cutoff, tids) if m.get("conditionId") else None
                    time.sleep(0.25)
                    if not y24: continue
                    p_gfs = bucket_p(gfs_f, GFS_BIAS, GFS_SIGMA, lo_f, hi_f)
                    p_multi = bucket_p(multi_f, MULTI_BIAS, MULTI_SIGMA, lo_f, hi_f) if multi_f else None
                    rows.append({"city":city,"date":ds,"yes_24h":round(y24,4),
                                 "p_gfs":round(p_gfs,4),"p_multi":round(p_multi,4) if p_multi else None,
                                 "won":1 if i==winner_idx else 0})
            except Exception as ex: print(f"  err {ex}", flush=True)
            time.sleep(0.2)
        d += timedelta(days=1)

    # Backtest both
    print(f"\n{len(rows)} buckets. Backtesting...", flush=True)
    for name, key, slip in [("GFS-only", "p_gfs", 0.015), ("Multi-model", "p_multi", 0.015)]:
        n=wins=0; pnl=0.0
        for r in rows:
            if not r['yes_24h'] or not r[key]: continue
            pmkt=float(r['yes_24h']); pmod=float(r[key])
            if pmkt - pmod < 0.12: continue
            p_no=1-pmkt; fill=min(p_no*(1+slip),0.99); shares=10.0/fill; cost=0.005*fill*10.0
            won = int(r['won'])==0
            pnl += (shares*(1-fill)-cost) if won else (-10.0-cost)
            n+=1; wins+=won
        print(f"  {name:12s}: n={n:3d} win={wins/max(n,1):.1%} pnl=${pnl:+.1f}")

    # Save
    with open("/home/hatch/workspace/pm-bot/research/replay/multi_compare.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["city","date","yes_24h","p_gfs","p_multi","won"])
        w.writeheader(); w.writerows(rows)
    print("Wrote multi_compare.csv")

if __name__ == "__main__": main()
