#!/usr/bin/env python3
"""S1 paper trader v2: WebSocket prices + ECMWF forecast.

Upgrades from v1:
  - Forecast: ECMWF IFS (best global model) instead of GFS
  - Prices: WebSocket real-time instead of REST polling
  - Still paper: logs signals with executable quotes, no real orders
"""
from __future__ import annotations

import asyncio
import csv
import json
import re
import sys
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
# ECMWF error characteristics (typically better than GFS; conservative)
BIAS_C = -0.30
SIGMA_C = 0.75
EDGE = 0.12
STAKE = 2.0
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def get_json(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "pm-bot-paper/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def get_ecmwf(lat: float, lon: float) -> tuple[float, float] | None:
    """ECMWF IFS ensemble forecast for today."""
    try:
        d = get_json(f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}"
                     f"&daily=temperature_2m_max&forecast_days=2&models=ecmwf_ifs&timezone=auto")
        daily = d.get("daily", {})
        members = [float(v[0]) for k, v in daily.items() if "member" in k and v and v[0] is not None]
        if len(members) < 10:
            # fallback to deterministic
            det = get_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                           f"&daily=temperature_2m_max&forecast_days=1&models=ecmwf_ifs&timezone=auto")
            v = det["daily"]["temperature_2m_max"][0]
            return (float(v), 1.0) if v else None
        import numpy as np
        return float(np.mean(members)), float(np.std(members))
    except Exception as e:
        print(f"  ecmwf err: {e}", file=sys.stderr)
        return None


def parse_bucket(question: str):
    m = re.search(r"between\s+(-?\d+)-(-?\d+)\s*°?F", question, re.I)
    if m: return (float(m.group(1)), float(m.group(2)))
    m = re.search(r"(-?\d+)\s*°?F\s+or\s+higher", question, re.I)
    if m: return (float(m.group(1)), None)
    m = re.search(r"between\s+(-?\d+)-(-?\d+)\s*°?C", question, re.I)
    if m: return (float(m.group(1))*9/5+32, float(m.group(2))*9/5+32)
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


async def get_ws_prices(token_ids: list[str], timeout: float = 10.0) -> dict[str, float]:
    """Get real-time prices via WebSocket."""
    try:
        import websockets
    except ImportError:
        return {}
    prices = {}
    try:
        async with websockets.connect(WS_URL) as ws:
            # subscribe to market channel for each token
            for tid in token_ids[:20]:  # limit
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "channel": "market",
                    "token_id": tid,
                }))
            # collect for timeout seconds
            import asyncio as aio
            end = aio.get_event_loop().time() + timeout
            while aio.get_event_loop().time() < end:
                try:
                    msg = await aio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    # parse price update (format varies)
                    if isinstance(data, dict):
                        tid = data.get("token_id") or data.get("asset_id", "")
                        px = data.get("price") or data.get("best_ask", "")
                        if tid and px:
                            prices[str(tid)] = float(px)
                except aio.TimeoutError:
                    continue
                if len(prices) >= len(token_ids[:20]):
                    break
    except Exception as e:
        print(f"  ws err: {e}", file=sys.stderr)
    return prices


def get_quote(token_id: str, amount: float) -> float | None:
    try:
        d = get_json(f"https://clob.polymarket.com/price?token_id={token_id}&side=BUY&amount={amount}")
        return float(d.get("price", 0))
    except Exception:
        return None


def main():
    today = datetime.now(timezone.utc)
    ds = today.strftime("%Y-%m-%d")
    month = MONTHS[today.month - 1]
    out_path = f"/home/hatch/workspace/pm-bot/research/replay/paper_v2_{ds}.csv"
    rows = []

    # collect all token IDs for WS subscription
    all_markets = []  # (city, market, question, bounds, p_model)
    for city, cfg in CITIES.items():
        print(f"\n{city}...", flush=True)
        q = f"highest-temperature-in-{cfg['slug']}-on-{month}-{today.day}-{today.year}"
        try:
            res = get_json(f"https://gamma-api.polymarket.com/public-search?q={q}")
            ev = next((e for e in res.get("events", [])
                       if str(today.day) in e.get("ticker","") and month in e.get("ticker","")), None)
            if not ev:
                print(f"  no event"); continue
        except Exception as e:
            print(f"  err: {e}"); continue

        fcst = get_ecmwf(cfg["lat"], cfg["lon"])
        if not fcst:
            print(f"  no forecast"); continue
        fmean, fstd = fcst
        print(f"  ECMWF: {fmean:.1f}C ± {fstd:.1f}")

        for m in ev.get("markets", []):
            question = m.get("question", "")
            bounds = parse_bucket(question)
            if not bounds: continue
            tids = m.get("clobTokenIds", [])
            if isinstance(tids, str):
                try: tids = json.loads(tids)
                except: tids = []
            if len(tids) < 2: continue
            p_model = bucket_p(fmean, *bounds)
            all_markets.append((city, m, question, bounds, p_model, tids))

    print(f"\n{len(all_markets)} buckets tracked. Getting WS prices...")
    # get WS prices for YES tokens
    yes_tids = []
    tid_to_info = {}
    for city, m, question, bounds, p_model, tids in all_markets:
        yes_tid = tids[0]
        yes_tids.append(yes_tid)
        tid_to_info[yes_tid] = (city, m, question, bounds, p_model, tids)

    ws_prices = asyncio.run(get_ws_prices(yes_tids))
    print(f"WS returned {len(ws_prices)} prices")

    for yes_tid, ws_px in ws_prices.items():
        city, m, question, bounds, p_model, tids = tid_to_info[yes_tid]
        p_displayed = float(m.get("lastTradePrice") or 0)
        # prefer WS price, fallback to displayed
        p_mkt = ws_px if 0.01 < ws_px < 0.99 else p_displayed
        if not (0.01 < p_mkt < 0.99): continue
        edge = p_mkt - p_model
        if edge < EDGE: continue

        no_tid = tids[1]
        no_displayed = 1.0 - p_displayed
        no_exec = get_quote(no_tid, STAKE)
        if not no_exec: continue

        rows.append({
            "city": city, "date": ds, "question": question[:50],
            "yes_ws": round(ws_px, 4), "yes_displayed": round(p_displayed, 4),
            "p_model_ecmwf": round(p_model, 4), "edge": round(edge, 4),
            "no_displayed": round(no_displayed, 4),
            "no_executable": round(no_exec, 4),
            "gap_c": round((no_exec - no_displayed)*100, 1),
        })
        print(f"  SIGNAL {city}: YES ws {ws_px:.2f} disp {p_displayed:.2f} model {p_model:.2f} | NO exec {no_exec:.3f}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["city","date","question","yes_ws","yes_displayed","p_model_ecmwf","edge","no_displayed","no_executable","gap_c"])
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} signals. Wrote {out_path}")


if __name__ == "__main__":
    main()
