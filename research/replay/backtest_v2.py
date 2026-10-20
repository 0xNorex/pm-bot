#!/usr/bin/env python3
"""S1 falsification rerun (2026-10-20): old vs new params on identical replay data.

Joins the saved replay CSV (prices/won from s1_data.csv) with the raw
previous-runs forecast per (city, date) — the old CSV only stored p_model
under the OLD error params (bias -0.54, sigma 0.90), so the forecast must
be re-fetched to recompute p_model under the NEW live params
(bias 0.0, sigma 1.60, edge cap 0.30).

Cells: {old,new} params x {backtest NO range 0.15-0.40, live NO range 0.05-0.95}.
Costs: 50bps fee + 1.5% slippage, flat $20 (live stake).

CAVEAT (must print with results): previous-runs forecasts are contaminated
(near-actual), so every cell is an UPPER BOUND. This is a falsification
test only — failing here kills the strategy; passing proves nothing live.
"""
from __future__ import annotations

import csv
import json
import statistics
import time
import urllib.request
from math import erf, sqrt

BASE = "/home/hatch/workspace/pm-bot/research/replay"
COORDS = {  # settlement-station coords, same as live
    "New York": (40.7772, -73.8726),
    "Chicago": (41.9742, -87.9073),
    "London": (51.5048, 0.0495),
    "Dallas": (32.8471, -96.8518),
    "San Francisco": (37.6213, -122.3790),
    "Miami": (25.7953, -80.2902),
    "Tokyo": (35.5522, 139.7796),
}

FEE_BPS, SLIPPAGE, STAKE = 50, 0.015, 20.0


def get_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "pm-bot-s1/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


_FC_CACHE: dict[tuple[float, float], dict[str, float]] = {}


def prev_fcst(lat: float, lon: float, date_str: str) -> float | None:
    key = (lat, lon)
    if key not in _FC_CACHE:
        try:
            d = get_json(
                f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max&models=gfs_seamless&past_days=35&timezone=auto"
            )
            times = d["daily"]["time"]
            vals = d["daily"]["temperature_2m_max"]
            _FC_CACHE[key] = {t: float(v) for t, v in zip(times, vals) if v is not None}
        except Exception:
            _FC_CACHE[key] = {}
    return _FC_CACHE[key].get(date_str)


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def bucket_p(fcst_c: float, lo_f, hi_f, bias: float, sigma: float) -> float:
    mu = fcst_c + bias
    def to_c(f): return (f - 32) * 5 / 9
    if hi_f is None:
        return 1.0 - norm_cdf((to_c(lo_f) - mu) / sigma)
    if lo_f is None:
        return norm_cdf((to_c(hi_f) - mu) / sigma)
    return norm_cdf((to_c(hi_f) - mu) / sigma) - norm_cdf((to_c(lo_f) - mu) / sigma)


def run(rows, bias, sigma, edge_lo, edge_hi, no_lo, no_hi, price_col="yes_24h"):
    n = wins = 0
    pnl = 0.0
    edges = []
    for r in rows:
        if r.get("fcst_c") in (None, "") or not r[price_col]:
            continue
        p_mkt = float(r[price_col])
        p_no = 1.0 - p_mkt
        if not (no_lo <= p_no <= no_hi):
            continue
        lo = float(r["lo_f"]) if r["lo_f"] not in ("", "None") else None
        hi = float(r["hi_f"]) if r["hi_f"] not in ("", "None") else None
        p_mod = bucket_p(float(r["fcst_c"]), lo, hi, bias, sigma)
        edge = p_mkt - p_mod
        if not (edge_lo <= edge <= edge_hi):
            continue
        fill = min(p_no * (1 + SLIPPAGE), 0.99)
        cost = FEE_BPS / 10000 * fill * STAKE
        won_no = int(r["won"]) == 0
        pnl += (STAKE * (1 - fill) / fill - cost) if won_no else (-STAKE - cost)
        n += 1
        wins += 1 if won_no else 0
        edges.append(edge)
    return n, wins, pnl, (statistics.mean(edges) if edges else 0.0)


def main() -> None:
    rows = list(csv.DictReader(open(f"{BASE}/s1_data.csv")))
    keys = sorted({(r["city"], r["date"]) for r in rows})
    print(f"{len(rows)} buckets, {len(keys)} city-days; fetching forecasts...")
    fc: dict[tuple[str, str], float] = {}
    miss = 0
    for city, date in keys:
        lat, lon = COORDS[city]
        v = prev_fcst(lat, lon, date)
        if v is None:
            miss += 1
        else:
            fc[(city, date)] = v
        time.sleep(0.15)
    print(f"forecasts: {len(fc)}/{len(keys)} fetched, {miss} missing")
    for r in rows:
        v = fc.get((r["city"], r["date"]))
        r["fcst_c"] = f"{v:.2f}" if v is not None else ""
    with open(f"{BASE}/s1_data_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== 24h window, $20 stake, fee 50bps + slip 1.5% ===")
    print(f"{'cell':38s} {'n':>4} {'win%':>6} {'PnL':>9} {'avgEdge':>8}")
    cells = [
        ("OLD s0.90 b-0.54, edge>=.12 | BT filter", -0.54, 0.90, 0.12, 9.9, 0.15, 0.40),
        ("OLD s0.90 b-0.54, edge>=.12 | LIVE filt", -0.54, 0.90, 0.12, 9.9, 0.05, 0.95),
        ("NEW s1.60 b0, edge .12-.30 | BT filter", 0.0, 1.60, 0.12, 0.30, 0.15, 0.40),
        ("NEW s1.60 b0, edge .12-.30 | LIVE filt", 0.0, 1.60, 0.12, 0.30, 0.05, 0.95),
        ("NEW params, NO cap (edge>=.12) | LIVE", 0.0, 1.60, 0.12, 9.9, 0.05, 0.95),
    ]
    for label, b, s, elo, ehi, nlo, nhi in cells:
        n, w, pnl, ae = run(rows, b, s, elo, ehi, nlo, nhi)
        wr = f"{w / n:.1%}" if n else "  -  "
        print(f"{label:38s} {n:>4} {wr:>6} {pnl:>+9.2f} {ae:>8.3f}")
    print("\nCAVEAT: previous-runs forecasts are contaminated (near-actual);")
    print("all cells are UPPER BOUNDS. Falsification only, not live evidence.")


if __name__ == "__main__":
    main()
