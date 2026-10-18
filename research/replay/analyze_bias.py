#!/usr/bin/env python3
"""Replay analysis: model P (previous-runs forecast + EMOS) vs market price.

For each tail bucket in tail_bias_data.csv:
  - Fetch the deterministic forecast for (city, date) from Open-Meteo previous-runs
  - EMOS-calibrate: mu = a + b*fcst, sigma = sqrt(c + d*var_typ)
  - P_model(tail) = 1 - Phi((threshold_c - mu) / sigma)
  - edge = P_model - market_price

Answers: were the 0.18%-priced tails actually underpriced?
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from math import erf, sqrt

CITIES = {
    "New York": (40.7772, -73.8726),
    "Chicago": (41.9742, -87.9073),
    "Miami": (25.7953, -80.2902),
    "London": (51.5048, 0.0495),
    "Tokyo": (35.5522, 139.7796),
}

PREV_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
VAR_TYP = 2.25  # typical ensemble variance (°C²)


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "pm-bot-replay/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


def norm_sf(z: float) -> float:
    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


def load_coeffs():
    with open("/home/hatch/workspace/pm-bot/data/emos_coeffs/emos_all.json") as f:
        return json.load(f)


def prev_forecast(lat: float, lon: float, date: str) -> float | None:
    """Deterministic GFS forecast for `date` (as issued ~24h before)."""
    try:
        d = get_json(f"{PREV_RUNS}?latitude={lat}&longitude={lon}"
                     f"&daily=temperature_2m_max&models=gfs_seamless"
                     f"&past_days=35&timezone=auto")
        times = d["daily"]["time"]
        vals = d["daily"]["temperature_2m_max"]
        if date in times:
            v = vals[times.index(date)]
            return float(v) if v is not None else None
    except Exception as e:
        print(f"  prev-runs {date}: {e}", file=sys.stderr)
    return None


def main():
    coeffs = load_coeffs()
    in_path = "/home/hatch/workspace/pm-bot/research/replay/tail_bias_data.csv"
    rows = list(csv.DictReader(open(in_path)))

    # Empirical 24h deterministic GFS error (measured 2026-10-18, n=150):
    #   mean_err = -0.54C (forecast warm bias), std_err = 0.90C
    # Using this instead of EMOS ensemble scale (which was overconfident
    # when applied to deterministic forecasts).
    BIAS_C = -0.54
    SIGMA_C = 0.90

    edges, priced_rows = [], 0
    for r in rows:
        city = r["city"]
        lat, lon = CITIES[city]
        fcst = prev_forecast(lat, lon, r["date"])
        if fcst is None:
            continue
        time.sleep(0.2)
        mu = fcst + BIAS_C
        sigma = SIGMA_C
        thresh_c = (float(r["tail_threshold_f"]) - 32) * 5 / 9
        p_model = norm_sf((thresh_c - mu) / sigma)
        p_mkt = float(r["price_24h_before"])
        edges.append((p_model, p_mkt, p_model - p_mkt, r["date"], city))
        priced_rows += 1

    n = len(edges)
    avg_model = sum(e[0] for e in edges) / n
    avg_mkt = sum(e[1] for e in edges) / n
    avg_edge = sum(e[2] for e in edges) / n
    n_pos = sum(1 for e in edges if e[2] > 0.02)  # edge > 2%
    n_big = sum(1 for e in edges if e[2] > 0.05)  # edge > 5%

    print(f"\nrows with forecast: {n}")
    print(f"avg P_model(tail): {avg_model:.4f} ({avg_model*100:.2f}%)")
    print(f"avg P_market:      {avg_mkt:.4f} ({avg_mkt*100:.2f}%)")
    print(f"avg edge:          {avg_edge:+.4f} ({avg_edge*100:+.2f}%)")
    print(f"edge > 2%: {n_pos}/{n}   edge > 5%: {n_big}/{n}")
    print(f"\nratio P_model / P_market: {avg_model/max(avg_mkt,1e-6):.1f}x")

    # show biggest edges
    edges.sort(key=lambda e: -e[2])
    print("\ntop 5 edges (model vs market):")
    for p_model, p_mkt, e, date, city in edges[:5]:
        print(f"  {city} {date}: model={p_model:.3f} mkt={p_mkt:.4f} edge={e:+.3f}")


if __name__ == "__main__":
    main()
