#!/usr/bin/env python3
"""Honest S1/S4 backtest on point-in-time data (honest_data.csv).

Arms:
  S1-live          current live logic: fixed bucket geometry (imported
                   NoFadeStrategy.bucket_model_p), EMOS sigma from the
                   cross-model spread, edge in [0.12, 0.30], NO in
                   [0.05, 0.95], $20 flat, cost model A.
  S1-live-costB    same signals, cost model B (real taker fee
                   0.05*p*(1-p)/share + 2.3c executable-price gap).
  S1-buggy-geom    OLD bucket geometry (raw printed bounds): quantifies
                   how much "edge" the geometry bug fabricated.
  S1-ballesjr      fixed geometry + NO band [0.15, 0.40] + edge >= 0.25.
  S1-fixed-sigma   fixed geometry, constant sigma = 1.6 (EMOS ablation).
  market-control   no model at all: buy NO on every YES in [0.15, 0.60].
  S4-lock          censored running-max model, ERA5 hourly running max
                   by 16:00 local as proxy, T-3h prices, live cities only.

Adversarial suite: forecast-shuffle placebo, winner-shuffle placebo,
day-block bootstrap CI, parameter sensitivity grid, negative-control
cities, time split, slippage / fill haircut stress.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import urllib.request
from collections import defaultdict
from math import erf, sqrt

sys.path.insert(0, "/home/hatch/workspace/pm-bot")
from pm_bot.models.market import TemperatureBucket  # noqa: E402
from pm_bot.strategies.no_fade import NoFadeStrategy, emos_sigma  # noqa: E402
from pm_bot.strategies.s4_lock import S4LockStrategy  # noqa: E402

STAKE = 20.0
LIVE_CITIES = {"New York", "Chicago", "London", "Dallas", "San Francisco"}
COORDS = {
    "New York": (40.7772, -73.8726), "Chicago": (41.9742, -87.9073),
    "London": (51.5048, 0.0495), "Dallas": (32.8471, -96.8518),
    "San Francisco": (37.6213, -122.3790), "Tokyo": (35.5522, 139.7796),
    "Paris": (48.9694, 2.4414), "Seoul": (37.4602, 126.4407),
}
_nf = NoFadeStrategy()
_s4 = S4LockStrategy()


def _cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def buggy_bucket_p(mu: float, sigma: float, lo, hi, unit) -> float:
    """Pre-2026-10-20 geometry: integrate over raw printed bounds."""
    if unit == "F":
        lo_c = _to_c(lo) if lo is not None else None
        hi_c = _to_c(hi) if hi is not None else None
    else:
        lo_c, hi_c = lo, hi
    if hi_c is None:
        return 1.0 - _cdf((lo_c - mu) / sigma)
    if lo_c is None:
        return _cdf((hi_c - mu) / sigma)
    return _cdf((hi_c - mu) / sigma) - _cdf((lo_c - mu) / sigma)


def make_bucket(r) -> TemperatureBucket:
    lo = float(r["lo"]) if r["lo"] != "" else -999.0
    hi = float(r["hi"]) if r["hi"] != "" else 999.0
    return TemperatureBucket(
        market_id="", question=r["question"], temp_low=lo, temp_high=hi,
        temp_unit=r["unit"], yes_price=float(r["yes_24h"] or 0),
        no_price=1.0 - float(r["yes_24h"] or 0), volume=0.0,
    )


def pnl_cost_a(p_no: float, won_no: bool) -> float:
    fill = min(p_no * 1.015, 0.99)
    cost = 0.005 * fill * STAKE
    return (STAKE * (1 - fill) / fill - cost) if won_no else (-STAKE - cost)


def pnl_cost_b(p_no: float, won_no: bool) -> float:
    fill = min(p_no + 0.023, 0.99)
    shares = STAKE / fill
    fee = shares * 0.05 * fill * (1 - fill)
    return (shares - STAKE - fee) if won_no else (-STAKE - fee)


def eval_s1(rows, p_model_fn, edge_lo=0.12, edge_hi=0.30, no_lo=0.05, no_hi=0.95,
            cost=pnl_cost_a, cities=None):
    trades = []
    for r in rows:
        if cities and r["city"] not in cities:
            continue
        if r["yes_24h"] == "":
            continue
        p_mkt = float(r["yes_24h"])
        if not (0.01 < p_mkt < 0.99):
            continue
        p_no = 1.0 - p_mkt
        if not (no_lo <= p_no <= no_hi):
            continue
        p_mod = p_model_fn(r)
        edge = p_mkt - p_mod
        if not (edge_lo <= edge <= edge_hi):
            continue
        won_no = int(r["won"]) == 0
        trades.append({"city": r["city"], "date": r["date"], "edge": edge,
                       "pnl": cost(p_no, won_no), "won": won_no})
    return trades


def summarize(name, trades):
    n = len(trades)
    if n == 0:
        print(f"{name:22s} n=0")
        return {"n": 0, "pnl": 0.0}
    w = sum(t["won"] for t in trades)
    pnl = sum(t["pnl"] for t in trades)
    print(f"{name:22s} n={n:4d} win={w / n * 100:5.1f}% pnl=${pnl:+9.2f} avgEdge={statistics.mean(t['edge']):.3f}")
    return {"n": n, "win_rate": w / n, "pnl": pnl, "trades": trades}


def p_live(r):
    b = make_bucket(r)
    return _nf.bucket_model_p(float(r["fcst_mu"]), b, emos_sigma(float(r["fcst_spread"])))


def p_live_fixed_sigma(r):
    b = make_bucket(r)
    return _nf.bucket_model_p(float(r["fcst_mu"]), b, 1.6)


def p_buggy(r):
    lo = float(r["lo"]) if r["lo"] != "" else None
    hi = float(r["hi"]) if r["hi"] != "" else None
    return buggy_bucket_p(float(r["fcst_mu"]), emos_sigma(float(r["fcst_spread"])), lo, hi, r["unit"])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/hatch/workspace/pm-bot/research/replay/honest_data.csv"
    rows = list(csv.DictReader(open(path)))
    days = sorted({r["date"] for r in rows})
    print(f"dataset: {len(rows)} buckets, {len(days)} days ({days[0]}..{days[-1]}), cities={sorted({r['city'] for r in rows})}")
    priced = [r for r in rows if r["yes_24h"] != ""]
    print(f"priced buckets: {len(priced)} ({len(priced) / len(rows) * 100:.0f}%)\n")

    print("=== ARMS (live cities, $20 flat) ===")
    live_rows = [r for r in rows if r["city"] in LIVE_CITIES]
    res = {}
    res["live"] = summarize("S1-live (EMOS, costA)", eval_s1(live_rows, p_live))
    res["liveB"] = summarize("S1-live (costB real)", eval_s1(live_rows, p_live, cost=pnl_cost_b))
    res["buggy"] = summarize("S1-buggy-geometry", eval_s1(live_rows, p_buggy))
    res["ballesjr"] = summarize("S1-ballesjr band", eval_s1(live_rows, p_live, edge_lo=0.25, edge_hi=9.9, no_lo=0.15, no_hi=0.40))
    res["fixed"] = summarize("S1-fixed-sigma1.6", eval_s1(live_rows, p_live_fixed_sigma))

    def p_control(r):
        return float(r["yes_24h"])  # edge always 0 -> use direct filter below
    ctrl = []
    for r in live_rows:
        if r["yes_24h"] == "":
            continue
        p = float(r["yes_24h"])
        if 0.15 <= p <= 0.60:
            ctrl.append({"city": r["city"], "date": r["date"], "edge": 0.0,
                         "pnl": pnl_cost_a(1 - p, int(r["won"]) == 0), "won": int(r["won"]) == 0})
    res["control"] = summarize("market-control (no model)", ctrl)

    print("\n=== S1-live per city (incl. negative controls) ===")
    for city in sorted({r["city"] for r in rows}):
        summarize(f"  {city}", eval_s1(rows, p_live, cities={city}))

    print("\n=== S1-live time split ===")
    mid = days[len(days) // 2]
    summarize(f"  first half (<{mid})", eval_s1([r for r in live_rows if r["date"] < mid], p_live))
    summarize(f"  second half (>={mid})", eval_s1([r for r in live_rows if r["date"] >= mid], p_live))

    trades = res["live"].get("trades", [])
    print("\n=== ADVERSARIAL ===")
    # 1) forecast-shuffle placebo: break forecast<->day link within city
    rng = random.Random(42)
    by_city_days = defaultdict(list)
    for r in live_rows:
        by_city_days[r["city"]].append(r)
    real_pnl = res["live"]["pnl"]
    placebo = []
    base = [dict(r) for r in live_rows]
    for _ in range(200):
        shuffled = [dict(r) for r in base]
        by_city = defaultdict(list)
        for r in shuffled:
            by_city[r["city"]].append(r)
        for city, rs in by_city.items():
            pairs = [(r["fcst_mu"], r["fcst_spread"]) for r in rs]
            # shuffle among distinct days only
            day_keys = sorted({r["date"] for r in rs})
            day_pair = {}
            src = [next(r for r in rs if r["date"] == d) for d in day_keys]
            src_pairs = [(r["fcst_mu"], r["fcst_spread"]) for r in src]
            perm = src_pairs[:]
            rng.shuffle(perm)
            for d, pr in zip(day_keys, perm):
                day_pair[d] = pr
            for r in rs:
                r["fcst_mu"], r["fcst_spread"] = day_pair[r["date"]]
        t = eval_s1(shuffled, p_live)
        placebo.append(sum(x["pnl"] for x in t))
    pct = sum(1 for x in placebo if x < real_pnl) / len(placebo) * 100
    print(f"forecast-shuffle placebo: real=${real_pnl:+.2f} vs placebo mean=${statistics.mean(placebo):+.2f} "
          f"p95=${sorted(placebo)[int(0.95 * len(placebo))]:+.2f} -> real beats {pct:.0f}% of placebos")

    # 2) winner-shuffle placebo: redraw each event's winner with
    # probability proportional to market YES prices. This preserves the
    # market's information, so under the null (model adds nothing beyond
    # prices) expected P&L ~= -costs. A uniform draw would be a WRONG
    # null: it changes base win rates of traded buckets.
    events = defaultdict(list)
    for r in live_rows:
        events[(r["city"], r["date"])].append(r)
    wplacebo = []
    for _ in range(200):
        shuffled = []
        for rs in events.values():
            rs = [dict(r) for r in rs]
            if any(int(r["won"]) == 1 for r in rs):
                weights = []
                for r in rs:
                    try:
                        weights.append(max(float(r["yes_24h"]), 0.001))
                    except (TypeError, ValueError):
                        weights.append(0.001)
                pick = rng.choices(range(len(rs)), weights=weights)[0]
                for r in rs:
                    r["won"] = "0"
                rs[pick]["won"] = "1"
            shuffled.extend(rs)
        t = eval_s1(shuffled, p_live)
        wplacebo.append(sum(x["pnl"] for x in t))
    print(f"winner-shuffle placebo (price-weighted): mean=${statistics.mean(wplacebo):+.2f} "
          f"(null expectation ~ -costs if the model adds nothing beyond market prices)")

    # 3) day-block bootstrap on S1-live trades
    by_day = defaultdict(float)
    by_day_n = defaultdict(int)
    for t in trades:
        by_day[t["date"]] += t["pnl"]
        by_day_n[t["date"]] += 1
    day_pnls = list(by_day.values())
    boots = []
    for _ in range(2000):
        boots.append(sum(rng.choice(day_pnls) for _ in day_pnls))
    boots.sort()
    print(f"day bootstrap (n_days={len(day_pnls)}): P&L 95% CI [${boots[int(0.025 * len(boots))]:+.0f}, "
          f"${boots[int(0.975 * len(boots))]:+.0f}], P(>0)={sum(1 for b in boots if b > 0) / len(boots) * 100:.0f}%")

    # 4) sensitivity grid
    print("\nsensitivity (edge_min x sigma_scale x bias):")
    for elo in (0.08, 0.12, 0.16, 0.20):
        line = []
        for ss in (0.7, 1.0, 1.3):
            def p_sens(r, ss=ss):
                b = make_bucket(r)
                return _nf.bucket_model_p(float(r["fcst_mu"]), b, emos_sigma(float(r["fcst_spread"])) * ss)
            t = eval_s1(live_rows, p_sens, edge_lo=elo)
            line.append(f"s{ss}: n={len(t)} ${sum(x['pnl'] for x in t):+.0f}")
        print(f"  edge>={elo:.2f}  " + " | ".join(line))
    for bias in (-0.63, 0.63):
        def p_bias(r, bias=bias):
            b = make_bucket(r)
            return _nf.bucket_model_p(float(r["fcst_mu"]) + bias, b, emos_sigma(float(r["fcst_spread"])))
        t = eval_s1(live_rows, p_bias)
        print(f"  bias {bias:+.2f}C: n={len(t)} pnl=${sum(x['pnl'] for x in t):+.2f}")

    # 5) slippage / fill stress on S1-live signals
    sig = []
    for r in live_rows:
        if r["yes_24h"] == "":
            continue
        p_mkt = float(r["yes_24h"])
        p_no = 1.0 - p_mkt
        if not (0.05 <= p_no <= 0.95) or not (0.01 < p_mkt < 0.99):
            continue
        e = p_mkt - p_live(r)
        if 0.12 <= e <= 0.30:
            sig.append((p_no, int(r["won"]) == 0))
    for extra in (0.0, 0.015, 0.03):
        tot = 0.0
        for p_no, won_no in sig:
            fill = min((p_no + extra) * 1.015, 0.99)
            cost = 0.005 * fill * STAKE
            tot += (STAKE * (1 - fill) / fill - cost) if won_no else (-STAKE - cost)
        print(f"slippage +{extra * 100:.1f}c: pnl=${tot:+.2f} (n={len(sig)})")
    hair = []
    for _ in range(200):
        tot = 0.0
        for p_no, won_no in sig:
            if won_no and rng.random() < 0.5:
                continue  # profitable fill never happened (adverse selection)
            fill = min(p_no * 1.015, 0.99)
            cost = 0.005 * fill * STAKE
            tot += (STAKE * (1 - fill) / fill - cost) if won_no else (-STAKE - cost)
        hair.append(tot)
    print(f"halve-winning-fills stress: mean pnl=${statistics.mean(hair):+.2f}")


if __name__ == "__main__":
    main()
