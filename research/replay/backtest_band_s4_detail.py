"""Detailed backtest of the two LIVE arms (2026-10-21, post no_fade halt).

band_control: uses the LIVE fair_yes() step calibration and live config
(band [0.15,0.60], min_edge 0.05) imported from the strategy module —
no re-implemented formulas. Compared against the raw buy-all-in-band
control, a sub-band x min_edge grid, time halves, per-city, a
price-weighted winner-shuffle placebo, and the halve-winning-fills
adverse-selection stress.

s4_lock: uses the LIVE S4LockStrategy._censored_p with config overrides
over (drift, sigma_rem, min_edge, max_no_price), on the ERA5 rm16 proxy
and yes_s4 (17:30-local last-trade) prices, same as backtest_honest.

Costs: pnl_cost_b copied verbatim from backtest_honest.py (real taker
fee formula + 2.3c executable spread), $20 flat stakes.
Dataset: honest_data.csv (Single Runs point-in-time, gamma winners).
"""
from __future__ import annotations

import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from pm_bot.models.market import TemperatureBucket
from pm_bot.strategies.band_control import BandControlConfig, fair_yes
from pm_bot.strategies.s4_lock import S4Config, S4LockStrategy

STAKE = 20.0
LIVE_CITIES = ["New York", "Chicago", "London", "Dallas", "San Francisco"]
HERE = os.path.dirname(os.path.abspath(__file__))


def pnl_cost_b(p_no: float, won_no: bool) -> float:
    fill = min(p_no + 0.023, 0.99)
    shares = STAKE / fill
    fee = shares * 0.05 * fill * (1 - fill)
    return (shares - STAKE - fee) if won_no else (-STAKE - fee)


def load_rows():
    with open(os.path.join(HERE, "honest_data.csv")) as f:
        rows = list(csv.DictReader(f))
    return rows


def make_bucket(r, price_key="yes_24h") -> TemperatureBucket:
    lo = float(r["lo"]) if r["lo"] != "" else -999.0
    hi = float(r["hi"]) if r["hi"] != "" else 999.0
    p = float(r[price_key] or 0)
    return TemperatureBucket(
        market_id="", question=r["question"], temp_low=lo, temp_high=hi,
        temp_unit=r["unit"], yes_price=p, no_price=1.0 - p, volume=0.0,
    )


def summ(name, trades):
    n = len(trades)
    if not n:
        print(f"{name:44s} n=0")
        return
    w = sum(1 for t in trades if t["won"])
    pnl = sum(t["pnl"] for t in trades)
    print(f"{name:44s} n={n:4d} win={w / n:5.1%} pnl=${pnl:+9.2f} avg=${pnl / n:+.2f}")


# ---------------- band arms ----------------

def band_trades(rows, lo, hi, min_edge, use_fair=True, cities=LIVE_CITIES):
    out = []
    for r in rows:
        if r["city"] not in cities:
            continue
        if r.get("yes_24h") in ("", None):
            continue
        p = float(r["yes_24h"])
        if not (lo <= p <= hi):
            continue
        if use_fair:
            fair = fair_yes(p)
            if fair is None:
                continue
            if p - fair < min_edge:
                continue
        won_no = int(r["won"]) == 0
        out.append({"city": r["city"], "date": r["date"], "p": p,
                    "won": won_no, "pnl": pnl_cost_b(1.0 - p, won_no)})
    return out


def band_detail(rows):
    print("=== BAND: live rule vs raw control (live cities, costB) ===")
    live = band_trades(rows, 0.15, 0.60, 0.05)
    raw = band_trades(rows, 0.15, 0.60, 0.0, use_fair=False)
    summ("band-LIVE (fair step, edge>=0.05)", live)
    summ("band-RAW (buy all in band)", raw)
    summ("band-LIVE all 8 cities", band_trades(rows, 0.15, 0.60, 0.05,
                                              cities=list({r['city'] for r in rows})))
    print()
    print("=== BAND grid (live fair, sub-band x min_edge) ===")
    for lo in (0.15, 0.20, 0.25):
        for hi in (0.45, 0.50, 0.55, 0.60):
            for me in (0.0, 0.05, 0.10):
                t = band_trades(rows, lo, hi, me)
                if len(t) >= 15:
                    summ(f"band [{lo:.2f},{hi:.2f}] edge>={me:.2f}", t)
    print()
    print("=== BAND time halves (split 2026-09-19) ===")
    for name, tr in (("LIVE", live), ("RAW", raw)):
        h1 = [t for t in tr if t["date"] <= "2026-09-19"]
        h2 = [t for t in tr if t["date"] > "2026-09-19"]
        summ(f"band-{name} first half", h1)
        summ(f"band-{name} second half", h2)
    print()
    print("=== BAND-LIVE per city ===")
    by = defaultdict(list)
    for t in live:
        by[t["city"]].append(t)
    for c in LIVE_CITIES:
        summ(f"band-LIVE {c}", by.get(c, []))
    print()
    # halve-winning-fills stress on live rule
    rng = random.Random(11)
    tot = 0.0
    for t in live:
        if t["won"] and rng.random() < 0.5:
            continue
        tot += t["pnl"]
    print(f"band-LIVE halve-winning-fills stress pnl=${tot:+.2f} (vs ${sum(t['pnl'] for t in live):+.2f})")
    # price-weighted winner-shuffle placebo within city-day
    days = defaultdict(list)
    for idx, r in enumerate(rows):
        if r.get("yes_24h") in ("", None):
            continue
        days[(r["city"], r["date"])].append(idx)
    real = sum(t["pnl"] for t in live)
    rng = random.Random(23)
    nulls = []
    for _ in range(2000):
        won_override = {}
        for key, idxs in days.items():
            ps = [max(float(rows[i]["yes_24h"]), 1e-6) for i in idxs]
            pick = rng.choices(idxs, weights=ps, k=1)[0]
            for i in idxs:
                won_override[i] = 1 if i == pick else 0
        tot = 0.0
        for idx, r in enumerate(rows):
            if r["city"] not in LIVE_CITIES or r.get("yes_24h") in ("", None):
                continue
            p = float(r["yes_24h"])
            if not (0.15 <= p <= 0.60):
                continue
            fair = fair_yes(p)
            if fair is None or p - fair < 0.05:
                continue
            tot += pnl_cost_b(1.0 - p, won_override[idx] == 0)
        nulls.append(tot)
    nulls.sort()
    frac = sum(1 for x in nulls if x >= real) / len(nulls)
    print(f"band-LIVE winner-shuffle placebo: real=${real:+.2f} null mean=${sum(nulls)/len(nulls):+.2f} "
          f"p95=${nulls[int(0.95 * len(nulls))]:+.2f} frac(null>=real)={frac:.3f}")


# ---------------- S4 arm ----------------

def s4_trades(rows, rm16, cfg: S4Config):
    strat = S4LockStrategy(cfg)
    out = []
    for r in rows:
        if r["city"] not in LIVE_CITIES or r["date"] < "2026-09-20":
            continue
        if r.get("yes_s4") in ("", None):
            continue
        rm = rm16.get((r["city"], r["date"]))
        if rm is None:
            continue
        p_mkt = float(r["yes_s4"])
        if not (0.01 < p_mkt < 0.99):
            continue
        p_no = 1.0 - p_mkt
        if not (cfg.min_no_price <= p_no <= cfg.max_no_price):
            continue
        b = make_bucket(r, "yes_s4")
        p_mod = strat._censored_p(rm, b)
        edge = p_mkt - p_mod
        if edge < cfg.min_edge:
            continue
        won_no = int(r["won"]) == 0
        out.append({"city": r["city"], "date": r["date"], "edge": edge,
                    "won": won_no, "pnl": pnl_cost_b(p_no, won_no)})
    return out


def s4_detail(rows):
    rm16 = {}
    with open(os.path.join(HERE, "runmax16.csv")) as f:
        for r in csv.DictReader(f):
            rm16[(r["city"], r["date"])] = float(r["rm16"])
    print()
    print("=== S4 grid (ERA5 rm16 proxy, yes_s4 prices, costB) ===")
    cells = []
    for drift in (0.0, 0.15, 0.30):
        for sig in (0.25, 0.35, 0.50):
            for me in (0.03, 0.08):
                for mx in (0.95, 0.98):
                    cfg = S4Config(drift_c=drift, sigma_rem_c=sig, min_edge=me, max_no_price=mx)
                    t = s4_trades(rows, rm16, cfg)
                    cells.append((sum(x["pnl"] for x in t), len(t), drift, sig, me, mx, t))
    cells.sort(reverse=True)
    for pnl, n, drift, sig, me, mx, _ in cells[:8]:
        print(f"s4 drift={drift:.2f} sig={sig:.2f} edge>={me:.2f} maxNO={mx:.2f}   n={n:4d} pnl=${pnl:+9.2f}")
    print("  ... worst cells:")
    for pnl, n, drift, sig, me, mx, _ in cells[-3:]:
        print(f"s4 drift={drift:.2f} sig={sig:.2f} edge>={me:.2f} maxNO={mx:.2f}   n={n:4d} pnl=${pnl:+9.2f}")
    live_cfg = S4Config()
    t = s4_trades(rows, rm16, live_cfg)
    print()
    summ("s4 LIVE config (drift .15 sig .35 e.03 max.98)", t)
    h1 = [x for x in t if x["date"] <= "2026-10-04"]
    h2 = [x for x in t if x["date"] > "2026-10-04"]
    summ("s4 LIVE first half (Sep20-Oct4)", h1)
    summ("s4 LIVE second half (Oct5-19)", h2)


def main():
    rows = load_rows()
    print(f"dataset: {len(rows)} buckets")
    cfg = BandControlConfig()
    print(f"band live config: [{cfg.band_lo},{cfg.band_hi}] min_edge={cfg.min_edge}")
    band_detail(rows)
    s4_detail(rows)


if __name__ == "__main__":
    main()
