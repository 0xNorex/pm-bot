#!/usr/bin/env python3
"""S1 backtest: buy NO on overpriced buckets.

Signal: P_market(YES) - P_model(YES) > edge AND NO_price in [0.15, 0.40]
Action: buy NO at p_no = 1 - P_market(YES)
Payout: win 1.0 if bucket loses (prob 1-q), lose stake if bucket wins.

Costs: taker fee 50bps + 1.5% slippage on entry.
Sizing: flat $10 per signal (Kelly later).
"""
from __future__ import annotations

import csv
import sys

FEE_BPS = 50
SLIPPAGE = 0.015
STAKE = 10.0


def run(edge: float, lead: str, no_lo: float = 0.15, no_hi: float = 0.40):
    rows = list(csv.DictReader(open("/home/hatch/workspace/pm-bot/research/replay/s1_data.csv")))
    price_col = "yes_24h" if lead == "24h" else "yes_6h"
    n = wins = 0
    pnl = 0.0
    for r in rows:
        if not r[price_col] or not r["p_model"]:
            continue
        p_mkt = float(r[price_col])
        p_mod = float(r["p_model"])
        p_no = 1.0 - p_mkt
        if not (no_lo <= p_no <= no_hi):
            continue
        if p_mkt - p_mod < edge:
            continue
        # buy NO
        fill = min(p_no * (1 + SLIPPAGE), 0.99)
        cost = FEE_BPS / 10000 * fill * STAKE
        won = int(r["won"]) == 0  # NO wins iff bucket loses
        trade_pnl = (STAKE * (1 - fill) / fill - cost) if won else (-STAKE - cost)
        # NOTE: stake convention — buying NO at price `fill` with $STAKE:
        # shares = STAKE / fill; win -> shares*(1-fill); lose -> -STAKE
        shares = STAKE / fill
        trade_pnl = (shares * (1 - fill) - cost) if won else (-STAKE - cost)
        pnl += trade_pnl
        n += 1
        wins += won
    return n, (wins / n if n else 0), pnl


def main():
    print(f"{'edge':>6} {'lead':>5} {'band':>11} | {'n':>4} {'win%':>6} {'pnl$':>8}")
    for edge in (0.03, 0.05, 0.08, 0.12):
        for lead in ("24h", "6h"):
            for band in ((0.15, 0.40), (0.20, 0.35)):
                n, wr, pnl = run(edge, lead, *band)
                print(f"{edge:>6.2f} {lead:>5} {str(band):>11} | {n:>4} {wr:>6.1%} {pnl:>+8.1f}")


if __name__ == "__main__":
    main()
