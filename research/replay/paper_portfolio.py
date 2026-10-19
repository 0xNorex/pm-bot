#!/usr/bin/env python3
"""S1 paper portfolio: simulated $1000 fund trading real markets.

Reads today's signals from paper_YYYY-MM-DD.csv, books paper positions
at the executable quote price, tracks until resolution.

Usage:
  paper_portfolio.py init          # create portfolio with $1000
  paper_portfolio.py buy           # book today's signals as positions
  paper_portfolio.py settle        # check resolved markets, settle P&L
  paper_portfolio.py status        # show portfolio
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PORTFOLIO = Path("/home/hatch/workspace/pm-bot/research/replay/paper_portfolio.json")
STARTING_CAPITAL = 1000.0
STAKE_PER_TRADE = 20.0


def get_json(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "pm-bot-paper/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load():
    if PORTFOLIO.exists():
        return json.loads(PORTFOLIO.read_text())
    return None


def save(p):
    PORTFOLIO.write_text(json.dumps(p, indent=2))


def cmd_init():
    if PORTFOLIO.exists():
        print("Portfolio already exists. Use 'status'.")
        return
    p = {
        "capital": STARTING_CAPITAL,
        "starting": STARTING_CAPITAL,
        "stake": STAKE_PER_TRADE,
        "positions": [],  # open
        "closed": [],     # settled
        "created": datetime.now(timezone.utc).isoformat(),
    }
    save(p)
    print(f"Paper portfolio created: ${STARTING_CAPITAL:.0f}, ${STAKE_PER_TRADE:.0f}/trade")


def cmd_buy():
    p = load()
    if not p:
        print("Run 'init' first."); return
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sig_path = Path(f"/home/hatch/workspace/pm-bot/research/replay/paper_{today}.csv")
    if not sig_path.exists():
        print(f"No signals file for {today}. Run paper_trade.py first.")
        return
    signals = list(csv.DictReader(sig_path.open()))
    if not signals:
        print("No signals today.")
        return
    booked = 0
    for s in signals:
        # check if already have position for this (city+question)
        key = (s["city"], s["question"])
        if any((x["city"], x["question"]) == key for x in p["positions"]):
            continue
        no_exec = float(s["no_executable"])
        shares = STAKE_PER_TRADE / no_exec
        cost = STAKE_PER_TRADE
        if p["capital"] < cost:
            print(f"  Insufficient capital for {s['city']}")
            continue
        p["capital"] -= cost
        p["positions"].append({
            "city": s["city"], "date": s["date"], "question": s["question"],
            "no_price": no_exec, "shares": round(shares, 2),
            "cost": round(cost, 2), "booked_at": datetime.now(timezone.utc).isoformat(),
        })
        booked += 1
        print(f"  PAPER BUY: {s['city']} NO @ {no_exec:.3f} x {shares:.1f} shares (${cost:.0f})")
    save(p)
    print(f"\nBooked {booked} positions. Capital: ${p['capital']:.2f}")


def cmd_settle():
    """Check each open position's market resolution, settle P&L."""
    p = load()
    if not p or not p["positions"]:
        print("No open positions."); return
    still_open = []
    for pos in p["positions"]:
        # Find the market by question to check resolution
        # We need the conditionId - not stored. Use question to search.
        # Simpler: check if the event date has passed and assume we can
        # fetch the outcome from gamma-api.
        # For now, mark as needing manual check if older than 1 day.
        pos_date = datetime.fromisoformat(pos["date"])
        age_days = (datetime.now(timezone.utc) - pos_date).days
        if age_days < 1:
            still_open.append(pos)
            continue
        # Try to find resolution via search
        print(f"  Checking {pos['city']} {pos['date']}...")
        # This is a simplification - in production, store conditionId at buy time
        still_open.append(pos)  # keep for now, manual settle
    p["positions"] = still_open
    save(p)
    print(f"{len(still_open)} positions still open (need conditionId for auto-settle)")


def cmd_status():
    p = load()
    if not p:
        print("No portfolio. Run 'init'."); return
    total_invested = sum(x["cost"] for x in p["positions"])
    closed_pnl = sum(x["pnl"] for x in p["closed"])
    print(f"Capital: ${p['capital']:.2f}")
    print(f"Open positions: {len(p['positions'])} (invested ${total_invested:.2f})")
    print(f"Closed P&L: ${closed_pnl:+.2f}")
    print(f"Total equity: ${p['capital'] + total_invested + closed_pnl:.2f}")
    for pos in p["positions"]:
        print(f"  {pos['city']}: NO @ {pos['no_price']:.3f} x {pos['shares']:.1f} (${pos['cost']:.0f})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"init": cmd_init, "buy": cmd_buy, "settle": cmd_settle, "status": cmd_status}[cmd]()
