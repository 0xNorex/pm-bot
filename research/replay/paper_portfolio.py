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

CITY_SLUGS = {
    "New York": "nyc", "Chicago": "chicago", "London": "london",
    "Dallas": "dallas", "San Francisco": "san-francisco",
}
MONTHS = ["january","february","march","april","may","june","july","august",
          "september","october","november","december"]


def get_json(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "pm-bot-paper/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_event_markets(city: str, date_ymd: str):
    """Same event lookup as paper_trade.py. Returns list of gamma market dicts or None."""
    slug = CITY_SLUGS.get(city)
    if not slug:
        return None
    dt = datetime.strptime(date_ymd, "%Y-%m-%d")
    month = MONTHS[dt.month - 1]
    q = f"highest-temperature-in-{slug}-on-{month}-{dt.day}-{dt.year}"
    try:
        res = get_json(f"https://gamma-api.polymarket.com/public-search?q={q}")
    except Exception as e:
        print(f"  event search err for {city}: {e}")
        return None
    for e in res.get("events", []):
        if str(dt.day) in str(e.get("ticker", "")) and month in str(e.get("ticker", "")):
            return e.get("markets", [])
    return None


def as_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return v or []


def market_yes_price(m: dict) -> float | None:
    outcomes = as_list(m.get("outcomes"))
    prices = as_list(m.get("outcomePrices"))
    if not prices:
        return None
    idx = 0
    for i, o in enumerate(outcomes):
        if str(o).strip().lower() == "yes":
            idx = i; break
    if idx >= len(prices):
        return None
    try:
        return float(prices[idx])
    except (TypeError, ValueError):
        return None


def resolve_winner(markets: list[dict]):
    """Event-level resolution via outcomePrices (settlement record).

    Returns the winning market dict, or None if unresolved/ambiguous."""
    winners = []
    for m in markets:
        yp = market_yes_price(m)
        if yp is not None and yp >= 0.99:
            winners.append(m)
    if len(winners) == 1:
        return winners[0]
    return None


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
        # dedupe: condition_id when available (survives truncated questions),
        # else fall back to (city, question)
        cond = (s.get("condition_id") or "").strip()
        if cond:
            if any(x.get("condition_id") == cond for x in p["positions"]):
                continue
        else:
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
            "condition_id": cond,
            "market_slug": (s.get("market_slug") or "").strip(),
            "no_price": no_exec, "shares": round(shares, 2),
            "cost": round(cost, 2), "booked_at": datetime.now(timezone.utc).isoformat(),
        })
        booked += 1
        print(f"  PAPER BUY: {s['city']} NO @ {no_exec:.3f} x {shares:.1f} shares (${cost:.0f})")
    save(p)
    print(f"\nBooked {booked} positions. Capital: ${p['capital']:.2f}")


def cmd_settle():
    """Check each open position's market resolution, settle P&L.

    Positions hold NO. Event-level resolution: the one bucket whose
    outcomePrices YES leg is ~1.0 won. If that is our market -> NO lost.
    Otherwise -> NO won, payout = shares * $1.
    """
    p = load()
    if not p or not p["positions"]:
        print("No open positions."); return

    # group by (city, date) so each event is fetched once
    groups: dict[tuple[str, str], list[dict]] = {}
    for pos in p["positions"]:
        groups.setdefault((pos["city"], pos["date"]), []).append(pos)

    still_open: list[dict] = []
    for (city, date), positions in groups.items():
        markets = fetch_event_markets(city, date)
        if not markets:
            print(f"  {city} {date}: event not found, {len(positions)} left open")
            still_open.extend(positions)
            continue
        winner = resolve_winner(markets)
        if winner is None:
            print(f"  {city} {date}: not yet resolved, {len(positions)} left open")
            still_open.extend(positions)
            continue
        wcond = str(winner.get("conditionId") or "")
        wq = str(winner.get("question") or "")[:60]
        for pos in positions:
            cond = pos.get("condition_id") or ""
            if not cond:
                print(f"  {city}: no condition_id stored -> manual ({pos['question'][:40]}...)")
                still_open.append(pos)
                continue
            if cond == wcond:
                result, payout = "LOST", 0.0
            else:
                result, payout = "WON", pos["shares"] * 1.0
            pnl = payout - pos["cost"]
            p["capital"] += payout
            p["closed"].append({
                "city": pos["city"], "date": pos["date"], "question": pos["question"],
                "condition_id": cond, "no_price": pos["no_price"],
                "shares": pos["shares"], "cost": pos["cost"],
                "payout": round(payout, 2), "pnl": round(pnl, 2),
                "result": result, "winning_bucket": wq,
                "settled_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"  {result}: {city} NO @ {pos['no_price']:.3f} -> "
                  f"payout ${payout:.2f}, pnl {pnl:+.2f}")

    p["positions"] = still_open
    save(p)
    print(f"Open: {len(still_open)} | Capital: ${p['capital']:.2f}")


def clob_buy_price(token_id: str, amount: float = 2.0) -> float | None:
    """Executable BUY price for a token from CLOB (same call paper_trade.py makes)."""
    try:
        d = get_json(f"https://clob.polymarket.com/price?token_id={token_id}&side=BUY&amount={amount}")
        return float(d.get("price", 0))
    except Exception:
        return None


def clean_prefix(question: str) -> str:
    """Strip appended '(YES x.xx)' style suffixes so the stored question is a
    true prefix of the real market question."""
    import re
    return re.sub(r"\s*\(YES [0-9.]+\)\s*$", "", question).strip()


def cmd_backfill():
    """One-off migration: attach conditionId to positions booked before the
    fix. Primary fingerprint: CLOB executable NO price (same quote
    paper_trade.py recorded as no_executable, amount=2.0). Fallback:
    lastTradePrice proximity to the CSV's yes_displayed."""
    import time
    p = load()
    if not p or not p["positions"]:
        print("No positions."); return
    missing = [x for x in p["positions"] if not x.get("condition_id")]
    if not missing:
        print("All positions already have condition_id."); return

    # index signal CSVs by (city, clean question, no_executable) -> yes_displayed
    sig_index: dict[tuple[str, str, float], float] = {}
    for pos in missing:
        csv_path = Path(f"/home/hatch/workspace/pm-bot/research/replay/paper_{pos['date']}.csv")
        if csv_path.exists():
            for row in csv.DictReader(csv_path.open()):
                try:
                    sig_index[(row["city"], row["question"],
                               round(float(row["no_executable"]), 2))] = float(row["yes_displayed"])
                except (KeyError, ValueError):
                    continue

    fixed = 0
    by_group: dict[tuple[str, str], list[dict]] = {}
    for pos in missing:
        by_group.setdefault((pos["city"], pos["date"]), []).append(pos)

    for (city, date), positions in by_group.items():
        markets = fetch_event_markets(city, date)
        if not markets:
            print(f"  {city} {date}: event not found, skipping")
            continue
        for pos in positions:
            prefix = clean_prefix(pos["question"])
            yes_sig = sig_index.get((pos["city"], prefix, round(pos["no_price"], 2)))
            cands = [m for m in markets
                     if str(m.get("question") or "").startswith(prefix)]
            if not cands:
                print(f"  {city}: no question-prefix match for '{prefix[:40]}...', skipped")
                continue
            # primary: CLOB executable NO price vs recorded no_executable
            scored = []
            for m in cands:
                tids = as_list(m.get("clobTokenIds"))
                if len(tids) < 2:
                    continue
                px = clob_buy_price(str(tids[1]), 2.0)
                time.sleep(0.2)
                if px is None:
                    continue
                scored.append((abs(px - pos["no_price"]), m, px))
            if scored:
                scored.sort(key=lambda t: t[0])
                best_d, best, best_px = scored[0]
                method = f"clob NO {best_px:.3f} vs {pos['no_price']:.3f}"
            else:
                # fallback: lastTradePrice vs yes_displayed
                best, best_d, method = None, 9e9, ""
                for m in cands:
                    try:
                        lp = float(m.get("lastTradePrice") or 0)
                    except (TypeError, ValueError):
                        continue
                    d = abs(lp - (yes_sig if yes_sig is not None else 0))
                    if d < best_d:
                        best, best_d, method = m, d, f"last {lp:.3f} vs yes {yes_sig}"
            if best is None or best_d > 0.10:
                print(f"  {city}: no confident match for '{prefix[:40]}...' (d={best_d:.2f}), skipped")
                continue
            pos["condition_id"] = str(best.get("conditionId") or "")
            pos["market_slug"] = str(best.get("slug") or "")
            fixed += 1
            print(f"  {city}: {str(best.get('question') or '')[:52]}... [{method}]")
    save(p)
    print(f"\nBackfilled {fixed}/{len(missing)} positions.")


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
    {"init": cmd_init, "buy": cmd_buy, "settle": cmd_settle,
     "status": cmd_status, "backfill": cmd_backfill}[cmd]()
