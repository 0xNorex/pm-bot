"""Intraday price archive (added 2026-10-21, S4-rebuild M1).

The honest dataset only has three price points per bucket per day
(yes_24h / yes_3h / yes_s4), which is why the S4 lock-in corridor could
never be calibrated: by the time a bucket is provably dead, the only
recorded price already shows NO > 0.98. The daemon already sees every
bucket's live prices each cycle — this module appends them, plus the
per-city running day-high, to an append-only CSV so future research
has real intraday paths:

    ~/.pm-bot/intraday_prices.csv
    ts_utc, city, event_date, local_time, market_id, lo, hi, unit,
    yes_price, no_price, volume, running_max_c

lo/hi are the bucket's printed native bounds; empty = tail side
(same convention as research/replay/honest_data.csv). running_max_c
is the settlement-station METAR running max (empty before the
observation fetcher's local-hour gate). One row per bucket per cycle;
gaps appear whenever the daemon is down (VM rotations) — consumers
must treat coverage as sampled, not continuous.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARCHIVE_PATH = Path.home() / ".pm-bot" / "intraday_prices.csv"

_FIELDS = [
    "ts_utc", "city", "event_date", "local_time", "market_id",
    "lo", "hi", "unit", "yes_price", "no_price", "volume",
    "running_max_c",
]

_TAIL = 999.0


def archive_cycle(events: list, day_highs: dict[str, Any], path: Path = ARCHIVE_PATH) -> int:
    """Append one row per bucket for this daemon cycle. Returns rows written."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for ev in events:
        dh = day_highs.get(ev.city)
        rm = dh.observed_c if dh is not None else ""
        lt = dh.local_time.isoformat(timespec="seconds") if dh is not None else ""
        for b in ev.buckets:
            rows.append({
                "ts_utc": ts,
                "city": ev.city,
                "event_date": ev.date,
                "local_time": lt,
                "market_id": b.market_id,
                "lo": "" if b.temp_low <= -_TAIL else b.temp_low,
                "hi": "" if b.temp_high >= _TAIL else b.temp_high,
                "unit": b.temp_unit,
                "yes_price": b.yes_price,
                "no_price": b.no_price,
                "volume": b.volume,
                "running_max_c": rm,
            })
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    return len(rows)
