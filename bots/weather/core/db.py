from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

DEFAULT_DB_PATH = Path.home() / ".pm-bot" / "pm-bot.db"

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE,
    market_id TEXT,
    condition_id TEXT,
    strategy TEXT,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    amount_usd REAL NOT NULL,
    kelly_fraction REAL,
    fill_status TEXT NOT NULL DEFAULT 'open',
    edge REAL,
    city TEXT,
    temp_label TEXT,
    reasoning TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    filled_at TEXT,
    cancelled_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id);
CREATE INDEX IF NOT EXISTS idx_trades_market_id ON trades(market_id);
CREATE INDEX IF NOT EXISTS idx_trades_fill_status ON trades(fill_status);
CREATE INDEX IF NOT EXISTS idx_trades_city ON trades(city);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT UNIQUE,
    market_id TEXT,
    city TEXT,
    strategy TEXT,
    side TEXT NOT NULL,
    total_shares REAL NOT NULL DEFAULT 0,
    avg_price REAL NOT NULL DEFAULT 0,
    unrealized_pnl REAL NOT NULL DEFAULT 0,
    realized_pnl REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_positions_city ON positions(city);
CREATE INDEX IF NOT EXISTS idx_positions_condition_id ON positions(condition_id);

CREATE TABLE IF NOT EXISTS daily_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE,
    total_spent REAL NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0,
    trade_count INTEGER NOT NULL DEFAULT 0,
    win_count INTEGER NOT NULL DEFAULT 0,
    loss_count INTEGER NOT NULL DEFAULT 0,
    bankroll_start REAL NOT NULL DEFAULT 0,
    bankroll_end REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS daemon_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class TradeDB:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        return self._conn

    def _migrate(self) -> None:
        assert self._conn is not None
        try:
            version = self._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = version[0] if version and version[0] else 0
        except sqlite3.OperationalError:
            current = 0

        if current < 1:
            self._conn.executescript(SCHEMA_V1)
            self._conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            self._conn.commit()
            log.info("migration_applied", version=1)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def record_trade(
        self,
        order_id: str,
        market_id: str,
        side: str,
        price: float,
        amount_usd: float,
        strategy: str,
        edge: float,
        city: str = "",
        temp_label: str = "",
        kelly_fraction_val: float = 0.0,
        reasoning: str = "",
        condition_id: str = "",
    ) -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO trades
                   (order_id, market_id, condition_id, strategy, side, price,
                    amount_usd, kelly_fraction, edge, city, temp_label, reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order_id,
                    market_id,
                    condition_id,
                    strategy,
                    side,
                    price,
                    amount_usd,
                    kelly_fraction_val,
                    edge,
                    city,
                    temp_label,
                    reasoning,
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            log.warning("duplicate_order", order_id=order_id)
            return False

    def update_fill_status(self, order_id: str, status: str) -> None:
        conn = self._get_conn()
        now = _utc_now()
        if status == "filled":
            conn.execute(
                "UPDATE trades SET fill_status = ?, filled_at = ? WHERE order_id = ?",
                (status, now, order_id),
            )
        elif status == "cancelled":
            conn.execute(
                "UPDATE trades SET fill_status = ?, cancelled_at = ? WHERE order_id = ?",
                (status, now, order_id),
            )
        else:
            conn.execute(
                "UPDATE trades SET fill_status = ? WHERE order_id = ?",
                (status, order_id),
            )
        conn.commit()

    def get_daily_spent(self) -> float:
        conn = self._get_conn()
        today = _utc_today()
        row = conn.execute(
            """SELECT COALESCE(SUM(amount_usd), 0) FROM trades
               WHERE date(created_at) = ? AND fill_status IN ('open', 'partial', 'filled')""",
            (today,),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def get_city_spent(self, city: str) -> float:
        conn = self._get_conn()
        today = _utc_today()
        row = conn.execute(
            """SELECT COALESCE(SUM(amount_usd), 0) FROM trades
               WHERE city = ? AND date(created_at) = ? AND fill_status IN ('open', 'partial', 'filled')""",
            (city, today),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def get_total_exposure(self) -> float:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT COALESCE(SUM(amount_usd), 0) FROM trades
               WHERE fill_status IN ('open', 'partial', 'filled')"""
        ).fetchone()
        return float(row[0]) if row else 0.0

    def get_open_trades(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM trades WHERE fill_status = 'open' ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_daily_state(self, date: str | None = None) -> dict[str, Any]:
        conn = self._get_conn()
        if date is None:
            date = _utc_today()
        row = conn.execute("SELECT * FROM daily_state WHERE date = ?", (date,)).fetchone()
        if row:
            return dict(row)
        return {
            "date": date,
            "total_spent": 0.0,
            "total_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "bankroll_start": 0.0,
            "bankroll_end": 0.0,
        }

    def update_daily_state(self, date: str | None = None, **kwargs: Any) -> None:
        conn = self._get_conn()
        if date is None:
            date = _utc_today()
        existing = self.get_daily_state(date)
        if existing and existing.get("id"):
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [_utc_now(), date]
            conn.execute(f"UPDATE daily_state SET {sets}, updated_at = ? WHERE date = ?", vals)
        else:
            cols = ["date"] + list(kwargs.keys())
            placeholders = ", ".join(["?"] * len(cols))
            vals = [date] + list(kwargs.values())
            conn.execute(f"INSERT INTO daily_state ({', '.join(cols)}) VALUES ({placeholders})", vals)
        conn.commit()

    def get_state(self, key: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM daemon_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO daemon_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, _utc_now()),
        )
        conn.commit()

    def get_state_json(self, key: str) -> dict | None:
        raw = self.get_state(key)
        if raw is None:
            return None
        try:
            result: dict = json.loads(raw)
            return result
        except (json.JSONDecodeError, TypeError):
            return None

    def set_state_json(self, key: str, value: dict) -> None:
        self.set_state(key, json.dumps(value))

    def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def check_duplicate_order(self, market_id: str, side: str, strategy: str | None = None) -> bool:
        # Per-strategy dedup (2026-10-20), same semantics as PaperTradeDB.
        conn = self._get_conn()
        if strategy is not None:
            row = conn.execute(
                """SELECT COUNT(*) FROM trades
                   WHERE market_id = ? AND side = ? AND fill_status = 'open' AND strategy = ?""",
                (market_id, side, strategy),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT COUNT(*) FROM trades
                   WHERE market_id = ? AND side = ? AND fill_status = 'open'""",
                (market_id, side),
            ).fetchone()
        return bool(row and row[0] > 0)

    def get_consecutive_losses(self) -> int:
        """Count consecutive losing days (conservative proxy for trade-level consecutive losses)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT date, total_pnl, trade_count FROM daily_state
               ORDER BY date DESC LIMIT 30"""
        ).fetchall()
        count = 0
        for row in rows:
            pnl = row["total_pnl"]
            tc = row["trade_count"] if "trade_count" in row.keys() else 0
            if tc == 0:
                continue  # skip days with no trades
            if pnl < 0:
                count += 1
            else:
                break
        return count

    def get_daily_pnl(self) -> float:
        conn = self._get_conn()
        today = _utc_today()
        row = conn.execute("SELECT COALESCE(total_pnl, 0) FROM daily_state WHERE date = ?", (today,)).fetchone()
        return float(row[0]) if row else 0.0

    def reconcile_open_orders(self, api_order_ids: set[str]) -> None:
        conn = self._get_conn()
        db_rows = conn.execute("SELECT order_id FROM trades WHERE fill_status = 'open'").fetchall()
        db_ids = {row[0] for row in db_rows}

        filled_or_cancelled = db_ids - api_order_ids
        for oid in filled_or_cancelled:
            self.update_fill_status(oid, "filled")
            log.info("reconcile_order_filled", order_id=oid)

        orphaned = api_order_ids - db_ids
        for oid in orphaned:
            log.warning("orphaned_order_found", order_id=oid)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
