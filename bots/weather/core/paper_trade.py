from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

PAPER_DB_PATH = Path.home() / ".pm-bot" / "paper-trades.db"

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE,
    market_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size_usd REAL NOT NULL,
    shares REAL NOT NULL,
    kelly_fraction REAL,
    edge REAL,
    city TEXT,
    temp_label TEXT,
    reasoning TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    settled_pnl REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    settled_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_market_id ON paper_trades(market_id);
CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_city ON paper_trades(city);

CREATE TABLE IF NOT EXISTS paper_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS paper_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE,
    starting_bankroll REAL NOT NULL,
    ending_bankroll REAL NOT NULL,
    trade_count INTEGER NOT NULL DEFAULT 0,
    win_count INTEGER NOT NULL DEFAULT 0,
    loss_count INTEGER NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_V2 = """
ALTER TABLE paper_trades ADD COLUMN exit_price REAL;
ALTER TABLE paper_trades ADD COLUMN exit_reason TEXT;
ALTER TABLE paper_trades ADD COLUMN fill_probability REAL;
"""

SCHEMA_V3 = """
ALTER TABLE paper_trades ADD COLUMN fill_reason TEXT;
"""


class PaperTradeDB:
    def __init__(self, db_path: Path | None = None, initial_bankroll: float = 100.0) -> None:
        self.db_path = db_path or PAPER_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._initial_bankroll = initial_bankroll

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        self._init_bankroll()
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
            log.info("paper_db_migration_applied", version=1)
            current = 1

        if current < 2:
            for statement in SCHEMA_V2.strip().split(";"):
                stmt = statement.strip()
                if not stmt:
                    continue
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise
            self._conn.execute("INSERT INTO schema_version (version) VALUES (2)")
            self._conn.commit()
            log.info("paper_db_migration_applied", version=2)

        if current < 3:
            for statement in SCHEMA_V3.strip().split(";"):
                stmt = statement.strip()
                if not stmt:
                    continue
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise
            self._conn.execute("INSERT INTO schema_version (version) VALUES (3)")
            self._conn.commit()
            log.info("paper_db_migration_applied", version=3)

    def _init_bankroll(self) -> None:
        assert self._conn is not None
        existing = self._conn.execute("SELECT value FROM paper_state WHERE key = 'bankroll'").fetchone()
        if not existing:
            self._conn.execute(
                "INSERT INTO paper_state (key, value) VALUES (?, ?)",
                ("bankroll", str(self._initial_bankroll)),
            )
            self._conn.execute(
                "INSERT INTO paper_state (key, value) VALUES (?, ?)",
                ("daily_spent", "0.0"),
            )
            self._conn.commit()

    @property
    def bankroll(self) -> float:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM paper_state WHERE key = 'bankroll'").fetchone()
        return float(row[0]) if row else self._initial_bankroll

    @bankroll.setter
    def bankroll(self, value: float) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value, updated_at) VALUES (?, ?, ?)",
            ("bankroll", str(value), _utc_now()),
        )
        conn.commit()

    @property
    def daily_spent(self) -> float:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM paper_state WHERE key = 'daily_spent'").fetchone()
        return float(row[0]) if row else 0.0

    @daily_spent.setter
    def daily_spent(self, value: float) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value, updated_at) VALUES (?, ?, ?)",
            ("daily_spent", str(value), _utc_now()),
        )
        conn.commit()

    def get_daily_spent(self) -> float:
        return self.daily_spent

    def get_city_spent(self, city: str) -> float:
        conn = self._get_conn()
        today = _utc_today()
        row = conn.execute(
            """SELECT COALESCE(SUM(size_usd), 0) FROM paper_trades
               WHERE city = ? AND date(created_at) = ? AND status IN ('open', 'filled')""",
            (city, today),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def get_total_exposure(self) -> float:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT COALESCE(SUM(size_usd), 0) FROM paper_trades
               WHERE status IN ('open', 'filled')"""
        ).fetchone()
        return float(row[0]) if row else 0.0

    def record_trade(
        self,
        order_id: str,
        market_id: str,
        side: str,
        price: float,
        size_usd: float,
        shares: float,
        strategy: str,
        edge: float,
        city: str = "",
        temp_label: str = "",
        kelly_fraction_val: float = 0.0,
        reasoning: str = "",
        fill_probability: float | None = None,
        status: str = "open",
        fill_reason: str | None = None,
    ) -> bool:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO paper_trades
                   (order_id, market_id, strategy, side, price, size_usd, shares,
                    kelly_fraction, edge, city, temp_label, reasoning, fill_probability, status, fill_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order_id,
                    market_id,
                    strategy,
                    side,
                    price,
                    size_usd,
                    shares,
                    kelly_fraction_val,
                    edge,
                    city,
                    temp_label,
                    reasoning,
                    fill_probability,
                    status,
                    fill_reason,
                ),
            )
            conn.commit()
            if cursor.rowcount == 1 and status in {"open", "filled"}:
                self.daily_spent = self.daily_spent + size_usd
            return True
        except sqlite3.IntegrityError:
            log.warning("paper_duplicate_order", order_id=order_id)
            return False

    def check_duplicate_order(self, market_id: str, side: str) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT COUNT(*) FROM paper_trades
               WHERE market_id = ? AND side = ? AND status = 'open'""",
            (market_id, side),
        ).fetchone()
        return bool(row and row[0] > 0)

    def close_trade_at_price(
        self,
        order_id: str,
        exit_price: float,
        reason: str,
    ) -> float:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT id, side, price, size_usd, shares FROM paper_trades
               WHERE order_id = ? AND status = 'open'""",
            (order_id,),
        ).fetchone()
        if not row:
            return 0.0

        size_usd = float(row["size_usd"])
        shares = float(row["shares"])
        pnl = shares * exit_price - size_usd

        conn.execute(
            """UPDATE paper_trades
               SET status = 'closed', settled_pnl = ?, settled_at = ?, exit_price = ?, exit_reason = ?
               WHERE id = ?""",
            (pnl, _utc_now(), exit_price, reason, row["id"]),
        )
        conn.commit()
        self.bankroll = self.bankroll + pnl
        log.info("paper_trade_closed", order_id=order_id, reason=reason, exit_price=exit_price, pnl=pnl)
        return pnl

    def settle_market(self, market_id: str, winning_side: str) -> float:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, order_id, side, price, size_usd, shares, strategy, city, temp_label
               FROM paper_trades WHERE market_id = ? AND status = 'open'""",
            (market_id,),
        ).fetchall()

        total_pnl = 0.0
        for row in rows:
            side = row["side"]
            price = row["price"]
            shares = row["shares"]
            size_usd = row["size_usd"]

            if side == "YES":
                if winning_side == "YES":
                    pnl = shares * (1.0 - price) - size_usd * 0.0
                    cost = size_usd
                    payout = shares * 1.0
                    pnl = payout - cost
                else:
                    pnl = -size_usd
            else:
                if winning_side == "NO":
                    payout = shares * 1.0
                    pnl = payout - size_usd
                else:
                    pnl = -size_usd

            total_pnl += pnl

            conn.execute(
                """UPDATE paper_trades SET status = 'settled', settled_pnl = ?, settled_at = ?
                   WHERE id = ?""",
                (pnl, _utc_now(), row["id"]),
            )

            log.info(
                "paper_trade_settled",
                order_id=row["order_id"],
                strategy=row["strategy"],
                city=row["city"],
                temp_label=row["temp_label"],
                side=side,
                winning=winning_side,
                pnl=pnl,
            )

        conn.commit()

        if total_pnl != 0:
            self.bankroll = self.bankroll + total_pnl

        return total_pnl

    def settle_by_temperature(self, market_id: str, observed_temp_c: float, temp_unit: str = "C") -> float:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT temp_label FROM paper_trades
               WHERE market_id = ? AND status = 'open' AND temp_label != '' LIMIT 1""",
            (market_id,),
        ).fetchone()
        if not row:
            return 0.0

        from pm_bot.core.parser import parse_bucket

        bucket = parse_bucket(str(row["temp_label"]), market_id=market_id)
        if bucket is None:
            return 0.0

        if bucket.is_low_tail:
            hit = floor(observed_temp_c) <= floor(bucket.temp_high_c)
        elif bucket.is_high_tail:
            hit = floor(observed_temp_c) >= floor(bucket.temp_low_c)
        elif bucket.temp_unit == "F" or temp_unit.upper() == "F":
            obs_f = floor(observed_temp_c * 1.8 + 32.0)
            hit = floor(bucket.temp_low) <= obs_f <= floor(bucket.temp_high)
        else:
            hit = floor(observed_temp_c) == floor(bucket.temp_low_c)

        return self.settle_market(market_id, winning_side="YES" if hit else "NO")

    def get_open_trades(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status = 'open' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM paper_trades ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_settled_pnl(self) -> float:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COALESCE(SUM(settled_pnl), 0) FROM paper_trades WHERE status IN ('settled', 'closed')"
        ).fetchone()
        return float(row[0]) if row else 0.0

    def get_trade_stats(self) -> dict[str, Any]:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status IN ('settled', 'closed')").fetchone()[0]
        wins = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status IN ('settled', 'closed') AND settled_pnl > 0").fetchone()[0]
        total_pnl = conn.execute("SELECT COALESCE(SUM(settled_pnl), 0) FROM paper_trades WHERE status IN ('settled', 'closed')").fetchone()[0]
        open_count = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status = 'open'").fetchone()[0]

        return {
            "total_settled": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0.0,
            "total_pnl": total_pnl,
            "open_positions": open_count,
            "bankroll": self.bankroll,
            "return_pct": (self.bankroll / self._initial_bankroll - 1.0) * 100 if self._initial_bankroll > 0 else 0.0,
        }

    def reset_daily(self) -> None:
        self.daily_spent = 0.0

    def city_guard(self, days: int = 60, min_trades: int = 15, alpha: float = 0.05) -> dict[str, dict[str, Any]]:
        """Rolling per-city win-rate guard (BallesJr-style auto-ban).

        Breakeven win probability for a bet bought at price p is p itself
        (win pays 1). If P(X <= wins | n, breakeven) < alpha with
        n >= min_trades settled trades in the window, the city's realized
        win rate is significantly below breakeven -> banned for new trades.
        """
        from math import comb

        conn = self._get_conn()
        rows = conn.execute(
            """SELECT city, COUNT(*),
                      SUM(CASE WHEN settled_pnl > 0 THEN 1 ELSE 0 END),
                      AVG(price)
               FROM paper_trades
               WHERE status IN ('settled', 'closed') AND settled_pnl IS NOT NULL
                 AND city IS NOT NULL
                 AND date(settled_at) >= date('now', ?)
               GROUP BY city""",
            (f"-{int(days)} days",),
        ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for city, n, wins, avg_price in rows:
            n = int(n)
            wins = int(wins or 0)
            be = float(avg_price or 0.5)
            p_val = sum(comb(n, k) * be ** k * (1 - be) ** (n - k) for k in range(wins + 1))
            out[str(city)] = {
                "n": n, "wins": wins, "breakeven": be, "p_value": p_val,
                "banned": bool(n >= min_trades and p_val < alpha),
            }
        return out

    def get_daily_pnl(self) -> float:
        conn = self._get_conn()
        today = _utc_today()
        row = conn.execute(
            """SELECT COALESCE(SUM(settled_pnl), 0) FROM paper_trades
               WHERE date(settled_at) = ? AND status IN ('settled', 'closed')""",
            (today,),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def get_consecutive_losses(self) -> int:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT settled_pnl FROM paper_trades
               WHERE status IN ('settled', 'closed') AND settled_pnl IS NOT NULL
               ORDER BY settled_at DESC LIMIT 20"""
        ).fetchall()
        count = 0
        for r in rows:
            if r[0] < 0:
                count += 1
            else:
                break
        return count

    def get_state(self, key: str, default: str = "") -> str:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM paper_state WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else default

    def set_state(self, key: str, value: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, _utc_now()),
        )
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
