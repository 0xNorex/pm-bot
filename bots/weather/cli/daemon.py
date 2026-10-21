from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import httpx
import structlog
from rich.console import Console
from rich.table import Table

from pm_bot.backtest.costs import FillModel
from pm_bot.core.clob import ClobTrader
from pm_bot.core.config_loader import (
    get_sizing,
    load_config,
    get_notifications,
)
from pm_bot.core.db import TradeDB
from pm_bot.core.kelly import compute_kelly_for_recommendation
from pm_bot.core.paper_trade import PaperTradeDB
from pm_bot.core.polymarket import fetch_weather_events
from pm_bot.core.risk import RiskManager, RiskCheckResult
from pm_bot.core.weather import fetch_forecast
from pm_bot.core.observation import fetch_day_high, fetch_observation, filter_recommendations
from pm_bot.core.price_archive import archive_cycle
from pm_bot.core.staged_entry import apply_staged_entry_for_event
from pm_bot.models.config import DEFAULT_CITIES, STRATEGY_DEFAULTS, resolve_city_alias
from pm_bot.models.market import Recommendation, ForecastResult
from pm_bot.strategies.base import ALL_STRATEGIES
from pm_bot.cli.notifications import notify, send_discord, send_telegram, format_daemon_message

console = Console()
log = structlog.get_logger()

PID_FILE = Path.home() / ".pm-bot" / "daemon.pid"
HEARTBEAT_FILE = Path.home() / ".pm-bot" / "heartbeat"
CONFIG_RELOAD_FLAG = Path.home() / ".pm-bot" / ".reload_config"


class TradingDaemon:
    def __init__(
        self,
        config: dict,
        dry_run: bool = False,
        strategy_names: list[str] | None = None,
        cities: list[str] | None = None,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.db = TradeDB()
        # Reuse single httpx client to avoid memory leak (SSL context + connection pool)
        self._http_client = httpx.AsyncClient(timeout=30.0)
        sizing = get_sizing(config)
        risk_cfg = config.get("risk", {})
        daemon_cfg = config.get("daemon", {})

        if strategy_names:
            self.strategies = {k: v for k, v in ALL_STRATEGIES.items() if k in strategy_names}
        else:
            self.strategies = ALL_STRATEGIES

        configured_cities = config.get("daemon", {}).get("cities") or config.get("live", {}).get("cities")
        if isinstance(configured_cities, str):
            configured_cities = [c.strip() for c in configured_cities.split(",")]
        selected_cities = cities if cities is not None else configured_cities
        if selected_cities is None:
            if dry_run:
                raise ValueError("Dry-run daemon requires explicit --cities or daemon.cities in config")
            selected_cities = DEFAULT_CITIES
            log.warning("daemon_default_cities_used", cities=selected_cities)
        self.cities = [resolve_city_alias(c.strip()) for c in selected_cities if c.strip()]
        self.city_set = set(self.cities)

        self.trader = ClobTrader(config)
        self.paper: PaperTradeDB | None = None
        self.fill_model = FillModel()

        if dry_run:
            paper_db_path = os.environ.get("PM_BOT_PAPER_DB", daemon_cfg.get("paper_db_path"))
            self.paper = PaperTradeDB(
                Path(paper_db_path).expanduser() if paper_db_path else None,
                initial_bankroll=float(os.environ.get("PM_BOT_BANKROLL", sizing.get("bankroll", 100.0))),
            )
            self.bankroll = self.paper.bankroll
        else:
            self.bankroll = float(os.environ.get("PM_BOT_BANKROLL", sizing.get("bankroll", 500.0)))

        self.kelly_fraction_val = float(os.environ.get("PM_BOT_KELLY", sizing.get("kelly_fraction", 0.25)))
        self.stop_loss = float(os.environ.get("PM_BOT_STOP_LOSS", sizing.get("stop_loss", 0.0)) or 0.0)
        if self.stop_loss > 0 and not dry_run:
            raise ValueError(
                "Live daemon stop-loss is not implemented. Use --dry-run or omit --stop-loss for live mode."
            )
        self.base_max_single = float(os.environ.get("PM_BOT_MAX_SINGLE", sizing.get("max_single", 50.0)))
        self.base_max_daily = float(os.environ.get("PM_BOT_MAX_DAILY", sizing.get("max_daily", 200.0)))
        self.base_max_per_city = float(os.environ.get("PM_BOT_MAX_PER_CITY", sizing.get("max_per_city", 100.0)))
        self.max_total_pct = float(os.environ.get("PM_BOT_MAX_TOTAL_PCT", sizing.get("max_total_pct", 0.30)))
        self.max_single = min(self.base_max_single, self.bankroll * 0.15)
        self.max_daily = min(self.base_max_daily, self.bankroll * 0.50)
        self.max_per_city = min(self.base_max_per_city, self.bankroll * 0.25)
        self.scan_interval = int(daemon_cfg.get("scan_interval", 300))
        self.heartbeat_path = Path(
            os.environ.get("PM_BOT_HEARTBEAT", daemon_cfg.get("heartbeat_path", str(HEARTBEAT_FILE)))
        ).expanduser()

        from pm_bot.core.risk import RiskDB

        risk_db: RiskDB
        if dry_run and self.paper is not None:
            risk_db = self.paper
        else:
            risk_db = self.db
        self.risk_manager = RiskManager(
            db=risk_db,
            bankroll=self.bankroll,
            circuit_breaker_l1=float(risk_cfg.get("circuit_breaker_l1", 0.05)),
            circuit_breaker_l2=float(risk_cfg.get("circuit_breaker_l2", 0.10)),
            circuit_breaker_l3=float(risk_cfg.get("circuit_breaker_l3", 0.15)),
            no_new_before_resolution_h=int(risk_cfg.get("no_new_position_before_resolution_h", 6)),
            consecutive_loss_pause_count=int(risk_cfg.get("consecutive_loss_pause_count", 5)),
            consecutive_loss_pause_minutes=int(risk_cfg.get("consecutive_loss_pause_minutes", 60)),
            max_spread=float(risk_cfg.get("max_spread", 0.10)),
            max_per_city=self.max_per_city,
            max_total_pct=self.max_total_pct,
            max_daily=self.max_daily,
        )

        self.shutdown_event = asyncio.Event()
        self.start_time = time.time()
        self.cycle_count = 0
        self.trades_this_cycle = 0

        # Load EMOS calibrators
        self.emos_calibrators: dict[str, 'EMOSCalibrator'] = {}
        self._load_emos_calibrators()


    def _load_emos_calibrators(self) -> None:
        """Load trained EMOS calibrators from disk."""
        from pathlib import Path
        from pm_bot.core.emos import EMOSCalibrator

        emos_dir = Path("data/emos")
        if not emos_dir.exists():
            return

        for path in emos_dir.glob("emos_*.json"):
            try:
                calibrator = EMOSCalibrator.load(path)
                if calibrator.city:
                    self.emos_calibrators[calibrator.city] = calibrator
                    log.info("loaded_emos", city=calibrator.city)
            except Exception as e:
                log.warning("emos_load_failed", path=str(path), error=str(e))

        if self.emos_calibrators:
            log.info("emos_calibrators_loaded", n=len(self.emos_calibrators))



    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, self._handle_sigterm)
        loop.add_signal_handler(signal.SIGINT, self._handle_sigterm)
        try:
            loop.add_signal_handler(signal.SIGUSR1, self._handle_sigusr1)
        except (OSError, AttributeError):
            pass

        self._write_pid()
        await self._send_notification("🟢 PM-Bot daemon started", "daemon_start")

        if not self.trader.is_configured():
            if not self.dry_run:
                log.error("trader_not_configured")
                await self._send_notification("🔴 PM-Bot daemon: trading credentials not configured", "daemon_error")
                return
            log.warning("trader_not_configured_dry_run", msg="Running in dry-run mode without credentials")

        if not self.dry_run:
            self.trader.start_heartbeat()

        try:
            self._recover_state()
            while not self.shutdown_event.is_set():
                cycle_start = time.time()
                try:
                    await self._trade_cycle()
                except Exception as e:
                    log.error("cycle_failed", error=str(e))
                import gc; gc.collect()
                self.cycle_count += 1
                self._write_heartbeat()

                self._check_daily_reset()

                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.scan_interval - elapsed)
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=sleep_time)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self._graceful_shutdown()

    async def _trade_cycle(self) -> None:
        self.trades_this_cycle = 0
        client = self._http_client

        events = await fetch_weather_events(client)
        events = [e for e in events if e.city in self.city_set]

        if not events:
            log.debug("no_events_found")
            return

        self._sync_dynamic_risk_limits()
        await self._apply_paper_stop_loss(events)
        self._sync_dynamic_risk_limits()

        forecasts: dict[str, ForecastResult] = {}
        obs_map: dict[str, Any] = {}
        for ev in events:
            fc = await fetch_forecast(client, ev.city, ev.date)
            if fc:
                forecasts[ev.city] = fc

        for city in {ev.city for ev in events}:
            obs = await fetch_observation(client, city)
            if obs:
                obs_map[city] = obs
                if obs.is_past_cutoff:
                    log.info("observation_locked", city=city, observed_c=obs.observed_c)

        # S4: running day-high per city (None before 15:00 local)
        day_highs: dict[str, Any] = {}
        for city in {ev.city for ev in events}:
            dh = await fetch_day_high(client, city)
            if dh:
                day_highs[city] = dh

        # S4-rebuild M1 (2026-10-21): archive every bucket's live prices
        # + running day-high each cycle (pre-guard, raw fetch) so the
        # early-trigger recalibration has real intraday paths.
        if self.dry_run:
            try:
                archive_cycle(events, day_highs)
            except Exception as e:
                log.warning("price_archive_failed", error=str(e))

        # City guard: skip cities whose rolling 60d win rate is
        # significantly below breakeven (binomial test, paper DB).
        banned = self._banned_cities()
        if banned:
            events = [e for e in events if e.city not in banned]

        all_recs: list[Recommendation] = []
        for ev in events:
            for strat_name, strat in self.strategies.items():
                kwargs: dict[str, Any] = {}
                for k, v in STRATEGY_DEFAULTS.get(strat_name, {}).items():
                    kwargs[k] = v
                if ev.city in forecasts:
                    kwargs["forecast"] = forecasts[ev.city]
                if ev.city in day_highs:
                    kwargs["day_high"] = day_highs[ev.city]
                if ev.city in self.emos_calibrators:
                    kwargs["emos_calibrator"] = self.emos_calibrators[ev.city]

                recs = strat.run(ev, **kwargs)

                if ev.city in obs_map:
                    recs = filter_recommendations(recs, obs_map[ev.city])

                # Apply staged entry scaling based on time to resolution
                recs = apply_staged_entry_for_event(recs, ev.date)

                for rec in recs:
                    if rec.edge < 0.05:
                        continue
                    dup_db = self.paper if self.dry_run else self.db
                    if dup_db is not None and dup_db.check_duplicate_order(rec.bucket.market_id, rec.direction, rec.strategy):
                        continue

                    risk_result = self.risk_manager.full_check(
                        city=rec.city,
                        amount_usd=self.max_single,
                        yes_price=rec.bucket.yes_price,
                        no_price=rec.bucket.no_price,
                        hours_to_resolution=_estimate_hours_to_resolution(ev.date),
                    )
                    if not risk_result.allowed:
                        log.info("risk_blocked", reason=risk_result.reason, city=rec.city)
                        # PRD 3E/3F: Send circuit breaker / consecutive loss notification
                        if risk_result.circuit_breaker_level > 0:
                            await self.send_circuit_breaker_alert(risk_result)
                        elif "Consecutive" in risk_result.reason:
                            await self._send_notification(
                                f"⚠️ {risk_result.reason}\nBankroll: ${self.bankroll:.2f}",
                                "consecutive_loss",
                            )
                        continue

                    effective_kelly = self.kelly_fraction_val * risk_result.kelly_adjustment
                    if self.dry_run and self.paper is not None:
                        daily_spent = self.paper.daily_spent
                        city_spent = self.paper.get_city_spent(rec.city)
                        total_exposure = self.paper.get_total_exposure()
                        sizing_bankroll = self.paper.bankroll
                    else:
                        daily_spent = self.db.get_daily_spent()
                        city_spent = self.db.get_city_spent(rec.city)
                        total_exposure = self.db.get_total_exposure()
                        sizing_bankroll = self.bankroll

                    sized = compute_kelly_for_recommendation(
                        rec,
                        bankroll=sizing_bankroll,
                        kelly_multiplier=effective_kelly,
                        max_single=self.max_single,
                        max_daily=self.max_daily,
                        daily_spent=daily_spent,
                        max_per_city=self.max_per_city,
                        city_spent=city_spent,
                        max_total_pct=self.max_total_pct,
                        total_exposure=total_exposure,
                    )
                    if sized is not None:
                        all_recs.append(sized)

        all_recs.sort(key=lambda r: r.edge, reverse=True)

        for rec in all_recs:
            if self.trades_this_cycle >= 10:
                break
            await self._execute_trade(rec)

        await self._auto_settle()

    def _sync_dynamic_risk_limits(self) -> None:
        if self.dry_run and self.paper is not None:
            self.bankroll = self.paper.bankroll
        self.max_single = min(self.base_max_single, self.bankroll * 0.15)
        self.max_daily = min(self.base_max_daily, self.bankroll * 0.50)
        self.max_per_city = min(self.base_max_per_city, self.bankroll * 0.25)
        self.risk_manager.bankroll = self.bankroll
        self.risk_manager.max_daily = self.max_daily
        self.risk_manager.max_per_city = self.max_per_city

    async def _apply_paper_stop_loss(self, events: list[Any]) -> None:
        if not self.dry_run or self.paper is None or self.stop_loss <= 0:
            return
        open_trades = self.paper.get_open_trades()
        if not open_trades:
            return

        price_by_market: dict[str, tuple[float, float]] = {}
        for ev in events:
            for bucket in ev.buckets:
                price_by_market[bucket.market_id] = (bucket.yes_price, bucket.no_price)

        for trade in open_trades:
            market_id = str(trade.get("market_id", ""))
            prices = price_by_market.get(market_id)
            if not prices:
                continue
            yes_price, no_price = prices
            side = str(trade.get("side", "YES"))
            entry = float(trade.get("price", 0.0))
            current_price = yes_price if side == "YES" else no_price
            if entry <= 0:
                continue
            drawdown = (entry - current_price) / entry
            if drawdown >= self.stop_loss:
                self.paper.close_trade_at_price(
                    order_id=str(trade.get("order_id", "")),
                    exit_price=current_price,
                    reason=f"stop_loss_{self.stop_loss:.0%}",
                )
                self.bankroll = self.paper.bankroll
                log.info(
                    "paper_stop_loss_triggered",
                    market_id=market_id,
                    side=side,
                    entry=entry,
                    current=current_price,
                    drawdown=drawdown,
                )



    async def _auto_settle(self) -> None:
        if self.dry_run:
            await self._paper_settle()
            return
        if not self.trader.is_configured():
            return
        try:
            positions = self.trader.get_redeemable_positions()
            if not positions:
                return
            condition_ids = list({str(p["conditionId"]) for p in positions if p.get("conditionId")})
            if not condition_ids:
                return
            log.info("auto_settle_start", redeemable=len(positions))
            result = self.trader.settle_resolved(condition_ids=condition_ids)
            redeemed = result.get("redeemed", 0)
            if redeemed > 0:
                total_redeemed = sum(float(p.get("size", 0)) for p in positions[:redeemed])
                self.bankroll += total_redeemed
                log.info("auto_settle_complete", redeemed=redeemed, bankroll=self.bankroll)
                await self._send_notification(
                    f"💰 Auto-settled {redeemed} position(s), bankroll: ${self.bankroll:.2f}",
                    "auto_settle",
                )
        except Exception as e:
            log.warning("auto_settle_failed", error=str(e))

    async def _paper_settle(self) -> None:
        if not self.paper:
            return
        try:
            open_trades = self.paper.get_open_trades()
            if not open_trades:
                return

            client = self._http_client
            settled_events = await fetch_weather_events(client, include_closed=True)
            settled_map: dict[str, Any] = {}
            for ev in settled_events:
                for b in ev.buckets:
                    if b.market_id:
                        settled_map[b.market_id] = ev

            total_settled = 0
            total_pnl = 0.0
            for trade in open_trades:
                mid = trade["market_id"]
                if mid in settled_map:
                    ev = settled_map[mid]
                    for b in ev.buckets:
                        if b.market_id == mid:
                            if b.yes_price > 0.95:
                                pnl = self.paper.settle_market(mid, winning_side="YES")
                                total_pnl += pnl
                                total_settled += 1
                            elif b.no_price > 0.95:
                                pnl = self.paper.settle_market(mid, winning_side="NO")
                                total_pnl += pnl
                                total_settled += 1
                            break

            if total_settled > 0:
                self.bankroll = self.paper.bankroll
                log.info(
                    "paper_settled",
                    count=total_settled,
                    pnl=total_pnl,
                    bankroll=self.bankroll,
                )
        except Exception as e:
            log.warning("paper_settle_failed", error=str(e))

    async def _execute_trade(self, rec: Recommendation) -> None:
        bucket = rec.bucket
        price = rec.price
        size_usd = rec.size_usd

        if size_usd < 1.0:
            return

        size_shares = size_usd / price if price > 0 else 1

        if self.dry_run and self.paper is not None:
            fill_probability = self.fill_model.fill_probability(price)
            fill_key = f"{rec.event.event_id}:{bucket.market_id}:{rec.direction}:{self.cycle_count}"
            order_seq = self.trades_this_cycle
            self.trades_this_cycle += 1
            order_prefix = f"DRY-{self.cycle_count}-{int(time.time() * 1000)}-{order_seq}-{uuid.uuid4().hex[:8]}"
            if not self._deterministic_fill(fill_key, fill_probability):
                recorded = self.paper.record_trade(
                    order_id=f"{order_prefix}-UNFILLED",
                    market_id=bucket.market_id,
                    side=rec.direction,
                    price=price,
                    size_usd=size_usd,
                    shares=0.0,
                    strategy=rec.strategy,
                    edge=rec.edge,
                    city=rec.city,
                    temp_label=rec.temp_label,
                    kelly_fraction_val=rec.kelly_fraction,
                    reasoning=rec.reasoning,
                    fill_probability=fill_probability,
                    status="unfilled",
                    fill_reason="deterministic_fill_model",
                )
                if not recorded:
                    log.warning("paper_trade_record_duplicate", order_id=f"{order_prefix}-UNFILLED")
                log.info(
                    "paper_trade_not_filled",
                    strategy=rec.strategy,
                    city=rec.city,
                    direction=rec.direction,
                    price=price,
                    fill_probability=fill_probability,
                )
                return
            shares = size_usd / price if price > 0 else 1
            recorded = self.paper.record_trade(
                order_id=order_prefix,
                market_id=bucket.market_id,
                side=rec.direction,
                price=price,
                size_usd=size_usd,
                shares=shares,
                strategy=rec.strategy,
                edge=rec.edge,
                city=rec.city,
                temp_label=rec.temp_label,
                kelly_fraction_val=rec.kelly_fraction,
                reasoning=rec.reasoning,
                fill_probability=fill_probability,
            )
            if not recorded:
                log.warning("paper_trade_record_duplicate", order_id=order_prefix)
                return
            log.info(
                "dry_run_trade",
                strategy=rec.strategy,
                city=rec.city,
                direction=rec.direction,
                temp_label=rec.temp_label,
                price=price,
                size_usd=size_usd,
                edge=rec.edge,
                fill_probability=fill_probability,
                bankroll=self.paper.bankroll if self.paper else self.bankroll,
            )
            return

        if rec.direction == "YES":
            result = self.trader.place_limit_buy(
                token_id=bucket.market_id,
                price=price,
                size=size_shares,
                neg_risk=True,
            )
        else:
            result = self.trader.place_limit_sell(
                token_id=bucket.market_id,
                price=price,
                size=size_shares,
                neg_risk=True,
            )

        if result:
            order_id = str(result.get("orderID", result.get("order_id", "")))
            self.db.record_trade(
                order_id=order_id,
                market_id=bucket.market_id,
                side=rec.direction,
                price=price,
                amount_usd=size_usd,
                strategy=rec.strategy,
                edge=rec.edge,
                city=rec.city,
                temp_label=rec.temp_label,
                kelly_fraction_val=rec.kelly_fraction,
                reasoning=rec.reasoning,
            )
            self.trades_this_cycle += 1
            log.info(
                "auto_trade",
                strategy=rec.strategy,
                city=rec.city,
                direction=rec.direction,
                temp_label=rec.temp_label,
                price=price,
                size_usd=size_usd,
                edge=rec.edge,
                order_id=order_id[:16],
            )
            await notify(
                self.config,
                "created",
                rec.strategy,
                rec.direction,
                rec.city,
                rec.temp_label,
                price,
                rec.edge,
                order_id,
            )
        else:
            log.warning("auto_trade_failed", city=rec.city, direction=rec.direction)

    @staticmethod
    def _deterministic_fill(key: str, probability: float) -> bool:
        import hashlib

        bounded = max(0.0, min(1.0, probability))
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big") / float(2**64 - 1)
        return value <= bounded

    def _recover_state(self) -> None:
        # PRD 3D: Check for crash recovery from previous run
        shutdown_state = self.db.get_state_json("shutdown_state")
        had_graceful_shutdown = self.db.get_state("graceful_shutdown") == "true"

        try:
            open_orders = self.trader.get_open_orders()
            api_ids = {str(o.get("id", o.get("orderID", ""))) for o in open_orders}
            self.db.reconcile_open_orders(api_ids)
            log.info("state_recovered", open_orders=len(open_orders))
        except Exception as e:
            log.warning("state_recovery_failed", error=str(e))

        # PRD 3F: Notify if crash recovery (no graceful shutdown flag)
        if not had_graceful_shutdown and shutdown_state:
            log.warning("crash_recovery_detected")
            asyncio.create_task(
                self._send_notification(
                    format_daemon_message("crash_recovery", "Reconciling with Polymarket API"),
                    "crash_recovery",
                )
            )

        # Reset graceful shutdown flag so next crash is detected
        self.db.set_state("graceful_shutdown", "false")

    async def _graceful_shutdown(self) -> None:
        log.info("graceful_shutdown_start")

        try:
            if not self.dry_run:
                self.trader.cancel_all_orders()
                log.info("all_orders_cancelled")
        except Exception as e:
            log.error("cancel_all_failed", error=str(e))

        self._persist_state()

        # Mark graceful shutdown for crash recovery detection
        self.db.set_state("graceful_shutdown", "true")

        pending = self.db.get_open_trades()
        if pending:
            log.info("waiting_pending_fills", count=len(pending))
            try:
                await asyncio.wait_for(self._poll_fills(), timeout=30)
            except asyncio.TimeoutError:
                log.warning("pending_fill_timeout")

        if not self.dry_run:
            self.trader.stop_heartbeat()

        self._update_daily_state()

        await self._send_notification("🔴 PM-Bot daemon stopped", "daemon_stop")

        await self._http_client.aclose()
        self._remove_pid()
        self.db.close()
        log.info("graceful_shutdown_complete")

    async def _poll_fills(self) -> None:
        open_trades = self.db.get_open_trades()
        for t in open_trades:
            order_id = t.get("order_id", "")
            if not order_id:
                continue
            status = self.trader.get_order_status(order_id)
            if status:
                fill_status = status.get("status", "open")
                if fill_status in ("filled", "matched"):
                    self.db.update_fill_status(order_id, "filled")
                elif fill_status in ("cancelled",):
                    self.db.update_fill_status(order_id, "cancelled")

    def _persist_state(self) -> None:
        daily_spent = self.db.get_daily_spent()
        self.db.set_state_json(
            "shutdown_state",
            {
                "daily_spent": daily_spent,
                "bankroll": self.bankroll,
                "last_cycle": time.time(),
            },
        )

    def _update_daily_state(self) -> None:
        daily_spent = self.db.get_daily_spent()
        daily_pnl = self.db.get_daily_pnl()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trades = self.db.get_recent_trades(limit=100)
        today_trades = [t for t in trades if t.get("created_at", "").startswith(today)]
        wins = sum(1 for t in today_trades if t.get("edge", 0) > 0)
        losses = len(today_trades) - wins
        self.db.update_daily_state(
            date=today,
            total_spent=daily_spent,
            total_pnl=daily_pnl,
            trade_count=len(today_trades),
            win_count=wins,
            loss_count=losses,
            bankroll_end=self.bankroll + daily_pnl,
        )

    def _banned_cities(self) -> set[str]:
        """Cities banned by the rolling win-rate guard (cached per UTC day)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache = getattr(self, "_guard_cache", None)
        if cache and cache[0] == today:
            return cache[1]
        banned: set[str] = set()
        if self.dry_run and self.paper is not None:
            try:
                stats = self.paper.city_guard()
                banned = {c for c, s in stats.items() if s["banned"]}
                for c in banned:
                    s = stats[c]
                    log.warning(
                        "city_guard_ban", city=c, n=s["n"], wins=s["wins"],
                        breakeven=round(s["breakeven"], 3), p_value=round(s["p_value"], 4),
                    )
            except Exception as e:
                log.error("city_guard_failed", error=str(e))
        self._guard_cache = (today, banned)
        return banned

    def _check_daily_reset(self) -> None:
        state = self.db.get_state_json("daily_reset")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state and state.get("date") == today:
            return

        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        self._update_daily_state()

        # Actually reset the paper daily spend — this was missing, so
        # daily_spent accumulated forever and wedged the daily limit
        # (found 2026-10-20 stuck at $260 from the previous day).
        if self.paper is not None:
            self.paper.reset_daily()

        daily_state = self.db.get_daily_state(yesterday)
        pnl = daily_state.get("total_pnl", 0)
        trades = daily_state.get("trade_count", 0)
        wins = daily_state.get("win_count", 0)

        msg = (
            f"📊 <b>Daily Summary — {yesterday}</b>\n"
            f"  P&L: ${pnl:.2f}\n"
            f"  Trades: {trades} (W:{wins} L:{trades - wins})\n"
            f"  Bankroll: ${self.bankroll:.2f}\n"
        )
        asyncio.create_task(self._send_notification(msg, "daily_summary"))

        self.db.set_state_json("daily_reset", {"date": today})

        # Backfill forecast_log actuals for elapsed target dates (the
        # settle-time backfill promised by no_fade never existed, so
        # actual_c stayed empty — found in the 2026-10-21 loss audit).
        try:
            from pm_bot.strategies.no_fade import backfill_forecast_actuals

            asyncio.create_task(backfill_forecast_actuals(self._http_client))
        except Exception:
            pass

    def _write_heartbeat(self) -> None:
        try:
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            self.heartbeat_path.write_text(
                json.dumps(
                    {
                        "ts": time.time(),
                        "status": "running",
                        "cycle": self.cycle_count,
                        "trades_this_cycle": self.trades_this_cycle,
                        "daily_spent": self.paper.daily_spent
                        if self.dry_run and self.paper
                        else self.db.get_daily_spent(),
                        "bankroll": self.bankroll,
                        "pid": os.getpid(),
                        "uptime": time.time() - self.start_time,
                        "dry_run": self.dry_run,
                    }
                )
            )
        except Exception as e:
            log.warning("heartbeat_write_failed", error=str(e))

    def _write_pid(self) -> None:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    def _handle_sigterm(self) -> None:
        log.info("sigterm_received")
        self.shutdown_event.set()

    def _handle_sigusr1(self) -> None:
        log.info("sigusr1_reload_config")
        try:
            self.config = load_config()
            log.info("config_reloaded")
        except Exception as e:
            log.error("config_reload_failed", error=str(e))

    async def _send_notification(self, msg: str, event_type: str) -> None:
        notifications = get_notifications(self.config)
        discord = notifications.get("discord", {})
        telegram = notifications.get("telegram", {})
        await send_discord(discord.get("webhook_url", ""), msg)
        await send_telegram(telegram.get("bot_token", ""), str(telegram.get("chat_id", "")), msg)

    async def send_circuit_breaker_alert(self, result: RiskCheckResult) -> None:
        level_emoji = {1: "🟡", 2: "🟠", 3: "🔴"}.get(result.circuit_breaker_level, "⚪")
        msg = (
            f"{level_emoji} [L{result.circuit_breaker_level} CIRCUIT BREAKER]\n"
            f"{result.reason}\n"
            f"Bankroll: ${self.bankroll:.2f} | Kelly adj: {result.kelly_adjustment:.0%}"
        )
        await self._send_notification(msg, "circuit_breaker")


async def _fetch_forecast_at(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    city: str,
    date: str = "",
    model: str = "gfs_seamless",
) -> ForecastResult | None:
    from pm_bot.core.weather import OPEN_METEO_BASE, ENSEMBLE_BASE, _MEMBER_KEYS

    params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "forecast_days": 3,
        "timezone": "auto",
    }
    try:
        params_model = {**params, "models": model}
        resp = await client.get(f"{OPEN_METEO_BASE}/forecast", params=params_model)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        log.error("forecast_at_error", lat=lat, lon=lon, error=str(e))
        return None

    daily = data.get("daily", {})
    temps = daily.get("temperature_2m_max", [])
    main_temp = float(temps[0]) if temps and isinstance(temps[0], (int, float)) else 0.0

    members: list[float] = []
    try:
        params_ens = {**params, "models": model}
        resp = await client.get(ENSEMBLE_BASE, params=params_ens)
        resp.raise_for_status()
        ens_data = resp.json()
        ens_daily = ens_data.get("daily", {})
        for mk in _MEMBER_KEYS:
            member_data = ens_daily.get(mk, [])
            if member_data:
                v = member_data[0]
                if isinstance(v, (int, float)):
                    members.append(float(v))
    except httpx.HTTPError:
        pass

    return ForecastResult(
        city=city,
        date=date,
        model=model,
        temp_high_c=main_temp,
        members=members,
    )


def _is_daemon_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return False


def _estimate_hours_to_resolution(date_str: str) -> float | None:
    """Estimate hours until market resolution (usually midnight ET on the date)."""
    if not date_str:
        return None
    try:
        # Weather markets typically resolve at midnight US Eastern time
        resolution = (
            datetime.strptime(date_str, "%Y-%m-%d")
            .replace(
                hour=5,
                minute=0,
                second=0,  # UTC 05:00 ≈ midnight ET
            )
            .replace(tzinfo=timezone.utc)
        )
        now = datetime.now(timezone.utc)
        delta = (resolution - now).total_seconds() / 3600.0
        return max(0.0, delta)
    except (ValueError, TypeError):
        return None


async def daemon_start(
    debug: bool = False,
    dry_run: bool = False,
    strategy_names: list[str] | None = None,
    cities: list[str] | None = None,
) -> None:
    _setup_logging(debug)

    if _is_daemon_running():
        pid = PID_FILE.read_text().strip()
        console.print(f"[red]Daemon already running (PID {pid})[/red]")
        return

    config = load_config()
    daemon = TradingDaemon(config, dry_run=dry_run, strategy_names=strategy_names, cities=cities)

    if not daemon.trader.is_configured() and not dry_run:
        console.print("[red]Trading credentials not configured. Set POLY_PK and [clob] in config.toml[/red]")
        return

    if dry_run:
        console.print("[bold yellow]Starting PM-Bot daemon in DRY-RUN mode (no orders will be placed)[/bold yellow]")
        if strategy_names:
            console.print(f"[dim]Strategies: {', '.join(strategy_names)}[/dim]")
        console.print(f"[dim]Cities: {', '.join(daemon.cities)}[/dim]")
    else:
        console.print("[bold green]Starting PM-Bot daemon...[/bold green]")
    await daemon.run()


async def daemon_stop(debug: bool = False) -> None:
    _setup_logging(debug)

    if not _is_daemon_running():
        console.print("[yellow]Daemon is not running[/yellow]")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Sent SIGTERM to daemon (PID {pid})[/green]")
    except ProcessLookupError:
        console.print("[yellow]Daemon process not found (stale PID file)[/yellow]")
        PID_FILE.unlink(missing_ok=True)
    except PermissionError:
        console.print("[red]Permission denied to send signal[/red]")


async def daemon_status(debug: bool = False) -> None:
    _setup_logging(debug)

    if not _is_daemon_running():
        console.print("[yellow]Daemon is not running[/yellow]")
        return

    pid = PID_FILE.read_text().strip()

    hb_data: dict[str, Any] = {}
    if HEARTBEAT_FILE.exists():
        try:
            hb_data = json.loads(HEARTBEAT_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    db = TradeDB()
    try:
        daily_spent = db.get_daily_spent()
        daily_pnl = db.get_daily_pnl()
        open_trades = len(db.get_open_trades())
        total_exposure = db.get_total_exposure()
    except Exception:
        daily_spent = 0.0
        daily_pnl = 0.0
        open_trades = 0
        total_exposure = 0.0
    finally:
        db.close()

    table = Table(title="PM-Bot Daemon Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    uptime = hb_data.get("uptime", 0)
    hours, remainder = divmod(int(uptime), 3600)
    minutes, seconds = divmod(remainder, 60)

    table.add_row("PID", pid)
    table.add_row("Uptime", f"{hours}h {minutes}m {seconds}s")
    table.add_row("Cycles", str(hb_data.get("cycle", 0)))
    table.add_row("Bankroll", f"${hb_data.get('bankroll', 0):.2f}")

    if hb_data.get("dry_run"):
        table.add_row("Mode", "[yellow]DRY-RUN[/yellow]")
        paper = PaperTradeDB()
        try:
            stats = paper.get_trade_stats()
            table.add_row("Paper P&L", f"${stats['total_pnl']:.2f}")
            table.add_row("Return", f"{stats['return_pct']:.1f}%")
            table.add_row("Win Rate", f"{stats['win_rate']:.0%}")
            table.add_row("Settled", str(stats["total_settled"]))
            table.add_row("Open Positions", str(stats["open_positions"]))
        except Exception:
            pass
        finally:
            paper.close()

    table.add_row("Daily Spent", f"${daily_spent:.2f}")
    table.add_row("Daily P&L", f"${daily_pnl:.2f}")
    table.add_row("Open Orders", str(open_trades))
    table.add_row("Total Exposure", f"${total_exposure:.2f}")
    table.add_row(
        "Last Heartbeat",
        datetime.fromtimestamp(hb_data.get("ts", 0)).strftime("%H:%M:%S") if hb_data.get("ts") else "N/A",
    )

    console.print(table)


def _setup_logging(debug: bool) -> None:
    import logging

    if debug:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG))
    else:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
