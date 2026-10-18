#!/usr/bin/env python3
"""Falsification test: resolve bets against TRUE_TEMP (the synthetic
realization) instead of member fractions.

The stock backtest resolves via `won = rng.random() < bucket.true_prob`
where true_prob = fraction of ensemble members in the bucket. That tests
whether EMOS(member stats) predicts *member fractions* better than the
market — circular, since both derive from the same members.

This script re-resolves against the actual synthetic outcome
(true_temp in bucket) — the honest question: does the model beat the
market at predicting realizations?
"""
from __future__ import annotations

import importlib.util
import random
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WS_PATH = ROOT / "bots" / "weather" / "backtest" / "weather_strategies.py"
RUN_PATH = ROOT / "run_weather_backtest.py"

stub_pkg = types.ModuleType("backtest")
stub_pkg.__path__ = [str(WS_PATH.parent)]
sys.modules["backtest"] = stub_pkg
spec = importlib.util.spec_from_file_location("backtest.weather_strategies", WS_PATH)
ws = importlib.util.module_from_spec(spec)
sys.modules["backtest.weather_strategies"] = ws
spec.loader.exec_module(ws)

from dataclasses import dataclass as _dc

@_dc
class _ExecutedTrade:
    city: str; date: str; bucket_low: float; bucket_high: float
    direction: str; entry_price: float; model_prob: float; edge: float
    size_usd: float; won: bool; pnl: float; cost: float; strategy: str
    lead_time_hours: float = 48.0

spec2 = importlib.util.spec_from_file_location("wb_sim", RUN_PATH)
wb = importlib.util.module_from_spec(spec2)
sys.modules["wb_sim"] = wb
wb.ExecutedTrade = _ExecutedTrade
spec2.loader.exec_module(wb)


class HighTailStrategy(ws.TailStrategy):
    name = "high_tail"
    description = "高温尾部：只买 ≥X°C 开口桶"

    def generate_signals(self, city, date, buckets, model_mu, model_sigma,
                         emos_coeffs=None, lead_time_hours=48.0):
        high_tails = [b for b in buckets if b.high_c >= 999.0]
        return super().generate_signals(
            city, date, high_tails, model_mu, model_sigma,
            emos_coeffs, lead_time_hours)


def run_honest(days=90, seed=42, bankroll=100.0):
    """Same sim, but resolve against true_temp (the realization)."""
    from math import erf as _erf, sqrt as _sqrt
    rng = random.Random(seed)
    np.random.seed(seed)
    emos_coeffs = wb.load_emos_coeffs()
    cities = list(wb.CITIES.keys())
    strat = HighTailStrategy()
    exec_cfg = wb.ExecutionConfig()
    market_cfg = wb.MarketSimConfig()
    today = datetime.now()

    # tail-augmented sim (same as tail_backtest.py)
    _orig_sim = wb.simulate_market_buckets

    def sim_with_tails(true_temp, members, rng_, city="", cfg=None):
        buckets = _orig_sim(true_temp, members, rng_, city, cfg or market_cfg)
        cfg = cfg or market_cfg
        arr = np.array(members)
        m_mean = float(np.mean(arr))
        m_std = max(float(np.std(arr)), 1.5)
        thresh = float(np.floor(m_mean + 1.2 * m_std))
        mkt_std = max(1.5, m_std * 0.85)
        z = (thresh - m_mean) / mkt_std
        mkt = 0.5 * (1.0 - _erf(z / _sqrt(2))) * cfg.tail_underprice_factor
        mkt = max(0.01, min(0.50, mkt + rng_.gauss(0, 0.012)))
        buckets.append(wb.Bucket(low_c=thresh, high_c=999.0, market_price=mkt))
        return buckets

    bal = bankroll
    wins = losses = 0
    pnls = []
    n_trades = 0
    for day_offset in range(days):
        date = today - timedelta(days=days - day_offset)
        for city in cities:
            city_cfg = wb.CITIES[city]
            coeffs = emos_coeffs.get(city, {})
            true_temp, members = wb.generate_forecast(city, city_cfg, date, rng)
            mu, sigma = (wb.emos_calibrate(members, coeffs) if coeffs
                         else (float(np.mean(members)), float(np.std(members))))
            buckets = sim_with_tails(true_temp, members, rng, city)
            signals = strat.generate_signals(
                city=city, date=date.strftime("%Y-%m-%d"), buckets=buckets,
                model_mu=mu, model_sigma=sigma,
                emos_coeffs=coeffs or None,
                lead_time_hours=rng.uniform(6, 72))
            for signal in signals:
                if bal < 1.0:
                    break
                # HONEST resolution: did the realization land in the bucket?
                b = signal.bucket
                won = (b.low_c <= true_temp < b.high_c) if b.high_c < 999 else (true_temp >= b.low_c)
                # price/fill/cost model copied from execute_trade
                fill = signal.bucket.market_price * (1.0 + exec_cfg.taker_fee_bps / 10000 * 0)
                fill = signal.bucket.market_price * (1.0 + exec_cfg.slippage_pct)
                fill = min(max(fill, 0.01), 0.99)
                size = max(min(signal.size_pct * bal, 50.0), 1.0)
                if size > bal * 0.5:
                    continue
                edge = signal.edge - exec_cfg.edge_decay_per_min * exec_cfg.fill_delay_minutes
                if edge <= 0:
                    continue
                cost = exec_cfg.taker_fee_bps / 10000.0 * fill * size
                pnl = size * (1.0 - fill) - cost if won else -size * fill - cost
                bal += pnl
                pnls.append(pnl)
                n_trades += 1
                wins += won
                losses += (not won)
    wr = wins / max(n_trades, 1)
    ret = (bal - bankroll) / bankroll * 100
    print(f"HONEST (resolve vs true_temp): trades={n_trades} win_rate={wr:.1%} "
          f"return={ret:+.1f}% final=${bal:.2f}")


if __name__ == "__main__":
    run_honest()
