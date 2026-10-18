#!/usr/bin/env python3
"""Standalone tail-focus backtest runner.

Bypasses the stale pm_bot package __init__ chain (strategies were deleted
2026-05-16; live imports are broken) by loading weather_strategies.py
directly. Compares:
  1. TailStrategy baseline (all tails, both directions)
  2. HighTailStrategy: HIGH-temperature tails only (>=X°C open buckets)

Usage: .venv/bin/python research/tail_backtest.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WS_PATH = ROOT / "bots" / "weather" / "backtest" / "weather_strategies.py"
RUN_PATH = ROOT / "run_weather_backtest.py"

# ── Stub the broken `backtest` package so its __init__ never executes ──
stub_pkg = types.ModuleType("backtest")
stub_pkg.__path__ = [str(WS_PATH.parent)]
sys.modules["backtest"] = stub_pkg

spec = importlib.util.spec_from_file_location("backtest.weather_strategies", WS_PATH)
ws = importlib.util.module_from_spec(spec)
sys.modules["backtest.weather_strategies"] = ws
spec.loader.exec_module(ws)

# ── Load the sim harness (its `from backtest.weather_strategies import …`
#    now resolves to the module above) ──
spec2 = importlib.util.spec_from_file_location("wb_sim", RUN_PATH)
wb = importlib.util.module_from_spec(spec2)
sys.modules["wb_sim"] = wb

# ── Patch staleness: run_weather_backtest.py references ExecutedTrade
#    but never defines it (dead code after the 2026-05-16 strategy purge).
#    Define a compatible dataclass here instead of editing the original. ──
from dataclasses import dataclass as _dc

@_dc
class _ExecutedTrade:
    city: str
    date: str
    bucket_low: float
    bucket_high: float
    direction: str
    entry_price: float
    model_prob: float
    edge: float
    size_usd: float
    won: bool
    pnl: float
    cost: float
    strategy: str
    lead_time_hours: float = 48.0

wb.ExecutedTrade = _ExecutedTrade
spec2.loader.exec_module(wb)


class HighTailStrategy(ws.TailStrategy):
    """Hans323 tail, HIGH-temperature tails only (>=X°C open-ended buckets)."""

    name = "high_tail"
    description = "高温尾部：只买 ≥X°C 开口桶"

    def generate_signals(self, city, date, buckets, model_mu, model_sigma,
                         emos_coeffs=None, lead_time_hours=48.0):
        high_tails = [b for b in buckets if b.high_c >= 999.0]
        return super().generate_signals(
            city, date, high_tails, model_mu, model_sigma,
            emos_coeffs, lead_time_hours,
        )


def show(title: str, r) -> None:
    print(f"\n=== {title} ===")
    print(f"  trades: {r.total_trades}  win_rate: {r.win_rate:.1%}  "
          f"return: {r.total_return_pct:+.1f}%  profit_factor: {r.profit_factor:.2f}")
    print(f"  avg_edge: {r.avg_edge:.2%}  avg_pnl/trade: ${r.avg_pnl_per_trade:+.2f}  "
          f"max_dd: {r.max_drawdown_pct:.1f}%  sharpe: {r.sharpe_ratio:.2f}")
    print(f"  best: ${r.best_trade_pnl:+.2f}  worst: ${r.worst_trade_pnl:+.2f}  "
          f"costs: ${r.total_costs:.2f}")


def main() -> None:
    emos_coeffs = wb.load_emos_coeffs()
    print(f"EMOS coeffs loaded for {len(emos_coeffs)} cities")

    # ── Augment the sim with TRUE open-ended tail buckets ──
    # The stock sim only makes 2°C interior buckets; real Polymarket
    # high-temp markets have a "≥X°C" open bucket. Append one high tail
    # (and one low tail for symmetry) per city-day.
    from math import erf as _erf, sqrt as _sqrt
    import numpy as _np
    _orig_sim = wb.simulate_market_buckets

    def sim_with_tails(true_temp, members, rng, city="", cfg=None):
        buckets = _orig_sim(true_temp, members, rng, city, cfg)
        if cfg is None:
            cfg = wb.MarketSimConfig()
        top = max(b.high_c for b in buckets)
        bot = min(b.low_c for b in buckets)
        # Rebuild market mean/std the same way the sim does (approx):
        # use ensemble mean/std as proxy for market center.
        arr = _np.array(members)
        m_mean, m_std = float(_np.mean(arr)), max(float(_np.std(arr)), 1.5)
        # High tail threshold: ~1.2σ above ensemble mean — the zone where a
        # real "≥X°C" bucket sits (model prob ~10%, market often <8¢).
        thresh = float(_np.floor(m_mean + 1.2 * m_std))
        # Market price: Gaussian tail at market's own (tighter) spread,
        # then the crowd's tail-underpricing factor.
        mkt_std = max(1.5, m_std * 0.85)
        z = (thresh - m_mean) / mkt_std
        mkt = 0.5 * (1.0 - _erf(z / _sqrt(2))) * cfg.tail_underprice_factor
        mkt = max(0.01, min(0.50, mkt + rng.gauss(0, 0.012)))
        true_p = float(_np.sum(arr >= thresh)) / len(arr)
        buckets.append(wb.Bucket(low_c=thresh, high_c=999.0,
                                 market_price=mkt, true_prob=true_p))
        # Low tail mirror for the baseline comparison.
        thresh_lo = float(_np.ceil(m_mean - 1.2 * m_std))
        z2 = (thresh_lo - m_mean) / mkt_std
        mkt2 = 0.5 * (1.0 + _erf(z2 / _sqrt(2))) * cfg.tail_underprice_factor
        mkt2 = max(0.01, min(0.50, mkt2 + rng.gauss(0, 0.012)))
        true_p2 = float(_np.sum(arr <= thresh_lo)) / len(arr)
        buckets.append(wb.Bucket(low_c=-999.0, high_c=thresh_lo,
                                 market_price=mkt2, true_prob=true_p2))
        return buckets

    wb.simulate_market_buckets = sim_with_tails

    strategies = {
        "tail (all tails)": ws.TailStrategy(),
        "high_tail (>=X°C only)": HighTailStrategy(),
    }
    for title, strat in strategies.items():
        r = wb.run_backtest(
            strategy=strat,
            scenario_name="tail_focus_90d",
            days=90,
            bankroll=100.0,
            seed=42,
            emos_coeffs=emos_coeffs,
        )
        show(title, r)

    print("\n--- sensitivity: crowd tail-underpricing factor ---")
    _base_sim = sim_with_tails  # the tail-augmented sim; wrap fresh each time
    for factor in (0.7, 0.85, 1.0):
        def sim_fixed(true_temp, members, rng, city="", cfg=None, _f=factor):
            c = wb.MarketSimConfig()
            c.tail_underprice_factor = _f
            return _base_sim(true_temp, members, rng, city, c)

        wb.simulate_market_buckets = sim_fixed
        r = wb.run_backtest(
            strategy=HighTailStrategy(),
            scenario_name=f"high_tail_f{factor}",
            days=90, bankroll=100.0, seed=42, emos_coeffs=emos_coeffs,
        )
        show(f"high_tail factor={factor}", r)


if __name__ == "__main__":
    main()
