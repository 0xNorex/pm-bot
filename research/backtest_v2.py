#!/usr/bin/env python3
"""Backtest v2 — honest DGP (first-principles rewrite).

v1 的三个原罪:
  1. 循环结算: won ~ Bernoulli(member_fraction)，模型和"真相"来自同一组成员
  2. DGP 失配: sim 用 σ≈2 生成成员，EMOS 系数(真实数据训练)却给出 σ≈0.8
  3. 合成市场把 edge 写死在假设里 (tail_underprice_factor=0.7)

v2 的第一性原理 (修正版 — 时间顺序!):
  现实的时间顺序是: 先有预报 → 后有实现。v2 初版搞反了 (先实现，后成员)，
  导致成员几乎完全揭示实现，EMOS 的 σ 反而成了误导。

  正确 DGP, 每 (city, day):
  1. 预报中心: f_mean ~ 季节模型 + N(0,3)      (预报员的最佳猜测)
  2. 成员: 31 × N(f_mean, 1.5)                  (集合离散度)
  3. 实现: true_temp ~ N(f_mean, σ_emos_city)   (真实预报误差!)
     σ_emos_city = sqrt(c + d*2.25)，即该城市 EMOS 系数隐含的尺度
     → EMOS(members) 按构造就是校准良好的 (up to 31 个成员的抽样噪声)
  4. 模型: P_model = EMOS(members)             (校准良好，唯一噪声是抽样)
  5. 市场: P_market = P_true * bias + noise    (bias 只作用于尾部，可扫)
     P_true = Φ(bucket | N(f_mean, σ_emos_city)) (完美预报者知道的)
  6. 结算: 1(true_temp ∈ bucket)               (对实现结算，无循环)

  诚实性: 模型和市场都从同一个底层真相派生，但走不同的噪声通道；
  唯一的系统性 edge 来源 = 市场 bias 参数。扫 bias 得 breakeven。
  模型估计噪声 (31 成员) 是真实的 — 这是小样本策略必须付的代价。
"""
from __future__ import annotations

import importlib.util
import random
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import erf, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WS_PATH = ROOT / "bots" / "weather" / "backtest" / "weather_strategies.py"

stub_pkg = types.ModuleType("backtest")
stub_pkg.__path__ = [str(WS_PATH.parent)]
sys.modules["backtest"] = stub_pkg
spec = importlib.util.spec_from_file_location("backtest.weather_strategies", WS_PATH)
ws = importlib.util.module_from_spec(spec)
sys.modules["backtest.weather_strategies"] = ws
spec.loader.exec_module(ws)

# ── City seasonal config (from run_weather_backtest.py CITIES) ──
CITIES = {
    "New York": {"temp_range": (-5, 38), "season_amp": 15},
    "Miami": {"temp_range": (15, 38), "season_amp": 5},
    "London": {"temp_range": (2, 28), "season_amp": 10},
    "Paris": {"temp_range": (3, 30), "season_amp": 11},
    "Tokyo": {"temp_range": (5, 34), "season_amp": 12},
    "Seoul": {"temp_range": (-3, 32), "season_amp": 14},
    "Shanghai": {"temp_range": (3, 36), "season_amp": 13},
    "Hong Kong": {"temp_range": (15, 33), "season_amp": 6},
    "Chicago": {"temp_range": (-8, 32), "season_amp": 16},
    "Atlanta": {"temp_range": (8, 35), "season_amp": 10},
    "Los Angeles": {"temp_range": (12, 30), "season_amp": 5},
    "San Francisco": {"temp_range": (10, 25), "season_amp": 4},
    "Dallas": {"temp_range": (8, 38), "season_amp": 12},
    "Beijing": {"temp_range": (-5, 33), "season_amp": 15},
}

TAIL_BOUND = 999.0


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def perfect_bucket_prob(mu: float, sigma: float, low_c: float, high_c: float) -> float:
    """P(bucket) under the perfect forecaster N(mu, sigma)."""
    if high_c >= TAIL_BOUND:
        return 1.0 - norm_cdf((low_c - mu) / sigma)
    if low_c <= -TAIL_BOUND:
        return norm_cdf((high_c - mu) / sigma)
    return norm_cdf((high_c - mu) / sigma) - norm_cdf((low_c - mu) / sigma)


@dataclass
class V2Config:
    days: int = 90
    bankroll: float = 100.0
    seed: int = 42
    tail_bias: float = 0.7        # market underpricing factor for tails
    market_noise: float = 0.012   # execution noise on market price
    slippage_pct: float = 0.015
    taker_fee_bps: int = 50
    min_price: float = 0.01


def city_sigma(city: str, coeffs: dict) -> float:
    """EMOS-implied scale for this city (DGP consistency anchor)."""
    c = coeffs.get("city", {})
    cc = c.get("c", 4.0)
    dd = c.get("d", 0.5)
    return max(0.8, sqrt(cc + dd * 2.25))


def run_v2(cfg: V2Config) -> dict:
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    all_coeffs = ws.load_emos_coeffs()
    today = datetime.now()

    strat = ws.TailStrategy()  # Hans323 tail, thresholds as coded
    bal = cfg.bankroll
    wins = n = 0
    pnls: list[float] = []

    for day_offset in range(cfg.days):
        date = today - timedelta(days=cfg.days - day_offset)
        doy = date.timetuple().tm_yday
        for city, ccfg in CITIES.items():
            lo, hi = ccfg["temp_range"]
            mid = (lo + hi) / 2.0
            coeffs = all_coeffs.get(city, {})
            sigma = city_sigma(city, all_coeffs)  # EMOS-implied error scale
            # 1. forecaster's best guess (BEFORE the outcome exists)
            f_mean = mid + ccfg["season_amp"] * np.sin(2 * np.pi * doy / 365.0 - np.pi / 2)
            f_mean += rng.gauss(0, 3.0)
            # 2. ensemble members around the forecast
            members = [f_mean + rng.gauss(0, 1.5) for _ in range(31)]
            # 3. realization drawn from the TRUE conditional distribution
            true_temp = f_mean + rng.gauss(0, sigma)
            # 4. model: EMOS calibration of members (well-calibrated by construction)
            mu_m, sigma_m = ws.emos_calibrate(members, coeffs)

            # 5. market: biased perfect-P + noise. Tail bucket at FIXED X
            #    (real markets fix X, e.g. NY ≥35°C — it does NOT move with forecast)
            buckets: list[ws.Bucket] = []
            t_lo = int(np.floor(f_mean)) - 12
            t_hi = int(np.ceil(f_mean)) + 12
            for low in range(t_lo, t_hi, 2):
                high = low + 2
                p_perf = perfect_bucket_prob(f_mean, sigma, float(low), float(high))
                p_mkt = max(cfg.min_price, min(0.99, p_perf + rng.gauss(0, cfg.market_noise)))
                buckets.append(ws.Bucket(low_c=float(low), high_c=float(high),
                                         market_price=p_mkt))
            thresh = float(ccfg["temp_range"][1] - 3)  # FIXED threshold
            p_perf_t = perfect_bucket_prob(f_mean, sigma, thresh, TAIL_BOUND)
            p_mkt_t = max(cfg.min_price,
                          min(0.50, p_perf_t * cfg.tail_bias + rng.gauss(0, cfg.market_noise)))
            buckets.append(ws.Bucket(low_c=thresh, high_c=TAIL_BOUND, market_price=p_mkt_t))

            # 6. strategy signals (model = EMOS on members)
            signals = strat.generate_signals(
                city=city, date=date.strftime("%Y-%m-%d"), buckets=buckets,
                model_mu=mu_m, model_sigma=sigma_m,
                emos_coeffs=coeffs or None, lead_time_hours=rng.uniform(6, 72))
            for s in signals:
                if bal < 1.0:
                    break
                # 7. HONEST settlement vs the realization
                b = s.bucket
                won = (true_temp >= b.low_c) if b.high_c >= TAIL_BOUND else (b.low_c <= true_temp < b.high_c)
                fill = min(max(s.bucket.market_price * (1 + cfg.slippage_pct), 0.01), 0.99)
                size = max(min(s.size_pct * bal, 50.0), 1.0)
                if size > bal * 0.5:
                    continue
                cost = cfg.taker_fee_bps / 10000.0 * fill * size
                pnl = (size * (1.0 - fill) - cost) if won else (-size * fill - cost)
                bal += pnl
                pnls.append(pnl)
                n += 1
                wins += won

    ret = (bal - cfg.bankroll) / cfg.bankroll * 100
    return {"trades": n, "win_rate": wins / max(n, 1), "return_pct": ret,
            "final": bal, "avg_pnl": float(np.mean(pnls)) if pnls else 0.0}


def main() -> None:
    print("v2 honest backtest — tail strategy, 90d, 14 cities")
    print("sweep: market tail-bias factor (1.0 = no mispricing)")
    for bias in (0.5, 0.7, 0.85, 1.0):
        r = run_v2(V2Config(tail_bias=bias))
        print(f"  bias={bias:.2f}: trades={r['trades']:4d} win={r['win_rate']:.1%} "
              f"ret={r['return_pct']:+7.1f}% avg=${r['avg_pnl']:+.2f}")


if __name__ == "__main__":
    main()
