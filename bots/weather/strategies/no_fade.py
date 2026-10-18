"""S1: NO Fade — buy NO on overpriced YES buckets.

Thesis (validated on 150-event replay, 2026-10-18):
  Polymarket daily high-temp markets systematically OVERPRICE buckets
  relative to an honest forecast model. Buy NO where the overpricing
  exceeds the edge threshold.

  Replay: 76 trades, 82.9% win, +$105.9 ($10 flat, fees incl).
  Robust across cities (NY/Chicago/London) and time halves.

Signal:  P_market(YES) - P_model(YES) > edge_threshold
  where P_model comes from the deterministic GFS forecast (previous-runs)
  plus the empirically measured 24h error distribution
  (bias -0.54C, sigma 0.90C, n=150).

Costs: taker fee 50bps + 1.5% slippage modeled in backtest.
Sizing: flat $10/trade for paper; Kelly once live calibration is confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

from pm_bot.models.market import (
    ForecastResult,
    Recommendation,
    TemperatureBucket,
    WeatherEvent,
)
from pm_bot.strategies.base import Strategy

# Empirically measured 24h GFS deterministic forecast error (n=150, 2026-10-18)
ERROR_BIAS_C = -0.54
ERROR_SIGMA_C = 0.90

_TAIL_BOUND = 999.0


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


@dataclass
class NoFadeConfig:
    edge_threshold: float = 0.10   # min overpricing to trade
    stake_usd: float = 10.0        # flat stake per signal (paper)
    max_no_price: float = 0.95     # don't buy NO above 95c (adverse selection)
    min_no_price: float = 0.05     # don't buy NO below 5c (dust)


class NoFadeStrategy(Strategy):
    """Buy NO on buckets the market overprices vs the forecast model."""

    name: str = "no_fade"

    def __init__(self, config: NoFadeConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config or NoFadeConfig()

    def bucket_model_p(self, fcst_c: float, bucket: TemperatureBucket) -> float:
        """P(bucket wins) under N(fcst + bias, sigma)."""
        mu = fcst_c + ERROR_BIAS_C
        s = ERROR_SIGMA_C
        lo, hi = bucket.temp_low, bucket.temp_high
        # buckets may be in F; normalize via temp_unit
        if bucket.temp_unit == "F":
            lo_c, hi_c = _to_c(lo), _to_c(hi) if hi < _TAIL_BOUND else None
        else:
            lo_c, hi_c = lo, hi if hi < _TAIL_BOUND else None
        if hi_c is None:
            return 1.0 - _norm_cdf((lo_c - mu) / s)
        if lo_c is None or lo <= -_TAIL_BOUND:
            return _norm_cdf((hi_c - mu) / s)
        return _norm_cdf((hi_c - mu) / s) - _norm_cdf((lo_c - mu) / s)

    def run(
        self,
        event: WeatherEvent,
        forecast: ForecastResult | None = None,
        **kwargs,
    ) -> list[Recommendation]:
        if forecast is None:
            return []
        # deterministic forecast: use ensemble mean if members exist, else temp_high_c
        fcst_c = forecast.mean if forecast.members else forecast.temp_high_c

        recs: list[Recommendation] = []
        for bucket in event.buckets:
            p_market = bucket.yes_price
            if not (0.01 < p_market < 0.99):
                continue
            p_no = 1.0 - p_market
            if not (self.cfg.min_no_price <= p_no <= self.cfg.max_no_price):
                continue
            p_model = self.bucket_model_p(fcst_c, bucket)
            edge = p_market - p_model
            if edge < self.cfg.edge_threshold:
                continue
            recs.append(Recommendation(
                strategy=self.name,
                event=event,
                bucket=bucket,
                direction="NO",
                edge=edge,
                reasoning=(
                    f"NO FADE | market YES {p_market:.1%} vs model {p_model:.1%} "
                    f"overpriced by {edge:.1%}; buy NO at {p_no:.1%}"
                ),
                size_usd=self.cfg.stake_usd,
            ))
        return recs
