"""Control arm: model-free price-band NO buying (added 2026-10-20).

A/B control for S1 (no_fade), approved by user 2026-10-20 after the
honest point-in-time backtest (research/replay/backtest_honest.py,
61 days, 481 city-days) showed:

- market-control (buy NO on every bucket with YES in [0.15, 0.60],
  no model at all): n=208, win 80.8%, +$555.26 — beat S1-live.
- forecast-shuffle placebo: the real forecast beat only 6% of
  placebos -> the S1 forecast filter adds no measurable value.
  The edge is the structural overpricing of mid-band YES buckets.

This strategy uses NO forecast. Its "edge" is the empirical
mispricing from the honest dataset's calibration curve (priced
buckets, YES in band, winner from gamma outcomePrices):

  YES bin      n    YES win rate   mean YES price   implied edge
  [0.15,0.25]  108     9.3%           0.192           +0.099
  [0.25,0.35]   65    16.9%           0.291           +0.122
  [0.35,0.45]   53    24.5%           0.394           +0.148
  [0.45,0.60]   31    25.8%           0.485           +0.227

edge(bucket) = yes_price - fair_yes(yes_price), fair from the table.
Runs in the same paper ledger as S1 (strategy='band_control') so
live fills / adverse selection decide the A/B, not another backtest.
Shares the daemon's daily/risk caps with the other strategies.
"""
from __future__ import annotations

from dataclasses import dataclass

from pm_bot.models.market import ForecastResult, Recommendation, WeatherEvent
from pm_bot.strategies.base import Strategy

# Empirical YES win rate by price bin, honest dataset 2026-08-20..10-19.
_FAIR_BINS: list[tuple[float, float, float]] = [
    # (bin_lo, bin_hi, empirical YES win rate)
    (0.15, 0.25, 0.093),
    (0.25, 0.35, 0.169),
    (0.35, 0.45, 0.245),
    (0.45, 0.60, 0.258),
]


def fair_yes(p_yes: float) -> float | None:
    for lo, hi, rate in _FAIR_BINS:
        if lo <= p_yes < hi or (p_yes == hi == 0.60):
            return rate
    return None


@dataclass
class BandControlConfig:
    band_lo: float = 0.15
    band_hi: float = 0.60
    min_edge: float = 0.05
    stake_usd: float = 20.0


class BandControlStrategy(Strategy):
    name = "band_control"

    def __init__(self, cfg: BandControlConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg or BandControlConfig()

    def run(
        self,
        event: WeatherEvent,
        forecast: ForecastResult | None = None,
        **kwargs,
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        for bucket in event.buckets:
            p_market = bucket.yes_price
            if not (self.cfg.band_lo <= p_market <= self.cfg.band_hi):
                continue
            fair = fair_yes(p_market)
            if fair is None:
                continue
            edge = p_market - fair
            if edge < self.cfg.min_edge:
                continue
            p_no = 1.0 - p_market
            recs.append(Recommendation(
                strategy=self.name,
                event=event,
                bucket=bucket,
                direction="NO",
                edge=edge,
                reasoning=(
                    f"BAND CONTROL | no model; market YES {p_market:.1%} in "
                    f"[{self.cfg.band_lo:.2f},{self.cfg.band_hi:.2f}] band vs "
                    f"empirical fair {fair:.1%} (honest-data calibration); "
                    f"buy NO at {p_no:.1%}"
                ),
                size_usd=self.cfg.stake_usd,
            ))
        return recs
