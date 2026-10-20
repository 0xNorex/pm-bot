"""S4: same-day running-max lock-in (added 2026-10-20).

Late in the local day the daily max is nearly determined: the final max
cannot fall below the running max, and the remaining upside shrinks fast.
Model: final_max ~ N(running_max + drift, sigma_rem), truncated below at
running_max (censored normal — Andreev's same-day censored regression cut
CRPS 0.81 -> 0.61C). Buckets the censored model calls dead but the market
still prices as live -> buy NO.

Conservative v1 parameters: start 16:00 local, drift +0.15C,
sigma_rem 0.35C, min edge 0.03, NO price in [0.50, 0.98]. Paper only.
Requires the daemon to pass day_high (core.observation.fetch_day_high:
METAR running max at the settlement station).
"""
from __future__ import annotations

from dataclasses import dataclass

from pm_bot.models.market import ForecastResult, Recommendation, TemperatureBucket, WeatherEvent
from pm_bot.strategies.base import Strategy
from pm_bot.strategies.no_fade import _TAIL_BOUND, _norm_cdf, _to_c


@dataclass
class S4Config:
    start_hour: int = 16          # local hour from which the max is ~locked
    drift_c: float = 0.15         # expected remaining rise after start_hour
    sigma_rem_c: float = 0.35     # std of the remaining rise
    min_edge: float = 0.03
    max_no_price: float = 0.98
    min_no_price: float = 0.50
    stake_usd: float = 20.0


class S4LockStrategy(Strategy):
    name = "s4_lock"

    def __init__(self, cfg: S4Config | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg or S4Config()

    def _censored_p(self, rm_c: float, bucket: TemperatureBucket) -> float:
        """P(final max in bucket | final >= running max), censored normal.

        Bucket edges use the same (temp_low, temp_high) semantics as
        NoFadeStrategy.bucket_model_p: rounding already baked in by the
        event parser; _TAIL_BOUND marks open ends.
        """
        lo, hi = bucket.temp_low, bucket.temp_high
        # Same displayed-value geometry as NoFadeStrategy.bucket_model_p:
        # printed [lo, hi] covers continuous [lo-0.5, hi+0.5] native.
        lo_n = lo - 0.5 if lo > -_TAIL_BOUND else lo
        hi_n = hi + 0.5 if hi < _TAIL_BOUND else hi
        if bucket.temp_unit == "F":
            lo_c = _to_c(lo_n) if lo > -_TAIL_BOUND else float("-inf")
            hi_c = _to_c(hi_n) if hi < _TAIL_BOUND else None
        else:
            lo_c = lo_n if lo > -_TAIL_BOUND else float("-inf")
            hi_c = hi_n if hi < _TAIL_BOUND else None
        mu = rm_c + self.cfg.drift_c
        s = self.cfg.sigma_rem_c
        denom = 1.0 - _norm_cdf((rm_c - mu) / s)
        if denom <= 1e-9:
            return 0.0
        lo_e = max(lo_c, rm_c)
        if hi_c is None:
            return max(0.0, min(1.0, (1.0 - _norm_cdf((lo_e - mu) / s)) / denom))
        if hi_c <= rm_c:
            return 0.0
        p = (_norm_cdf((hi_c - mu) / s) - _norm_cdf((lo_e - mu) / s)) / denom
        return max(0.0, min(1.0, p))

    def run(
        self,
        event: WeatherEvent,
        forecast: ForecastResult | None = None,
        day_high=None,
        **kwargs,
    ) -> list[Recommendation]:
        if day_high is None:
            return []
        lt = day_high.local_time
        if lt.hour < self.cfg.start_hour:
            return []
        # The daemon fetches events by server date; only trade when the
        # event is the city's local today, else the running max is for
        # the wrong day.
        if event.date != f"{lt:%B} {lt.day}":
            return []
        rm_c = day_high.observed_c
        recs: list[Recommendation] = []
        for bucket in event.buckets:
            p_market = bucket.yes_price
            if not (0.01 < p_market < 0.99):
                continue
            p_no = 1.0 - p_market
            if not (self.cfg.min_no_price <= p_no <= self.cfg.max_no_price):
                continue
            p_model = self._censored_p(rm_c, bucket)
            edge = p_market - p_model
            if edge < self.cfg.min_edge:
                continue
            recs.append(Recommendation(
                strategy=self.name,
                event=event,
                bucket=bucket,
                direction="NO",
                edge=edge,
                reasoning=(
                    f"S4 LOCK | running max {rm_c:.1f}C at {lt:%H:%M} local; "
                    f"market YES {p_market:.1%} vs censored model {p_model:.1%}; "
                    f"buy NO at {p_no:.1%}"
                ),
                size_usd=self.cfg.stake_usd,
            ))
        return recs
