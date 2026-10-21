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

CORRECTION (2026-10-20): the replay sigma/bias came from previous-runs,
whose "forecasts" are contaminated by near-actual data (audit finding).
First live paper day (2026-10-19): 13 trades, 5W/8L, -$40 — the model
assigned ~0% to buckets that WON in all 5 cities. Live 24h forecast
error is much larger than 0.90C. Until live (forecast, actual) pairs
accumulate in ~/.pm-bot/forecast_log.csv for a true recalibration, use
a literature-grade conservative sigma (1.60C) and cap the edge: a
signal whose implied edge exceeds max_edge is treated as model error,
not market error, and skipped.

Costs: taker fee 50bps + 1.5% slippage modeled in backtest.
Sizing: flat $20/trade for paper; Kelly once live calibration is confirmed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from math import erf, sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

from pm_bot.core.observation import CITY_TZ, fetch_day_high_for_date
from pm_bot.models.market import (
    ForecastResult,
    Recommendation,
    TemperatureBucket,
    WeatherEvent,
)
from pm_bot.strategies.base import Strategy

# EMOS spread-dependent sigma (2026-10-20): sigma = sqrt(a + b * spread^2).
# a=1.44 -> sigma floor 1.2C on zero-spread days (raw ensembles are
# systematically underdispersive); b=1.0 per Gneiting et al. 2005 scale.
# Refit (a, b) by min-CRPS once >= 30 (forecast, actual) pairs accumulate
# in ~/.pm-bot/forecast_log.csv. Replaces the fixed sigma=1.6 (2026-10-20
# stopgap, itself replacing the contaminated replay sigma 0.90).
EMOS_A = 1.44
EMOS_B = 1.0
EMOS_SIGMA_FLOOR = 1.2
EMOS_SIGMA_CAP = 3.0


def emos_sigma(spread_c: float | None) -> float:
    """Calibrated predictive sigma from the ensemble spread."""
    s2 = float(spread_c or 0.0)
    return min(EMOS_SIGMA_CAP, max(EMOS_SIGMA_FLOOR, sqrt(EMOS_A + EMOS_B * s2 * s2)))


# Replay-measured error (previous-runs, n=150) — DO NOT USE for live:
# contaminated by near-actual data, see module docstring / 2026-10-20 audit.
REPLAY_BIAS_C = -0.54
REPLAY_SIGMA_C = 0.90

# Live calibration in effect (2026-10-20): conservative literature-grade
# 24h daily-max forecast error until live pairs accumulate for refit.
ERROR_BIAS_C = 0.0
ERROR_SIGMA_C = 1.60
FORECAST_LOG_PATH = Path.home() / ".pm-bot" / "forecast_log.csv"

_TAIL_BOUND = 999.0


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


@dataclass
class NoFadeConfig:
    edge_threshold: float = 0.12   # min overpricing to trade (walk-forward optimal)
    max_edge: float = 0.30         # SKIP signals above this (2026-10-20: model error, not market error)
    stake_usd: float = 20.0        # flat stake per signal (paper)
    max_no_price: float = 0.95     # don't buy NO above 95c (adverse selection)
    min_no_price: float = 0.05     # don't buy NO below 5c (dust)


_FORECAST_LOG_HEADER = ["fetched_at", "city", "target_date", "fcst_c", "spread_c",
                        "bias_c", "sigma_c", "members", "model", "actual_c"]


def _append_forecast_log(city: str, target: str, fcst_c: float, spread_c: float, bias_c: float, sigma_c: float, members: int = 0, model: str = "") -> None:
    """Append (forecast, spread, emos sigma, members, model) for live recalibration.

    Actuals are backfilled by backfill_forecast_actuals() once the target
    date has elapsed; rows with spread_c logged let us refit EMOS (a, b)
    by min-CRPS once enough pairs accumulate.
    """
    try:
        FORECAST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_file = not FORECAST_LOG_PATH.exists()
        with FORECAST_LOG_PATH.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(_FORECAST_LOG_HEADER)
            w.writerow([date.today().isoformat(), city, target, f"{fcst_c:.2f}",
                        f"{spread_c:.2f}", f"{bias_c:.2f}", f"{sigma_c:.2f}",
                        members, model, ""])
    except OSError:
        pass  # logging must never break trading


def _read_forecast_log() -> list[list[str]]:
    """Tolerant read of forecast_log.csv.

    Legacy rows (2026-10-20/21) were written unquoted, so the model field
    may itself contain commas. The first 8 fields are fixed and the last
    field is actual_c; everything in between is rejoined as the model.
    """
    if not FORECAST_LOG_PATH.exists():
        return []
    rows: list[list[str]] = []
    with FORECAST_LOG_PATH.open(newline="") as f:
        for raw in csv.reader(f):
            if not raw or raw[0] == "fetched_at" or len(raw) < 10:
                continue
            rows.append(raw[:8] + [",".join(raw[8:-1]), raw[-1]])
    return rows


def _parse_target_date(target: str, fetched_at: str) -> date | None:
    try:
        year = int(fetched_at[:4])
        return datetime.strptime(f"{target} {year}", "%B %d %Y").date()
    except (ValueError, IndexError):
        return None


async def backfill_forecast_actuals(client) -> int:
    """Fill actual_c for elapsed target dates from historical METAR.

    The append path promised settle-time backfill but none existed, so
    actual_c stayed empty for every row (found 2026-10-21, live-loss
    audit). Fetches the settlement-station day high once per
    (city, target date) and rewrites the log with proper CSV quoting.
    Returns the number of rows filled; never raises.
    """
    try:
        rows = _read_forecast_log()
        if not rows:
            return 0
        pending: set[tuple[str, date]] = set()
        for r in rows:
            if r[9].strip():
                continue
            day = _parse_target_date(r[2], r[0])
            if day is None:
                continue
            tz = ZoneInfo(CITY_TZ.get(r[1], "UTC"))
            if day < datetime.now(tz).date():
                pending.add((r[1], day))
        actuals: dict[tuple[str, date], float] = {}
        for city, day in sorted(pending):
            v = await fetch_day_high_for_date(client, city, day)
            if v is not None:
                actuals[(city, day)] = v
        if not actuals:
            return 0
        filled = 0
        with FORECAST_LOG_PATH.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(_FORECAST_LOG_HEADER)
            for r in rows:
                day = _parse_target_date(r[2], r[0])
                if not r[9].strip() and day is not None and (r[1], day) in actuals:
                    r[9] = f"{actuals[(r[1], day)]:.1f}"
                    filled += 1
                w.writerow(r)
        return filled
    except Exception:
        return 0


class NoFadeStrategy(Strategy):
    """Buy NO on buckets the market overprices vs the forecast model."""

    name: str = "no_fade"

    def __init__(self, config: NoFadeConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config or NoFadeConfig()

    def bucket_model_p(self, fcst_c: float, bucket: TemperatureBucket, sigma_c: float | None = None) -> float:
        """P(bucket wins) under N(fcst + bias, sigma). Sigma defaults to EMOS floor."""
        mu = fcst_c + ERROR_BIAS_C
        s = sigma_c if sigma_c is not None else emos_sigma(None)
        lo, hi = bucket.temp_low, bucket.temp_high
        # Settlement is on the DISPLAYED (rounded) value, so a bucket
        # printed [lo, hi] covers the continuous range [lo-0.5, hi+0.5]
        # in native unit. Integrating over the raw printed bounds was a
        # live bug (found 2026-10-20): single-value C buckets got P=0
        # (zero width -> fake edge = full price; London 2026-10-19), and
        # F range buckets got half their true width (edge inflated).
        lo_n = lo - 0.5 if lo > -_TAIL_BOUND else lo
        hi_n = hi + 0.5 if hi < _TAIL_BOUND else hi
        if bucket.temp_unit == "F":
            lo_c = _to_c(lo_n) if lo > -_TAIL_BOUND else None
            hi_c = _to_c(hi_n) if hi < _TAIL_BOUND else None
        else:
            lo_c = lo_n if lo > -_TAIL_BOUND else None
            hi_c = hi_n if hi < _TAIL_BOUND else None
        if hi_c is None:
            return 1.0 - _norm_cdf((lo_c - mu) / s)
        if lo_c is None:
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
        spread_c = forecast.std if forecast.members else 0.0
        sigma_c = emos_sigma(spread_c)
        _append_forecast_log(event.city, event.date, fcst_c, spread_c, ERROR_BIAS_C, sigma_c, len(forecast.members), forecast.model)

        recs: list[Recommendation] = []
        for bucket in event.buckets:
            p_market = bucket.yes_price
            if not (0.01 < p_market < 0.99):
                continue
            p_no = 1.0 - p_market
            if not (self.cfg.min_no_price <= p_no <= self.cfg.max_no_price):
                continue
            p_model = self.bucket_model_p(fcst_c, bucket, sigma_c)
            edge = p_market - p_model
            if edge < self.cfg.edge_threshold:
                continue
            if edge > self.cfg.max_edge:
                # Edge this large = model error, not market error (2026-10-19 lesson:
                # London edge 55% lost). Skip.
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
