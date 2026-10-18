"""EMOS Forecast Edge Strategy — Optimized.

Compares EMOS-calibrated ensemble probabilities against Polymarket
market prices. Includes all optimizations:

1. Multi-model ensemble (GFS, ECMWF, ICON, GEM)
2. EMOS calibration (bias + variance correction)
3. Station-level bias correction (airport vs city center)
4. METAR real-time observation integration (final 6h)
5. Laddering strategy (multiple adjacent buckets)
6. Lead-time adaptive sizing (larger bets closer to resolution)
7. Dead-man switch (auto-halt on consecutive losses)
8. Bucket probability with underdispersion correction

Edge sources:
  - Model bias correction (trained a parameter)
  - Variance calibration (trained c,d parameters)
  - Multi-model consensus (4 models vs market crowd)
  - Airport microclimate difference (e.g., KLGA vs Manhattan)
  - METAR real-time observation (final hours)
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from pm_bot.core.weather import (
    bucket_probability,
    _get_emos_calibrator,
    _last_metar_obs,
    _hours_until_resolution,
)
from pm_bot.models.market import (
    ForecastResult,
    Recommendation,
    TemperatureBucket,
    WeatherEvent,
)
from pm_bot.strategies.base import Strategy


@dataclass
class EMOSEdgeConfig:
    """Configuration for EMOS edge strategy."""
    edge_threshold: float = 0.04       # Minimum edge to trade (4%)
    min_model_prob: float = 0.06       # Skip buckets with <6% model probability
    max_market_price: float = 0.90     # Don't buy above 90 cents
    min_market_price: float = 0.03     # Don't buy below 3 cents
    kelly_fraction: float = 0.25       # Quarter Kelly
    max_single_pct: float = 0.15       # Max 15% of bankroll per trade
    max_total_pct: float = 0.60        # Max 60% total exposure
    min_notional: float = 1.0          # Minimum $1 position
    use_emos: bool = True              # Use EMOS calibration

    # ── Laddering ──
    enable_laddering: bool = True      # Buy multiple adjacent buckets
    max_ladder_buckets: int = 4        # Max buckets to ladder
    min_ladder_edge: float = 0.03      # Min edge for ladder buckets

    # ── Lead-time adaptive sizing ──
    enable_lead_time_sizing: bool = True
    lead_time_multiplier_24h: float = 1.5   # 1.5x size when <24h to resolution
    lead_time_multiplier_6h: float = 2.0    # 2x size when <6h
    lead_time_multiplier_72h: float = 0.7   # 0.7x size when >72h

    # ── METAR boost ──
    enable_metar_boost: bool = True    # Use METAR observations
    metar_edge_boost: float = 0.02     # Extra edge when METAR confirms model
    metar_boost_hours: float = 6.0     # Apply METAR boost in final 6h

    # ── Dead-man switch ──
    enable_deadman: bool = True
    max_consecutive_losses: int = 3    # Halt after 3 consecutive losses
    halt_duration_s: int = 86400       # Halt for 24 hours
    daily_loss_limit_pct: float = 0.05 # Halt if daily loss > 5% of bankroll
    brier_score_threshold: float = 0.35  # Halt if 30-day Brier > 0.35

    # ── Tail bucket strategy ──
    enable_tail_strategy: bool = False  # Hans323 mode: bet on low-prob buckets
    tail_min_edge: float = 0.08        # Higher edge for tail buckets
    tail_max_price: float = 0.10       # Only buy below 10 cents
    tail_kelly_fraction: float = 0.10  # More conservative Kelly for tails


@dataclass
class DeadManState:
    """Dead-man switch state tracking."""
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    daily_bankroll_start: float = 0.0
    halted_until: float = 0.0  # Unix timestamp
    trade_outcomes: list[bool] = field(default_factory=list)  # Last 100 trades

    def is_halted(self) -> bool:
        """Check if trading should be halted."""
        return time.time() < self.halted_until

    def record_loss(self, halt_duration_s: int) -> None:
        """Record a loss and potentially halt."""
        self.consecutive_losses += 1
        if self.consecutive_losses >= 3:
            self.halted_until = time.time() + halt_duration_s

    def record_win(self) -> None:
        """Record a win, reset consecutive losses."""
        self.consecutive_losses = 0
        self.halted_until = 0.0

    def record_trade(self, won: bool) -> None:
        """Record trade outcome for Brier score tracking."""
        self.trade_outcomes.append(won)
        if len(self.trade_outcomes) > 100:
            self.trade_outcomes = self.trade_outcomes[-100:]

    def check_daily_loss(self, current_bankroll: float, limit_pct: float) -> bool:
        """Check if daily loss limit exceeded."""
        if self.daily_bankroll_start <= 0:
            return False
        loss_pct = (self.daily_bankroll_start - current_bankroll) / self.daily_bankroll_start
        return loss_pct > limit_pct

    def brier_score(self) -> float:
        """Compute Brier score from recent trades."""
        if len(self.trade_outcomes) < 10:
            return 0.0
        # Brier = mean((outcome - prediction)^2)
        # For binary outcomes: mean(outcome) when we bet, = 1 - win_rate
        wins = sum(1 for o in self.trade_outcomes if o)
        win_rate = wins / len(self.trade_outcomes)
        # Approximate Brier: if we bet on events with ~60% confidence
        # and our actual win rate is lower, Brier is high
        return 1.0 - win_rate  # Simplified


class EMOSEdgeStrategy(Strategy):
    """EMOS-based forecast edge strategy with all optimizations.

    Features:
    - Multi-model ensemble with EMOS calibration
    - Station-level bias correction (airport coordinates)
    - METAR real-time observation integration
    - Laddering strategy (multiple adjacent buckets)
    - Lead-time adaptive sizing
    - Dead-man switch (auto-halt on consecutive losses)
    - Tail bucket strategy (Hans323 mode)
    """

    name = "emos_edge"

    def __init__(
        self,
        edge_threshold: float = 0.04,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        max_single_pct: float = 0.15,
        min_notional: float = 1.0,
        max_position_usd: float = 50.0,
        config: EMOSEdgeConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            edge_threshold=edge_threshold,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            max_single_pct=max_single_pct,
            min_notional=min_notional,
            max_position_usd=max_position_usd,
            **kwargs,
        )
        self.cfg = config or EMOSEdgeConfig(
            edge_threshold=edge_threshold,
            kelly_fraction=kelly_fraction,
            max_single_pct=max_single_pct,
        )
        self.deadman = DeadManState()

    def run(
        self,
        event: WeatherEvent,
        forecast: ForecastResult | None = None,
        bankroll: float | None = None,
        emos_calibrator: Any | None = None,
        metar_temp_c: float | None = None,
        **kwargs,
    ) -> list[Recommendation]:
        """Generate trading recommendations for a weather event.

        Args:
            event: Weather event with market buckets
            forecast: Forecast result with ensemble members
            bankroll: Current bankroll
            emos_calibrator: Pre-loaded EMOS calibrator
            metar_temp_c: Current METAR observation (°C)

        Returns:
            List of trading recommendations (possibly laddered)
        """
        if not forecast or not forecast.members:
            return []

        effective_bankroll = bankroll or self.bankroll

        # ── Dead-man switch check ──
        if self.cfg.enable_deadman and self.deadman.is_halted():
            log.debug("deadman_halted", city=forecast.city)
            return []

        if self.cfg.enable_deadman and self.deadman.check_daily_loss(
            effective_bankroll, self.cfg.daily_loss_limit_pct
        ):
            log.warning("daily_loss_limit", city=forecast.city)
            return []

        # ── Brier score check ──
        if self.cfg.enable_deadman and self.deadman.brier_score() > self.cfg.brier_score_threshold:
            log.warning("brier_score_exceeded", score=self.deadman.brier_score())
            return []

        # ── Get EMOS calibrator ──
        calibrator = emos_calibrator
        if calibrator is None and self.cfg.use_emos:
            calibrator = _get_emos_calibrator(forecast.city)

        # ── Compute model probabilities for all buckets ──
        # NOTE (2026-10-18 fix): tail buckets used to be dropped here, so
        # _tail_strategy() never saw a real tail. They are now collected
        # separately with their own model probabilities.
        bucket_probs: list[tuple[TemperatureBucket, float]] = []
        tail_bucket_probs: list[tuple[TemperatureBucket, float]] = []
        for bucket in event.buckets:
            model_prob = bucket_probability(
                forecast,
                bucket.temp_low_c,
                bucket.temp_high_c,
                bucket.temp_unit,
                use_emos=self.cfg.use_emos,
            )
            if bucket.is_low_tail or bucket.is_high_tail:
                tail_bucket_probs.append((bucket, model_prob))
                continue
            if bucket.yes_price < self.cfg.min_market_price:
                continue
            if bucket.yes_price > self.cfg.max_market_price:
                continue

            if model_prob < self.cfg.min_model_prob:
                continue

            bucket_probs.append((bucket, model_prob))

        # ── Sort by edge (highest first) ──
        bucket_edges = []
        for bucket, model_prob in bucket_probs:
            edge = model_prob - bucket.yes_price

            # ── METAR boost ──
            if self.cfg.enable_metar_boost and metar_temp_c is not None:
                hours_left = _hours_until_resolution(event.date)
                if hours_left < self.cfg.metar_boost_hours:
                    # Check if METAR confirms model direction
                    forecast_mean = forecast.temp_high_c
                    metar_diff = abs(metar_temp_c - forecast_mean)
                    if metar_diff < 2.0:  # METAR close to forecast
                        edge += self.cfg.metar_edge_boost
                        log.debug(
                            "metar_boost",
                            city=forecast.city,
                            metar=f"{metar_temp_c:.1f}",
                            forecast=f"{forecast_mean:.1f}",
                            boost=f"+{self.cfg.metar_edge_boost:.2f}",
                        )

            bucket_edges.append((bucket, model_prob, edge))

        # Sort by edge descending
        bucket_edges.sort(key=lambda x: x[2], reverse=True)

        # ── Generate recommendations ──
        recs: list[Recommendation] = []

        if self.cfg.enable_laddering:
            recs = self._laddering_strategy(
                bucket_edges, event, forecast, effective_bankroll
            )
        else:
            recs = self._single_bucket_strategy(
                bucket_edges, event, forecast, effective_bankroll
            )

        # ── Tail bucket strategy (Hans323 mode) ──
        # Gets the REAL tail buckets (with model probs), not the filtered list.
        if self.cfg.enable_tail_strategy:
            tail_bucket_edges = [
                (b, p, p - b.yes_price) for b, p in tail_bucket_probs
            ]
            tail_recs = self._tail_strategy(
                tail_bucket_edges, event, forecast, effective_bankroll
            )
            recs.extend(tail_recs)

        return recs

    def _laddering_strategy(
        self,
        bucket_edges: list[tuple[TemperatureBucket, float, float]],
        event: WeatherEvent,
        forecast: ForecastResult,
        bankroll: float,
    ) -> list[Recommendation]:
        """Laddering: buy YES on multiple adjacent high-probability buckets.

        Instead of concentrating on the single most likely bucket,
        spread across the top 3-4 adjacent buckets. This reduces
        variance while retaining expected value.
        """
        recs: list[Recommendation] = []
        total_exposure = 0.0
        buckets_traded = 0

        # Filter to ladder-worthy buckets
        ladder_candidates = [
            (b, mp, e) for b, mp, e in bucket_edges
            if e >= self.cfg.min_ladder_edge
        ]

        # Find the most likely bucket and its neighbors
        if not ladder_candidates:
            return recs

        best_bucket, best_prob, best_edge = ladder_candidates[0]

        # Find adjacent buckets (within 2°C of the best)
        adjacent = []
        for bucket, prob, edge in ladder_candidates:
            if abs(bucket.temp_center_c - best_bucket.temp_center_c) <= 4.0:
                adjacent.append((bucket, prob, edge))

        # Limit to max_ladder_buckets
        adjacent = adjacent[:self.cfg.max_ladder_buckets]

        for bucket, model_prob, edge in adjacent:
            if edge < self.cfg.edge_threshold:
                continue

            # Lead-time adaptive sizing
            size_mult = self._get_lead_time_multiplier(event.date)

            # Kelly sizing
            payout = (1.0 - bucket.yes_price) / max(bucket.yes_price, 0.01)
            full_kelly = (model_prob * payout - (1.0 - model_prob)) / max(payout, 0.01)

            if full_kelly <= 0:
                continue

            kelly_pct = full_kelly * self.cfg.kelly_fraction * size_mult
            kelly_pct = min(kelly_pct, self.cfg.max_single_pct / len(adjacent))

            size_usd = kelly_pct * bankroll
            size_usd = min(size_usd, self.max_position_usd / len(adjacent))
            size_usd = max(size_usd, self.cfg.min_notional)

            # Check total exposure
            if total_exposure + size_usd > bankroll * self.cfg.max_total_pct:
                continue

            total_exposure += size_usd
            buckets_traded += 1

            reasoning = (
                f"LADDER #{buckets_traded} | "
                f"EMOS P={model_prob:.1%} vs market {bucket.yes_price:.1%} "
                f"edge={edge:+.1%} kelly={kelly_pct:.1%} "
                f"forecast={forecast.temp_high_c:.1f}°C ±{forecast.std:.1f}°C"
            )

            recs.append(Recommendation(
                strategy=self.name,
                event=event,
                bucket=bucket,
                direction="YES",
                edge=edge,
                reasoning=reasoning,
                size_usd=size_usd,
                kelly_fraction=kelly_pct,
            ))

        return recs

    def _single_bucket_strategy(
        self,
        bucket_edges: list[tuple[TemperatureBucket, float, float]],
        event: WeatherEvent,
        forecast: ForecastResult,
        bankroll: float,
    ) -> list[Recommendation]:
        """Single bucket: trade only the highest-edge bucket."""
        recs: list[Recommendation] = []

        for bucket, model_prob, edge in bucket_edges:
            if edge < self.cfg.edge_threshold:
                continue

            size_mult = self._get_lead_time_multiplier(event.date)

            payout = (1.0 - bucket.yes_price) / max(bucket.yes_price, 0.01)
            full_kelly = (model_prob * payout - (1.0 - model_prob)) / max(payout, 0.01)

            if full_kelly <= 0:
                continue

            kelly_pct = full_kelly * self.cfg.kelly_fraction * size_mult
            kelly_pct = min(kelly_pct, self.cfg.max_single_pct)

            size_usd = kelly_pct * bankroll
            size_usd = min(size_usd, self.max_position_usd)
            size_usd = max(size_usd, self.cfg.min_notional)

            total_exposure = sum(r.size_usd for r in recs)
            if total_exposure + size_usd > bankroll * self.cfg.max_total_pct:
                continue

            reasoning = (
                f"EMOS P={model_prob:.1%} vs market {bucket.yes_price:.1%} "
                f"edge={edge:+.1%} kelly={kelly_pct:.1%} "
                f"forecast={forecast.temp_high_c:.1f}°C ±{forecast.std:.1f}°C"
            )

            recs.append(Recommendation(
                strategy=self.name,
                event=event,
                bucket=bucket,
                direction="YES",
                edge=edge,
                reasoning=reasoning,
                size_usd=size_usd,
                kelly_fraction=kelly_pct,
            ))
            break  # Only one bucket in single mode

        return recs

    def _tail_strategy(
        self,
        bucket_edges: list[tuple[TemperatureBucket, float, float]],
        event: WeatherEvent,
        forecast: ForecastResult,
        bankroll: float,
    ) -> list[Recommendation]:
        """Tail bucket strategy (Hans323 mode).

        Buy low-probability, high-payout buckets when model gives
        them significantly more probability than the market.

        This is high-variance but can yield 10x+ returns.
        Only use with money you can afford to lose.
        """
        recs: list[Recommendation] = []

        for bucket, model_prob, edge in bucket_edges:
            # Only tail buckets: low market price, high model probability
            if bucket.yes_price > self.cfg.tail_max_price:
                continue
            if edge < self.cfg.tail_min_edge:
                continue

            # Conservative Kelly for tails
            payout = (1.0 - bucket.yes_price) / max(bucket.yes_price, 0.01)
            full_kelly = (model_prob * payout - (1.0 - model_prob)) / max(payout, 0.01)

            if full_kelly <= 0:
                continue

            kelly_pct = full_kelly * self.cfg.tail_kelly_fraction
            kelly_pct = min(kelly_pct, self.cfg.max_single_pct * 0.5)  # Half size for tails

            size_usd = kelly_pct * bankroll
            size_usd = min(size_usd, self.max_position_usd * 0.3)  # Cap at 30% of max
            size_usd = max(size_usd, self.cfg.min_notional)

            reasoning = (
                f"TAIL BET | EMOS P={model_prob:.1%} vs market {bucket.yes_price:.1%} "
                f"edge={edge:+.1%} payout={payout:.1f}x "
                f"kelly={kelly_pct:.1%}"
            )

            recs.append(Recommendation(
                strategy=self.name + "_tail",
                event=event,
                bucket=bucket,
                direction="YES",
                edge=edge,
                reasoning=reasoning,
                size_usd=size_usd,
                kelly_fraction=kelly_pct,
            ))

        return recs

    def _get_lead_time_multiplier(self, date: str) -> float:
        """Get position size multiplier based on lead time.

        Closer to resolution = more certainty = larger positions.
        """
        if not self.cfg.enable_lead_time_sizing:
            return 1.0

        hours_left = _hours_until_resolution(date)

        if hours_left < 6:
            return self.cfg.lead_time_multiplier_6h
        elif hours_left < 24:
            return self.cfg.lead_time_multiplier_24h
        elif hours_left > 72:
            return self.cfg.lead_time_multiplier_72h
        else:
            return 1.0

    def record_trade_outcome(self, won: bool, pnl: float) -> None:
        """Record trade outcome for dead-man switch tracking.

        Call this after each trade resolves.
        """
        if self.cfg.enable_deadman:
            if won:
                self.deadman.record_win()
            else:
                self.deadman.record_loss(self.cfg.halt_duration_s)
            self.deadman.record_trade(won)
            self.deadman.daily_pnl += pnl

    def reset_daily(self, bankroll: float) -> None:
        """Reset daily tracking. Call at start of each day."""
        self.deadman.daily_pnl = 0.0
        self.deadman.daily_bankroll_start = bankroll


def create_strategy(**kwargs) -> EMOSEdgeStrategy:
    """Factory function for the strategy registry."""
    return EMOSEdgeStrategy(**kwargs)
