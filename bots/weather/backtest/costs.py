from __future__ import annotations

from dataclasses import dataclass

from pm_bot.core.clob import compute_v2_taker_fee


@dataclass
class FillModel:
    """Maker fill probability model for realistic backtesting.

    Weather market thin books mean limit orders don't always fill.
    This model applies Bernoulli sampling to simulate fill likelihood.
    """

    fill_prob_at_best: float = 0.50
    fill_prob_inside: float = 0.25
    # Calibrated for Polymarket's thin orderbook weather markets.
    # Tail buckets ($0.01-$0.15 and $0.85+) have very few resting orders,
    # so limit orders rarely fill — 10% fill rate reflects observed liquidity.
    fill_prob_tail: float = 0.10
    tail_low: float = 0.01
    tail_high: float = 0.15
    tail_high2: float = 0.85
    tail_very_high: float = 0.99

    def fill_probability(self, price: float) -> float:
        """Return fill probability for a given order price."""
        if price <= self.tail_low or price >= self.tail_very_high:
            return self.fill_prob_tail
        if price <= self.tail_high or price >= self.tail_high2:
            # Inside-spread tier (was returning fill_prob_tail, which made
            # fill_prob_inside unreachable dead config — fixed 2026-10-21).
            return self.fill_prob_inside
        return self.fill_prob_at_best


class CostModel:
    taker_fee_rate_bps: int = 50
    taker_fee_exponent: float = 0.5
    maker_fee_rate: float = 0.00
    default_spread_pct: float = 0.02
    default_slippage_pct: float = 0.01
    stop_loss_slippage_pct: float = 0.03
    tail_price_penalty_pct: float = 0.05
    tail_price_upper: float = 0.15
    tail_price_lower: float = 0.01
    ghost_trade_loss_pct: float = 0.02

    live_mode: bool = False
    live_max_position_usd: float = 10.0  # $10 max for small capital
    live_min_edge: float = 0.08          # 8% minimum edge (industry standard)
    live_side: str = "maker"

    forecast_penalty_pct: float = 0.05
    fill_model: FillModel = None  # type: ignore[assignment]

    def __init__(self) -> None:
        self.fill_model = FillModel()

    def _taker_fee_rate(self, price: float) -> float:
        return min(compute_v2_taker_fee(self.taker_fee_rate_bps, price, self.taker_fee_exponent), 0.0125)

    def calculate_cost(self, side: str, price: float, amount_usd: float) -> float:
        wager = price * amount_usd
        spread_cost = wager * self.default_spread_pct / 2.0
        if side == "taker":
            fee = self._taker_fee_rate(price) * wager
            slippage = wager * self.default_slippage_pct
            base = fee + spread_cost + slippage
        else:
            base = spread_cost

        if self.live_mode:
            base += wager * self.ghost_trade_loss_pct

        if self.tail_price_lower <= price <= self.tail_price_upper:
            base += wager * self.tail_price_penalty_pct

        return base

    def passes_live_filter(self, edge: float, notional: float) -> bool:
        if not self.live_mode:
            return True
        if edge < self.live_min_edge:
            return False
        if notional > self.live_max_position_usd:
            return False
        return True

    def net_edge(self, gross_edge: float, side: str, price: float, amount_usd: float) -> float:
        cost = self.calculate_cost(side, price, amount_usd)
        return gross_edge * amount_usd - cost

    def forecast_penalty_cost(self, price: float, amount_usd: float) -> float:
        """Additional cost when price source is forecast-derived (conservative penalty)."""
        return price * amount_usd * self.forecast_penalty_pct

    def stop_loss_slippage(self, wager: float) -> float:
        return wager * self.stop_loss_slippage_pct
