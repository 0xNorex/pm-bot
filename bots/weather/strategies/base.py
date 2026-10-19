from __future__ import annotations

from pm_bot.models.market import Recommendation, WeatherEvent, ForecastResult


class Strategy:
    """Base strategy for Polymarket weather trading.

    Principles:
    - Quarter Kelly (kelly_fraction=0.25) for safety
    - 8% minimum edge threshold for weather trades
    - Max 2% of bankroll per trade
    - Keep 30% cash reserve (max_total_pct=0.70)
    """

    name: str = "base"

    def __init__(
        self,
        edge_threshold: float = 0.08,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        max_single_pct: float = 0.02,
        min_notional: float = 1.0,
        max_position_usd: float = 2.0,
        **kwargs,
    ):
        self.edge_threshold = edge_threshold
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_single_pct = max_single_pct
        self.min_notional = min_notional
        self.max_position_usd = max_position_usd

    @property
    def supports_backtest(self) -> bool:
        return True

    def get_defaults(self) -> dict[str, float]:
        from pm_bot.models.config import STRATEGY_DEFAULTS

        return STRATEGY_DEFAULTS.get(self.name, {})

    def run(self, event: WeatherEvent, **kwargs) -> list[Recommendation]:
        return []


_all_strategies: dict[str, Strategy] | None = None


def get_all_strategies() -> dict[str, Strategy]:
    """All active strategies for Polymarket temperature markets.

    Currently active:
    - no_fade (S1): Buy NO on overpriced YES buckets. Validated 2026-10-19:
      30-day replay, 66 trades, 81.8% win, Sharpe 9.5. Paper trading live.

    Deleted 2026-05-16 (historical):
    - gopfan2, emos_gopfan2: pure tail buying, unprofitable
    - smart_wallet, adaptive_smart_wallet: no infrastructure
    - forecast_arb, emos_forecast_arb: model vs market, unprofitable
    - barbell, adaptive_barbell: tail+central, marginal

    See: .trellis/spec/backend/trading-config.md for strategy design history.
    """
    global _all_strategies
    if _all_strategies is None:
        _all_strategies = {}
        try:
            from pm_bot.strategies.no_fade import NoFadeStrategy
            _all_strategies["no_fade"] = NoFadeStrategy()
        except ImportError:
            pass
    return _all_strategies


ALL_STRATEGIES: dict[str, Strategy] = get_all_strategies()
