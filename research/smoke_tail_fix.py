#!/usr/bin/env python3
"""Smoke test: tail plumbing fix — real tail buckets must reach _tail_strategy."""
import sys
sys.path.insert(0, "/home/hatch/workspace/pm-bot")

from pm_bot.strategies.emos_edge import EMOSEdgeStrategy, EMOSEdgeConfig
from pm_bot.models.market import ForecastResult, TemperatureBucket, WeatherEvent

cfg = EMOSEdgeConfig(enable_tail_strategy=True, use_emos=False)
strat = EMOSEdgeStrategy(config=cfg)

# members centered at 33°C, std ~2.4 → P(X≥35) ≈ 20%, market 5¢ → edge 15%
members = [33.0 + (i - 15) * 0.27 for i in range(31)]
forecast = ForecastResult(city="TestCity", date="2026-10-19", model="test",
                          temp_high_c=33.0, members=members)

def mkbucket(low, high, yes, no):
    return TemperatureBucket(market_id="m1", question="q", temp_low=low,
                             temp_high=high, temp_unit="C",
                             yes_price=yes, no_price=no, volume=100.0)

buckets = [
    mkbucket(28, 30, 0.40, 0.60),
    mkbucket(30, 32, 0.30, 0.70),
    # real high-tail open bucket, mispriced: model should see ~15%+, market 5¢
    mkbucket(35, 999.0, 0.05, 0.95),
]
event = WeatherEvent(event_id="test", title="t", slug="s", city="TestCity",
                     date="2026-10-19", measure_type="high", buckets=buckets)

recs = strat.run(event, forecast=forecast, bankroll=1000.0)
tail_recs = [r for r in recs if "tail" in r.strategy]
print(f"total recs: {len(recs)}, tail recs: {len(tail_recs)}")
for r in tail_recs:
    print(f"  {r.strategy}: {r.reasoning}")
assert tail_recs, "FAIL: tail strategy produced nothing for a mispriced real tail"
print("PASS: real tail bucket reached _tail_strategy")
