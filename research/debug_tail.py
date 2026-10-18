#!/usr/bin/env python3
"""Debug: why do high-tail signals fire? Print model_prob vs market_price
for sample tail signals."""
from __future__ import annotations

import importlib.util
import random
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

import numpy as np

ROOT = Path("/home/hatch/workspace/pm-bot")
stub_pkg = types.ModuleType("backtest")
stub_pkg.__path__ = [str(ROOT / "bots" / "weather" / "backtest")]
sys.modules["backtest"] = stub_pkg
spec = importlib.util.spec_from_file_location(
    "backtest.weather_strategies", ROOT / "bots" / "weather" / "backtest" / "weather_strategies.py")
ws = importlib.util.module_from_spec(spec)
sys.modules["backtest.weather_strategies"] = ws
spec.loader.exec_module(ws)

rng = random.Random(42)
np.random.seed(42)
coeffs = ws.load_emos_coeffs()
print("coeff cities:", len(coeffs), "| New York coeffs:", coeffs.get("New York"))

# replicate one city-day of generate_forecast
city_cfg = {"temp_range": (-5, 38), "season_amp": 15, "station_bias": 1.5}
date = datetime.now()
doy = date.timetuple().tm_yday
mid = (-5 + 38) / 2.0
true_temp = mid + 15 * np.sin(2 * np.pi * doy / 365.0 - np.pi / 2) + rng.gauss(0, 3.0)
true_station = true_temp - 1.5
members = [true_station + rng.gauss(0, 0.3) + rng.gauss(0, 2.0) for _ in range(31)]

mu, sigma = ws.emos_calibrate(members, coeffs.get("New York", {}))
raw_std = float(np.std(members))
print(f"true_temp={true_temp:.2f} raw_mean={np.mean(members):.2f} raw_std={raw_std:.2f}")
print(f"EMOS mu={mu:.2f} sigma={sigma:.2f}")

thresh = float(np.floor(np.mean(members) + 1.2 * raw_std))
p = ws.emos_bucket_prob(mu, sigma, thresh, 999.0)
print(f"tail thresh={thresh:.0f}  model_prob={p:.3f}  (min_model_prob filter = 0.08)")
