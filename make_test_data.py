#!/usr/bin/env python3
"""Generate synthetic daily OHLCV csv files containing engineered
impulse -> micro-pullback -> (breakout | breakdown) sequences, for
testing micro_pullback.py offline."""

import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def make_series(seed: int, days: int = 900, start_price: float = 100.0):
    rng = np.random.default_rng(seed)
    prices, vols = [], []
    price = start_price
    base_vol = 1_000_000
    i = 0
    while i < days:
        regime = rng.choice(["drift", "impulse"], p=[0.6, 0.4])
        if regime == "drift":
            for _ in range(rng.integers(8, 20)):
                price *= 1 + rng.normal(0.0002, 0.010)
                prices.append(price)
                vols.append(base_vol * rng.uniform(0.7, 1.3))
                i += 1
        else:
            # impulse leg: 8-12 strong up days on rising volume
            for _ in range(rng.integers(8, 13)):
                price *= 1 + abs(rng.normal(0.012, 0.006))
                prices.append(price)
                vols.append(base_vol * rng.uniform(1.3, 2.2))
                i += 1
            # micro pullback: 1-4 small down days on lighter volume
            pb_len = rng.integers(1, 5)
            for _ in range(pb_len):
                price *= 1 - abs(rng.normal(0.004, 0.003))
                prices.append(price)
                vols.append(base_vol * rng.uniform(0.5, 0.9))
                i += 1
            # resolution: 70% breakout up, 30% breakdown
            if rng.random() < 0.7:
                for _ in range(rng.integers(3, 7)):
                    price *= 1 + abs(rng.normal(0.010, 0.005))
                    prices.append(price)
                    vols.append(base_vol * rng.uniform(1.2, 1.9))
                    i += 1
            else:
                for _ in range(rng.integers(3, 7)):
                    price *= 1 - abs(rng.normal(0.012, 0.006))
                    prices.append(price)
                    vols.append(base_vol * rng.uniform(1.0, 1.6))
                    i += 1

    prices = np.array(prices[:days])
    vols = np.array(vols[:days])
    n = len(prices)

    opens = prices * (1 + rng.normal(0, 0.003, n))
    highs = np.maximum(opens, prices) * (1 + abs(rng.normal(0, 0.004, n)))
    lows = np.minimum(opens, prices) * (1 - abs(rng.normal(0, 0.004, n)))
    dates = pd.bdate_range(end="2026-08-07", periods=n)
    return pd.DataFrame(
        {"Open": opens.round(2), "High": highs.round(2), "Low": lows.round(2),
         "Close": prices.round(2), "Volume": vols.astype(int)},
        index=pd.Index(dates, name="Date"),
    )


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    os.makedirs(OUT_DIR, exist_ok=True)
    for k in range(n_files):
        df = make_series(seed=k)
        path = os.path.join(OUT_DIR, f"SYN{k:02d}.csv")
        df.to_csv(path)
        print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
