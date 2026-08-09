#!/usr/bin/env python3
"""Monkey benchmark: random entries with the exact same trade mechanics
(structure-scaled stop, 3R target, 25d time exit, same costs) on the same
universe. If the pattern has no edge, its expectancy should sit inside the
monkey distribution.

Two monkey species:
  A. uniform  - random ticker, random date (enjoys the same bull decade)
  B. date-matched - enters on the SAME dates as the strategy's trades,
     random ticker (controls for market timing / regime exposure)

Usage:  python monkey.py [n_runs=100]
Requires trades_best.json (the strategy's trade dataset) and the price cache.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import micro_pullback as mp  # noqa: E402

P = mp.DEFAULTS
COST = P["trade_cost_bps"] / 1e4


def load_universe():
    with open("sp500.txt") as f:
        tickers = [ln.strip() for ln in f if ln.strip()]
    data = {}
    for t in tickers:
        path = os.path.join("cache", f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if len(df) >= 300:
            data[t] = dict(dates=df.index.to_numpy(),
                           o=df["Open"].to_numpy(float),
                           h=df["High"].to_numpy(float),
                           lo=df["Low"].to_numpy(float),
                           c=df["Close"].to_numpy(float))
    return data


def sim(arr, k, risk_pct):
    """One monkey trade: enter at Open[k], stop risk_pct below, 3R target,
    time exit - identical resolution rules to simulate_trade."""
    entry = arr["o"][k]
    stop = entry * (1 - risk_pct)
    risk = entry - stop
    target = (entry + P["trade_target_r"] * risk
              if P["trade_target_r"] > 0 else None)
    n = len(arr["c"])
    for j in range(k, min(k + P["trade_max_hold"], n)):
        if arr["lo"][j] <= stop:
            exit_px = min(stop, arr["o"][j]) if j > k else stop
            break
        if target is not None and arr["h"][j] >= target:
            exit_px = max(target, arr["o"][j]) if j > k else target
            break
    else:
        j = min(k + P["trade_max_hold"], n) - 1
        exit_px = arr["c"][j]
    if j == n - 1 and exit_px == arr["c"][j] and n - k < P["trade_max_hold"]:
        return None  # would still be open - skip like the backtest does
    return (exit_px - entry) / risk - COST * entry / risk


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rng = np.random.default_rng(7)
    data = load_universe()
    names = list(data)

    strat = pd.DataFrame(json.load(open("trades_best.json")))
    n_trades = len(strat)
    risk_dist = strat["risk_pct"].to_numpy(float)
    strat_dates = pd.to_datetime(strat["entry_date"]).to_numpy()
    print(f"strategy: {n_trades} trades, avg R "
          f"{strat['outcome_r'].mean():+.3f}\n")

    # precompute date -> index maps for date-matched sampling
    date_pos = {t: pd.Series(np.arange(len(d["dates"])), index=d["dates"])
                for t, d in data.items()}

    for species in ("uniform", "date-matched"):
        means = []
        for run in range(n_runs):
            rs = []
            i = 0
            attempts = 0
            while len(rs) < n_trades and i < n_trades:
                t = names[rng.integers(len(names))]
                arr = data[t]
                if species == "uniform":
                    k = rng.integers(1, len(arr["c"]) - 2)
                else:
                    attempts += 1
                    if attempts > 50:  # no ticker has data for this date
                        i += 1
                        attempts = 0
                        continue
                    d = strat_dates[i]
                    pos = date_pos[t].index.searchsorted(d)
                    if pos < 1 or pos >= len(arr["c"]) - 2:
                        continue
                    k = pos
                r = sim(arr, int(k), float(rng.choice(risk_dist)))
                if r is not None:
                    rs.append(r)
                    i += 1
                    attempts = 0
            means.append(np.mean(rs))
            if (run + 1) % 20 == 0:
                print(f"  ...{species} {run + 1}/{n_runs}", flush=True)
        means = np.array(means)
        strat_avg = strat["outcome_r"].mean()
        pval = float((means >= strat_avg).mean())
        print(f"monkey ({species}, {n_runs} runs x {n_trades} trades):")
        print(f"  avg R: mean {means.mean():+.3f}  "
              f"[p5 {np.percentile(means, 5):+.3f} .. "
              f"p95 {np.percentile(means, 95):+.3f}]  best {means.max():+.3f}")
        print(f"  strategy {strat_avg:+.3f} -> beats "
              f"{(means < strat_avg).mean():.0%} of monkeys "
              f"(p-value {pval:.4f})\n")


if __name__ == "__main__":
    main()
