#!/usr/bin/env python3
"""Parameter sweep for the micro pullback strategy.

Honest protocol: configurations are ranked on TRAIN years only
(entries <= 2022-12-31); the top configs are then checked on the
untouched HOLDOUT years (2023+). Run after the price cache is populated
(e.g. `python micro_pullback.py backtest --tickers-file sp500.txt --period 10y`).

Usage:
  python sweep.py ofat      # stage A: one-factor-at-a-time around defaults
  python sweep.py combo     # stage B: combinations of per-axis winners
  python sweep.py show      # print ranked results collected so far
"""

import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import micro_pullback as mp  # noqa: E402

CACHE = "cache"
TICKERS_FILE = "sp500.txt"
TRAIN_END = "2022-12-31"
RESULTS = "sweep_results.json"

AXES = {
    "impulse_days": [8, 10, 15],
    "impulse_min_gain": [0.04, 0.05, 0.06, 0.08],
    "impulse_min_green": [0.50, 0.55, 0.60],
    "impulse_min_rvol": [0.9, 1.2, 1.5],
    "pullback_max_len": [3, 4, 6],
    "max_retrace": [0.45, 0.618, 0.80],
    "vol_contraction": [0.9, 1.1, 1.3],
    "confirm_within": [3, 5, 8],
    "trade_target_r": [1.5, 2.0, 3.0],
    "trade_max_hold": [10, 15, 25],
}

_DATA = None  # loaded once, shared with fork()ed workers


def load_data():
    global _DATA
    with open(TICKERS_FILE) as f:
        tickers = [ln.strip() for ln in f if ln.strip()]
    data = {}
    for t in tickers:
        path = os.path.join(CACHE, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if len(df) >= 300:
            data[t] = mp.add_indicators(df)
    _DATA = data
    print(f"loaded {len(data)} tickers into memory")


def run_config(overrides: dict) -> dict:
    p = dict(mp.DEFAULTS)
    p.update(overrides)
    tr_r, ho_r = [], []
    for name, df in _DATA.items():
        for ev in mp.detect_events(df, p):
            t = mp.simulate_trade(df, ev, p)
            if t and t["reason"] != "open":
                (tr_r if t["entry_date"] <= TRAIN_END else ho_r).append(t["outcome_r"])
    tr, ho = np.array(tr_r), np.array(ho_r)

    def stats(a):
        if len(a) < 30:
            return dict(n=int(len(a)), avg=np.nan, win=np.nan, pf=np.nan, tot=np.nan)
        wins, losses = a[a > 0], a[a <= 0]
        pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
        return dict(n=int(len(a)), avg=round(float(a.mean()), 4),
                    win=round(float((a > 0).mean()), 3), pf=round(pf, 3),
                    tot=round(float(a.sum()), 1))

    return dict(overrides=overrides, train=stats(tr), holdout=stats(ho))


def save(rows):
    old = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            old = json.load(f)
    seen = {json.dumps(r["overrides"], sort_keys=True) for r in old}
    old += [r for r in rows
            if json.dumps(r["overrides"], sort_keys=True) not in seen]
    with open(RESULTS, "w") as f:
        json.dump(old, f, indent=1)
    return old


def report(rows, min_train_n=600, top=25):
    ok = [r for r in rows if r["train"]["n"] >= min_train_n
          and not np.isnan(r["train"]["avg"])]
    ok.sort(key=lambda r: -r["train"]["avg"])
    print(f"\n{'train_avgR':>10} {'n':>6} {'pf':>6} | "
          f"{'hold_avgR':>9} {'n':>6} | overrides")
    for r in ok[:top]:
        t, ho = r["train"], r["holdout"]
        print(f"{t['avg']:>+10.3f} {t['n']:>6} {t['pf']:>6.2f} | "
              f"{ho['avg']:>+9.3f} {ho['n']:>6} | "
              f"{json.dumps(r['overrides'])}")
    return ok


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "ofat"
    if mode == "show":
        with open(RESULTS) as f:
            report(json.load(f))
        return

    load_data()
    if mode == "ofat":
        configs = [{}]  # baseline
        for k, vals in AXES.items():
            for v in vals:
                if v != mp.DEFAULTS[k]:
                    configs.append({k: v})
    elif mode == "combo":
        # combinations of the per-axis values whose OFAT train avg R beat
        # the baseline, capped via random sampling
        with open(RESULTS) as f:
            prev = json.load(f)
        base = next(r for r in prev if not r["overrides"])
        base_avg = base["train"]["avg"]
        good = {}
        for r in prev:
            if len(r["overrides"]) == 1 and r["train"]["n"] >= 600 \
                    and r["train"]["avg"] > base_avg:
                (k, v), = r["overrides"].items()
                good.setdefault(k, []).append(v)
        print("axes that beat baseline:", good)
        keys = sorted(good)
        pools = [[mp.DEFAULTS[k]] + good[k] for k in keys]
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*pools)]
        combos = [c for c in combos
                  if any(c[k] != mp.DEFAULTS[k] for k in keys)]
        rng = np.random.default_rng(1)
        if len(combos) > 60:
            combos = list(rng.choice(combos, size=60, replace=False))
        configs = combos
    else:
        sys.exit(f"unknown mode {mode}")

    print(f"running {len(configs)} configs...")
    with Pool(min(8, os.cpu_count() or 4)) as pool:
        rows = []
        for i, res in enumerate(pool.imap_unordered(run_config, configs)):
            rows.append(res)
            print(f"  [{i + 1}/{len(configs)}] train avgR "
                  f"{res['train']['avg']} n={res['train']['n']} "
                  f"{res['overrides']}", flush=True)
    allrows = save(rows)
    report(allrows)


if __name__ == "__main__":
    main()
