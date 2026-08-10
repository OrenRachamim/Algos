#!/usr/bin/env python3
"""Support/resistance laboratory for the micro pullback strategy.

Computes horizontal S/R features for every detected event - strictly from
data available at the pullback end (swing points need 3 later days to
confirm, so only swings confirmed by pb_end are used) - then measures
whether they separate winning trades from losers under the champion exit.

Features per event (zone = pullback low +- 0.5*ATR):
  sup_touches   days in the 126d window before the impulse whose LOW is in
                the zone - classic horizontal support strength
  res_touches   days in that window whose HIGH is in the zone - prior
                resistance now retested from above (flip zone)
  headroom_r    distance (in R) from the trigger to the nearest confirmed
                swing high above it in the past 250d; 10 = clear skies
  gap_support   1 if the impulse contains a gap-up whose zone overlaps the
                pullback low area
  round_dist    distance of the pullback low to the nearest $5/$10
                multiple, in ATR units

Usage:  python support_lab.py
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import micro_pullback as mp  # noqa: E402

TRAIN_END = "2022-12-31"
SWING_W = 3  # bars each side to confirm a swing point


def sr_features(df, arr, ev):
    h, lo = arr["h"], arr["lo"]
    pb_end = ev["pb_end"]
    leg_start = ev["leg_start"]
    pb_low, trigger, atr = ev["pb_low"], ev["pb_high"], ev["atr"]
    risk = max(trigger - pb_low, 1e-9)

    z_lo, z_hi = pb_low - 0.5 * atr, pb_low + 0.5 * atr
    w0 = max(0, leg_start - 126)
    win_lo = lo[w0:leg_start]
    win_hi = h[w0:leg_start]
    sup_touches = int(((win_lo >= z_lo) & (win_lo <= z_hi)).sum())
    res_touches = int(((win_hi >= z_lo) & (win_hi <= z_hi)).sum())

    # confirmed swing highs up to pb_end (need SWING_W later bars)
    last_ok = pb_end - SWING_W
    r0 = max(SWING_W, last_ok - 250)
    headroom_r = 10.0
    for i in range(r0, last_ok):
        if h[i] == h[i - SWING_W: i + SWING_W + 1].max() and h[i] > trigger:
            headroom_r = min(headroom_r, (h[i] - trigger) / risk)

    gap_support = 0
    for i in range(leg_start + 1, ev["leg_end"] + 1):
        if lo[i] > h[i - 1]:  # gap up; zone [h[i-1], lo[i]]
            if h[i - 1] <= z_hi and lo[i] >= z_lo - 0.5 * atr:
                gap_support = 1
                break

    step = 10.0 if pb_low >= 100 else 5.0
    round_dist = abs(pb_low - round(pb_low / step) * step) / max(atr, 1e-9)

    return dict(sup_touches=sup_touches, res_touches=res_touches,
                headroom_r=round(float(headroom_r), 2),
                gap_support=gap_support,
                round_dist=round(float(round_dist), 3))


_DATA = None


def load():
    global _DATA
    with open("sp500.txt") as f:
        tickers = [ln.strip() for ln in f if ln.strip()]
    out = {}
    for t in tickers:
        path = os.path.join("cache", f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if len(df) >= 300:
            out[t] = mp.add_indicators(df)
    _DATA = out
    print(f"loaded {len(out)} tickers", flush=True)


def one_ticker(t):
    df = _DATA[t]
    arr = dict(h=df["High"].to_numpy(float), lo=df["Low"].to_numpy(float))
    rows = []
    for ev in mp.detect_events(df, mp.DEFAULTS):
        tr = mp.simulate_trade(df, ev, mp.DEFAULTS)
        if not tr or tr["reason"] == "open":
            continue
        row = dict(ticker=t, entry_date=tr["entry_date"],
                   exit_date=tr["exit_date"], outcome_r=tr["outcome_r"],
                   risk_pct=round(tr["risk"] / tr["entry_px"], 5))
        row.update(sr_features(df, arr, ev))
        rows.append(row)
    return rows


def main():
    load()
    with Pool(min(8, os.cpu_count() or 4)) as pool:
        chunks = pool.map(one_ticker, list(_DATA))
    rows = [r for ch in chunks for r in ch]
    with open("trades_sr.json", "w") as f:
        json.dump(rows, f)
    d = pd.DataFrame(rows)
    tr = d[d.entry_date <= TRAIN_END]
    ho = d[d.entry_date > TRAIN_END]
    print(f"\ntrades: {len(d)}  train avg {tr.outcome_r.mean():+.3f} "
          f"(n={len(tr)})  holdout avg {ho.outcome_r.mean():+.3f} (n={len(ho)})")

    print("\n--- bucket analysis (train | holdout avg R) ---")
    specs = {
        "sup_touches": [0, 1, 3, 6, 100],
        "res_touches": [0, 1, 3, 6, 100],
        "headroom_r": [0, 1, 2, 3, 9.99, 100],
        "gap_support": [0, 1, 2],
        "round_dist": [0, 0.1, 0.3, 1, 100],
    }
    for feat, edges in specs.items():
        parts = []
        for a, b in zip(edges[:-1], edges[1:]):
            mt = (tr[feat] >= a) & (tr[feat] < b)
            mh = (ho[feat] >= a) & (ho[feat] < b)
            if mt.sum() < 100:
                parts.append(f"[{a}-{b}) n<100")
                continue
            parts.append(f"[{a}-{b}) {tr[mt].outcome_r.mean():+.3f}|"
                         f"{ho[mh].outcome_r.mean():+.3f} (n={mt.sum()}/{mh.sum()})")
        print(f"{feat:12s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
