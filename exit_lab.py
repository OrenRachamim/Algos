#!/usr/bin/env python3
"""Exit-mechanism laboratory for the micro pullback strategy.

Detection (exit-independent) runs ONCE with the champion parameters; every
exit policy is then simulated over the identical event set, so differences
are attributable to the exit alone.

Exit policy fields (all optional on top of the base stop):
  target_r      fixed profit target in R (0 = none)
  max_hold      time exit after N days (close)
  be_at_r       move stop to entry after an excursion of +X R
  trail_pct     trail stop X% below the highest high since entry
  trail_atr     trail stop k*ATR(entry) below the highest high since entry
  trail_after_r activate the trail only after an excursion of +X R
  partial       (fraction, target_r): sell `fraction` at that target, the
                remainder continues under the other rules

Anti-lookahead: the stop level used on day k is the one computed through
day k-1's close; peaks/ratchets update after the day's stop check.
Conservative: stop checked before target on the same day.

Usage:  python exit_lab.py stage1|refine|portfolio
Results accumulate in exit_lab_results.json.
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
RESULTS = "exit_lab_results.json"

_DATA = None  # {ticker: (arrays, events)} shared via fork


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
        if len(df) < 300:
            continue
        df = mp.add_indicators(df)
        events = mp.detect_events(df, mp.DEFAULTS)
        if not events:
            continue
        arrays = dict(
            dates=np.array([str(d.date()) for d in df.index]),
            o=df["Open"].to_numpy(float), h=df["High"].to_numpy(float),
            lo=df["Low"].to_numpy(float), c=df["Close"].to_numpy(float),
            atr=df["atr14"].to_numpy(float),
        )
        out[t] = (arrays, events)
    _DATA = out
    n_ev = sum(len(e) for _, e in out.values())
    print(f"loaded {len(out)} tickers, {n_ev} events", flush=True)


COST = mp.DEFAULTS["trade_cost_bps"] / 1e4


def sim_exit(arr, ev, pol):
    """Simulate one trade under an exit policy. Returns trade dict or None."""
    n = len(arr["c"])
    trigger, init_stop, atr = ev["pb_high"], ev["pb_low"], ev["atr"]
    entry_i = None
    for k in range(ev["pb_end"] + 1,
                   min(ev["pb_end"] + 1 + mp.DEFAULTS["confirm_within"], n)):
        if arr["h"][k] > trigger:
            entry = max(arr["o"][k], trigger)
            entry_i = k
            break
        if arr["lo"][k] <= init_stop:
            return None
    if entry_i is None:
        return None
    risk = entry - init_stop
    if risk <= 0 or risk / entry < 1e-4:
        return None

    target_r = pol.get("target_r", 0)
    target = entry + target_r * risk if target_r > 0 else None
    max_hold = pol.get("max_hold", 25)
    part = pol.get("partial")  # (fraction, target_r)
    part_px = entry + part[1] * risk if part else None
    part_done_r = None

    stop = init_stop          # level in force for the CURRENT day
    peak = entry
    frac_open = 1.0
    exit_r, exit_i = None, None

    for k in range(entry_i, n):
        hi, lo, op, cl = arr["h"][k], arr["lo"][k], arr["o"][k], arr["c"][k]

        if lo <= stop:  # conservative: stop first
            px = min(stop, op)
            exit_r = (px - entry) / risk
            exit_i = k
            break
        if part and part_done_r is None and hi >= part_px:
            px = max(part_px, op) if k > entry_i else part_px
            part_done_r = (px - entry) / risk
        if target is not None and hi >= target:
            px = max(target, op) if k > entry_i else target
            exit_r = (px - entry) / risk
            exit_i = k
            break
        if k - entry_i + 1 >= max_hold:
            exit_r = (cl - entry) / risk
            exit_i = k
            break

        # ---- end of day: update peak and ratchet the stop for tomorrow
        peak = max(peak, hi)
        exc_r = (peak - entry) / risk  # best excursion so far
        lvl = stop
        if pol.get("be_at_r") and exc_r >= pol["be_at_r"]:
            lvl = max(lvl, entry)
        active = exc_r >= pol.get("trail_after_r", 0)
        if pol.get("trail_pct") and active:
            lvl = max(lvl, peak * (1 - pol["trail_pct"]))
        if pol.get("trail_atr") and active:
            lvl = max(lvl, peak - pol["trail_atr"] * atr)
        # adaptive trails: width interpolates start->end along a driver
        if pol.get("trail_profit"):  # (start_pct, end_pct, r_span)
            s, e, span = pol["trail_profit"]
            w = s + (e - s) * min(exc_r / span, 1.0)
            lvl = max(lvl, peak * (1 - w))
        if pol.get("trail_time"):    # (start_pct, end_pct) over max_hold
            s, e = pol["trail_time"]
            w = s + (e - s) * min((k - entry_i) / max_hold, 1.0)
            lvl = max(lvl, peak * (1 - w))
        if pol.get("trail_atr_roll"):  # k * rolling ATR below the peak
            lvl = max(lvl, peak - pol["trail_atr_roll"] * arr["atr"][k])
        if pol.get("ratchet_r") and exc_r >= pol["ratchet_r"]:
            steps = int(exc_r / pol["ratchet_r"])
            lvl = max(lvl, entry + (steps - 1) * pol["ratchet_r"] * risk)
        stop = lvl

    if exit_r is None:  # ran off the end of data - still open, skip
        return None

    if part and part_done_r is not None:
        f = part[0]
        total_r = f * part_done_r + (1 - f) * exit_r
    else:
        total_r = exit_r
    total_r -= COST * entry / risk
    return dict(entry_date=arr["dates"][entry_i], exit_date=arr["dates"][exit_i],
                outcome_r=round(float(total_r), 4),
                risk_pct=round(float(risk / entry), 5),
                days=int(exit_i - entry_i + 1))


def run_policy(item):
    name, pol = item
    trades = []
    for t, (arr, events) in _DATA.items():
        for ev in events:
            tr = sim_exit(arr, ev, pol)
            if tr:
                trades.append(tr)
    d = pd.DataFrame(trades)
    r = d.outcome_r.to_numpy()
    tr_m = (d.entry_date <= TRAIN_END).to_numpy()

    def block(mask):
        a = r[mask]
        return dict(n=int(len(a)), avg=round(float(a.mean()), 4),
                    win=round(float((a > 0).mean()), 3),
                    wins5=round(float(np.clip(a, -3, 5).mean()), 4),
                    days=round(float(d.days.to_numpy()[mask].mean()), 1))

    res = dict(name=name, policy=pol, train=block(tr_m), holdout=block(~tr_m))
    res["_trades"] = trades
    return res


def portfolio(trades, risk=0.0025, capital=100_000.0, max_expo=1.0):
    d = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    equity, invested = capital, 0.0
    open_pos, curve = [], []
    by_day = {k: g.index.tolist() for k, g in d.groupby("entry_date")}
    days = sorted(set(d.entry_date) | set(d.exit_date))
    for day in days:
        still = []
        for xd, pnl, pv in open_pos:
            if xd <= day:
                equity += pnl
                invested -= pv
            else:
                still.append((xd, pnl, pv))
        open_pos = still
        for i in by_day.get(day, []):
            row = d.iloc[i]
            rd = risk * equity
            pv = rd / max(row.risk_pct, 1e-4)
            if invested + pv > equity * max_expo:
                continue
            invested += pv
            open_pos.append((row.exit_date, row.outcome_r * rd, pv))
        curve.append(equity)
    for _, pnl, _ in open_pos:
        equity += pnl
    c = pd.Series(curve)
    yrs = (pd.to_datetime(days[-1]) - pd.to_datetime(days[0])).days / 365.25
    cagr = (equity / capital) ** (1 / yrs) - 1
    dd = float(((c.cummax() - c) / c.cummax()).max())
    return cagr, dd


STAGE1 = {
    "champ target3/h25":        dict(target_r=3.0, max_hold=25),
    "target2.5/h25":            dict(target_r=2.5, max_hold=25),
    "target3.5/h25":            dict(target_r=3.5, max_hold=25),
    "target3/h15":              dict(target_r=3.0, max_hold=15),
    "target3/h35":              dict(target_r=3.0, max_hold=35),
    "trail5% h60":              dict(trail_pct=0.05, max_hold=60),
    "trail8% h60":              dict(trail_pct=0.08, max_hold=60),
    "trail10% h60":             dict(trail_pct=0.10, max_hold=60),
    "trail12% h60":             dict(trail_pct=0.12, max_hold=60),
    "trail15% h60":             dict(trail_pct=0.15, max_hold=60),
    "trail8%+target3":          dict(trail_pct=0.08, target_r=3.0, max_hold=25),
    "trail10%+target3":         dict(trail_pct=0.10, target_r=3.0, max_hold=25),
    "chandelier2atr h60":       dict(trail_atr=2.0, max_hold=60),
    "chandelier3atr h60":       dict(trail_atr=3.0, max_hold=60),
    "chandelier4atr h60":       dict(trail_atr=4.0, max_hold=60),
    "BE@1R target3/h25":        dict(be_at_r=1.0, target_r=3.0, max_hold=25),
    "BE@1.5R target3/h25":      dict(be_at_r=1.5, target_r=3.0, max_hold=25),
    "BE@1R trail10%after2R t3": dict(be_at_r=1.0, trail_pct=0.10,
                                     trail_after_r=2.0, target_r=3.0, max_hold=25),
    "ratchet1R h40":            dict(ratchet_r=1.0, max_hold=40),
    "partial50@2R rest3R":      dict(partial=(0.5, 2.0), target_r=3.0, max_hold=25),
    "partial50@1.5R rest4R":    dict(partial=(0.5, 1.5), target_r=4.0, max_hold=25),
    "partial50@2R trail10%":    dict(partial=(0.5, 2.0), trail_pct=0.10, max_hold=40),
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    load()
    if mode == "stage1":
        battery = STAGE1
    else:
        with open("exit_lab_extra.json") as f:  # refine stages drop configs here
            battery = {k: v for k, v in json.load(f).items()}

    with Pool(min(8, os.cpu_count() or 4)) as pool:
        results = list(pool.imap_unordered(run_policy, battery.items()))

    # portfolio sim for every config (cheap enough at this event count)
    for res in results:
        cagr, dd = portfolio(res.pop("_trades"))
        res["cagr"] = round(float(cagr), 4)
        res["dd"] = round(float(dd), 4)

    old = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            old = json.load(f)
    seen = {r["name"] for r in results}
    merged = [r for r in old if r["name"] not in seen] + results
    with open(RESULTS, "w") as f:
        json.dump(merged, f, indent=1)

    merged.sort(key=lambda r: -r["holdout"]["avg"])
    print(f"\n{'name':26s} {'trainR':>7} {'holdR':>7} {'hWins5':>7} "
          f"{'win%':>5} {'days':>5} {'CAGR':>6} {'DD':>6}")
    for r in merged:
        t, h = r["train"], r["holdout"]
        print(f"{r['name']:26s} {t['avg']:+7.3f} {h['avg']:+7.3f} "
              f"{h['wins5']:+7.3f} {h['win']:5.0%} {h['days']:5.1f} "
              f"{r['cagr']:6.1%} {r['dd']:6.1%}")


if __name__ == "__main__":
    main()
