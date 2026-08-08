#!/usr/bin/env python3
"""Micro Pullback pattern scanner & predictor for daily OHLCV data.

Detects the pattern:  strong impulse leg up -> short shallow pullback
(1-4 small red/doji candles holding above support) -> breakout above the
pullback high.  A gradient-boosting model estimates the probability that
an *active* (unconfirmed) pullback resolves upward.

Usage:
  python micro_pullback.py scan    --tickers PAYX,MSFT,NVDA --period 2y
  python micro_pullback.py train   --tickers <basket>       --period 5y
  python micro_pullback.py predict --tickers PAYX           --period 2y
  python micro_pullback.py scan    --csv data/PAYX.csv

CSV format: Date,Open,High,Low,Close,Volume (header required).
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_model.joblib")

# ---------------------------------------------------------------- parameters

DEFAULTS = dict(
    impulse_days=10,        # lookback window for the impulse leg
    impulse_min_gain=0.05,  # leg must gain >= 5%
    impulse_min_green=0.55, # >= 55% of leg candles close up
    pullback_min_len=1,
    pullback_max_len=4,
    max_retrace=0.618,      # pullback may retrace <= 61.8% of the leg
    vol_contraction=1.10,   # pullback avg volume <= 110% of impulse avg volume
    confirm_within=5,       # breakout must happen within N days to "confirm"
    label_horizon=10,       # days ahead used to label success for the model
    label_target_atr=1.5,   # success = +1.5 ATR above pullback high ...
    label_stop_atr=1.0,     # ... before -1.0 ATR below pullback low
)

INT_PARAMS = {"impulse_days", "pullback_min_len", "pullback_max_len",
              "confirm_within", "label_horizon"}


def build_params(args) -> dict:
    """Merge pattern parameters: DEFAULTS < --config file < CLI flags."""
    p = dict(DEFAULTS)
    if getattr(args, "config", None):
        with open(args.config) as f:
            cfg = json.load(f)
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            sys.exit(f"unknown parameter(s) in {args.config}: {', '.join(sorted(unknown))}")
        p.update(cfg)
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            p[k] = v
    return p


def explicit_cli_params(args) -> dict:
    """Only the parameters the user set explicitly via CLI flags."""
    return {k: getattr(args, k) for k in DEFAULTS if getattr(args, k, None) is not None}

# ---------------------------------------------------------------- indicators


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    df["ema10"] = close.ewm(span=10, adjust=False).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["sma50"] = close.rolling(50).mean()
    df["vol_sma20"] = df["Volume"].rolling(20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    df["slope20"] = df["ema20"].pct_change(5)  # 5-day slope of EMA20
    return df


# ---------------------------------------------------------------- detection


def _is_red_or_small(row, atr):
    """Pullback candle: closes down, or a small-bodied candle (< 0.5 ATR)."""
    body = abs(row["Close"] - row["Open"])
    return row["Close"] < row["Open"] or body < 0.5 * atr


def detect_events(df: pd.DataFrame, p: dict = DEFAULTS):
    """Return a list of pattern events found in df.

    Each event dict has: leg_start/leg_end/pb_end indices, status
    ('confirmed' | 'failed' | 'active'), feature values, and (when the
    future window is available) a success label for model training.
    """
    df = add_indicators(df)
    events = []
    n = len(df)
    i = p["impulse_days"] + 50  # room for indicators to warm up

    while i < n - 1:
        leg = df.iloc[i - p["impulse_days"]: i + 1]
        leg_gain = leg["Close"].iloc[-1] / leg["Close"].iloc[0] - 1
        green_ratio = (leg["Close"] > leg["Open"]).mean()
        row = df.iloc[i]

        trend_ok = (
            leg_gain >= p["impulse_min_gain"]
            and green_ratio >= p["impulse_min_green"]
            and row["Close"] > row["ema20"]
            and (np.isnan(row["sma50"]) or row["Close"] > row["sma50"])
        )
        if not trend_ok:
            i += 1
            continue

        leg_high = leg["High"].max()
        leg_low = leg["Low"].min()
        leg_range = max(leg_high - leg_low, 1e-9)
        atr = row["atr14"]

        # ---- walk forward through a candidate pullback
        j = i + 1
        pb_rows = []
        while (
            j < n
            and len(pb_rows) < p["pullback_max_len"]
            and _is_red_or_small(df.iloc[j], atr)
            and df.iloc[j]["High"] <= leg_high * 1.002  # not making new highs
        ):
            pb_rows.append(j)
            j += 1

        if len(pb_rows) < p["pullback_min_len"]:
            i += 1
            continue

        pb = df.iloc[pb_rows]
        pb_low = pb["Low"].min()
        pb_high = pb["High"].max()
        retrace = (leg_high - pb_low) / leg_range
        impulse_vol = leg["Volume"].mean()
        pb_vol_ratio = pb["Volume"].mean() / max(impulse_vol, 1)

        shallow_ok = retrace <= p["max_retrace"] and pb_low > df.iloc[i]["ema20"] * 0.985
        vol_ok = pb_vol_ratio <= p["vol_contraction"]
        if not (shallow_ok and vol_ok):
            i += 1
            continue

        pb_end = pb_rows[-1]
        event = dict(
            leg_start=int(i - p["impulse_days"]),
            leg_end=int(i),
            pb_end=int(pb_end),
            date_leg_end=str(df.index[i].date()),
            date_pb_end=str(df.index[pb_end].date()),
            leg_gain=round(float(leg_gain), 4),
            pb_len=len(pb_rows),
            retrace=round(float(retrace), 3),
            pb_vol_ratio=round(float(pb_vol_ratio), 3),
            pb_high=round(float(pb_high), 2),
            pb_low=round(float(pb_low), 2),
        )
        event["features"] = extract_features(df, i, pb_rows, leg_gain, retrace,
                                             pb_vol_ratio, p)

        # ---- outcome: does a breakout above pb_high happen within confirm_within days?
        status, label = "active", None
        fw_start = pb_end + 1
        confirm_idx = None
        for k in range(fw_start, min(fw_start + p["confirm_within"], n)):
            if df.iloc[k]["Low"] < pb_low - p["label_stop_atr"] * atr:
                status = "failed"
                break
            if df.iloc[k]["Close"] > pb_high:
                status, confirm_idx = "confirmed", k
                break
        if status == "active" and fw_start + p["confirm_within"] <= n:
            status = "failed"  # window elapsed without breakout

        # ---- training label: after pullback end, +1.5 ATR before -1.0 ATR?
        target = pb_high + p["label_target_atr"] * atr
        stop = pb_low - p["label_stop_atr"] * atr
        horizon_end = min(pb_end + 1 + p["label_horizon"], n)
        if pb_end + 1 + p["label_horizon"] <= n:
            label = 0
            for k in range(pb_end + 1, horizon_end):
                if df.iloc[k]["Low"] <= stop:
                    label = 0
                    break
                if df.iloc[k]["High"] >= target:
                    label = 1
                    break

        event["status"] = status
        event["confirm_date"] = str(df.index[confirm_idx].date()) if confirm_idx else None
        event["label"] = label
        events.append(event)
        i = pb_end + 1

    return events


FEATURE_NAMES = [
    "leg_gain", "retrace", "pb_len", "pb_vol_ratio", "rsi14",
    "dist_ema10", "dist_ema20", "slope20", "atr_pct", "green_ratio",
]


def extract_features(df, leg_end_i, pb_rows, leg_gain, retrace, pb_vol_ratio, p):
    last = df.iloc[pb_rows[-1]]
    leg = df.iloc[leg_end_i - p["impulse_days"]: leg_end_i + 1]
    close = last["Close"]
    return dict(
        leg_gain=float(leg_gain),
        retrace=float(retrace),
        pb_len=float(len(pb_rows)),
        pb_vol_ratio=float(pb_vol_ratio),
        rsi14=float(last["rsi14"]),
        dist_ema10=float(close / last["ema10"] - 1),
        dist_ema20=float(close / last["ema20"] - 1),
        slope20=float(last["slope20"]),
        atr_pct=float(last["atr14"] / close),
        green_ratio=float((leg["Close"] > leg["Open"]).mean()),
    )


# ---------------------------------------------------------------- data I/O


def load_ticker(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def gather(tickers, period, csv):
    """Yield (name, df) pairs from tickers and/or CSV files.

    --csv accepts a single file, a comma separated list, or a directory
    (all *.csv files inside it are used).
    """
    paths = []
    if csv:
        for part in csv.split(","):
            if os.path.isdir(part):
                paths += sorted(
                    os.path.join(part, f) for f in os.listdir(part)
                    if f.endswith(".csv")
                )
            else:
                paths.append(part)
    for path in paths:
        yield os.path.splitext(os.path.basename(path))[0], load_csv(path)
    for t in tickers:
        try:
            df = load_ticker(t, period)
            if len(df) >= 120:
                yield t, df
            else:
                print(f"  [skip] {t}: only {len(df)} rows", file=sys.stderr)
        except Exception as e:
            print(f"  [skip] {t}: {e}", file=sys.stderr)


# ---------------------------------------------------------------- commands


def cmd_scan(args):
    p = build_params(args)
    out = []
    for name, df in gather(args.tickers, args.period, args.csv):
        events = detect_events(df, p)
        for ev in events:
            ev["ticker"] = name
        out.extend(events)
        conf = sum(1 for e in events if e["status"] == "confirmed")
        act = [e for e in events if e["status"] == "active"]
        print(f"{name}: {len(events)} pullback events "
              f"({conf} confirmed, {len(act)} active now)")
        for e in events[-args.show:]:
            print(f"  {e['date_pb_end']}  status={e['status']:9s} "
                  f"leg=+{e['leg_gain']:.1%} retrace={e['retrace']:.0%} "
                  f"len={e['pb_len']} vol={e['pb_vol_ratio']:.2f} "
                  f"trigger>{e['pb_high']} stop<{e['pb_low']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {len(out)} events -> {args.json}")


def _dataset(tickers, period, csv, p):
    X, y, meta = [], [], []
    for name, df in gather(tickers, period, csv):
        for ev in detect_events(df, p):
            if ev["label"] is None:
                continue
            X.append([ev["features"][k] for k in FEATURE_NAMES])
            y.append(ev["label"])
            meta.append((name, ev["date_pb_end"]))
    return np.array(X), np.array(y), meta


def cmd_train(args):
    from joblib import dump
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_score

    p = build_params(args)
    X, y, meta = _dataset(args.tickers, args.period, args.csv, p)
    if len(y) < 40:
        sys.exit(f"only {len(y)} labeled events - need more tickers/history")

    order = np.argsort([m[1] for m in meta])  # chronological split
    X, y = X[order], y[order]
    split = int(len(y) * 0.75)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=42,
    )
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
    picked = proba >= 0.60
    prec = precision_score(yte, picked, zero_division=0) if picked.any() else float("nan")

    print(f"dataset: {len(y)} events, base success rate {y.mean():.1%}")
    print(f"test AUC: {auc:.3f}")
    print(f"precision @ p>=0.60: {prec:.1%} ({picked.sum()} signals in test set)")
    print("\nfeature importance:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                            key=lambda t: -t[1]):
        print(f"  {name:14s} {imp:.3f}")

    dump({"model": model, "features": FEATURE_NAMES, "params": p}, MODEL_PATH)
    print(f"\nsaved model (with pattern params) -> {MODEL_PATH}")


def cmd_predict(args):
    from joblib import load

    if not os.path.exists(MODEL_PATH):
        sys.exit("no trained model found - run `train` first")
    bundle = load(MODEL_PATH)
    model = bundle["model"]

    # detect with the exact params the model was trained on, unless the
    # user explicitly overrides via --config / CLI flags
    p = dict(bundle.get("params", DEFAULTS))
    if getattr(args, "config", None):
        p = build_params(args)
    else:
        overrides = explicit_cli_params(args)
        if overrides:
            print(f"note: overriding trained params: {overrides}", file=sys.stderr)
            p.update(overrides)

    found = False
    for name, df in gather(args.tickers, args.period, args.csv):
        events = [e for e in detect_events(df, p) if e["status"] == "active"]
        for ev in events:
            x = np.array([[ev["features"][k] for k in bundle["features"]]])
            p = model.predict_proba(x)[0, 1]
            found = True
            print(f"{name}: ACTIVE pullback ended {ev['date_pb_end']} | "
                  f"P(upward resolution) = {p:.1%}")
            print(f"   entry trigger > {ev['pb_high']}, stop < {ev['pb_low']}, "
                  f"retrace {ev['retrace']:.0%}, len {ev['pb_len']}d")
    if not found:
        print("no active (unconfirmed) micro pullbacks right now")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, fn in [("scan", cmd_scan), ("train", cmd_train), ("predict", cmd_predict)]:
        s = sub.add_parser(cmd)
        s.add_argument("--tickers", type=lambda v: v.split(",") if v else [],
                       default=[], help="comma separated tickers")
        s.add_argument("--period", default="2y", help="yfinance period (e.g. 2y, 5y)")
        s.add_argument("--csv", help="path to a Date,OHLCV csv file")
        s.add_argument("--json", help="scan only: write events to this json file")
        s.add_argument("--show", type=int, default=5,
                       help="scan only: how many recent events to print")
        s.add_argument("--config", help="json file with pattern parameters")
        pat = s.add_argument_group("pattern parameters (override config/defaults)")
        for k, v in DEFAULTS.items():
            pat.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None,
                             type=int if k in INT_PARAMS else float,
                             help=f"default: {v}")
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
