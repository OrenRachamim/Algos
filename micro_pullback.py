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

# Defaults are the winning configuration from the 2016-2022 train /
# 2023+ holdout parameter sweep (see sweep.py): high-volume impulses
# (rvol >= 1.5) with a wide 3R target and patient 25d hold dominated
# every alternative on both train and holdout expectancy.
DEFAULTS = dict(
    impulse_days=10,        # lookback window for the impulse leg
    impulse_min_gain=0.04,  # leg must gain >= 4%
    impulse_min_green=0.50, # >= 50% of leg candles close up
    impulse_min_rvol=1.5,   # leg avg volume >= 1.5x the 20d avg volume before the leg
    pullback_min_len=1,
    pullback_max_len=4,
    max_retrace=0.618,      # pullback may retrace <= 61.8% of the leg
    vol_contraction=1.10,   # pullback avg volume <= 110% of impulse avg volume
    confirm_within=8,       # breakout must happen within N days to "confirm"
    label_horizon=10,       # days ahead used to label success for the model
    label_target_atr=1.5,   # success = +1.5 ATR above pullback high ...
    label_stop_atr=1.0,     # ... before -1.0 ATR below pullback low
    trade_target_r=3.0,     # backtest: profit target in R (0 = time exit only)
    trade_max_hold=25,      # backtest: max holding days after entry
    trade_cost_bps=10.0,    # backtest: round-trip cost+slippage in basis points
    stop_buffer_atr=0.0,    # extra stop distance below the pullback low, in ATR
    trade_trail_ema=0,      # exit on close < EMA(n) instead of waiting for target (0=off)
)

INT_PARAMS = {"impulse_days", "pullback_min_len", "pullback_max_len",
              "confirm_within", "label_horizon", "trade_max_hold",
              "trade_trail_ema"}


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
    if "hi252" in df.columns:  # already computed (idempotent)
        return df
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
    df["hi252"] = high.rolling(252, min_periods=60).max()
    df["ret63"] = close.pct_change(63)
    return df


# ---------------------------------------------------------------- detection


def detect_events(df: pd.DataFrame, p: dict = DEFAULTS, market=None):
    """Return a list of pattern events found in df.

    Each event dict has: leg_start/leg_end/pb_end indices, status
    ('confirmed' | 'failed' | 'active'), feature values, and (when the
    future window is available) a success label for model training.
    Pass `market` (from load_market) to add regime/RS features.
    """
    df = add_indicators(df)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    lo_ = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    ema10 = df["ema10"].to_numpy(float)
    ema20 = df["ema20"].to_numpy(float)
    sma50 = df["sma50"].to_numpy(float)
    vol20 = df["vol_sma20"].to_numpy(float)
    rsi = df["rsi14"].to_numpy(float)
    atr14 = df["atr14"].to_numpy(float)
    slope20 = df["slope20"].to_numpy(float)
    hi252 = df["hi252"].to_numpy(float)
    ret63 = df["ret63"].to_numpy(float)
    green = c > o
    body = np.abs(c - o)

    events = []
    n = len(df)
    L = p["impulse_days"]
    i = L + 50  # room for indicators to warm up

    while i < n - 1:
        ls = i - L  # leg start index; leg spans [ls, i]
        leg_gain = c[i] / c[ls] - 1
        green_ratio = green[ls: i + 1].mean()

        # impulse volume quality: leg volume relative to the 20d average
        # volume *before* the leg started (so the leg doesn't inflate its
        # own baseline)
        base_vol = vol20[ls]
        leg_vol = vol[ls: i + 1].mean()
        impulse_rvol = leg_vol / base_vol if base_vol and not np.isnan(base_vol) else 1.0

        trend_ok = (
            leg_gain >= p["impulse_min_gain"]
            and green_ratio >= p["impulse_min_green"]
            and impulse_rvol >= p["impulse_min_rvol"]
            and c[i] > ema20[i]
            and (np.isnan(sma50[i]) or c[i] > sma50[i])
        )
        if not trend_ok:
            i += 1
            continue

        leg_high = h[ls: i + 1].max()
        leg_low = lo_[ls: i + 1].min()
        leg_range = max(leg_high - leg_low, 1e-9)
        atr = atr14[i]

        # ---- walk forward through a candidate pullback: red or small
        # candles that don't make new highs
        j = i + 1
        while (
            j < n
            and j - (i + 1) < p["pullback_max_len"]
            and (not green[j] or body[j] < 0.5 * atr)
            and h[j] <= leg_high * 1.002
        ):
            j += 1
        pb_len = j - (i + 1)
        if pb_len < p["pullback_min_len"]:
            i += 1
            continue

        pb_s, pb_end = i + 1, j - 1
        pb_low = lo_[pb_s: j].min()
        pb_high = h[pb_s: j].max()
        retrace = (leg_high - pb_low) / leg_range
        pb_vol = vol[pb_s: j]
        pb_vol_ratio = pb_vol.mean() / max(leg_vol, 1)

        # volume trend inside the pullback: negative slope = sellers drying
        # up (bullish), positive = selling pressure building (bearish)
        if pb_len >= 2 and pb_vol.mean() > 0:
            pb_vol_slope = float(np.polyfit(np.arange(pb_len),
                                            pb_vol / pb_vol.mean(), 1)[0])
        else:
            pb_vol_slope = 0.0

        shallow_ok = retrace <= p["max_retrace"] and pb_low > ema20[i] * 0.985
        vol_ok = pb_vol_ratio <= p["vol_contraction"]
        if not (shallow_ok and vol_ok):
            i += 1
            continue

        features = dict(
            leg_gain=float(leg_gain),
            retrace=float(retrace),
            pb_len=float(pb_len),
            pb_vol_ratio=float(pb_vol_ratio),
            impulse_rvol=float(impulse_rvol),
            pb_vol_slope=float(pb_vol_slope),
            rsi14=float(rsi[pb_end]),
            dist_ema10=float(c[pb_end] / ema10[pb_end] - 1),
            dist_ema20=float(c[pb_end] / ema20[pb_end] - 1),
            slope20=float(slope20[pb_end]),
            atr_pct=float(atr / c[pb_end]),
            green_ratio=float(green_ratio),
            # where the last pullback candle closed within its range:
            # near the high = buyers stepping back in (bullish)
            pb_close_pos=float((c[pb_end] - lo_[pb_end])
                               / max(h[pb_end] - lo_[pb_end], 1e-9)),
            dist_52w=float(c[pb_end] / hi252[pb_end] - 1)
            if not np.isnan(hi252[pb_end]) else np.nan,
            rs_63=float(ret63[pb_end]) if not np.isnan(ret63[pb_end]) else np.nan,
        )
        features.update(market_features(market, df.index[pb_end]))
        if not np.isnan(features["rs_63"]) and not np.isnan(features.get("spy_ret63", np.nan)):
            features["rs_63"] -= features["spy_ret63"]  # relative strength vs SPY

        event = dict(
            leg_start=int(ls),
            leg_end=int(i),
            pb_end=int(pb_end),
            date_leg_end=str(df.index[i].date()),
            date_pb_end=str(df.index[pb_end].date()),
            leg_gain=round(float(leg_gain), 4),
            pb_len=pb_len,
            retrace=round(float(retrace), 3),
            pb_vol_ratio=round(float(pb_vol_ratio), 3),
            impulse_rvol=round(float(impulse_rvol), 2),
            pb_vol_slope=round(pb_vol_slope, 3),
            pb_high=round(float(pb_high), 4),
            pb_low=round(float(pb_low), 4),
            atr=round(float(atr), 4),
            features=features,
        )

        # ---- outcome: does a breakout above pb_high happen within confirm_within days?
        status, label = "active", None
        fw_start = pb_end + 1
        confirm_idx = None
        for k in range(fw_start, min(fw_start + p["confirm_within"], n)):
            if lo_[k] < pb_low - p["label_stop_atr"] * atr:
                status = "failed"
                break
            if c[k] > pb_high:
                status, confirm_idx = "confirmed", k
                break
        if status == "active" and fw_start + p["confirm_within"] <= n:
            status = "failed"  # window elapsed without breakout

        # ---- training label: after pullback end, +1.5 ATR before -1.0 ATR?
        target = pb_high + p["label_target_atr"] * atr
        stop = pb_low - p["label_stop_atr"] * atr
        if fw_start + p["label_horizon"] <= n:
            label = 0
            for k in range(fw_start, fw_start + p["label_horizon"]):
                if lo_[k] <= stop:
                    label = 0
                    break
                if h[k] >= target:
                    label = 1
                    break

        event["status"] = status
        event["confirm_date"] = str(df.index[confirm_idx].date()) if confirm_idx else None
        event["label"] = label
        events.append(event)
        i = pb_end + 1

    return events


FEATURE_NAMES = [
    "leg_gain", "retrace", "pb_len", "pb_vol_ratio", "impulse_rvol",
    "pb_vol_slope", "rsi14", "dist_ema10", "dist_ema20", "slope20",
    "atr_pct", "green_ratio", "pb_close_pos",
    # phase 3: stock context + market regime
    "dist_52w", "rs_63", "spy_trend200", "spy_above50", "spy_ret20",
    "vix_level", "vix_pctl",
]

MARKET_KEYS = ["spy_trend200", "spy_above50", "spy_ret20", "spy_ret63",
               "vix_level", "vix_pctl"]


def load_market(period: str, cache_dir: str = None):
    """Load SPY + VIX context series used for regime features."""
    try:
        spy = add_indicators_market(load_ticker("SPY", period, cache_dir))
        vix = load_ticker("^VIX", period, cache_dir)
        vix["pctl252"] = vix["Close"].rolling(252, min_periods=60).rank(pct=True)
        return {"spy": spy, "vix": vix}
    except Exception as e:
        print(f"  [warn] market data unavailable ({e}) - regime features NaN",
              file=sys.stderr)
        return None


def add_indicators_market(spy: pd.DataFrame) -> pd.DataFrame:
    spy = spy.copy()
    close = spy["Close"]
    spy["sma50"] = close.rolling(50).mean()
    spy["sma200"] = close.rolling(200).mean()
    spy["ret20"] = close.pct_change(20)
    spy["ret63"] = close.pct_change(63)
    return spy


def market_features(market, ts) -> dict:
    """Regime features as of timestamp ts (NaN when unavailable)."""
    out = {k: np.nan for k in MARKET_KEYS}
    if not market:
        return out
    spy, vix = market["spy"], market["vix"]
    i = spy.index.get_indexer([ts], method="pad")[0]
    if i >= 0:
        row = spy.iloc[i]
        if not np.isnan(row["sma200"]):
            out["spy_trend200"] = float(row["Close"] / row["sma200"] - 1)
        if not np.isnan(row["sma50"]):
            out["spy_above50"] = float(row["Close"] > row["sma50"])
        out["spy_ret20"] = float(row["ret20"])
        out["spy_ret63"] = float(row["ret63"])
    j = vix.index.get_indexer([ts], method="pad")[0]
    if j >= 0:
        out["vix_level"] = float(vix["Close"].iloc[j])
        out["vix_pctl"] = float(vix["pctl252"].iloc[j])
    return out


# ---------------------------------------------------------------- data I/O


PERIOD_DAYS = {"6mo": 126, "1y": 252, "2y": 504, "5y": 1260,
               "10y": 2520, "max": 10 ** 9}


def _slice_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    days = PERIOD_DAYS.get(period)
    if days is None:  # e.g. "3y" -> 3*252
        try:
            days = int(float(period.rstrip("y")) * 252)
        except ValueError:
            days = 10 ** 9
    return df.iloc[-days:]


def load_ticker(ticker: str, period: str, cache_dir: str = None) -> pd.DataFrame:
    """Load daily bars, with an optional on-disk cache.

    The cache always stores 10y of history; the requested period is sliced
    from it. A cached file is reused if its last bar is < 4 days old.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{ticker.replace('^', '_')}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            if len(df) and (pd.Timestamp.now() - df.index[-1]).days < 4:
                return _slice_period(df, period)
        df = _download(ticker, "10y")
        df.to_csv(path)
        return _slice_period(df, period)
    return _download(ticker, period)


_YF_BROKEN = False  # set after the first yfinance failure to skip it thereafter


def _download(ticker: str, period: str) -> pd.DataFrame:
    global _YF_BROKEN
    if not _YF_BROKEN:
        try:
            import logging

            import yfinance as yf

            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
            df = yf.download(ticker, period=period, interval="1d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df):
                return df
            _YF_BROKEN = True
        except Exception:
            _YF_BROKEN = True
    return _yahoo_chart_api(ticker, period)


def _yahoo_chart_api(ticker: str, period: str) -> pd.DataFrame:
    """Fallback: fetch daily bars straight from Yahoo's chart API.

    Works in environments where yfinance's curl_cffi transport fails
    (e.g. behind TLS-intercepting proxies).
    """
    import time

    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    for attempt in range(5):
        r = requests.get(
            url,
            params={"range": period, "interval": "1d", "events": "div,split"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=30,
        )
        if r.status_code == 429 and attempt < 4:  # rate limited - back off
            time.sleep(2 ** attempt)
            continue
        break
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    quote = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")

    df = pd.DataFrame(
        {"Open": quote["open"], "High": quote["high"], "Low": quote["low"],
         "Close": quote["close"], "Volume": quote["volume"]},
        index=pd.to_datetime(res["timestamp"], unit="s").normalize(),
    ).dropna()
    df.index.name = "Date"

    if adj is not None:  # adjust OHLC the same way yfinance auto_adjust does
        adj = pd.Series(adj, index=pd.to_datetime(res["timestamp"], unit="s").normalize())
        factor = (adj / df["Close"]).reindex(df.index)
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * factor
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def gather(tickers, period, csv, tickers_file=None, cache_dir=None):
    """Yield (name, df) pairs from tickers, a tickers file and/or CSV files.

    --csv accepts a single file, a comma separated list, or a directory
    (all *.csv files inside it are used).  --tickers-file is a text file
    with one ticker per line (# comments allowed).
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

    tickers = list(tickers)
    if tickers_file:
        with open(tickers_file) as f:
            tickers += [ln.strip() for ln in f
                        if ln.strip() and not ln.startswith("#")]
    for k, t in enumerate(tickers):
        try:
            df = load_ticker(t, period, cache_dir)
            if len(df) >= 120:
                yield t, df
            else:
                print(f"  [skip] {t}: only {len(df)} rows", file=sys.stderr)
        except Exception as e:
            print(f"  [skip] {t}: {e}", file=sys.stderr)
        if len(tickers) > 20 and (k + 1) % 50 == 0:
            print(f"  ...loaded {k + 1}/{len(tickers)} tickers", file=sys.stderr)


# ---------------------------------------------------------------- trading


def simulate_trade(df: pd.DataFrame, ev: dict, p: dict):
    """Simulate one trade from a detected event.

    Entry: first day the High crosses the pullback high (buy-stop at the
    trigger, or at the Open on a gap up).  Stop: pullback low.  Exit at
    +trade_target_r * R, at the stop, or at the Close after trade_max_hold
    days.  Round-trip costs are subtracted in R terms.  Intraday ambiguity
    (both stop and target touched) resolves to the stop - conservative.

    Returns None when the trigger never fires (or breaks down first).
    """
    n = len(df)
    trigger = ev["pb_high"]
    stop = ev["pb_low"] - p["stop_buffer_atr"] * ev.get("atr", 0.0)
    trail = None
    if p["trade_trail_ema"]:
        trail = df["Close"].ewm(span=p["trade_trail_ema"],
                                adjust=False).mean().to_numpy(float)
    entry_i = None
    for k in range(ev["pb_end"] + 1, min(ev["pb_end"] + 1 + p["confirm_within"], n)):
        row = df.iloc[k]
        if row["High"] > trigger:
            entry_px = max(row["Open"], trigger)
            entry_i = k
            break
        if row["Low"] <= stop:  # broke down before ever triggering
            return None
    if entry_i is None:
        return None

    risk = entry_px - stop
    if risk <= 0 or risk / entry_px < 1e-4:
        return None
    target = entry_px + p["trade_target_r"] * risk if p["trade_target_r"] > 0 else None

    exit_px, exit_i, reason = None, None, None
    for k in range(entry_i, n):
        row = df.iloc[k]
        if row["Low"] <= stop:
            exit_px = min(stop, row["Open"])  # gap through the stop fills lower
            exit_i, reason = k, "stop"
            break
        if target is not None and row["High"] >= target:
            exit_px = max(target, row["Open"]) if k > entry_i else target
            exit_i, reason = k, "target"
            break
        if trail is not None and k > entry_i and row["Close"] < trail[k]:
            exit_px, exit_i, reason = row["Close"], k, "time"
            break
        if k - entry_i + 1 >= p["trade_max_hold"]:
            exit_px, exit_i, reason = row["Close"], k, "time"
            break
    if exit_i is None:  # still open at end of data
        exit_px, exit_i, reason = df.iloc[-1]["Close"], n - 1, "open"

    cost_r = (p["trade_cost_bps"] / 1e4) * entry_px / risk
    return dict(
        entry_date=str(df.index[entry_i].date()),
        exit_date=str(df.index[exit_i].date()),
        entry_px=round(float(entry_px), 4),
        exit_px=round(float(exit_px), 4),
        risk=round(float(risk), 4),
        reason=reason,
        outcome_r=round(float((exit_px - entry_px) / risk - cost_r), 3),
    )


def backtest_stats(trades):
    """Aggregate closed trades into performance stats."""
    closed = [t for t in trades if t["reason"] != "open"]
    if not closed:
        return None
    r = np.array([t["outcome_r"] for t in closed])
    wins, losses = r[r > 0], r[r <= 0]
    order = np.argsort([t["entry_date"] for t in closed])
    equity = np.cumsum(r[order])
    dd = float((np.maximum.accumulate(equity) - equity).max())
    return dict(
        trades=len(closed),
        open_trades=len(trades) - len(closed),
        win_rate=float((r > 0).mean()),
        avg_r=float(r.mean()),
        median_r=float(np.median(r)),
        profit_factor=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        total_r=float(r.sum()),
        max_drawdown_r=dd,
        by_reason={k: int(sum(1 for t in closed if t["reason"] == k))
                   for k in ("target", "stop", "time")},
    )


def print_stats(stats, title):
    print(f"\n=== {title} ===")
    print(f"trades: {stats['trades']}   (open/excluded: {stats['open_trades']})")
    print(f"win rate: {stats['win_rate']:.1%}   avg R: {stats['avg_r']:+.3f}   "
          f"median R: {stats['median_r']:+.3f}")
    print(f"profit factor: {stats['profit_factor']:.2f}   total R: "
          f"{stats['total_r']:+.1f}   max DD: {stats['max_drawdown_r']:.1f}R")
    br = stats["by_reason"]
    print(f"exits: target {br['target']} / stop {br['stop']} / time {br['time']}")


# ---------------------------------------------------------------- commands


def cmd_backtest(args):
    p = build_params(args)
    trades = []
    n_events = 0
    for name, df in gather(args.tickers, args.period, args.csv,
                           args.tickers_file, args.cache_dir):
        for ev in detect_events(df, p):
            n_events += 1
            tr = simulate_trade(df, ev, p)
            if tr:
                tr["ticker"] = name
                trades.append(tr)

    if not trades:
        sys.exit("no trades simulated - relax parameters or add data")
    stats = backtest_stats(trades)
    print(f"\nevents detected: {n_events}, triggered trades: {len(trades)} "
          f"({len(trades) / n_events:.0%})")
    print_stats(stats, f"BACKTEST  target={p['trade_target_r']}R "
                       f"stop=1R hold<={p['trade_max_hold']}d "
                       f"costs={p['trade_cost_bps']}bps")

    years = {}
    for t in trades:
        if t["reason"] == "open":
            continue
        years.setdefault(t["entry_date"][:4], []).append(t["outcome_r"])
    print("\nyear   trades  win%   avgR    sumR")
    for y in sorted(years):
        r = np.array(years[y])
        print(f"{y}   {len(r):5d}  {(r > 0).mean():5.1%}  {r.mean():+.3f}  {r.sum():+7.1f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"stats": stats, "trades": trades}, f, indent=2)
        print(f"\nwrote {len(trades)} trades -> {args.json}")


def build_trade_dataset(args, p):
    """Detect events, simulate trades, return one row per closed trade
    (features + realized R). This is the training table for the model."""
    market = load_market(args.period, args.cache_dir or None)
    rows = []
    for name, df in gather(args.tickers, args.period, args.csv,
                           args.tickers_file, args.cache_dir):
        for ev in detect_events(df, p, market):
            tr = simulate_trade(df, ev, p)
            if tr and tr["reason"] != "open":
                row = dict(ticker=name, entry_date=tr["entry_date"],
                           exit_date=tr["exit_date"],
                           outcome_r=tr["outcome_r"], reason=tr["reason"],
                           risk_pct=round(tr["risk"] / tr["entry_px"], 5))
                row.update({f"f_{k}": ev["features"].get(k, np.nan)
                            for k in FEATURE_NAMES})
                rows.append(row)
    return rows


def _clean_X(X: np.ndarray) -> np.ndarray:
    """Impute NaNs with the column median (0 for all-NaN columns).

    sklearn's HistGradientBoosting crashes when a column is entirely NaN
    inside any internal CV fold / chronological slice, so we impute up
    front instead of relying on native NaN passthrough."""
    X = X.copy()
    med = np.nanmedian(np.where(np.isinf(X), np.nan, X), axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(med, idx[1])
    return X


class REnsemble:
    """Small seed-ensemble of gradient boosting regressors predicting the
    trade's R outcome (meta-labeling by expected value). Ranking by
    predicted R separates far better than a calibrated win/loss classifier
    on this data - probabilities compress around the base rate while R
    magnitudes keep the ordering information. Averaging across seeds
    stabilizes the ranking."""

    N_SEEDS = 5

    def __init__(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.models = [
            HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=3,
                min_samples_leaf=40, l2_regularization=1.0, random_state=s,
            )
            for s in range(self.N_SEEDS)
        ]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def _make_model():
    return REnsemble()


R_CLIP = 3.0  # clip training target to +-3R so outliers don't dominate

QUANTILE_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def _pick_threshold(pred, r, min_trades=100):
    """Choose the predicted-R cutoff (from quantiles of pred) that
    maximizes TOTAL R on (pred, r) - i.e. cut trades only when dropping
    them adds money. Returns (threshold, quantile); (-inf, None) means no
    cutoff beats taking every signal, which with a strong pattern
    configuration is often the honest answer."""
    best_thr, best_q, best_sum = -np.inf, None, r.sum()
    for q in QUANTILE_GRID:
        thr = np.quantile(pred, q)
        sel = pred >= thr
        if sel.sum() >= min_trades and r[sel].sum() > best_sum:
            best_sum, best_thr, best_q = r[sel].sum(), thr, q
    return best_thr, best_q


def cmd_evaluate(args):
    """Walk-forward evaluation: train on all years before Y, pick an
    EV-maximizing probability threshold on the train set, apply it
    out-of-sample on year Y. Reports whether model filtering beats the
    take-every-trade baseline."""
    p = build_params(args)

    if args.dataset and os.path.exists(args.dataset):
        with open(args.dataset) as f:
            rows = json.load(f)
        print(f"loaded {len(rows)} trades from {args.dataset}")
    else:
        rows = build_trade_dataset(args, p)
        if args.dataset:
            with open(args.dataset, "w") as f:
                json.dump(rows, f)
            print(f"built and saved {len(rows)} trades -> {args.dataset}")

    data = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
    if len(data) < 500:
        sys.exit(f"only {len(data)} trades - need a larger universe/history")

    X = _clean_X(data[[f"f_{k}" for k in FEATURE_NAMES]].to_numpy(float))
    r = data["outcome_r"].to_numpy(float)
    years = data["entry_date"].str[:4].to_numpy()
    dates = data["entry_date"].to_numpy()

    print(f"\ntrades: {len(data)}  years {years.min()}-{years.max()}  "
          f"baseline: win {(r > 0).mean():.1%}, avg R {r.mean():+.3f}")

    from scipy.stats import spearmanr

    folds = []
    for ty in sorted(set(years)):
        # purge: drop trades entered in Dec of the prior year (their exits
        # can overlap the test year -> leakage)
        train = (years < ty) & (dates < f"{int(ty) - 1}-12-01")
        test = years == ty
        if train.sum() < 400 or test.sum() < 30:
            continue
        model = _make_model()
        model.fit(X[train], np.clip(r[train], -R_CLIP, R_CLIP))
        thr, _ = _pick_threshold(model.predict(X[train]), r[train])

        pred = model.predict(X[test])
        sel = pred >= thr
        ic = spearmanr(pred, r[test]).statistic if test.sum() > 10 else np.nan
        folds.append(dict(
            year=ty, thr=None if thr == -np.inf else round(float(thr), 3),
            ic=round(float(ic), 3),
            base_n=int(test.sum()), base_avg=float(r[test].mean()),
            filt_n=int(sel.sum()),
            filt_avg=float(r[test][sel].mean()) if sel.any() else np.nan,
            filt_sum=float(r[test][sel].sum()),
            _sel_r=r[test][sel],
        ))

    if not folds:
        sys.exit("not enough history for walk-forward folds")

    print("\nyear    thr      IC   base_n baseAvgR   filt_n filtAvgR   filtSumR")
    for f in folds:
        thr = "  none" if f["thr"] is None else f"{f['thr']:+.3f}"
        print(f"{f['year']}  {thr}  {f['ic']:+.3f}   {f['base_n']:5d}  "
              f"{f['base_avg']:+.3f}    {f['filt_n']:5d}   {f['filt_avg']:+.3f}"
              f"   {f['filt_sum']:+8.1f}")

    oos_years = [f["year"] for f in folds]
    base_all = r[np.isin(years, oos_years)]
    filt_r = np.concatenate([f.pop("_sel_r") for f in folds])
    filt_avg = filt_r.mean() if len(filt_r) else np.nan
    improved = sum(1 for f in folds
                   if f["filt_n"] > 0 and f["filt_avg"] > f["base_avg"])
    print(f"\nOOS pooled:  baseline avg R {base_all.mean():+.3f} "
          f"({len(base_all)} trades)  |  filtered avg R {filt_avg:+.3f} "
          f"({len(filt_r)} trades)")
    print(f"years where filter beat baseline: {improved}/{len(folds)}")

    # bootstrap: how often would random subsets of the same size do as well?
    rng = np.random.default_rng(0)
    boot = np.array([
        rng.choice(base_all, size=len(filt_r), replace=True).mean()
        for _ in range(5000)
    ])
    pval = float((boot >= filt_avg).mean())
    print(f"bootstrap p-value (random subset >= filtered avg): {pval:.4f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(folds, f, indent=2)


def cmd_portfolio(args):
    """Portfolio simulation with real capital constraints: fixed fraction
    of current equity risked per trade, total exposure capped at 100% (no
    leverage), compounding, and - when signals exceed capacity - optional
    prioritization by walk-forward model score (each year's model trained
    only on prior years). Equity is marked on realized exits."""
    p = build_params(args)
    if args.dataset and os.path.exists(args.dataset):
        with open(args.dataset) as f:
            rows = json.load(f)
    else:
        rows = build_trade_dataset(args, p)
        if args.dataset:
            with open(args.dataset, "w") as f:
                json.dump(rows, f)
    data = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
    if "risk_pct" not in data.columns:
        sys.exit("dataset lacks risk_pct/exit_date - rebuild it (delete the file)")

    # walk-forward model scores for prioritization
    X = _clean_X(data[[f"f_{k}" for k in FEATURE_NAMES]].to_numpy(float))
    r = data["outcome_r"].to_numpy(float)
    years = data["entry_date"].str[:4].to_numpy()
    dates_arr = data["entry_date"].to_numpy()
    score = np.zeros(len(data))
    if not args.no_model:
        for ty in sorted(set(years)):
            train = (years < ty) & (dates_arr < f"{int(ty) - 1}-12-01")
            test = years == ty
            if train.sum() < 400:
                continue
            m = _make_model()
            m.fit(X[train], np.clip(r[train], -R_CLIP, R_CLIP))
            score[test] = m.predict(X[test])

    equity = args.capital
    invested = 0.0
    open_pos = []  # (exit_date, pnl_dollars, pos_value)
    curve = []
    skipped = taken = 0

    by_day = {d: g.index.tolist() for d, g in data.groupby("entry_date")}
    all_days = sorted(set(data["entry_date"]) | set(data["exit_date"]))
    for day in all_days:
        # close exits first - frees capital for the same day's entries
        still = []
        for exit_date, pnl, pv in open_pos:
            if exit_date <= day:
                equity += pnl
                invested -= pv
            else:
                still.append((exit_date, pnl, pv))
        open_pos = still

        idxs = by_day.get(day, [])
        idxs.sort(key=lambda i: -score[i])  # best expected R first
        for i in idxs:
            row = data.iloc[i]
            risk_d = args.risk * equity
            pos_val = risk_d / max(row["risk_pct"], 1e-4)
            if invested + pos_val > equity * args.max_exposure:
                skipped += 1
                continue
            invested += pos_val
            open_pos.append((row["exit_date"], row["outcome_r"] * risk_d, pos_val))
            taken += 1
        curve.append((day, equity))

    for _, pnl, _ in open_pos:  # flush anything still open
        equity += pnl
    curve.append((all_days[-1], equity))

    c = pd.Series(dict(curve))
    c.index = pd.to_datetime(c.index)
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    cagr = (equity / args.capital) ** (1 / yrs) - 1
    dd = float(((c.cummax() - c) / c.cummax()).max())
    label = "FIFO (no model)" if args.no_model else "model-prioritized"
    print(f"\n=== PORTFOLIO  {label}  risk {args.risk:.2%}/trade, "
          f"exposure<= {args.max_exposure:.0%}, start ${args.capital:,.0f} ===")
    print(f"trades taken: {taken}   skipped (no capacity): {skipped}")
    print(f"final equity: ${equity:,.0f}   CAGR: {cagr:+.1%}   "
          f"max drawdown: {dd:.1%}")
    yearly = c.resample("YE").last()
    prev = args.capital
    print("\nyear    equity        return")
    for ts, v in yearly.items():
        print(f"{ts.year}   ${v:>12,.0f}   {v / prev - 1:+8.1%}")
        prev = v
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"curve": {str(k.date()): round(float(v), 2)
                                 for k, v in c.items()},
                       "cagr": cagr, "max_dd": dd, "final": equity,
                       "taken": taken, "skipped": skipped}, f, indent=1)
        print(f"\nwrote equity curve -> {args.json}")


def cmd_universe(args):
    """Fetch the current S&P 500 ticker list from Wikipedia."""
    import io

    import requests

    r = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
    )
    r.raise_for_status()
    table = pd.read_html(io.StringIO(r.text))[0]
    symbols = sorted(s.replace(".", "-") for s in table["Symbol"])
    out = args.out or "sp500.txt"
    with open(out, "w") as f:
        f.write("\n".join(symbols) + "\n")
    print(f"wrote {len(symbols)} tickers -> {out}")


def cmd_scan(args):
    p = build_params(args)
    out = []
    for name, df in gather(args.tickers, args.period, args.csv,
                           args.tickers_file, args.cache_dir):
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
                  f"rvol={e['impulse_rvol']:.1f} vslope={e['pb_vol_slope']:+.2f} "
                  f"trigger>{e['pb_high']} stop<{e['pb_low']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {len(out)} events -> {args.json}")


def cmd_train(args):
    """Train the final meta-labeling model on realized trades and pick the
    EV-maximizing probability threshold; both are saved with the model."""
    from joblib import dump

    p = build_params(args)
    if args.dataset and os.path.exists(args.dataset):
        with open(args.dataset) as f:
            rows = json.load(f)
        print(f"loaded {len(rows)} trades from {args.dataset}")
    else:
        rows = build_trade_dataset(args, p)
        if args.dataset:
            with open(args.dataset, "w") as f:
                json.dump(rows, f)
    data = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
    if len(data) < 200:
        sys.exit(f"only {len(data)} trades - need more tickers/history")

    X = _clean_X(data[[f"f_{k}" for k in FEATURE_NAMES]].to_numpy(float))
    r = data["outcome_r"].to_numpy(float)

    # holdout report (chronological last 25%) before refitting on everything
    from scipy.stats import spearmanr

    split = int(len(r) * 0.75)
    hold = _make_model()
    hold.fit(X[:split], np.clip(r[:split], -R_CLIP, R_CLIP))
    pred_h = hold.predict(X[split:])
    ic = spearmanr(pred_h, r[split:]).statistic
    top = pred_h >= np.quantile(pred_h, 0.7)
    print(f"trades: {len(r)}  baseline win {(r > 0).mean():.1%}, "
          f"avg R {r.mean():+.3f}")
    print(f"holdout (last 25%): IC {ic:+.3f}, top-30% avg R "
          f"{r[split:][top].mean():+.3f} vs {r[split:].mean():+.3f} all")

    # pick the cutoff on the chronological validation tail (fit on the
    # first 75%, select on the last 25%) - an in-sample pick looks
    # brilliant and validates nothing
    _, q = _pick_threshold(pred_h, r[split:], min_trades=50)

    model = _make_model()
    model.fit(X, np.clip(r, -R_CLIP, R_CLIP))
    if q is None:
        thr = -np.inf
        print("no cutoff beats taking every trade on validation -> "
              "threshold disabled (all signals TAKE); expected R still "
              "ranks setups for prioritization")
    else:
        thr = float(np.quantile(model.predict(X), q))
        sel = model.predict(X) >= thr
        print(f"threshold (total-R max on validation, q={q}): predicted R "
              f">= {thr:+.3f} -> keeps {sel.mean():.0%} of trades")

    dump({"model": model, "features": FEATURE_NAMES, "params": p,
          "threshold": float(thr)}, MODEL_PATH)
    print(f"saved model (params + threshold) -> {MODEL_PATH}")


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

    threshold = bundle.get("threshold", 0.5)
    market = load_market(args.period, args.cache_dir or None)
    found = False
    for name, df in gather(args.tickers, args.period, args.csv,
                           args.tickers_file, args.cache_dir):
        events = [e for e in detect_events(df, p, market)
                  if e["status"] == "active"]
        for ev in events:
            x = np.array([[ev["features"].get(k, np.nan)
                           for k in bundle["features"]]], dtype=float)
            x = np.nan_to_num(x, nan=0.0)
            pred_r = model.predict(x)[0]
            verdict = "TAKE" if pred_r >= threshold else "skip"
            found = True
            print(f"{name}: ACTIVE pullback ended {ev['date_pb_end']} | "
                  f"expected R = {pred_r:+.2f} -> {verdict} "
                  f"(thr {threshold:+.2f})")
            print(f"   entry trigger > {ev['pb_high']}, stop < {ev['pb_low']}, "
                  f"retrace {ev['retrace']:.0%}, len {ev['pb_len']}d")
    if not found:
        print("no active (unconfirmed) micro pullbacks right now")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("universe", help="fetch S&P 500 ticker list")
    u.add_argument("--out", help="output file (default sp500.txt)")
    u.set_defaults(fn=cmd_universe)

    for cmd, fn in [("scan", cmd_scan), ("train", cmd_train),
                    ("predict", cmd_predict), ("backtest", cmd_backtest),
                    ("evaluate", cmd_evaluate), ("portfolio", cmd_portfolio)]:
        s = sub.add_parser(cmd)
        s.add_argument("--tickers", type=lambda v: v.split(",") if v else [],
                       default=[], help="comma separated tickers")
        s.add_argument("--tickers-file", help="text file, one ticker per line")
        s.add_argument("--period", default="2y", help="yfinance period (e.g. 2y, 5y)")
        s.add_argument("--csv", help="path to a Date,OHLCV csv file")
        s.add_argument("--cache-dir", default="cache",
                       help="on-disk price cache dir ('' disables)")
        s.add_argument("--json", help="scan/backtest/evaluate: write results to json")
        s.add_argument("--dataset", help="train/evaluate/portfolio: trade-dataset "
                                         "json (built if missing, reused if present)")
        s.add_argument("--capital", type=float, default=100_000,
                       help="portfolio: starting equity (default 100k)")
        s.add_argument("--risk", type=float, default=0.0025,
                       help="portfolio: fraction of equity risked per trade")
        s.add_argument("--max-exposure", type=float, default=1.0,
                       help="portfolio: max invested fraction of equity")
        s.add_argument("--no-model", action="store_true",
                       help="portfolio: first-come-first-served, no ranking")
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
