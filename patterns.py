"""
Additional pattern detection: RSI divergence, liquidity sweeps, order blocks.

Usage:
    from patterns import rsi_divergence, liquidity_sweep, order_blocks
    from analysis import to_df
    from mexc import get_ohlcv

    candles = get_ohlcv("ETHUSDT", "1h", 200)
    df = to_df(candles)
    divs   = rsi_divergence(df)
    sweeps = liquidity_sweep(df)
    obs    = order_blocks(df)
"""

import pandas as pd
import numpy as np
from analysis import rsi as calc_rsi, swing_points


# ─── RSI Divergence ───────────────────────────────────────────────────────────

def rsi_divergence(
    df: pd.DataFrame,
    lookback: int = 5,
    rsi_period: int = 14,
    scan_window: int = 50,
) -> list[dict]:
    """
    Detect RSI divergences in the last `scan_window` candles.

    Bullish divergence: price makes a lower low, RSI makes a higher low.
    Bearish divergence: price makes a higher high, RSI makes a lower high.

    Returns a list of divergence dicts:
        {type, price_level_a, price_level_b, rsi_a, rsi_b, time_a, time_b}
    """
    df = df.copy().tail(scan_window + lookback * 2 + rsi_period)
    df["rsi"] = calc_rsi(df, rsi_period)
    df = swing_points(df, lookback)

    divergences = []

    # ── Bullish: swing lows ────────────────────────────────────────────────────
    lows_idx = df["swing_low"].dropna().index
    lows_list = list(lows_idx)
    for i in range(1, len(lows_list)):
        t_a = lows_list[i - 1]
        t_b = lows_list[i]
        price_a = df.loc[t_a, "low"]
        price_b = df.loc[t_b, "low"]
        rsi_a   = df.loc[t_a, "rsi"]
        rsi_b   = df.loc[t_b, "rsi"]

        if pd.isna(rsi_a) or pd.isna(rsi_b):
            continue

        # Price: lower low  /  RSI: higher low  → bullish divergence
        if price_b < price_a and rsi_b > rsi_a:
            divergences.append({
                "type":          "bullish",
                "price_a":       round(price_a, 6),
                "price_b":       round(price_b, 6),
                "rsi_a":         round(rsi_a, 2),
                "rsi_b":         round(rsi_b, 2),
                "time_a":        t_a,
                "time_b":        t_b,
                "price_diff_pct": round((price_b - price_a) / price_a * 100, 3),
                "rsi_diff":      round(rsi_b - rsi_a, 2),
            })

    # ── Bearish: swing highs ───────────────────────────────────────────────────
    highs_idx = df["swing_high"].dropna().index
    highs_list = list(highs_idx)
    for i in range(1, len(highs_list)):
        t_a = highs_list[i - 1]
        t_b = highs_list[i]
        price_a = df.loc[t_a, "high"]
        price_b = df.loc[t_b, "high"]
        rsi_a   = df.loc[t_a, "rsi"]
        rsi_b   = df.loc[t_b, "rsi"]

        if pd.isna(rsi_a) or pd.isna(rsi_b):
            continue

        # Price: higher high  /  RSI: lower high  → bearish divergence
        if price_b > price_a and rsi_b < rsi_a:
            divergences.append({
                "type":          "bearish",
                "price_a":       round(price_a, 6),
                "price_b":       round(price_b, 6),
                "rsi_a":         round(rsi_a, 2),
                "rsi_b":         round(rsi_b, 2),
                "time_a":        t_a,
                "time_b":        t_b,
                "price_diff_pct": round((price_b - price_a) / price_a * 100, 3),
                "rsi_diff":      round(rsi_b - rsi_a, 2),
            })

    # Return most recent first
    divergences.sort(key=lambda x: x["time_b"], reverse=True)
    return divergences


# ─── Liquidity Sweeps ─────────────────────────────────────────────────────────

def liquidity_sweep(
    df: pd.DataFrame,
    lookback: int = 5,
    scan_window: int = 60,
) -> list[dict]:
    """
    Detect liquidity sweeps (stop hunts) in the last `scan_window` candles.

    Bullish sweep: candle wicks BELOW a prior swing low, but closes ABOVE it.
    Bearish sweep: candle wicks ABOVE a prior swing high, but closes BELOW it.

    Returns a list of sweep dicts (most recent first):
        {type, sweep_candle_time, swept_level, wick_extreme, close, wick_size_pct}
    """
    window = df.tail(scan_window + lookback * 2).copy()
    window = swing_points(window, lookback)

    opens  = window["open"].values
    closes = window["close"].values
    highs  = window["high"].values
    lows   = window["low"].values
    times  = window.index

    # Build rolling lists of prior swing highs/lows at each candle position
    swing_highs = window["swing_high"].values
    swing_lows  = window["swing_low"].values

    sweeps = []

    for i in range(lookback, len(window)):
        # Collect prior swing levels (not including current candle)
        prior_lows  = [v for v in swing_lows[:i]  if not np.isnan(v)]
        prior_highs = [v for v in swing_highs[:i] if not np.isnan(v)]

        if not prior_lows and not prior_highs:
            continue

        c_high  = highs[i]
        c_low   = lows[i]
        c_close = closes[i]

        # Bullish sweep: wick below prior swing low, closes back above it
        if prior_lows:
            nearest_low = max(l for l in prior_lows if l > c_low * 0.995)  if any(l > c_low * 0.995 for l in prior_lows) else None
            if nearest_low and c_low < nearest_low and c_close > nearest_low:
                wick_size_pct = round((nearest_low - c_low) / nearest_low * 100, 4)
                sweeps.append({
                    "type":             "bullish",
                    "time":             times[i],
                    "swept_level":      round(nearest_low, 6),
                    "wick_extreme":     round(c_low, 6),
                    "close":            round(c_close, 6),
                    "wick_size_pct":    wick_size_pct,
                })

        # Bearish sweep: wick above prior swing high, closes back below it
        if prior_highs:
            nearest_high = min(h for h in prior_highs if h < c_high * 1.005) if any(h < c_high * 1.005 for h in prior_highs) else None
            if nearest_high and c_high > nearest_high and c_close < nearest_high:
                wick_size_pct = round((c_high - nearest_high) / nearest_high * 100, 4)
                sweeps.append({
                    "type":             "bearish",
                    "time":             times[i],
                    "swept_level":      round(nearest_high, 6),
                    "wick_extreme":     round(c_high, 6),
                    "close":            round(c_close, 6),
                    "wick_size_pct":    wick_size_pct,
                })

    sweeps.sort(key=lambda x: x["time"], reverse=True)
    return sweeps


# ─── Order Blocks ─────────────────────────────────────────────────────────────

def order_blocks(
    df: pd.DataFrame,
    min_move_pct: float = 1.0,
    scan_window: int = 100,
) -> list[dict]:
    """
    Detect order blocks (OBs).

    Bullish OB:  last DOWN candle before a strong impulsive BULLISH move.
    Bearish OB:  last UP candle before a strong impulsive BEARISH move.

    A "strong move" is defined as a single candle body >= min_move_pct of its open price.

    Returns a list of OB dicts (most recent first):
        {type, time, top, bottom, body_size_pct, invalidated}
    invalidated=True if price has since closed through the OB zone.
    """
    window = df.tail(scan_window).copy()
    obs = []

    closes = window["close"].values
    opens  = window["open"].values
    highs  = window["high"].values
    lows   = window["low"].values
    times  = window.index

    current_price = closes[-1]

    for i in range(1, len(window) - 1):
        body_pct = abs(closes[i] - opens[i]) / opens[i] * 100

        if body_pct < min_move_pct:
            continue

        # Strong BULLISH candle at position i → look for last bearish candle before it
        if closes[i] > opens[i]:
            # Find the last bearish candle before i
            ob_idx = None
            for j in range(i - 1, max(i - 10, -1), -1):
                if closes[j] < opens[j]:
                    ob_idx = j
                    break
            if ob_idx is not None:
                ob_top    = max(opens[ob_idx], closes[ob_idx])
                ob_bottom = min(opens[ob_idx], closes[ob_idx])
                # Invalidated if price has since traded through and closed below OB bottom
                future_closes = closes[ob_idx + 1:]
                invalidated = bool(any(c < ob_bottom for c in future_closes))
                obs.append({
                    "type":          "bullish",
                    "time":          times[ob_idx],
                    "top":           round(ob_top, 6),
                    "bottom":        round(ob_bottom, 6),
                    "wick_high":     round(highs[ob_idx], 6),
                    "wick_low":      round(lows[ob_idx], 6),
                    "body_size_pct": round(body_pct, 3),
                    "dist_pct":      round((ob_top - current_price) / current_price * 100, 3),
                    "invalidated":   invalidated,
                })

        # Strong BEARISH candle at position i → look for last bullish candle before it
        elif closes[i] < opens[i]:
            ob_idx = None
            for j in range(i - 1, max(i - 10, -1), -1):
                if closes[j] > opens[j]:
                    ob_idx = j
                    break
            if ob_idx is not None:
                ob_top    = max(opens[ob_idx], closes[ob_idx])
                ob_bottom = min(opens[ob_idx], closes[ob_idx])
                future_closes = closes[ob_idx + 1:]
                invalidated = bool(any(c > ob_top for c in future_closes))
                obs.append({
                    "type":          "bearish",
                    "time":          times[ob_idx],
                    "top":           round(ob_top, 6),
                    "bottom":        round(ob_bottom, 6),
                    "wick_high":     round(highs[ob_idx], 6),
                    "wick_low":      round(lows[ob_idx], 6),
                    "body_size_pct": round(body_pct, 3),
                    "dist_pct":      round((ob_top - current_price) / current_price * 100, 3),
                    "invalidated":   invalidated,
                })

    # Deduplicate by time (keep last occurrence per time index)
    seen: set = set()
    unique_obs = []
    for ob in reversed(obs):
        key = (ob["time"], ob["type"])
        if key not in seen:
            seen.add(key)
            unique_obs.append(ob)

    unique_obs.sort(key=lambda x: x["time"], reverse=True)
    return unique_obs


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json
    from mexc import get_ohlcv
    from analysis import to_df

    symbol   = sys.argv[1].upper() if len(sys.argv) > 1 else "ETHUSDT"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1h"

    print(f"Pattern analysis: {symbol}  {interval}\n")
    candles = get_ohlcv(symbol, interval, 200)
    df = to_df(candles)

    divs   = rsi_divergence(df)
    sweeps = liquidity_sweep(df)
    obs    = order_blocks(df)

    print(f"RSI Divergences ({len(divs)}):")
    for d in divs[:5]:
        print(f"  {d['type']:<10}  price {d['price_a']} → {d['price_b']} ({d['price_diff_pct']:+.2f}%)  "
              f"RSI {d['rsi_a']} → {d['rsi_b']}  @ {d['time_b']}")

    print(f"\nLiquidity Sweeps ({len(sweeps)}):")
    for s in sweeps[:5]:
        print(f"  {s['type']:<10}  swept ${s['swept_level']}  wick={s['wick_size_pct']}%  @ {s['time']}")

    print(f"\nOrder Blocks ({len(obs)}):")
    valid_obs = [o for o in obs if not o["invalidated"]]
    for o in valid_obs[:5]:
        print(f"  {o['type']:<10}  zone ${o['bottom']}–${o['top']}  dist={o['dist_pct']:+.2f}%  @ {o['time']}")
