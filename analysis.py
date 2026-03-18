"""
Trading analysis models — EMA, RSI, structure, FVGs, key levels
"""

import pandas as pd
import numpy as np
from typing import Optional


def to_df(candles: list[dict]) -> pd.DataFrame:
    """Convert candle list to DataFrame."""
    df = pd.DataFrame(candles)
    df.set_index("time", inplace=True)
    return df


# ─── Indicators ───────────────────────────────────────────────────────────────

def ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False).mean()


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(period).mean()


# ─── Market Structure ─────────────────────────────────────────────────────────

def swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Identify swing highs and lows."""
    df = df.copy()
    df["swing_high"] = df["high"][(df["high"] == df["high"].rolling(lookback * 2 + 1, center=True).max())]
    df["swing_low"]  = df["low"][(df["low"]  == df["low"].rolling(lookback * 2 + 1, center=True).min())]
    return df


def market_structure(df: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Detect trend via HH/HL or LH/LL.
    Returns: trend (bullish/bearish/ranging), last BOS level, last CHoCH level
    """
    df = swing_points(df, lookback)
    highs = df["swing_high"].dropna()
    lows  = df["swing_low"].dropna()

    if len(highs) < 2 or len(lows) < 2:
        return {"trend": "unknown", "bos": None, "choch": None}

    last_hh = highs.iloc[-1] > highs.iloc[-2]
    last_hl = lows.iloc[-1]  > lows.iloc[-2]
    last_lh = highs.iloc[-1] < highs.iloc[-2]
    last_ll = lows.iloc[-1]  < lows.iloc[-2]

    if last_hh and last_hl:
        trend = "bullish"
    elif last_lh and last_ll:
        trend = "bearish"
    else:
        trend = "ranging"

    return {
        "trend":      trend,
        "last_high":  round(highs.iloc[-1], 4),
        "last_low":   round(lows.iloc[-1], 4),
        "prev_high":  round(highs.iloc[-2], 4),
        "prev_low":   round(lows.iloc[-2], 4),
    }


# ─── Fair Value Gaps ──────────────────────────────────────────────────────────

def find_fvgs(df: pd.DataFrame, min_gap_pct: float = 0.1) -> list[dict]:
    """
    Find Fair Value Gaps (FVGs).
    Bullish FVG: candle[i-1].high < candle[i+1].low
    Bearish FVG: candle[i-1].low  > candle[i+1].high
    """
    fvgs = []
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    times  = df.index

    for i in range(1, len(df) - 1):
        mid_price = closes[i]

        # Bearish FVG (price likely to fill from below)
        if lows[i - 1] > highs[i + 1]:
            gap_size = (lows[i - 1] - highs[i + 1]) / mid_price * 100
            if gap_size >= min_gap_pct:
                fvgs.append({
                    "type":   "bearish",
                    "top":    round(lows[i - 1], 4),
                    "bottom": round(highs[i + 1], 4),
                    "time":   times[i],
                    "gap_pct": round(gap_size, 3),
                })

        # Bullish FVG
        if highs[i - 1] < lows[i + 1]:
            gap_size = (lows[i + 1] - highs[i - 1]) / mid_price * 100
            if gap_size >= min_gap_pct:
                fvgs.append({
                    "type":   "bullish",
                    "top":    round(lows[i + 1], 4),
                    "bottom": round(highs[i - 1], 4),
                    "time":   times[i],
                    "gap_pct": round(gap_size, 3),
                })

    return fvgs


# ─── Key Levels ───────────────────────────────────────────────────────────────

def key_levels(df: pd.DataFrame, lookback: int = 50, tolerance_pct: float = 0.3) -> list[dict]:
    """
    Find significant support/resistance levels by clustering swing points.
    """
    df = swing_points(df.tail(lookback))
    highs = df["swing_high"].dropna().tolist()
    lows  = df["swing_low"].dropna().tolist()
    points = highs + lows

    if not points:
        return []

    points.sort()
    levels = []
    cluster = [points[0]]

    for p in points[1:]:
        if (p - cluster[-1]) / cluster[-1] * 100 <= tolerance_pct:
            cluster.append(p)
        else:
            levels.append(round(np.mean(cluster), 4))
            cluster = [p]
    levels.append(round(np.mean(cluster), 4))

    current = df["close"].iloc[-1]
    result = []
    for lvl in levels:
        result.append({
            "level": lvl,
            "type":  "resistance" if lvl > current else "support",
            "dist_pct": round((lvl - current) / current * 100, 2),
        })
    result.sort(key=lambda x: abs(x["dist_pct"]))
    return result


# ─── Full Analysis ────────────────────────────────────────────────────────────

def full_analysis(candles: list[dict], symbol: str = "") -> dict:
    """
    Run full analysis on a candle set.
    Returns structured dict with all indicators + signals.
    """
    df = to_df(candles)
    current = df["close"].iloc[-1]

    # EMAs
    e21  = ema(df, 21).iloc[-1]
    e55  = ema(df, 55).iloc[-1]
    e200 = ema(df, 200).iloc[-1]

    ema_trend = "bullish" if e21 > e55 > e200 else \
                "bearish" if e21 < e55 < e200 else "mixed"

    # RSI
    rsi_val = rsi(df).iloc[-1]
    rsi_signal = "overbought" if rsi_val > 70 else \
                 "oversold"   if rsi_val < 30 else "neutral"

    # ATR
    atr_val = atr(df).iloc[-1]

    # Volume
    vol_ma  = volume_ma(df).iloc[-1]
    vol_now = df["volume"].iloc[-1]
    vol_signal = "high" if vol_now > vol_ma * 1.5 else \
                 "low"  if vol_now < vol_ma * 0.5 else "normal"

    # Structure
    structure = market_structure(df)

    # FVGs (last 30 candles)
    fvgs = find_fvgs(df.tail(30))
    nearby_fvgs = [f for f in fvgs if abs((f["top"] + f["bottom"]) / 2 - current) / current * 100 < 3]

    # Key levels
    levels = key_levels(df)[:6]

    # Bias
    bearish_score = sum([
        ema_trend == "bearish",
        rsi_val > 60,
        structure["trend"] == "bearish",
        vol_signal == "high" and df["close"].iloc[-1] < df["open"].iloc[-1],
    ])
    bullish_score = sum([
        ema_trend == "bullish",
        rsi_val < 40,
        structure["trend"] == "bullish",
        vol_signal == "high" and df["close"].iloc[-1] > df["open"].iloc[-1],
    ])
    bias = "bearish" if bearish_score > bullish_score else \
           "bullish" if bullish_score > bearish_score else "neutral"

    return {
        "symbol":    symbol,
        "price":     round(current, 4),
        "bias":      bias,
        "ema":       {"e21": round(e21, 4), "e55": round(e55, 4), "e200": round(e200, 4), "trend": ema_trend},
        "rsi":       {"value": round(rsi_val, 2), "signal": rsi_signal},
        "atr":       round(atr_val, 4),
        "volume":    {"signal": vol_signal, "ratio": round(vol_now / vol_ma, 2) if vol_ma else None},
        "structure": structure,
        "fvgs_nearby": nearby_fvgs,
        "key_levels": levels,
    }


if __name__ == "__main__":
    from mexc import get_ohlcv
    candles = get_ohlcv("ETHUSDT", "1h", 200)
    result = full_analysis(candles, "ETHUSDT")
    import json
    print(json.dumps(result, indent=2, default=str))
