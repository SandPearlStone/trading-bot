"""
entry_finder.py — Precision entry timing

Zooms into 15m candles to determine whether the current price action
presents a high-confidence entry at the confluence zone identified
by the master scoring engine.
"""

from __future__ import annotations

import sys
from typing import Optional

import pandas as pd

# ── Local imports (graceful) ──────────────────────────────────────────────────
try:
    from mexc import get_ohlcv
except ImportError:
    print("[entry_finder] WARNING: mexc not available", file=sys.stderr)
    get_ohlcv = None  # type: ignore

try:
    from analysis import find_fvgs, rsi as calc_rsi
except ImportError:
    print("[entry_finder] WARNING: analysis not available", file=sys.stderr)
    find_fvgs = calc_rsi = None  # type: ignore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def _in_zone(price: float, zone_low: float, zone_high: float, tolerance_pct: float = 0.3) -> bool:
    """Return True if price is inside the zone or within tolerance of its edges."""
    tol = (zone_high - zone_low) * tolerance_pct
    return (zone_low - tol) <= price <= (zone_high + tol)


def _is_bullish_engulfing(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1:
        return False
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    prev_bearish = prev["close"] < prev["open"]
    curr_bullish = curr["close"] > curr["open"]
    engulfs = curr["close"] > prev["open"] and curr["open"] < prev["close"]
    return bool(prev_bearish and curr_bullish and engulfs)


def _is_bearish_engulfing(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1:
        return False
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    prev_bullish = prev["close"] > prev["open"]
    curr_bearish = curr["close"] < curr["open"]
    engulfs = curr["close"] < prev["open"] and curr["open"] > prev["close"]
    return bool(prev_bullish and curr_bearish and engulfs)


def _is_bullish_pin_bar(df: pd.DataFrame, idx: int) -> bool:
    row  = df.iloc[idx]
    body = abs(row["close"] - row["open"])
    wick = row["open"] - row["low"] if row["close"] > row["open"] else row["close"] - row["low"]
    total = row["high"] - row["low"]
    if total == 0:
        return False
    return bool(wick >= body * 2 and wick / total >= 0.5)


def _is_bearish_pin_bar(df: pd.DataFrame, idx: int) -> bool:
    row  = df.iloc[idx]
    body = abs(row["close"] - row["open"])
    wick = row["high"] - row["close"] if row["close"] < row["open"] else row["high"] - row["open"]
    total = row["high"] - row["low"]
    if total == 0:
        return False
    return bool(wick >= body * 2 and wick / total >= 0.5)


# ── Main entry-finding function ───────────────────────────────────────────────

def find_entry(
    symbol: str,
    direction: str,
    entry_zone: tuple[float, float],
    interval: str = "15m",
) -> dict:
    """
    Zoom into *interval* (default 15m) to find a precision entry signal.

    Parameters
    ----------
    symbol      : e.g. "BTC"
    direction   : "LONG" or "SHORT"
    entry_zone  : (zone_low, zone_high) from confluence.score_setup
    interval    : timeframe to examine (default "15m")

    Returns
    -------
    dict with keys:
        ready        (bool)   — True if a valid trigger fired
        entry_price  (float)  — suggested entry
        confidence   (0–100) — signal strength
        triggers     (list)  — what confirmed the entry
        warnings     (list)  — what didn't confirm
    """
    result: dict = {
        "symbol":      symbol,
        "direction":   direction,
        "ready":       False,
        "entry_price": None,
        "confidence":  0,
        "triggers":    [],
        "warnings":    [],
    }

    if get_ohlcv is None:
        result["warnings"].append("mexc module unavailable")
        return result

    zone_low, zone_high = entry_zone
    if zone_low is None or zone_high is None:
        result["warnings"].append("Entry zone not defined")
        return result

    candles = get_ohlcv(symbol, interval, limit=100)
    if not candles or len(candles) < 3:
        result["warnings"].append("Insufficient candle data")
        return result

    df   = _candles_to_df(candles)
    last = df.iloc[-1]
    price = float(last["close"])
    idx   = len(df) - 1

    # ── Check 1: Is price in the entry zone? ─────────────────────────────────
    in_zone = _in_zone(price, zone_low, zone_high)
    if in_zone:
        result["triggers"].append(f"Price in entry zone (${zone_low:,.4f}–${zone_high:,.4f})")
        result["confidence"] += 30
        result["entry_price"] = price
    else:
        dist_pct = min(
            abs(price - zone_low) / zone_low * 100,
            abs(price - zone_high) / zone_high * 100,
        )
        result["warnings"].append(
            f"Price ${price:,.4f} outside zone by {dist_pct:.1f}%"
        )

    # ── Check 2: Candlestick pattern at zone ─────────────────────────────────
    # Check last 2 candles for pattern
    pattern_found = False
    for check_idx in [idx, idx - 1]:
        if check_idx < 1:
            continue
        if direction == "LONG":
            if _is_bullish_engulfing(df, check_idx):
                result["triggers"].append("Bullish engulfing at zone")
                result["confidence"] += 25
                pattern_found = True
                break
            if _is_bullish_pin_bar(df, check_idx):
                result["triggers"].append("Bullish pin bar at zone")
                result["confidence"] += 20
                pattern_found = True
                break
        else:  # SHORT
            if _is_bearish_engulfing(df, check_idx):
                result["triggers"].append("Bearish engulfing at zone")
                result["confidence"] += 25
                pattern_found = True
                break
            if _is_bearish_pin_bar(df, check_idx):
                result["triggers"].append("Bearish pin bar at zone")
                result["confidence"] += 20
                pattern_found = True
                break

    if not pattern_found:
        result["warnings"].append("No confirming candlestick pattern")

    # ── Check 3: 15m FVG inside the entry zone ───────────────────────────────
    if find_fvgs:
        try:
            fvgs = find_fvgs(df)
            fvg_type_want = "bullish" if direction == "LONG" else "bearish"
            zone_fvgs = [
                f for f in fvgs
                if f["type"] == fvg_type_want
                and f["bottom"] >= zone_low * 0.998
                and f["top"] <= zone_high * 1.002
            ]
            if zone_fvgs:
                best = zone_fvgs[-1]  # most recent
                result["triggers"].append(
                    f"15m FVG inside zone (${best['bottom']:,.4f}–${best['top']:,.4f})"
                )
                result["confidence"] += 20
                # Refine entry price to FVG mid
                if in_zone:
                    result["entry_price"] = round((best["top"] + best["bottom"]) / 2, 8)
            else:
                result["warnings"].append("No 15m FVG inside entry zone")
        except Exception as e:
            result["warnings"].append(f"FVG check error: {e}")
    else:
        result["warnings"].append("FVG module unavailable")

    # ── Check 4: RSI confirming ───────────────────────────────────────────────
    if calc_rsi:
        try:
            rsi_series = calc_rsi(df)
            rsi_val    = float(rsi_series.iloc[-1])
            if direction == "LONG" and rsi_val < 40:
                result["triggers"].append(f"RSI oversold ({rsi_val:.0f}) on {interval}")
                result["confidence"] += 25
            elif direction == "SHORT" and rsi_val > 60:
                result["triggers"].append(f"RSI overbought ({rsi_val:.0f}) on {interval}")
                result["confidence"] += 25
            elif direction == "LONG" and rsi_val < 50:
                result["triggers"].append(f"RSI below midline ({rsi_val:.0f})")
                result["confidence"] += 10
            elif direction == "SHORT" and rsi_val > 50:
                result["triggers"].append(f"RSI above midline ({rsi_val:.0f})")
                result["confidence"] += 10
            else:
                result["warnings"].append(f"RSI not confirming ({rsi_val:.0f})")
        except Exception as e:
            result["warnings"].append(f"RSI check error: {e}")
    else:
        result["warnings"].append("RSI module unavailable")

    # ── Cap confidence at 100 ─────────────────────────────────────────────────
    result["confidence"] = min(result["confidence"], 100)

    # ── Ready? Require price in zone + at least one pattern OR FVG ───────────
    has_zone    = any("Price in entry zone" in t for t in result["triggers"])
    has_pattern = any("engulfing" in t or "pin bar" in t for t in result["triggers"])
    has_fvg     = any("FVG" in t for t in result["triggers"])

    result["ready"] = bool(has_zone and (has_pattern or has_fvg))

    # Default entry to current price if not refined
    if result["entry_price"] is None:
        result["entry_price"] = price

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python3 entry_finder.py <symbol> <LONG|SHORT> <zone_low> <zone_high> [interval]")
        sys.exit(1)

    sym  = sys.argv[1]
    dirn = sys.argv[2].upper()
    zl   = float(sys.argv[3])
    zh   = float(sys.argv[4])
    tf   = sys.argv[5] if len(sys.argv) > 5 else "15m"

    res = find_entry(sym, dirn, (zl, zh), tf)

    status = "✅ READY" if res["ready"] else "⏳ NOT READY"
    print(f"\n{sym} {dirn} Entry Check — {status}")
    print(f"  Confidence: {res['confidence']}%")
    if res["entry_price"]:
        print(f"  Entry price: ${res['entry_price']:,.4f}")
    print("\n  Triggers:")
    for t in res["triggers"]:
        print(f"    ✅ {t}")
    print("\n  Warnings:")
    for w in res["warnings"]:
        print(f"    ⚠️  {w}")
