"""
Entry scanner — finds potential setups across symbols/timeframes
Run directly: python3 scanner.py [SYMBOL] [INTERVAL]

Optional enrichment (graceful imports):
  --mtf     : include multi-timeframe confluence via mtf.py
  --ob      : include orderbook walls via orderbook.py
  --patterns: include RSI divergence / sweeps / order blocks via patterns.py
"""

import sys
import json
from mexc import get_ohlcv, get_24h
from analysis import full_analysis, to_df, rsi, ema, find_fvgs, key_levels

# ─── Graceful optional imports ────────────────────────────────────────────────

try:
    from mtf import mtf_analysis as _mtf_analysis
    _HAS_MTF = True
except ImportError:
    _HAS_MTF = False

try:
    from orderbook import analyze_orderbook as _analyze_orderbook
    _HAS_OB = True
except ImportError:
    _HAS_OB = False

try:
    from patterns import rsi_divergence as _rsi_divergence, order_blocks as _order_blocks
    _HAS_PATTERNS = True
except ImportError:
    _HAS_PATTERNS = False


DEFAULT_SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_INTERVAL = "1h"


def score_short_entry(result: dict, candles: list[dict]) -> dict:
    """
    Score a short entry opportunity (0–10).
    Higher = better short setup.
    """
    score = 0
    reasons = []

    # Bearish bias
    if result["bias"] == "bearish":
        score += 2
        reasons.append("✅ Bearish bias")

    # EMA stack bearish
    if result["ema"]["trend"] == "bearish":
        score += 2
        reasons.append("✅ EMA stack bearish (21<55<200)")

    # RSI overbought (good for short entry timing)
    rsi_val = result["rsi"]["value"]
    if rsi_val > 65:
        score += 2
        reasons.append(f"✅ RSI overbought ({rsi_val:.1f})")
    elif rsi_val > 55:
        score += 1
        reasons.append(f"⚠️  RSI elevated ({rsi_val:.1f})")

    # Bearish structure
    if result["structure"]["trend"] == "bearish":
        score += 2
        reasons.append("✅ Bearish market structure (LH/LL)")

    # Nearby bearish FVG (resistance)
    bearish_fvgs = [f for f in result["fvgs_nearby"] if f["type"] == "bearish"]
    if bearish_fvgs:
        score += 1
        reasons.append(f"✅ Bearish FVG nearby ({bearish_fvgs[0]['bottom']}–{bearish_fvgs[0]['top']})")

    # High volume on last red candle
    if result["volume"]["signal"] == "high":
        df = to_df(candles)
        last = df.iloc[-1]
        if last["close"] < last["open"]:
            score += 1
            reasons.append("✅ High volume on bearish candle")

    return {"score": score, "max": 10, "reasons": reasons}


def score_long_entry(result: dict, candles: list[dict]) -> dict:
    """Score a long entry opportunity."""
    score = 0
    reasons = []

    if result["bias"] == "bullish":
        score += 2
        reasons.append("✅ Bullish bias")

    if result["ema"]["trend"] == "bullish":
        score += 2
        reasons.append("✅ EMA stack bullish (21>55>200)")

    rsi_val = result["rsi"]["value"]
    if rsi_val < 35:
        score += 2
        reasons.append(f"✅ RSI oversold ({rsi_val:.1f})")
    elif rsi_val < 45:
        score += 1
        reasons.append(f"⚠️  RSI low ({rsi_val:.1f})")

    if result["structure"]["trend"] == "bullish":
        score += 2
        reasons.append("✅ Bullish market structure (HH/HL)")

    bullish_fvgs = [f for f in result["fvgs_nearby"] if f["type"] == "bullish"]
    if bullish_fvgs:
        score += 1
        reasons.append(f"✅ Bullish FVG nearby ({bullish_fvgs[0]['bottom']}–{bullish_fvgs[0]['top']})")

    if result["volume"]["signal"] == "high":
        df = to_df(candles)
        last = df.iloc[-1]
        if last["close"] > last["open"]:
            score += 1
            reasons.append("✅ High volume on bullish candle")

    return {"score": score, "max": 10, "reasons": reasons}


def scan_symbol(
    symbol: str,
    interval: str = "1h",
    include_mtf: bool = False,
    mtf_timeframes: list[str] = ["4h", "1h", "15m"],
    include_ob: bool = False,
    ob_limit: int = 50,
    include_patterns: bool = False,
) -> dict:
    """
    Scan a single symbol and return the full result dict.

    Optional enrichment flags (require corresponding modules):
        include_mtf      — add MTF confluence via mtf.py
        include_ob       — add orderbook wall analysis via orderbook.py
        include_patterns — add RSI divergence + order blocks via patterns.py
    """
    candles = get_ohlcv(symbol, interval, 200)
    result  = full_analysis(candles, symbol)
    short   = score_short_entry(result, candles)
    long_   = score_long_entry(result, candles)

    # Suggested SL/TP based on ATR
    atr_val = result["atr"]
    price   = result["price"]
    suggestion = {
        "short": {
            "entry": round(price, 4),
            "sl":    round(price + atr_val * 1.5, 4),
            "tp1":   round(price - atr_val * 2, 4),
            "tp2":   round(price - atr_val * 4, 4),
        },
        "long": {
            "entry": round(price, 4),
            "sl":    round(price - atr_val * 1.5, 4),
            "tp1":   round(price + atr_val * 2, 4),
            "tp2":   round(price + atr_val * 4, 4),
        }
    }

    scan_result = {
        "symbol":      symbol,
        "interval":    interval,
        "price":       price,
        "bias":        result["bias"],
        "short_score": short,
        "long_score":  long_,
        "ema":         result["ema"],
        "rsi":         result["rsi"],
        "structure":   result["structure"],
        "key_levels":  result["key_levels"][:4],
        "fvgs_nearby": result["fvgs_nearby"],
        "suggestion":  suggestion,
        "mtf":         None,
        "orderbook":   None,
        "patterns":    None,
    }

    # ── Optional: MTF confluence ───────────────────────────────────────────────
    if include_mtf and _HAS_MTF:
        try:
            scan_result["mtf"] = _mtf_analysis(symbol, mtf_timeframes)
        except Exception as e:
            print(f"[scanner] MTF error for {symbol}: {e}")

    # ── Optional: Orderbook walls ─────────────────────────────────────────────
    if include_ob and _HAS_OB:
        try:
            scan_result["orderbook"] = _analyze_orderbook(symbol, ob_limit)
        except Exception as e:
            print(f"[scanner] Orderbook error for {symbol}: {e}")

    # ── Optional: Pattern detection ───────────────────────────────────────────
    if include_patterns and _HAS_PATTERNS:
        try:
            df = to_df(candles)
            valid_obs = [o for o in _order_blocks(df) if not o["invalidated"]]
            scan_result["patterns"] = {
                "rsi_divergences": _rsi_divergence(df)[:3],
                "order_blocks":    valid_obs[:4],
            }
        except Exception as e:
            print(f"[scanner] Patterns error for {symbol}: {e}")

    return scan_result


def print_report(scan: dict):
    s = scan
    print(f"\n{'='*50}")
    print(f"  {s['symbol']} | {s['interval']} | ${s['price']}")
    print(f"  Bias: {s['bias'].upper()}  |  Structure: {s['structure']['trend'].upper()}")
    print(f"  EMA trend: {s['ema']['trend']}  (21:{s['ema']['e21']} 55:{s['ema']['e55']})")
    print(f"  RSI: {s['rsi']['value']} ({s['rsi']['signal']})")
    print(f"  Short score: {s['short_score']['score']}/{s['short_score']['max']}")
    for r in s['short_score']['reasons']:
        print(f"    {r}")
    print(f"  Long score:  {s['long_score']['score']}/{s['long_score']['max']}")
    for r in s['long_score']['reasons']:
        print(f"    {r}")
    print(f"  Key levels:")
    for l in s['key_levels']:
        print(f"    {l['type']:10} ${l['level']}  ({l['dist_pct']:+.2f}%)")
    if s['fvgs_nearby']:
        print(f"  FVGs nearby:")
        for f in s['fvgs_nearby']:
            print(f"    {f['type']:8} ${f['bottom']}–${f['top']} ({f['gap_pct']}%)")

    # ── MTF confluence (if present) ───────────────────────────────────────────
    if s.get("mtf"):
        mtf = s["mtf"]
        status = "ALIGNED" if mtf["aligned"] else "mixed"
        print(f"  MTF ({status}):  {mtf['confluence_score']}/{mtf['total_timeframes']} agree → {mtf['bias'].upper()}")
        for tf, b in mtf["bias_per_tf"].items():
            mark = "✅" if b == mtf["bias"] else "⚠️ "
            print(f"    {mark} {tf}: {b}")

    # ── Orderbook walls (if present) ──────────────────────────────────────────
    if s.get("orderbook"):
        ob = s["orderbook"]
        print(f"  Orderbook: imbalance={ob['imbalance']} ({ob['imbalance_signal']})")
        if ob.get("nearest_support"):
            print(f"    Support wall:     ${ob['nearest_support']}")
        if ob.get("nearest_resistance"):
            print(f"    Resistance wall:  ${ob['nearest_resistance']}")

    # ── Patterns (if present) ─────────────────────────────────────────────────
    if s.get("patterns"):
        pats = s["patterns"]
        divs = pats.get("rsi_divergences", [])
        obs  = pats.get("order_blocks", [])
        if divs:
            d = divs[0]
            print(f"  RSI divergence: {d['type']} (price {d['price_diff_pct']:+.2f}%, RSI {d['rsi_diff']:+.2f})")
        if obs:
            o = obs[0]
            print(f"  Order block: {o['type']} zone ${o['bottom']}–${o['top']} ({o['dist_pct']:+.2f}%)")

    best = "SHORT" if s['short_score']['score'] >= s['long_score']['score'] else "LONG"
    sug = s['suggestion'][best.lower()]
    print(f"\n  📍 Best setup: {best}")
    print(f"     Entry: ${sug['entry']}  SL: ${sug['sl']}  TP1: ${sug['tp1']}  TP2: ${sug['tp2']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    symbols  = [sys.argv[1].upper()] if len(sys.argv) > 1 else DEFAULT_SYMBOLS
    interval = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_INTERVAL

    # Optional enrichment flags
    use_mtf      = "--mtf"      in sys.argv
    use_ob       = "--ob"       in sys.argv
    use_patterns = "--patterns" in sys.argv

    if use_mtf and not _HAS_MTF:
        print("Warning: --mtf requested but mtf.py not available.")
    if use_ob and not _HAS_OB:
        print("Warning: --ob requested but orderbook.py not available.")
    if use_patterns and not _HAS_PATTERNS:
        print("Warning: --patterns requested but patterns.py not available.")

    print(f"Scanning {symbols} on {interval}...")
    for sym in symbols:
        try:
            result = scan_symbol(
                sym, interval,
                include_mtf=use_mtf,
                include_ob=use_ob,
                include_patterns=use_patterns,
            )
            print_report(result)
        except Exception as e:
            print(f"Error scanning {sym}: {e}")
