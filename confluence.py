"""
confluence.py — Master confluence engine

Scores a trading setup across multiple dimensions (0–100) and returns
a rich dict with entry zone, SL, TPs, grade, and alignment details.
"""

from __future__ import annotations

import sys
from typing import Optional

import pandas as pd

# ── Local imports (graceful) ──────────────────────────────────────────────────
try:
    from mexc import get_ohlcv, get_price
except ImportError:
    print("[confluence] WARNING: mexc not available", file=sys.stderr)
    get_ohlcv = get_price = None  # type: ignore

try:
    from analysis import full_analysis, find_fvgs, market_structure, ema, rsi, atr, swing_points
except ImportError:
    print("[confluence] WARNING: analysis not available", file=sys.stderr)
    full_analysis = find_fvgs = market_structure = ema = rsi = atr = swing_points = None  # type: ignore

try:
    from mtf import mtf_analysis
except ImportError:
    print("[confluence] WARNING: mtf not available", file=sys.stderr)
    mtf_analysis = None  # type: ignore

try:
    from orderbook import analyze_orderbook
except ImportError:
    print("[confluence] WARNING: orderbook not available", file=sys.stderr)
    analyze_orderbook = None  # type: ignore

try:
    from patterns import rsi_divergence, liquidity_sweep, order_blocks
except ImportError:
    print("[confluence] WARNING: patterns not available", file=sys.stderr)
    rsi_divergence = liquidity_sweep = order_blocks = None  # type: ignore


# ── Scoring weights (must sum to 100) ─────────────────────────────────────────
WEIGHTS = {
    "mtf_bias":        25,   # 4h + 1h + 15m agreement
    "market_structure": 15,   # BOS, HH/HL or LH/LL
    "ema_stack":       10,   # EMA alignment
    "rsi_position":    10,   # RSI zone
    "fvg_nearby":      10,   # FVG near price
    "order_block":     10,   # OB nearby
    "ob_wall":         10,   # Orderbook wall aligns with direction
    "rsi_divergence":   5,   # RSI divergence
    "liquidity_sweep":  5,   # Stop hunt / sweep before entry
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def _grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "F"


def _nearest_ob(obs: list[dict], price: float, direction: str) -> Optional[dict]:
    """Return the closest non-invalidated OB that matches the setup direction."""
    candidates = [
        ob for ob in obs
        if not ob.get("invalidated", False) and ob["type"] == direction
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda ob: abs((ob["top"] + ob["bottom"]) / 2 - price))


def _nearest_fvg(fvgs: list[dict], price: float, direction: str) -> Optional[dict]:
    """Return the closest FVG whose type matches the setup direction."""
    # For a long setup we want a bullish FVG; for short, a bearish FVG.
    fvg_type = "bullish" if direction == "bullish" else "bearish"
    candidates = [f for f in fvgs if f["type"] == fvg_type]
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs((f["top"] + f["bottom"]) / 2 - price))


# ── Main scoring function ─────────────────────────────────────────────────────

def score_setup(symbol: str, interval: str = "1h", higher_tf: str = "4h") -> dict:
    """
    Score a trading setup for *symbol* on *interval* with *higher_tf* context.

    Returns a dict with keys:
        score, direction, entry_zone, optimal_entry, stop_loss,
        tp1, tp2, tp3, rr_ratio, invalidation,
        confluence_reasons, missing, grade, details
    """
    result: dict = {
        "symbol": symbol,
        "interval": interval,
        "score": 0,
        "direction": "NO_TRADE",
        "entry_zone": (None, None),
        "optimal_entry": None,
        "stop_loss": None,
        "tp1": None,
        "tp2": None,
        "tp3": None,
        "rr_ratio": None,
        "invalidation": None,
        "confluence_reasons": [],
        "missing": [],
        "grade": "F",
        "details": {},
    }

    if get_ohlcv is None or full_analysis is None:
        result["missing"].append("Core modules unavailable")
        return result

    # ── Fetch data ──────────────────────────────────────────────────────────
    candles_1h  = get_ohlcv(symbol, interval, limit=200)
    candles_htf = get_ohlcv(symbol, higher_tf, limit=200)
    candles_15m = get_ohlcv(symbol, "15m", limit=200)

    if not candles_1h:
        result["missing"].append("No candle data")
        return result

    df_1h  = _candles_to_df(candles_1h)
    df_htf = _candles_to_df(candles_htf) if candles_htf else None
    df_15m = _candles_to_df(candles_15m) if candles_15m else None

    price = float(df_1h["close"].iloc[-1])
    result["details"]["price"] = price

    # ── Full analysis on primary TF ─────────────────────────────────────────
    analysis = full_analysis(candles_1h, symbol)
    primary_bias = analysis.get("bias", "neutral")          # bullish/bearish/neutral
    atr_val      = analysis.get("atr", price * 0.01)
    ema_data     = analysis.get("ema", {})
    rsi_data     = analysis.get("rsi", {})
    structure    = analysis.get("structure", {})
    fvgs_nearby  = analysis.get("fvgs_nearby", [])
    key_lvls     = analysis.get("key_levels", [])

    # ── 1. MTF bias agreement (25 pts) ──────────────────────────────────────
    mtf_score = 0
    mtf_detail: dict = {}
    if mtf_analysis:
        tfs = [higher_tf, interval, "15m"]
        mtf_res = mtf_analysis(symbol, timeframes=tfs)
        bias_per_tf = mtf_res.get("bias_per_tf", {})
        mtf_detail  = bias_per_tf
        agreement   = mtf_res.get("confluence_score", 0)  # count of agreeing TFs

        if agreement == len(tfs):           # all 3 agree
            mtf_score = WEIGHTS["mtf_bias"]
            result["confluence_reasons"].append(f"{higher_tf}/{interval}/15m all {'bullish' if primary_bias=='bullish' else 'bearish'}")
        elif agreement == 2:
            mtf_score = int(WEIGHTS["mtf_bias"] * 0.6)
            result["missing"].append("Partial MTF alignment (2/3)")
        else:
            result["missing"].append("MTF not aligned")

        # Determine dominant direction from HTF
        htf_bias = bias_per_tf.get(higher_tf, "neutral")
        if htf_bias != "neutral":
            primary_bias = htf_bias  # let HTF override
    else:
        result["missing"].append("MTF module unavailable")

    result["details"]["mtf"] = mtf_detail

    # ── 2. Market structure (15 pts) ────────────────────────────────────────
    ms_score = 0
    trend = structure.get("trend", "sideways")
    if primary_bias == "bullish" and trend in ("uptrend", "bullish"):
        ms_score = WEIGHTS["market_structure"]
        result["confluence_reasons"].append("Bullish market structure (HH/HL)")
    elif primary_bias == "bearish" and trend in ("downtrend", "bearish"):
        ms_score = WEIGHTS["market_structure"]
        result["confluence_reasons"].append("Bearish market structure (LH/LL)")
    else:
        result["missing"].append(f"Market structure not aligned ({trend})")

    # ── 3. EMA stack alignment (10 pts) ─────────────────────────────────────
    ema_score = 0
    ema_trend = ema_data.get("trend", "")
    if primary_bias == "bullish" and "bullish" in ema_trend.lower():
        ema_score = WEIGHTS["ema_stack"]
        result["confluence_reasons"].append("Bullish EMA stack (21>55>200)")
    elif primary_bias == "bearish" and "bearish" in ema_trend.lower():
        ema_score = WEIGHTS["ema_stack"]
        result["confluence_reasons"].append("Bearish EMA stack (21<55<200)")
    else:
        result["missing"].append(f"EMA stack not aligned ({ema_trend})")

    # ── 4. RSI positioning (10 pts) ─────────────────────────────────────────
    rsi_score = 0
    rsi_val    = rsi_data.get("value", 50)
    rsi_signal = rsi_data.get("signal", "neutral")
    if primary_bias == "bullish" and rsi_val < 45:
        rsi_score = WEIGHTS["rsi_position"]
        result["confluence_reasons"].append(f"RSI oversold ({rsi_val:.0f})")
    elif primary_bias == "bearish" and rsi_val > 55:
        rsi_score = WEIGHTS["rsi_position"]
        result["confluence_reasons"].append(f"RSI overbought ({rsi_val:.0f})")
    elif 45 <= rsi_val <= 55:
        rsi_score = int(WEIGHTS["rsi_position"] * 0.3)
        result["missing"].append(f"RSI neutral ({rsi_val:.0f})")
    else:
        result["missing"].append(f"RSI not confirming ({rsi_val:.0f})")

    # ── 5. FVG near price (10 pts) ──────────────────────────────────────────
    fvg_score = 0
    best_fvg: Optional[dict] = None
    if fvgs_nearby:
        fvg_type_want = "bullish" if primary_bias == "bullish" else "bearish"
        matching = [f for f in fvgs_nearby if f.get("type") == fvg_type_want]
        if matching:
            best_fvg = min(matching, key=lambda f: abs((f["top"] + f["bottom"]) / 2 - price))
            fvg_score = WEIGHTS["fvg_nearby"]
            mid = (best_fvg["top"] + best_fvg["bottom"]) / 2
            result["confluence_reasons"].append(
                f"{'Bullish' if primary_bias=='bullish' else 'Bearish'} FVG at "
                f"${best_fvg['bottom']:,.2f}–${best_fvg['top']:,.2f}"
            )
        else:
            result["missing"].append("No matching FVG near price")
    else:
        result["missing"].append("No FVGs nearby")

    # ── 6. Order block nearby (10 pts) ──────────────────────────────────────
    ob_score = 0
    best_ob: Optional[dict] = None
    if order_blocks and df_1h is not None:
        obs = order_blocks(df_1h, scan_window=100)
        ob_type_want = "bullish" if primary_bias == "bullish" else "bearish"
        valid_obs = [
            ob for ob in obs
            if ob["type"] == ob_type_want and not ob.get("invalidated", False)
        ]
        if valid_obs:
            best_ob = min(valid_obs, key=lambda ob: abs((ob["top"] + ob["bottom"]) / 2 - price))
            dist = abs((best_ob["top"] + best_ob["bottom"]) / 2 - price) / price * 100
            if dist < 5:  # within 5% of price
                ob_score = WEIGHTS["order_block"]
                result["confluence_reasons"].append(
                    f"{'Bullish' if primary_bias=='bullish' else 'Bearish'} OB at "
                    f"${best_ob['bottom']:,.2f}–${best_ob['top']:,.2f} ({dist:.1f}% away)"
                )
            else:
                result["missing"].append(f"Nearest OB too far ({dist:.1f}%)")
        else:
            result["missing"].append("No valid OB nearby")
    else:
        result["missing"].append("Order blocks module unavailable")

    # ── 7. Orderbook wall confluence (10 pts) ────────────────────────────────
    wall_score = 0
    if analyze_orderbook:
        try:
            ob_data = analyze_orderbook(symbol, limit=50)
            imbalance = ob_data.get("imbalance_signal", "neutral")
            if primary_bias == "bullish" and "bullish" in imbalance:
                wall_score = WEIGHTS["ob_wall"]
                result["confluence_reasons"].append(f"Orderbook wall supports long ({imbalance})")
            elif primary_bias == "bearish" and "bearish" in imbalance:
                wall_score = WEIGHTS["ob_wall"]
                result["confluence_reasons"].append(f"Orderbook wall supports short ({imbalance})")
            else:
                result["missing"].append(f"Orderbook wall not aligned ({imbalance})")
            result["details"]["orderbook_imbalance"] = imbalance
        except Exception as e:
            result["missing"].append(f"Orderbook error: {e}")
    else:
        result["missing"].append("Orderbook module unavailable")

    # ── 8. RSI divergence (5 pts) ────────────────────────────────────────────
    div_score = 0
    if rsi_divergence and df_1h is not None:
        divs = rsi_divergence(df_1h)
        div_type_want = "bullish" if primary_bias == "bullish" else "bearish"
        recent_divs = [d for d in divs if d["type"] == div_type_want]
        if recent_divs:
            div_score = WEIGHTS["rsi_divergence"]
            result["confluence_reasons"].append(
                f"RSI {'bullish' if primary_bias=='bullish' else 'bearish'} divergence"
            )
        else:
            result["missing"].append("No RSI divergence")
    else:
        result["missing"].append("RSI divergence module unavailable")

    # ── 9. Liquidity sweep (5 pts) ───────────────────────────────────────────
    sweep_score = 0
    if liquidity_sweep and df_1h is not None:
        sweeps = liquidity_sweep(df_1h)
        # For long: want a bearish sweep (stop hunt below), then reversal
        # For short: want a bullish sweep (stop hunt above), then reversal
        sweep_type_want = "bearish" if primary_bias == "bullish" else "bullish"
        recent_sweeps = [s for s in sweeps[-5:] if s.get("type") == sweep_type_want]
        if recent_sweeps:
            sweep_score = WEIGHTS["liquidity_sweep"]
            result["confluence_reasons"].append(
                f"Liquidity sweep ({'above' if primary_bias=='bearish' else 'below'} key level)"
            )
        else:
            result["missing"].append("No liquidity sweep detected")
    else:
        result["missing"].append("Liquidity sweep module unavailable")

    # ── Total score ──────────────────────────────────────────────────────────
    total_score = (
        mtf_score + ms_score + ema_score + rsi_score +
        fvg_score + ob_score + wall_score + div_score + sweep_score
    )
    result["score"] = total_score

    # ── Direction ────────────────────────────────────────────────────────────
    if total_score >= 40 and primary_bias in ("bullish", "bearish"):
        result["direction"] = "LONG" if primary_bias == "bullish" else "SHORT"
    else:
        result["direction"] = "NO_TRADE"

    # ── Entry zone (from FVG or OB) ──────────────────────────────────────────
    ez_low = ez_high = None

    if best_fvg:
        ez_low  = best_fvg["bottom"]
        ez_high = best_fvg["top"]
    elif best_ob:
        ez_low  = best_ob["bottom"]
        ez_high = best_ob["top"]
    else:
        # Fall back to ATR-based zone around price
        ez_low  = round(price - atr_val * 0.5, 8)
        ez_high = round(price + atr_val * 0.5, 8)

    result["entry_zone"]    = (ez_low, ez_high)
    result["optimal_entry"] = round((ez_low + ez_high) / 2, 8) if ez_low and ez_high else price

    # ── Stop loss ────────────────────────────────────────────────────────────
    atr_buffer = atr_val * 0.5
    if result["direction"] == "LONG":
        sl_base = best_ob["bottom"] if best_ob else ez_low
        result["stop_loss"] = round(sl_base - atr_buffer, 8) if sl_base else round(price - atr_val * 2, 8)
    elif result["direction"] == "SHORT":
        sl_base = best_ob["top"] if best_ob else ez_high
        result["stop_loss"] = round(sl_base + atr_buffer, 8) if sl_base else round(price + atr_val * 2, 8)
    else:
        result["stop_loss"] = None

    # ── TPs and R:R ──────────────────────────────────────────────────────────
    if result["optimal_entry"] and result["stop_loss"]:
        entry = result["optimal_entry"]
        sl    = result["stop_loss"]
        risk  = abs(entry - sl)

        if result["direction"] == "LONG":
            result["tp1"] = round(entry + risk * 1.5, 8)
            result["tp2"] = round(entry + risk * 3.0, 8)
            # TP3: nearest resistance above entry
            resistances = [
                k["level"] for k in key_lvls
                if k["type"] == "resistance" and k["level"] > entry
            ]
            result["tp3"] = round(min(resistances), 8) if resistances else round(entry + risk * 5, 8)
        else:
            result["tp1"] = round(entry - risk * 1.5, 8)
            result["tp2"] = round(entry - risk * 3.0, 8)
            supports = [
                k["level"] for k in key_lvls
                if k["type"] == "support" and k["level"] < entry
            ]
            result["tp3"] = round(max(supports), 8) if supports else round(entry - risk * 5, 8)

        tp3 = result["tp3"]
        result["rr_ratio"] = round(abs(tp3 - entry) / risk, 2) if risk > 0 else None

    # ── Invalidation level ───────────────────────────────────────────────────
    if result["direction"] == "LONG":
        result["invalidation"] = result["stop_loss"]
    elif result["direction"] == "SHORT":
        result["invalidation"] = result["stop_loss"]

    # ── Grade ────────────────────────────────────────────────────────────────
    result["grade"] = _grade(total_score)

    # ── Store component scores in details ────────────────────────────────────
    result["details"]["component_scores"] = {
        "mtf_bias":         mtf_score,
        "market_structure": ms_score,
        "ema_stack":        ema_score,
        "rsi_position":     rsi_score,
        "fvg_nearby":       fvg_score,
        "order_block":      ob_score,
        "ob_wall":          wall_score,
        "rsi_divergence":   div_score,
        "liquidity_sweep":  sweep_score,
    }
    result["details"]["price"]     = price
    result["details"]["rsi"]       = rsi_val
    result["details"]["atr"]       = atr_val
    result["details"]["bias"]      = primary_bias

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    tf  = sys.argv[2] if len(sys.argv) > 2 else "1h"
    htf = sys.argv[3] if len(sys.argv) > 3 else "4h"

    res = score_setup(sym, tf, htf)

    print(f"\n{'═'*50}")
    print(f"  {sym}/USDT | {res['direction']} | Grade: {res['grade']} ({res['score']}/100)")
    print(f"{'═'*50}")
    print(f"  Price:         ${res['details'].get('price', 0):,.4f}")
    if res["entry_zone"][0]:
        print(f"  Entry zone:    ${res['entry_zone'][0]:,.4f} – ${res['entry_zone'][1]:,.4f}")
        print(f"  Optimal entry: ${res['optimal_entry']:,.4f}")
    if res["stop_loss"]:
        print(f"  Stop loss:     ${res['stop_loss']:,.4f}")
    if res["tp1"]:
        print(f"  TP1:           ${res['tp1']:,.4f}  (1.5R)")
        print(f"  TP2:           ${res['tp2']:,.4f}  (3.0R)")
        print(f"  TP3:           ${res['tp3']:,.4f}  (key level)")
    if res["rr_ratio"]:
        print(f"  R:R:           1:{res['rr_ratio']}")
    print()
    for reason in res["confluence_reasons"]:
        print(f"  ✅ {reason}")
    for m in res["missing"]:
        print(f"  ❌ {m}")
    if res["invalidation"]:
        print(f"\n  Invalidation: ${res['invalidation']:,.4f}")
    print(f"{'═'*50}\n")
