"""
Multi-timeframe (MTF) analysis and confluence scoring.

mtf_analysis(symbol, timeframes) — runs full_analysis on each timeframe,
returns per-TF results plus a confluence score (how many TFs agree on bias).

mtf_scan(symbols, timeframes) — returns symbols where ALL timeframes agree.

Usage:
    from mtf import mtf_analysis, mtf_scan
    result = mtf_analysis("ETHUSDT", ["4h", "1h", "15m"])
    aligned = mtf_scan(["ETHUSDT", "BTCUSDT"], ["4h", "1h", "15m"])
"""

from mexc import get_ohlcv
from analysis import full_analysis


# ─── Single-symbol MTF ─────────────────────────────────────────────────────────

def mtf_analysis(
    symbol: str,
    timeframes: list[str] = ["4h", "1h", "15m"],
    candle_limit: int = 200,
) -> dict:
    """
    Run full_analysis on each timeframe for a symbol.

    Returns:
        {
          "symbol": str,
          "timeframes": { "4h": full_analysis_result, ... },
          "bias_per_tf": { "4h": "bullish", ... },
          "confluence_score": int,           # how many TFs agree with majority bias
          "total_timeframes": int,
          "bias": "bullish" | "bearish" | "mixed",
          "aligned": bool,                   # True when ALL TFs agree
        }
    """
    tf_results: dict[str, dict] = {}
    bias_per_tf: dict[str, str] = {}

    for tf in timeframes:
        try:
            candles = get_ohlcv(symbol, tf, candle_limit)
            analysis = full_analysis(candles, symbol)
            tf_results[tf] = analysis
            bias_per_tf[tf] = analysis["bias"]
        except Exception as e:
            print(f"[mtf] Warning: failed to analyse {symbol} on {tf}: {e}")
            bias_per_tf[tf] = "unknown"

    # Determine majority bias
    biases = [b for b in bias_per_tf.values() if b != "unknown"]
    bull_count = biases.count("bullish")
    bear_count = biases.count("bearish")

    if bull_count > bear_count:
        majority_bias = "bullish"
        confluence_score = bull_count
    elif bear_count > bull_count:
        majority_bias = "bearish"
        confluence_score = bear_count
    else:
        majority_bias = "mixed"
        confluence_score = max(bull_count, bear_count, 0)

    total = len(timeframes)
    aligned = confluence_score == total and majority_bias != "mixed"

    return {
        "symbol":            symbol,
        "timeframes":        tf_results,
        "bias_per_tf":       bias_per_tf,
        "confluence_score":  confluence_score,
        "total_timeframes":  total,
        "bias":              majority_bias,
        "aligned":           aligned,
    }


def mtf_summary(mtf_result: dict) -> str:
    """Return a compact human-readable summary of MTF confluence."""
    sym    = mtf_result["symbol"]
    bias   = mtf_result["bias"].upper()
    score  = mtf_result["confluence_score"]
    total  = mtf_result["total_timeframes"]
    status = "ALIGNED" if mtf_result["aligned"] else "MIXED"

    lines = [f"{sym} MTF [{status}]  Confluence: {score}/{total}  Bias: {bias}"]
    for tf, b in mtf_result["bias_per_tf"].items():
        mark = "✅" if b == mtf_result["bias"] else "⚠️ "
        lines.append(f"  {mark} {tf}: {b}")
    return "\n".join(lines)


# ─── Multi-symbol scan ─────────────────────────────────────────────────────────

def mtf_scan(
    symbols: list[str],
    timeframes: list[str] = ["4h", "1h", "15m"],
    require_full_alignment: bool = True,
) -> list[dict]:
    """
    Scan a list of symbols and return those with MTF confluence.

    require_full_alignment=True  → only symbols where ALL timeframes agree.
    require_full_alignment=False → any symbol where majority bias is consistent.

    Returns list of mtf_analysis() results for matching symbols.
    """
    aligned_symbols = []

    for sym in symbols:
        result = mtf_analysis(sym, timeframes)
        if require_full_alignment:
            if result["aligned"]:
                aligned_symbols.append(result)
        else:
            if result["bias"] != "mixed":
                aligned_symbols.append(result)

    # Sort by confluence score descending
    aligned_symbols.sort(key=lambda x: x["confluence_score"], reverse=True)
    return aligned_symbols


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    symbol     = sys.argv[1].upper() if len(sys.argv) > 1 else "ETHUSDT"
    timeframes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["4h", "1h", "15m"]

    print(f"MTF analysis: {symbol}  timeframes={timeframes}\n")
    result = mtf_analysis(symbol, timeframes)
    print(mtf_summary(result))

    print("\n--- Bias per TF detail ---")
    for tf, analysis in result["timeframes"].items():
        print(
            f"  {tf}  bias={analysis['bias']:<8}  "
            f"rsi={analysis['rsi']['value']:.1f}  "
            f"ema={analysis['ema']['trend']}"
        )
