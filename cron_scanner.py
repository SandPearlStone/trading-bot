"""
Scheduled scanner — designed to be run directly or via cron.

Reads config from scanner_config.json, runs full scan + optional MTF confluence
+ optional orderbook walls + optional patterns, then formats a summary report.
Sends via Telegram if alert_enabled=True.

Cron example (every 15 minutes):
    */15 * * * * cd /path/to/trading && python3 cron_scanner.py

Or run directly:
    python3 cron_scanner.py
    python3 cron_scanner.py --dry-run   # print alert but don't send
"""

import json
import sys
import os
from datetime import datetime, timezone
from typing import Optional

# ─── Config loading ────────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "scanner_config.json")


def load_config(path: str = CONFIG_FILE) -> dict:
    with open(path) as f:
        return json.load(f)


# ─── Optional module imports ───────────────────────────────────────────────────

def _try_import_mtf():
    try:
        from mtf import mtf_analysis
        return mtf_analysis
    except ImportError:
        return None


def _try_import_orderbook():
    try:
        from orderbook import analyze_orderbook
        return analyze_orderbook
    except ImportError:
        return None


def _try_import_patterns():
    try:
        from patterns import rsi_divergence, liquidity_sweep, order_blocks
        from analysis import to_df
        from mexc import get_ohlcv
        return rsi_divergence, liquidity_sweep, order_blocks, to_df, get_ohlcv
    except ImportError:
        return None


# ─── Per-symbol scan ──────────────────────────────────────────────────────────

def run_full_scan(
    symbol: str,
    config: dict,
    mtf_fn=None,
    ob_fn=None,
    pattern_fns=None,
) -> Optional[dict]:
    """
    Run the complete scan for one symbol.
    Returns None on error.
    """
    from scanner import scan_symbol

    interval        = config["interval"]
    score_threshold = config["score_threshold"]

    try:
        scan = scan_symbol(symbol, interval)
    except Exception as e:
        print(f"[cron_scanner] scan_symbol({symbol}) failed: {e}")
        return None

    result = {
        "symbol":     symbol,
        "interval":   interval,
        "scan":       scan,
        "mtf":        None,
        "orderbook":  None,
        "patterns":   None,
        "triggered":  False,
        "direction":  None,
        "score":      0,
    }

    # Determine if any score meets threshold
    ls = scan["long_score"]["score"]
    ss = scan["short_score"]["score"]
    if ls >= score_threshold or ss >= score_threshold:
        result["triggered"] = True
        result["direction"] = "LONG" if ls >= ss else "SHORT"
        result["score"]     = max(ls, ss)

    # MTF analysis
    if mtf_fn and config.get("mtf_enabled", True):
        try:
            tfs = config.get("mtf_timeframes", ["4h", "1h", "15m"])
            result["mtf"] = mtf_fn(symbol, tfs)
        except Exception as e:
            print(f"[cron_scanner] MTF failed for {symbol}: {e}")

    # Orderbook analysis
    if ob_fn and config.get("orderbook_enabled", True):
        try:
            depth = config.get("orderbook_depth", 50)
            result["orderbook"] = ob_fn(symbol, depth)
        except Exception as e:
            print(f"[cron_scanner] Orderbook failed for {symbol}: {e}")

    # Pattern analysis
    if pattern_fns and config.get("patterns_enabled", True):
        try:
            rsi_div_fn, sweep_fn, ob_detect_fn, to_df_fn, get_ohlcv_fn = pattern_fns
            candles = get_ohlcv_fn(symbol, interval, 200)
            df = to_df_fn(candles)
            result["patterns"] = {
                "rsi_divergences": rsi_div_fn(df)[:3],
                "liquidity_sweeps": sweep_fn(df)[:3],
                "order_blocks": [o for o in ob_detect_fn(df) if not o["invalidated"]][:4],
            }
        except Exception as e:
            print(f"[cron_scanner] Patterns failed for {symbol}: {e}")

    return result


# ─── Report formatting ────────────────────────────────────────────────────────

def _format_symbol_block(r: dict) -> str:
    """Format one symbol's result for the summary report (plain text)."""
    scan  = r["scan"]
    sym   = r["symbol"]
    tf    = r["interval"]
    price = scan["price"]
    bias  = scan["bias"].upper()

    ls    = scan["long_score"]["score"]
    ss    = scan["short_score"]["score"]
    max_s = scan["long_score"]["max"]

    lines = [
        f"{'─'*45}",
        f"  {sym}  [{tf}]  ${price}  Bias: {bias}",
        f"  Long: {ls}/{max_s}   Short: {ss}/{max_s}",
    ]

    # MTF
    if r.get("mtf"):
        mtf = r["mtf"]
        lines.append(
            f"  MTF: {mtf['confluence_score']}/{mtf['total_timeframes']} agree → {mtf['bias'].upper()}"
            + ("  ✅ ALIGNED" if mtf["aligned"] else "")
        )

    # Orderbook
    if r.get("orderbook"):
        ob = r["orderbook"]
        lines.append(f"  OB Imbalance: {ob['imbalance']} ({ob['imbalance_signal']})")
        if ob.get("nearest_support"):
            lines.append(f"  Support wall:     ${ob['nearest_support']}")
        if ob.get("nearest_resistance"):
            lines.append(f"  Resistance wall:  ${ob['nearest_resistance']}")

    # Patterns
    if r.get("patterns"):
        pats = r["patterns"]
        if pats.get("rsi_divergences"):
            d = pats["rsi_divergences"][0]
            lines.append(f"  RSI div: {d['type']} @ {d['time_b']}")
        if pats.get("liquidity_sweeps"):
            s = pats["liquidity_sweeps"][0]
            lines.append(f"  Sweep: {s['type']} swept ${s['swept_level']} @ {s['time']}")
        if pats.get("order_blocks"):
            o = pats["order_blocks"][0]
            lines.append(f"  OB: {o['type']} zone ${o['bottom']}–${o['top']}")

    # Entry suggestion if triggered
    if r["triggered"]:
        dir_ = r["direction"].lower()
        sug  = scan["suggestion"][dir_]
        lines.append(f"\n  *** {r['direction']} SIGNAL (score {r['score']}/{max_s}) ***")
        lines.append(f"  Entry: ${sug['entry']}  SL: ${sug['sl']}  TP1: ${sug['tp1']}  TP2: ${sug['tp2']}")

    return "\n".join(lines)


def build_report(results: list[dict], config: dict) -> str:
    """Build the full summary report as a string."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    threshold = config["score_threshold"]
    triggered = [r for r in results if r.get("triggered")]

    header = [
        f"{'='*45}",
        f"  CRON SCANNER REPORT  —  {now}",
        f"  Interval: {config['interval']}  Threshold: {threshold}/10",
        f"  Symbols scanned: {len(results)}  |  Signals: {len(triggered)}",
        f"{'='*45}",
    ]

    # Triggered first, then rest
    ordered = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
    body = [_format_symbol_block(r) for r in ordered]

    return "\n".join(header + [""] + body + [f"{'='*45}"])


def build_telegram_message(results: list[dict], config: dict) -> str:
    """Build a concise Telegram HTML message for triggered signals only."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    triggered = [r for r in results if r.get("triggered")]

    if not triggered:
        return f"<b>Scanner [{now}]</b>\nNo signals above threshold {config['score_threshold']}."

    lines = [f"<b>🔔 Scanner Alert [{now}]</b>"]
    lines.append(f"Threshold: {config['score_threshold']}/10  |  {len(triggered)} signal(s)\n")

    for r in sorted(triggered, key=lambda x: x["score"], reverse=True):
        scan = r["scan"]
        sym  = r["symbol"]
        dir_ = r["direction"]
        sug  = scan["suggestion"][dir_.lower()]
        arrow = "🟢" if dir_ == "LONG" else "🔴"

        lines.append(
            f"{arrow} <b>{sym} {dir_}</b>  score {r['score']}/10  ${scan['price']}"
        )
        lines.append(
            f"   SL: <code>{sug['sl']}</code>  TP: <code>{sug['tp1']}</code>"
        )

        if r.get("mtf") and r["mtf"]["aligned"]:
            lines.append(f"   MTF: all {r['mtf']['total_timeframes']} TFs aligned {r['mtf']['bias'].upper()}")

        if r.get("orderbook"):
            ob = r["orderbook"]
            lines.append(f"   OB: {ob['imbalance_signal']}  imbalance={ob['imbalance']}")

        lines.append("")

    return "\n".join(lines).strip()


# ─── Main entry point ─────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    config = load_config()

    symbols   = config["symbols"]
    alert_on  = config.get("alert_enabled", False)
    log_file  = config.get("log_file") if config.get("log_to_file") else None

    # Load optional modules
    mtf_fn       = _try_import_mtf()
    ob_fn        = _try_import_orderbook()
    pattern_fns  = _try_import_patterns()

    print(f"[cron_scanner] Scanning {len(symbols)} symbols on {config['interval']}...")
    if not mtf_fn:
        print("[cron_scanner] MTF module not available, skipping.")
    if not ob_fn:
        print("[cron_scanner] Orderbook module not available, skipping.")
    if not pattern_fns:
        print("[cron_scanner] Patterns module not available, skipping.")

    results = []
    for sym in symbols:
        print(f"  → {sym}")
        r = run_full_scan(sym, config, mtf_fn, ob_fn, pattern_fns)
        if r:
            results.append(r)

    # Build and print report
    report = build_report(results, config)
    print("\n" + report)

    # Log to file if configured
    if log_file:
        with open(log_file, "a") as f:
            f.write(report + "\n\n")
        print(f"[cron_scanner] Log appended to {log_file}")

    # Telegram alert
    if alert_on or dry_run:
        tg_msg = build_telegram_message(results, config)
        if dry_run:
            print("\n--- TELEGRAM MESSAGE (dry-run) ---")
            print(tg_msg)
            print("----------------------------------")
        else:
            try:
                from alerts import send_alert
                ok = send_alert(tg_msg)
                print(f"[cron_scanner] Telegram alert {'sent' if ok else 'FAILED'}")
            except Exception as e:
                print(f"[cron_scanner] Alert error: {e}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
