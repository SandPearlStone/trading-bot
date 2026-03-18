"""
Simple walk-forward backtest engine.

backtest(symbol, interval, lookback_days) — fetches historical candles,
walks through with a 200-candle context window, runs scanner scoring at each
point, simulates entries when score >= 7, tracks outcomes using ATR-based SL/TP.

Returns comprehensive stats dict.

Usage:
    from backtest import backtest
    stats = backtest("ETHUSDT", "1h", lookback_days=90)
"""

import math
from datetime import datetime, timedelta
from mexc import get_ohlcv
from analysis import full_analysis, to_df, atr as calc_atr
from scanner import score_short_entry, score_long_entry


# ─── Config ───────────────────────────────────────────────────────────────────

CONTEXT_WINDOW  = 200     # candles used for indicator context
MIN_SCORE       = 7       # entry threshold
SL_ATR_MULT     = 1.5     # stop-loss = entry ± ATR * this
TP_ATR_MULT     = 2.0     # take-profit = entry ± ATR * this (TP1)
MAX_CONCURRENT  = 1       # don't stack trades (wait for previous to close)


# ─── Interval → minutes lookup ────────────────────────────────────────────────

_INTERVAL_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
}


def _interval_to_minutes(interval: str) -> int:
    return _INTERVAL_MINUTES.get(interval, 60)


def _candles_for_days(days: int, interval: str) -> int:
    mins_per_candle = _interval_to_minutes(interval)
    return math.ceil(days * 24 * 60 / mins_per_candle)


# ─── Trade simulation ─────────────────────────────────────────────────────────

def _simulate_trade(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    future_candles: list[dict],
) -> dict:
    """
    Walk future candles to determine if SL or TP was hit first.

    Returns:
        {outcome: "win"|"loss"|"open", exit_price, exit_candle_idx, r_multiple}
    """
    risk = abs(entry - sl)
    if risk == 0:
        return {"outcome": "open", "exit_price": entry, "exit_candle_idx": -1, "r_multiple": 0.0}

    for idx, c in enumerate(future_candles):
        high = c["high"]
        low  = c["low"]

        if direction == "LONG":
            if low <= sl:
                return {"outcome": "loss", "exit_price": sl, "exit_candle_idx": idx, "r_multiple": -1.0}
            if high >= tp:
                r = round((tp - entry) / risk, 3)
                return {"outcome": "win", "exit_price": tp, "exit_candle_idx": idx, "r_multiple": r}
        else:  # SHORT
            if high >= sl:
                return {"outcome": "loss", "exit_price": sl, "exit_candle_idx": idx, "r_multiple": -1.0}
            if low <= tp:
                r = round((entry - tp) / risk, 3)
                return {"outcome": "win", "exit_price": tp, "exit_candle_idx": idx, "r_multiple": r}

    # Trade still open at end of data
    last_price = future_candles[-1]["close"] if future_candles else entry
    r = round((last_price - entry) / risk, 3) if direction == "LONG" else round((entry - last_price) / risk, 3)
    return {"outcome": "open", "exit_price": last_price, "exit_candle_idx": len(future_candles) - 1, "r_multiple": r}


# ─── Main backtest ────────────────────────────────────────────────────────────

def backtest(
    symbol: str = "ETHUSDT",
    interval: str = "1h",
    lookback_days: int = 90,
    min_score: int = MIN_SCORE,
    verbose: bool = False,
) -> dict:
    """
    Walk-forward backtest.

    Fetches (lookback_days worth of candles + CONTEXT_WINDOW) and steps through
    each candle, running the full scanner. When score >= min_score, simulates
    an ATR-based entry and tracks the outcome.

    Returns stats dict with: total_trades, wins, losses, win_rate, avg_rr,
    profit_factor, best_trade, worst_trade, trades (list).
    """
    total_needed = _candles_for_days(lookback_days, interval) + CONTEXT_WINDOW
    # MEXC max limit per request is 1000
    fetch_limit = min(total_needed, 1000)

    print(f"[backtest] Fetching {fetch_limit} candles for {symbol} {interval}...")
    all_candles = get_ohlcv(symbol, interval, fetch_limit)

    if len(all_candles) < CONTEXT_WINDOW + 10:
        return {"error": f"Not enough candles: got {len(all_candles)}, need >{CONTEXT_WINDOW}"}

    trades = []
    in_trade_until: int = -1   # candle index until which we're locked in a trade

    # Walk from CONTEXT_WINDOW to len-1 (need future candles for trade outcome)
    for i in range(CONTEXT_WINDOW, len(all_candles) - 1):
        # Skip if we're currently in a trade
        if i <= in_trade_until:
            continue

        context = all_candles[i - CONTEXT_WINDOW : i]
        future  = all_candles[i:]          # includes current candle onward

        try:
            analysis = full_analysis(context, symbol)
        except Exception as e:
            if verbose:
                print(f"[backtest] analysis failed at candle {i}: {e}")
            continue

        short_sc = score_short_entry(analysis, context)
        long_sc  = score_long_entry(analysis, context)

        best_dir   = None
        best_score = 0
        if short_sc["score"] >= min_score and short_sc["score"] > best_score:
            best_dir, best_score = "SHORT", short_sc["score"]
        if long_sc["score"]  >= min_score and long_sc["score"]  > best_score:
            best_dir, best_score = "LONG",  long_sc["score"]

        if best_dir is None:
            continue

        # Compute ATR-based SL/TP from the context window
        df_ctx = to_df(context)
        atr_series = calc_atr(df_ctx)
        atr_val = atr_series.iloc[-1]
        if math.isnan(atr_val) or atr_val == 0:
            continue

        entry = all_candles[i]["open"]  # enter at next candle's open
        if best_dir == "LONG":
            sl = entry - atr_val * SL_ATR_MULT
            tp = entry + atr_val * TP_ATR_MULT
        else:
            sl = entry + atr_val * SL_ATR_MULT
            tp = entry - atr_val * TP_ATR_MULT

        # Simulate trade on future candles (skip candle i itself since we enter at its open)
        result = _simulate_trade(best_dir, entry, sl, tp, future[1:])

        trade_record = {
            "candle_idx":    i,
            "time":          all_candles[i]["time"],
            "direction":     best_dir,
            "score":         best_score,
            "entry":         round(entry, 6),
            "sl":            round(sl, 6),
            "tp":            round(tp, 6),
            "atr":           round(atr_val, 6),
            "outcome":       result["outcome"],
            "exit_price":    round(result["exit_price"], 6),
            "r_multiple":    result["r_multiple"],
            "exit_candle":   i + result["exit_candle_idx"] + 1,
        }
        trades.append(trade_record)

        # Lock until trade closes
        in_trade_until = trade_record["exit_candle"]

        if verbose:
            ts = all_candles[i]["time"]
            print(f"  [{ts}] {best_dir} score={best_score}  entry={entry:.4f}  → {result['outcome']}  R={result['r_multiple']}")

    # ── Compute statistics ────────────────────────────────────────────────────
    closed = [t for t in trades if t["outcome"] in ("win", "loss")]
    wins   = [t for t in closed if t["outcome"] == "win"]
    losses = [t for t in closed if t["outcome"] == "loss"]

    win_rate  = round(len(wins) / len(closed) * 100, 2) if closed else 0.0
    avg_rr    = round(sum(t["r_multiple"] for t in closed) / len(closed), 3) if closed else 0.0

    gross_win  = sum(t["r_multiple"] for t in wins)
    gross_loss = abs(sum(t["r_multiple"] for t in losses))
    profit_factor = round(gross_win / gross_loss, 3) if gross_loss else float("inf")

    best_trade  = max(closed, key=lambda t: t["r_multiple"]) if closed else None
    worst_trade = min(closed, key=lambda t: t["r_multiple"]) if closed else None

    long_trades  = [t for t in closed if t["direction"] == "LONG"]
    short_trades = [t for t in closed if t["direction"] == "SHORT"]

    return {
        "symbol":         symbol,
        "interval":       interval,
        "lookback_days":  lookback_days,
        "total_candles":  len(all_candles),
        "total_trades":   len(trades),
        "closed_trades":  len(closed),
        "open_trades":    len(trades) - len(closed),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate_pct":   win_rate,
        "avg_r_multiple": avg_rr,
        "profit_factor":  profit_factor,
        "gross_win_r":    round(gross_win, 3),
        "gross_loss_r":   round(gross_loss, 3),
        "net_r":          round(gross_win - gross_loss, 3),
        "best_trade":     best_trade,
        "worst_trade":    worst_trade,
        "long_trades":    len(long_trades),
        "short_trades":   len(short_trades),
        "long_win_rate":  round(len([t for t in long_trades  if t["outcome"]=="win"]) / len(long_trades)  * 100, 2) if long_trades  else 0.0,
        "short_win_rate": round(len([t for t in short_trades if t["outcome"]=="win"]) / len(short_trades) * 100, 2) if short_trades else 0.0,
        "trades":         trades,
    }


def print_backtest_report(stats: dict) -> None:
    """Pretty-print backtest results."""
    if "error" in stats:
        print(f"Backtest error: {stats['error']}")
        return

    print(f"\n{'='*55}")
    print(f"  BACKTEST: {stats['symbol']}  {stats['interval']}  ({stats['lookback_days']}d)")
    print(f"{'='*55}")
    print(f"  Total candles:  {stats['total_candles']}")
    print(f"  Total trades:   {stats['total_trades']}  (closed: {stats['closed_trades']})")
    print(f"  Win rate:       {stats['win_rate_pct']}%  ({stats['wins']}W / {stats['losses']}L)")
    print(f"  Avg R:R:        {stats['avg_r_multiple']}")
    print(f"  Profit factor:  {stats['profit_factor']}")
    print(f"  Net R:          {stats['net_r']}R")
    print(f"  Long:  {stats['long_trades']} trades  {stats['long_win_rate']}% win")
    print(f"  Short: {stats['short_trades']} trades  {stats['short_win_rate']}% win")
    if stats.get("best_trade"):
        b = stats["best_trade"]
        print(f"\n  Best:  {b['direction']} @ {b['entry']}  R={b['r_multiple']}  ({b['time']})")
    if stats.get("worst_trade"):
        w = stats["worst_trade"]
        print(f"  Worst: {w['direction']} @ {w['entry']}  R={w['r_multiple']}  ({w['time']})")
    print(f"{'='*55}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    symbol   = sys.argv[1].upper() if len(sys.argv) > 1 else "ETHUSDT"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1h"
    days     = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    verbose  = "--verbose" in sys.argv

    stats = backtest(symbol, interval, days, verbose=verbose)
    print_backtest_report(stats)
