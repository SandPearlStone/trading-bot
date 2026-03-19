#!/usr/bin/env python3
"""
run_backtest.py — Interactive backtest runner with multi-symbol, param tuning

Usage:
    python3 run_backtest.py                   # Interactive mode
    python3 run_backtest.py ETHUSDT 1h 90     # Quick backtest
    python3 run_backtest.py --symbols BTC ETH SOL --days 180 --interval 4h
    python3 run_backtest.py --batch           # Run all default symbols
"""

import sys
import argparse
from backtest import backtest, print_backtest_report
from watchlist import DEFAULT_SYMBOLS


def interactive_mode():
    """Prompt user for symbol, interval, days."""
    print("\n" + "="*60)
    print("  📊 Trading Backtest Engine")
    print("="*60)
    
    symbols = input("\nSymbols (space-separated, or Enter for defaults): ").strip().split()
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT"]
    
    interval = input("Interval (1m/5m/15m/30m/1h/4h/1d, default=1h): ").strip() or "1h"
    days = input("Lookback days (default=90): ").strip() or "90"
    
    try:
        days = int(days)
    except ValueError:
        days = 90
    
    print(f"\nBacktesting {', '.join(symbols)} on {interval} over {days}d...\n")
    
    for symbol in symbols:
        try:
            stats = backtest(symbol, interval, days, verbose=False)
            print_backtest_report(stats)
        except Exception as e:
            print(f"❌ {symbol} failed: {e}\n")


def batch_mode():
    """Run all default symbols with standard params."""
    print("\n" + "="*60)
    print("  📊 Batch Backtest — All Symbols")
    print("="*60)
    
    interval = "1h"
    days = 90
    
    results = []
    for symbol in DEFAULT_SYMBOLS:
        try:
            print(f"\n[{symbol}] Running...", end=" ", flush=True)
            stats = backtest(symbol, interval, days, verbose=False)
            results.append(stats)
            print(f"✓ {stats['win_rate_pct']}% WR, {stats['net_r']}R net")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Summary table
    print("\n" + "="*60)
    print("  Summary Across All Symbols")
    print("="*60)
    print(f"{'Symbol':<12} {'Trades':<10} {'Win%':<10} {'Avg R':<10} {'PF':<10}")
    print("-"*60)
    
    for stats in sorted(results, key=lambda s: s.get('profit_factor', 0), reverse=True):
        sym = stats['symbol']
        trades = stats['total_trades']
        wr = stats['win_rate_pct']
        avg_r = stats['avg_r_multiple']
        pf = stats['profit_factor']
        print(f"{sym:<12} {trades:<10} {wr:<10.1f} {avg_r:<10.3f} {pf:<10.3f}")
    
    # Best overall
    if results:
        best = max(results, key=lambda s: s.get('profit_factor', 0))
        print("\n✅ Best: {} (PF {})".format(best['symbol'], best['profit_factor']))


def single_backtest(symbol, interval, days):
    """Run a single backtest."""
    try:
        stats = backtest(symbol.upper(), interval, int(days), verbose=False)
        print_backtest_report(stats)
        return stats
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading backtest engine")
    parser.add_argument("symbol", nargs="?", help="Symbol to test (e.g., ETHUSDT)")
    parser.add_argument("interval", nargs="?", default="1h", help="Interval (1h/4h/1d)")
    parser.add_argument("days", nargs="?", type=int, default=90, help="Lookback days")
    parser.add_argument("--symbols", nargs="+", help="Multiple symbols")
    parser.add_argument("--batch", action="store_true", help="Batch test all symbols")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_mode()
    elif args.symbols:
        for sym in args.symbols:
            single_backtest(sym, args.interval, args.days)
            print()
    elif args.symbol:
        single_backtest(args.symbol, args.interval, args.days)
    else:
        interactive_mode()
