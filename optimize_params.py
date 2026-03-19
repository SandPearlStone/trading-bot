#!/usr/bin/env python3
"""
optimize_params.py — Find optimal min_score + ATR multipliers for a symbol

Tests combinations of:
  - min_score: [5, 6, 7, 8, 9] (entry threshold)
  - SL ATR mult: [1.0, 1.5, 2.0] (stop loss distance)
  - TP ATR mult: [1.5, 2.0, 3.0] (take profit distance)

Usage:
    python3 optimize_params.py ETHUSDT 1h 90     # Optimize ETH
    python3 optimize_params.py --symbols BTC ETH --quick  # Quick sweep
"""

import sys
import argparse
from backtest import backtest as _backtest


def backtest_with_params(symbol, interval, days, min_score, sl_mult, tp_mult):
    """Wrapper to override backtest params."""
    import backtest
    
    # Save originals
    orig_min = backtest.MIN_SCORE
    orig_sl = backtest.SL_ATR_MULT
    orig_tp = backtest.TP_ATR_MULT
    
    try:
        # Override
        backtest.MIN_SCORE = min_score
        backtest.SL_ATR_MULT = sl_mult
        backtest.TP_ATR_MULT = tp_mult
        
        # Run
        stats = _backtest(symbol, interval, days, min_score=min_score, verbose=False)
        return stats
    finally:
        # Restore
        backtest.MIN_SCORE = orig_min
        backtest.SL_ATR_MULT = orig_sl
        backtest.TP_ATR_MULT = orig_tp


def optimize_single(symbol, interval, days, quick=False):
    """Optimize one symbol."""
    print(f"\n{'='*70}")
    print(f"  Optimizing {symbol.upper()} ({interval}, {days}d)")
    print(f"{'='*70}")
    
    min_scores = [5, 6, 7, 8] if quick else [5, 6, 7, 8, 9]
    sl_mults = [1.5, 2.0] if quick else [1.0, 1.5, 2.0]
    tp_mults = [2.0, 3.0] if quick else [1.5, 2.0, 3.0]
    
    results = []
    total = len(min_scores) * len(sl_mults) * len(tp_mults)
    count = 0
    
    print(f"\nTesting {total} combinations...\n")
    print(f"{'Score':<8} {'SL Mult':<10} {'TP Mult':<10} {'Trades':<10} {'WR%':<10} {'Net R':<10} {'PF':<10}")
    print("-"*70)
    
    for min_s in min_scores:
        for sl_m in sl_mults:
            for tp_m in tp_mults:
                count += 1
                try:
                    stats = backtest_with_params(symbol, interval, days, min_s, sl_m, tp_m)
                    
                    if "error" not in stats:
                        results.append({
                            "min_score": min_s,
                            "sl_mult": sl_m,
                            "tp_mult": tp_m,
                            "trades": stats["total_trades"],
                            "wr": stats["win_rate_pct"],
                            "net_r": stats["net_r"],
                            "pf": stats["profit_factor"],
                            "stats": stats,
                        })
                        
                        print(f"{min_s:<8} {sl_m:<10.1f} {tp_m:<10.1f} {stats['total_trades']:<10} {stats['win_rate_pct']:<10.1f} {stats['net_r']:<10.2f} {stats['profit_factor']:<10.3f}")
                    else:
                        print(f"{min_s:<8} {sl_m:<10.1f} {tp_m:<10.1f} {'ERROR':<10} {'-':<10} {'-':<10} {'-':<10}")
                
                except Exception as e:
                    print(f"{min_s:<8} {sl_m:<10.1f} {tp_m:<10.1f} {'FAIL':<10} {'-':<10} {'-':<10} {'-':<10}")
                    # Uncomment for debugging: print(f"  {e}")
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("  Best by Profit Factor")
        print("="*70)
        
        sorted_pf = sorted(results, key=lambda r: r["pf"], reverse=True)
        for i, r in enumerate(sorted_pf[:5], 1):
            print(f"\n{i}. Score={r['min_score']} SL={r['sl_mult']} TP={r['tp_mult']}")
            print(f"   Trades: {r['trades']}  WR: {r['wr']:.1f}%  Net: {r['net_r']:.2f}R  PF: {r['pf']:.3f}")
        
        print("\n" + "="*70)
        print("  Best by Net R (Absolute Profit)")
        print("="*70)
        
        sorted_net = sorted(results, key=lambda r: r["net_r"], reverse=True)
        for i, r in enumerate(sorted_net[:5], 1):
            print(f"\n{i}. Score={r['min_score']} SL={r['sl_mult']} TP={r['tp_mult']}")
            print(f"   Trades: {r['trades']}  WR: {r['wr']:.1f}%  Net: {r['net_r']:.2f}R  PF: {r['pf']:.3f}")
        
        # Best balanced (good PF + decent trades)
        print("\n" + "="*70)
        print("  Best Balanced (PF * Trades)")
        print("="*70)
        
        sorted_balanced = sorted(results, key=lambda r: r["pf"] * (r["trades"] / 10), reverse=True)
        for i, r in enumerate(sorted_balanced[:3], 1):
            print(f"\n{i}. Score={r['min_score']} SL={r['sl_mult']} TP={r['tp_mult']}")
            print(f"   Trades: {r['trades']}  WR: {r['wr']:.1f}%  Net: {r['net_r']:.2f}R  PF: {r['pf']:.3f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter optimization for backtest")
    parser.add_argument("symbol", nargs="?", help="Symbol (e.g., ETHUSDT)")
    parser.add_argument("interval", nargs="?", default="1h", help="Interval")
    parser.add_argument("days", nargs="?", type=int, default=90, help="Days")
    parser.add_argument("--symbols", nargs="+", help="Multiple symbols")
    parser.add_argument("--quick", action="store_true", help="Faster sweep (fewer params)")
    
    args = parser.parse_args()
    
    if args.symbols:
        all_results = {}
        for sym in args.symbols:
            results = optimize_single(sym, args.interval, args.days, quick=args.quick)
            all_results[sym] = results
        
        print("\n\n" + "="*70)
        print("  SUMMARY ACROSS ALL SYMBOLS")
        print("="*70)
        for sym, results in all_results.items():
            if results:
                best = max(results, key=lambda r: r["pf"])
                print(f"\n{sym}: Score={best['min_score']} SL={best['sl_mult']} TP={best['tp_mult']} → PF {best['pf']:.3f}")
    
    elif args.symbol:
        optimize_single(args.symbol, args.interval, args.days, quick=args.quick)
    else:
        print("Usage: python3 optimize_params.py ETHUSDT [1h] [90]")
        print("   or: python3 optimize_params.py --symbols BTC ETH SOL")
