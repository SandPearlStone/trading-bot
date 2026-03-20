#!/usr/bin/env python3
"""
daily_summary.py — Generate daily trade summary and stats

Usage:
    python3 daily_summary.py               # Today's stats
    python3 daily_summary.py --days 7      # Last 7 days
    python3 daily_summary.py --all         # All-time stats
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("data/trades.db")


def get_trades(days=1, all_trades=False):
    """Fetch trades from database."""
    if not DB_PATH.exists():
        print("❌ Database not found:", DB_PATH)
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if all_trades:
        cursor.execute("SELECT * FROM trades WHERE status='closed' ORDER BY exit_time DESC")
    else:
        cutoff = datetime.now() - timedelta(days=days)
        cursor.execute(
            "SELECT * FROM trades WHERE status='closed' AND exit_time > ? ORDER BY exit_time DESC",
            (cutoff.isoformat(),),
        )

    trades = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return trades


def calculate_stats(trades):
    """Calculate trading statistics."""
    if not trades:
        return {}

    wins = [t for t in trades if float(t.get("pnl_pct", 0)) > 0]
    losses = [t for t in trades if float(t.get("pnl_pct", 0)) < 0]
    breakeven = [t for t in trades if float(t.get("pnl_pct", 0)) == 0]

    win_pcts = [float(t.get("pnl_pct", 0)) for t in wins]
    loss_pcts = [float(t.get("pnl_pct", 0)) for t in losses]

    total_pnl = sum(float(t.get("pnl_pct", 0)) for t in trades)

    stats = {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate": round((len(wins) / len(trades) * 100), 2) if trades else 0,
        "avg_win": round(sum(win_pcts) / len(win_pcts), 2) if win_pcts else 0,
        "avg_loss": round(sum(loss_pcts) / len(loss_pcts), 2) if loss_pcts else 0,
        "best_trade": round(max(win_pcts), 2) if win_pcts else 0,
        "worst_trade": round(min(loss_pcts), 2) if loss_pcts else 0,
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(sum(win_pcts) / abs(sum(loss_pcts)), 2) if loss_pcts and sum(loss_pcts) != 0 else 0,
    }

    return stats, trades


def print_summary(stats, trades, period="Today"):
    """Print formatted summary."""
    if not stats:
        print(f"❌ No trades for {period}")
        return

    print("\n" + "=" * 80)
    print(f"📊 {period.upper()} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print(f"\n📈 OVERVIEW")
    print(f"  Total trades:    {stats['total_trades']}")
    print(f"  Wins:            {stats['wins']} ({stats['win_rate']:.1f}%)")
    print(f"  Losses:          {stats['losses']}")
    print(f"  Breakeven:       {stats['breakeven']}")

    print(f"\n💰 PERFORMANCE")
    print(f"  Total PnL:       {stats['total_pnl']:+.2f}%")
    print(f"  Avg Win:         {stats['avg_win']:+.2f}%")
    print(f"  Avg Loss:        {stats['avg_loss']:+.2f}%")
    print(f"  Best Trade:      {stats['best_trade']:+.2f}%")
    print(f"  Worst Trade:     {stats['worst_trade']:+.2f}%")
    print(f"  Profit Factor:   {stats['profit_factor']:.2f}")

    print(f"\n🎯 BY SYMBOL")
    symbol_stats = {}
    for trade in trades:
        sym = trade.get("symbol", "UNKNOWN")
        pnl = float(trade.get("pnl_pct", 0))
        if sym not in symbol_stats:
            symbol_stats[sym] = {"wins": 0, "losses": 0, "pnl": 0}
        if pnl > 0:
            symbol_stats[sym]["wins"] += 1
        elif pnl < 0:
            symbol_stats[sym]["losses"] += 1
        symbol_stats[sym]["pnl"] += pnl

    for sym in sorted(symbol_stats.keys()):
        s = symbol_stats[sym]
        total = s["wins"] + s["losses"]
        wr = round(s["wins"] / total * 100, 1) if total > 0 else 0
        print(f"  {sym:12} | {s['wins']:2}W-{s['losses']:2}L ({wr:5.1f}%) | {s['pnl']:+7.2f}%")

    print("\n" + "=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Daily trading summary")
    parser.add_argument("--days", type=int, default=1, help="Days to summarize")
    parser.add_argument("--all", action="store_true", help="All-time stats")
    args = parser.parse_args()

    if args.all:
        stats, trades = calculate_stats(get_trades(all_trades=True))
        print_summary(stats, trades, "ALL-TIME")
    else:
        stats, trades = calculate_stats(get_trades(days=args.days))
        period = f"LAST {args.days} DAY(S)"
        print_summary(stats, trades, period)


if __name__ == "__main__":
    main()
