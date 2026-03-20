#!/usr/bin/env python3
"""
monitor_positions.py — Check open positions, alert on SL/TP events

Usage:
    python3 monitor_positions.py                    # Check current state
    python3 monitor_positions.py --watch 5          # Check every 5 minutes
    python3 monitor_positions.py --alert            # Send alerts only if state changed
"""

import time
import json
from datetime import datetime
from pathlib import Path

try:
    from mexc import get_24h, get_position
except ImportError:
    get_24h = lambda s: {"price": 0}
    get_position = lambda s: None

try:
    from db import get_open_positions, update_position
except ImportError:
    get_open_positions = lambda: []
    update_position = lambda *args: None


def load_state(file="monitor_state.json"):
    """Load last known state."""
    if Path(file).exists():
        with open(file) as f:
            return json.load(f)
    return {}


def save_state(state, file="monitor_state.json"):
    """Save current state for comparison."""
    with open(file, "w") as f:
        json.dump(state, f, indent=2)


def check_positions():
    """Check all open positions."""
    try:
        positions = get_open_positions()
    except:
        print("❌ Could not fetch positions from database")
        return []

    results = []
    for pos in positions:
        symbol = pos.get("symbol", "UNKNOWN")
        try:
            ticker = get_24h(symbol)
            current_price = ticker.get("price", 0)
        except:
            current_price = pos.get("entry_price", 0)

        entry = float(pos.get("entry_price", 0))
        sl = float(pos.get("sl", entry * 0.99))
        tp1 = float(pos.get("tp1", entry * 1.01))
        tp2 = float(pos.get("tp2", entry * 1.02))
        direction = pos.get("direction", "LONG")

        # Calculate PnL %
        if direction == "LONG":
            pnl = ((current_price - entry) / entry) * 100 if entry else 0
            sl_hit = current_price <= sl
            tp1_hit = current_price >= tp1
            tp2_hit = current_price >= tp2
            below_entry = current_price < entry
        else:  # SHORT
            pnl = ((entry - current_price) / entry) * 100 if entry else 0
            sl_hit = current_price >= sl
            tp1_hit = current_price <= tp1
            tp2_hit = current_price <= tp2
            below_entry = current_price > entry

        result = {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "current": current_price,
            "pnl_pct": round(pnl, 2),
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "sl_hit": sl_hit,
            "tp1_hit": tp1_hit,
            "tp2_hit": tp2_hit,
            "below_entry": below_entry,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

        # Update DB
        try:
            update_position(symbol, current_price)
        except:
            pass

    return results


def print_status(positions, alerts_only=False):
    """Print formatted position status."""
    if not positions:
        print("✅ No open positions")
        return

    print("\n" + "=" * 100)
    print(f"📊 POSITION MONITOR — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    for pos in positions:
        sym = pos["symbol"]
        direction = pos["direction"]
        entry = pos["entry"]
        current = pos["current"]
        pnl = pos["pnl_pct"]
        sl_hit = pos["sl_hit"]
        tp1_hit = pos["tp1_hit"]
        tp2_hit = pos["tp2_hit"]

        # Status indicator
        if sl_hit:
            status = "🔴 SL HIT"
        elif tp2_hit:
            status = "🟢 TP2 HIT"
        elif tp1_hit:
            status = "🟡 TP1 HIT"
        elif pnl < -5:
            status = "⚠️  UNDERWATER"
        elif pnl > 0:
            status = "📈 PROFITABLE"
        else:
            status = "⏳ WAITING"

        print(f"\n{sym} {direction} | {status}")
        print(f"  Entry:   ${entry:.8f}")
        print(f"  Current: ${current:.8f} ({pnl:+.2f}%)")
        print(f"  SL:      ${pos['sl']:.8f}")
        print(f"  TP1:     ${pos['tp1']:.8f}")
        print(f"  TP2:     ${pos['tp2']:.8f}")

        if alerts_only and not (sl_hit or tp1_hit or tp2_hit):
            print(f"  [No alerts]")

    print("\n" + "=" * 100)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor open trading positions")
    parser.add_argument(
        "--watch",
        type=int,
        help="Check every N minutes",
        default=0,
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Only print alerts (skip normal status)",
    )
    args = parser.parse_args()

    last_state = load_state()

    try:
        while True:
            positions = check_positions()

            # Check for changes
            state_changed = json.dumps(positions) != json.dumps(
                last_state.get("positions", [])
            )
            has_alerts = any(
                p.get("sl_hit") or p.get("tp1_hit") or p.get("tp2_hit")
                for p in positions
            )

            if has_alerts or state_changed:
                print_status(positions, alerts_only=args.alert)

            last_state = {"positions": positions}
            save_state(last_state)

            if args.watch > 0:
                print(f"\n⏳ Next check in {args.watch} minutes...")
                time.sleep(args.watch * 60)
            else:
                break

    except KeyboardInterrupt:
        print("\n✅ Monitor stopped")


if __name__ == "__main__":
    main()
