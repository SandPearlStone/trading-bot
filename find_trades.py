"""
find_trades.py — The main entrypoint

Scans the watchlist, checks entry timing, prints a rich report,
optionally sends Telegram alerts, and offers to log setups to the trade journal.

Usage:
    python3 find_trades.py [--symbols BTC ETH SOL] [--interval 1h]
                           [--higher-tf 4h] [--min-grade B] [--alert]
                           [--no-entry] [--no-journal]
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

# ── Local imports ─────────────────────────────────────────────────────────────
try:
    from watchlist import scan_watchlist, DEFAULT_SYMBOLS
except ImportError as e:
    print(f"[find_trades] FATAL: watchlist unavailable — {e}")
    sys.exit(1)

try:
    from confluence import score_setup
except ImportError:
    score_setup = None  # type: ignore

try:
    from entry_finder import find_entry
except ImportError:
    find_entry = None  # type: ignore

try:
    from alerts import send_alert
except ImportError:
    print("[find_trades] WARNING: alerts not available — Telegram disabled")
    send_alert = None  # type: ignore

try:
    from trades import add_trade
except ImportError:
    print("[find_trades] WARNING: trades not available — journal disabled")
    add_trade = None  # type: ignore


# ── Rich report formatter ─────────────────────────────────────────────────────

def _fmt_setup(r: dict, idx: int) -> str:
    """Format a single setup result into the canonical report block."""
    sym   = r.get("symbol", "?")
    dirn  = r.get("direction", "?")
    grade = r.get("grade", "?")
    score = r.get("score", 0)
    price = r.get("details", {}).get("price", 0)
    ez    = r.get("entry_zone", (None, None))
    opt   = r.get("optimal_entry")
    sl    = r.get("stop_loss")
    tp1   = r.get("tp1")
    tp2   = r.get("tp2")
    tp3   = r.get("tp3")
    rr    = r.get("rr_ratio")
    inv   = r.get("invalidation")
    entry_info = r.get("entry")
    pos_rec = r.get("position_recommendation")
    kelly_info = r.get("kelly_information")

    lines = [
        f"{'═'*50}",
        f"  {sym}/USDT | {dirn} | Grade: {grade} ({score}/100)",
        f"  Price: ${price:,.4f}",
        "",
    ]

    if ez[0] is not None:
        lines.append(f"  Entry zone:    ${ez[0]:,.4f} – ${ez[1]:,.4f}")
    if opt:
        lines.append(f"  Optimal entry: ${opt:,.4f}")
    if sl:
        sl_pct = abs((opt or price) - sl) / (opt or price) * 100
        if tp1:
            tp1_str = f"  TP1: ${tp1:,.4f}  (1.5R)"
            entry_timing = ""
            if entry_info:
                if entry_info.get("ready"):
                    entry_timing = f"  ✓ Entry timing: READY ({entry_info['confidence']}%)"
                else:
                    entry_timing = f"  ⏳ Entry timing: WAITING"
            lines.append(f"  Stop loss:     ${sl:,.4f}  ({sl_pct:.2f}%)")
            lines.append(tp1_str + (f"  {entry_timing}" if entry_timing else ""))
        else:
            lines.append(f"  Stop loss:     ${sl:,.4f}  ({sl_pct:.2f}%)")
    if tp2:
        lines.append(f"  TP2: ${tp2:,.4f}  (3.0R)")
    if tp3:
        lines.append(f"  TP3: ${tp3:,.4f}  (key level)")
    if rr:
        lines.append(f"  R:R: 1:{rr}")

    # Entry timing on its own line if TP1 didn't include it
    if entry_info and tp1:
        pass  # already shown inline above
    elif entry_info:
        if entry_info.get("ready"):
            lines.append(f"  ✅ Entry timing: READY ({entry_info['confidence']}%)")
        else:
            lines.append(f"  ⏳ Entry timing: WAITING ({entry_info['confidence']}%)")

    # Kelly position sizing section
    if pos_rec and kelly_info:
        lines.append("")
        lines.append(f"  ✨ KELLY SIZING (Account: $10,000):")
        lines.append(f"     f*/4:         {kelly_info['kelly_f_quarter']:.1%} (base)")
        regime = r.get("details", {}).get("regime", {}).get("regime", "UNKNOWN")
        lines.append(f"     Regime:       {regime} ({pos_rec['regime_multiplier']}x multiplier)")
        lines.append(f"     Position:     ${pos_rec['position_size']:,.0f} ({pos_rec['adjusted_risk_pct']:.2f}% risk)")
        lines.append(f"     Leverage:     {pos_rec['recommended_leverage']:.1f}x")

    lines.append("")

    # Confluence reasons (check marks)
    for reason in r.get("confluence_reasons", []):
        lines.append(f"  ✅ {reason}")

    # Missing / warnings
    missing = r.get("missing", [])
    for m in missing:
        # Distinguish between "not aligned" (❌) and "unavailable" (⚠️)
        icon = "⚠️ " if "unavailable" in m.lower() or "error" in m.lower() else "❌"
        lines.append(f"  {icon} {m}")

    if inv:
        lines.append(f"\n  Invalidation: ${inv:,.4f}")

    lines.append(f"{'═'*50}")
    return "\n".join(lines)


def _fmt_telegram(results: list[dict]) -> str:
    """HTML Telegram message for top setups."""
    if not results:
        return "No A/B grade setups found."

    lines = ["<b>🎯 Trade Setups Found</b>"]
    for r in results[:5]:
        sym   = r.get("symbol", "?")
        dirn  = r.get("direction", "?")
        grade = r.get("grade", "?")
        score = r.get("score", 0)
        price = r.get("details", {}).get("price", 0)
        ez    = r.get("entry_zone", (None, None))
        sl    = r.get("stop_loss")
        tp1   = r.get("tp1")
        rr    = r.get("rr_ratio")

        icon  = "🟢" if dirn == "LONG" else "🔴"
        ready = r.get("entry") and r["entry"].get("ready")
        entry_flag = " 🎯 <b>ENTRY NOW</b>" if ready else ""

        lines.append(
            f"\n{icon} <b>{sym}/USDT</b> | {dirn} | Grade <b>{grade}</b> ({score}/100)"
            f"{entry_flag}"
        )
        lines.append(f"   Price: ${price:,.4f}")
        if ez[0]:
            lines.append(f"   Zone:  ${ez[0]:,.4f} – ${ez[1]:,.4f}")
        if sl and tp1:
            rr_str = f" | R:R 1:{rr}" if rr else ""
            lines.append(f"   SL: ${sl:,.4f} | TP1: ${tp1:,.4f}{rr_str}")
        top_reasons = r.get("confluence_reasons", [])[:3]
        for reason in top_reasons:
            lines.append(f"   ✅ {reason}")

    return "\n".join(lines)


def _log_to_journal(r: dict) -> None:
    """Interactive prompt to log a setup to the trade journal."""
    if add_trade is None:
        print("  Trade journal not available.")
        return

    sym   = r.get("symbol", "?")
    dirn  = r.get("direction", "?")
    opt   = r.get("optimal_entry")
    sl    = r.get("stop_loss")
    tp1   = r.get("tp1")
    tp2   = r.get("tp2")
    tp3   = r.get("tp3")
    score = r.get("score", 0)
    grade = r.get("grade", "")

    if not opt or not sl:
        print("  Cannot log: missing entry or SL.")
        return

    print(f"\n  Logging {sym} {dirn} @ ${opt:,.4f}  SL: ${sl:,.4f}")
    size_str  = input("  Position size USDT (0=skip): ").strip()
    lev_str   = input("  Leverage (default 1): ").strip()
    notes_str = input("  Notes (Enter to skip): ").strip()

    size     = float(size_str) if size_str else 0.0
    leverage = float(lev_str)  if lev_str  else 1.0
    notes    = notes_str or f"Grade {grade} | Score {score}/100"

    reasons = "; ".join(r.get("confluence_reasons", []))
    if reasons:
        notes += f" | {reasons}"

    trade_id = add_trade(
        symbol=sym, direction=dirn, entry_price=opt, sl=sl,
        tp1=tp1, tp2=tp2, tp3=tp3, size_usdt=size, leverage=leverage,
        confluence_score=score, grade=grade, setup_notes=notes,
    )
    print(f"  ✅ Logged as trade #{trade_id}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find high-confluence trade setups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 find_trades.py
  python3 find_trades.py --symbols BTC ETH SOL
  python3 find_trades.py --min-grade A --alert
  python3 find_trades.py --interval 4h --higher-tf 1d --no-journal
""",
    )
    parser.add_argument("--symbols",    nargs="+", default=None,  help="Symbols to scan (base only, e.g. BTC ETH)")
    parser.add_argument("--interval",   default="1h",             help="Primary timeframe (default: 1h)")
    parser.add_argument("--higher-tf",  default="4h",             help="Higher timeframe (default: 4h)")
    parser.add_argument("--min-grade",  default="B",              help="Minimum grade to show: A, B, C (default: B)")
    parser.add_argument("--alert",      action="store_true",      help="Send top setups via Telegram")
    parser.add_argument("--no-entry",   action="store_true",      help="Skip entry timing check (faster)")
    parser.add_argument("--no-journal", action="store_true",      help="Skip trade journal prompt")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS
    print(f"\n{'═'*43}")
    print(f"  find_trades.py — Trade Finder")
    print(f"{'═'*43}")
    print(f"  Symbols:   {', '.join(symbols)}")
    print(f"  Interval:  {args.interval} / {args.higher_tf}")
    print(f"  Min grade: {args.min_grade}")
    print(f"  Alerts:    {'yes' if args.alert else 'no'}")
    print(f"{'─'*43}\n")

    # ── Step 1: Scan watchlist ────────────────────────────────────────────────
    print("Step 1/3 — Scanning symbols for A/B grade setups...\n")
    results = scan_watchlist(
        symbols=symbols,
        interval=args.interval,
        higher_tf=args.higher_tf,
        min_grade=args.min_grade,
        check_entry=not args.no_entry,
    )

    if not results:
        print(f"\nNo {args.min_grade}+ grade setups found. Try a lower --min-grade or different symbols.")
        return

    print(f"\nStep 2/3 — Displaying {len(results)} qualifying setup(s):\n")

    # ── Step 2: Print report ──────────────────────────────────────────────────
    ready_setups = []
    for i, r in enumerate(results, 1):
        print(_fmt_setup(r, i))
        print()
        if r.get("entry") and r["entry"].get("ready"):
            ready_setups.append(r)

    # Summary
    print(f"\n{'─'*43}")
    print(f"  Summary: {len(results)} setup(s) found, {len(ready_setups)} with READY entry")
    print(f"{'─'*43}\n")

    # ── Step 3: Telegram alerts ───────────────────────────────────────────────
    if args.alert:
        print("Step 3/3 — Sending Telegram alert...")
        if send_alert:
            msg = _fmt_telegram(results)
            ok  = send_alert(msg, parse_mode="HTML")
            if ok:
                print("  ✅ Alert sent")
            else:
                print("  ❌ Alert failed (check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        else:
            print("  ❌ alerts module not available")
    else:
        print("Step 3/3 — Skipping Telegram (use --alert to enable)")

    # ── Step 4: Trade journal prompt ─────────────────────────────────────────
    if not args.no_journal and add_trade is not None and results:
        print(f"\n{'─'*43}")
        print("  Log a setup to the trade journal?")
        print(f"{'─'*43}")
        for i, r in enumerate(results, 1):
            sym   = r.get("symbol", "?")
            dirn  = r.get("direction", "?")
            grade = r.get("grade", "?")
            print(f"  [{i}] {sym} {dirn} | Grade {grade}")
        print("  [0] Skip")

        choice = input("\n  Enter number: ").strip()
        try:
            n = int(choice)
            if 1 <= n <= len(results):
                _log_to_journal(results[n - 1])
        except (ValueError, IndexError):
            print("  Skipped.")

    print()


if __name__ == "__main__":
    main()
