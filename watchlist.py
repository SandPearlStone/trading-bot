"""
watchlist.py — Active opportunity tracker

Runs confluence.score_setup on a list of symbols, filters for Grade A/B
setups, and returns a ranked report sorted by score descending.
"""

from __future__ import annotations

import sys
from typing import Optional

# ── Local imports (graceful) ──────────────────────────────────────────────────
try:
    from confluence import score_setup, score_setup_with_ml
except ImportError:
    print("[watchlist] WARNING: confluence not available", file=sys.stderr)
    score_setup = None  # type: ignore
    score_setup_with_ml = None  # type: ignore

try:
    from entry_finder import find_entry
except ImportError:
    print("[watchlist] WARNING: entry_finder not available", file=sys.stderr)
    find_entry = None  # type: ignore


# ── Default universe ──────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "ARBUSDT", "OPUSDT",
    "PEPEUSDT", "WIFUSDT",
]

PASSING_GRADES = {"A", "B"}


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_watchlist(
    symbols: Optional[list[str]] = None,
    interval: str = "1h",
    higher_tf: str = "4h",
    min_grade: str = "B",
    check_entry: bool = True,
    use_ml: bool = False,
) -> list[dict]:
    """
    Score every symbol in *symbols*, filter by grade >= *min_grade*,
    and return results sorted by score descending.

    Parameters
    ----------
    symbols    : list of base symbols (default: DEFAULT_SYMBOLS)
    interval   : primary timeframe (default "1h")
    higher_tf  : context timeframe (default "4h")
    min_grade  : minimum grade to keep ("A", "B", "C")
    check_entry: also run entry_finder on qualifying setups
    use_ml     : if True, use ML-enhanced scoring (Phase 4)

    Returns
    -------
    list of score dicts (from confluence.score_setup or score_setup_with_ml)
    with optional "entry" key from entry_finder.
    """
    if score_setup is None:
        print("[watchlist] ERROR: confluence module not available")
        return []

    symbols = symbols or DEFAULT_SYMBOLS
    grade_order = {"A": 4, "B": 3, "C": 2, "F": 1}
    min_grade_rank = grade_order.get(min_grade.upper(), 3)

    results = []
    for sym in symbols:
        try:
            print(f"  Scanning {sym}...", end=" ", flush=True)
            # Use ML-enhanced scoring if requested and available
            if use_ml and score_setup_with_ml is not None:
                setup = score_setup_with_ml(sym, interval, higher_tf)
            else:
                setup = score_setup(sym, interval, higher_tf)
            grade_rank = grade_order.get(setup.get("grade", "F"), 1)

            if grade_rank >= min_grade_rank and setup.get("direction") != "NO_TRADE":
                # Optionally check entry timing
                if check_entry and find_entry:
                    ez = setup.get("entry_zone", (None, None))
                    if ez[0] is not None:
                        entry_res = find_entry(sym, setup["direction"], ez, "15m")
                        setup["entry"] = entry_res
                    else:
                        setup["entry"] = None
                else:
                    setup["entry"] = None

                results.append(setup)
                print(f"Grade {setup['grade']} ({setup['score']}/100) {setup['direction']}")
            else:
                print(f"Grade {setup.get('grade','F')} — filtered out")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results


# ── Report formatter ──────────────────────────────────────────────────────────

def format_report(results: list[dict]) -> str:
    """
    Return a clean human-readable string summarising the top setups.
    """
    if not results:
        return "No qualifying setups found.\n"

    lines: list[str] = []
    lines.append(f"{'═'*50}")
    lines.append(f"  WATCHLIST SCAN — {len(results)} setup(s) found")
    lines.append(f"{'═'*50}")

    for i, r in enumerate(results, 1):
        sym   = r.get("symbol", "?")
        dirn  = r.get("direction", "?")
        grade = r.get("grade", "?")
        score = r.get("score", 0)
        price = r.get("details", {}).get("price", 0)
        ez    = r.get("entry_zone", (None, None))
        sl    = r.get("stop_loss")
        tp1   = r.get("tp1")
        tp2   = r.get("tp2")
        tp3   = r.get("tp3")
        rr    = r.get("rr_ratio")
        inv   = r.get("invalidation")

        entry_info = r.get("entry")
        entry_status = ""
        if entry_info:
            if entry_info.get("ready"):
                entry_status = f"  ✅ Entry timing: READY ({entry_info['confidence']}%)"
            else:
                entry_status = f"  ⏳ Entry timing: WAITING ({entry_info['confidence']}%)"

        lines.append("")
        lines.append(f"  #{i}  {sym}/USDT | {dirn} | Grade: {grade} ({score}/100)")
        lines.append(f"  Price: ${price:,.4f}")
        lines.append("")
        if ez[0]:
            lines.append(f"  Entry zone:    ${ez[0]:,.4f} – ${ez[1]:,.4f}")
            lines.append(f"  Optimal entry: ${r.get('optimal_entry', 0):,.4f}")
        if sl:
            sl_pct = abs(r.get("optimal_entry", price) - sl) / r.get("optimal_entry", price) * 100
            lines.append(f"  Stop loss:     ${sl:,.4f}  ({sl_pct:.2f}%)")
        if tp1:
            lines.append(f"  TP1: ${tp1:,.4f}  (1.5R)")
        if tp2:
            lines.append(f"  TP2: ${tp2:,.4f}  (3.0R)")
        if tp3:
            lines.append(f"  TP3: ${tp3:,.4f}  (key level)")
        if rr:
            lines.append(f"  R:R: 1:{rr}")
        if entry_status:
            lines.append(entry_status)
        lines.append("")

        for reason in r.get("confluence_reasons", []):
            lines.append(f"  ✅ {reason}")
        for m in r.get("missing", [])[:4]:  # cap at 4 for readability
            lines.append(f"  ❌ {m}")

        if inv:
            lines.append(f"\n  Invalidation: ${inv:,.4f}")
        lines.append(f"{'─'*50}")

    return "\n".join(lines)


def format_telegram(results: list[dict]) -> str:
    """Return an HTML-formatted Telegram message for the top setups."""
    if not results:
        return "No qualifying setups found."

    lines = ["<b>📊 Watchlist Scan Results</b>"]
    for r in results[:5]:  # cap at 5 for Telegram
        sym   = r.get("symbol", "?")
        dirn  = r.get("direction", "?")
        grade = r.get("grade", "?")
        score = r.get("score", 0)
        price = r.get("details", {}).get("price", 0)
        sl    = r.get("stop_loss")
        tp1   = r.get("tp1")
        rr    = r.get("rr_ratio")

        icon = "🟢" if dirn == "LONG" else "🔴"
        lines.append(
            f"\n{icon} <b>{sym}/USDT</b> | {dirn} | Grade <b>{grade}</b> ({score}/100)\n"
            f"   Price: ${price:,.4f}"
        )
        if sl:
            lines.append(f"   SL: ${sl:,.4f} | TP1: ${tp1:,.4f}" + (f" | R:R 1:{rr}" if rr else ""))
        top_reasons = r.get("confluence_reasons", [])[:3]
        for reason in top_reasons:
            lines.append(f"   ✅ {reason}")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan watchlist for top setups")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to scan")
    parser.add_argument("--interval", default="1h", help="Primary timeframe")
    parser.add_argument("--higher-tf", default="4h", help="Higher timeframe")
    parser.add_argument("--min-grade", default="B", help="Minimum grade (A/B/C)")
    parser.add_argument("--no-entry", action="store_true", help="Skip entry timing check")
    args = parser.parse_args()

    print(f"\nScanning {len(args.symbols or DEFAULT_SYMBOLS)} symbols on {args.interval}...\n")
    results = scan_watchlist(
        symbols=args.symbols,
        interval=args.interval,
        higher_tf=args.higher_tf,
        min_grade=args.min_grade,
        check_entry=not args.no_entry,
    )

    print("\n" + format_report(results))
