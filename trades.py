"""
trades.py — Trade journal

SQLite-backed trade log with open/close lifecycle, PnL tracking,
and performance statistics.

CLI:
    python3 trades.py add
    python3 trades.py close <id>
    python3 trades.py open
    python3 trades.py stats [--days 30]
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Optional

# ── Database setup ────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "trades.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,          -- LONG / SHORT
    entry_price     REAL    NOT NULL,
    sl              REAL    NOT NULL,
    tp1             REAL,
    tp2             REAL,
    tp3             REAL,
    size_usdt       REAL    DEFAULT 0,
    leverage        REAL    DEFAULT 1,
    confluence_score INTEGER DEFAULT 0,
    grade           TEXT    DEFAULT '',
    setup_notes     TEXT    DEFAULT '',
    entry_time      TEXT    NOT NULL,
    exit_price      REAL,
    exit_time       TEXT,
    pnl_pct         REAL,
    pnl_usdt        REAL,
    result          TEXT    DEFAULT 'open',    -- open / win / loss / be
    screenshot_path TEXT    DEFAULT ''
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _calc_pnl(direction: str, entry: float, exit_price: float, size_usdt: float, leverage: float) -> tuple[float, float]:
    """Return (pnl_pct, pnl_usdt)."""
    if direction.upper() == "LONG":
        pnl_pct = (exit_price - entry) / entry * 100 * leverage
    else:
        pnl_pct = (entry - exit_price) / entry * 100 * leverage
    pnl_usdt = size_usdt * pnl_pct / 100
    return round(pnl_pct, 4), round(pnl_usdt, 4)


def _result_label(pnl_pct: float, be_threshold: float = 0.1) -> str:
    if pnl_pct > be_threshold:
        return "win"
    if pnl_pct < -be_threshold:
        return "loss"
    return "be"


# ── CRUD operations ───────────────────────────────────────────────────────────

def add_trade(
    symbol: str,
    direction: str,
    entry_price: float,
    sl: float,
    tp1: Optional[float] = None,
    tp2: Optional[float] = None,
    tp3: Optional[float] = None,
    size_usdt: float = 0.0,
    leverage: float = 1.0,
    confluence_score: int = 0,
    grade: str = "",
    setup_notes: str = "",
    entry_time: Optional[str] = None,
    screenshot_path: str = "",
) -> int:
    """
    Log a new trade. Returns the new trade id.
    """
    entry_time = entry_time or _now_utc()
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO trades (
                symbol, direction, entry_price, sl, tp1, tp2, tp3,
                size_usdt, leverage, confluence_score, grade, setup_notes,
                entry_time, screenshot_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol.upper(), direction.upper(), entry_price, sl,
                tp1, tp2, tp3, size_usdt, leverage,
                confluence_score, grade, setup_notes,
                entry_time, screenshot_path,
            ),
        )
        conn.commit()
        return cur.lastrowid


def close_trade(
    trade_id: int,
    exit_price: float,
    exit_time: Optional[str] = None,
) -> Optional[dict]:
    """
    Mark a trade as closed. Calculates PnL and returns the updated row.
    """
    exit_time = exit_time or _now_utc()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        if not row:
            print(f"[trades] Trade #{trade_id} not found")
            return None
        if row["result"] != "open":
            print(f"[trades] Trade #{trade_id} already closed ({row['result']})")
            return None

        pnl_pct, pnl_usdt = _calc_pnl(
            row["direction"], row["entry_price"], exit_price,
            row["size_usdt"] or 0, row["leverage"] or 1,
        )
        result = _result_label(pnl_pct)

        conn.execute(
            """
            UPDATE trades
               SET exit_price = ?, exit_time = ?,
                   pnl_pct = ?, pnl_usdt = ?, result = ?
             WHERE id = ?
            """,
            (exit_price, exit_time, pnl_pct, pnl_usdt, result, trade_id),
        )
        conn.commit()

        updated = conn.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        return dict(updated)


def open_trades() -> list[dict]:
    """Return all currently open trades."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE result = 'open' ORDER BY entry_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_trade(trade_id: int) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None


def all_trades(days: int = 0) -> list[dict]:
    """Return all trades, optionally limited to the last *days* days."""
    with _get_conn() as conn:
        if days > 0:
            rows = conn.execute(
                """
                SELECT * FROM trades
                 WHERE entry_time >= datetime('now', ?)
                 ORDER BY entry_time DESC
                """,
                (f"-{days} days",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC"
            ).fetchall()
        return [dict(r) for r in rows]


# ── Statistics ────────────────────────────────────────────────────────────────

def stats(days: int = 30) -> dict:
    """
    Compute performance statistics over the last *days* days.

    Returns dict with:
        total, wins, losses, breakevens, open_count,
        win_rate_pct, avg_pnl_pct, avg_rr,
        profit_factor, total_pnl_usdt,
        best_trade, worst_trade
    """
    trades = all_trades(days)
    closed = [t for t in trades if t["result"] != "open"]

    wins       = [t for t in closed if t["result"] == "win"]
    losses     = [t for t in closed if t["result"] == "loss"]
    breakevens = [t for t in closed if t["result"] == "be"]
    opens      = [t for t in trades if t["result"] == "open"]

    win_rate = len(wins) / len(closed) * 100 if closed else 0.0

    avg_pnl = (
        sum(t["pnl_pct"] or 0 for t in closed) / len(closed)
        if closed else 0.0
    )

    gross_profit = sum(t["pnl_usdt"] or 0 for t in wins)
    gross_loss   = abs(sum(t["pnl_usdt"] or 0 for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = sum(t["pnl_usdt"] or 0 for t in closed)

    best  = max(closed, key=lambda t: t["pnl_pct"] or 0) if closed else None
    worst = min(closed, key=lambda t: t["pnl_pct"] or 0) if closed else None

    # Average R:R: estimate from entry/sl/exit
    rr_values = []
    for t in wins + losses:
        e = t.get("entry_price") or 0
        s = t.get("sl") or 0
        x = t.get("exit_price") or 0
        if e and s and x and abs(e - s) > 0:
            risk   = abs(e - s)
            reward = abs(x - e)
            rr_values.append(reward / risk)
    avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0.0

    return {
        "period_days":    days,
        "total":          len(trades),
        "closed":         len(closed),
        "wins":           len(wins),
        "losses":         len(losses),
        "breakevens":     len(breakevens),
        "open_count":     len(opens),
        "win_rate_pct":   round(win_rate, 1),
        "avg_pnl_pct":    round(avg_pnl, 2),
        "avg_rr":         round(avg_rr, 2),
        "profit_factor":  round(profit_factor, 2),
        "total_pnl_usdt": round(total_pnl, 2),
        "best_trade":     best,
        "worst_trade":    worst,
    }


def format_stats(days: int = 30) -> str:
    """Return a Telegram-friendly stats string."""
    s = stats(days)
    lines = [
        f"<b>📈 Trade Stats — Last {s['period_days']} Days</b>",
        f"",
        f"Total trades:   {s['total']}  ({s['open_count']} open)",
        f"Closed:         {s['closed']}  (W:{s['wins']} / L:{s['losses']} / BE:{s['breakevens']})",
        f"Win rate:       {s['win_rate_pct']}%",
        f"Avg PnL:        {s['avg_pnl_pct']:+.2f}%",
        f"Avg R:R:        1:{s['avg_rr']}",
        f"Profit factor:  {s['profit_factor']}",
        f"Total PnL:      ${s['total_pnl_usdt']:+,.2f}",
    ]
    if s["best_trade"]:
        b = s["best_trade"]
        lines.append(f"Best trade:     {b['symbol']} {b['direction']} {b['pnl_pct']:+.2f}%")
    if s["worst_trade"]:
        w = s["worst_trade"]
        lines.append(f"Worst trade:    {w['symbol']} {w['direction']} {w['pnl_pct']:+.2f}%")
    return "\n".join(lines)


def _print_trade(t: dict) -> None:
    dirn_icon = "🟢" if t["direction"] == "LONG" else "🔴"
    result_icons = {"win": "✅", "loss": "❌", "be": "〰️", "open": "🔵"}
    icon = result_icons.get(t["result"], "?")
    pnl_str = f"{t['pnl_pct']:+.2f}%" if t["pnl_pct"] is not None else "open"
    print(
        f"  [{t['id']:>3}] {dirn_icon} {t['symbol']:<6} {t['direction']:<5}  "
        f"Entry: ${t['entry_price']:<12,.4f}  "
        f"SL: ${t['sl']:<12,.4f}  "
        f"{icon} {pnl_str:>8}  "
        f"Grade: {t['grade'] or '—'}  "
        f"{t['entry_time'][:16]}"
    )


# ── Interactive prompts ───────────────────────────────────────────────────────

def _prompt_add() -> None:
    """Interactive prompt to add a new trade."""
    print("\nAdd new trade (press Enter to skip optional fields)\n")

    def ask(prompt: str, cast=str, required: bool = True) -> Optional[object]:
        while True:
            val = input(f"  {prompt}: ").strip()
            if not val:
                if required:
                    print("  (required)")
                    continue
                return None
            try:
                return cast(val)
            except ValueError:
                print(f"  Invalid input, expected {cast.__name__}")

    symbol    = ask("Symbol (e.g. BTC)")
    direction = ask("Direction (LONG/SHORT)").upper()
    entry     = ask("Entry price", float)
    sl        = ask("Stop loss", float)
    tp1       = ask("TP1 (optional)", float, required=False)
    tp2       = ask("TP2 (optional)", float, required=False)
    tp3       = ask("TP3 (optional)", float, required=False)
    size      = ask("Size USDT (optional, 0=skip)", float, required=False) or 0.0
    leverage  = ask("Leverage (default 1)", float, required=False) or 1.0
    score     = ask("Confluence score (optional)", int, required=False) or 0
    grade     = ask("Grade (A/B/C/F, optional)", str, required=False) or ""
    notes     = ask("Setup notes (optional)", str, required=False) or ""

    trade_id = add_trade(
        symbol=symbol, direction=direction, entry_price=entry, sl=sl,
        tp1=tp1, tp2=tp2, tp3=tp3, size_usdt=size, leverage=leverage,
        confluence_score=score, grade=grade, setup_notes=notes,
    )
    print(f"\n  ✅ Trade #{trade_id} logged.")


def _prompt_close(trade_id: int) -> None:
    t = get_trade(trade_id)
    if not t:
        print(f"Trade #{trade_id} not found.")
        return
    print(f"\n  Closing trade #{trade_id}: {t['symbol']} {t['direction']} @ ${t['entry_price']}")
    exit_price = float(input("  Exit price: ").strip())
    result = close_trade(trade_id, exit_price)
    if result:
        icon = "✅" if result["result"] == "win" else ("❌" if result["result"] == "loss" else "〰️")
        print(f"\n  {icon} Closed: {result['pnl_pct']:+.2f}% (${result['pnl_usdt']:+,.2f})")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Trade journal CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("add",   help="Log a new trade interactively")
    close_p = sub.add_parser("close", help="Close a trade by ID")
    close_p.add_argument("id", type=int, help="Trade ID")
    sub.add_parser("open",  help="List open trades")
    stats_p = sub.add_parser("stats", help="Show performance stats")
    stats_p.add_argument("--days", type=int, default=30)
    list_p = sub.add_parser("list",  help="List all trades")
    list_p.add_argument("--days", type=int, default=0)

    args = parser.parse_args()

    if args.command == "add":
        _prompt_add()

    elif args.command == "close":
        _prompt_close(args.id)

    elif args.command == "open":
        opens = open_trades()
        if not opens:
            print("No open trades.")
        else:
            print(f"\nOpen trades ({len(opens)}):")
            for t in opens:
                _print_trade(t)

    elif args.command == "stats":
        s = stats(args.days)
        print(f"\nStats — Last {s['period_days']} days")
        print(f"  Trades:        {s['total']}  ({s['open_count']} open, {s['closed']} closed)")
        print(f"  Win/Loss/BE:   {s['wins']}/{s['losses']}/{s['breakevens']}")
        print(f"  Win rate:      {s['win_rate_pct']}%")
        print(f"  Avg PnL:       {s['avg_pnl_pct']:+.2f}%")
        print(f"  Avg R:R:       1:{s['avg_rr']}")
        print(f"  Profit factor: {s['profit_factor']}")
        print(f"  Total PnL:     ${s['total_pnl_usdt']:+,.2f}")
        if s["best_trade"]:
            b = s["best_trade"]
            print(f"  Best:          {b['symbol']} {b['direction']} {b['pnl_pct']:+.2f}%")
        if s["worst_trade"]:
            w = s["worst_trade"]
            print(f"  Worst:         {w['symbol']} {w['direction']} {w['pnl_pct']:+.2f}%")

    elif args.command == "list":
        all_t = all_trades(args.days)
        if not all_t:
            print("No trades found.")
        else:
            print(f"\nAll trades ({len(all_t)}):")
            for t in all_t:
                _print_trade(t)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
