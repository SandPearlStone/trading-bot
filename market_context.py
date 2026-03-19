"""
Hourly market context generator.
Produces ~/workspace/trading/context.md with:
  - Live prices + 24h stats for top symbols
  - Quick technical bias (1h + 4h) per symbol
  - Recent crypto news headlines (via web search)
  - Fear & Greed index
  - Open trades summary
"""

import os, sys, json, sqlite3, requests
from datetime import datetime, timezone
from pathlib import Path

# Load env
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.openclaw/.env"))

BASE_DIR = Path(__file__).parent
CONTEXT_FILE = BASE_DIR / "context.md"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

from mexc import get_24h, get_ohlcv
from analysis import full_analysis


def get_fear_greed():
    """Fetch Fear & Greed index from alternative.me."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        d = r.json()["data"][0]
        return {"value": d["value"], "label": d["value_classification"]}
    except:
        return {"value": "?", "label": "unavailable"}


def get_btc_dominance():
    """BTC dominance from CoinGecko."""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=8)
        dom = r.json()["data"]["market_cap_percentage"]["btc"]
        return round(dom, 1)
    except:
        return None


def quick_bias(symbol: str) -> dict:
    """Fast 1h + 4h bias without heavy confluence scoring."""
    try:
        c1h = get_ohlcv(symbol, "1h", 100)
        r1h = full_analysis(c1h, symbol)
        c4h = get_ohlcv(symbol, "4h", 100)
        r4h = full_analysis(c4h, symbol)
        return {
            "1h": {"bias": r1h["bias"], "rsi": round(r1h["rsi"]["value"], 1), "ema": r1h["ema"]["trend"]},
            "4h": {"bias": r4h["bias"], "rsi": round(r4h["rsi"]["value"], 1), "ema": r4h["ema"]["trend"]},
            "structure_1h": r1h["structure"]["trend"],
            "key_levels": r1h["key_levels"][:3],
        }
    except Exception as e:
        return {"error": str(e)}


def get_open_trades() -> list:
    db = BASE_DIR / "trades.db"
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        "SELECT id, symbol, direction, entry_price, sl, tp1, leverage, entry_time FROM trades WHERE result='open'"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "symbol": r[1], "direction": r[2], "entry": r[3],
             "sl": r[4], "tp1": r[5], "leverage": r[6], "entry_time": r[7]} for r in rows]


def build_context():
    now = datetime.now(timezone.utc)
    lines = []
    lines.append(f"# Market Context\n_Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}_\n")

    # Fear & Greed
    fg = get_fear_greed()
    btc_dom = get_btc_dominance()
    lines.append("## Sentiment")
    lines.append(f"- **Fear & Greed:** {fg['value']}/100 — {fg['label']}")
    if btc_dom:
        lines.append(f"- **BTC Dominance:** {btc_dom}%")
    lines.append("")

    # Market overview
    lines.append("## Market Overview")
    for sym in SYMBOLS:
        try:
            stats = get_24h(sym)
            bias = quick_bias(sym)
            change_emoji = "📈" if stats["change_pct"] > 0 else "📉"
            lines.append(f"\n### {sym.replace('USDT','')} — ${stats['price']:,.2f}  {change_emoji} {stats['change_pct']:+.2f}%")
            lines.append(f"24h: H ${stats['high']:,.2f}  L ${stats['low']:,.2f}  Vol ${stats['volume_usdt']/1e6:.0f}M")
            if "error" not in bias:
                b1 = bias["1h"]
                b4 = bias["4h"]
                lines.append(f"1H: {b1['bias'].upper()} | RSI {b1['rsi']} | EMA {b1['ema']} | Structure {bias['structure_1h']}")
                lines.append(f"4H: {b4['bias'].upper()} | RSI {b4['rsi']} | EMA {b4['ema']}")
                if bias["key_levels"]:
                    lvls = " | ".join([f"{l['type']} ${l['level']} ({l['dist_pct']:+.1f}%)" for l in bias["key_levels"]])
                    lines.append(f"Levels: {lvls}")
        except Exception as e:
            lines.append(f"\n### {sym} — error: {e}")
    lines.append("")

    # Open trades
    open_trades = get_open_trades()
    lines.append("## Open Trades")
    if open_trades:
        for t in open_trades:
            try:
                current = get_24h(t["symbol"])["price"]
                pnl = ((t["entry"] - current) / t["entry"] * 100) if t["direction"] == "short" else \
                      ((current - t["entry"]) / t["entry"] * 100)
                pnl_lev = pnl * (t["leverage"] or 1)
                lines.append(f"- **#{t['id']} {t['symbol']} {t['direction'].upper()} {t['leverage']}x**")
                lines.append(f"  Entry ${t['entry']}  SL ${t['sl']}  TP1 ${t['tp1']}")
                lines.append(f"  Current ${current:,.2f}  PnL {pnl:+.2f}% ({pnl_lev:+.1f}% levered)")
            except:
                lines.append(f"- #{t['id']} {t['symbol']} {t['direction'].upper()}")
    else:
        lines.append("- No open trades")
    lines.append("")

    # News placeholder — filled by agent during cron run
    lines.append("## Crypto News & Sentiment")
    lines.append("_Run `python3 market_context.py --with-news` or let the hourly cron agent fill this._")
    lines.append("")
    lines.append("<!-- NEWS_PLACEHOLDER -->")

    CONTEXT_FILE.write_text("\n".join(lines))
    print(f"Context written to {CONTEXT_FILE}")
    return str(CONTEXT_FILE)


if __name__ == "__main__":
    build_context()
