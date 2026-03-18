"""
Telegram alert system for trading signals.

Loads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment or .env file.
Uses the Bot API sendMessage endpoint directly (no external library needed).

Usage:
    from alerts import send_alert, check_and_alert
    send_alert("Test message")
    check_and_alert(["ETHUSDT", "BTCUSDT"], "1h", score_threshold=7)
"""

import os
import requests
from typing import Optional

# ─── Env / .env loading ────────────────────────────────────────────────────────

def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (if not already set)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

TELEGRAM_BOT_TOKEN: Optional[str] = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID:   Optional[str] = os.environ.get("TELEGRAM_CHAT_ID")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# ─── Core send ─────────────────────────────────────────────────────────────────

def send_alert(text: str, parse_mode: str = "HTML") -> bool:
    """
    Send a Telegram message.
    Returns True on success, False on failure.
    Raises RuntimeError if credentials are missing.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. "
            "Set them in your environment or .env file."
        )
    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[alerts] Telegram send failed: {e}")
        return False


# ─── Format helpers ────────────────────────────────────────────────────────────

def _format_signal_alert(scan: dict, direction: str) -> str:
    """
    Build a nicely formatted HTML Telegram message from a scan_symbol() result.

    direction: "LONG" or "SHORT"
    """
    sym   = scan["symbol"]
    price = scan["price"]
    tf    = scan["interval"]

    score_key = "long_score" if direction == "LONG" else "short_score"
    score_data = scan[score_key]
    score = score_data["score"]
    max_s = score_data["max"]

    sug = scan["suggestion"][direction.lower()]
    sl  = sug["sl"]
    tp1 = sug["tp1"]
    tp2 = sug["tp2"]

    # Risk:Reward
    risk   = abs(price - sl)
    rr1    = round(abs(tp1 - price) / risk, 2) if risk else 0
    rr2    = round(abs(tp2 - price) / risk, 2) if risk else 0

    arrow = "🟢" if direction == "LONG" else "🔴"
    lines = [
        f"{arrow} <b>{sym} — {direction} SIGNAL</b>  [{tf}]",
        f"Score: <b>{score}/{max_s}</b>",
        f"Price:  <code>{price}</code>",
        f"SL:     <code>{sl}</code>",
        f"TP1:    <code>{tp1}</code>  (R:R {rr1})",
        f"TP2:    <code>{tp2}</code>  (R:R {rr2})",
        f"Bias:   {scan['bias'].upper()}  |  Structure: {scan['structure']['trend'].upper()}",
        f"RSI:    {scan['rsi']['value']}  |  EMA trend: {scan['ema']['trend']}",
    ]

    # Attach top reasons (strip emoji prefix for brevity)
    reasons = score_data.get("reasons", [])
    if reasons:
        lines.append("")
        lines.append("<i>Reasons:</i>")
        for r in reasons[:5]:
            lines.append(f"  {r}")

    # MTF confluence if present
    mtf = scan.get("mtf")
    if mtf:
        lines.append("")
        lines.append(f"MTF confluence: {mtf.get('confluence_score', '?')}/{mtf.get('total_timeframes', '?')} timeframes agree ({mtf.get('bias', '?')})")

    return "\n".join(lines)


# ─── High-level scanner + alert ────────────────────────────────────────────────

def check_and_alert(
    symbols: list[str],
    interval: str = "1h",
    score_threshold: int = 7,
    dry_run: bool = False,
) -> list[dict]:
    """
    Scan each symbol and send a Telegram alert if long_score or short_score
    meets or exceeds score_threshold.

    dry_run=True prints the message instead of sending.

    Returns list of triggered alerts: [{symbol, direction, score, message}]
    """
    from scanner import scan_symbol  # local import to avoid circular deps

    triggered = []
    for sym in symbols:
        try:
            scan = scan_symbol(sym, interval)
        except Exception as e:
            print(f"[alerts] Error scanning {sym}: {e}")
            continue

        for direction in ("LONG", "SHORT"):
            key   = f"{direction.lower()}_score"
            score = scan[key]["score"]
            if score >= score_threshold:
                msg = _format_signal_alert(scan, direction)
                if dry_run:
                    print(f"\n--- DRY RUN [{sym} {direction}] ---\n{msg}\n")
                else:
                    ok = send_alert(msg)
                    status = "sent" if ok else "failed"
                    print(f"[alerts] {sym} {direction} score={score} → alert {status}")
                triggered.append({
                    "symbol":    sym,
                    "direction": direction,
                    "score":     score,
                    "message":   msg,
                })

    return triggered


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    symbols  = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ETHUSDT", "BTCUSDT"]
    interval = sys.argv[2] if len(sys.argv) > 2 else "1h"
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 7

    print(f"Checking {symbols} on {interval}, threshold={threshold} (dry_run=True)...")
    results = check_and_alert(symbols, interval, threshold, dry_run=True)
    print(f"\n{len(results)} alert(s) would have been sent.")
