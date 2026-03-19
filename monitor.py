"""
Live monitor — fast 5-min check on ETH/BTC using lightweight analysis.
Full confluence scan triggered only when conditions look interesting.
Sends Telegram alert on signals.
"""

import time, os, sys
from datetime import datetime, timezone

# Load env
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.openclaw/.env"))

from mexc import get_ohlcv, get_price
from analysis import full_analysis, to_df, atr as calc_atr
from alerts import send_alert

SYMBOLS = ["ETHUSDT", "BTCUSDT"]
INTERVAL = "15m"
CHECK_EVERY = 300  # 5 min
FULL_SCAN_SCORE_THRESHOLD = 3  # trigger full confluence if quick score >= this

last_prices = {}
last_quick_scores = {}

def quick_score(result: dict) -> tuple[int, list]:
    """Fast 0-5 score from basic indicators."""
    score = 0
    reasons = []
    direction = "neutral"

    bias = result["bias"]
    ema = result["ema"]["trend"]
    rsi = result["rsi"]["value"]
    struct = result["structure"]["trend"]

    # Bearish signals
    bear = sum([bias == "bearish", ema == "bearish", struct == "bearish", rsi > 60])
    bull = sum([bias == "bullish", ema == "bullish", struct == "bullish", rsi < 40])

    if bear >= 2:
        score = bear
        direction = "SHORT"
        if bias == "bearish": reasons.append("bearish bias")
        if ema == "bearish": reasons.append("bearish EMA stack")
        if struct == "bearish": reasons.append("bearish structure")
        if rsi > 60: reasons.append(f"RSI {rsi:.0f}")
    elif bull >= 2:
        score = bull
        direction = "LONG"
        if bias == "bullish": reasons.append("bullish bias")
        if ema == "bullish": reasons.append("bullish EMA stack")
        if struct == "bullish": reasons.append("bullish structure")
        if rsi < 40: reasons.append(f"RSI {rsi:.0f}")

    return score, direction, reasons


def check_symbol(symbol: str):
    price = get_price(symbol)
    candles = get_ohlcv(symbol, INTERVAL, 100)
    result = full_analysis(candles, symbol)
    df = to_df(candles)
    atr_val = calc_atr(df).iloc[-1]

    score, direction, reasons = quick_score(result)
    rsi = result["rsi"]["value"]
    ema_trend = result["ema"]["trend"]
    struct = result["structure"]["trend"]

    now = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f"[{now}] {symbol} ${price:,.2f} | {direction} {score}/5 | RSI:{rsi:.1f} EMA:{ema_trend} STR:{struct}")

    # Alert if score crosses threshold
    prev = last_quick_scores.get(symbol, 0)
    last_quick_scores[symbol] = score

    if score >= FULL_SCAN_SCORE_THRESHOLD:
        reasons_str = " | ".join(reasons)
        msg = (
            f"⚡ *{symbol} SETUP* ({INTERVAL})\n"
            f"Direction: *{direction}*  Score: {score}/5\n"
            f"Price: ${price:,.2f}\n"
            f"RSI: {rsi:.1f} | EMA: {ema_trend} | Structure: {struct}\n"
            f"Signals: {reasons_str}\n"
            f"ATR: ${atr_val:.2f}"
        )
        send_alert(msg)
        print(f"  ⚡ ALERT SENT")
    elif prev >= FULL_SCAN_SCORE_THRESHOLD and score < FULL_SCAN_SCORE_THRESHOLD:
        send_alert(f"📉 *{symbol}* signal faded (was {prev}/5, now {score}/5) at ${price:,.2f}")
        print(f"  📉 Signal faded alert sent")


if __name__ == "__main__":
    print(f"🔍 Monitor started | Symbols: {SYMBOLS} | Interval: {INTERVAL} | Every {CHECK_EVERY//60}min")
    print(f"Alert on: quick score >= {FULL_SCAN_SCORE_THRESHOLD}/5")
    print()
    while True:
        for sym in SYMBOLS:
            try:
                check_symbol(sym)
            except Exception as e:
                print(f"  Error [{sym}]: {e}")
        time.sleep(CHECK_EVERY)
