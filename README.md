# Trading Toolkit

Python tools for MEXC crypto futures analysis.

## Files

| File | Purpose |
|------|---------|
| `mexc.py` | MEXC REST API client (price, OHLCV, orderbook, 24h stats) |
| `analysis.py` | Technical analysis models (EMA, RSI, ATR, structure, FVGs, key levels) |
| `scanner.py` | Entry scanner — scores long/short setups, suggests SL/TP |

## Quick Usage

```bash
# Scan a symbol (default 1h)
python3 scanner.py ETHUSDT

# Scan with custom interval (1m 5m 15m 30m 1h 4h 1d)
python3 scanner.py BTCUSDT 15m

# Scan multiple default symbols (ETH BTC SOL BNB)
python3 scanner.py

# Raw analysis output
python3 analysis.py
```

## Models Used

- **EMA stack** — 21/55/200 for trend direction
- **RSI** — 14-period, overbought/oversold signals
- **ATR** — 14-period, used for SL/TP sizing
- **Market structure** — swing highs/lows → HH/HL (bull) or LH/LL (bear)
- **FVGs** — Fair Value Gaps for magnet zones
- **Key levels** — clustered swing points → S/R zones

## Scoring (0–10)

Short or long setups are scored by confluence:
- Bias alignment (+2)
- EMA trend match (+2)
- RSI signal (+1–2)
- Market structure (+2)
- Nearby FVG (+1)
- Volume confirmation (+1)
