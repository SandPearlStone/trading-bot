# Trading System

Crypto futures trading toolkit with ML-assisted signal generation.

## Architecture

```
Live Market Data (Binance)
        ↓
SQLite Database (market.db: 1.1M+ bars)
        ↓
Feature Engineering (47 features, TA-Lib)
        ↓
P1-3 Confluence Gate (score ≥ 40)
        ↓
ML Filter (LGBMRegressor, |return| > 0.2%)
        ↓
LLM Review (contextual analysis)
        ↓
Human Decision (manual execution on MEXC)
        ↓
Trade Logger (trades.db)
        ↓
Weekly Retraining (model improves over time)
```

## Performance (Backtest)

**FreqTrade backtest: Jul 2024 → Mar 2026 (20 months, 10 pairs)**

| Metric | Value |
|--------|-------|
| Starting Balance | $1,000 |
| Final Balance | $2,125 |
| Total Profit | +112.5% |
| CAGR | 57.2% |
| Sharpe | 8.40 |
| Sortino | 55.27 |
| Max Drawdown | 17.08% |
| Profit Factor | 1.32 |
| Trades | 2,024 (~3.3/day) |
| Win Rate | 44.6% |
| All pairs profitable | ✅ |

Market dropped -23% during this period. System was net profitable through bear market.

## Core Files

### Trading Tools
| File | Description |
|------|-------------|
| `trading_tools.py` | Live scanning: price, TA, confluence, funding, OI |
| `confluence.py` | Phase 1-3 confluence scoring (FVG, OB, regime, divergence) |
| `divergence_detector.py` | Hidden divergence detection |
| `kelly_calculator.py` | Kelly criterion position sizing |

### ML Pipeline
| File | Description |
|------|-------------|
| `freqai_features.py` | 47-feature stable contract (TA-Lib based) |
| `freqai_labels.py` | Forward return labels with IQR filtering |
| `freqai_model.py` | LGBMRegressor with walk-forward validation |
| `backtest_system.py` | Custom backtest (P1-3 + ML vs baselines) |

### FreqTrade Integration
| File | Description |
|------|-------------|
| `freqtrade_userdata/strategies/MainStrategy.py` | FreqAI strategy with P1-3 gate + Kelly |
| `freqtrade_userdata/config_backtest.json` | Backtest config (10 pairs, futures) |

### Data
| Source | Location | Size |
|--------|----------|------|
| OHLCV 1h/4h | `data/market.db` (ohlcv table) | 262K bars |
| OHLCV 15m | `data/market.db` (ohlcv_15m table) | 840K bars |
| OHLCV 5m | `data/market.db` (ohlcv table) | 420K bars |
| Funding rates | `data/market.db` (funding_rates) | 4.3K rows |
| Open interest | `data/market.db` (open_interest) | 1.3K rows |
| Trade log | `trades.db` | User trades |

## Quick Start

### Prerequisites
```bash
# Python venv with all deps
source /home/sandro/trading-venv/bin/activate

# Key packages: freqtrade, lightgbm, xgboost, ta-lib, ccxt, pandas-ta
```

### Scan Market
```bash
python3 -c "
from trading_tools import scan_market
results = scan_market(['BTCUSDT','ETHUSDT','SOLUSDT'])
for r in results:
    print(f\"{r['symbol']}: \${r['price']} | RSI={r['rsi_1h']:.0f} | Score={r['confluence_score']} | Grade={r['confluence_grade']}\")
"
```

### Run Backtest
```bash
# Custom backtest (fast)
python3 backtest_system.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

# FreqTrade backtest (full, with walk-forward ML retraining)
freqtrade backtesting \
  --config freqtrade_userdata/config_backtest.json \
  --strategy MainStrategy \
  --freqaimodel LightGBMRegressor \
  --strategy-path freqtrade_userdata/strategies \
  --userdir freqtrade_userdata \
  --timerange 20240701-20260301
```

### Live Data Ingestion
```bash
# Cron (already configured):
# */5 * * * * python3 scripts/ingest_live.py
```

## Feature Contract (47 features)

| Group | Features | Count |
|-------|----------|-------|
| Base TA (1h) | RSI, MACD, ATR, BB, ADX, EMA ratios, vol, ROC, stoch, MFI | 19 |
| Shifted | T-1, T-2, T-5 lags for key indicators | 10 |
| 15m context | RSI, EMA slope, vol spike, momentum, trend | 5 |
| 4h context | RSI, trend, ADX, ATR% | 4 |
| BTC context | RSI, trend, ATR%, ROC | 4 |
| Temporal | hour/dow sin/cos | 4 |
| Confluence | Approximated P1-3 score | 1 |

## Symbols
BTC, ETH, SOL, BNB, XRP, DOGE, AVAX, LINK, ARB, OP, PEPE, WIF

## Stack
- **FreqTrade 2026.2** + FreqAI
- **LightGBM** (primary model)
- **TA-Lib** + pandas-ta (indicators)
- **CCXT** (exchange connectivity)
- **SQLite** (data storage)
- **Kelly criterion** (position sizing, f*/4)

## License
Private. Not for redistribution.
