#!/home/sandro/trading-venv/bin/python3
"""Full system backtest: A (Random) vs B (P1-3) vs C (P1-3 + ML)."""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import sqlite3
from freqai_features import TradingFeatures
from freqai_labels import TradingLabels
from freqai_model import TradingModel


def load_data(symbol, interval):
    """Load OHLCV from SQLite market.db. Falls back to CSV if DB fails."""
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'market.db')
    try:
        conn = sqlite3.connect(db_path)
        if interval == '15m':
            query = "SELECT open_time as timestamp, open, high, low, close, volume FROM ohlcv_15m WHERE symbol=? ORDER BY open_time"
        else:
            query = "SELECT open_time as timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND interval=? ORDER BY open_time"
        
        if interval == '15m':
            df = pd.read_sql_query(query, conn, params=(symbol,))
        else:
            df = pd.read_sql_query(query, conn, params=(symbol, interval))
        conn.close()
        
        if len(df) > 0:
            return df
    except Exception as e:
        pass
    
    # Fallback to CSV
    csv_path = f"data/binance_raw/{symbol}_{interval}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'open_time' in df.columns:
            df = df.rename(columns={'open_time': 'timestamp'})
        # Convert ISO strings to int ms if needed
        if df['timestamp'].dtype == object:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**6
        return df
    return None


def approx_confluence(row):
    """Approximate Phase 1-3 confluence score (0-100)."""
    score = 0
    rsi = row.get('rsi_14', 50)
    adx = row.get('adx_14', 20)
    bb_pct = row.get('bb_pct_b', 0.5)
    ema_ratio = row.get('ema_8_21_ratio', 1.0)
    vol_ratio = row.get('vol_ratio_20', 1.0)
    rsi_lag5 = row.get('rsi_14_lag5', 50)

    if rsi < 30 or rsi > 70:
        score += 20
    elif rsi < 40 or rsi > 60:
        score += 10
    if adx > 25:
        score += 15
    elif adx > 20:
        score += 5
    if bb_pct < 0.1 or bb_pct > 0.9:
        score += 15
    elif bb_pct < 0.2 or bb_pct > 0.8:
        score += 5
    if ema_ratio > 1.002 or ema_ratio < 0.998:
        score += 15
    if vol_ratio > 1.5:
        score += 15
    elif vol_ratio > 1.2:
        score += 5
    rsi_change = rsi - rsi_lag5
    if abs(rsi_change) > 10:
        score += 10
    return min(score, 100)


def approx_direction(row):
    """Approximate direction from features."""
    rsi = row.get('rsi_14', 50)
    ema = row.get('ema_8_21_ratio', 1.0)
    if rsi < 40 and ema < 0.999:
        return 'SHORT'
    elif rsi > 60 and ema > 1.001:
        return 'LONG'
    elif ema > 1.001:
        return 'LONG'
    elif ema < 0.999:
        return 'SHORT'
    return 'HOLD'


def simulate_trade(df, i, direction, atr, max_bars=20):
    """Simulate trade from bar i. Return PnL or None if no exit."""
    entry_price = df['close'].iloc[i]
    sl_dist = 1.5 * atr
    tp_dist = 2.5 * atr

    if direction == 'LONG':
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    elif direction == 'SHORT':
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist
    else:
        return None

    for j in range(i + 1, min(i + max_bars + 1, len(df))):
        if direction == 'LONG':
            if df['low'].iloc[j] <= sl:
                return -sl_dist / entry_price - 0.001
            if df['high'].iloc[j] >= tp:
                return tp_dist / entry_price - 0.001
        elif direction == 'SHORT':
            if df['high'].iloc[j] >= sl:
                return -sl_dist / entry_price - 0.001
            if df['low'].iloc[j] <= tp:
                return tp_dist / entry_price - 0.001

    # Timeout: close at bar i+max_bars
    close_idx = min(i + max_bars, len(df) - 1)
    if direction == 'LONG':
        return (df['close'].iloc[close_idx] - entry_price) / entry_price - 0.001
    else:
        return (entry_price - df['close'].iloc[close_idx]) / entry_price - 0.001


def compute_metrics(trades):
    """Compute metrics from list of PnL values."""
    if not trades:
        return {'n': 0, 'win_rate': 0, 'avg_return': 0, 'sharpe': 0, 'max_dd': 0, 'pf': 0}
    trades = np.array(trades)
    n = len(trades)
    win_rate = (trades > 0).sum() / n if n > 0 else 0
    avg_return = trades.mean()
    sharpe = (trades.mean() / trades.std() * np.sqrt(252)) if trades.std() > 0 else 0
    cum_pnl = np.cumsum(trades)
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = (cum_pnl - peak).min() * 100 if len(cum_pnl) > 0 else 0
    wins = trades[trades > 0].sum()
    losses = abs(trades[trades < 0].sum())
    pf = wins / losses if losses > 0 else 0
    return {'n': n, 'win_rate': win_rate, 'avg_return': avg_return, 'sharpe': sharpe, 'max_dd': max_dd, 'pf': pf}


def backtest_symbol(symbol, n_forward):
    """Run all 3 strategies on symbol."""
    df_1h = load_data(symbol, '1h')
    if df_1h is None or len(df_1h) < 100:
        print(f"❌ {symbol}: Not enough data")
        return None

    df_4h = load_data(symbol, '4h')
    df_15m = load_data(symbol, '15m')

    # Compute features
    feat_engine = TradingFeatures()
    features = feat_engine.compute(df_1h, df_15m, df_4h, None)
    features['confluence_score'] = features.apply(approx_confluence, axis=1)

    # Compute labels
    label_engine = TradingLabels()
    labels = label_engine.compute(df_1h, n_forward)

    # Train/test split: 70/30
    split_idx = int(len(df_1h) * 0.7)
    features_train = features.iloc[:split_idx]
    labels_train = labels.iloc[:split_idx]
    features_test = features.iloc[split_idx:]
    labels_test = labels.iloc[split_idx:]
    df_test = df_1h.iloc[split_idx:].reset_index(drop=True)
    features_test = features_test.reset_index(drop=True)

    # Train ML model
    model = TradingModel()
    print(f"Training ML model on {symbol}...")
    model.train(features_train, labels_train)

    # === Strategy A: Random ===
    np.random.seed(42)
    trades_a = []
    i = 0
    while i < len(df_test) - 20:
        direction = np.random.choice(['LONG', 'SHORT'])
        atr = features_test['atr_14'].iloc[i]
        if atr > 0:
            pnl = simulate_trade(df_test, i, direction, atr)
            if pnl is not None:
                trades_a.append(pnl)
        i += 20

    # === Strategy B: P1-3 Only ===
    trades_b = []
    for i in range(len(df_test)):
        score = features_test['confluence_score'].iloc[i]
        if score >= 40:
            direction = approx_direction(features_test.iloc[i])
            atr = features_test['atr_14'].iloc[i]
            if direction != 'HOLD' and atr > 0:
                pnl = simulate_trade(df_test, i, direction, atr)
                if pnl is not None:
                    trades_b.append(pnl)

    # === Strategy C: P1-3 + ML ===
    trades_c = []
    for i in range(len(features_test)):
        score = features_test['confluence_score'].iloc[i]
        if score >= 40:
            pred = model.predict(features_test.iloc[i:i+1])
            ml_ret = pred['expected_return']
            if abs(ml_ret) > 0.002:
                direction = 'LONG' if ml_ret > 0 else 'SHORT'
                atr = features_test['atr_14'].iloc[i]
                if atr > 0:
                    pnl = simulate_trade(df_test, i, direction, atr)
                    if pnl is not None:
                        trades_c.append(pnl)

    # Compute metrics
    m_a = compute_metrics(trades_a)
    m_b = compute_metrics(trades_b)
    m_c = compute_metrics(trades_c)

    return {
        'symbol': symbol,
        'n_bars': len(df_1h),
        'n_days': len(df_1h) / 24,
        'a': m_a, 'b': m_b, 'c': m_c
    }


def print_results(results):
    """Print backtest results."""
    for res in results:
        symbol = res['symbol']
        n_bars = res['n_bars']
        n_days = res['n_days']
        a, b, c = res['a'], res['b'], res['c']

        print("\n" + "=" * 70)
        print(f"BACKTEST RESULTS — {symbol} 1h ({n_bars} bars, ~{n_days:.0f} days)")
        print("=" * 70)
        print()
        print(f"Strategy A (Random):")
        print(f"  Trades: {a['n']:3d}  |  Win Rate: {a['win_rate']:5.1%}  |  Sharpe: {a['sharpe']:6.2f}  |  Max DD: {a['max_dd']:6.1f}%  |  PF: {a['pf']:5.2f}")
        print()
        print(f"Strategy B (P1-3 Only):")
        print(f"  Trades: {b['n']:3d}  |  Win Rate: {b['win_rate']:5.1%}  |  Sharpe: {b['sharpe']:6.2f}  |  Max DD: {b['max_dd']:6.1f}%  |  PF: {b['pf']:5.2f}")
        print()
        print(f"Strategy C (P1-3 + ML):")
        print(f"  Trades: {c['n']:3d}  |  Win Rate: {c['win_rate']:5.1%}  |  Sharpe: {c['sharpe']:6.2f}  |  Max DD: {c['max_dd']:6.1f}%  |  PF: {c['pf']:5.2f}")
        print()

        delta_wr = c['win_rate'] - b['win_rate']
        delta_s = c['sharpe'] - b['sharpe']
        delta_dd = c['max_dd'] - b['max_dd']

        print(f"ML Improvement over P1-3:")
        print(f"  Win Rate: {delta_wr:+.1%}  |  Sharpe: {delta_s:+.2f}  |  DD: {delta_dd:+.1f}%")
        print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full System Backtest')
    parser.add_argument('--symbols', type=str, default='BTCUSDT,ETHUSDT,SOLUSDT',
                        help='Comma-separated symbols')
    parser.add_argument('--n-forward', type=int, default=10,
                        help='Label lookahead (bars)')
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]
    results = []

    for symbol in symbols:
        print(f"\n📊 Backtesting {symbol}...")
        res = backtest_symbol(symbol, args.n_forward)
        if res:
            results.append(res)

    print_results(results)
    print("\n✅ backtest_system.py DONE")
