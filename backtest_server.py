#!/usr/bin/env python3
"""
backtest_server.py — Web GUI for backtesting

Flask server with real-time backtest execution, param tuning, and results visualization.

Usage:
    python3 backtest_server.py
    
    Then visit: http://localhost:5555

Features:
  - Single/batch symbol testing
  - Real-time param tuning (min_score, SL mult, TP mult)
  - Results table with sortable columns
  - Trade list with P&L details
  - Profit charts
  - Parameter optimization grid
"""

from flask import Flask, render_template, request, jsonify
import json
import os
from datetime import datetime
from backtest import backtest
from watchlist import DEFAULT_SYMBOLS


app = Flask(__name__, static_folder='templates/static', template_folder='templates')

# Store results in memory for session
RESULTS = {}
OPTIMIZATION_RESULTS = {}


@app.route('/')
def index():
    """Main backtest UI."""
    return render_template('backtest.html', symbols=DEFAULT_SYMBOLS)


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Run a single backtest."""
    data = request.json
    symbol = data.get('symbol', 'ETHUSDT').upper()
    interval = data.get('interval', '1h')
    days = int(data.get('days', 90))
    min_score = int(data.get('min_score', 7))
    sl_mult = float(data.get('sl_mult', 1.5))
    tp_mult = float(data.get('tp_mult', 2.0))
    
    try:
        # Override backtest params
        import backtest as bt_module
        orig_min, orig_sl, orig_tp = bt_module.MIN_SCORE, bt_module.SL_ATR_MULT, bt_module.TP_ATR_MULT
        
        bt_module.MIN_SCORE = min_score
        bt_module.SL_ATR_MULT = sl_mult
        bt_module.TP_ATR_MULT = tp_mult
        
        stats = backtest(symbol, interval, days, min_score=min_score, verbose=False)
        
        bt_module.MIN_SCORE, bt_module.SL_ATR_MULT, bt_module.TP_ATR_MULT = orig_min, orig_sl, orig_tp
        
        # Store in session
        key = f"{symbol}_{interval}_{days}_{min_score}_{sl_mult}_{tp_mult}"
        RESULTS[key] = stats
        
        # Format for response
        resp = {
            "symbol": symbol,
            "interval": interval,
            "days": days,
            "min_score": min_score,
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "total_trades": stats.get('total_trades', 0),
            "closed_trades": stats.get('closed_trades', 0),
            "wins": stats.get('wins', 0),
            "losses": stats.get('losses', 0),
            "win_rate": stats.get('win_rate_pct', 0),
            "avg_r": stats.get('avg_r_multiple', 0),
            "profit_factor": stats.get('profit_factor', 0),
            "net_r": stats.get('net_r', 0),
            "long_trades": stats.get('long_trades', 0),
            "short_trades": stats.get('short_trades', 0),
            "long_wr": stats.get('long_win_rate', 0),
            "short_wr": stats.get('short_win_rate', 0),
            "best_trade": stats.get('best_trade'),
            "worst_trade": stats.get('worst_trade'),
        }
        
        return jsonify({"ok": True, "result": resp})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route('/api/batch', methods=['POST'])
def api_batch():
    """Run batch backtest on multiple symbols."""
    data = request.json
    symbols = data.get('symbols', DEFAULT_SYMBOLS)
    interval = data.get('interval', '1h')
    days = int(data.get('days', 90))
    
    results = []
    for symbol in symbols:
        try:
            stats = backtest(symbol, interval, days, verbose=False)
            results.append({
                "symbol": symbol,
                "total_trades": stats.get('total_trades'),
                "win_rate": stats.get('win_rate_pct'),
                "net_r": stats.get('net_r'),
                "profit_factor": stats.get('profit_factor'),
            })
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})
    
    return jsonify({"ok": True, "results": results})


@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """Grid-search parameter optimization."""
    data = request.json
    symbol = data.get('symbol', 'ETHUSDT').upper()
    interval = data.get('interval', '1h')
    days = int(data.get('days', 90))
    quick = data.get('quick', False)
    
    min_scores = [5, 6, 7, 8] if quick else [5, 6, 7, 8, 9]
    sl_mults = [1.5, 2.0] if quick else [1.0, 1.5, 2.0]
    tp_mults = [2.0, 3.0] if quick else [1.5, 2.0, 3.0]
    
    results = []
    import backtest as bt_module
    orig_min, orig_sl, orig_tp = bt_module.MIN_SCORE, bt_module.SL_ATR_MULT, bt_module.TP_ATR_MULT
    
    for min_s in min_scores:
        for sl_m in sl_mults:
            for tp_m in tp_mults:
                try:
                    bt_module.MIN_SCORE = min_s
                    bt_module.SL_ATR_MULT = sl_m
                    bt_module.TP_ATR_MULT = tp_m
                    
                    stats = backtest(symbol, interval, days, min_score=min_s, verbose=False)
                    
                    if "error" not in stats:
                        results.append({
                            "min_score": min_s,
                            "sl_mult": sl_m,
                            "tp_mult": tp_m,
                            "trades": stats['total_trades'],
                            "wr": stats['win_rate_pct'],
                            "net_r": stats['net_r'],
                            "pf": stats['profit_factor'],
                        })
                except:
                    pass
    
    bt_module.MIN_SCORE, bt_module.SL_ATR_MULT, bt_module.TP_ATR_MULT = orig_min, orig_sl, orig_tp
    
    # Sort by PF
    results = sorted(results, key=lambda r: r['pf'], reverse=True)
    
    return jsonify({"ok": True, "symbol": symbol, "results": results[:10]})


@app.route('/api/trades/<result_key>')
def api_trades(result_key):
    """Get detailed trade list for a backtest result."""
    if result_key not in RESULTS:
        return jsonify({"ok": False, "error": "Result not found"})
    
    stats = RESULTS[result_key]
    trades = stats.get('trades', [])
    
    # Format trades
    formatted = []
    for t in trades:
        formatted.append({
            "time": t.get('time'),
            "direction": t.get('direction'),
            "entry": t.get('entry'),
            "sl": t.get('sl'),
            "tp": t.get('tp'),
            "exit_price": t.get('exit_price'),
            "outcome": t.get('outcome'),
            "r_multiple": t.get('r_multiple'),
        })
    
    return jsonify({"ok": True, "trades": formatted})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🎯 Backtest GUI Server")
    print("="*60)
    print("\n  Open your browser: http://localhost:5555")
    print("  Ctrl+C to stop\n")
    
    app.run(host='127.0.0.1', port=5555, debug=False)
