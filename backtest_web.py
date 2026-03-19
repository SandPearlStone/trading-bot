#!/usr/bin/env python3
"""
backtest_web.py — Lightweight backtest web server (no external deps)

Uses Python's built-in http.server + minimal JSON APIs.

Usage:
    python3 backtest_web.py
    
    Then visit: http://localhost:5555
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import json
import os
from backtest import backtest
from watchlist import DEFAULT_SYMBOLS


class BacktestHandler(BaseHTTPRequestHandler):
    """HTTP handler for backtest APIs."""
    
    def do_GET(self):
        """Serve HTML UI or handle GET requests."""
        if self.path == '/':
            self.serve_html()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle API requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body)
        except:
            self.send_json({"ok": False, "error": "Invalid JSON"}, 400)
            return
        
        if self.path == '/api/backtest':
            self.api_backtest(data)
        elif self.path == '/api/batch':
            self.api_batch(data)
        elif self.path == '/api/optimize':
            self.api_optimize(data)
        else:
            self.send_error(404)
    
    def api_backtest(self, data):
        """Single backtest."""
        try:
            symbol = data.get('symbol', 'ETHUSDT').upper()
            interval = data.get('interval', '1h')
            days = int(data.get('days', 90))
            min_score = int(data.get('minScore', 7))
            sl_mult = float(data.get('slMult', 1.5))
            tp_mult = float(data.get('tpMult', 2.0))
            
            # Override params
            import backtest as bt
            orig = (bt.MIN_SCORE, bt.SL_ATR_MULT, bt.TP_ATR_MULT)
            bt.MIN_SCORE = min_score
            bt.SL_ATR_MULT = sl_mult
            bt.TP_ATR_MULT = tp_mult
            
            stats = backtest(symbol, interval, days, min_score=min_score, verbose=False)
            
            bt.MIN_SCORE, bt.SL_ATR_MULT, bt.TP_ATR_MULT = orig
            
            result = {
                "symbol": symbol,
                "interval": interval,
                "days": days,
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
            self.send_json({"ok": True, "result": result})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)
    
    def api_batch(self, data):
        """Batch test multiple symbols."""
        try:
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
            
            self.send_json({"ok": True, "results": results})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)
    
    def api_optimize(self, data):
        """Parameter optimization."""
        try:
            symbol = data.get('symbol', 'ETHUSDT').upper()
            interval = data.get('interval', '1h')
            days = int(data.get('days', 90))
            quick = data.get('quick', False)
            
            min_scores = [5, 6, 7, 8] if quick else [5, 6, 7, 8, 9]
            sl_mults = [1.5, 2.0] if quick else [1.0, 1.5, 2.0]
            tp_mults = [2.0, 3.0] if quick else [1.5, 2.0, 3.0]
            
            results = []
            import backtest as bt
            orig = (bt.MIN_SCORE, bt.SL_ATR_MULT, bt.TP_ATR_MULT)
            
            for min_s in min_scores:
                for sl_m in sl_mults:
                    for tp_m in tp_mults:
                        try:
                            bt.MIN_SCORE = min_s
                            bt.SL_ATR_MULT = sl_m
                            bt.TP_ATR_MULT = tp_m
                            
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
            
            bt.MIN_SCORE, bt.SL_ATR_MULT, bt.TP_ATR_MULT = orig
            
            results = sorted(results, key=lambda r: r['pf'], reverse=True)[:10]
            
            self.send_json({"ok": True, "symbol": symbol, "results": results})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)
    
    def serve_html(self):
        """Serve HTML UI."""
        html = open('templates/backtest.html').read()
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


if __name__ == '__main__':
    port = 5555
    server = HTTPServer(('127.0.0.1', port), BacktestHandler)
    
    print("\n" + "="*60)
    print("  🎯 Backtest GUI Server (No Dependencies)")
    print("="*60)
    print(f"\n  Open your browser: http://localhost:{port}")
    print("  Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()
