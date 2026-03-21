"""
phase5_scanner.py — Market scanner for Phase 5 ML signals.

Scan all symbols:
  1. Fetch latest OHLCV
  2. Compute Phase 1-3 confluence score
  3. Run Phase 5 ML inference
  4. Output clean signal cards (symbol, direction, scores, entry/SL/TP)

Usage:
    python3 phase5_scanner.py [--symbols BTC,ETH,SOL] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase5_feature_builder import Phase5FeatureBuilder
from phase5_inference import Phase5Inference

try:
    from confluence import score_symbol
except ImportError:
    score_symbol = None

try:
    import ccxt
    exchange = ccxt.binance()
except ImportError:
    exchange = None


class Phase5Scanner:
    """Market scanner for Phase 5 ML signals."""
    
    def __init__(self, symbols: Optional[list[str]] = None):
        self.symbols = symbols or [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "ARBUSDT",
            "BNBUSDT", "XRPUSDT", "LINKUSDT", "OPUSDT", "WIFUSDT", "PEPEUSDT", "DOGEUSDT",
        ]
        self.inference = Phase5Inference()
        self.feature_builder = Phase5FeatureBuilder()
    
    def fetch_latest_bars(self, symbol: str, timeframe: str = "4h", limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch latest OHLCV bars from Binance (or fallback to CSV)."""
        if exchange is None:
            # Fallback: load from CSV
            csv_path = f"data/processed/{symbol}_4h_features.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                return df.tail(limit)
            return None
        
        try:
            # Convert 4h to minutes
            since = exchange.parse8601((pd.Timestamp.now() - pd.Timedelta(days=30)).isoformat())
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df["symbol"] = symbol
            
            return df
        except Exception as e:
            print(f"  ⚠️  Failed to fetch {symbol}: {e}")
            return None
    
    def scan_symbol(self, symbol: str) -> dict:
        """Scan single symbol, return signal dict."""
        # Fetch data
        df = self.fetch_latest_bars(symbol)
        if df is None:
            return {
                "symbol": symbol,
                "status": "SKIP",
                "reason": "Data not available",
            }
        
        try:
            # Build features
            df_features = self.feature_builder.build(df)
            
            # Phase 1-3 confluence
            confluence_score = 0.0
            if "confluence" in df_features.columns:
                confluence_score = df_features["confluence"].iloc[-1]
            elif score_symbol is not None:
                try:
                    result = score_symbol(symbol)
                    confluence_score = result.get("score", 0)
                except Exception as e:
                    confluence_score = 0.0
            
            # Phase 5 ML inference
            result = self.inference.infer(df_features, confluence_score=confluence_score)
            
            # Calculate entry, SL, TP (simple ATR-based)
            latest_close = df["close"].iloc[-1]
            atr = df_features.get("atr", pd.Series([latest_close * 0.02])).iloc[-1]
            
            if result["direction"] == "LONG":
                entry = latest_close
                sl = latest_close - 2 * atr
                tp = latest_close + 3 * atr
            elif result["direction"] == "SHORT":
                entry = latest_close
                sl = latest_close + 2 * atr
                tp = latest_close - 3 * atr
            else:
                entry = sl = tp = latest_close
            
            return {
                "symbol": symbol,
                "status": "SIGNAL" if result["should_trade"] else "MONITOR",
                "direction": result["direction"],
                "predicted_return": result["predicted_return"],
                "signal_strength": result["signal_strength"],
                "confluence": confluence_score,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "reason": result["reason"],
                "timestamp": str(datetime.now()),
            }
        
        except Exception as e:
            return {
                "symbol": symbol,
                "status": "ERROR",
                "reason": str(e),
            }
    
    def scan_all(self, symbols: Optional[list[str]] = None) -> list[dict]:
        """Scan all symbols, return list of signal dicts."""
        symbols = symbols or self.symbols
        results = []
        
        print(f"\n[Scanner] Scanning {len(symbols)} symbols...")
        print("=" * 80)
        
        for symbol in symbols:
            result = self.scan_symbol(symbol)
            results.append(result)
            
            # Print compact
            if result["status"] == "SIGNAL":
                print(f"✅ {symbol:12} | {result['direction']:5} | "
                      f"Return: {result['predicted_return']:+.2%} | "
                      f"Confluence: {result['confluence']:.0f}")
            elif result["status"] == "MONITOR":
                print(f"👀 {symbol:12} | {result['direction']:5} | "
                      f"Return: {result.get('predicted_return', 0):+.2%} | "
                      f"Confluence: {result.get('confluence', 0):.0f}")
            else:
                print(f"⚠️  {symbol:12} | {result['status']:5} | {result.get('reason', 'N/A')}")
        
        print("=" * 80)
        
        return results
    
    def format_signal_card(self, result: dict) -> str:
        """Format single signal as card."""
        if result["status"] == "ERROR" or result["status"] == "SKIP":
            return f"[{result['symbol']}] {result['status']}: {result.get('reason', 'N/A')}"
        
        lines = [
            f"╔════════════════════════════════════════╗",
            f"║ {result['symbol']:36} ║",
            f"╠════════════════════════════════════════╣",
            f"║ Status: {result['status']:30} ║",
            f"║ Direction: {result['direction']:27} ║",
            f"║ Return: {result['predicted_return']:+.2%} ({result['signal_strength']:.2%}) {' '*15} ║",
            f"║ Confluence: {result['confluence']:.0f}{' '*28} ║",
            f"╠════════════════════════════════════════╣",
            f"║ Entry: {result['entry']:.2f}{' '*29} ║",
            f"║ SL: {result['sl']:.2f}{' '*32} ║",
            f"║ TP: {result['tp']:.2f}{' '*32} ║",
            f"╚════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 5 ML market scanner")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (default: all)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    scanner = Phase5Scanner(symbols=symbols)
    results = scanner.scan_all()
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        # Print formatted cards
        for result in results:
            if result["status"] == "SIGNAL":
                print(scanner.format_signal_card(result))
                print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Scanner] Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
