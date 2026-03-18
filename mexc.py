"""
MEXC API client — fetch OHLCV, ticker, orderbook
MEXC uses Binance-compatible REST API (v3)
"""

import requests
from datetime import datetime

BASE_URL = "https://api.mexc.com"

INTERVAL_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "60m",
    "4h":  "4h",
    "1d":  "1d",
    "1w":  "1W",
}

def get_price(symbol: str = "ETHUSDT") -> float:
    """Get current price for a symbol."""
    r = requests.get(f"{BASE_URL}/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])


def get_ohlcv(symbol: str = "ETHUSDT", interval: str = "1h", limit: int = 200) -> list[dict]:
    """
    Fetch OHLCV candles from MEXC.
    Returns list of dicts: {time, open, high, low, close, volume}
    """
    mapped = INTERVAL_MAP.get(interval, interval)
    r = requests.get(
        f"{BASE_URL}/api/v3/klines",
        params={"symbol": symbol, "interval": mapped, "limit": limit},
        timeout=10
    )
    r.raise_for_status()
    raw = r.json()
    candles = []
    for c in raw:
        candles.append({
            "time":   datetime.utcfromtimestamp(c[0] / 1000),
            "open":   float(c[1]),
            "high":   float(c[2]),
            "low":    float(c[3]),
            "close":  float(c[4]),
            "volume": float(c[5]),
        })
    return candles


def get_orderbook(symbol: str = "ETHUSDT", limit: int = 20) -> dict:
    """Fetch top N bids/asks."""
    r = requests.get(f"{BASE_URL}/api/v3/depth", params={"symbol": symbol, "limit": limit}, timeout=10)
    r.raise_for_status()
    data = r.json()
    return {
        "bids": [(float(p), float(q)) for p, q in data["bids"]],
        "asks": [(float(p), float(q)) for p, q in data["asks"]],
    }


def get_24h(symbol: str = "ETHUSDT") -> dict:
    """24h stats: price change, high, low, volume."""
    r = requests.get(f"{BASE_URL}/api/v3/ticker/24hr", params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    d = r.json()
    return {
        "symbol":       d["symbol"],
        "price":        float(d["lastPrice"]),
        "change_pct":   float(d["priceChangePercent"]),
        "high":         float(d["highPrice"]),
        "low":          float(d["lowPrice"]),
        "volume_usdt":  float(d["quoteVolume"]),
    }


if __name__ == "__main__":
    print("ETH price:", get_price("ETHUSDT"))
    stats = get_24h("ETHUSDT")
    print(f"24h: {stats['change_pct']:+.2f}%  H:{stats['high']}  L:{stats['low']}")
