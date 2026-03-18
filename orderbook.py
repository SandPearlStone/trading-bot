"""
Orderbook analysis — detect walls, imbalance, significant support/resistance.

Uses get_orderbook() from mexc.py.

Usage:
    from orderbook import analyze_orderbook
    result = analyze_orderbook("ETHUSDT", limit=50)
"""

from mexc import get_orderbook
import statistics


# ─── Constants ────────────────────────────────────────────────────────────────

# A level is a "wall" if its size is >= this multiple of the median level size
WALL_THRESHOLD_MULTIPLIER = 3.0

# Number of top walls to return
MAX_WALLS = 5


# ─── Core analysis ────────────────────────────────────────────────────────────

def analyze_orderbook(symbol: str, limit: int = 50) -> dict:
    """
    Fetch and analyse the orderbook for `symbol`.

    Returns:
        {
          "symbol":         str,
          "bid_walls":      [ {price, size, dist_pct}, ... ],   # large buy orders (support)
          "ask_walls":      [ {price, size, dist_pct}, ... ],   # large sell orders (resistance)
          "imbalance":      float,      # bid_volume / ask_volume  (>1 = more bids = bullish pressure)
          "total_bid_vol":  float,
          "total_ask_vol":  float,
          "best_bid":       float,
          "best_ask":       float,
          "spread_pct":     float,
          "nearest_support":    float | None,   # price of nearest bid wall
          "nearest_resistance": float | None,   # price of nearest ask wall
        }
    """
    book = get_orderbook(symbol, limit)
    bids: list[tuple[float, float]] = book["bids"]  # (price, size), sorted best→worst
    asks: list[tuple[float, float]] = book["asks"]  # (price, size), sorted best→worst

    if not bids or not asks:
        return {"symbol": symbol, "error": "empty orderbook"}

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    spread_pct = round((best_ask - best_bid) / mid_price * 100, 4)

    # Aggregate totals
    total_bid_vol = sum(q for _, q in bids)
    total_ask_vol = sum(q for _, q in asks)
    imbalance = round(total_bid_vol / total_ask_vol, 4) if total_ask_vol else 0.0

    # Detect walls: levels with size >= WALL_THRESHOLD_MULTIPLIER * median
    bid_walls = _find_walls(bids, mid_price, "bid")
    ask_walls = _find_walls(asks, mid_price, "ask")

    # Nearest significant wall
    nearest_support    = bid_walls[0]["price"]  if bid_walls else None
    nearest_resistance = ask_walls[0]["price"]  if ask_walls else None

    return {
        "symbol":             symbol,
        "best_bid":           best_bid,
        "best_ask":           best_ask,
        "spread_pct":         spread_pct,
        "mid_price":          round(mid_price, 6),
        "total_bid_vol":      round(total_bid_vol, 4),
        "total_ask_vol":      round(total_ask_vol, 4),
        "imbalance":          imbalance,          # >1 bullish, <1 bearish
        "imbalance_signal":   _imbalance_signal(imbalance),
        "bid_walls":          bid_walls,
        "ask_walls":          ask_walls,
        "nearest_support":    nearest_support,
        "nearest_resistance": nearest_resistance,
    }


def _find_walls(
    levels: list[tuple[float, float]],
    mid_price: float,
    side: str,
) -> list[dict]:
    """
    Identify wall levels (abnormally large orders).

    side: "bid" or "ask" — used only for distance sign convention.
    """
    if not levels:
        return []

    sizes = [q for _, q in levels]
    if len(sizes) < 2:
        return []

    med = statistics.median(sizes)
    if med == 0:
        return []

    walls = []
    for price, size in levels:
        if size >= med * WALL_THRESHOLD_MULTIPLIER:
            dist_pct = round((price - mid_price) / mid_price * 100, 4)
            walls.append({
                "price":    price,
                "size":     round(size, 4),
                "dist_pct": dist_pct,
                "ratio":    round(size / med, 2),  # how many times bigger than median
            })

    # Sort: bids by price descending (nearest first), asks by price ascending
    reverse = (side == "bid")
    walls.sort(key=lambda x: x["price"], reverse=reverse)
    return walls[:MAX_WALLS]


def _imbalance_signal(ratio: float) -> str:
    if ratio >= 2.0:
        return "strong_bullish"
    elif ratio >= 1.3:
        return "bullish"
    elif ratio <= 0.5:
        return "strong_bearish"
    elif ratio <= 0.77:
        return "bearish"
    else:
        return "neutral"


# ─── Pretty print ─────────────────────────────────────────────────────────────

def print_orderbook_report(ob: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  Orderbook: {ob['symbol']}")
    print(f"  Bid: {ob['best_bid']}  Ask: {ob['best_ask']}  Spread: {ob['spread_pct']}%")
    print(f"  Imbalance: {ob['imbalance']}  ({ob['imbalance_signal']})")
    print(f"  Total bid vol: {ob['total_bid_vol']}  |  Total ask vol: {ob['total_ask_vol']}")

    if ob.get("bid_walls"):
        print(f"\n  BID WALLS (support):")
        for w in ob["bid_walls"]:
            print(f"    ${w['price']}  size={w['size']}  ({w['dist_pct']:+.2f}%)  {w['ratio']}x median")

    if ob.get("ask_walls"):
        print(f"\n  ASK WALLS (resistance):")
        for w in ob["ask_walls"]:
            print(f"    ${w['price']}  size={w['size']}  ({w['dist_pct']:+.2f}%)  {w['ratio']}x median")

    if ob.get("nearest_support"):
        print(f"\n  Nearest support wall:    ${ob['nearest_support']}")
    if ob.get("nearest_resistance"):
        print(f"  Nearest resistance wall: ${ob['nearest_resistance']}")
    print(f"{'='*50}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "ETHUSDT"
    limit  = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"Analysing orderbook for {symbol} (depth={limit})...")
    result = analyze_orderbook(symbol, limit)
    print_orderbook_report(result)
