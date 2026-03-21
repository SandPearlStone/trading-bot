"""
phase5/data_fetcher.py — Download 2-year OHLCV data from Binance via ccxt.

Usage:
    python3 data_fetcher.py                     # all symbols, 1h+4h
    python3 data_fetcher.py --symbols BTCUSDT   # single symbol
    python3 data_fetcher.py --tf 1h 4h          # specific timeframes
    python3 data_fetcher.py --validate           # check alignment only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

# ── Local config ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    SYMBOLS, TIMEFRAMES, LOOKBACK_DAYS, DATA_DIR, LOGS_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "data_fetcher.log")),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _csv_path(symbol: str, tf: str) -> str:
    safe = symbol.replace("/", "")
    return os.path.join(DATA_DIR, f"{safe}_{tf}.csv")


def _tf_to_ms(tf: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    return int(tf[:-1]) * units[tf[-1]]


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    tf: str,
    since_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Paginate through Binance OHLCV and return a DataFrame."""
    rows = []
    tf_ms = _tf_to_ms(tf)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while since_ms < now_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, tf, since=since_ms, limit=limit)
        except ccxt.RateLimitExceeded:
            log.warning("Rate limit hit — sleeping 30s")
            time.sleep(30)
            continue
        except ccxt.NetworkError as e:
            log.warning(f"Network error ({e}) — retry in 5s")
            time.sleep(5)
            continue
        except Exception as e:
            log.error(f"fetch_ohlcv error: {e}")
            break

        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + tf_ms
        time.sleep(exchange.rateLimit / 1000)

        if len(batch) < limit:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["symbol"] = symbol
    df["timeframe"] = tf
    return df


def download_symbol(exchange: ccxt.Exchange, symbol: str, tf: str, days: int) -> pd.DataFrame:
    """Download or update local cache for one symbol+timeframe."""
    path = _csv_path(symbol, tf)
    since_ms: int

    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["timestamp"])
        last_ts = pd.to_datetime(existing["timestamp"].max(), utc=True)
        since_ms = int(last_ts.timestamp() * 1000) + _tf_to_ms(tf)
        log.info(f"{symbol}/{tf}: resuming from {last_ts.date()} ({len(existing)} rows existing)")
    else:
        existing = pd.DataFrame()
        since_dt = datetime.now(timezone.utc) - timedelta(days=days)
        since_ms = int(since_dt.timestamp() * 1000)
        log.info(f"{symbol}/{tf}: full download from {since_dt.date()}")

    df_new = fetch_ohlcv(exchange, symbol, tf, since_ms)

    if df_new.empty:
        log.info(f"{symbol}/{tf}: no new data")
        return existing if not existing.empty else pd.DataFrame()

    if not existing.empty:
        df = pd.concat([existing, df_new], ignore_index=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    else:
        df = df_new

    df.to_csv(path, index=False)
    log.info(f"{symbol}/{tf}: saved {len(df)} rows → {path}")
    return df


def validate_alignment(symbols: list[str], timeframes: list[str]) -> dict:
    """Check that all symbols have aligned timestamps and no large gaps."""
    results = {}
    for tf in timeframes:
        dfs = {}
        for sym in symbols:
            path = _csv_path(sym, tf)
            if os.path.exists(path):
                dfs[sym] = pd.read_csv(path, parse_dates=["timestamp"])

        if not dfs:
            results[tf] = {"status": "missing", "symbols": []}
            continue

        counts = {s: len(df) for s, df in dfs.items()}
        date_ranges = {
            s: (df["timestamp"].min().strftime("%Y-%m-%d"), df["timestamp"].max().strftime("%Y-%m-%d"))
            for s, df in dfs.items()
        }

        # Check gap: max consecutive missing candles
        tf_ms = _tf_to_ms(tf)
        gaps = {}
        for sym, df in dfs.items():
            ts = pd.to_datetime(df["timestamp"]).sort_values()
            diffs = ts.diff().dropna()
            max_gap = diffs.max()
            expected = pd.Timedelta(milliseconds=tf_ms)
            if max_gap > expected * 3:
                gaps[sym] = str(max_gap)

        results[tf] = {
            "status": "ok" if not gaps else "gaps_detected",
            "row_counts": counts,
            "date_ranges": date_ranges,
            "gaps": gaps,
        }
        log.info(f"[{tf}] Validation: {len(dfs)} symbols, gaps={gaps or 'none'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Download Binance OHLCV data")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--tf", nargs="+", default=TIMEFRAMES, dest="timeframes")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        results = validate_alignment(args.symbols, args.timeframes)
        for tf, info in results.items():
            print(f"\n[{tf}] {info['status'].upper()}")
            for sym, cnt in info.get("row_counts", {}).items():
                rng = info["date_ranges"].get(sym, ("?", "?"))
                print(f"  {sym}: {cnt:,} rows  {rng[0]} → {rng[1]}")
            if info.get("gaps"):
                print(f"  GAPS: {info['gaps']}")
        return

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    total = len(args.symbols) * len(args.timeframes)
    done = 0
    for sym in args.symbols:
        for tf in args.timeframes:
            log.info(f"[{done+1}/{total}] {sym} {tf}")
            try:
                df = download_symbol(exchange, sym, tf, args.days)
                if not df.empty:
                    log.info(f"  ✓ {len(df):,} rows")
            except Exception as e:
                log.error(f"  ✗ {sym}/{tf}: {e}")
            done += 1

    log.info("Download complete. Running validation...")
    validate_alignment(args.symbols, args.timeframes)
    log.info("Done.")


if __name__ == "__main__":
    main()
