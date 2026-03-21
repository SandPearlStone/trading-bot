"""
scripts/ingest_live.py — Live ingestion daemon for Phase 5.5.

Reads ingest_state for each configured (symbol, interval) and fetches
any new candles since last_ts. Designed to run every 5 minutes via cron:

    */5 * * * * cd /path/to/trading && python3 scripts/ingest_live.py

Features:
  - Resume from last_ts in ingest_state
  - INSERT OR IGNORE (idempotent)
  - Retry + exponential backoff on API errors
  - Gap detection: alerts if >2 consecutive candles are missing
  - WAL mode + timeout for concurrent DB access

Usage:
    python3 scripts/ingest_live.py
    python3 scripts/ingest_live.py --db /path/to/market.db
    python3 scripts/ingest_live.py --symbols BTCUSDT --intervals 5m 1h
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRADING = os.path.dirname(_HERE)
DEFAULT_DB = os.path.join(_TRADING, "data", "market.db")
LOGS_DIR = os.path.join(_TRADING, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "ingest_live.log")),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "https://fapi.binance.com"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_INTERVALS = ["5m", "1h", "4h"]

INTERVAL_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

MAX_MISSING_CANDLES_ALERT = 2   # alert if gap > this many candles
LIMIT = 1000                    # max candles per request


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 5) -> Optional[list]:
    for attempt in range(retries):
        backoff = min(2 ** attempt, 60)
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                log.warning(f"Rate limited. Sleeping {retry_after}s")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            log.warning(f"Timeout (attempt {attempt+1}/{retries}), retrying in {backoff}s")
            time.sleep(backoff)
        except requests.RequestException as e:
            log.warning(f"Request error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(backoff)
    log.error(f"All {retries} retries failed for {url}")
    return None


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_con(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def get_last_ts(con: sqlite3.Connection, symbol: str, interval: str) -> Optional[int]:
    """Get last ingested timestamp (ms) for (symbol, interval)."""
    cur = con.execute(
        "SELECT last_ts FROM ingest_state WHERE symbol=? AND interval=?",
        (symbol, interval)
    )
    row = cur.fetchone()
    return row[0] if row else None


def update_state(con: sqlite3.Connection, symbol: str, interval: str, last_ts: int):
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        "INSERT OR REPLACE INTO ingest_state (symbol, interval, last_ts, updated_at) VALUES (?,?,?,?)",
        (symbol, interval, last_ts, now)
    )
    con.commit()


# ── Gap detection ─────────────────────────────────────────────────────────────

def check_for_gaps(rows: list[tuple], interval_ms: int, symbol: str, interval: str):
    """Alert if there are gaps > MAX_MISSING_CANDLES_ALERT between rows."""
    if len(rows) < 2:
        return

    for i in range(1, len(rows)):
        expected_ts = rows[i-1][2] + interval_ms
        actual_ts = rows[i][2]
        gap_candles = (actual_ts - expected_ts) // interval_ms

        if gap_candles > MAX_MISSING_CANDLES_ALERT:
            gap_start = datetime.fromtimestamp(expected_ts / 1000, tz=timezone.utc)
            gap_end = datetime.fromtimestamp(actual_ts / 1000, tz=timezone.utc)
            log.warning(
                f"⚠️  GAP DETECTED {symbol}/{interval}: "
                f"{gap_candles} missing candles "
                f"({gap_start.isoformat()} → {gap_end.isoformat()})"
            )


# ── Core ingestion ────────────────────────────────────────────────────────────

def ingest_symbol_interval(db_path: str, symbol: str, interval: str) -> dict:
    """Fetch and insert new candles for one (symbol, interval) pair."""
    interval_ms = INTERVAL_MS.get(interval)
    if interval_ms is None:
        log.error(f"Unknown interval: {interval}")
        return {"error": f"unknown interval {interval}"}

    con = get_con(db_path)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Resume from last known position
    last_ts = get_last_ts(con, symbol, interval)
    if last_ts:
        since_ms = last_ts + interval_ms
    else:
        # First time: just get last 100 candles
        since_ms = now_ms - (100 * interval_ms)
        log.info(f"{symbol}/{interval}: No state found, fetching last 100 candles")

    if since_ms >= now_ms:
        log.info(f"{symbol}/{interval}: Already up to date (last_ts={last_ts})")
        con.close()
        return {"inserted": 0, "skipped": 0}

    # Fetch new candles
    data = _get(
        f"{BASE_URL}/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "startTime": since_ms, "limit": LIMIT}
    )

    if not data:
        log.warning(f"{symbol}/{interval}: No data returned")
        con.close()
        return {"inserted": 0, "skipped": 0, "warning": "no data"}

    rows = [
        (symbol, interval, int(k[0]), float(k[1]), float(k[2]),
         float(k[3]), float(k[4]), float(k[5]))
        for k in data
    ]

    # Gap check
    check_for_gaps(rows, interval_ms, symbol, interval)

    # Insert
    cur = con.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND interval=?", (symbol, interval)
    )
    before = cur.fetchone()[0]

    con.executemany(
        "INSERT OR IGNORE INTO ohlcv (symbol, interval, open_time, open, high, low, close, volume) "
        "VALUES (?,?,?,?,?,?,?,?)",
        rows
    )
    con.commit()

    after = con.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND interval=?", (symbol, interval)
    ).fetchone()[0]

    inserted = after - before
    skipped = len(rows) - inserted

    # Update state
    last_row_ts = rows[-1][2]
    update_state(con, symbol, interval, last_row_ts)

    last_dt = datetime.fromtimestamp(last_row_ts / 1000, tz=timezone.utc)
    log.info(
        f"{symbol}/{interval}: fetched={len(rows)} inserted={inserted} "
        f"skipped={skipped} last={last_dt.strftime('%Y-%m-%d %H:%M')}"
    )

    con.close()
    return {"inserted": inserted, "skipped": skipped, "fetched": len(rows)}


def main():
    parser = argparse.ArgumentParser(description="Live market data ingestion for Phase 5.5")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--intervals", nargs="+", default=DEFAULT_INTERVALS)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        log.error(f"DB not found: {args.db}. Run db_init.py first.")
        sys.exit(1)

    log.info(
        f"=== ingest_live.py: symbols={args.symbols} intervals={args.intervals} "
        f"@ {datetime.now(timezone.utc).isoformat()} ==="
    )

    total_inserted = 0
    errors = []

    for symbol in args.symbols:
        for interval in args.intervals:
            try:
                result = ingest_symbol_interval(args.db, symbol, interval)
                total_inserted += result.get("inserted", 0)
            except Exception as e:
                log.error(f"ERROR {symbol}/{interval}: {e}")
                errors.append((symbol, interval, str(e)))

            time.sleep(0.2)  # Small delay between API calls

    log.info(f"=== ingest_live.py done: {total_inserted:,} new rows ===")
    if errors:
        log.warning(f"Errors encountered: {errors}")


if __name__ == "__main__":
    main()
