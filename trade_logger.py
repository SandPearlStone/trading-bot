"""
trade_logger.py — Log user-executed trades to SQLite with full metadata.

Stores in trades.db:
  - trade_id (auto-increment)
  - symbol, direction, entry_price
  - SL, TP, position_size
  - entry_time, exit_time, exit_price
  - P&L, P&L%, duration
  - model_confidence (what model predicted)
  - user_notes (approval comments)
  - regime_at_entry (CHOPPY/NORMAL/TRENDING)

Usage:
    from trade_logger import TradeLogger
    
    logger = TradeLogger()
    
    # Log entry
    trade_id = logger.log_entry(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=70730,
        sl=69500,
        tp=72950,
        position_size=0.5,
        model_confidence=0.82,
        user_notes="Approved signal per dashboard review",
        regime='TRENDING',
    )
    print(f"Trade {trade_id} logged")
    
    # Log exit
    logger.log_exit(
        trade_id=trade_id,
        exit_price=72800,
        exit_notes="TP hit, closed position",
    )
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

log = logging.getLogger(__name__)

# Database path
_HERE = os.path.dirname(os.path.abspath(__file__))
TRADES_DB_PATH = os.path.join(_HERE, "trades.db")


class TradeLogger:
    """Log user-executed trades to SQLite."""

    def __init__(self, db_path: str = TRADES_DB_PATH):
        """
        Initialize logger.
        
        Args:
            db_path: Path to trades.db
        """
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self):
        """Create phase55_trades table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Create table if not exists (use phase55_trades to avoid conflicts)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS phase55_trades (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
                    entry_price REAL NOT NULL,
                    sl REAL NOT NULL,
                    tp REAL NOT NULL,
                    position_size REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    exit_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    duration_minutes INTEGER,
                    model_confidence REAL NOT NULL,
                    user_notes TEXT,
                    regime_at_entry TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create index on symbol for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_phase55_symbol ON phase55_trades(symbol)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_phase55_entry_time ON phase55_trades(entry_time)
            """)
            
            conn.commit()
            conn.close()
            log.info(f"✓ Phase 5.5 trade schema ready at {self.db_path}")
        
        except Exception as e:
            log.error(f"Error creating schema: {e}")
            raise

    def log_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        position_size: float,
        model_confidence: float,
        user_notes: str = "",
        regime: str = "UNKNOWN",
    ) -> int:
        """
        Log a new trade entry.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            sl: Stop loss price
            tp: Take profit price
            position_size: Position size (0-1, fraction of capital)
            model_confidence: Model confidence (0-1)
            user_notes: User's approval notes
            regime: Market regime at entry
        
        Returns:
            trade_id (auto-increment)
        
        Raises:
            ValueError: if validation fails
        """
        
        # Validate inputs
        if direction not in ('LONG', 'SHORT'):
            raise ValueError(f"Invalid direction: {direction}")
        if not 0 < entry_price:
            raise ValueError(f"Invalid entry_price: {entry_price}")
        if not 0 <= position_size <= 1:
            raise ValueError(f"Invalid position_size: {position_size}")
        if not 0 <= model_confidence <= 1:
            raise ValueError(f"Invalid model_confidence: {model_confidence}")
        
        # Validate SL/TP
        if direction == 'LONG':
            if not sl < entry_price:
                raise ValueError(f"LONG: SL ({sl}) must be below entry ({entry_price})")
            if not tp > entry_price:
                raise ValueError(f"LONG: TP ({tp}) must be above entry ({entry_price})")
        else:  # SHORT
            if not sl > entry_price:
                raise ValueError(f"SHORT: SL ({sl}) must be above entry ({entry_price})")
            if not tp < entry_price:
                raise ValueError(f"SHORT: TP ({tp}) must be below entry ({entry_price})")
        
        now = datetime.utcnow().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO phase55_trades (
                    symbol, direction, entry_price, sl, tp, position_size,
                    entry_time, model_confidence, user_notes, regime_at_entry,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, direction, entry_price, sl, tp, position_size,
                now, model_confidence, user_notes, regime,
                now, now
            ))
            
            trade_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            log.info(f"✓ Trade {trade_id} logged: {symbol} {direction} @ {entry_price}")
            return trade_id
        
        except Exception as e:
            log.error(f"Error logging entry: {e}")
            raise

    def log_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_notes: str = "",
    ) -> Dict[str, Any]:
        """
        Log trade exit and calculate P&L.
        
        Args:
            trade_id: Trade ID from log_entry()
            exit_price: Exit price
            exit_notes: Exit notes (TP hit, SL hit, manual close, etc.)
        
        Returns:
            Dict with trade details and P&L
        
        Raises:
            ValueError: if trade not found
        """
        
        now = datetime.utcnow().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get existing trade
            cursor.execute("SELECT * FROM phase55_trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Trade {trade_id} not found")
            
            # Extract entry data
            direction = row['direction']
            entry_price = row['entry_price']
            entry_time = row['entry_time']
            position_size = row['position_size']
            
            # Calculate P&L
            if direction == 'LONG':
                pnl = exit_price - entry_price
            else:  # SHORT
                pnl = entry_price - exit_price
            
            pnl_pct = pnl / entry_price
            
            # Calculate duration
            entry_dt = datetime.fromisoformat(entry_time)
            exit_dt = datetime.fromisoformat(now)
            duration_minutes = int((exit_dt - entry_dt).total_seconds() / 60)
            
            # Update trade
            cursor.execute("""
                UPDATE phase55_trades SET
                    exit_time = ?,
                    exit_price = ?,
                    pnl = ?,
                    pnl_pct = ?,
                    duration_minutes = ?,
                    user_notes = COALESCE(user_notes || '; ', '') || ?,
                    updated_at = ?
                WHERE trade_id = ?
            """, (
                now, exit_price, pnl, pnl_pct, duration_minutes,
                exit_notes, now, trade_id
            ))
            
            conn.commit()
            conn.close()
            
            result = {
                'trade_id': trade_id,
                'symbol': row['symbol'],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration_minutes': duration_minutes,
                'position_size': position_size,
            }
            
            log.info(f"✓ Trade {trade_id} closed: {pnl:+.2f} ({pnl_pct:+.1%})")
            return result
        
        except Exception as e:
            log.error(f"Error logging exit: {e}")
            raise

    def get_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Get trade details by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM phase55_trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()
            conn.close()
            
            return dict(row) if row else None
        
        except Exception as e:
            log.error(f"Error fetching trade: {e}")
            return None

    def get_recent_trades(self, limit: int = 10, symbol: Optional[str] = None) -> list[Dict[str, Any]]:
        """Get recent trades (optionally filtered by symbol)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute(
                    "SELECT * FROM phase55_trades WHERE symbol = ? ORDER BY entry_time DESC LIMIT ?",
                    (symbol, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM phase55_trades ORDER BY entry_time DESC LIMIT ?",
                    (limit,)
                )
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        
        except Exception as e:
            log.error(f"Error fetching trades: {e}")
            return []

    def get_closed_trades(self, limit: Optional[int] = None, symbol: Optional[str] = None) -> list[Dict[str, Any]]:
        """Get all closed trades (with exit_price set)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if symbol:
                query = "SELECT * FROM phase55_trades WHERE symbol = ? AND exit_price IS NOT NULL ORDER BY exit_time DESC"
                cursor.execute(query, (symbol,))
            else:
                query = "SELECT * FROM phase55_trades WHERE exit_price IS NOT NULL ORDER BY exit_time DESC"
                cursor.execute(query)
            
            rows = cursor.fetchall()
            
            if limit:
                rows = rows[:limit]
            
            conn.close()
            return [dict(row) for row in rows]
        
        except Exception as e:
            log.error(f"Error fetching closed trades: {e}")
            return []

    def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get trade statistics."""
        closed = self.get_closed_trades(symbol=symbol)
        
        if not closed:
            return {
                'total_trades': 0,
                'closed_trades': 0,
                'open_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_pnl': 0.0,
            }
        
        wins = [t['pnl'] for t in closed if t['pnl'] > 0]
        losses = [t['pnl'] for t in closed if t['pnl'] < 0]
        
        return {
            'total_trades': len(closed),
            'closed_trades': len(closed),
            'open_trades': 0,  # Could query for exits where exit_price IS NULL
            'win_rate': len(wins) / len(closed) if closed else 0.0,
            'wins': len(wins),
            'losses': len(losses),
            'avg_win': sum(wins) / len(wins) if wins else 0.0,
            'avg_loss': sum(losses) / len(losses) if losses else 0.0,
            'total_pnl': sum(t['pnl'] for t in closed),
            'total_pnl_pct': sum(t['pnl_pct'] for t in closed),
        }


def main():
    """Test trade logger."""
    logging.basicConfig(level=logging.INFO)
    
    logger = TradeLogger()
    
    # Log a sample entry
    print("\n📝 Logging sample trade entry...")
    trade_id = logger.log_entry(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=70730.00,
        sl=69500.00,
        tp=72950.00,
        position_size=0.5,
        model_confidence=0.82,
        user_notes="Strong confluence: RSI overbought, ATR expanding",
        regime='TRENDING',
    )
    
    print(f"✓ Trade {trade_id} logged")
    
    # Retrieve trade
    print("\n🔍 Retrieving trade...")
    trade = logger.get_trade(trade_id)
    print(f"Trade: {trade}")
    
    # Log exit
    print("\n📤 Logging trade exit...")
    result = logger.log_exit(
        trade_id=trade_id,
        exit_price=72800.00,
        exit_notes="TP hit, closed at +$2070 profit",
    )
    
    print(f"✓ Trade closed: {result['pnl_pct']:+.1%}")
    
    # Get stats
    print("\n📊 Trade statistics...")
    stats = logger.get_stats()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
