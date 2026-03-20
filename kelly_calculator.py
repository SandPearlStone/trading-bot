#!/usr/bin/env python3
"""
kelly_calculator.py — Calculate optimal position sizing using Kelly Criterion

Loads closed trade history from trades.db and computes:
  - f* (optimal Kelly fraction)
  - f*/2 (moderate)
  - f*/4 (conservative)
  - Recommended position sizes for different account sizes

Kelly Formula: f* = (p × b − (1−p) × a) / b
  p = win_rate
  b = avg_win_pct
  a = avg_loss_pct

Output:
  - kelly_results.csv (metrics + fractions + position sizes)
  - Console: Detailed report with examples
  - Integration snippet: Code for confluence.py position sizing

Usage:
  python3 kelly_calculator.py
  # Output: kelly_results.csv
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import sys

DB_PATH = Path(__file__).parent / "data" / "trades.db"

# ─── TRADE HISTORY LOADER ───────────────────────────────────────────────────

class TradeHistoryLoader:
    """Load and filter closed trades from database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    def load_closed_trades(self) -> pd.DataFrame:
        """Load all closed trades."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT symbol, direction, entry_price, exit_price, pnl_pct, 
                   entry_time, exit_time, status
            FROM trades
            WHERE status = 'closed' OR pnl_pct IS NOT NULL
            ORDER BY created_at DESC
        '''
        
        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading trades: {e}")
            conn.close()
            return pd.DataFrame()
    
    def load_trades_by_symbol(self, symbol: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Load trades grouped by symbol."""
        df_all = self.load_closed_trades()
        
        if df_all.empty:
            return {}
        
        if symbol:
            return {symbol: df_all[df_all['symbol'] == symbol]}
        else:
            return {sym: df_all[df_all['symbol'] == sym] for sym in df_all['symbol'].unique()}


# ─── KELLY CALCULATOR ───────────────────────────────────────────────────────

class KellyCalculator:
    """Calculate Kelly Criterion and optimal position sizing."""
    
    @staticmethod
    def calculate_kelly(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> Dict:
        """
        Calculate Kelly fractions.
        
        Args:
            win_rate: 0.0 to 1.0 (e.g., 0.55 = 55%)
            avg_win_pct: Average win as % (e.g., 2.5)
            avg_loss_pct: Average loss as % (e.g., 1.0)
        
        Returns:
            {f_star, f_half, f_quarter, recommendations}
        """
        
        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            raise ValueError("Win rate must be between 0 and 1")
        if avg_win_pct <= 0 or avg_loss_pct <= 0:
            raise ValueError("Win/loss percentages must be positive")
        
        # Kelly formula: f* = (p × b − (1−p) × a) / b
        # where:
        #   p = win_rate
        #   b = avg_win_pct (as decimal: 2.5% = 0.025)
        #   a = avg_loss_pct
        
        p = win_rate
        b = avg_win_pct / 100  # Convert to decimal
        a = avg_loss_pct / 100
        
        numerator = (p * b) - ((1 - p) * a)
        denominator = b
        
        f_star = numerator / denominator if denominator != 0 else 0
        
        # Fractional Kelly
        f_half = f_star / 2
        f_quarter = f_star / 4
        
        # Safety checks
        if f_star < 0:
            print(f"⚠️  WARNING: Negative Kelly ({f_star:.2%}) — system is not profitable!")
            f_star = 0
            f_half = 0
            f_quarter = 0
        
        if f_star > 0.5:
            print(f"⚠️  WARNING: Kelly exceeds 50% ({f_star:.2%}) — extremely aggressive!")
        
        return {
            "f_star": f_star,
            "f_half": f_half,
            "f_quarter": f_quarter,
            "win_rate": win_rate,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct
        }
    
    @staticmethod
    def calculate_position_size(account_size: float, kelly_fraction: float, risk_pct: float = 1.0) -> Dict:
        """
        Calculate position size from account and Kelly fraction.
        
        Position size = Account × Kelly / Risk%
        
        Args:
            account_size: Account balance (e.g., $10,000)
            kelly_fraction: Kelly fraction (e.g., 0.083 for f*/4)
            risk_pct: Risk per trade (default 1%)
        
        Returns:
            {position_size, risk_amount, leverage}
        """
        
        # Risk amount per trade
        risk_amount = account_size * (risk_pct / 100)
        
        # Position size from Kelly
        position_size = account_size * kelly_fraction
        
        # Implied leverage (if position > account)
        leverage = position_size / account_size if account_size > 0 else 1.0
        
        return {
            "position_size": position_size,
            "risk_amount": risk_amount,
            "leverage": leverage,
            "pct_of_account": kelly_fraction * 100
        }


# ─── REPORT GENERATOR ───────────────────────────────────────────────────────

class ReportGenerator:
    """Generate Kelly analysis reports."""
    
    @staticmethod
    def generate_overview(trades_df: pd.DataFrame) -> Dict:
        """Generate overview stats from trades."""
        if trades_df.empty:
            return {}
        
        pnl_list = trades_df['pnl_pct'].dropna().tolist()
        
        if not pnl_list:
            return {}
        
        total_trades = len(pnl_list)
        wins = sum(1 for p in pnl_list if p > 0)
        losses = sum(1 for p in pnl_list if p <= 0)
        
        win_pnls = [p for p in pnl_list if p > 0]
        loss_pnls = [p for p in pnl_list if p <= 0]
        
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = abs(np.mean(loss_pnls)) if loss_pnls else 0
        
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "total_pnl": sum(pnl_list),
            "best_trade": max(pnl_list),
            "worst_trade": min(pnl_list)
        }
    
    @staticmethod
    def format_report(symbol: str, overview: Dict, kelly_data: Dict, account_size: float = 10000) -> str:
        """Format Kelly report as readable text."""
        
        if not overview or not kelly_data:
            return f"❌ {symbol}: No data available\n"
        
        lines = []
        lines.append(f"\n{'=' * 70}")
        lines.append(f"KELLY ANALYSIS: {symbol}")
        lines.append(f"{'=' * 70}")
        
        # Trade stats
        lines.append(f"\n📊 TRADE STATISTICS:")
        lines.append(f"   Total Trades:    {overview['total_trades']}")
        lines.append(f"   Wins:            {overview['wins']}")
        lines.append(f"   Losses:          {overview['losses']}")
        lines.append(f"   Win Rate:        {overview['win_rate']:.1%}")
        lines.append(f"   Avg Win:         {overview['avg_win_pct']:.2f}%")
        lines.append(f"   Avg Loss:        {overview['avg_loss_pct']:.2f}%")
        lines.append(f"   Total PnL:       {overview['total_pnl']:.2f}%")
        lines.append(f"   Best Trade:      {overview['best_trade']:.2f}%")
        lines.append(f"   Worst Trade:     {overview['worst_trade']:.2f}%")
        
        # Kelly fractions
        lines.append(f"\n🎯 KELLY FRACTIONS:")
        lines.append(f"   f* (Optimal):    {kelly_data['f_star']:.1%}")
        lines.append(f"   f*/2 (Moderate): {kelly_data['f_half']:.1%}")
        lines.append(f"   f*/4 (Safe):     {kelly_data['f_quarter']:.1%}")
        
        # Position sizing examples
        lines.append(f"\n💰 POSITION SIZING (${account_size:,.0f} account):")
        
        for fraction_name, fraction_val in [("f* (Optimal)", kelly_data['f_star']), 
                                             ("f*/2 (Moderate)", kelly_data['f_half']),
                                             ("f*/4 (Safe)", kelly_data['f_quarter'])]:
            ps = KellyCalculator.calculate_position_size(account_size, fraction_val)
            lines.append(f"\n   {fraction_name}:")
            lines.append(f"      Position Size: ${ps['position_size']:,.0f} ({ps['pct_of_account']:.2f}% of account)")
            lines.append(f"      Risk (1%):     ${ps['risk_amount']:,.0f}")
            lines.append(f"      Leverage:      {ps['leverage']:.1f}x")
        
        # Recommendation
        lines.append(f"\n✅ RECOMMENDATION:")
        if kelly_data['f_star'] <= 0:
            lines.append(f"   ⚠️  System is not profitable. Review trading strategy.")
        elif kelly_data['f_star'] > 0.5:
            lines.append(f"   Use f*/4 ({kelly_data['f_quarter']:.1%}) for safety. System is profitable but risky.")
        else:
            lines.append(f"   Recommended: f*/2 ({kelly_data['f_half']:.1%}) for balanced growth")
            lines.append(f"   Conservative: f*/4 ({kelly_data['f_quarter']:.1%}) if drawdown concerns")
            lines.append(f"   Aggressive: f* ({kelly_data['f_star']:.1%}) only if very confident")
        
        return "\n".join(lines)


# ─── INTEGRATION CODE GENERATOR ──────────────────────────────────────────────

def generate_integration_snippet(kelly_data: Dict) -> str:
    """Generate Python snippet for confluence.py position sizing."""
    
    snippet = f'''
# ─── KELLY-BASED POSITION SIZING ───────────────────────────────────────────
# Generated from kelly_calculator.py
# Add to confluence.py or position_sizer.py

KELLY_F_STAR = {kelly_data['f_star']:.4f}
KELLY_F_HALF = {kelly_data['f_half']:.4f}
KELLY_F_QUARTER = {kelly_data['f_quarter']:.4f}

def calculate_position_size(account_balance: float, setup_score: float, kelly_fraction=KELLY_F_QUARTER):
    \"\"\"
    Calculate position size using Kelly fraction.
    
    Args:
        account_balance: Account size in USD
        setup_score: Confluence score (0-100)
        kelly_fraction: Kelly fraction to use (default: f*/4)
    
    Returns:
        position_size (in USD)
    \"\"\"
    
    # Scale Kelly by confidence (setup score)
    confidence = setup_score / 100
    adjusted_kelly = kelly_fraction * confidence
    
    # Position size
    position_size = account_balance * adjusted_kelly
    
    return position_size

# Example usage:
# account = 10000  # USD 10k
# score = 75       # 75 percent confidence
# pos_size = calculate_position_size(account, score, KELLY_F_QUARTER)
# print(f"Position size: USD {{pos_size:,.0f}}")
'''
    
    return snippet


# ─── MAIN SCRIPT ─────────────────────────────────────────────────────────────

def main():
    """Run Kelly calculator."""
    print("=" * 80)
    print("KELLY CRITERION POSITION SIZER")
    print("=" * 80)
    print()
    
    # Load trades
    loader = TradeHistoryLoader(DB_PATH)
    trades_df = loader.load_closed_trades()
    
    if trades_df.empty:
        print("❌ No closed trades found in database.")
        print()
        print("To populate trades, run:")
        print("  1. python3 confluence.py  # Score setups")
        print("  2. Log trades manually in db.py")
        print()
        
        # Generate sample calculation for demo
        print("=" * 80)
        print("SAMPLE CALCULATION (55% win rate, +2% avg win, -1% avg loss)")
        print("=" * 80)
        
        sample_overview = {
            "total_trades": 20,
            "wins": 11,
            "losses": 9,
            "win_rate": 0.55,
            "avg_win_pct": 2.0,
            "avg_loss_pct": 1.0,
            "total_pnl": 13.0,
            "best_trade": 5.0,
            "worst_trade": -1.0
        }
        
        sample_kelly = KellyCalculator.calculate_kelly(0.55, 2.0, 1.0)
        
        print(ReportGenerator.format_report("SAMPLE", sample_overview, sample_kelly, account_size=10000))
        print(generate_integration_snippet(sample_kelly))
        
        return
    
    # Generate reports for all symbols
    print(f"📊 Analyzing {len(trades_df)} trades...\n")
    
    symbols = trades_df['symbol'].unique()
    all_results = []
    
    for symbol in symbols:
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        
        overview = ReportGenerator.generate_overview(symbol_trades)
        if not overview:
            continue
        
        # Calculate Kelly
        try:
            kelly_data = KellyCalculator.calculate_kelly(
                overview['win_rate'],
                overview['avg_win_pct'],
                overview['avg_loss_pct']
            )
        except Exception as e:
            print(f"❌ Error calculating Kelly for {symbol}: {e}")
            continue
        
        # Print report
        print(ReportGenerator.format_report(symbol, overview, kelly_data))
        
        # Store for CSV
        for kelly_type, kelly_frac in [("f*", kelly_data['f_star']), 
                                        ("f*/2", kelly_data['f_half']),
                                        ("f*/4", kelly_data['f_quarter'])]:
            ps = KellyCalculator.calculate_position_size(10000, kelly_frac)
            all_results.append({
                "symbol": symbol,
                "total_trades": overview['total_trades'],
                "win_rate": f"{overview['win_rate']:.1%}",
                "avg_win": f"{overview['avg_win_pct']:.2f}%",
                "avg_loss": f"{overview['avg_loss_pct']:.2f}%",
                "kelly_type": kelly_type,
                "kelly_fraction": f"{kelly_frac:.4f}",
                "kelly_pct": f"{kelly_frac*100:.2f}%",
                "position_size_10k": f"${ps['position_size']:,.0f}",
                "risk_amount_1pct": f"${ps['risk_amount']:,.0f}",
                "leverage": f"{ps['leverage']:.1f}x"
            })
    
    # Output CSV
    if all_results:
        df_results = pd.DataFrame(all_results)
        csv_path = Path(__file__).parent / "kelly_results.csv"
        df_results.to_csv(csv_path, index=False)
        
        print()
        print("=" * 80)
        print(f"📁 Results saved to: {csv_path}")
        print("=" * 80)
        print()
        print("CSV PREVIEW:")
        print(df_results.to_string(index=False))
        print()
        
        # Integration snippet
        if len(symbols) > 0:
            first_symbol_trades = trades_df[trades_df['symbol'] == symbols[0]]
            overview = ReportGenerator.generate_overview(first_symbol_trades)
            kelly_data = KellyCalculator.calculate_kelly(
                overview['win_rate'],
                overview['avg_win_pct'],
                overview['avg_loss_pct']
            )
            print()
            print("=" * 80)
            print("INTEGRATION SNIPPET (for confluence.py)")
            print("=" * 80)
            print(generate_integration_snippet(kelly_data))
    else:
        print("❌ No valid Kelly calculations. Check trade data.")


if __name__ == "__main__":
    main()
