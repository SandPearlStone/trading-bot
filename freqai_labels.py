#!/home/sandro/trading-venv/bin/python3
"""Forward return labeling for ML training."""

import pandas as pd
import numpy as np


class TradingLabels:
    """Forward return labels with meta-labeling filter."""

    def compute(self, df: pd.DataFrame, n_forward: int = 10, fee: float = 0.001) -> pd.Series:
        """
        Compute forward return N bars ahead, net of fees.
        
        Args:
            df: DataFrame with 'close' column
            n_forward: bars to look ahead (default 10 = 10 hours on 1h)
            fee: round-trip fee to subtract (default 0.1%)
        
        Returns:
            pd.Series named 'target' with forward returns.
            Outliers removed via IQR filter (set to NaN).
            Last n_forward rows are NaN (no future data).
        """
        # Compute forward returns
        target = df['close'].shift(-n_forward) / df['close'] - 1 - fee
        target.name = 'target'
        
        # Apply IQR filter (1st/99th percentile for fat tails)
        q1 = target.quantile(0.01)
        q3 = target.quantile(0.99)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        target[(target < lower) | (target > upper)] = np.nan
        
        # Set last n_forward rows to NaN (no future data available)
        target.iloc[-n_forward:] = np.nan
        
        # Print statistics
        valid = target.dropna()
        pct_pos = (valid > 0).mean() * 100
        pct_neg = (valid < 0).mean() * 100
        print(f"Labels: {len(target)} total, {len(valid)} valid, {pct_pos:.1f}% +, {pct_neg:.1f}% -, μ={valid.mean():.6f}, σ={valid.std():.6f}")
        
        return target

    def meta_filter(self, df: pd.DataFrame, min_score: int = 40) -> pd.Series:
        """
        Return boolean mask: True where confluence_score >= min_score.
        
        Args:
            df: DataFrame, may or may not have 'confluence_score' column
            min_score: minimum score to keep (default 40)
        
        Returns:
            pd.Series of bool, same length as df.
            If 'confluence_score' not in df.columns: return all True.
        """
        if 'confluence_score' not in df.columns:
            return pd.Series(True, index=df.index)
        return df['confluence_score'] >= min_score
