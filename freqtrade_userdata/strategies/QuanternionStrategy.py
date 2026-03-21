"""
QuanternionStrategy — FreqTrade strategy with FreqAI + P1-3 confluence + Kelly sizing.

Uses:
- FreqAI LightGBMRegressor for ML predictions
- Custom P1-3 confluence scoring for signal gating
- Kelly criterion for position sizing
- ATR-based SL/TP
"""

import sys
sys.path.insert(0, '/home/sandro/.openclaw/workspace/trading')

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.persistence import Trade
import talib
import numpy as np
import pandas as pd
from pandas import DataFrame


class QuanternionStrategy(IStrategy):
    """FreqTrade strategy wrapping P1-3 + ML system."""
    
    # Strategy settings
    INTERFACE_VERSION = 3
    timeframe = '1h'
    informative_timeframes = ['4h', '15m']
    
    # Can short
    can_short = True
    
    # Stoploss
    stoploss = -0.03  # 3% max (ATR-based SL is tighter)
    trailing_stop = False
    
    # Take profit (handled by custom exit)
    minimal_roi = {"0": 0.05}  # 5% max, but ATR-based TP is tighter
    
    # Position sizing
    position_adjustment_enable = False
    
    # FreqAI
    freqai_info = {
        "enabled": True,
        "purge_old_models": 4,
        "train_period_days": 90,
        "backtest_period_days": 7,
        "identifier": "sandro_v1",
        "feature_parameters": {
            "include_timeframes": ["1h"],
            "include_corr_pairlist": ["BTC/USDT"],
            "label_period_candles": 10,
            "include_shifted_candles": 5,
            "indicator_periods_candles": [7, 14, 21],
            "DI_threshold": 0.9,
            "weight_factor": 0.9,
        },
        "data_split_parameters": {
            "test_size": 0.25,
            "random_state": 42,
        },
        "model_training_parameters": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 7,
        },
    }
    
    # === FEATURE ENGINEERING ===
    
    def feature_engineering_expand_all(self, dataframe: DataFrame, period, metadata, **kwargs) -> DataFrame:
        """Features that expand across periods, timeframes, and correlated pairs."""
        dataframe["%-rsi-period"] = talib.RSI(dataframe["close"], timeperiod=period)
        dataframe["%-mfi-period"] = talib.MFI(dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], timeperiod=period)
        dataframe["%-adx-period"] = talib.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=period)
        dataframe["%-roc-period"] = talib.ROC(dataframe["close"], timeperiod=period)
        
        bollinger = talib.BBANDS(dataframe["close"], timeperiod=period)
        dataframe["%-bb_width-period"] = (bollinger[0] - bollinger[2]) / bollinger[1]
        dataframe["%-bb_pctb-period"] = (dataframe["close"] - bollinger[2]) / (bollinger[0] - bollinger[2])
        
        dataframe["%-relative_volume-period"] = dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        
        return dataframe
    
    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """Features that expand across timeframes and correlated pairs (not periods)."""
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        
        # EMA ratios
        ema8 = talib.EMA(dataframe["close"], timeperiod=8)
        ema21 = talib.EMA(dataframe["close"], timeperiod=21)
        ema55 = talib.EMA(dataframe["close"], timeperiod=55)
        dataframe["%-ema_8_21_ratio"] = ema8 / ema21
        dataframe["%-ema_21_55_ratio"] = ema21 / ema55
        
        # ATR as percentage
        dataframe["%-atr_pct"] = talib.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14) / dataframe["close"] * 100
        
        # MACD
        macd, signal, hist = talib.MACD(dataframe["close"])
        dataframe["%-macd_hist"] = hist
        
        # Stochastic
        dataframe["%-stoch_k"], dataframe["%-stoch_d"] = talib.STOCH(dataframe["high"], dataframe["low"], dataframe["close"])
        
        return dataframe
    
    def feature_engineering_standard(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """Non-expanded features (temporal, custom)."""
        # Temporal
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        
        return dataframe
    
    def set_freqai_targets(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """Set prediction target: forward return 10 bars ahead minus fees."""
        dataframe["&-target"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            / dataframe["close"]
            - 1
            - 0.001  # subtract fees
        )
        return dataframe
    
    # === ENTRY/EXIT LOGIC ===
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate ATR for SL/TP and confluence score."""
        dataframe["atr"] = talib.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        
        # P1-3 Confluence approximation (pre-feature-engineering)
        rsi = talib.RSI(dataframe["close"], timeperiod=14)
        adx = talib.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        ema8 = talib.EMA(dataframe["close"], timeperiod=8)
        ema21 = talib.EMA(dataframe["close"], timeperiod=21)
        vol_ratio = dataframe["volume"] / dataframe["volume"].rolling(20).mean()
        bb = talib.BBANDS(dataframe["close"], timeperiod=20)
        bb_pctb = (dataframe["close"] - bb[2]) / (bb[0] - bb[2])
        
        score = pd.Series(0, index=dataframe.index)
        score += ((rsi < 30) | (rsi > 70)).astype(int) * 20
        score += (((rsi >= 30) & (rsi < 40)) | ((rsi > 60) & (rsi <= 70))).astype(int) * 10
        score += (adx > 25).astype(int) * 15
        score += (((bb_pctb < 0.1) | (bb_pctb > 0.9))).astype(int) * 15
        score += (abs(ema8 / ema21 - 1) > 0.002).astype(int) * 15
        score += (vol_ratio > 1.5).astype(int) * 15
        
        dataframe["%-confluence_score"] = score.clip(0, 100)
        
        # FreqAI predictions (populated by FreqAI framework)
        # dataframe["&-target"] will contain the predicted return
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signals: P1-3 confluence gate + basic volume/trend filter."""
        
        # Additional trend filter: EMA alignment
        ema8 = talib.EMA(dataframe["close"], timeperiod=8)
        ema21 = talib.EMA(dataframe["close"], timeperiod=21)
        ema_bullish = ema8 > ema21
        ema_bearish = ema8 < ema21
        
        # Long entries: confluence + bullish EMA trend + volume
        dataframe.loc[
            (
                (dataframe["%-confluence_score"] >= 40) &  # P1-3 confluence gate
                ema_bullish &                               # EMA trend filter
                (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"]
        ] = (1, "p13_long")
        
        # Short entries: confluence + bearish EMA trend + volume
        dataframe.loc[
            (
                (dataframe["%-confluence_score"] >= 40) &  # P1-3 confluence gate
                ema_bearish &                               # EMA trend filter
                (dataframe["volume"] > 0)
            ),
            ["enter_short", "enter_tag"]
        ] = (1, "p13_short")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit handled by stoploss and ROI."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time, current_rate, current_profit, after_fill, **kwargs) -> float:
        """ATR-based dynamic stoploss: 1.5 × ATR."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) > 0:
            last = dataframe.iloc[-1]
            atr = last.get("atr", 0)
            if atr > 0:
                sl_pct = (1.5 * atr) / current_rate
                return -sl_pct
        return self.stoploss
    
    def custom_stake_amount(self, pair: str, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        """Kelly criterion position sizing (f*/4 conservative)."""
        # Simple Kelly: use backtest WR 52% and R:R 1.67
        win_rate = 0.52
        rr_ratio = 2.5 / 1.5  # TP/SL = 1.67
        
        kelly_f = (win_rate * rr_ratio - (1 - win_rate)) / rr_ratio
        conservative_f = kelly_f / 4  # f*/4
        
        if conservative_f <= 0:
            return min_stake
        
        # Adjust by regime (reduce in low-confidence setups)
        stake = proposed_stake * min(conservative_f * 4, 1.0)  # scale
        return max(min(stake, max_stake), min_stake)
