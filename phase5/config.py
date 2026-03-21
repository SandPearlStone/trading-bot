"""
phase5/config.py — Centralized configuration for Phase 5 ML system.
All hyperparameters, paths, thresholds in one place.
"""

from __future__ import annotations
import os

# ── Base paths ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRADING = os.path.dirname(_HERE)

DATA_DIR       = os.path.join(_TRADING, "data", "binance_raw")
PROCESSED_DIR  = os.path.join(_TRADING, "data", "processed")
MODELS_DIR     = os.path.join(_TRADING, "models")
LOGS_DIR       = os.path.join(_TRADING, "logs")

for _d in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Symbols & Timeframes ──────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "ARBUSDT", "OPUSDT", "PEPEUSDT", "WIFUSDT",
]

TIMEFRAMES    = ["1h", "4h"]
PRIMARY_TF    = "4h"
SECONDARY_TF  = "1h"
LOOKBACK_DAYS = 730  # 2 years

# ── Technical Indicator Periods ───────────────────────────────────────────────
RSI_PERIOD     = 14
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
ADX_PERIOD     = 14
STOCH_K        = 14
STOCH_D        = 3
BB_PERIOD      = 20
BB_STD         = 2.0
KELTNER_PERIOD = 20
KELTNER_ATR    = 1.5
ROC_PERIOD     = 10
ATR_PERIOD     = 14
HIST_VOL_PERIOD= 20
OBV_SMOOTH     = 10
MFI_PERIOD     = 14
EMA_FAST       = 9
EMA_SLOW       = 21
EMA_TREND      = 50

# ── Triple Barrier Labeling ───────────────────────────────────────────────────
TP_ATR_MULTIPLE = 2.0    # profit target = entry + 2*ATR
SL_ATR_MULTIPLE = 1.5    # stop loss     = entry - 1.5*ATR
TIMEOUT_CANDLES = 10     # max hold time in candles

# ── Walk-Forward CV ───────────────────────────────────────────────────────────
WF_N_SPLITS    = 4
WF_TRAIN_FRAC  = 0.75    # 75% train, 25% test per split
WF_EMBARGO     = 2       # candles embargo gap to prevent leakage

# ── LightGBM Baseline ─────────────────────────────────────────────────────────
LGB_PARAMS = {
    "n_estimators":      200,
    "learning_rate":     0.05,
    "max_depth":         7,
    "num_leaves":        63,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

# ── XGBoost Baseline ──────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":   200,
    "learning_rate":  0.05,
    "max_depth":      6,
    "min_child_weight": 5,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":      0.1,
    "reg_lambda":     1.0,
    "random_state":   42,
    "n_jobs":         -1,
    "verbosity":      0,
    "eval_metric":    "logloss",
}

# ── Optuna Search Space ───────────────────────────────────────────────────────
OPTUNA_N_TRIALS = 200
OPTUNA_TIMEOUT  = None   # seconds; None = unlimited

LGB_SEARCH_SPACE = {
    "num_leaves":        (20, 200),
    "learning_rate":     (0.01, 0.2, "log"),
    "min_child_samples": (5, 100),
    "subsample":         (0.5, 1.0),
    "colsample_bytree":  (0.5, 1.0),
    "reg_alpha":         (0.0, 1.0),
    "reg_lambda":        (0.0, 1.0),
}

XGB_SEARCH_SPACE = {
    "max_depth":          (3, 10),
    "learning_rate":      (0.01, 0.2, "log"),
    "min_child_weight":   (1, 20),
    "subsample":          (0.5, 1.0),
    "colsample_bytree":   (0.5, 1.0),
    "reg_alpha":          (0.0, 1.0),
    "reg_lambda":         (0.5, 5.0),
}

# ── Ensemble ──────────────────────────────────────────────────────────────────
ENSEMBLE_LGB_WEIGHT = 0.5
ENSEMBLE_XGB_WEIGHT = 0.5

# ── Meta-Labeling / Confidence ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65   # only trade when confidence > this

# ── Model Paths ───────────────────────────────────────────────────────────────
LGB_MODEL_PATH      = os.path.join(MODELS_DIR, "phase5_lgb.pkl")
XGB_MODEL_PATH      = os.path.join(MODELS_DIR, "phase5_xgb.pkl")
META_MODEL_PATH     = os.path.join(MODELS_DIR, "phase5_meta.pkl")
LGB_OPT_MODEL_PATH  = os.path.join(MODELS_DIR, "phase5_lgb_opt.pkl")
XGB_OPT_MODEL_PATH  = os.path.join(MODELS_DIR, "phase5_xgb_opt.pkl")
FEATURE_NAMES_PATH  = os.path.join(MODELS_DIR, "phase5_feature_names.pkl")
SCALER_PATH         = os.path.join(MODELS_DIR, "phase5_scaler.pkl")

# ── Feature list (populated after engineering) ────────────────────────────────
FEATURE_NAMES: list[str] = []   # filled by feature_engineer at runtime

# ── Deployment ────────────────────────────────────────────────────────────────
RETRAINING_SCHEDULE = "weekly"   # "weekly" | "daily" | "monthly"
LIVE_EXCHANGE       = "binance"  # ccxt exchange id for live data
LIVE_DELAY_SECS     = 10        # poll interval in live mode
