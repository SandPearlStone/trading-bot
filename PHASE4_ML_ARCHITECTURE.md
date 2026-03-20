# Phase 4 ML Integration Architecture
## Ready-to-implement design for Haiku

---

## 1. ML Scoring Module (`ml_scorer.py`)

### Startup & model loading

```python
MODEL_PATH = Path(__file__).parent / "phase4_model.pkl"

FEATURE_NAMES = [
    'vol_ratio', 'rsi_strength', 'rsi_div_regular', 'rsi_div_hidden',
    'macd_div', 'sentiment_fg', 'mtf_bias', 'confluence_score',
    'entry_to_sl', 'direction_encoded', 'duration_candles',
    'regime_CHOPPY', 'regime_NORMAL', 'regime_VOLATILE',
]

FEATURE_DEFAULTS = {
    'vol_ratio': 1.0,        # neutral volatility
    'rsi_strength': 0.5,     # neutral RSI
    'rsi_div_regular': 0,    # no divergence
    'rsi_div_hidden': 0,
    'macd_div': 0,
    'sentiment_fg': 0.5,     # neutral sentiment
    'mtf_bias': 0,           # neutral
    'confluence_score': 50,  # midpoint
    'entry_to_sl': 2.0,      # 2 ATR (reasonable default)
    'direction_encoded': 1,
    'duration_candles': 24,  # 1 day
    'regime_CHOPPY': 0,
    'regime_NORMAL': 1,      # assume normal
    'regime_VOLATILE': 0,
}

_model = None

def load_model() -> bool:
    """Load phase4_model.pkl at module init. Returns True if loaded."""
    global _model
    if _model is not None:
        return True
    try:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        return True
    except FileNotFoundError:
        print(f"[ml_scorer] Model not found: {MODEL_PATH}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ml_scorer] Model load error: {e}", file=sys.stderr)
        return False

# Auto-load at import time
load_model()
```

### Core scoring function

```python
def score_with_ml(features: dict) -> float:
    """
    Score a trade setup using the trained RandomForest.
    
    Args:
        features: partial or full dict of feature values
    
    Returns:
        confidence: float 0.0–1.0 (P(win))
        Returns 0.5 (neutral) if model not available or features invalid.
    
    Missing features are filled with FEATURE_DEFAULTS (graceful degradation).
    """
    if _model is None:
        return 0.5  # neutral fallback

    # Fill missing features with defaults
    filled = {k: features.get(k, FEATURE_DEFAULTS[k]) for k in FEATURE_NAMES}
    
    try:
        X = pd.DataFrame([filled])[FEATURE_NAMES]
        prob = _model.predict_proba(X)[0][1]  # P(win)
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as e:
        print(f"[ml_scorer] Prediction error: {e}", file=sys.stderr)
        return 0.5
```

---

## 2. Feature Extraction (`ml_scorer.py` continued)

```python
def extract_features_from_setup(setup: dict, df_1h: pd.DataFrame) -> dict:
    """
    Build ML feature dict from a live confluence result + candle data.
    
    Args:
        setup:  result dict from confluence.score_setup()
        df_1h:  1h OHLCV DataFrame (for vol_ratio, rsi_strength calculation)
    
    Returns:
        features dict (all keys in FEATURE_NAMES)
    """
    features = {}
    details  = setup.get("details", {})
    
    # ── regime ───────────────────────────────────────────────────────────────
    regime_raw = details.get("regime", {})
    regime_str = regime_raw.get("regime", "NORMAL") if isinstance(regime_raw, dict) else "NORMAL"
    features["regime_CHOPPY"]   = 1 if regime_str == "CHOPPY"   else 0
    features["regime_NORMAL"]   = 1 if regime_str == "NORMAL"   else 0
    features["regime_VOLATILE"] = 1 if regime_str == "VOLATILE" else 0
    
    # ── vol_ratio ─────────────────────────────────────────────────────────────
    # ATR_current / ATR_MA(50)
    if len(df_1h) >= 50:
        atr_series = calc_atr(df_1h, period=14)
        current_atr = atr_series.iloc[-1]
        hist_atr    = atr_series.iloc[-50:-1].mean()
        features["vol_ratio"] = float(current_atr / hist_atr) if hist_atr > 0 else 1.0
    else:
        features["vol_ratio"] = 1.0
    
    # ── rsi_strength ─────────────────────────────────────────────────────────
    # Normalized distance from RSI midline: abs(rsi - 50) / 50
    rsi_val = details.get("rsi", 50)
    features["rsi_strength"] = float(abs(rsi_val - 50) / 50)
    
    # ── divergences ──────────────────────────────────────────────────────────
    reasons = " ".join(setup.get("confluence_reasons", [])).lower()
    features["rsi_div_regular"] = 1 if "rsi" in reasons and "divergence" in reasons and "hidden" not in reasons else 0
    features["rsi_div_hidden"]  = 1 if "hidden divergence" in reasons else 0
    features["macd_div"]        = 1 if "macd divergence" in reasons or "dual divergence" in reasons else 0
    
    # ── sentiment_fg ─────────────────────────────────────────────────────────
    # Normalize 0-100 F&G index to 0-1; use 0.5 if unavailable
    # TODO: wire to live F&G source when available
    fg_raw = details.get("fear_greed", 50)
    features["sentiment_fg"] = float(fg_raw) / 100.0
    
    # ── mtf_bias ─────────────────────────────────────────────────────────────
    # +1 bullish, -1 bearish, 0 neutral
    direction = setup.get("direction", "NO_TRADE")
    bias_str  = details.get("bias", "neutral")
    if bias_str == "bullish":
        features["mtf_bias"] = 1
    elif bias_str == "bearish":
        features["mtf_bias"] = -1
    else:
        features["mtf_bias"] = 0
    
    # ── confluence_score (raw, before ML blend) ───────────────────────────────
    features["confluence_score"] = float(setup.get("score", 50))
    
    # ── sl_distance in ATR units ─────────────────────────────────────────────
    entry  = setup.get("optimal_entry")
    sl     = setup.get("stop_loss")
    atr_v  = details.get("atr", None)
    if entry and sl and atr_v and atr_v > 0:
        features["entry_to_sl"] = float(abs(entry - sl) / atr_v)
    else:
        features["entry_to_sl"] = 2.0
    
    # ── direction encoded ─────────────────────────────────────────────────────
    features["direction_encoded"] = 1 if direction == "LONG" else -1
    
    # ── duration (expected, in candles) ──────────────────────────────────────
    # Rough estimate: SL distance in ATR × 12 (heuristic for 1h TF)
    features["duration_candles"] = float(features["entry_to_sl"] * 12)
    
    return features
```

---

## 3. Confluence Integration

### Changes to `confluence.py` → `confluence_with_ml.py`

**Add after Kelly Sizing section (end of `score_setup`):**

```python
def score_setup_with_ml(
    symbol: str,
    interval: str = "1h",
    higher_tf: str = "4h",
    df_1h: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Drop-in replacement for score_setup() with ML scoring layered on top.
    
    Additional keys in returned dict:
        ml_confidence: float 0-1  (RandomForest P(win))
        ml_score:      float       (final blended score)
        ml_available:  bool        (False if model not loaded)
    
    Blending formula:
        final_score = raw_score × (0.7 + 0.3 × ml_prob)
    
    This means:
        ml_prob=0.0  →  final = raw × 0.70  (max 30% penalty)
        ml_prob=0.5  →  final = raw × 0.85  (slight discount)
        ml_prob=1.0  →  final = raw × 1.00  (no change, cap at raw)
    
    The model can only reduce a score, never inflate above raw.
    To allow boosting above raw, change multiplier cap to e.g. 1.1.
    """
    # --- Step 1: Run original scoring ---
    result = score_setup(symbol, interval, higher_tf)
    
    # --- Step 2: Get candle data for feature extraction ---
    if df_1h is None:
        # Fetch if not provided (reuse cached if possible)
        try:
            candles = get_ohlcv(symbol, interval, limit=200)
            df_1h = _candles_to_df(candles) if candles else None
        except Exception:
            df_1h = None
    
    # --- Step 3: Extract features ---
    ml_prob = 0.5  # safe default
    ml_available = False
    
    if df_1h is not None:
        try:
            from ml_scorer import score_with_ml, extract_features_from_setup, load_model
            
            if load_model():
                features   = extract_features_from_setup(result, df_1h)
                ml_prob    = score_with_ml(features)
                ml_available = True
        except ImportError:
            pass
        except Exception as e:
            result["missing"].append(f"ML scoring error: {e}")
    
    # --- Step 4: Blend score ---
    raw_score = result["score"]
    if ml_available and result["direction"] != "NO_TRADE":
        multiplier  = 0.7 + 0.3 * ml_prob       # range: [0.70, 1.00]
        final_score = round(raw_score * multiplier, 1)
        final_score = min(final_score, 100.0)
    else:
        final_score = raw_score
    
    result["score"]          = final_score
    result["raw_score"]      = raw_score
    result["ml_confidence"]  = round(ml_prob, 3)
    result["ml_available"]   = ml_available
    result["grade"]          = _grade(final_score)  # re-grade after blend
    
    return result
```

**Integration point in `confluence.py`:**
- Import: `from ml_scorer import score_with_ml, extract_features_from_setup`
- Only import if available (wrap in try/except)
- `score_setup()` stays untouched (backward compat)
- New `score_setup_with_ml()` wraps it

---

## 4. Scanner Output (`find_trades.py`)

### Changes to `_fmt_setup()`

```python
# Replace grade/score header line:

# OLD:
f"  {sym}/USDT | {dirn} | Grade: {grade} ({score}/100)"

# NEW:
def _fmt_grade_line(r: dict) -> str:
    sym   = r.get("symbol", "?")
    dirn  = r.get("direction", "?")
    grade = r.get("grade", "?")
    score = r.get("score", 0)
    ml_conf = r.get("ml_confidence")
    raw   = r.get("raw_score")
    
    grade_str = f"Grade {grade} ({score:.0f}/100)"
    
    if ml_conf is not None and r.get("ml_available"):
        ml_pct    = f"{ml_conf * 100:.0f}%"
        raw_str   = f" [raw: {raw:.0f}]" if raw and raw != score else ""
        grade_str += f"{raw_str} — ML: {ml_pct}"
    
    return f"  {sym}/USDT | {dirn} | {grade_str}"
    # Example: "  BTC/USDT | LONG | Grade A (81/100) [raw: 88] — ML: 67%"
```

### Sort order in `_fmt_telegram()` and main loop

```python
def _ml_sort_key(r: dict) -> float:
    """
    Sort key: grade_numeric × ml_confidence
    Higher = better setup. NO_TRADE always last.
    """
    if r.get("direction") == "NO_TRADE":
        return -1.0
    
    grade_map = {"A": 4, "B": 3, "C": 2, "F": 1}
    grade_num = grade_map.get(r.get("grade", "F"), 1)
    
    ml_conf   = r.get("ml_confidence", 0.5)
    score_norm = r.get("score", 0) / 100.0
    
    # Composite: grade tier × ml × normalized score
    return grade_num * ml_conf * score_norm


# In scan loop / results sorting:
results.sort(key=_ml_sort_key, reverse=True)
```

### `find_trades.py` argument change

```python
# Add to argparse:
parser.add_argument(
    "--with-ml",
    action="store_true",
    help="Use ML-enhanced scoring (requires phase4_model.pkl)"
)

# In scan call:
score_fn = score_setup_with_ml if args.with_ml else score_setup
```

---

## 5. Backtest Integration (`compare_phases.py`)

### New Phase 4 backend

```python
class Phase4Backend:
    """
    Phase 4: Phase 2 + ML confidence blending.
    
    Uses same base scoring as Phase2Backend, then applies:
        final_score = base_score × (0.7 + 0.3 × ml_prob)
    """
    
    name = "Phase4_ML"
    
    def __init__(self):
        self.phase2   = Phase2Backend()
        self._ml_ready = False
        try:
            from ml_scorer import load_model
            self._ml_ready = load_model()
        except ImportError:
            pass
    
    def score_setup(self, symbol: str, df_1h: pd.DataFrame) -> dict:
        """Phase 2 scoring + ML blend."""
        base = self.phase2.score_setup(symbol, df_1h)
        
        if not self._ml_ready or base["direction"] == "NO_TRADE":
            base["ml_confidence"] = 0.5
            base["ml_available"]  = False
            return base
        
        try:
            from ml_scorer import score_with_ml, extract_features_from_setup
            
            # Build minimal setup dict for feature extraction
            pseudo_setup = {
                "score":     base["score"],
                "direction": base["direction"],
                "stop_loss": base.get("sl"),
                "optimal_entry": df_1h["close"].iloc[-1],
                "details": {
                    "regime": {"regime": base.get("regime", "NORMAL")},
                    "rsi":    50,   # simplified for backtest
                    "atr":    None,
                    "bias":   "bullish" if base["direction"] == "LONG" else "bearish",
                },
                "confluence_reasons": [],
            }
            
            features   = extract_features_from_setup(pseudo_setup, df_1h)
            ml_prob    = score_with_ml(features)
            multiplier = 0.7 + 0.3 * ml_prob
            final      = min(base["score"] * multiplier, 100.0)
            
            base["score"]          = final
            base["raw_score"]      = base.get("score")
            base["ml_confidence"]  = ml_prob
            base["ml_available"]   = True
        except Exception as e:
            base["ml_confidence"] = 0.5
            base["ml_available"]  = False
        
        return base
    
    def should_enter(self, setup: dict) -> bool:
        """
        Phase 4 gating:
          - Same regime rules as Phase 2
          - Additional: skip if ML confidence < 0.35 (low conviction)
        """
        if not self.phase2.should_enter(setup):
            return False
        
        ml_conf = setup.get("ml_confidence", 0.5)
        if setup.get("ml_available") and ml_conf < 0.35:
            return False
        
        return True
```

### `--with-ml` flag in `compare_phases.py`

```python
# Add to argparse (or just always run 3-way):
parser.add_argument("--with-ml", action="store_true",
                    help="Include Phase 4 (ML) in comparison")

# In main():
backends = [Phase1Backend(), Phase2Backend()]
if args.with_ml:
    backends.append(Phase4Backend())

# Run all backends in parallel (ThreadPoolExecutor):
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_backend(backend, symbol, candles):
    engine = BacktestEngine(backend, symbol, candles, lookback=200)
    trades = engine.run()
    stats  = StatsCalculator.calculate(trades)
    return backend.name, stats

results_by_phase = {}
for symbol in symbols:
    candles = load_ohlcv_from_db(symbol)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(run_backend, b, symbol, candles): b.name
            for b in backends
        }
        for future in as_completed(futures):
            phase_name, stats = future.result()
            results_by_phase.setdefault(symbol, {})[phase_name] = stats
```

### 3-Way comparison output

```python
def print_3way_comparison(results_by_phase: dict, backends: list):
    """
    Print formatted 3-way comparison table.
    
    Example output:
    ╔════════════════╦══════════╦══════════╦══════════╗
    ║ Symbol         ║ Phase 1  ║ Phase 2  ║ Phase 4  ║
    ╠════════════════╬══════════╬══════════╬══════════╣
    ║ BTCUSDT        ║          ║          ║          ║
    ║   Win Rate     ║  52.3%   ║  61.7%   ║  68.2% ✓ ║
    ║   Profit Factor║  1.21    ║  1.54    ║  1.89  ✓ ║
    ║   Max Drawdown ║ -8.4%    ║ -5.1%    ║ -4.2%  ✓ ║
    ║   Total PnL    ║ +12.3%   ║ +18.9%   ║ +24.1% ✓ ║
    ╚════════════════╩══════════╩══════════╩══════════╝
    """
    METRICS = ["win_rate", "profit_factor", "max_drawdown", "sharpe_ratio", "total_pnl"]
    HIGHER_IS_BETTER = {"win_rate", "profit_factor", "sharpe_ratio", "total_pnl"}
    
    headers = ["Symbol/Metric"] + [b.name for b in backends]
    
    rows = []
    for symbol, phase_stats in results_by_phase.items():
        rows.append([f"── {symbol} ──"] + [""] * len(backends))
        for metric in METRICS:
            values = [phase_stats.get(b.name, {}).get(metric, 0) for b in backends]
            
            # Find best value
            if metric in HIGHER_IS_BETTER:
                best_idx = max(range(len(values)), key=lambda i: values[i])
            else:
                best_idx = min(range(len(values)), key=lambda i: values[i])
            
            cells = []
            for i, v in enumerate(values):
                mark = " ✓" if i == best_idx else "  "
                cells.append(f"{v:+.2f}{mark}" if isinstance(v, float) else str(v))
            
            rows.append([f"  {metric}"] + cells)
    
    # Print with tabulate or simple formatting
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    
    # Summary winner count
    print("\n── OVERALL WINNER ──")
    wins = {b.name: 0 for b in backends}
    # ... count wins across all metrics × symbols
    for name, count in wins.items():
        print(f"  {name}: {count} wins")
```

---

## 6. Integration Points Summary

| File | Change | Method |
|------|--------|--------|
| `ml_scorer.py` | **NEW** | `load_model()`, `score_with_ml(features)`, `extract_features_from_setup(setup, df)` |
| `confluence.py` | **ADD** | `score_setup_with_ml()` at bottom (wraps existing `score_setup`) |
| `find_trades.py` | **MODIFY** | `--with-ml` flag → use `score_setup_with_ml`; update `_fmt_setup` + sort |
| `compare_phases.py` | **MODIFY** | `Phase4Backend` class + `--with-ml` flag + `print_3way_comparison()` |

## 7. Key Design Decisions

1. **Backward compat**: `score_setup()` is untouched. `score_setup_with_ml()` is a thin wrapper — drop-in compatible.

2. **Graceful degradation**: Model absent → `ml_confidence=0.5` → multiplier=`0.85` (minor penalty, not a crash).

3. **Score can only go down** (0.7–1.0 multiplier). This is conservative. If you want ML to boost, change to `0.7 + 0.4 × ml_prob` (range 0.70–1.10).

4. **ML gate at 35%**: In Phase4Backend, if model says P(win) < 35%, skip the trade. This is configurable.

5. **Feature extraction is stateless**: `extract_features_from_setup` takes a setup dict + DataFrame. No side effects.

6. **Parallel backtesting**: ThreadPoolExecutor runs all 3 phases concurrently per symbol. Safe since backends are stateless.

---

## 8. File to create

```
trading/
  ml_scorer.py          ← NEW: model loader + score_with_ml + extract_features
  confluence_with_ml.py ← NEW: score_setup_with_ml() (or append to confluence.py)
  find_trades.py        ← MODIFY: --with-ml flag, display, sort
  compare_phases.py     ← MODIFY: Phase4Backend + 3-way output + --with-ml
```

Model path (hardcoded, adjustable):
```
trading/phase4_model.pkl
```
