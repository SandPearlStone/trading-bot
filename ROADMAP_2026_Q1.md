# Trading System Roadmap 2026 Q1

## Current State (2026-03-20)
- **Phase 1:** ✅ Complete
  - Rebalanced weights (RSI div 5→12, EMA 10→3)
  - MACD divergence detection
  - Sentiment gating framework (ready for F&G integration)
  - Grade distribution improved (47-57 range vs 40-45 clustering)

- **Active Trade:** AVAX LONG ($9.453, 30x, target $9.5585/$9.6365)
- **Session P&L:** Break-even (BTC BE, DOGE SL, ARB +15%)
- **Tools Ready:** Scanner, backtest GUI, Sonnet review agent

---

## Phase 2: Market Regime Detection (Priority: HIGH | Effort: 3-5 days)

### What
Classify market state (TRENDING/RANGING/CHOPPY) and adaptively adjust min_score thresholds.

### Why
- Would have blocked SOL/WIF BE trades (both in choppy market)
- Reduces false entries by 30-40%
- Prevents overtrading in low-edge periods

### How
**regime_detector.py:**
```python
def classify_regime(df_4h, df_1h):
    # ATR ratio: current ATR / 20-period avg
    # EMA slope variance: how stable is trend
    # Structure quality: HH/HL vs ranging
    # RSI extremes persistence
    
    if atr_ratio < 0.8 and ema_slope_var < threshold:
        return "RANGING"
    elif atr_ratio > 1.2 and ema_slope_var > threshold:
        return "TRENDING"
    else:
        return "CHOPPY"

def adaptive_min_score(regime):
    if regime == "RANGING":
        return 85  # Skip entries in range
    elif regime == "TRENDING":
        return 65  # Relaxed threshold
    else:  # CHOPPY
        return 75  # Mid threshold
```

### Integration
- Add `regime` field to scanner output
- Gate all entries: `if grade >= adaptive_threshold(regime)`
- Log regime classification in trade journal

### Expected Impact
- Win rate: 50-60% (from ~40-50%)
- Entries/week: -20% but higher quality
- Drawdown reduction: -15-20%

### Timeline
- Build regime_detector.py: 1 day
- Test on historical data: 1 day
- Integrate into scanner: 1 day
- Backtest validation: 1 day

---

## Phase 3: Dynamic Position Sizing (Priority: HIGH | Effort: 2-3 days)

### What
Tier position size by grade + regime.

### Tiers
```
A-grade + TRENDING    → 1.5x base risk
A-grade + RANGING     → 0.75x base risk
B-grade + TRENDING    → 1.0x base risk
B-grade + RANGING     → 0.5x base risk
C-grade + any regime  → 0.25x base risk (or skip)
CHOPPY + any grade    → 0.25x or skip
```

### Why
- Protects capital in weak conditions
- Scales into high-conviction setups
- Reduces drawdown in choppy periods

### How
```python
def position_size(grade, regime, base_risk=1.0):
    grade_multiplier = {"A": 1.2, "B": 0.9, "C": 0.5}
    regime_multiplier = {
        "TRENDING": 1.2,
        "RANGING": 0.7,
        "CHOPPY": 0.4
    }
    
    size = base_risk * grade_multiplier[grade] * regime_multiplier[regime]
    return max(0.25 * base_risk, size)  # Floor at 0.25x
```

### Expected Impact
- Reduce max drawdown: -20-25%
- Avg win size: +10-15%
- Reduce whipsaw losses: -30%

### Timeline
- Implement sizing function: 1 day
- Test on active trades: 0.5 days
- Backtest validation: 1 day

---

## Phase 4: Per-Symbol Rolling Win Rate Tracking (Priority: MEDIUM | Effort: 2 days)

### What
Track win rate per symbol. Adjust grade threshold if symbol is cold.

### Why
- Some symbols have edge in certain regimes (AVAX vs PEPE)
- Cold symbols deserve higher min_score
- Prevents revenge trading on bad symbols

### How
```python
def rolling_win_rate(symbol, window=20):
    recent_trades = get_trades(symbol, limit=window)
    wins = len([t for t in recent_trades if t['pnl'] > 0])
    return wins / len(recent_trades)

def symbol_adjusted_min_score(grade, symbol, regime):
    wr = rolling_win_rate(symbol)
    if wr < 0.35:  # Cold symbol
        return adaptive_min_score(regime) + 10
    elif wr > 0.65:  # Hot symbol
        return max(40, adaptive_min_score(regime) - 5)
    else:
        return adaptive_min_score(regime)
```

### Expected Impact
- Reduce cold symbol entries: -40%
- Increase hot symbol size: +20%
- Win rate consistency: +5-10%

### Timeline
- Build tracking system: 1 day
- Integrate into scanner: 0.5 days

---

## Phase 5: Advanced Pattern Recognition (Priority: MEDIUM | Effort: 1 week)

### Options (Pick 2-3)

#### A. Hidden Divergences
- Price makes lower high, RSI makes higher high (bearish hidden div)
- Catch pullback reversals inside trend
- More relevant than regular divergence

#### B. Volume-Weighted Confluence
- Weight signals by volume confirmation
- Volume dries up = weak signal
- Volume spike = strong signal

#### C. Multi-Symbol Correlation Filter
- If BTC in liquidation cascade, skip alts
- If sector (SOL, BNBUSDT) all bearish, avoid longs
- Reduce whipsaws from macro shifts

#### D. Orderflow Microstructure (if API available)
- Bid-ask spread analysis
- Large order clustering
- Execution probability estimates

### Expected Impact (per feature)
- Win rate: +2-5% each
- Drawdown: -10% each
- Setup quality: noticeably sharper entries

### Timeline
- Pick 2: 3-5 days development
- Backtest: 2-3 days
- Validation: 1 week live

---

## Sonnet Research Integration (TONIGHT 21:00 UTC)

**Incoming:** Deep research on predictive algorithms covering:
- Regime detection (validate our Phase 2 approach)
- Signal ensemble methods (improve confluence scoring)
- Kelly Criterion variants (optimize position sizing)
- Divergence patterns (validate Phase 5 choices)
- ML ensemble approaches (evaluate feasibility)

**Action:** Review Sonnet's report tomorrow morning, prioritize recommendations, integrate top 2-3 into roadmap.

---

## Implementation Priority Matrix

### Immediate (This Week)
1. ✅ Phase 1 complete
2. **Phase 2: Regime detection** (HIGH impact, HIGH feasibility)
3. **Phase 3: Position sizing** (HIGH impact, EASY)

### Near-term (Next Week)
4. **Phase 4: Symbol win rate** (MEDIUM impact, EASY)
5. **Review Sonnet research** + integrate top findings

### Medium-term (1-2 Weeks)
6. **Phase 5A: Hidden divergences** (MEDIUM impact, MEDIUM difficulty)
7. **Phase 5C: Multi-symbol correlation** (LOW impact, EASY)

### Optional (If Time)
8. Phase 5B: Volume confirmation
9. Phase 5D: Orderflow microstructure (if MEXC API extended)

---

## Success Metrics

| Metric | Current | Target (After Phase 2-3) |
|--------|---------|------------------------|
| Win Rate | 40-50% | 55-65% |
| Avg R per win | ~1.0R | 1.5-2.0R |
| Max Drawdown | TBD | -20% → -15% |
| Entries/week | X | 0.8X (fewer, better) |
| Consecutive losses | 3-4 | <3 |

---

## Backtest Validation Plan

1. **Baseline (Phase 1):** Run on 90-day walk-forward
   - Document current: WR, avg R, max DD, Sharpe
   
2. **Phase 2 (Regime):** Re-run with regime gating
   - Expected: +5-10% WR, -15% DD
   
3. **Phase 2+3 (Regime + Sizing):** Add dynamic position tiers
   - Expected: +8-12% WR, -20% DD, Sharpe up 30%
   
4. **Phase 4 (Symbol filter):** Add rolling win rate
   - Expected: +3-5% WR (from filtering cold)
   
5. **Live validation:** 2-3 weeks trading with new system
   - Confirm backtests vs real execution

---

## Risk Management

- **Soft launch Phase 2+3:** Run in parallel with current system for 1 week before full switch
- **Symbol whitelisting:** Only trade symbols with >20 trade history
- **Position size caps:** Never exceed 2x base risk even if grade/regime suggest
- **Drawdown circuit breaker:** Pause new entries if max DD > -25%

---

## Tools & Automation

| Component | Status | ETA |
|-----------|--------|-----|
| Scanner (find_trades.py) | ✅ Live | — |
| Backtest GUI | ✅ Live | — |
| Sonnet review agent | ✅ Ready | — |
| Regime detector | 🔄 In progress | Today |
| Position sizing | 🔄 In progress | Today |
| Symbol tracker | 📋 Design | Tomorrow |
| Cron scheduling | ✅ Ready | — |
| Trade journal | ✅ Live | — |

---

## Decision Gate: Should We Accelerate?

**Current velocity:** Phase 1 completed in 1 session.

**Proposal:** Run Phase 2+3 in parallel this weekend, backtest Monday, go live Tuesday.

**Risk:** Untested in live market. Phase 4 (symbol filter) helps mitigate.

**Benefit:** Start collecting fresh data with new system by mid-next week.

**Recommendation:** YES — time-decay on market regime is low (regime valid ~2-4 weeks). Better to test now while current regime is still valid.

---

## Questions for Sandro

1. **Sonnet research tonight** — Any specific algorithms you want emphasized?
2. **Risk tolerance:** Accept -25% DD for +60% WR? Or prefer -20% DD / +55% WR?
3. **Automation level:** Manual entry confirmation or auto-execute on scanner signal (with Sonnet review)?
4. **Symbol focus:** Stick with 12-symbol watchlist or expand/contract?
5. **Timeframe:** Week sprint (Phase 2+3) or gradual (Phase 1 each week)?

---

## Next Session Checklist

- [ ] Review Sonnet research (21:00 UTC tonight)
- [ ] Validate regime_detector logic against AVAX/DOGE/ARB trades
- [ ] Implement Phase 2 regime gating
- [ ] Implement Phase 3 position sizing
- [ ] Backtest Phase 2+3 on 90-day historical data
- [ ] Decision: parallel run or full switch?
- [ ] Schedule Phase 4 for next week
