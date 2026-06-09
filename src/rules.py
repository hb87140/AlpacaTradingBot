"""
Entry rule definitions — Alligator Swing Trading Strategy (Bill Williams).

Architecture for easy rule changes:
  • All numeric thresholds live in src/config.py (env-var overridable).
  • Each rule is a standalone function: (ctx: dict) -> (bool, str).
    The string is the human-readable fail reason (empty string on pass).
  • PERMANENT_DAY_RULES  — slow-changing filters cached for the trading day.
  • CYCLE_RULES          — checked every scan cycle.

Entry signal: the two faster SMMAs (periods 5 and 8) cross above the slow SMMA
(period 13) from below, with the crossover occurring within ALLIGATOR_CROSS_LOOKBACK
bars. All three SMAs are offset-adjusted per Bill Williams' original specification
(offsets 3, 5, 8 — all Fibonacci numbers).

Exit signals (in check_velocity_exits):
  Early  — fast SMMA crosses below medium SMMA.
  Full   — both fast and medium cross below slow SMMA (confirmed reversal).

Context keys consumed by rules
────────────────────────────────────────────────────────────────────────────────
  live_price        float   latest trade price
  smma_fast         float   SMMA(5) at offset 3 (green line, current aligned value)
  smma_med          float   SMMA(8) at offset 5 (red line)
  smma_slow         float   SMMA(13) at offset 8 (blue line / trend anchor)
  alligator_crossed bool    True when fast+med crossed above slow within lookback
  rsi               float   current RSI(14) value
  rsi_prev          float   previous day's RSI(14)
  rvol              float   intraday relative volume vs 20-day avg at current tod_frac
  spread_pct        float   (ask-bid)/mid — 0.0 when unavailable
  volume            int     latest daily bar volume
  avg_20d_vol       float   20-day average daily share volume
  dollar_vol_20d    float   20-day average daily dollar volume
  intraday_open     float   today's opening price (from snapshot daily_bar)
  intraday_high     float   today's session high so far
  intraday_low      float   today's session low so far
  _is_friday        bool    injected by engine; enables 2× volume threshold
  _effective_rvol_min float injected by engine; overrides RVOL_MIN in bearish regime
"""
from __future__ import annotations

import math
from typing import Callable, List, Tuple

from src.config import (
    SCAN_MIN_PRICE,
    SCAN_MIN_VOLUME,
    SCAN_MIN_DOLLAR_VOL,
    SPREAD_MAX_PCT,
    RSI_MIN_DELTA,
    RVOL_MIN,
    DAY_STRENGTH_OPEN_PCT,
    VOL_MULT_FRIDAY,
    ALLIGATOR_CROSS_LOOKBACK,
    SCORE_ALLIGATOR_MAX,
    SCORE_RVOL_MAX,
    SCORE_RSI_DELTA_MAX,
    SCORE_LIQUIDITY_MAX,
)

# Type alias
Rule = Tuple[str, Callable[[dict], Tuple[bool, str]]]


# ── Individual rule functions ─────────────────────────────────────────────────

def check_price_floor(ctx: dict) -> Tuple[bool, str]:
    """Price must be above the minimum universe threshold."""
    price = ctx.get('live_price', 0.0)
    ok = price >= SCAN_MIN_PRICE
    return ok, f'Price ${price:.2f} < min ${SCAN_MIN_PRICE:.2f}'


def check_spread(ctx: dict) -> Tuple[bool, str]:
    """Bid-ask spread must be within the maximum allowed threshold."""
    spread = ctx.get('spread_pct', 0.0)
    ok = spread <= SPREAD_MAX_PCT
    return ok, f'Spread {spread*100:.2f}% > max {SPREAD_MAX_PCT*100:.2f}%'


def check_volume(ctx: dict) -> Tuple[bool, str]:
    """20-day avg daily share volume must clear the universe floor (2× on Fridays)."""
    avg_vol = ctx.get('avg_20d_vol', 0.0)
    is_friday = ctx.get('_is_friday', False)
    threshold = SCAN_MIN_VOLUME * (VOL_MULT_FRIDAY if is_friday else 1.0)
    ok = avg_vol >= threshold
    return ok, f'AvgVol {avg_vol:,.0f} < {threshold:,.0f}'


def check_dollar_vol(ctx: dict) -> Tuple[bool, str]:
    """20-day avg dollar volume must clear the liquidity floor (2× on Fridays)."""
    dv = ctx.get('dollar_vol_20d', 0.0)
    is_friday = ctx.get('_is_friday', False)
    threshold = SCAN_MIN_DOLLAR_VOL * (VOL_MULT_FRIDAY if is_friday else 1.0)
    ok = dv >= threshold
    return ok, f'DolVol20d ${dv/1e6:.1f}M < ${threshold/1e6:.1f}M'


def check_alligator_bullish(ctx: dict) -> Tuple[bool, str]:
    """Alligator indicator — bullish setup: all three offset-adjusted SMMAs aligned upward
    with a recent crossover.

    Conditions (all must pass):
    1. All three SMMA values are finite and positive.
    2. fast SMMA > medium SMMA > slow SMMA — 'mouth open upward', trend confirmed.
    3. The crossover is fresh: within ALLIGATOR_CROSS_LOOKBACK bars, fast+medium
       transitioned from below the slow line to above it.  Prevents entering into
       mature trends where the best risk/reward has already passed.
    """
    smma_fast = ctx.get('smma_fast', float('nan'))
    smma_med  = ctx.get('smma_med',  float('nan'))
    smma_slow = ctx.get('smma_slow', float('nan'))

    if any(math.isnan(v) or v <= 0 for v in (smma_fast, smma_med, smma_slow)):
        return False, 'Alligator SMMA values unavailable'

    if smma_fast <= smma_med:
        return False, (
            f'Fast SMMA ({smma_fast:.2f}) ≤ med SMMA ({smma_med:.2f}) — not aligned bullish'
        )
    if smma_med <= smma_slow:
        return False, (
            f'Med SMMA ({smma_med:.2f}) ≤ slow SMMA ({smma_slow:.2f}) — trend not confirmed'
        )

    if not ctx.get('alligator_crossed', False):
        return False, (
            f'No bullish crossover within last {ALLIGATOR_CROSS_LOOKBACK} bars '
            '— enter only at the signal, not mid-trend'
        )

    return True, ''


def check_rsi_trend(ctx: dict) -> Tuple[bool, str]:
    """RSI must show upward momentum — rising by at least RSI_MIN_DELTA points.

    For Alligator swing entries we want accelerating momentum, not exhaustion.
    RSI > 50 confirms the stock is in bullish territory; rising RSI confirms
    the trend is still gaining strength rather than topping.
    """
    rsi      = ctx.get('rsi', 0.0)
    rsi_prev = ctx.get('rsi_prev', rsi)
    delta    = rsi - rsi_prev

    if delta < RSI_MIN_DELTA:
        return False, f'RSI delta {delta:.1f} < {RSI_MIN_DELTA:.1f} required (momentum slowing)'

    if rsi < 50:
        return False, f'RSI {rsi:.1f} < 50 — stock not yet in bullish territory'

    return True, ''


def check_rvol(ctx: dict) -> Tuple[bool, str]:
    """Intraday RVOL must confirm institutional participation.

    Uses _effective_rvol_min from ctx so the engine can tighten the threshold
    during a bearish SPY regime without modifying this function.
    """
    rvol     = ctx.get('rvol', 0.0)
    rvol_min = ctx.get('_effective_rvol_min', RVOL_MIN)
    ok = rvol >= rvol_min
    return ok, f'RVOL {rvol:.2f}x < {rvol_min:.2f}x required'


def check_day_strength(ctx: dict) -> Tuple[bool, str]:
    """Price must be showing intraday strength, not fading.

    Two conditions:
    1. Price ≥ DAY_STRENGTH_OPEN_PCT above today's open.
    2. Price is in the upper half of today's intraday range.
    """
    price = ctx.get('live_price', 0.0)
    o     = ctx.get('intraday_open', 0.0)
    h     = ctx.get('intraday_high', price)
    l     = ctx.get('intraday_low',  price)

    if o > 0:
        above_open = (price - o) / o
        if above_open < DAY_STRENGTH_OPEN_PCT:
            return False, (
                f'Price {above_open*100:.2f}% above open '
                f'(min {DAY_STRENGTH_OPEN_PCT*100:.1f}%)'
            )

    intraday_range = h - l
    if intraday_range > 0:
        range_pos = (price - l) / intraday_range
        if range_pos < 0.5:
            return False, f'Price in lower {range_pos*100:.0f}% of intraday range'

    return True, ''


# ── Rule lists — edit these to add/remove/reorder rules ──────────────────────

# Permanent day-level rules: computed once per day per symbol; result cached in
# _daily_scan_skip.  These should be filters whose inputs change at most daily.
PERMANENT_DAY_RULES: List[Rule] = [
    ('dollar_vol', check_dollar_vol),
]

# Cycle-level rules: evaluated on every scan cycle.
# Order: cheapest / most-selective checks first for fast rejection.
CYCLE_RULES: List[Rule] = [
    ('price_floor',       check_price_floor),
    ('spread',            check_spread),
    ('volume',            check_volume),
    ('alligator_bullish', check_alligator_bullish),
    ('rsi_trend',         check_rsi_trend),
    ('rvol',              check_rvol),
    ('day_strength',      check_day_strength),
]


# ── Rule runner ───────────────────────────────────────────────────────────────

def check_rules(ctx: dict, rules: List[Rule]) -> Tuple[bool, List[Tuple[str, str]]]:
    """Run a list of (name, fn) rules against ctx.

    Returns (all_passed, failures) where failures is a list of (name, reason) pairs
    for every rule that did not pass.
    """
    failures: List[Tuple[str, str]] = []
    for name, fn in rules:
        passed, reason = fn(ctx)
        if not passed:
            failures.append((name, reason))
    return not failures, failures


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_candidate(ctx: dict) -> float:
    """Score a candidate 0-100 using the 4-component Alligator swing formula.

    Component              Max pts  Description
    ─────────────────────  ───────  ────────────────────────────────────────────
    Alligator Alignment      30     Wider SMMA spread = stronger confirmed trend
    Time-Segmented RVOL      25     Higher with stronger institutional volume
    RSI Momentum             25     Higher with faster RSI acceleration
    Spread + Dollar Vol      20     Higher with tighter spread and larger volume

    All weights are driven by config constants (SCORE_*_MAX).
    """
    smma_fast  = ctx.get('smma_fast', 0.0)
    smma_slow  = ctx.get('smma_slow', 0.0)
    rvol       = ctx.get('rvol', 0.0)
    rvol_min   = ctx.get('_effective_rvol_min', RVOL_MIN)
    rsi        = ctx.get('rsi', 0.0)
    rsi_prev   = ctx.get('rsi_prev', rsi)
    spread_pct = ctx.get('spread_pct', 0.0)
    dol_vol    = ctx.get('dollar_vol_20d', 0.0)

    # 1. Alligator alignment (SCORE_ALLIGATOR_MAX pts)
    # Linear: 0 pts when fast == slow; full score at 5% separation.
    if smma_slow > 0 and smma_fast > smma_slow:
        alignment_pct   = (smma_fast - smma_slow) / smma_slow
        alligator_score = min(SCORE_ALLIGATOR_MAX, alignment_pct / 0.05 * SCORE_ALLIGATOR_MAX)
    else:
        alligator_score = 0.0

    # 2. Time-segmented RVOL (SCORE_RVOL_MAX pts)
    rvol_excess = max(0.0, rvol - rvol_min)
    rvol_score  = min(SCORE_RVOL_MAX, rvol_excess / max(5.0 - rvol_min, 0.01) * SCORE_RVOL_MAX)

    # 3. RSI momentum acceleration (SCORE_RSI_DELTA_MAX pts)
    rsi_delta = max(0.0, rsi - rsi_prev)
    rsi_score = min(SCORE_RSI_DELTA_MAX, rsi_delta / 10.0 * SCORE_RSI_DELTA_MAX)

    # 4. Spread & dollar-volume liquidity (SCORE_LIQUIDITY_MAX pts)
    half = SCORE_LIQUIDITY_MAX / 2.0
    spread_pts = max(0.0, (1.0 - spread_pct / SPREAD_MAX_PCT) * half) if SPREAD_MAX_PCT > 0 else half
    vol_pts    = min(half, (dol_vol / SCAN_MIN_DOLLAR_VOL) * half) if SCAN_MIN_DOLLAR_VOL > 0 else 0.0
    liq_score  = spread_pts + vol_pts

    return round(alligator_score + rvol_score + rsi_score + liq_score, 2)
