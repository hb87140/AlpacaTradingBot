"""
Entry rule definitions — Donchian Bounce Strategy.

Architecture for easy rule changes:
  • All numeric thresholds live in src/config.py (env-var overridable).
  • Each rule is a standalone function: (ctx: dict) -> (bool, str).
    The string is the human-readable fail reason (empty string on pass).
  • PERMANENT_DAY_RULES  — slow-changing filters cached for the trading day.
  • CYCLE_RULES          — checked every scan cycle.

To add a rule    : write a check function, append (name, fn) to the list.
To remove a rule : delete the (name, fn) entry from the list.
To change a threshold: edit the constant in src/config.py — no engine edits needed.

Context keys consumed by rules
────────────────────────────────────────────────────────────────────────────────
  live_price        float   latest trade price
  donchian_lower    float   20-day low of lows (the bounce floor)
  donchian_upper    float   20-day high of highs (resistance target)
  rsi               float   current RSI(14) value
  rsi_prev          float   previous day's RSI(14)
  rsi_history       list    last RSI_OVERSOLD_LOOKBACK+2 daily RSI values (oldest first)
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

from typing import Callable, List, Tuple

from src.config import (
    SCAN_MIN_PRICE,
    SCAN_MIN_VOLUME,
    SCAN_MIN_DOLLAR_VOL,
    SPREAD_MAX_PCT,
    DONCHIAN_FLOOR_TOL_PCT,
    RSI_MIN_DELTA,
    RSI_OVERSOLD_THRESHOLD,
    RSI_OVERSOLD_LOOKBACK,
    RVOL_MIN,
    DAY_STRENGTH_OPEN_PCT,
    VOL_MULT_FRIDAY,
    SCORE_DONCHIAN_MAX,
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


def check_donchian_floor(ctx: dict) -> Tuple[bool, str]:
    """Price must be within DONCHIAN_FLOOR_TOL_PCT of the 20-day Donchian lower band.

    This is the core setup: the stock is touching its floor, setting up for a bounce.
    Closer to the band = higher Donchian proximity score.
    """
    price = ctx.get('live_price', 0.0)
    lower = ctx.get('donchian_lower', 0.0)
    if lower <= 0:
        return False, 'Donchian lower band unavailable'
    proximity = (price - lower) / lower
    ok = proximity <= DONCHIAN_FLOOR_TOL_PCT
    return ok, (
        f'Price {proximity*100:.2f}% above Donchian floor '
        f'(max {DONCHIAN_FLOOR_TOL_PCT*100:.1f}%)'
    )


def check_rsi_momentum(ctx: dict) -> Tuple[bool, str]:
    """RSI must show a meaningful momentum turn from oversold territory.

    Two conditions must both pass:
    1. RSI was below RSI_OVERSOLD_THRESHOLD within the last RSI_OVERSOLD_LOOKBACK days.
    2. Current daily RSI is rising by at least RSI_MIN_DELTA points vs prior day.
    """
    rsi      = ctx.get('rsi', 0.0)
    rsi_prev = ctx.get('rsi_prev', rsi)
    delta    = rsi - rsi_prev

    if delta < RSI_MIN_DELTA:
        return False, f'RSI delta {delta:.1f} < {RSI_MIN_DELTA:.1f} required'

    rsi_hist = ctx.get('rsi_history', [])
    # rsi_history contains recent values oldest-first; exclude current bar (last element)
    prior_vals = rsi_hist[-(RSI_OVERSOLD_LOOKBACK + 1):-1] if rsi_hist else []
    if not prior_vals:
        # Fall back to rsi_prev when history is unavailable
        prior_vals = [rsi_prev]
    was_oversold = any(r < RSI_OVERSOLD_THRESHOLD for r in prior_vals)
    if not was_oversold:
        return False, (
            f'RSI not below {RSI_OVERSOLD_THRESHOLD} '
            f'in last {RSI_OVERSOLD_LOOKBACK} days'
        )
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
    l     = ctx.get('intraday_low', price)

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
    ('price_floor',    check_price_floor),
    ('spread',         check_spread),
    ('volume',         check_volume),
    ('donchian_floor', check_donchian_floor),
    ('rsi_momentum',   check_rsi_momentum),
    ('rvol',           check_rvol),
    ('day_strength',   check_day_strength),
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
    """Score a candidate 0-100 using the 4-component Donchian bounce formula.

    Component           Max pts  Description
    ──────────────────  ───────  ─────────────────────────────────────────────
    Donchian Proximity    30     Higher when price is closer to the lower band
    Time-Segmented RVOL   25     Higher with stronger institutional volume
    RSI Delta Accel       25     Higher with faster RSI recovery from oversold
    Spread + Dollar Vol   20     Higher with tighter spread and larger volume

    All weights are driven by config constants (SCORE_*_MAX).
    """
    price = ctx.get('live_price', 0.0)
    lower = ctx.get('donchian_lower', 0.0)
    rvol       = ctx.get('rvol', 0.0)
    rvol_min   = ctx.get('_effective_rvol_min', RVOL_MIN)
    rsi        = ctx.get('rsi', 0.0)
    rsi_prev   = ctx.get('rsi_prev', rsi)
    spread_pct = ctx.get('spread_pct', 0.0)
    dol_vol    = ctx.get('dollar_vol_20d', 0.0)

    # 1. Donchian floor proximity (SCORE_DONCHIAN_MAX pts)
    # Linear: max pts when price == lower band; 0 pts at tolerance ceiling.
    if lower > 0:
        proximity = (price - lower) / lower
        donchian_score = max(0.0, (1.0 - proximity / DONCHIAN_FLOOR_TOL_PCT) * SCORE_DONCHIAN_MAX)
    else:
        donchian_score = 0.0

    # 2. Time-segmented RVOL (SCORE_RVOL_MAX pts)
    # Linear from rvol_min to 5×; capped at max.
    rvol_excess = max(0.0, rvol - rvol_min)
    rvol_score  = min(SCORE_RVOL_MAX, rvol_excess / max(5.0 - rvol_min, 0.01) * SCORE_RVOL_MAX)

    # 3. RSI delta acceleration (SCORE_RSI_DELTA_MAX pts)
    # Linear: 0 pts at delta=0; max pts at delta≥10.
    rsi_delta = max(0.0, rsi - rsi_prev)
    rsi_score = min(SCORE_RSI_DELTA_MAX, rsi_delta / 10.0 * SCORE_RSI_DELTA_MAX)

    # 4. Spread & dollar-volume liquidity (SCORE_LIQUIDITY_MAX pts)
    # Split equally: half from spread quality, half from volume quality.
    half = SCORE_LIQUIDITY_MAX / 2.0
    spread_pts = max(0.0, (1.0 - spread_pct / SPREAD_MAX_PCT) * half) if SPREAD_MAX_PCT > 0 else half
    vol_pts    = min(half, (dol_vol / SCAN_MIN_DOLLAR_VOL) * half) if SCAN_MIN_DOLLAR_VOL > 0 else 0.0
    liq_score  = spread_pts + vol_pts

    return round(donchian_score + rvol_score + rsi_score + liq_score, 2)
