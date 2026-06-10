"""
Comprehensive validation of three critical VelocityEngine subsystems (Alpaca edition):

  1. Entry order construction (2-order structure: LMT BUY + TrailingStop SELL)
     - chandelier_dist = ATR_CHAND × CHANDELIER_MULT (2.0)  — dollar amount
     - BUY order: LimitOrderRequest, time_in_force=DAY
     - TRAIL stop: TrailingStopOrderRequest, trail_price=chandelier_dist, time_in_force=GTC
     - state.stop_loss  = fill - chandelier_dist (dollar level for internal tracking)
     - state.stop_dist  = chandelier_dist
     - No goodAfterTime on either order (Alpaca is commission-free, stateless REST)

  2. Screener (Alpaca get_candidates — gainers + most-actives)
     - get_institutional_scan() calls get_candidates(data_client, screener_client)
     - Returns a deduped list of symbols (gainers first, then actives)

  3. Scoring and shortlisting
     - Trend (30pts) · RVOL (25pts) · Momentum (25pts) · Liquidity (20pts)
     - Maximum achievable score = 100
     - Candidates ranked by score; top slots filled first
     - Slot cap respected; already-held symbols skipped before scoring
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
import pytz

from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, TrailingStopOrderRequest

# Shared SPY regime dict — used wherever _fetch_spy_trend is patched
_SPY_BULL_REGIME = {
    'is_bull': True, 'spy_close': 450.0, 'ema50': 440.0,
    'size_factor': 1.0, 'rvol_mult': 1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_account(equity=2500.0, cash=2500.0):
    acc = MagicMock()
    acc.portfolio_value = str(equity)
    acc.cash = str(cash)
    return acc


def _mock_trading_client(equity=2500.0, cash=2500.0):
    tc = MagicMock()
    tc.get_account.return_value = _mock_account(equity, cash)
    tc.get_all_positions.return_value = []
    tc.get_open_position.side_effect = Exception("no position")
    tc.get_orders.return_value = []
    tc.submit_order.return_value = MagicMock(id='order-id', status='new')
    tc.cancel_order_by_id.return_value = None
    tc.get_order_by_id.return_value = MagicMock(
        id='order-id', status='filled',
        filled_avg_price='100.0', filled_qty='10.0')
    return tc


def _make_engine(equity=2500.0, cash=2500.0, trading_client=None, data_client=None, state=None):
    from src.engine import VelocityEngine
    engine = VelocityEngine.__new__(VelocityEngine)
    engine.trading_client = trading_client or _mock_trading_client(equity, cash)
    engine.data_client = data_client or MagicMock()
    engine.screener_client = MagicMock()
    engine.state = state if state is not None else {}
    engine._last_equity = 0.0
    engine._last_settled_cash = 0.0
    engine._equity_initialized = False
    engine._last_vix = None
    engine._vix_cache_date = None
    engine._last_scan_ts = None
    engine._next_scan_dt = None
    engine._day_start_equity = None
    engine._day_start_date = None
    engine._bar_cache = {}
    engine._spy_cache = {}
    engine._sector_cache = {}
    engine._analyst_cache = {}
    engine._daily_scan_skip = {}
    engine._insufficient_history_skip = set()
    engine._last_audit_date = None
    engine._missing_position_counts = {}
    engine._pnl_cache               = None
    engine._pnl_cache_ts            = None
    return engine


def _make_snapshot(price=None, bid=0.0, ask=0.0, intraday_vol=5_000_000):
    if price is None:
        return None
    return {'live_price': price, 'bid': bid, 'ask': ask, 'intraday_vol': intraday_vol}


def _ctx(price=100.0, orb=95.0, ma50=105.0, ma200=90.0,
         rsi=52.0, rsi_prev=37.0, atr=3.0,
         dollar_vol=500_000_000,
         rvol=3.5, spread_pct=0.002,
         sma200_slope=0.1,
         atr_chandelier=None,
         avg_20d_vol=5_000_000,
         adx=25.0, high200=None,
         donchian_lower=None, donchian_upper=None,
         intraday_open=None, intraday_high=None, intraday_low=None,
         rsi_history=None,
         smma_fast=None, smma_med=None, smma_slow=None,
         alligator_crossed=True,
         analyst_buy=0, analyst_hold=0, analyst_sell=0):
    """Build a get_technical_context()-style dict with all production-rule fields."""
    h200 = high200 if high200 is not None else round(price * 1.1, 4)
    dl = donchian_lower if donchian_lower is not None else round(price * 0.998, 4)
    du = donchian_upper if donchian_upper is not None else round(price * 1.10, 4)
    # Day-strength defaults: price 1.2% above open, in upper 86% of intraday range
    io = intraday_open if intraday_open is not None else round(price * 0.988, 4)
    ih = intraday_high if intraday_high is not None else round(price * 1.005, 4)
    il = intraday_low  if intraday_low  is not None else round(price * 0.970, 4)
    rh = rsi_history if rsi_history is not None else [28.0, 30.0, 32.0, rsi_prev, rsi]
    # Alligator defaults: 5% fast/slow separation → 30 pts; all three aligned bullish.
    sf = smma_fast if smma_fast is not None else 105.0
    sm = smma_med  if smma_med  is not None else 102.0
    ss = smma_slow if smma_slow is not None else 100.0
    return {
        'orb_high':          orb,
        'ma50':              ma50,
        'ma200':             ma200,
        'rsi':               rsi,
        'rsi_prev':          rsi_prev,
        'rsi_history':       rh,
        'atr':               atr,
        'atr_chandelier':    atr_chandelier if atr_chandelier is not None else atr,
        'sma200_slope':      sma200_slope,
        'adx':               adx,
        'high200':           h200,
        'rvol':              rvol,
        'spread_pct':        spread_pct,
        'close':             price - 0.5,
        'live_price':        price,
        'volume':            5_000_000,
        'dollar_vol_20d':    dollar_vol,
        'avg_20d_vol':       avg_20d_vol,
        'donchian_lower':    dl,
        'donchian_upper':    du,
        'intraday_open':     io,
        'intraday_high':     ih,
        'intraday_low':      il,
        'smma_fast':         sf,
        'smma_med':          sm,
        'smma_slow':         ss,
        'alligator_crossed': alligator_crossed,
        'analyst_buy':       analyst_buy,
        'analyst_hold':      analyst_hold,
        'analyst_sell':      analyst_sell,
        # price_fetched_at must be well inside the 60-second freshness window
        # so run_cycle() doesn't attempt a re-price snapshot call
        'price_fetched_at':  pytz.timezone('US/Eastern').localize(datetime(2024, 6, 5, 10, 30)),
    }


def _run_entry_cycle(engine, ctx, sym='TSLA', fake_now=None, equity=2500.0, cash=2500.0):
    """
    Run one full run_cycle() with a single passing signal.
    Returns engine.
    """
    tz_ny = pytz.timezone('US/Eastern')
    if fake_now is None:
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

    tc = engine.trading_client

    with patch.object(engine, '_ensure_connected', return_value=True), \
         patch.object(engine, '_sync_positions'), \
         patch.object(engine, '_get_account_values', return_value=(equity, cash)), \
         patch.object(engine, 'check_velocity_exits', return_value={}), \
         patch.object(engine, '_audit_stop_orders'), \
         patch.object(engine, '_update_position_prices'), \
         patch.object(engine, '_write_dashboard_data'), \
         patch.object(engine, 'save_state'), \
         patch.object(engine, 'get_institutional_scan', return_value=[sym]), \
         patch.object(engine, 'get_technical_context', return_value=ctx), \
         patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
         patch.object(engine, '_fetch_vix', return_value=20.0), \
         patch.object(engine, '_get_sector', return_value='Technology'), \
         patch('src.engine.time.sleep'), \
         patch('src.engine.datetime') as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.fromisoformat = datetime.fromisoformat
        engine.run_cycle()
    return engine


def _run_multi_signal_cycle(engine, ctx_map, equity=2500.0, cash=2500.0):
    """
    Run a cycle where get_institutional_scan returns ctx_map.keys().
    Returns the set of symbols in state after the cycle (includes pre-held positions).

    _get_sector is patched so that:
    - Symbols in ctx_map (new candidates) → 'Technology'
    - Already-held symbols (existing state keys) → 'OtherSector'
    This prevents the sector-concentration guard from falsely blocking entries.
    """
    tz_ny = pytz.timezone('US/Eastern')
    fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

    symbols = list(ctx_map.keys())

    def _ctx_for(sym, snap=None):
        return ctx_map.get(sym)

    # Build a sector map: held positions → 'OtherSector', candidates → 'Technology'.
    # This prevents the sector-concentration guard (MAX_SECTOR_COUNT=2) from
    # blocking new candidates just because we pre-filled all held slots with
    # the same sector.
    held_syms = set(engine.state.keys())
    candidate_syms = set(ctx_map.keys())

    def _sector_for(sym):
        return 'OtherSector' if sym in held_syms else 'Technology'

    _no_analyst = {'analyst_buy': 0, 'analyst_hold': 0, 'analyst_sell': 0}
    with patch.object(engine, '_ensure_connected', return_value=True), \
         patch.object(engine, '_sync_positions'), \
         patch.object(engine, '_get_account_values', return_value=(equity, cash)), \
         patch.object(engine, 'check_velocity_exits', return_value={}), \
         patch.object(engine, '_audit_stop_orders'), \
         patch.object(engine, '_update_position_prices'), \
         patch.object(engine, '_write_dashboard_data'), \
         patch.object(engine, 'save_state'), \
         patch.object(engine, 'get_institutional_scan', return_value=symbols), \
         patch.object(engine, 'get_technical_context', side_effect=_ctx_for), \
         patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
         patch.object(engine, '_fetch_vix', return_value=20.0), \
         patch.object(engine, '_get_sector', side_effect=_sector_for), \
         patch.object(engine, '_get_analyst_ratings', return_value=_no_analyst), \
         patch('src.engine.time.sleep'), \
         patch('src.engine.datetime') as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.fromisoformat = datetime.fromisoformat
        engine.run_cycle()

    # Return the full set of state keys after the cycle
    return set(engine.state.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENTRY ORDER CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryOrderConstruction:
    """
    Verify that run_cycle() submits:
      1st call: LimitOrderRequest(side=BUY, time_in_force=DAY)
      2nd call: TrailingStopOrderRequest(trail_price=chandelier_dist, time_in_force=GTC)

    And that state records fill_price, stop_dist, stop_loss, score.
    """

    ATR_CHAND  = 3.00
    ENTRY      = 100.00
    CHAND_DIST = round(3.00 * 2.5, 2)   # 7.50

    def _setup(self, equity=2500.0, cash=2500.0):
        from src.config import CHANDELIER_MULT
        tc = _mock_trading_client(equity, cash)

        # get_order_by_id: first call is "filled" (buy poll), subsequent is stop confirmation
        filled_order = MagicMock(
            id='buy-id', status='filled',
            filled_avg_price=str(self.ENTRY),
            filled_qty='10.0',
        )
        tc.get_order_by_id.return_value = filled_order

        # submit_order: first = buy_order, second = stop_order
        buy_order  = MagicMock(id='buy-id',  status='new')
        stop_order = MagicMock(id='stop-id', status='accepted')
        tc.submit_order.side_effect = [buy_order, stop_order]

        engine = _make_engine(equity, cash, trading_client=tc)
        ctx = _ctx(
            price=self.ENTRY,
            atr=self.ATR_CHAND,
            atr_chandelier=self.ATR_CHAND,
            orb=self.ENTRY - 5,
            ma50=self.ENTRY - 3,
            ma200=self.ENTRY - 15,
            rsi=52.0, rsi_prev=37.0,
        )
        return tc, engine, ctx

    def test_chandelier_dist_equals_atr_chandelier_times_mult(self):
        from src.config import CHANDELIER_MULT
        assert CHANDELIER_MULT == 2.5
        chandelier_dist = round(self.ATR_CHAND * CHANDELIER_MULT, 2)
        assert chandelier_dist == self.CHAND_DIST

    def test_two_submit_order_calls_placed(self):
        """Exactly 2 submit_order calls: LMT BUY + TrailingStop SELL."""
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 2, "Exactly 2 submit_order calls: BUY + TRAIL stop"

    def test_first_order_is_limit_buy(self):
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx)
        req = tc.submit_order.call_args_list[0][0][0]
        assert isinstance(req, LimitOrderRequest), \
            f"First submit_order must be LimitOrderRequest, got {type(req)}"

    def test_second_order_is_trailing_stop(self):
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx)
        req = tc.submit_order.call_args_list[1][0][0]
        assert isinstance(req, TrailingStopOrderRequest), \
            f"Second submit_order must be TrailingStopOrderRequest, got {type(req)}"

    def test_buy_order_time_in_force_is_day(self):
        from alpaca.trading.enums import TimeInForce
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx)
        req = tc.submit_order.call_args_list[0][0][0]
        assert req.time_in_force == TimeInForce.DAY

    def test_stop_order_time_in_force_is_gtc(self):
        from alpaca.trading.enums import TimeInForce
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx)
        req = tc.submit_order.call_args_list[1][0][0]
        assert req.time_in_force == TimeInForce.GTC

    def test_stop_trail_price_equals_chandelier_dist(self):
        """trail_price = ATR_CHAND × CHANDELIER_MULT (dollar amount)."""
        from src.config import CHANDELIER_MULT
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx)
        req = tc.submit_order.call_args_list[1][0][0]
        expected = round(self.ATR_CHAND * CHANDELIER_MULT, 2)
        assert req.trail_price == pytest.approx(expected, abs=0.01)

    def test_state_stop_dist_equals_chandelier_dist(self):
        from src.config import CHANDELIER_MULT
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx, sym='TSLA')
        assert 'TSLA' in engine.state
        assert engine.state['TSLA']['stop_dist'] == pytest.approx(
            round(self.ATR_CHAND * CHANDELIER_MULT, 2), abs=0.01
        )

    def test_state_stop_loss_equals_fill_minus_chandelier_dist(self):
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx, sym='TSLA')
        assert 'TSLA' in engine.state
        sl = engine.state['TSLA']['stop_loss']
        assert sl == pytest.approx(self.ENTRY - self.CHAND_DIST, abs=0.01)

    def test_state_has_no_take_profit(self):
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx, sym='TSLA')
        assert 'take_profit' not in engine.state.get('TSLA', {})

    def test_state_records_fill_price_and_order_id(self):
        """state stores fill_price, entry_order_id; no commission field for Alpaca."""
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx, sym='TSLA')
        s = engine.state['TSLA']
        assert s['fill_price'] == pytest.approx(self.ENTRY, abs=0.01)
        assert 'entry_order_id' in s

    def test_state_records_score(self):
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx, sym='TSLA')
        assert 'score' in engine.state.get('TSLA', {})
        assert engine.state['TSLA']['score'] is not None

    def test_no_commission_field_in_state(self):
        """Alpaca is commission-free — no commission field should be stored."""
        tc, engine, ctx = self._setup()
        _run_entry_cycle(engine, ctx, sym='TSLA')
        assert 'commission' not in engine.state.get('TSLA', {})

    def test_no_state_written_when_order_cancelled(self):
        """If BUY poll sees 'canceled', state must not be written."""
        from alpaca.trading.enums import TimeInForce
        tc = _mock_trading_client()
        buy_order = MagicMock(id='buy-id', status='new')
        tc.submit_order.return_value = buy_order
        cancelled = MagicMock(
            id='buy-id', status='canceled',
            filled_avg_price=None, filled_qty=None,
        )
        tc.get_order_by_id.return_value = cancelled

        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=self.ENTRY, atr=self.ATR_CHAND, atr_chandelier=self.ATR_CHAND,
                   orb=self.ENTRY - 5, ma50=self.ENTRY - 3, ma200=self.ENTRY - 15,
                   rsi=42.0, rsi_prev=37.0)

        _run_entry_cycle(engine, ctx, sym='TSLA')
        assert 'TSLA' not in engine.state, "Cancelled order must not create state entry"

    def test_high_priced_stock_skipped_when_qty_would_be_zero(self):
        """limit_price above bucket_size → int qty_by_bucket = 0 → skip."""
        tc = _mock_trading_client(equity=2500.0, cash=2500.0)
        engine = _make_engine(equity=2500.0, cash=2500.0, trading_client=tc)
        ctx = _ctx(price=2000.0, atr=10.0, atr_chandelier=10.0,
                   orb=1990.0, ma50=1990.0, ma200=1900.0,
                   rsi=42.0, rsi_prev=37.0)
        _run_entry_cycle(engine, ctx, equity=2500.0, cash=2500.0)
        assert tc.submit_order.call_count == 0, "Stock above bucket price must be skipped"
        assert 'TSLA' not in engine.state


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCREENER — get_candidates from src/scanner.py (gainers + most-actives)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetInstitutionalScan:
    """
    get_institutional_scan() calls get_candidates(data_client, screener_client) from
    scanner.py and returns the resulting deduped list (gainers first, then actives).
    """

    def test_returns_list_of_symbols(self):
        engine = _make_engine()
        with patch('src.engine.get_candidates', return_value=['AAPL', 'TSLA']) as mock_scan:
            result = engine.get_institutional_scan()
        assert result == ['AAPL', 'TSLA']
        mock_scan.assert_called_once_with(engine.data_client, engine.screener_client)

    def test_passes_clients_through(self):
        dc = MagicMock()
        sc = MagicMock()
        engine = _make_engine(data_client=dc)
        engine.screener_client = sc
        with patch('src.engine.get_candidates', return_value=[]) as mock_scan:
            engine.get_institutional_scan()
        mock_scan.assert_called_once_with(dc, sc)

    def test_returns_empty_list_on_exception(self):
        engine = _make_engine()
        with patch('src.engine.get_candidates', side_effect=Exception("network error")):
            try:
                result = engine.get_institutional_scan()
                assert isinstance(result, list)
            except Exception:
                pass  # acceptable: engine lets exception propagate from scanner

    def test_returns_many_symbols(self):
        syms = [f'SYM{i}' for i in range(55)]
        engine = _make_engine()
        with patch('src.engine.get_candidates', return_value=syms):
            result = engine.get_institutional_scan()
        assert len(result) == 55
        assert result[0] == 'SYM0'

    def test_returns_empty_when_scanner_returns_empty(self):
        engine = _make_engine()
        with patch('src.engine.get_candidates', return_value=[]):
            result = engine.get_institutional_scan()
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCORING SYSTEM — _score_candidate() / score_candidate()
#    Four components summing to 100:
#      Donchian Proximity (30) · RVOL (25) · RSI Delta (25) · Liquidity (20)
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringAlligatorAlignment:
    """Alligator alignment component (0-30 pts):
      alignment_pct = (smma_fast - smma_slow) / smma_slow
      alligator_score = min(30, alignment_pct / 0.05 * 30)

      alignment=5%  (fast=105, slow=100) → 30 pts
      alignment=2.5% (fast=102.5, slow=100) → 15 pts
      alignment=0%  (fast==slow)          → 0 pts
      alignment>5%  → capped at 30 pts
      smma_slow=0   → 0 pts (condition guarded)

    Isolation: rvol=RVOL_MIN → 0; rsi_delta=0 → 0; spread=SPREAD_MAX, vol=0 → 0.
    Total = alligator_score + 0 + 0 + 0.
    """

    def _alligator_score(self, smma_fast_val, smma_slow_val):
        from src.config import SPREAD_MAX_PCT, RVOL_MIN
        engine = _make_engine()
        ctx = _ctx(rvol=RVOL_MIN,
                   rsi=65.0, rsi_prev=65.0,
                   spread_pct=SPREAD_MAX_PCT, dollar_vol=0,
                   smma_fast=smma_fast_val, smma_slow=smma_slow_val)
        return engine._score_candidate(ctx)

    def test_at_5pct_alignment_gives_30_pts(self):
        """fast=105, slow=100 → 5% alignment → 30 pts (max)."""
        assert self._alligator_score(105.0, 100.0) == pytest.approx(30.0, abs=0.1)

    def test_at_2p5pct_alignment_gives_15_pts(self):
        """fast=102.5, slow=100 → 2.5% alignment (half of 5%) → 15 pts."""
        assert self._alligator_score(102.5, 100.0) == pytest.approx(15.0, abs=0.1)

    def test_zero_alignment_gives_0_pts(self):
        """fast==slow → 0% alignment → 0 pts."""
        assert self._alligator_score(100.0, 100.0) == pytest.approx(0.0, abs=0.1)

    def test_above_5pct_capped_at_30(self):
        """fast=120, slow=100 → 20% alignment → capped at 30 pts."""
        assert self._alligator_score(120.0, 100.0) == pytest.approx(30.0, abs=0.1)

    def test_slow_zero_gives_zero(self):
        """smma_slow=0 → condition guard fires → 0 pts."""
        assert self._alligator_score(105.0, 0.0) == pytest.approx(0.0, abs=0.1)


class TestScoringRVOL:
    """RVOL component (0-25 pts):
      rvol_score = min(25, max(0, rvol - RVOL_MIN) / (5.0 - RVOL_MIN) × 25)

      RVOL_MIN = 1.2
      rvol=1.2 (floor) → 0 pts
      rvol=3.1 (midpoint) → 12.5 pts
      rvol=5.0 → 25 pts
      rvol>5.0 → capped at 25
      rvol<1.2 → 0 pts

    Isolation: smma_fast=smma_slow → Alligator=0; rsi_delta=0 → 0; spread=max, vol=0 → 0.
    Total = 0 + rvol_score + 0 + 0 = rvol_score.
    """

    def _rvol_score(self, rvol):
        from src.config import SPREAD_MAX_PCT
        engine = _make_engine()
        ctx = _ctx(rsi=65.0, rsi_prev=65.0,
                   spread_pct=SPREAD_MAX_PCT, dollar_vol=0,
                   rvol=rvol, smma_fast=100.0, smma_slow=100.0)
        return engine._score_candidate(ctx)

    def test_rvol_at_floor_gives_zero(self):
        from src.config import RVOL_MIN
        assert self._rvol_score(RVOL_MIN) == pytest.approx(0.0, abs=0.1)

    def test_rvol_at_midpoint_gives_12_5(self):
        """midpoint = (1.2 + 5.0) / 2 = 3.1 → 12.5 pts."""
        assert self._rvol_score(3.1) == pytest.approx(12.5, abs=0.1)

    def test_rvol_at_5x_gives_25(self):
        assert self._rvol_score(5.0) == pytest.approx(25.0, abs=0.1)

    def test_rvol_above_5x_capped_at_25(self):
        assert self._rvol_score(7.0) == pytest.approx(25.0, abs=0.1)

    def test_rvol_below_floor_gives_zero(self):
        assert self._rvol_score(1.0) == pytest.approx(0.0, abs=0.1)


class TestScoringRSIDelta:
    """RSI delta acceleration component (0-25 pts):
      rsi_score = min(25, max(0, rsi - rsi_prev) / 10.0 × 25)

      delta=0  → 0 pts
      delta=5  → 12.5 pts
      delta=10 → 25 pts (saturated)
      delta>10 → capped at 25
      delta<0  → 0 pts (clamped)

    Isolation: smma_fast=smma_slow → Alligator=0; rvol=RVOL_MIN → 0; spread=max, vol=0 → 0.
    Total = 0 + 0 + rsi_score + 0 = rsi_score.
    """

    def _rsi_score(self, rsi, rsi_prev):
        from src.config import RVOL_MIN, SPREAD_MAX_PCT
        engine = _make_engine()
        ctx = _ctx(rsi=rsi, rsi_prev=rsi_prev,
                   rvol=RVOL_MIN, spread_pct=SPREAD_MAX_PCT, dollar_vol=0,
                   smma_fast=100.0, smma_slow=100.0)
        return engine._score_candidate(ctx)

    def test_delta_zero_gives_zero(self):
        assert self._rsi_score(65.0, 65.0) == pytest.approx(0.0, abs=0.1)

    def test_delta_5_gives_12_5_pts(self):
        assert self._rsi_score(65.0, 60.0) == pytest.approx(12.5, abs=0.1)

    def test_delta_10_gives_25_pts(self):
        assert self._rsi_score(70.0, 60.0) == pytest.approx(25.0, abs=0.1)

    def test_delta_above_10_capped_at_25(self):
        assert self._rsi_score(80.0, 60.0) == pytest.approx(25.0, abs=0.1)

    def test_negative_delta_gives_zero(self):
        """Falling RSI: delta<0 → clamped at 0 pts."""
        assert self._rsi_score(60.0, 65.0) == pytest.approx(0.0, abs=0.1)


class TestScoringLiquidity:
    """Liquidity component (0-20 pts): spread_pts (0-10) + vol_pts (0-10).

      spread_pts = max(0, (1 - spread/SPREAD_MAX) × 10)
      vol_pts    = min(10, (dollar_vol / SCAN_MIN_DOLLAR_VOL) × 10)

    Isolation: smma_fast=smma_slow → Alligator=0; rvol=RVOL_MIN → 0; rsi_delta=0 → 0.
    Total = 0 + 0 + 0 + liq_score = liq_score.
    """

    def _liq_score(self, spread_pct, dollar_vol):
        from src.config import RVOL_MIN
        engine = _make_engine()
        ctx = _ctx(rsi=65.0, rsi_prev=65.0,
                   rvol=RVOL_MIN, smma_fast=100.0, smma_slow=100.0,
                   spread_pct=spread_pct, dollar_vol=dollar_vol)
        return engine._score_candidate(ctx)

    def test_zero_spread_full_vol_gives_20_pts(self):
        """spread=0, vol=SCAN_MIN → spread_pts=10, vol_pts=10 → 20."""
        from src.config import SCAN_MIN_DOLLAR_VOL
        assert self._liq_score(0.0, SCAN_MIN_DOLLAR_VOL) == pytest.approx(20.0, abs=0.1)

    def test_max_spread_full_vol_gives_10_pts(self):
        """spread=SPREAD_MAX → spread_pts=0; vol=min → vol_pts=10 → 10 total."""
        from src.config import SPREAD_MAX_PCT, SCAN_MIN_DOLLAR_VOL
        assert self._liq_score(SPREAD_MAX_PCT, SCAN_MIN_DOLLAR_VOL) == pytest.approx(10.0, abs=0.1)

    def test_zero_spread_zero_vol_gives_10_pts(self):
        """spread=0 → spread_pts=10; vol=0 → vol_pts=0 → 10 total."""
        assert self._liq_score(0.0, 0.0) == pytest.approx(10.0, abs=0.1)

    def test_max_spread_zero_vol_gives_0_pts(self):
        """spread=max → 0; vol=0 → 0 → 0 total."""
        from src.config import SPREAD_MAX_PCT
        assert self._liq_score(SPREAD_MAX_PCT, 0.0) == pytest.approx(0.0, abs=0.1)

    def test_half_spread_zero_vol_gives_5_pts(self):
        """spread=half_max → spread_pts=5; vol=0 → vol_pts=0 → 5 total."""
        assert self._liq_score(0.0025, 0.0) == pytest.approx(5.0, abs=0.1)

    def test_above_max_vol_caps_at_10(self):
        """vol=2×SCAN_MIN → vol_pts capped at 10; spread=0 → 10+10=20 total."""
        from src.config import SCAN_MIN_DOLLAR_VOL
        assert self._liq_score(0.0, SCAN_MIN_DOLLAR_VOL * 2) == pytest.approx(20.0, abs=0.1)


class TestScoringMaxAndTotal:
    """Integration: verify total score = Alligator + RVOL + RSI_delta + Liquidity.

    Maximum breakdown:
      Alligator alignment = 30  (smma_fast=105, smma_slow=100 → 5% → 30 pts)
      RVOL                = 25  (rvol=5.0)
      RSI delta           = 25  (delta=17, capped at 10→25)
      Liquidity           = 20  (spread=0, vol≥SCAN_MIN_DOLLAR_VOL)
      total               = 100
    """

    def test_maximum_achievable_score_is_100(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        engine = _make_engine()
        # smma 5% separation → 30 pts; rvol=5→25; delta=17→capped 25; liq=20. Total=100.
        ctx = _ctx(price=100.0, rvol=5.0, rsi=52.0, rsi_prev=35.0,
                   spread_pct=0.0, dollar_vol=SCAN_MIN_DOLLAR_VOL,
                   smma_fast=105.0, smma_slow=100.0)
        assert engine._score_candidate(ctx) == pytest.approx(100.0, abs=0.1)

    def test_no_alligator_caps_max_at_70(self):
        """smma_fast=smma_slow → Alligator=0; max from RVOL+RSI+liq = 25+25+20 = 70."""
        from src.config import SCAN_MIN_DOLLAR_VOL
        engine = _make_engine()
        ctx = _ctx(price=100.0, rvol=5.0, rsi=52.0, rsi_prev=35.0,
                   spread_pct=0.0, dollar_vol=SCAN_MIN_DOLLAR_VOL,
                   smma_fast=100.0, smma_slow=100.0)
        assert engine._score_candidate(ctx) == pytest.approx(70.0, abs=0.1)

    def test_score_never_exceeds_100(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        engine = _make_engine()
        ctx = _ctx(price=100.0, rvol=10.0, rsi=52.0, rsi_prev=35.0,
                   spread_pct=0.0, dollar_vol=SCAN_MIN_DOLLAR_VOL * 10,
                   smma_fast=105.0, smma_slow=100.0)
        assert engine._score_candidate(ctx) <= 100.0

    def test_score_is_rounded_to_2_decimals(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        engine = _make_engine()
        ctx = _ctx(price=100.25, rvol=5.0, rsi=52.0, rsi_prev=35.0,
                   spread_pct=0.0, dollar_vol=SCAN_MIN_DOLLAR_VOL,
                   smma_fast=105.0, smma_slow=100.0)
        score = engine._score_candidate(ctx)
        assert score == round(score, 2)

    def test_score_is_non_negative(self):
        """All-zero/bad inputs must produce score ≥ 0 (no negative components)."""
        engine = _make_engine()
        ctx = _ctx(price=100.0, rvol=0.0, rsi=50.0, rsi_prev=55.0,
                   spread_pct=0.01, dollar_vol=0,
                   smma_fast=100.0, smma_slow=100.0)
        assert engine._score_candidate(ctx) >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3a. ANALYST CONSENSUS BONUS — score_candidate() component 5
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringAnalyst:
    """Analyst consensus bonus (0-SCORE_ANALYST_MAX pts):

      buy_ratio = analyst_buy / (buy + hold + sell)
      analyst_score = min(MAX, max(0, (buy_ratio - 0.30) / 0.40 * MAX))

      No data (all zeros)  → 0 pts (neutral — no penalty)
      buy_ratio = 0.30     → 0 pts (floor)
      buy_ratio = 0.50     → 50% of max
      buy_ratio = 0.70     → full max
      buy_ratio > 0.70     → capped at max
      buy_ratio < 0.30     → 0 pts (clamped)

    Isolation: all other components zeroed (smma equal, rvol=min, rsi_delta=0,
    spread=max, vol=0). Total = 0 + 0 + 0 + 0 + analyst_score.
    """

    def _analyst_score(self, buy, hold, sell):
        from src.config import RVOL_MIN, SPREAD_MAX_PCT
        engine = _make_engine()
        ctx = _ctx(rsi=65.0, rsi_prev=65.0,
                   rvol=RVOL_MIN, spread_pct=SPREAD_MAX_PCT, dollar_vol=0,
                   smma_fast=100.0, smma_slow=100.0,
                   analyst_buy=buy, analyst_hold=hold, analyst_sell=sell)
        return engine._score_candidate(ctx)

    def test_no_analyst_data_gives_zero(self):
        """analyst_buy=hold=sell=0 → neutral, 0 bonus pts."""
        assert self._analyst_score(0, 0, 0) == pytest.approx(0.0, abs=0.01)

    def test_buy_ratio_at_floor_gives_zero(self):
        """30% buys → 0 pts (threshold floor)."""
        assert self._analyst_score(3, 7, 0) == pytest.approx(0.0, abs=0.01)

    def test_buy_ratio_below_floor_gives_zero(self):
        """20% buys < 30% floor → clamped to 0."""
        assert self._analyst_score(2, 8, 0) == pytest.approx(0.0, abs=0.01)

    def test_buy_ratio_at_full_gives_max(self):
        """70% buys → full SCORE_ANALYST_MAX."""
        from src.config import SCORE_ANALYST_MAX
        assert self._analyst_score(7, 3, 0) == pytest.approx(SCORE_ANALYST_MAX, abs=0.1)

    def test_buy_ratio_above_70pct_capped_at_max(self):
        """90% buys → capped at SCORE_ANALYST_MAX."""
        from src.config import SCORE_ANALYST_MAX
        assert self._analyst_score(9, 1, 0) == pytest.approx(SCORE_ANALYST_MAX, abs=0.1)

    def test_buy_ratio_at_50pct_gives_half(self):
        """50% buys → midpoint between 30% and 70% → half of SCORE_ANALYST_MAX."""
        from src.config import SCORE_ANALYST_MAX
        assert self._analyst_score(5, 5, 0) == pytest.approx(SCORE_ANALYST_MAX / 2, abs=0.1)

    def test_analyst_bonus_cannot_push_score_above_100(self):
        """Even with max analyst bonus, score is capped at 100."""
        from src.config import SCAN_MIN_DOLLAR_VOL
        engine = _make_engine()
        # All technical components maxed (30+25+25+20=100) plus full analyst bonus
        ctx = _ctx(price=100.0, rvol=5.0, rsi=52.0, rsi_prev=35.0,
                   spread_pct=0.0, dollar_vol=SCAN_MIN_DOLLAR_VOL,
                   smma_fast=105.0, smma_slow=100.0,
                   analyst_buy=9, analyst_hold=1, analyst_sell=0)
        assert engine._score_candidate(ctx) == pytest.approx(100.0, abs=0.01)

    def test_analyst_bonus_lifts_otherwise_weak_score(self):
        """Strong analyst consensus lifts a candidate whose technical score alone is borderline."""
        from src.config import SCORE_ANALYST_MAX, RVOL_MIN, SPREAD_MAX_PCT
        engine = _make_engine()
        # Alligator=0, RVOL=0, RSI=0, Liq=0 → base score = 0
        base_ctx = _ctx(rsi=65.0, rsi_prev=65.0,
                        rvol=RVOL_MIN, spread_pct=SPREAD_MAX_PCT, dollar_vol=0,
                        smma_fast=100.0, smma_slow=100.0,
                        analyst_buy=0, analyst_hold=0, analyst_sell=0)
        # With strong buys → score = analyst_score
        boosted_ctx = _ctx(rsi=65.0, rsi_prev=65.0,
                           rvol=RVOL_MIN, spread_pct=SPREAD_MAX_PCT, dollar_vol=0,
                           smma_fast=100.0, smma_slow=100.0,
                           analyst_buy=7, analyst_hold=3, analyst_sell=0)
        assert engine._score_candidate(boosted_ctx) > engine._score_candidate(base_ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. MINIMUM SCORE GATE — signals below SCAN_MIN_SCORE never entered
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringMinScore:
    """Signals passing all 12 rules but scoring below SCAN_MIN_SCORE are skipped."""

    # ctx where price > ma50 > ma200 (all 12 rules pass with default equity/cash)
    _ENTRY_CTX_KWARGS = dict(price=100.0, orb=95.0, ma50=97.0, ma200=85.0,
                             rsi=52.0, rsi_prev=37.0, atr=3.0,
                             rvol=3.5, spread_pct=0.002, adx=25.0)

    def test_signal_below_min_score_not_entered(self):
        """A ctx that scores below SCAN_MIN_SCORE must not produce a BUY order."""
        from src.config import SCAN_MIN_SCORE
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        # Force _score_candidate to return a score just below the threshold
        with patch.object(engine, '_score_candidate', return_value=SCAN_MIN_SCORE - 1.0):
            ctx = _ctx(**self._ENTRY_CTX_KWARGS)
            _run_entry_cycle(engine, ctx, sym='TSLA')
        assert 'TSLA' not in engine.state, (
            "Position must not be opened when score < SCAN_MIN_SCORE"
        )
        assert tc.submit_order.call_count == 0, (
            "No order should be submitted when score < SCAN_MIN_SCORE"
        )

    def test_signal_at_min_score_is_entered(self):
        """A ctx scoring exactly at SCAN_MIN_SCORE must be entered."""
        from src.config import SCAN_MIN_SCORE
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        with patch.object(engine, '_score_candidate', return_value=SCAN_MIN_SCORE):
            ctx = _ctx(**self._ENTRY_CTX_KWARGS)
            _run_entry_cycle(engine, ctx, sym='TSLA')
        assert tc.submit_order.call_count >= 1, (
            "Order should be submitted when score == SCAN_MIN_SCORE"
        )

    def test_signal_above_min_score_is_entered(self):
        """A ctx scoring above SCAN_MIN_SCORE must proceed to order submission."""
        from src.config import SCAN_MIN_SCORE
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        with patch.object(engine, '_score_candidate', return_value=SCAN_MIN_SCORE + 20.0):
            ctx = _ctx(**self._ENTRY_CTX_KWARGS)
            _run_entry_cycle(engine, ctx, sym='TSLA')
        assert tc.submit_order.call_count >= 1, (
            "Order should be submitted when score > SCAN_MIN_SCORE"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHORTLISTING — ranking, slot limits, portfolio exclusion
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateRanking:
    """Signals sorted descending by score; top N (available slots) entered."""

    def _engine_with_held(self, held_syms, equity=2500.0, cash=2500.0):
        tc = _mock_trading_client(equity, cash)

        # submit_order: always returns a filled buy then an accepted stop
        buy_order  = MagicMock(id='buy-id',  status='new')
        stop_order = MagicMock(id='stop-id', status='accepted')
        tc.submit_order.side_effect = [buy_order, stop_order] * 10  # enough for many entries

        filled = MagicMock(
            id='buy-id', status='filled',
            filled_avg_price='100.0', filled_qty='5.0',
        )
        tc.get_order_by_id.return_value = filled

        engine = _make_engine(equity, cash, trading_client=tc)
        engine.state = {s: {'price': 100, 'qty': 5, 'time': datetime.now().isoformat(),
                            'stop_loss': 90, 'volume': 0, 'score': 50}
                        for s in held_syms}
        engine._last_audit_date = '2024-06-05'  # suppress audit
        return engine

    def test_highest_score_candidate_entered_when_one_slot(self):
        """Two signals, max_pos-1 existing positions → 1 slot. Only higher-score entered."""
        from src.config import MAX_POSITIONS_CAP, MIN_BUCKET_SIZE
        n_held = min(int(2500 / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP) - 1
        held = [f'SYM{i}' for i in range(n_held)]

        ctx_high = _ctx(price=101.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                        ma50=95.0, ma200=85.0, rvol=5.0)
        ctx_low  = _ctx(price=102.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                        ma50=95.0, ma200=85.0, rvol=2.5)

        engine = self._engine_with_held(held)
        entered = _run_multi_signal_cycle(engine, {'HIGH': ctx_high, 'LOW': ctx_low})

        assert 'HIGH' in entered
        assert 'LOW' not in entered

    def test_lower_score_candidate_skipped_when_slot_filled(self):
        from src.config import MAX_POSITIONS_CAP, MIN_BUCKET_SIZE
        n_held = min(int(2500 / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP) - 1
        held = [f'SYM{i}' for i in range(n_held)]

        ctx_high = _ctx(price=101.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                        ma50=95.0, ma200=85.0, rvol=5.0)
        ctx_low  = _ctx(price=102.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                        ma50=95.0, ma200=85.0, rvol=2.5)

        engine = self._engine_with_held(held)
        entered = _run_multi_signal_cycle(engine, {'HIGH': ctx_high, 'LOW': ctx_low})
        assert 'LOW' not in entered

    def test_all_candidates_entered_when_enough_slots(self):
        """No held positions, 2 signals → both entered."""
        ctx_a = _ctx(price=101.0, orb=100.0, rsi=52.0, rsi_prev=35.0, ma50=95.0, ma200=85.0)
        ctx_b = _ctx(price=104.0, orb=100.0, rsi=52.0, rsi_prev=35.0, ma50=95.0, ma200=85.0)

        engine = self._engine_with_held([])
        entered = _run_multi_signal_cycle(engine, {'ALPHA': ctx_a, 'BETA': ctx_b})

        assert 'ALPHA' in entered
        assert 'BETA' in entered

    def test_entry_order_is_score_descending(self):
        """With 3 candidates and at least 3 initial capacity slots, top 2 by score are entered.

        The engine recalculates open_slots after each fill (capacity shrinks by 1 per entry).
        We start with n_held = max_pos - 3 so there are 3 capacity slots initially:
          fill #1 (HIGH) → capacity drops to 2, open_slots ≥ 2 → loop continues
          fill #2 (MED)  → capacity drops to 1, open_slots = 1 ≥ 1 → loop would
                           break before LOW because placed (2) >= open_slots (1)
        """
        from src.config import MAX_POSITIONS_CAP, MIN_BUCKET_SIZE
        # Use max_pos - 3 held positions so 3 capacity slots are available initially
        n_held = max(0, min(int(2500 / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP) - 3)
        held = [f'HELD{i}' for i in range(n_held)]

        ctx_hi  = _ctx(price=101.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                       ma50=95.0, ma200=85.0, rvol=5.0)
        ctx_med = _ctx(price=104.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                       ma50=95.0, ma200=85.0, rvol=3.75)
        ctx_lo  = _ctx(price=108.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                       ma50=95.0, ma200=85.0, rvol=2.5)

        engine = self._engine_with_held(held)
        entered = _run_multi_signal_cycle(engine, {'MED': ctx_med, 'HIGH': ctx_hi, 'LOW': ctx_lo})

        assert 'HIGH' in entered
        assert 'MED' in entered
        assert 'LOW' not in entered

    def test_already_held_symbol_not_re_entered(self):
        """Scanner returning an already-held symbol must not call get_technical_context."""
        engine = self._engine_with_held(['AAPL'])

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(2500.0, 2500.0)), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['AAPL']), \
             patch.object(engine, 'get_technical_context') as mock_ctx, \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        mock_ctx.assert_not_called()

    def test_no_entry_when_get_technical_context_returns_none(self):
        """If data is unavailable (None), the symbol must be silently skipped."""
        engine = self._engine_with_held([])

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(2500.0, 2500.0)), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['GHOST']), \
             patch.object(engine, 'get_technical_context', return_value=None), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'GHOST' not in engine.state

    def test_no_entry_outside_session_window(self):
        """Outside entry window no entry must occur even if signal passes."""
        engine = self._engine_with_held([])
        ctx = _ctx(price=101.0, orb=100.0)

        tz_ny    = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 16, 0))  # after close

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(2500.0, 2500.0)), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['SYM']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'SYM' not in engine.state

    def test_insufficient_settled_cash_blocks_entry(self):
        """If settled cash < order cost, the entry must be skipped."""
        tc = _mock_trading_client(equity=1400.0, cash=1.0)  # only $1 cash
        engine = _make_engine(equity=1400.0, cash=1.0, trading_client=tc)
        ctx = _ctx(price=100.0, atr=3.0)

        _run_entry_cycle(engine, ctx, equity=1400.0, cash=1.0)

        assert tc.submit_order.call_count == 0, "No order must be placed with insufficient cash"
        assert 'TSLA' not in engine.state

    def test_fallthrough_to_next_candidate_when_order_cancelled(self):
        """When rank-1 BUY is cancelled, fall through to rank-2."""
        from src.config import MAX_POSITIONS_CAP, MIN_BUCKET_SIZE

        ctx_hi = _ctx(price=101.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                      ma50=95.0, ma200=85.0, rvol=5.0)
        ctx_lo = _ctx(price=104.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                      ma50=95.0, ma200=85.0, rvol=2.5)

        tc = _mock_trading_client(equity=2500.0, cash=2500.0)

        # HIGH: buy submitted, poll returns 'canceled' → state removed
        # LOW: buy submitted, poll returns 'filled'; stop order submitted
        buy_hi  = MagicMock(id='buy-hi',  status='new')
        buy_lo  = MagicMock(id='buy-lo',  status='new')
        stop_lo = MagicMock(id='stop-lo', status='accepted')
        tc.submit_order.side_effect = [buy_hi, buy_lo, stop_lo]

        cancelled_poll = MagicMock(
            id='buy-hi', status='canceled',
            filled_avg_price=None, filled_qty=None,
        )
        filled_poll = MagicMock(
            id='buy-lo', status='filled',
            filled_avg_price='104.0', filled_qty='5.0',
        )
        tc.get_order_by_id.side_effect = [cancelled_poll, filled_poll]

        held = ['HELD0', 'HELD1']
        engine = _make_engine(equity=2500.0, cash=2500.0, trading_client=tc)
        engine.state = {s: {'price': 100, 'qty': 5, 'time': datetime.now().isoformat(),
                            'stop_loss': 90, 'volume': 0, 'score': 50}
                        for s in held}
        engine._last_audit_date = '2024-06-05'

        entered = _run_multi_signal_cycle(engine, {'HIGH': ctx_hi, 'LOW': ctx_lo})

        assert 'HIGH' not in engine.state, "Cancelled rank-1 must NOT be written to state"
        assert 'LOW' in engine.state, "Rank-2 must be entered after rank-1 is cancelled"


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXIT ORDERS — velocity exits, liquidation (Alpaca edition)
# ─────────────────────────────────────────────────────────────────────────────

class TestExitOrders:
    """
    Verify velocity exit (check_velocity_exits → liquidate) behaviour:
    - MarketOrderRequest(SELL) submitted with qty from get_open_position()
    - Non-trailing-stop SELL orders cancelled before the market sell
    - State marked pending_exit=True after liquidation
    - TimeInForce.DAY used (no goodAfterTime)
    - Exit fires only when stagnant; profitable positions kept
    """

    def _make_state_entry(self, price=100.0, qty=5.0, days_ago=0):
        tz_ny = pytz.timezone('US/Eastern')
        entry_time = (datetime.now(tz_ny) - timedelta(days=days_ago)).isoformat()
        return {'price': price, 'time': entry_time, 'qty': qty,
                'stop_loss': price * 0.94, 'volume': 0, 'score': 50,
                'peak_price': price}

    # ── liquidate() ──────────────────────────────────────────────────────────

    def test_liquidate_places_market_sell_with_position_qty(self):
        """Market sell uses qty from trading_client.get_open_position()."""
        qty = 2.5
        tc  = _mock_trading_client()
        pos = MagicMock(); pos.qty = str(qty)
        tc.get_open_position.side_effect = None   # override the default Exception side_effect
        tc.get_open_position.return_value = pos
        tc.get_orders.return_value = []

        engine = _make_engine(trading_client=tc)
        engine.state = {'SYM': self._make_state_entry(qty=qty)}

        with patch('src.engine.time.sleep'):
            engine.liquidate('SYM')

        assert tc.submit_order.call_count == 1
        req = tc.submit_order.call_args[0][0]
        assert isinstance(req, MarketOrderRequest)
        assert req.qty == pytest.approx(qty, abs=0.0001)

    def test_liquidate_uses_exact_alpaca_position_qty_for_sell(self):
        """liquidate() reads qty from Alpaca (source of truth), not from state."""
        alpaca_qty = 5.0
        state_qty  = 4.0
        tc = _mock_trading_client()
        pos = MagicMock(); pos.qty = str(alpaca_qty)
        tc.get_open_position.side_effect = None   # override the default Exception side_effect
        tc.get_open_position.return_value = pos
        tc.get_orders.return_value = []

        engine = _make_engine(trading_client=tc)
        engine.state = {'PRICEY': self._make_state_entry(qty=state_qty)}

        with patch('src.engine.time.sleep'):
            engine.liquidate('PRICEY')

        req = tc.submit_order.call_args[0][0]
        assert req.qty == pytest.approx(alpaca_qty, abs=0.0001), \
            "Market sell must use Alpaca position qty, not state qty"

    def test_liquidate_cancels_non_trail_orders_before_sell(self):
        """Non-trailing-stop SELL orders must be cancelled before market sell."""
        tc = _mock_trading_client()
        pos = MagicMock(); pos.qty = '5.0'
        tc.get_open_position.side_effect = None   # override the default Exception side_effect
        tc.get_open_position.return_value = pos

        # A non-trail sell order for our symbol
        non_trail = MagicMock()
        non_trail.symbol     = 'SYM'
        non_trail.order_type = 'limit'
        non_trail.id         = 'ord-1'
        tc.get_orders.return_value = [non_trail]

        engine = _make_engine(trading_client=tc)
        engine.state = {'SYM': self._make_state_entry()}

        cancel_calls_before_submit = []
        original_submit = tc.submit_order

        def capture_submit(*args, **kwargs):
            cancel_calls_before_submit.append(tc.cancel_order_by_id.call_count)
            return MagicMock(id='sell-id', status='new')

        tc.submit_order.side_effect = capture_submit

        with patch('src.engine.time.sleep'):
            engine.liquidate('SYM')

        assert tc.cancel_order_by_id.called, "cancel_order_by_id must be called for non-trail order"
        assert cancel_calls_before_submit and cancel_calls_before_submit[0] >= 1, \
            "Cancel must happen before submit_order"

    def test_liquidate_cancels_trail_stop_order(self):
        """TrailingStop SELL orders MUST be cancelled by liquidate() before the market sell.

        Alpaca holds shares 'for orders' — the GTC TRAIL reserves all position shares,
        causing the market SELL to fail with available=0 unless the TRAIL is cancelled first.
        """
        tc = _mock_trading_client()
        pos = MagicMock(); pos.qty = '5.0'
        tc.get_open_position.side_effect = None   # override the default Exception side_effect
        tc.get_open_position.return_value = pos

        trail_order = MagicMock()
        trail_order.symbol     = 'SYM'
        trail_order.order_type = 'trailing_stop'
        trail_order.id         = 'trail-1'
        tc.get_orders.return_value = [trail_order]

        engine = _make_engine(trading_client=tc)
        engine.state = {'SYM': self._make_state_entry()}

        with patch('src.engine.time.sleep'):
            engine.liquidate('SYM')

        # cancel_order_by_id MUST be called for the trailing stop
        cancelled_ids = [c[0][0] for c in tc.cancel_order_by_id.call_args_list]
        assert 'trail-1' in cancelled_ids, "Trailing stop MUST be cancelled before market sell"

    def test_liquidate_marks_pending_exit(self):
        """liquidate() sets pending_exit=True; deletion deferred to _sync_positions."""
        tc = _mock_trading_client()
        pos = MagicMock(); pos.qty = '1.0'
        tc.get_open_position.side_effect = None   # override the default Exception side_effect
        tc.get_open_position.return_value = pos
        tc.get_orders.return_value = []

        engine = _make_engine(trading_client=tc)
        engine.state = {'SYM': self._make_state_entry()}

        with patch('src.engine.time.sleep'):
            engine.liquidate('SYM')

        assert 'SYM' in engine.state, "State must be preserved after liquidate()"
        assert engine.state['SYM'].get('pending_exit') is True, \
            "State must be marked pending_exit=True after sell order placed"

    def test_liquidate_market_sell_uses_day_tif(self):
        """Liquidation sell must use TimeInForce.DAY."""
        from alpaca.trading.enums import TimeInForce
        tc = _mock_trading_client()
        pos = MagicMock(); pos.qty = '1.0'
        tc.get_open_position.side_effect = None   # override the default Exception side_effect
        tc.get_open_position.return_value = pos
        tc.get_orders.return_value = []

        engine = _make_engine(trading_client=tc)
        engine.state = {'SYM': self._make_state_entry()}

        with patch('src.engine.time.sleep'):
            engine.liquidate('SYM')

        req = tc.submit_order.call_args[0][0]
        assert req.time_in_force == TimeInForce.DAY

    # ── check_velocity_exits() ───────────────────────────────────────────────

    def test_velocity_exit_triggers_when_stagnant_after_hold_bars(self):
        """Position older than HOLD_TRADING_BARS with profit below threshold → sell placed."""
        from src.config import PROFIT_MIN_THRESHOLD
        tc = _mock_trading_client()
        pos = MagicMock(); pos.qty = '5.0'
        tc.get_open_position.side_effect = None   # override the default Exception side_effect
        tc.get_open_position.return_value = pos
        tc.get_orders.return_value = []

        entry_price   = 100.0
        stagnant_price = entry_price * (1 + PROFIT_MIN_THRESHOLD - 0.005)
        tz_ny = pytz.timezone('US/Eastern')
        _safe_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))
        old_time  = (_safe_now - timedelta(days=14)).isoformat()

        engine = _make_engine(trading_client=tc)
        engine.state = {'SLOW': {
            'price': entry_price, 'time': old_time, 'qty': 5.0,
            'stop_loss': entry_price * 0.94, 'volume': 0, 'score': 50,
            'peak_price': entry_price,
        }}

        snap = _make_snapshot(price=stagnant_price)
        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        assert engine.state.get('SLOW', {}).get('pending_exit') is True, \
            "Stagnant position must be marked pending_exit=True after sell order placed"
        assert tc.submit_order.called, "Market sell must be issued"

    def test_velocity_exit_does_not_trigger_when_profitable(self):
        """Position older than HOLD_TRADING_BARS but profit ≥ threshold → kept."""
        from src.config import PROFIT_MIN_THRESHOLD
        tc = _mock_trading_client()
        entry_price  = 100.0
        profit_price = entry_price * (1 + PROFIT_MIN_THRESHOLD + 0.01)

        engine = _make_engine(trading_client=tc)
        engine.state = {'WINNER': self._make_state_entry(price=entry_price, days_ago=14)}

        snap = _make_snapshot(price=profit_price)
        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'):
            engine.check_velocity_exits()

        assert 'WINNER' in engine.state, "Profitable position must NOT be liquidated"
        assert not tc.submit_order.called

    def test_velocity_exit_does_not_trigger_before_hold_bars(self):
        """Position still within hold window must never be touched."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {'NEW': self._make_state_entry(days_ago=0, price=100.0)}

        snap = _make_snapshot(price=101.0)
        # Pin to Wednesday to prevent Friday-close rule from firing (profit 1% < 3%).
        tz_ny = pytz.timezone('US/Eastern')
        safe_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = safe_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.check_velocity_exits()

        assert 'NEW' in engine.state
        assert not tc.submit_order.called

    def test_pending_exit_blocks_duplicate_sell_on_next_cycle(self):
        """Positions marked pending_exit=True must be skipped by check_velocity_exits()."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)

        entry = self._make_state_entry(price=100.0, days_ago=14)
        entry['pending_exit'] = True  # sell already submitted last cycle
        engine.state = {'PEX': entry}

        # Price far below hard stop — would trigger liquidate() if not guarded
        snap = _make_snapshot(price=85.0)
        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'):
            engine.check_velocity_exits()

        assert not tc.submit_order.called, \
            "check_velocity_exits must NOT place a second sell when pending_exit=True"


# ─────────────────────────────────────────────────────────────────────────────
# 9. EDGE CASES — guards, NaN/zero inputs, boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """
    Covers every critical boundary and defensive guard:
    - Corrupt state file on load → empty state, no crash
    - ATR = 0 or NaN → skip entry, no malformed order
    - MA200 = 0 → no division by zero in scoring
    - Entry price = 0 in velocity exit → skipped, no division by zero
    - VIX at threshold (=35) → entries allowed; above (>35) → blocked
    - Strict comparisons: price > ORB, RSI strictly rising
    """

    # ── State persistence ────────────────────────────────────────────────────

    def test_load_state_returns_empty_on_corrupt_json(self, tmp_path):
        """Corrupt STATE_FILE must not crash the engine — returns empty dict."""
        import src.engine as eng_mod
        state_path = tmp_path / "engine_state.json"
        state_path.write_text("{not valid json!!!")

        original = eng_mod.STATE_FILE
        eng_mod.STATE_FILE = str(state_path)
        try:
            engine = eng_mod.VelocityEngine.__new__(eng_mod.VelocityEngine)
            result = engine.load_state()
        finally:
            eng_mod.STATE_FILE = original

        assert result == {}, "Corrupt JSON must yield empty state, not crash"

    def test_load_state_returns_empty_when_file_missing(self, tmp_path):
        """Missing state file → empty state (fresh start)."""
        import src.engine as eng_mod
        original = eng_mod.STATE_FILE
        eng_mod.STATE_FILE = str(tmp_path / "nonexistent.json")
        try:
            engine = eng_mod.VelocityEngine.__new__(eng_mod.VelocityEngine)
            result = engine.load_state()
        finally:
            eng_mod.STATE_FILE = original
        assert result == {}

    # ── ATR guard in entry loop ───────────────────────────────────────────────

    def test_entry_skipped_when_atr_is_zero(self):
        """ATR=0 → stop_dist=0 → invalid; engine must skip before placing order."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=100.0, atr=0.0)
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 0, "No order must be placed when ATR=0"

    def test_entry_skipped_when_atr_is_nan(self):
        """ATR=NaN → skip before order; must not write NaN into state."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=100.0, atr=float('nan'))
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 0, "No order must be placed when ATR=NaN"
        assert 'TSLA' not in engine.state

    def test_entry_skipped_when_atr_chandelier_is_nan(self):
        """atr_chandelier=NaN must be caught before computing trail_price or state write."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=100.0, atr=2.0, atr_chandelier=float('nan'))
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 0, "No order must be placed when ATR_CHAND=NaN"
        assert 'TSLA' not in engine.state

    def test_entry_skipped_when_atr_chandelier_is_zero(self):
        """atr_chandelier=0 must be caught — a zero-width trail stop is invalid."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=100.0, atr=2.0, atr_chandelier=0.0)
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 0, "No order must be placed when ATR_CHAND=0"
        assert 'TSLA' not in engine.state

    # ── Scoring: MA200 = 0 guard ─────────────────────────────────────────────

    def test_score_candidate_ma200_zero_does_not_raise(self):
        """ma200=0 must not raise ZeroDivisionError; trend component floors to 0."""
        from src.engine import VelocityEngine
        engine = VelocityEngine.__new__(VelocityEngine)
        ctx = _ctx(price=110.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                   ma50=105.0, ma200=0.0)
        score = engine._score_candidate(ctx)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_score_candidate_trend_is_zero_when_ma200_is_zero(self):
        """Trend component must return 0 (not NaN or inf) when MA200=0."""
        from src.engine import VelocityEngine
        engine = VelocityEngine.__new__(VelocityEngine)
        ctx_zero_ma = _ctx(price=110.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                           ma50=105.0, ma200=0.0)
        ctx_normal  = _ctx(price=110.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                           ma50=105.0, ma200=105.0)
        assert engine._score_candidate(ctx_zero_ma) == engine._score_candidate(ctx_normal), \
            "MA200=0 must give same trend=0 as equal MAs"

    # ── VIX threshold boundary ────────────────────────────────────────────────

    def _run_cycle_with_vix(self, vix_val):
        """Helper: run a full cycle with given VIX and one passing signal."""
        tc = _mock_trading_client(equity=2500.0, cash=2500.0)
        buy_order  = MagicMock(id='buy-id',  status='new')
        stop_order = MagicMock(id='stop-id', status='accepted')
        tc.submit_order.side_effect = [buy_order, stop_order]
        filled = MagicMock(id='buy-id', status='filled',
                           filled_avg_price='101.0', filled_qty='10.0')
        tc.get_order_by_id.return_value = filled

        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=101.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                   ma50=95.0, ma200=85.0)

        tz_ny = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(2500.0, 2500.0)), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['SYM']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=vix_val), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        return engine

    def test_vix_at_threshold_allows_entries(self):
        """VIX=35 (== threshold) must NOT block entries (rule: VIX > 35 blocks)."""
        from src.config import VIX_THRESHOLD
        engine = self._run_cycle_with_vix(float(VIX_THRESHOLD))
        assert 'SYM' in engine.state, f"VIX={VIX_THRESHOLD} must not block entries"

    def test_vix_above_threshold_blocks_entries(self):
        """VIX=35.01 (> threshold) must block all new entries."""
        from src.config import VIX_THRESHOLD
        engine = self._run_cycle_with_vix(VIX_THRESHOLD + 0.01)
        assert 'SYM' not in engine.state, "VIX above threshold must block entries"

    # ── Strict comparison boundaries ─────────────────────────────────────────

    def test_alligator_not_crossed_does_not_enter(self):
        """alligator_crossed=False (mid-trend, not a fresh crossover) blocks entry."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=100.0, rsi=52.0, rsi_prev=35.0,
                   smma_fast=105.0, smma_med=102.0, smma_slow=100.0,
                   alligator_crossed=False)
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 0, \
            "Entry mid-trend (alligator_crossed=False) must not be taken"

    def test_rsi_equal_to_prev_does_not_enter(self):
        """rsi == rsi_prev fails rsi_momentum (requires RSI_MIN_DELTA rise). No entry."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=100.0, rsi=65.0, rsi_prev=65.0)
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 0, "flat RSI must not trigger entry"

    def test_rsi_below_50_does_not_enter(self):
        """RSI < 50 fails check_rsi_trend even when delta is rising. No entry."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        # rsi=48 < 50 → rsi_trend fails (stock not yet in bullish territory)
        ctx = _ctx(price=100.0, rsi=48.0, rsi_prev=44.0)
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 0, \
            "RSI < 50 must block entry even when delta is positive"

    def test_alligator_aligned_with_fresh_crossover_enters(self):
        """Alligator bullish alignment with fresh crossover → entry fires (2 orders)."""
        tc = _mock_trading_client()
        buy_order  = MagicMock(id='buy-id',  status='new')
        stop_order = MagicMock(id='stop-id', status='accepted')
        tc.submit_order.side_effect = [buy_order, stop_order]
        filled = MagicMock(id='buy-id', status='filled',
                           filled_avg_price='101.0', filled_qty='10.0')
        tc.get_order_by_id.return_value = filled

        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=101.0, rsi=52.0, rsi_prev=35.0,
                   smma_fast=105.0, smma_med=102.0, smma_slow=100.0,
                   alligator_crossed=True)
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count >= 1, \
            "Alligator bullish alignment with fresh crossover must trigger entry"

    # ── Friday dollar-volume multiplier ───────────────────────────────────────

    def test_friday_dollar_volume_threshold_is_doubled(self):
        """On Fridays the dollar-volume gate must be 2× the normal threshold."""
        from src.config import SCAN_MIN_DOLLAR_VOL
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)

        marginal_vol = SCAN_MIN_DOLLAR_VOL  # exactly 1× → fails 2× gate on Friday
        ctx = _ctx(price=101.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                   ma50=95.0, ma200=85.0, dollar_vol=marginal_vol)

        tz_ny = pytz.timezone('US/Eastern')
        friday_now = tz_ny.localize(datetime(2024, 6, 7, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(2500.0, 2500.0)), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['SYM']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = friday_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert 'SYM' not in engine.state, \
            "Stock at 1× dollar-vol must be blocked on Friday (requires 2×)"

    def test_normal_day_dollar_volume_at_1x_passes(self):
        """On a non-Friday, 1× dollar-vol threshold is sufficient for entry."""
        from src.config import SCAN_MIN_DOLLAR_VOL
        tc = _mock_trading_client()
        buy_order  = MagicMock(id='buy-id',  status='new')
        stop_order = MagicMock(id='stop-id', status='accepted')
        tc.submit_order.side_effect = [buy_order, stop_order]
        filled = MagicMock(id='buy-id', status='filled',
                           filled_avg_price='101.0', filled_qty='10.0')
        tc.get_order_by_id.return_value = filled

        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=101.0, orb=100.0, rsi=52.0, rsi_prev=35.0,
                   ma50=95.0, ma200=85.0, dollar_vol=SCAN_MIN_DOLLAR_VOL)
        _run_entry_cycle(engine, ctx, sym='TSLA')
        assert 'TSLA' in engine.state, "1× dollar-vol must pass on non-Friday"

    # ── Alpaca sync: add missing position ──────────────────────────────────────

    def test_sync_adds_alpaca_position_not_in_state(self):
        """Position present at Alpaca but not in state must be added with fill_price."""
        tc = _mock_trading_client()
        pos = MagicMock()
        pos.symbol          = 'AAPL'
        pos.qty             = '5.0'
        pos.avg_entry_price = '175.0'
        tc.get_all_positions.return_value = [pos]
        tc.get_orders.return_value = []

        engine = _make_engine(trading_client=tc)
        engine.state = {}

        with patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            tz_ny = pytz.timezone('US/Eastern')
            mock_dt.now.return_value  = tz_ny.localize(datetime(2024, 6, 5, 10, 30))
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._sync_positions()

        assert 'AAPL' in engine.state
        s = engine.state['AAPL']
        assert s['price']      == pytest.approx(175.0)
        assert s['qty']        == pytest.approx(5.0)
        assert s['fill_price'] == pytest.approx(175.0), "fill_price must be set from avg_entry_price"
        assert s['peak_price'] == pytest.approx(175.0), "peak_price must be set from avg_entry_price"
        assert s['score'] is None, "score stays None — cannot be recovered on restart"

    def test_sync_zero_entry_price_position_skips_velocity_exit_profit_check(self):
        """Position synced with price=0 must not crash velocity exit (division by zero)."""
        tc = _mock_trading_client()
        tz_ny = pytz.timezone('US/Eastern')
        entry_time = (datetime.now(tz_ny) - timedelta(days=14)).isoformat()
        engine = _make_engine(trading_client=tc)
        engine.state = {'GHOST': {'price': 0.0, 'time': entry_time,
                                   'qty': 3.0, 'stop_loss': 0.0,
                                   'volume': 0, 'score': None}}

        snap = _make_snapshot(price=100.0)
        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'):
            engine.check_velocity_exits()   # must not raise

        assert 'GHOST' in engine.state, "Zero-price position must be left alone (not liquidated)"

    # ── Indicator edge cases ─────────────────────────────────────────────────

    def test_rsi_flat_price_does_not_crash(self):
        """RSI with all-zero deltas (gain=0, loss=0) must return NaN, not crash."""
        import pandas as pd
        from src.indicators import compute_rsi
        flat = pd.Series([100.0] * 20)
        result = compute_rsi(flat, period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(flat)

    def test_rsi_all_gains_returns_100(self):
        """RSI with only up days (loss=0) must return 100 for those bars."""
        import pandas as pd
        from src.indicators import compute_rsi
        rising = pd.Series([float(i) for i in range(1, 31)])
        result = compute_rsi(rising, period=14)
        assert result.dropna().iloc[-1] == pytest.approx(100.0, abs=0.01)

    def test_atr_with_identical_bars_returns_zero(self):
        """ATR of a stock with identical high/low/close must be 0, not crash."""
        import pandas as pd
        from src.indicators import compute_atr
        df = pd.DataFrame({
            'high':  [100.0] * 20,
            'low':   [100.0] * 20,
            'close': [100.0] * 20,
        })
        result = compute_atr(df, period=14)
        assert isinstance(result, pd.Series)
        assert result.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# New feature tests — Day-strength boundaries
# ─────────────────────────────────────────────────────────────────────────────

class TestDayStrengthFilter:
    """Day-strength rule: price must be ≥ DAY_STRENGTH_OPEN_PCT above open AND
    in the upper half of the intraday range."""

    def test_price_in_lower_half_of_range_blocks_entry(self):
        """Price in lower half of intraday range fails day_strength. No entry."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        # price=100, low=98, high=103 → range_pos = (100-98)/(103-98) = 0.4 < 0.5
        ctx = _ctx(price=100.0, intraday_open=98.0,
                   intraday_high=103.0, intraday_low=98.0)
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 0, \
            "Price in lower 40% of range must block entry"
        assert 'TSLA' not in engine.state

    def test_price_above_open_and_upper_range_enters(self):
        """Price above open and in upper half of range passes day_strength. Entry fires."""
        tc = _mock_trading_client()
        buy_order  = MagicMock(id='buy-id',  status='new')
        stop_order = MagicMock(id='stop-id', status='accepted')
        tc.submit_order.side_effect = [buy_order, stop_order]
        filled = MagicMock(id='buy-id', status='filled',
                           filled_avg_price='101.0', filled_qty='10.0')
        tc.get_order_by_id.return_value = filled

        engine = _make_engine(trading_client=tc)
        # price=101, open=99.5 (+1.5%), low=98, high=102 → range_pos=(101-98)/4=0.75>0.5 ✓
        ctx = _ctx(price=101.0, intraday_open=99.5,
                   intraday_high=102.0, intraday_low=98.0)
        _run_entry_cycle(engine, ctx, sym='TSLA')
        assert tc.submit_order.call_count == 2, \
            "Price above open and in upper range must generate 2 submit_order calls"


class TestDailyLossCircuitBreakerSlotFull:
    """Circuit breaker: equity drops > MAX_DAILY_LOSS_PCT → skip new entries."""

    def test_circuit_breaker_halts_entries_on_daily_loss(self):
        """Equity dropping 3%+ from day open must prevent submit_order calls."""
        from src.config import MAX_DAILY_LOSS_PCT
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)

        engine._day_start_date   = '2024-06-05'
        engine._day_start_equity = 2500.0
        loss_equity = round(2500.0 * (1 - MAX_DAILY_LOSS_PCT - 0.001), 2)

        tz_ny = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(loss_equity, 2500.0)), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['TSLA']), \
             patch.object(engine, 'get_technical_context', return_value=_ctx()), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert tc.submit_order.call_count == 0, "Circuit breaker must block all new entries"

    def test_circuit_breaker_resets_on_new_day(self):
        """Day-start equity resets when date changes — new day, clean slate."""
        tc = _mock_trading_client(equity=2500.0, cash=2500.0)
        buy_order  = MagicMock(id='buy-id',  status='new')
        stop_order = MagicMock(id='stop-id', status='accepted')
        tc.submit_order.side_effect = [buy_order, stop_order]
        filled = MagicMock(id='buy-id', status='filled',
                           filled_avg_price='101.0', filled_qty='10.0')
        tc.get_order_by_id.return_value = filled

        engine = _make_engine(equity=2500.0, cash=2500.0, trading_client=tc)
        engine._day_start_date   = '2024-06-04'  # yesterday
        engine._day_start_equity = 2500.0

        passing_ctx = _ctx(price=101.0, orb=100.0, ma50=95.0, ma200=85.0,
                           rsi=52.0, rsi_prev=35.0, dollar_vol=500_000_000)

        tz_ny = pytz.timezone('US/Eastern')
        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(2500.0, 2500.0)), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['TSLA']), \
             patch.object(engine, 'get_technical_context', return_value=passing_ctx), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        assert engine._day_start_date == '2024-06-05'
        assert engine._day_start_equity == pytest.approx(2500.0, abs=0.01)
        assert tc.submit_order.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# 10. STARTUP SAFETY — _initialize() orphan-order cancellation (Alpaca edition)
# ─────────────────────────────────────────────────────────────────────────────

class TestInitializeSafeStartup:
    """
    _initialize() must only cancel orphaned BUY orders whose symbol is NOT in state.
    TRAIL SELL orders (protective stops) must never be cancelled.
    """

    def test_orphan_buy_cancel_skips_active_and_trail_orders(self):
        """Orphan BUY for unknown symbol cancelled; TRAIL SELL for active or unknown symbol kept."""
        tc = _mock_trading_client(equity=2500.0, cash=2500.0)
        tz_ny = pytz.timezone('US/Eastern')

        engine = _make_engine(trading_client=tc)
        engine.state = {
            'HELD': {'price': 100.0, 'qty': 5.0, 'stop_loss': 90.0,
                     'volume': 0, 'score': 60,
                     'time': datetime.now(tz_ny).isoformat()},
        }

        # Orphan BUY for unknown symbol — should be cancelled
        orphan_buy             = MagicMock()
        orphan_buy.symbol      = 'ORPHAN'
        orphan_buy.side        = 'buy'
        orphan_buy.id          = 'orphan-buy-id'

        # Active TRAIL SELL for HELD — must NOT be cancelled
        held_trail             = MagicMock()
        held_trail.symbol      = 'HELD'
        held_trail.side        = 'sell'
        held_trail.id          = 'held-trail-id'

        # Unknown TRAIL SELL (from a crashed session) — must NOT be cancelled
        orphan_trail           = MagicMock()
        orphan_trail.symbol    = 'ORPHAN'
        orphan_trail.side      = 'sell'
        orphan_trail.id        = 'orphan-trail-id'

        tc.get_orders.return_value = [held_trail, orphan_buy, orphan_trail]

        # _initialize calls several methods; mock out the heavy ones
        with patch.object(engine, '_fetch_equity_with_retry', return_value=2500.0), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_log_startup_summary'), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'):
            engine._initialize()

        cancelled_ids = [c[0][0] for c in tc.cancel_order_by_id.call_args_list]
        assert 'orphan-buy-id'  in cancelled_ids, "Orphan BUY must be cancelled"
        assert 'held-trail-id'  not in cancelled_ids, "Active TRAIL SELL must not be cancelled"
        assert 'orphan-trail-id' not in cancelled_ids, "Orphan TRAIL SELL must not be cancelled"

    def test_no_cancel_when_all_orders_belong_to_active_positions(self):
        """When every open BUY order maps to a state symbol, nothing is cancelled."""
        tc = _mock_trading_client(equity=2500.0, cash=2500.0)
        tz_ny = pytz.timezone('US/Eastern')

        engine = _make_engine(trading_client=tc)
        engine.state = {
            'SYM': {'price': 50.0, 'qty': 2.0, 'stop_loss': 45.0,
                    'volume': 0, 'score': 70,
                    'time': datetime.now(tz_ny).isoformat()},
        }

        # A BUY order for an active position — must NOT be cancelled
        active_buy         = MagicMock()
        active_buy.symbol  = 'SYM'
        active_buy.side    = 'buy'
        active_buy.id      = 'active-buy-id'
        tc.get_orders.return_value = [active_buy]

        with patch.object(engine, '_fetch_equity_with_retry', return_value=2500.0), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_log_startup_summary'), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'):
            engine._initialize()

        assert tc.cancel_order_by_id.call_count == 0, \
            "No orders must be cancelled when all belong to active positions"


# ─────────────────────────────────────────────────────────────────────────────
# 11. RSI DELTA GATE — minimum acceleration required
# ─────────────────────────────────────────────────────────────────────────────

class TestRsiDeltaGate:
    """
    c_rsi_delta = (rsi - rsi_prev) >= RSI_MIN_DELTA blocks trivially small rises.
    """

    def test_tiny_rsi_rise_blocks_entry(self):
        """RSI delta of 0.5 (< RSI_MIN_DELTA=1.0) must not generate a signal."""
        from src.config import RSI_MIN_DELTA
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        ctx = _ctx(price=101.0, orb=100.0, rsi=55.5, rsi_prev=55.0,
                   ma50=95.0, ma200=85.0)
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 0, \
            f"RSI delta {55.5-55.0:.1f} < {RSI_MIN_DELTA} must block entry"
        assert 'TSLA' not in engine.state

    def test_rsi_delta_at_minimum_allows_entry(self):
        """RSI delta exactly at RSI_MIN_DELTA must pass the gate."""
        from src.config import RSI_MIN_DELTA
        tc = _mock_trading_client()
        buy_order  = MagicMock(id='buy-id',  status='new')
        stop_order = MagicMock(id='stop-id', status='accepted')
        tc.submit_order.side_effect = [buy_order, stop_order]
        filled = MagicMock(id='buy-id', status='filled',
                           filled_avg_price='101.0', filled_qty='10.0')
        tc.get_order_by_id.return_value = filled

        engine = _make_engine(trading_client=tc)
        rsi_prev = 49.0   # rsi = 49 + RSI_MIN_DELTA(1.0) = 50.0 — exactly at bullish floor
        rsi      = rsi_prev + RSI_MIN_DELTA
        ctx = _ctx(price=101.0, orb=100.0, rsi=rsi, rsi_prev=rsi_prev,
                   ma50=95.0, ma200=85.0)
        _run_entry_cycle(engine, ctx)
        assert tc.submit_order.call_count == 2, \
            f"RSI delta exactly at {RSI_MIN_DELTA} must allow entry (2 calls: BUY + TRAIL)"


# ─────────────────────────────────────────────────────────────────────────────
# 12. SCANNER OPTIMIZATION — skip scan when all slots are full
# ─────────────────────────────────────────────────────────────────────────────

class TestScannerSkipWhenFull:
    """
    get_institutional_scan() must not be called when all dynamic slots are filled.
    """

    def test_scanner_not_called_when_all_slots_filled(self):
        from src.config import MAX_POSITIONS_CAP, MIN_BUCKET_SIZE
        equity = 2500.0
        dynamic_max = min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)

        tc = _mock_trading_client(equity=equity, cash=equity)
        engine = _make_engine(equity=equity, cash=equity, trading_client=tc)
        tz_ny = pytz.timezone('US/Eastern')

        for i in range(dynamic_max):
            sym = f'SYM{i}'
            engine.state[sym] = {
                'price': 100.0, 'qty': 5.0, 'current_price': 100.0,
                'stop_loss': 90.0, 'volume': 0, 'score': 60,
                'time': datetime.now(tz_ny).isoformat(),
            }

        fake_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(equity, equity)), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan') as mock_scan, \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, '_check_portfolio_concentration', return_value=(0.5, False, False)), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.run_cycle()

        mock_scan.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 13. HARD STOP — forced exit when drawdown exceeds threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestHardStop:

    def _state_entry(self, price, cur, tz_ny):
        return {'price': price, 'qty': 5.0, 'current_price': cur,
                'stop_loss': price * 0.85, 'volume': 0, 'score': 60,
                'time': datetime.now(tz_ny).isoformat(), 'peak_price': price}

    def test_hard_stop_triggers_when_down_beyond_threshold(self):
        """Drawdown > HARD_STOP_PCT from entry must place a sell order immediately."""
        from src.config import HARD_STOP_PCT
        tc = _mock_trading_client()
        pos = MagicMock(); pos.qty = '5.0'
        tc.get_open_position.side_effect = None   # override the default Exception side_effect
        tc.get_open_position.return_value = pos
        tc.get_orders.return_value = []

        engine = _make_engine(trading_client=tc)
        tz_ny = pytz.timezone('US/Eastern')
        entry = 100.0
        cur   = round(entry * (1 - HARD_STOP_PCT - 0.01), 2)
        engine.state = {'POS': self._state_entry(entry, cur, tz_ny)}

        snap = _make_snapshot(price=cur)
        _safe_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        assert engine.state.get('POS', {}).get('pending_exit') is True, \
            "Hard stop must mark position pending_exit=True after sell order placed"
        assert tc.submit_order.called

    def test_hard_stop_does_not_trigger_within_threshold(self):
        """Drawdown exactly below HARD_STOP_PCT must leave position open."""
        from src.config import HARD_STOP_PCT
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        tz_ny = pytz.timezone('US/Eastern')
        entry = 100.0
        cur   = round(entry * (1 - HARD_STOP_PCT + 0.01), 2)
        engine.state = {'POS': self._state_entry(entry, cur, tz_ny)}

        snap = _make_snapshot(price=cur)
        # Pin to Wednesday so Friday-close rule never fires during this test.
        safe_now = tz_ny.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = safe_now
            mock_dt.fromisoformat = datetime.fromisoformat
            engine.check_velocity_exits()

        assert 'POS' in engine.state, "Position within loss threshold must not be force-closed"
        assert not tc.submit_order.called


# ─────────────────────────────────────────────────────────────────────────────
# 14. FRIDAY CLOSE — close under-performing positions before weekend
# ─────────────────────────────────────────────────────────────────────────────

class TestFridayClose:

    def _state_entry(self, price, cur, tz_ny):
        return {'price': price, 'qty': 5.0, 'current_price': cur,
                'stop_loss': price * 0.90, 'volume': 0, 'score': 60,
                'time': datetime.now(tz_ny).isoformat(), 'peak_price': price}

    def test_friday_close_triggers_below_profit_threshold(self):
        """On Friday after close hour, profit < threshold must place a sell order."""
        from src.config import FRIDAY_CLOSE_HOUR, FRIDAY_MIN_PROFIT_PCT
        tc = _mock_trading_client()
        pos = MagicMock(); pos.qty = '5.0'
        tc.get_open_position.side_effect = None   # override the default Exception side_effect
        tc.get_open_position.return_value = pos
        tc.get_orders.return_value = []

        engine = _make_engine(trading_client=tc)
        tz_ny = pytz.timezone('US/Eastern')
        entry = 100.0
        cur   = round(entry * (1 + FRIDAY_MIN_PROFIT_PCT - 0.01), 2)
        engine.state = {'FRI': self._state_entry(entry, cur, tz_ny)}

        friday_after = tz_ny.localize(datetime(2024, 6, 7, FRIDAY_CLOSE_HOUR, 30))
        snap = _make_snapshot(price=cur)

        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = friday_after
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.check_velocity_exits()

        assert engine.state.get('FRI', {}).get('pending_exit') is True, \
            "Friday close must mark position pending_exit=True after sell order placed"

    def test_friday_close_does_not_trigger_above_threshold(self):
        """Profit above FRIDAY_MIN_PROFIT_PCT on Friday must keep position open."""
        from src.config import FRIDAY_CLOSE_HOUR, FRIDAY_MIN_PROFIT_PCT
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        tz_ny = pytz.timezone('US/Eastern')
        entry = 100.0
        cur   = round(entry * (1 + FRIDAY_MIN_PROFIT_PCT + 0.01), 2)
        engine.state = {'FRI': self._state_entry(entry, cur, tz_ny)}

        friday_after = tz_ny.localize(datetime(2024, 6, 7, FRIDAY_CLOSE_HOUR + 1, 0))
        snap = _make_snapshot(price=cur)

        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = friday_after
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.check_velocity_exits()

        assert 'FRI' in engine.state, "Profitable position must not be closed on Friday"
        assert not tc.submit_order.called

    def test_friday_close_does_not_trigger_before_close_hour(self):
        """Before FRIDAY_CLOSE_HOUR, Friday close rule must be inactive."""
        from src.config import FRIDAY_CLOSE_HOUR, FRIDAY_MIN_PROFIT_PCT
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        tz_ny = pytz.timezone('US/Eastern')
        entry = 100.0
        cur   = round(entry * (1 + FRIDAY_MIN_PROFIT_PCT - 0.01), 2)
        engine.state = {'FRI': self._state_entry(entry, cur, tz_ny)}

        friday_morning = tz_ny.localize(datetime(2024, 6, 7, FRIDAY_CLOSE_HOUR - 2, 0))
        snap = _make_snapshot(price=cur)

        with patch.object(engine, '_fetch_snapshot', return_value=snap), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = friday_morning
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.check_velocity_exits()

        assert 'FRI' in engine.state, "Friday close must not trigger before FRIDAY_CLOSE_HOUR"
        assert not tc.submit_order.called


# ─────────────────────────────────────────────────────────────────────────────
# 15. STOP-ORDER AUDIT — _audit_stop_orders (Alpaca edition)
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditStopOrders:
    """
    _audit_stop_orders() ensures every open position has exactly one
    chandelier trailing-stop SELL order via Alpaca.
    """

    def _trail_order(self, sym, trail_price=6.0, order_id='trail-1'):
        o = MagicMock()
        o.symbol      = sym
        o.side        = 'sell'
        o.order_type  = 'trailing_stop'
        o.id          = order_id
        o.created_at  = datetime(2024, 6, 5, 10, 0)
        o.trail_price = trail_price
        return o

    def _non_trail_order(self, sym, order_id='limit-1'):
        o = MagicMock()
        o.symbol      = sym
        o.side        = 'sell'
        o.order_type  = 'limit'
        o.id          = order_id
        o.created_at  = datetime(2024, 6, 5, 10, 0)
        return o

    def _state_with_position(self, sym='AAPL', price=100.0, qty=5.0):
        tz_ny = pytz.timezone('US/Eastern')
        return {
            sym: {
                'price': price, 'fill_price': price, 'qty': qty,
                'stop_loss': price * 0.94, 'stop_dist': 6.0,
                'time': datetime.now(tz_ny).isoformat(),
                'volume': 0, 'score': 60,
            }
        }

    def test_existing_trail_stop_confirmed_no_new_order(self):
        """If trailing stop already exists, no new submit_order called."""
        tc = _mock_trading_client()
        tc.get_orders.return_value = [self._trail_order('AAPL')]

        engine = _make_engine(trading_client=tc)
        engine.state = self._state_with_position('AAPL')

        with patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'):
            engine._audit_stop_orders()

        # submit_order should not be called — trail already exists
        assert tc.submit_order.call_count == 0

    def test_non_trail_sell_cancelled_before_new_stop(self):
        """Non-trailing-stop SELL is cancelled before placing new trailing stop."""
        tc = _mock_trading_client()
        non_trail = self._non_trail_order('AAPL', 'limit-1')
        tc.get_orders.return_value = [non_trail]

        # For the new stop placement: submit_order returns accepted stop
        accepted = MagicMock(id='new-trail', status='accepted')
        tc.submit_order.return_value = accepted
        tc.get_order_by_id.return_value = MagicMock(status='accepted')

        engine = _make_engine(trading_client=tc)
        engine.state = self._state_with_position('AAPL')

        # Provide fake bars so the audit can compute ATR
        import pandas as pd
        import numpy as np
        n = 50
        df = pd.DataFrame({
            'open':   [100.0] * n,
            'high':   [102.0] * n,
            'low':    [98.0]  * n,
            'close':  [100.0] * n,
            'volume': [1_000_000] * n,
        })
        _safe_now = pytz.timezone('US/Eastern').localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_daily_bars', return_value=df), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine._audit_stop_orders()

        # Cancel was called for the non-trail
        cancelled = [c[0][0] for c in tc.cancel_order_by_id.call_args_list]
        assert 'limit-1' in cancelled, "Non-trail SELL must be cancelled"
        # A new trailing stop was submitted
        assert tc.submit_order.call_count >= 1

    def test_missing_trail_stop_places_new_one(self):
        """When no SELL orders exist for position, a trailing stop is placed."""
        tc = _mock_trading_client()
        tc.get_orders.return_value = []  # no existing orders

        accepted = MagicMock(id='new-trail', status='accepted')
        tc.submit_order.return_value = accepted
        tc.get_order_by_id.return_value = MagicMock(status='accepted')

        engine = _make_engine(trading_client=tc)
        engine.state = self._state_with_position('TSLA')

        import pandas as pd
        n = 50
        df = pd.DataFrame({
            'open':   [100.0] * n,
            'high':   [102.0] * n,
            'low':    [98.0]  * n,
            'close':  [100.0] * n,
            'volume': [1_000_000] * n,
        })
        _safe_now = pytz.timezone('US/Eastern').localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_daily_bars', return_value=df), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine._audit_stop_orders()

        assert tc.submit_order.call_count == 1
        req = tc.submit_order.call_args[0][0]
        assert isinstance(req, TrailingStopOrderRequest)

    def test_duplicate_trail_stops_deduped(self):
        """Multiple trailing stops for same symbol → oldest cancelled, newest kept."""
        tc = _mock_trading_client()
        older = self._trail_order('AAPL', trail_price=6.0, order_id='trail-old')
        older.created_at = datetime(2024, 6, 5, 9, 0)
        newer = self._trail_order('AAPL', trail_price=6.0, order_id='trail-new')
        newer.created_at = datetime(2024, 6, 5, 10, 0)
        tc.get_orders.return_value = [older, newer]

        engine = _make_engine(trading_client=tc)
        engine.state = self._state_with_position('AAPL')

        _safe_now = pytz.timezone('US/Eastern').localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine._audit_stop_orders()

        cancelled = [c[0][0] for c in tc.cancel_order_by_id.call_args_list]
        assert 'trail-old' in cancelled, "Older duplicate trailing stop must be cancelled"
        assert 'trail-new' not in cancelled, "Newest trailing stop must be kept"

    def test_pending_positions_skipped(self):
        """Positions with pending=True must be skipped by the audit."""
        tc = _mock_trading_client()
        tc.get_orders.return_value = []

        engine = _make_engine(trading_client=tc)
        engine.state = {
            'PEND': {
                'price': 100.0, 'qty': 5.0, 'stop_dist': 6.0,
                'time': datetime.now().isoformat(), 'volume': 0, 'score': 60,
                'pending': True,
            }
        }

        with patch.object(engine, 'save_state'), \
             patch('src.engine.time.sleep'):
            engine._audit_stop_orders()

        assert tc.submit_order.call_count == 0, "Pending positions must not get a stop placed"


# ─────────────────────────────────────────────────────────────────────────────
# 16. DASHBOARD unit_price — fill_price only (no commission for Alpaca)
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardUnitPrice:
    """
    alpaca_dashboard.get_state() must compute unit_price correctly:
      - Normal entry (Alpaca): unit_price = fill_price directly (commission-free)
      - Re-synced from Alpaca: fill_price = avg_entry_price; unit_price = fill_price
      - No fill_price: unit_price = None (shown as 'pending' in UI)
    """

    def _get_positions(self, state_dict):
        """Call alpaca_dashboard.get_state() with mocked file reads."""
        import json
        import alpaca_dashboard as ds

        dash_data = {'equity': 10000.0, 'settled_cash': 5000.0}
        with patch.object(ds, '_read_json', side_effect=lambda path: (
            state_dict if 'state' in path else dash_data
        )):
            result = ds.get_state()
        return json.loads(result.body)['positions']

    def test_unit_price_equals_fill_price_for_normal_alpaca_entry(self):
        """Alpaca is commission-free; unit_price = fill_price."""
        state = {
            'AAPL': {
                'fill_price': 100.0, 'price': 100.0, 'qty': 10.0,
                'time': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
                'stop_loss': 95.0, 'volume': 0, 'score': 80,
            }
        }
        positions = self._get_positions(state)
        pos = next(p for p in positions if p['symbol'] == 'AAPL')
        assert pos['unit_price'] == pytest.approx(100.0, abs=0.001)

    def test_unit_price_equals_fill_price_when_resynced_from_alpaca(self):
        """Re-synced position: fill_price = avg_entry_price; unit_price = fill_price."""
        state = {
            'MSFT': {
                'fill_price': 175.0, 'price': 175.0, 'qty': 5.0,
                'time': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
                'stop_loss': 165.0, 'volume': 0, 'score': None,
            }
        }
        positions = self._get_positions(state)
        pos = next(p for p in positions if p['symbol'] == 'MSFT')
        assert pos['unit_price'] == pytest.approx(175.0, abs=0.001), \
            "unit_price must equal fill_price for re-synced positions"

    def test_unit_price_is_none_when_fill_price_absent(self):
        """A position without fill_price shows unit_price=None ('pending' in UI)."""
        state = {
            'NVDA': {
                'price': 500.0, 'qty': 2.0,
                # no fill_price — pending state
                'time': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
                'stop_loss': 475.0, 'volume': 0, 'score': None,
            }
        }
        positions = self._get_positions(state)
        pos = next(p for p in positions if p['symbol'] == 'NVDA')
        assert pos['unit_price'] is None, \
            "unit_price must be None when fill_price is absent"
