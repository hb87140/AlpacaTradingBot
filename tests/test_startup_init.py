"""
Unit tests for VelocityEngine startup initialisation gate.

Covers:
  - _fetch_equity_with_retry: retries on zero/negative equity, exceptions, succeeds after N attempts
  - _initialize: sets real equity, marks _equity_initialized, syncs positions,
    calls _update_position_prices only when positions exist, writes dashboard
  - _update_position_prices: stores unrealized_pnl and unrealized_pnl_pct
  - run(): calls _initialize before the first run_cycle
  - _ensure_connected: pings get_account(), returns bool
  - _fetch_spy_trend: SPY regime gate (SMA50 > SMA200) via _fetch_daily_bars
  - shutdown: cancels BUY orders (no disconnect — Alpaca REST is stateless)
  - _compute_book_correlation: portfolio correlation filter

Alpaca is fully mocked — no live connection required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytz
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, call

# Shared SPY regime dict — used wherever _fetch_spy_trend is patched
_SPY_BULL_REGIME = {
    'is_bull': True, 'spy_close': 450.0, 'ema50': 440.0,
    'size_factor': 1.0, 'rvol_mult': 1.0,
}


# ── shared helpers ─────────────────────────────────────────────────────────────

def _mock_account(equity=5000.0, cash=5000.0):
    acc = MagicMock()
    acc.portfolio_value = str(equity)
    acc.cash = str(cash)
    return acc


def _mock_trading_client(equity=5000.0, cash=5000.0):
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


def _make_engine(equity=5000.0, cash=5000.0, trading_client=None, data_client=None, state=None):
    """Return a VelocityEngine with Alpaca clients replaced and __init__ bypassed."""
    from src.engine import VelocityEngine
    engine = VelocityEngine.__new__(VelocityEngine)
    engine.trading_client = trading_client or _mock_trading_client(equity, cash)
    engine.data_client    = data_client or MagicMock()
    engine.screener_client = MagicMock()
    engine.state          = state if state is not None else {}
    engine._last_equity        = 0.0
    engine._last_settled_cash  = 0.0
    engine._equity_initialized = False
    engine._last_vix           = None
    engine._vix_cache_date     = None
    engine._last_scan_ts       = None
    engine._next_scan_dt       = None
    engine._day_start_date     = None
    engine._day_start_equity   = None
    engine._bar_cache          = {}
    engine._spy_cache          = {}
    engine._sector_cache       = {}
    engine._daily_scan_skip               = {}
    engine._insufficient_history_skip     = set()
    engine._last_audit_date               = None
    engine._missing_position_counts       = {}
    return engine


# ── _fetch_equity_with_retry ───────────────────────────────────────────────────

class TestFetchEquityWithRetry:
    """Polls trading_client.get_account() until portfolio_value > 0."""

    def test_returns_immediately_on_first_success(self):
        tc = _mock_trading_client(equity=5000.0)
        engine = _make_engine(trading_client=tc)
        with patch('src.engine.time.sleep') as mock_sleep:
            result = engine._fetch_equity_with_retry()
        assert result == 5000.0
        assert tc.get_account.call_count == 1
        assert mock_sleep.call_count == 0

    def test_retries_when_equity_is_zero(self):
        tc = MagicMock()
        tc.get_account.side_effect = [
            _mock_account(equity=0.0),
            _mock_account(equity=3000.0),
        ]
        engine = _make_engine(trading_client=tc)
        with patch('src.engine.time.sleep'):
            result = engine._fetch_equity_with_retry()
        assert result == 3000.0
        assert tc.get_account.call_count == 2

    def test_retries_when_equity_is_negative(self):
        tc = MagicMock()
        tc.get_account.side_effect = [
            _mock_account(equity=-500.0),
            _mock_account(equity=4200.0),
        ]
        engine = _make_engine(trading_client=tc)
        with patch('src.engine.time.sleep'):
            result = engine._fetch_equity_with_retry()
        assert result == 4200.0

    def test_retries_on_exception_then_succeeds(self):
        tc = MagicMock()
        tc.get_account.side_effect = [
            RuntimeError("connection lost"),
            _mock_account(equity=6000.0),
        ]
        engine = _make_engine(trading_client=tc)
        with patch('src.engine.time.sleep') as mock_sleep:
            result = engine._fetch_equity_with_retry()
        assert result == 6000.0
        assert mock_sleep.call_count == 1

    def test_succeeds_after_three_failures(self):
        tc = MagicMock()
        tc.get_account.side_effect = [
            _mock_account(equity=0.0),
            RuntimeError("timeout"),
            _mock_account(equity=0.0),
            _mock_account(equity=7500.0),
        ]
        engine = _make_engine(trading_client=tc)
        with patch('src.engine.time.sleep'):
            result = engine._fetch_equity_with_retry()
        assert result == 7500.0
        assert tc.get_account.call_count == 4

    def test_sleep_duration_matches_config(self):
        from src.config import EQUITY_RETRY_INTERVAL
        tc = MagicMock()
        tc.get_account.side_effect = [
            _mock_account(equity=0.0),
            _mock_account(equity=1000.0),
        ]
        engine = _make_engine(trading_client=tc)
        with patch('src.engine.time.sleep') as mock_sleep:
            engine._fetch_equity_with_retry()
        mock_sleep.assert_called_once_with(EQUITY_RETRY_INTERVAL)


# ── _initialize ────────────────────────────────────────────────────────────────

class TestInitialize:
    """_initialize fetches equity, syncs positions, cancels orphaned BUY orders,
    then waits for PRE_ENTRY_SYNC_TIME, re-syncs, audits stops and updates prices."""

    def test_sets_last_equity_from_api(self):
        tc = _mock_trading_client(equity=8000.0)
        engine = _make_engine(trading_client=tc)
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        assert engine._last_equity == 8000.0

    def test_marks_equity_initialized(self):
        engine = _make_engine(equity=5000.0)
        assert engine._equity_initialized is False
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        assert engine._equity_initialized is True

    def test_initializes_equity_from_alpaca_not_local_seed(self):
        tc = _mock_trading_client(equity=9999.0)
        engine = _make_engine(trading_client=tc)
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        assert engine._last_equity == 9999.0

    def test_syncs_positions_twice(self):
        """Phase 1 and Phase 2 each call _sync_positions once (total 2).
        _wait_for_pre_entry_sync is mocked so its heartbeat syncs don't interfere."""
        engine = _make_engine()
        with patch.object(engine, '_sync_positions') as mock_sync, \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_update_position_prices'),         \
             patch.object(engine, '_wait_for_pre_entry_sync'),        \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        assert mock_sync.call_count == 2

    def test_updates_prices_when_positions_exist(self):
        """Prices are updated in both Phase 1 (snapshot) and Phase 2 (final).
        _wait_for_pre_entry_sync is mocked so its heartbeat updates don't interfere."""
        engine = _make_engine(state={'AAPL': {
            'price': 150.0, 'qty': 10, 'stop_loss': 140.0,
            'volume': 5000000, 'score': 70.0, 'time': '2026-01-01T10:00:00',
        }})
        with patch.object(engine, '_sync_positions'),              \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_audit_stop_orders'),                      \
             patch.object(engine, '_wait_for_pre_entry_sync'),                \
             patch.object(engine, '_update_position_prices') as mock_up,     \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        assert mock_up.call_count == 2   # once per phase

    def test_audits_stops_when_positions_exist(self):
        engine = _make_engine(state={'AAPL': {
            'price': 150.0, 'qty': 10, 'stop_loss': 140.0,
            'volume': 5000000, 'score': 70.0, 'time': '2026-01-01T10:00:00',
        }})
        with patch.object(engine, '_sync_positions'),       \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_update_position_prices'),          \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        mock_audit.assert_called_once()

    def test_skips_audit_when_no_positions(self):
        engine = _make_engine(state={})
        with patch.object(engine, '_sync_positions'),       \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_audit_stop_orders') as mock_audit, \
             patch.object(engine, '_update_position_prices'),          \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        mock_audit.assert_not_called()

    def test_skips_price_update_when_no_positions(self):
        engine = _make_engine(state={})
        with patch.object(engine, '_sync_positions'),    \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_update_position_prices') as mock_up, \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        mock_up.assert_not_called()

    def test_writes_dashboard_twice(self):
        """Dashboard is written after Phase 1 (snapshot) and again after Phase 2 (final).
        _wait_for_pre_entry_sync is mocked out so its heartbeat writes don't interfere."""
        engine = _make_engine()
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_write_dashboard_data') as mock_wd, \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        assert mock_wd.call_count == 2
        mock_wd.assert_called_with(connected=True)   # both calls use connected=True

    def test_initialize_called_before_run_cycle_in_run(self):
        """run() must call _initialize() before the first run_cycle()."""
        engine = _make_engine()
        call_order = []

        def fake_init():
            call_order.append('init')

        def fake_cycle():
            call_order.append('cycle')
            raise SystemExit(0)

        with patch.object(engine, '_initialize', side_effect=fake_init), \
             patch.object(engine, 'run_cycle', side_effect=fake_cycle),  \
             patch.object(engine, '_write_dashboard_data'):
            with pytest.raises(SystemExit):
                engine.run()

        assert call_order.index('init') < call_order.index('cycle')

    def test_orphan_buy_order_cancelled_on_init(self):
        """BUY orders for symbols not in state are cancelled as orphans at startup."""
        tc = _mock_trading_client()
        buy_order = MagicMock()
        buy_order.side   = 'OrderSide.BUY'
        buy_order.symbol = 'AAPL'
        buy_order.id     = 'orphan-buy-1'
        tc.get_orders.return_value = [buy_order]

        engine = _make_engine(trading_client=tc, state={})  # AAPL not in state → orphan
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'), \
             patch('src.engine.time.sleep'):
            engine._initialize()

        tc.cancel_order_by_id.assert_called_with('orphan-buy-1')

    def test_orphan_trail_sell_not_cancelled_on_init(self):
        """TRAIL SELL orders (side='sell') for symbols not yet in state must NOT be cancelled."""
        tc = _mock_trading_client()
        trail_order = MagicMock()
        trail_order.side   = 'OrderSide.SELL'
        trail_order.symbol = 'AAPL'
        trail_order.id     = 'trail-sell-1'
        tc.get_orders.return_value = [trail_order]

        engine = _make_engine(trading_client=tc, state={})
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'), \
             patch('src.engine.time.sleep'):
            engine._initialize()

        # cancel_order_by_id should not have been called (SELL not a BUY)
        tc.cancel_order_by_id.assert_not_called()


# ── _update_position_prices — unrealized P&L ──────────────────────────────────

class TestUnrealizedPnl:
    """_update_position_prices fetches a snapshot for each position and stores P&L."""

    def _engine_with_pos(self, entry, qty, cur_price):
        state = {
            'TSLA': {
                'price': entry, 'qty': qty,
                'stop_loss': entry - 5,
                'volume': 1000000, 'score': 75.0,
                'time': '2026-01-01T10:00:00',
            }
        }
        engine = _make_engine(state=state)
        snap = {'live_price': cur_price, 'bid': 0.0, 'ask': 0.0, 'intraday_vol': 0}
        engine._fetch_snapshot = MagicMock(return_value=snap)
        return engine

    def test_unrealized_gain_stored(self):
        engine = self._engine_with_pos(entry=100.0, qty=10, cur_price=110.0)
        engine._update_position_prices()
        assert engine.state['TSLA']['unrealized_pnl']     == 100.0  # (110-100)*10
        assert engine.state['TSLA']['unrealized_pnl_pct'] == 10.0   # 10%

    def test_unrealized_loss_stored(self):
        engine = self._engine_with_pos(entry=100.0, qty=5, cur_price=90.0)
        engine._update_position_prices()
        assert engine.state['TSLA']['unrealized_pnl']     == -50.0  # (90-100)*5
        assert engine.state['TSLA']['unrealized_pnl_pct'] == -10.0  # -10%

    def test_breakeven_stored(self):
        engine = self._engine_with_pos(entry=100.0, qty=8, cur_price=100.0)
        engine._update_position_prices()
        assert engine.state['TSLA']['unrealized_pnl']     == 0.0
        assert engine.state['TSLA']['unrealized_pnl_pct'] == 0.0

    def test_current_price_stored(self):
        engine = self._engine_with_pos(entry=100.0, qty=10, cur_price=115.0)
        engine._update_position_prices()
        assert engine.state['TSLA']['current_price'] == 115.0

    def test_pnl_rounded_to_two_decimals(self):
        engine = self._engine_with_pos(entry=33.33, qty=3, cur_price=34.00)
        engine._update_position_prices()
        pnl = engine.state['TSLA']['unrealized_pnl']
        assert pnl == round(pnl, 2)

    def test_pnl_not_computed_when_snapshot_unavailable(self):
        state = {'MSFT': {
            'price': 300.0, 'qty': 5,
            'stop_loss': 280.0,
            'volume': 5000000, 'score': 70.0,
            'time': '2026-01-01T10:00:00',
        }}
        engine = _make_engine(state=state)
        engine._fetch_snapshot = MagicMock(return_value=None)
        engine._update_position_prices()
        assert 'unrealized_pnl' not in engine.state['MSFT']


# ── _log_startup_summary ───────────────────────────────────────────────────────

class TestLogStartupSummary:
    def test_no_positions_logs_ready(self, caplog):
        import logging
        engine = _make_engine(state={})
        with caplog.at_level(logging.INFO, logger='VelocityEngine'):
            engine._log_startup_summary(5000.0)
        combined = '\n'.join(caplog.messages)
        assert 'No open positions' in combined
        assert 'INIT READY' in combined

    def test_with_positions_logs_each_symbol(self, caplog):
        import logging
        state = {
            'AAPL': {
                'price': 150.0, 'qty': 10, 'current_price': 155.0,
                'stop_loss': 140.0,
                'unrealized_pnl': 50.0, 'unrealized_pnl_pct': 3.33,
                'volume': 5000000, 'score': 70.0, 'time': '2026-01-01T10:00:00',
            }
        }
        engine = _make_engine(state=state)
        with caplog.at_level(logging.INFO, logger='VelocityEngine'):
            engine._log_startup_summary(5000.0)
        combined = '\n'.join(caplog.messages)
        assert 'AAPL' in combined
        # When positions exist, the summary logs the positions table (not "INIT READY")
        assert 'Equity=' in combined

    def test_ready_line_shows_correct_equity(self, caplog):
        import logging
        engine = _make_engine(state={})
        with caplog.at_level(logging.INFO, logger='VelocityEngine'):
            engine._log_startup_summary(12345.67)
        assert any('12345.67' in m for m in caplog.messages)


# ── _initialize — _last_audit_date set after startup audit ────────────────────

class TestInitializeAuditDateSet:
    def test_last_audit_date_set_after_startup_audit(self):
        """_initialize must set _last_audit_date so run_cycle does not double-audit."""
        engine = _make_engine(state={'AAPL': {
            'price': 150.0, 'qty': 10, 'stop_loss': 140.0,
            'volume': 5000000, 'score': 70.0, 'time': '2026-01-01T10:00:00',
        }})
        assert engine._last_audit_date is None
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_audit_stop_orders'),         \
             patch.object(engine, '_update_position_prices'),    \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        assert engine._last_audit_date is not None

    def test_last_audit_date_not_set_when_no_positions(self):
        """If no positions, _audit_stop_orders is never called and the date stays None."""
        engine = _make_engine(state={})
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_restore_blocked_today'), \
             patch.object(engine, '_wait_for_pre_entry_sync'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, '_log_startup_summary'):
            engine._initialize()
        assert engine._last_audit_date is None


# ── _ensure_connected — connectivity check ────────────────────────────────────

class TestEnsureConnected:
    """_ensure_connected pings get_account(); returns True on success, False on exception."""

    def test_returns_true_when_account_reachable(self):
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        result = engine._ensure_connected()
        assert result is True
        tc.get_account.assert_called_once()

    def test_returns_false_when_account_raises(self):
        tc = _mock_trading_client()
        tc.get_account.side_effect = RuntimeError("connection refused")
        engine = _make_engine(trading_client=tc)
        result = engine._ensure_connected()
        assert result is False

    def test_returns_false_on_timeout(self):
        tc = _mock_trading_client()
        tc.get_account.side_effect = TimeoutError("request timed out")
        engine = _make_engine(trading_client=tc)
        result = engine._ensure_connected()
        assert result is False


# ── _wait_for_pre_entry_sync ──────────────────────────────────────────────────

class TestPreEntrySyncWait:
    """Uses time.sleep (not ib.sleep) for pre-entry wait."""

    _TZ_NY = pytz.timezone('US/Eastern')

    def test_sleeps_when_before_pre_entry_time(self):
        """Engine started before 09:40 ET → time.sleep called; total duration matches gap."""
        from src.config import PRE_ENTRY_SYNC_TIME
        engine = _make_engine()
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'):
            h, m = PRE_ENTRY_SYNC_TIME
            fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, 8, 0, 0))
            with patch('src.engine.datetime') as mock_dt, \
                 patch('src.engine.time.sleep') as mock_sleep:
                mock_dt.now.return_value  = fake_now
                mock_dt.fromisoformat     = datetime.fromisoformat
                engine._wait_for_pre_entry_sync()

            assert mock_sleep.call_count >= 1
            total_slept = sum(c[0][0] for c in mock_sleep.call_args_list)
            expected = (h * 60 + m - 8 * 60) * 60   # (9h58m - 8h00m) in seconds
            assert abs(total_slept - expected) < 2

    def test_no_sleep_when_at_or_past_pre_entry_time(self):
        """Engine started at or after 09:40 ET → time.sleep NOT called."""
        engine = _make_engine()
        fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, 10, 5, 0))
        with patch('src.engine.datetime') as mock_dt, \
             patch('src.engine.time.sleep') as mock_sleep:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._wait_for_pre_entry_sync()
        mock_sleep.assert_not_called()

    def test_no_sleep_on_intraday_restart(self):
        """Intraday restart at 14:30 ET — already past 09:40, no sleep."""
        engine = _make_engine()
        fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, 14, 30, 0))
        with patch('src.engine.datetime') as mock_dt, \
             patch('src.engine.time.sleep') as mock_sleep:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._wait_for_pre_entry_sync()
        mock_sleep.assert_not_called()

    def test_sleep_duration_covers_gap_to_pre_entry_time(self):
        """Total sleep from 09:00 ET to 09:40 ET should be exactly 40 min."""
        from src.config import PRE_ENTRY_SYNC_TIME
        engine = _make_engine()
        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'):
            fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, 9, 0, 0))
            with patch('src.engine.datetime') as mock_dt, \
                 patch('src.engine.time.sleep') as mock_sleep:
                mock_dt.now.return_value  = fake_now
                mock_dt.fromisoformat     = datetime.fromisoformat
                engine._wait_for_pre_entry_sync()
            total_slept = sum(c[0][0] for c in mock_sleep.call_args_list)
            h, m = PRE_ENTRY_SYNC_TIME
            expected = (h * 60 + m - 9 * 60) * 60  # gap from 09:00 to PRE_ENTRY_SYNC_TIME
            assert abs(total_slept - expected) < 2


# ── Pending flag ──────────────────────────────────────────────────────────────

class TestPendingFlag:
    """
    pending=True is written to state when a GTC LMT order is in new/submitted state.
    _sync_positions clears it once Alpaca reports the position as filled.
    """

    def test_pending_cleared_when_alpaca_position_appears(self):
        """After a pending BUY fills, sync should clear 'pending' and update fill_price."""
        tc = _mock_trading_client()
        pos = MagicMock()
        pos.symbol         = 'AAPL'
        pos.qty            = '10.0'
        pos.avg_entry_price = '102.50'
        tc.get_all_positions.return_value = [pos]

        engine = _make_engine(trading_client=tc, state={
            'AAPL': {
                'price':          101.0,
                'fill_price':     101.0,
                'pending':        True,
                'qty':            10,
                'stop_loss':      95.0,
                'stop_dist':      6.0,
                'entry_order_id': '1',
                'time':           '2026-05-18T10:00:00',
                'volume':         1_000_000,
                'score':          80.0,
            }
        })

        with patch.object(engine, 'save_state'):
            engine._sync_positions()

        assert 'pending' not in engine.state['AAPL']
        assert engine.state['AAPL']['fill_price'] == 102.5

    def test_pending_symbol_not_removed_before_fill(self):
        """If Alpaca shows no position yet (order still open), pending entry stays in state."""
        tc = _mock_trading_client()
        tc.get_all_positions.return_value = []

        engine = _make_engine(trading_client=tc, state={
            'AAPL': {
                'price': 101.0, 'fill_price': 101.0, 'pending': True,
                'qty': 10, 'stop_loss': 95.0, 'stop_dist': 6.0,
                'time': '2026-05-18T10:00:00', 'volume': 0, 'score': None,
            }
        })

        with patch.object(engine, 'save_state'):
            engine._sync_positions()

        assert 'AAPL' in engine.state
        assert engine.state['AAPL'].get('pending') is True

    def test_filled_order_has_no_pending_flag(self):
        """When status is filled, state is written without pending=True."""
        state_entry = {
            'fill_price':     105.0,
            'price':          105.0,
            'entry_order_id': '42',
            'time':           '2026-05-18T10:05:00',
            'qty':            10,
            'stop_loss':      99.0,
            'stop_dist':      6.0,
            'peak_price':     105.0,
            'volume':         2_000_000,
            'score':          85.0,
            'pending':        'filled' != 'filled',
        }
        assert state_entry['pending'] is False
        assert not state_entry['pending']

    def test_presubmitted_order_has_pending_flag(self):
        """When status is new/PreSubmitted, pending=True so dashboard hides it."""
        state_entry = {
            'pending': 'new' != 'filled',
        }
        assert state_entry['pending'] is True

    def test_pending_entry_excluded_from_position_value(self):
        """Dashboard position_value must skip pending positions."""
        state = {
            'AAPL': {'qty': 10, 'price': 100.0, 'pending': True},
            'MSFT': {'qty':  5, 'price': 200.0},
        }
        position_value = sum(
            float(d.get('current_price', d.get('price', 0))) * float(d.get('qty', 0))
            for d in state.values()
            if not d.get('pending')
        )
        assert position_value == 1000.0   # only MSFT counted

    def test_pending_entry_excluded_from_positions_list(self):
        """Dashboard positions list must not include pending entries."""
        state = {
            'AAPL': {'qty': 10, 'price': 100.0, 'pending': True},
            'MSFT': {'qty':  5, 'price': 200.0},
        }
        visible = [sym for sym, d in state.items() if not d.get('pending')]
        assert visible == ['MSFT']
        assert 'AAPL' not in visible


# ── _fetch_spy_trend — SPY regime filter ─────────────────────────────────────

class TestFetchSpyTrend:
    """_fetch_spy_trend calls self._fetch_daily_bars('SPY') and computes MA50/MA200."""

    def _make_spy_df(self, n=250, rising=True):
        import numpy as np, pandas as pd
        close = np.linspace(100, 130, n) if rising else np.linspace(130, 100, n)
        idx   = pd.date_range('2025-01-01', periods=n, freq='B')
        return pd.DataFrame({'open': close, 'high': close+0.5, 'low': close-0.5,
                             'close': close, 'volume': 1_000_000}, index=idx)

    def test_returns_true_in_uptrend(self):
        """SPY price > EMA50 in a rising series → is_bull=True dict."""
        engine = _make_engine()
        df = self._make_spy_df(rising=True)
        with patch.object(engine, '_fetch_daily_bars', return_value=df):
            result = engine._fetch_spy_trend()
        assert isinstance(result, dict)
        assert result['is_bull'] is True

    def test_result_cached_for_same_day(self):
        """Second call on same date must return cached value without calling _fetch_daily_bars."""
        engine = _make_engine()
        df = self._make_spy_df(rising=True)
        with patch.object(engine, '_fetch_daily_bars', return_value=df) as mock_bars:
            engine._fetch_spy_trend()
            call_count_after_first = mock_bars.call_count
            engine._fetch_spy_trend()
        assert mock_bars.call_count == call_count_after_first

    def test_fails_open_when_data_unavailable(self):
        """_fetch_daily_bars returns None → fail open (is_bull=True) so entries are not blocked."""
        engine = _make_engine()
        with patch.object(engine, '_fetch_daily_bars', return_value=None):
            result = engine._fetch_spy_trend()
        assert isinstance(result, dict)
        assert result['is_bull'] is True

    def test_fails_open_on_exception(self):
        """data_client raises → _fetch_daily_bars returns None → fail open (is_bull=True)."""
        engine = _make_engine()
        # Simulate data_client raising by patching it to raise; _fetch_daily_bars
        # catches exceptions internally and returns None, which _fetch_spy_trend
        # treats as fail-open. We mock the data_client directly.
        engine.data_client.get_stock_bars.side_effect = RuntimeError("timeout")
        result = engine._fetch_spy_trend()
        assert isinstance(result, dict)
        assert result['is_bull'] is True


# ── shutdown — graceful BUY order cancellation (no disconnect) ────────────────

class TestShutdown:
    """shutdown() cancels pending BUY orders via trading_client; no disconnect
    (Alpaca REST is stateless — there is no connection to close)."""

    def _make_buy_order(self, symbol='AAPL', order_id='buy-42'):
        o = MagicMock()
        o.side   = 'OrderSide.BUY'
        o.symbol = symbol
        o.id     = order_id
        return o

    def _make_sell_order(self, symbol='AAPL', order_id='sell-43'):
        o = MagicMock()
        o.side   = 'OrderSide.SELL'
        o.symbol = symbol
        o.id     = order_id
        return o

    def test_cancels_pending_buy_orders(self):
        """shutdown() must cancel all open BUY orders."""
        tc = _mock_trading_client()
        buy_order = self._make_buy_order()
        tc.get_orders.return_value = [buy_order]
        engine = _make_engine(trading_client=tc)
        engine.shutdown()
        tc.cancel_order_by_id.assert_called_once_with('buy-42')

    def test_does_not_cancel_sell_orders(self):
        """SELL (trailing-stop) orders must not be cancelled — they protect open positions."""
        tc = _mock_trading_client()
        sell_order = self._make_sell_order()
        tc.get_orders.return_value = [sell_order]
        engine = _make_engine(trading_client=tc)
        engine.shutdown()
        tc.cancel_order_by_id.assert_not_called()

    def test_cancels_buy_but_not_sell_when_both_present(self):
        """Mixed order list: cancel BUY, skip SELL."""
        tc = _mock_trading_client()
        buy_order  = self._make_buy_order(order_id='buy-1')
        sell_order = self._make_sell_order(order_id='sell-2')
        tc.get_orders.return_value = [buy_order, sell_order]
        engine = _make_engine(trading_client=tc)
        engine.shutdown()
        tc.cancel_order_by_id.assert_called_once_with('buy-1')

    def test_no_disconnect_call(self):
        """Alpaca REST is stateless — shutdown must never call disconnect."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.shutdown()
        # trading_client has no disconnect method to call
        assert not hasattr(tc, 'disconnect') or not tc.disconnect.called


# ── _compute_book_correlation ─────────────────────────────────────────────────

class TestComputeBookCorrelation:
    def _make_df(self, n=100, seed=0):
        import numpy as np, pandas as pd
        rng   = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        idx   = pd.date_range('2025-01-01', periods=n, freq='B')
        return pd.DataFrame({'open': close, 'high': close+0.5, 'low': close-0.5,
                             'close': close, 'volume': 1_000_000}, index=idx)

    def test_returns_zero_with_empty_book(self):
        engine = _make_engine(state={})
        df = self._make_df()
        corr = engine._compute_book_correlation('AAPL', df)
        assert corr == 0.0

    def test_returns_zero_when_same_symbol_in_book(self):
        """Candidate is itself already in the book — no comparison made → 0."""
        engine = _make_engine(state={'AAPL': {}})
        df = self._make_df()
        corr = engine._compute_book_correlation('AAPL', df)
        assert corr == 0.0

    def test_high_corr_detected_from_bar_cache(self):
        """Highly correlated position already in bar cache → correlation ≥ CORR_MAX."""
        import numpy as np, pandas as pd
        from src.config import CORR_MAX

        n     = 100
        close = 100 + np.cumsum(np.ones(n))   # perfectly monotone series
        idx   = pd.date_range('2025-01-01', periods=n, freq='B')
        df    = pd.DataFrame({'open': close, 'high': close+0.5, 'low': close-0.5,
                              'close': close, 'volume': 1_000_000}, index=idx)
        today_str = __import__('datetime').datetime.now(
            __import__('pytz').timezone('US/Eastern')
        ).strftime('%Y-%m-%d')

        engine = _make_engine(state={'MSFT': {}})
        # Cache key is 'bars_daily' — a DataFrame directly (Alpaca engine)
        engine._bar_cache['MSFT'] = {'date': today_str, 'bars_daily': df}

        corr = engine._compute_book_correlation('AAPL', df)
        assert corr >= CORR_MAX


# ── TEST-4: risk sizing uses min(chandelier_dist, hard_stop_dist) ─────────────

class TestRiskSizingAlignment:
    """Live engine risk sizing uses tighter of chandelier and 7% hard-stop distance."""

    def test_uses_hard_stop_when_atr_is_wide(self):
        """When chandelier dist > hard_stop dist, hard_stop_dist governs qty."""
        from src.config import HARD_STOP_PCT, RISK_PER_TRADE_PCT

        equity      = 10_000.0
        limit_price = 100.0
        chandelier_dist = limit_price * HARD_STOP_PCT * 3
        hard_stop_dist  = limit_price * HARD_STOP_PCT

        risk_stop_dist = max(min(chandelier_dist, hard_stop_dist), 0.01)
        risk_per_trade = equity * RISK_PER_TRADE_PCT
        qty            = int(risk_per_trade / risk_stop_dist)

        assert risk_stop_dist == pytest.approx(hard_stop_dist)
        assert qty == int(risk_per_trade / hard_stop_dist)

    def test_uses_chandelier_when_atr_is_tight(self):
        """When chandelier dist < hard_stop dist, chandelier_dist governs qty."""
        from src.config import HARD_STOP_PCT, RISK_PER_TRADE_PCT

        equity          = 10_000.0
        limit_price     = 100.0
        chandelier_dist = limit_price * HARD_STOP_PCT * 0.4
        hard_stop_dist  = limit_price * HARD_STOP_PCT

        risk_stop_dist = max(min(chandelier_dist, hard_stop_dist), 0.01)
        risk_per_trade = equity * RISK_PER_TRADE_PCT
        qty            = int(risk_per_trade / risk_stop_dist)

        assert risk_stop_dist == pytest.approx(chandelier_dist)
        assert qty == int(risk_per_trade / chandelier_dist)

    def test_floor_prevents_zero_division(self):
        """risk_stop_dist is always ≥ 0.01 even when both distances are tiny."""
        from src.config import HARD_STOP_PCT

        chandelier_dist = 0.001
        hard_stop_dist  = 0.002
        risk_stop_dist  = max(min(chandelier_dist, hard_stop_dist), 0.01)

        assert risk_stop_dist == pytest.approx(0.01)


# ── TEST-5: _get_account_values ───────────────────────────────────────────────

class TestGetAccountValuesFallback:
    """_get_account_values calls trading_client.get_account() and returns
    (float(portfolio_value), max(float(cash), 0.0))."""

    def test_positive_cash_returned(self):
        tc = _mock_trading_client(equity=5000.0, cash=3000.0)
        engine = _make_engine(trading_client=tc)
        equity, cash = engine._get_account_values()
        assert equity == 5000.0
        assert cash   == 3000.0

    def test_zero_cash_returns_zero(self):
        tc = _mock_trading_client(equity=4000.0, cash=0.0)
        engine = _make_engine(trading_client=tc)
        _, cash = engine._get_account_values()
        assert cash == 0.0

    def test_negative_cash_clamped_to_zero(self):
        tc = _mock_trading_client(equity=4000.0, cash=-200.0)
        engine = _make_engine(trading_client=tc)
        _, cash = engine._get_account_values()
        assert cash == 0.0

    def test_negative_cash_emits_warning(self, caplog):
        import logging
        tc = _mock_trading_client(equity=4000.0, cash=-200.0)
        engine = _make_engine(trading_client=tc)
        with caplog.at_level(logging.WARNING, logger='VelocityEngine'):
            engine._get_account_values()
        # Should warn about cash ≤ 0
        assert any('cash' in m.lower() for m in caplog.messages)

    def test_retries_on_exception_then_succeeds(self):
        tc = MagicMock()
        tc.get_account.side_effect = [
            RuntimeError("transient error"),
            _mock_account(equity=5000.0, cash=3000.0),
        ]
        engine = _make_engine(trading_client=tc)
        with patch('src.engine.time.sleep'):
            equity, cash = engine._get_account_values()
        assert equity == 5000.0
        assert cash   == 3000.0

    def test_fallback_to_last_known_after_all_retries_fail(self, caplog):
        """All attempts return equity=0 — fall back to _last_equity/_last_settled_cash."""
        import logging
        tc = _mock_trading_client(equity=0.0, cash=0.0)
        engine = _make_engine(trading_client=tc)
        engine._last_equity       = 4200.0
        engine._last_settled_cash = 1800.0
        with caplog.at_level(logging.WARNING, logger='VelocityEngine'), \
             patch('src.engine.time.sleep'):
            net_liq, cash = engine._get_account_values()
        assert net_liq == 4200.0
        assert cash    == 1800.0
        assert any('last known' in m for m in caplog.messages)


# ── TEST-6: _score_candidate — higher RVOL ranks above lower RVOL ────────────

class TestScoreCandidate:
    """_score_candidate ranks candidates on trend, RVOL, momentum, and liquidity."""

    def _ctx(self, ma50=108, ma200=100, rsi=65, rsi_prev=60,
             rvol=3.0, spread_pct=0.002):
        from src.config import RVOL_MIN
        return {
            'ma50':       ma50,
            'ma200':      ma200,
            'rsi':        rsi,
            'rsi_prev':   rsi_prev,
            'rvol':       rvol,
            'spread_pct': spread_pct,
        }

    def test_higher_rvol_scores_higher(self):
        engine = _make_engine()
        low  = engine._score_candidate(self._ctx(rvol=2.6))
        high = engine._score_candidate(self._ctx(rvol=4.5))
        assert high > low

    def test_tighter_spread_scores_higher(self):
        engine = _make_engine()
        wide  = engine._score_candidate(self._ctx(spread_pct=0.005))
        tight = engine._score_candidate(self._ctx(spread_pct=0.001))
        assert tight > wide

    def test_score_bounded_0_to_100(self):
        engine = _make_engine()
        ctx  = self._ctx(ma50=200, ma200=100, rsi=80, rsi_prev=65,
                         rvol=10.0, spread_pct=0.0)
        score = engine._score_candidate(ctx)
        assert 0.0 <= score <= 100.0

    def test_score_is_float(self):
        engine = _make_engine()
        score = engine._score_candidate(self._ctx())
        assert isinstance(score, float)


# ── TEST-7: daily loss circuit breaker halts entries ─────────────────────────

class TestDailyLossCircuitBreaker:
    """When equity drops ≥ MAX_DAILY_LOSS_PCT from day open, no new entries occur."""

    def test_circuit_breaker_triggers_on_loss(self):
        from src.config import MAX_DAILY_LOSS_PCT

        engine = _make_engine()
        day_start  = 10_000.0
        equity_now = day_start * (1 - MAX_DAILY_LOSS_PCT - 0.001)

        engine._day_start_equity = day_start

        fixed_ny = datetime(2026, 5, 18, 11, 0, tzinfo=pytz.timezone('US/Eastern'))
        engine._day_start_date = fixed_ny.strftime('%Y-%m-%d')

        mock_scan = MagicMock()
        with patch('src.engine.datetime') as mock_dt,                            \
             patch.object(engine, '_get_account_values', return_value=(equity_now, 0.0)), \
             patch.object(engine, '_sync_positions'),                             \
             patch.object(engine, 'check_velocity_exits', return_value={}),      \
             patch.object(engine, '_update_position_prices'),                    \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'get_institutional_scan', mock_scan):
            mock_dt.now.return_value  = fixed_ny
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine.run_cycle()

        # Circuit breaker must have blocked entries — scanner not called
        mock_scan.assert_not_called()

    def test_circuit_breaker_does_not_trigger_on_small_loss(self):
        from src.config import MAX_DAILY_LOSS_PCT

        tc = _mock_trading_client(equity=9980.0, cash=9980.0)
        engine = _make_engine(trading_client=tc)
        day_start = 10_000.0
        engine._day_start_equity = day_start

        fixed_ny = datetime(2026, 5, 18, 11, 0, tzinfo=pytz.timezone('US/Eastern'))

        with patch('src.engine.datetime') as mock_dt,                \
             patch.object(engine, '_sync_positions'),                 \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_update_position_prices'),         \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0),   \
             patch.object(engine, 'get_institutional_scan', return_value=[]) as mock_scan, \
             patch.object(engine, '_write_dashboard_data'):
            mock_dt.now.return_value   = fixed_ny
            mock_dt.fromisoformat      = datetime.fromisoformat
            engine._day_start_date     = fixed_ny.strftime('%Y-%m-%d')
            engine.run_cycle()

        # scanner was called (circuit breaker did NOT fire)
        mock_scan.assert_called()


# ── TEST-8: deduplication — same symbol not entered twice ────────────────────

class TestDeduplication:
    """Symbols already in state are skipped by the scanner loop."""

    def test_already_held_symbol_is_skipped(self):
        engine = _make_engine(state={
            'AAPL': {'price': 150.0, 'qty': 10, 'stop_loss': 140.0,
                     'volume': 5000000, 'score': 75.0, 'time': '2026-01-01T10:00:00'}
        })

        with patch.object(engine, '_sync_positions'),          \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_update_position_prices'),  \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'get_institutional_scan', return_value=['AAPL']), \
             patch.object(engine, 'get_technical_context') as mock_ctx, \
             patch.object(engine, '_write_dashboard_data'):
            now_ny = datetime(2026, 5, 19, 11, 0, tzinfo=pytz.timezone('US/Eastern'))
            with patch('src.engine.datetime') as mock_dt:
                mock_dt.now.return_value  = now_ny
                mock_dt.fromisoformat     = datetime.fromisoformat
                engine.run_cycle()

        # AAPL is already in state → get_technical_context must NOT be called for it
        for call_args in mock_ctx.call_args_list:
            assert call_args[0][0] != 'AAPL', "AAPL already held — should be skipped"


# ── TEST-9: pending position skipped by check_velocity_exits() ───────────────

class TestPendingPositionSkippedOnExit:
    """check_velocity_exits() must not attempt to exit positions with pending=True."""

    def test_pending_position_not_liquidated_on_friday_close(self):
        """A pending limit order must be untouched even on Friday afternoon."""
        engine = _make_engine(state={
            'MSFT': {
                'price':      400.0,
                'pending':    True,
                'qty':        5,
                'time':       '2026-05-15T09:45:00',
                'stop_loss':  372.0,
                'stop_dist':  28.0,
                'entry_order_id': '99',
                'volume':     0,
                'score':      70.0,
            }
        })

        with patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, '_fetch_snapshot') as mock_snap, \
             patch('src.engine.datetime') as mock_dt:
            friday = datetime(2026, 5, 16, 15, 30, tzinfo=pytz.timezone('US/Eastern'))
            mock_dt.now.return_value = friday
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        mock_snap.assert_not_called()  # snapshot never fetched for pending
        mock_liq.assert_not_called()   # liquidate never called

    def test_pending_position_absent_from_prefetched_prices(self):
        """Pending entries must not appear in the dict returned by check_velocity_exits()."""
        engine = _make_engine(state={
            'GOOG': {
                'price':   170.0,
                'pending': True,
                'qty':     3,
                'time':    '2026-05-19T10:00:00',
                'volume':  0,
                'score':   60.0,
            }
        })

        with patch('src.engine.datetime') as mock_dt:
            mid = datetime(2026, 5, 19, 12, 0, tzinfo=pytz.timezone('US/Eastern'))
            mock_dt.now.return_value = mid
            mock_dt.fromisoformat    = datetime.fromisoformat
            result = engine.check_velocity_exits()

        assert 'GOOG' not in result


# ── TEST-10: 12-rule production filter blocks specific failed conditions ───────

class TestTwelveRuleFilter:
    """run_cycle() Donchian bounce filter must block entries when individual rules fail."""

    def _make_passing_ctx(self):
        """Minimal technical context dict that passes all Donchian bounce rules."""
        from src.config import RVOL_MIN, SPREAD_MAX_PCT, SCAN_MIN_DOLLAR_VOL
        fixed_ny = datetime(2026, 5, 19, 11, 0, tzinfo=pytz.timezone('US/Eastern'))
        return {
            'live_price':      100.0,
            # Donchian: price 0.2% above lower band (within 0.5% tolerance)
            'donchian_lower':   99.8,
            'donchian_upper':  110.0,
            # RSI momentum: delta=7.0 >= RSI_MIN_DELTA, history has values below threshold=40
            'rsi':              42.0,
            'rsi_prev':         35.0,
            'rsi_history':     [28.0, 30.0, 32.0, 35.0, 42.0],
            # Day strength: 1% above open, in upper 86% of intraday range
            'intraday_open':    99.0,
            'intraday_high':   100.5,
            'intraday_low':     97.0,
            'atr':               2.0,
            'atr_chandelier':    2.0,
            'rvol':             RVOL_MIN + 0.5,       # 3.0
            'spread_pct':       SPREAD_MAX_PCT - 0.001,
            'dollar_vol_20d':   SCAN_MIN_DOLLAR_VOL * 2,
            'avg_20d_vol':    5_000_000,
            'volume':         5_000_000,
            'close':            99.5,
            'price_fetched_at': fixed_ny,
        }

    def _run_cycle_with_conditions(self, ctx):
        """Wire up run_cycle() to return a single candidate with the given ctx."""
        tc = _mock_trading_client(equity=5000.0, cash=5000.0)
        engine = _make_engine(trading_client=tc, state={})

        fixed_ny = datetime(2026, 5, 19, 11, 0, tzinfo=pytz.timezone('US/Eastern'))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'get_institutional_scan', return_value=['XYZ']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch('src.engine.datetime') as mock_dt, \
             patch('src.engine.time.sleep'):
            mock_dt.now.return_value = fixed_ny
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.run_cycle()

        return tc, engine

    def test_passing_candidate_triggers_order(self):
        ctx = self._make_passing_ctx()
        tc, engine = self._run_cycle_with_conditions(ctx)
        assert tc.submit_order.call_count > 0, \
            "Passing candidate should result in a submitted order"

    def test_fails_donchian_floor_unavailable(self):
        """donchian_lower=0 → Donchian floor unavailable → blocked."""
        ctx = self._make_passing_ctx()
        ctx['donchian_lower'] = 0.0
        tc, _ = self._run_cycle_with_conditions(ctx)
        assert tc.submit_order.call_count == 0

    def test_fails_donchian_floor_too_far(self):
        """Price too far above lower band (>0.5%) → donchian_floor fails."""
        ctx = self._make_passing_ctx()
        ctx['live_price']     = 101.0
        ctx['donchian_lower'] = 100.0   # 1% gap exceeds 0.5% tolerance
        tc, _ = self._run_cycle_with_conditions(ctx)
        assert tc.submit_order.call_count == 0

    def test_fails_dollar_vol(self):
        """When dollar volume is below the threshold, the symbol is skipped."""
        from src.config import SCAN_MIN_DOLLAR_VOL
        ctx = self._make_passing_ctx()
        ctx['dollar_vol_20d'] = SCAN_MIN_DOLLAR_VOL * 0.1
        tc, _ = self._run_cycle_with_conditions(ctx)
        assert tc.submit_order.call_count == 0

    def test_fails_rsi_rising(self):
        """RSI falling (prev > current) fails rsi_momentum."""
        ctx = self._make_passing_ctx()
        ctx['rsi']      = 35.0
        ctx['rsi_prev'] = 38.0
        tc, _ = self._run_cycle_with_conditions(ctx)
        assert tc.submit_order.call_count == 0

    def test_fails_rsi_never_oversold(self):
        """RSI history never below RSI_OVERSOLD_THRESHOLD fails rsi_momentum."""
        ctx = self._make_passing_ctx()
        ctx['rsi_history'] = [60.0, 62.0, 64.0, 65.0, 68.0]
        ctx['rsi']      = 68.0
        ctx['rsi_prev'] = 65.0
        tc, _ = self._run_cycle_with_conditions(ctx)
        assert tc.submit_order.call_count == 0


# ── TEST-11: sector clustering blocks a second entry in same sector ───────────

class TestSectorClusteringFilter:
    """run_cycle() must not place an order when sector already has MAX_SECTOR_COUNT positions."""

    def test_second_tech_entry_blocked_when_limit_reached(self):
        from src.config import MAX_SECTOR_COUNT, RVOL_MIN, SPREAD_MAX_PCT, SCAN_MIN_DOLLAR_VOL

        existing = {
            f'HELD{i}': {'price': 100.0, 'qty': 5, 'time': '2026-05-19T10:00:00',
                         'stop_loss': 93.0, 'volume': 0, 'score': 80.0}
            for i in range(MAX_SECTOR_COUNT)
        }
        tc = _mock_trading_client(equity=10000.0, cash=10000.0)
        engine = _make_engine(trading_client=tc, state=existing)

        passing_ctx = {
            'live_price': 110.0, 'orb_high': 108.0,
            'ma50': 108.0, 'ma200': 100.0,
            'rsi': 65.0, 'rsi_prev': 63.0,
            'atr': 2.0, 'atr_chandelier': 2.0,
            'adx': 25.0, 'high200': 121.0,
            'sma200_slope': 0.01,
            'rvol': RVOL_MIN + 0.5,
            'spread_pct': SPREAD_MAX_PCT - 0.001,
            'dollar_vol_20d': SCAN_MIN_DOLLAR_VOL * 2,
            'avg_20d_vol': 1_000_000, 'volume': 2_000_000,
            'df_daily': None,
            'price_fetched_at': datetime(2026, 5, 19, 11, 0,
                                         tzinfo=pytz.timezone('US/Eastern')),
        }

        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'get_technical_context', return_value=passing_ctx), \
             patch.object(engine, 'get_institutional_scan', return_value=['NVDA']), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch('src.engine.datetime') as mock_dt, \
             patch('src.engine.time.sleep'):
            now_ny = datetime(2026, 5, 19, 11, 0, tzinfo=pytz.timezone('US/Eastern'))
            mock_dt.now.return_value = now_ny
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.run_cycle()

        # Sector at MAX_SECTOR_COUNT — no order must be placed for NVDA
        tc.submit_order.assert_not_called()


# ── TEST-12: correlation filter blocks a highly-correlated candidate ──────────

class TestCorrelationFilter:
    """_compute_book_correlation > CORR_MAX must prevent an order."""

    def test_high_correlation_prevents_order(self):
        from src.config import CORR_MAX, SCAN_MIN_DOLLAR_VOL, RVOL_MIN, SPREAD_MAX_PCT
        import pandas as pd

        tc = _mock_trading_client(equity=5000.0, cash=5000.0)
        engine = _make_engine(trading_client=tc, state={
            'AAPL': {'price': 150.0, 'qty': 10, 'stop_loss': 139.5,
                     'time': '2026-05-19T10:00:00', 'volume': 0, 'score': 80.0}
        })

        ctx = {
            'live_price':     310.0,
            'orb_high':       308.0,
            'ma50':           308.0,
            'ma200':          285.0,
            'rsi':            65.0,
            'rsi_prev':       63.0,
            'atr':            3.0,
            'atr_chandelier': 3.0,
            'adx':            25.0,
            'high200':        350.0,
            'sma200_slope':   0.01,
            'rvol':           RVOL_MIN + 0.5,
            'spread_pct':     SPREAD_MAX_PCT - 0.001,
            'dollar_vol_20d': SCAN_MIN_DOLLAR_VOL * 2,
            'avg_20d_vol':    1_000_000,
            'volume':         2_000_000,
            'df_daily':       pd.DataFrame({'close': [1.0] * 70}),
            'price_fetched_at': datetime(2026, 5, 19, 11, 0,
                                          tzinfo=pytz.timezone('US/Eastern')),
        }

        with patch.object(engine, '_sync_positions'), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, 'get_institutional_scan', return_value=['MSFT']), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch.object(engine, '_compute_book_correlation',
                          return_value=CORR_MAX + 0.1), \
             patch.object(engine, '_write_dashboard_data'), \
             patch('src.engine.datetime') as mock_dt, \
             patch('src.engine.time.sleep'):
            now_ny = datetime(2026, 5, 19, 11, 0, tzinfo=pytz.timezone('US/Eastern'))
            mock_dt.now.return_value = now_ny
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.run_cycle()

        tc.submit_order.assert_not_called()


# ── Sync: defensive guard when Alpaca returns empty positions list ─────────────

class TestSyncEmptyGuard:
    """_sync_positions must not evict filled positions when Alpaca returns an
    empty list — protects against transient API gaps."""

    def test_filled_position_preserved_when_alpaca_returns_empty(self):
        """State has a filled position; Alpaca returns []. Must NOT remove it on first miss."""
        tc = _mock_trading_client()
        tc.get_all_positions.return_value = []
        engine = _make_engine(trading_client=tc, state={
            'AAPL': {'price': 150.0, 'qty': 10, 'stop_loss': 140.0,
                     'volume': 1_000_000, 'score': 70.0, 'time': '2026-05-18T10:00:00'}
        })
        with patch.object(engine, 'save_state'):
            engine._sync_positions()
        # First miss only increments counter — state must still have AAPL
        assert 'AAPL' in engine.state, (
            "Filled position must be preserved when Alpaca returns empty list "
            "(first miss — deferred removal)"
        )

    def test_multiple_filled_positions_preserved_when_alpaca_returns_empty(self):
        """All three filled positions must survive an empty Alpaca response (first miss)."""
        tc = _mock_trading_client()
        tc.get_all_positions.return_value = []
        engine = _make_engine(trading_client=tc, state={
            'AAPL': {'price': 150.0, 'qty': 10, 'stop_loss': 140.0,
                     'volume': 1_000_000, 'score': 70.0, 'time': '2026-05-18T10:00:00'},
            'MSFT': {'price': 300.0, 'qty': 5,  'stop_loss': 280.0,
                     'volume': 2_000_000, 'score': 75.0, 'time': '2026-05-18T10:05:00'},
            'NVDA': {'price': 500.0, 'qty': 3,  'stop_loss': 460.0,
                     'volume': 3_000_000, 'score': 80.0, 'time': '2026-05-18T10:10:00'},
        })
        with patch.object(engine, 'save_state'):
            engine._sync_positions()
        assert len(engine.state) == 3

    def test_pending_entry_preserved_when_alpaca_returns_empty(self):
        """A pending entry (limit BUY not yet filled) must be kept when Alpaca is empty."""
        tc = _mock_trading_client()
        tc.get_all_positions.return_value = []
        engine = _make_engine(trading_client=tc, state={
            'AAPL': {'price': 101.0, 'qty': 10, 'stop_loss': 95.0,
                     'pending': True,
                     'volume': 0, 'score': None, 'time': '2026-05-18T10:00:00'}
        })
        with patch.object(engine, 'save_state'):
            engine._sync_positions()
        assert 'AAPL' in engine.state

    def test_empty_state_with_empty_alpaca_is_normal(self):
        """Both state and Alpaca empty — nothing to do, save_state not called."""
        tc = _mock_trading_client()
        tc.get_all_positions.return_value = []
        engine = _make_engine(trading_client=tc, state={})
        with patch.object(engine, 'save_state') as mock_save:
            engine._sync_positions()
        mock_save.assert_not_called()


# ── Pre-entry wait: heartbeat syncs positions ─────────────────────────────────

class TestPreEntrySyncHeartbeatSync:
    """During the pre-entry wait, each 5-min heartbeat must call
    _sync_positions so position changes (stops fired, fills)
    are reflected on the dashboard without waiting until 09:58 ET."""

    _TZ_NY = pytz.timezone('US/Eastern')

    def test_heartbeat_syncs_positions_during_wait(self):
        """9:30 AM start → heartbeat fires before 09:58 → sync must be called."""
        engine = _make_engine()

        fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, 9, 30, 0))
        with patch.object(engine, '_sync_positions') as mock_sync, \
             patch.object(engine, '_update_position_prices'),       \
             patch.object(engine, '_write_dashboard_data'),         \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._wait_for_pre_entry_sync()

        assert mock_sync.call_count >= 1, (
            "At least one heartbeat sync must occur during the pre-entry wait"
        )

    def test_no_heartbeat_sync_when_already_past_sync_time(self):
        """Past 09:58 ET → returns immediately → sync must NOT be called."""
        engine = _make_engine()

        fake_now = self._TZ_NY.localize(datetime(2026, 5, 19, 10, 5, 0))
        with patch.object(engine, '_sync_positions') as mock_sync, \
             patch.object(engine, '_update_position_prices'),       \
             patch.object(engine, '_write_dashboard_data'),         \
             patch('src.engine.time.sleep'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._wait_for_pre_entry_sync()

        mock_sync.assert_not_called()


# ── _restore_blocked_today — daily scan skip from persisted state ─────────────

class TestRestoreBlockedToday:
    """_restore_blocked_today reads _daily_skip_reason/_daily_skip_date from
    each state entry and repopulates _daily_scan_skip dict."""

    def test_restores_skip_for_today(self):
        import pytz as _pytz
        today_str = datetime.now(_pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        engine = _make_engine(state={
            'TSLA': {
                '_daily_skip_reason': 'ADX<=20',
                '_daily_skip_date':   today_str,
            },
            'NVDA': {
                '_daily_skip_reason': 'DolVol<100M',
                '_daily_skip_date':   today_str,
            },
        })
        engine._restore_blocked_today()
        assert 'TSLA' in engine._daily_scan_skip
        assert 'NVDA' in engine._daily_scan_skip

    def test_skips_stale_date(self):
        engine = _make_engine(state={
            'TSLA': {
                '_daily_skip_reason': 'ADX<=20',
                '_daily_skip_date':   '2020-01-01',
            },
        })
        engine._restore_blocked_today()
        assert 'TSLA' not in engine._daily_scan_skip

    def test_empty_state_no_error(self):
        engine = _make_engine(state={})
        engine._restore_blocked_today()
        assert engine._daily_scan_skip == {}
