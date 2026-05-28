# VELOCITY TRADING BOT — PROJECT CONTEXT

## Project Purpose
Fully automated equity swing trading bot using the **Alpaca** brokerage API.
Optimized for **small cash accounts** with T+1 settlement. Not a margin/futures system.

## Environment

- **Python**: venv at `venv/` uses Python 3.13 (symlinked to current snap release). Run via `venv/bin/python`.
- **Broker**: Alpaca (`alpaca-trade-api` / `alpaca-py`). Do NOT use IB/ib_async imports.
- **Run tests**: `venv/bin/python -m pytest tests/ -q`  *(352+ tests — all must pass)*
- **Start engine**: `venv/bin/python AutoTrader.py`
- **Run backtest**: `venv/bin/python run_backtest.py`

## Architecture
```
AutoTrader.py          ← entry point (signal handling, restart loop)
src/engine.py          ← core trading engine (VelocityEngine class)
src/config.py          ← all tunable parameters (edit here, not inline)
src/indicators.py      ← technical indicators (ATR, RSI, MA, etc.)
src/scanner.py         ← Alpaca screener (top-gainers + most-actives candidate pool)
backtest/strategy.py   ← offline backtester (yfinance data)
dashboard_server.py    ← web dashboard for monitoring
run_backtest.py        ← CLI entry point for backtesting
tests/                 ← pytest test suite (352+ tests)
```

## Critical Design Decisions (Do NOT "Fix" These)

### goodAfterTime = 10:00 AM ET on TRAIL Orders — CONDITIONAL

TRAIL orders set `goodAfterTime = today 10:00 AM ET` **only when the current time is
before 10:00 AM**. If it is already past 10 AM the field is omitted entirely — a past
`goodAfterTime` causes IB error 201 "Invalid effective time" and the stop is rejected.
This applies to both `_audit_stop_orders` (line ~923) and the entry-flow TRAIL placement.
Since `ENTRY_START = (10,0)`, entries never fire before 10 AM, so the entry-flow TRAIL
never sets `goodAfterTime` in practice. The audit path sets it only on cold starts before
10 AM (e.g., pre-market restart). Do NOT change this back to unconditional.

### Cash Account — No Margin
This system trades a **cash account only**. There is no leverage. Position sizing is
based on settled cash / open slots, not margin. Do not introduce leverage concepts.

### ib_async Package
Always use `from ib_async import ...`. Never suggest or change to `ib_insync`.

### T+1 Settlement Awareness
Exit logic and capital bucketing account for T+1 settlement days. Settled cash
(not total equity) determines available capital for new entries.

### Dynamic Position Slots
`MAX_POSITIONS` is NOT a fixed constant. Maximum simultaneous positions are computed from
total account equity: `floor(NetLiquidation / MIN_BUCKET_SIZE)`, capped at
`MAX_POSITIONS_CAP=8`. New entry slots are then constrained by settled cash:
`floor(SettledCash / MIN_BUCKET_SIZE)`. This lets position capacity compound with equity
while still preventing a cash account from spending unsettled proceeds. Import
`MAX_POSITIONS_CAP` and `MIN_BUCKET_SIZE` from config — not `MAX_POSITIONS`.

### Break-Even Floor: Dual Enforcement
The 4% break-even floor (`BREAK_EVEN_PCT`) is enforced in two places:
1. **Dashboard / state tracking** (`_update_position_prices`): `effective_stop` floored at entry.
2. **Programmatic exit** (`check_velocity_exits`): if price retraces below entry after peak
   reached break-even threshold, the engine exits immediately via `liquidate()`.
The IBKR TRAIL order itself is NOT modified — it can fire below entry on wide-ATR stocks.
The software check in step 2 closes this gap entirely.

### Live Market Data (MARKET_DATA_TYPE=1)
The engine uses live streaming data via the **US Equity and Options Add-On Streaming Bundle**.
Real-time bid/ask is available, so the spread filter (`SPREAD_MAX_PCT=0.5%`) is active on every
candidate. RVOL uses the full elapsed-time normalizer (no delay subtraction). The adaptive
limit-price buffer (`LIMIT_BUF_MIN_PCT=0.2%` – `LIMIT_BUF_MAX_PCT=1.5%`) is an ATR-scaled
slippage buffer above the live ask, not a delay-compensation estimate.

### Spread Filter with Live Data
With live streaming, real-time bid/ask is populated. `spread_pct` is computed from the actual
bid/ask spread and filtered against `SPREAD_MAX_PCT (0.5%)`. If bid/ask is momentarily
unavailable in a snapshot (pre-market, illiquid tick), `spread_pct` falls back to `0.0` and
the entry limit order provides cost protection. The scoring liquidity component awards full
points only when spread data is valid.

### Combined Candidate Pool (Scanner)

`src/scanner.py` builds the candidate universe each scan cycle from two Alpaca endpoints:

1. `ScreenerClient.get_market_movers` → top-gainers by intraday % change, filtered to
   `≥ SCAN_MIN_GAIN_PCT (2.0%)`.  These are the primary momentum signal.
2. `StockHistoricalDataClient.get_stock_most_actives` → top stocks by intraday volume.
   Supplements gainers with high-volume movers that may not yet show a large % gain.

The two lists are merged (gainers first, then actives) with duplicates removed.
`get_candidates(data_client, screener_client)` is the single entry point; the engine calls
it as `self.get_institutional_scan()`.  `ScreenerClient` is instantiated separately from
`StockHistoricalDataClient` and stored on the engine as `self.screener_client`.

Do NOT revert to a single most-actives source — the combined pool catches more valid setups.
Do NOT call `get_most_actives` or `get_top_gainers` directly from engine code; use `get_candidates`.

### Price Floor Filter

`get_technical_context` returns `None` (skip) if `live_price < SCAN_MIN_PRICE ($20)`.
This prevents sub-$20 stocks from ever reaching the 12-rule screener or consuming a slot.
The check happens after the live snapshot is fetched, so the price used is real-time, not
yesterday's close. `SCAN_MIN_PRICE` is imported from `src/config.py`.

### VIX: Day-Level Caching and Graceful Degradation

`_fetch_vix` caches the fetched value for the remainder of the trading day (keyed on
`_vix_cache_date`).  On the first miss, `_last_vix` (last-known value) is returned with a
warning rather than blocking entries.  Only when no value has ever been fetched is `None`
returned; in that case `run_cycle` logs a warning and continues without the VIX filter —
it does NOT halt entries.  Do NOT revert to halt-on-None behavior.

### Sector Data via yfinance

`_get_sector` uses `yf.Ticker(symbol).info['sector']` to retrieve real GICS sector strings
(e.g. `'Technology'`).  Results are cached for the session in `self._sector_cache`.
`MAX_SECTOR_COUNT=2` limits simultaneous positions per sector.  Do NOT revert to Alpaca
`asset_class='us_equity'` — it returns the same string for every stock and breaks the
sector diversification filter entirely.

### `_tod_frac` — Cumulative Volume CDF

`_tod_frac(elapsed_min)` returns the cumulative fraction of daily volume expected to have
traded by `elapsed_min` minutes into the session.  It is a piecewise-linear CDF, NOT a rate:

```text
elapsed ≤ 30 min  →  f = max(0.01, elapsed / 30 × 0.22)   (22% by 30 min)
elapsed > 30 min  →  f = 0.22 + (elapsed − 30) / 360 × 0.78  (100% by 390 min)
```

RVOL is computed as `intraday_vol / avg_20d_vol / tod_frac` (higher early, lower late).
Do NOT invert this to a rate function — that would make RVOL monotonically increase toward
close and never clear the `RVOL_MIN` threshold early in the session.

### `_calc_max_positions` Returns 0 for Insufficient Equity

`_calc_max_positions(equity)` returns `0` (not `1`) when `equity < MIN_BUCKET_SIZE`.
A cash account below the $500 bucket floor cannot safely size even one position. Tests
`test_engine_returns_zero_not_one_when_insufficient` and `test_matches_engine_calc_max_positions`
validate this. Do NOT use `max(1, int(equity / MIN_BUCKET_SIZE))`.

### SPY Regime: Slope Check Required (Live Engine Matches Backtest)

`_fetch_spy_trend()` checks three conditions: `SPY close > SMA50 > SMA200` **AND**
`SMA200 slope > 0` (measured over `SMA200_SLOPE_LOOKBACK=5` days). The slope check blocks
entries during recovery rallies where price has crossed above a still-falling SMA200 — the
highest-false-breakout window in bear-market bounces. The backtest `_download_regime_data`
already enforced this; the live engine was missing it. Both must remain in sync.
Do NOT remove the slope check from either the live engine or the backtest.

### Backtest Scoring Formula Aligned with Live `_score_candidate`

`_daily_scan` scoring uses the same structural formulas as live `_score_candidate`:

- **RVOL pts**: `min(25.0, (rvol - self._rvol_min) / (5.0 - self._rvol_min) * 25.0)` — linear
  scale from threshold to cap of 5×. Previously used `/ self._rvol_min` (relative rate), which
  diverged from the live formula.
- **RSI quality pts**: `min(10.0, max(0.0, (rsi - RSI_THRESHOLD) / 20.0 * 10.0))` — continuous
  linear scale from 55 (0 pts) to 75 (10 pts). Previously a stepwise function (10→5→0 for
  ≤70/≤75/>75) that rewarded low RSI differently than the live engine.
- **Trend pts (30 max)**: MA separation (0-22 pts) + ADX quality (0-8 pts). MA sep formula:
  `min(22.0, (ma50-ma200)/ma200 / 0.06 * 22.0)`. ADX quality: `min(8.0, max(0.0, (adx-25.0)/25.0*8.0))`
  — activates above ADX=25 (not the entry gate of 20). Same formula in both live and backtest.

Do NOT revert any of these formulas independently — they must match `_score_candidate` exactly.

### Minimum Composite Score Gate (`SCAN_MIN_SCORE`)

After `_score_candidate` runs, the engine skips any candidate whose score is below
`SCAN_MIN_SCORE = 30.0` (out of 100). This prevents low-conviction entries that pass all
12 binary rules but score weakly on trend strength, RVOL quality, and RSI momentum. The
gate is applied in `run_cycle` before appending to `signals`:

```python
score = self._score_candidate(ctx)
if score < SCAN_MIN_SCORE:
    logger.debug(f"SCAN {sym}: SKIP — score={score:.1f} < min {SCAN_MIN_SCORE:.0f}")
    continue
signals.append((score, sym, ctx))
```

`SCAN_MIN_SCORE` is defined in `src/config.py`. Do NOT hardcode the threshold inline.

### Backtest Constants Must Match Config (No Hardcoded Threshold Values)

`_entry_signal` in `backtest/strategy.py` must reference config constants:

- `adx_val > ADX_THRESHOLD` (not `adx_val > 20`)
- `h200_val * HIGH200_MIN_PCT` (not `h200_val * 0.85`)

These constants are imported from `src.config`. If threshold values are changed in config,
both live and backtest filters change automatically. Hardcoding creates silent divergence.
Do NOT hardcode numeric thresholds that have named constants in `src/config.py`.

## Strategy Summary (12-Rule Filter)
**Entry** (11 stock rules + 1 market regime):

- SPY uptrend (SPY close > SMA50 > SMA200 **AND** SMA200 slope > 0 over last 5 days)
- price > MA50 > MA200 with MA50/MA200 separation ≥ 3%
- ADX(14) > 20 — confirms trend has real momentum
- Close ≥ 85% of 200-day rolling high — leadership/momentum
- RVOL ≥ 2.5× intraday (empirical U-shaped time-of-day normalizer)
- Bid-ask spread ≤ 0.5%
- 20-day avg dollar volume ≥ threshold (Friday: 2× threshold)
- ORB breakout: live price > 15-min opening range high
- Gap limit: price ≤ ORB high × 1.10 (no chasing)
- RSI(14) > 55 AND RSI rising ≥ 1 point (acceleration, not exhaustion)

**Exit**: Chandelier trailing stop (ATR22 × 2.0) + 7% hard stop + break-even floor
at 4% profit (programmatically enforced) + velocity exit (< 5% profit after 2 trading
days) + Friday close rule

**Universe**: NASDAQ Global Select + NYSE, price > $20, avg dollar vol > $100M/day

## Risk Parameters
- 2% equity risk per trade (ATR-based position sizing)
- Dynamic capacity: `floor(equity / $500)` capped at 8; entries also require settled cash
- Daily loss circuit breaker: 3% intraday drawdown halts new entries
- Portfolio concentration: warn at 85% deployed, halt at 95%
- VIX threshold: 35 (blocks new entries above)

## Running Tests

```bash
venv/bin/python -m pytest tests/ -q              # all tests
venv/bin/python -m pytest tests/test_engine.py   # engine only
```

All 352+ tests should pass. If any fail, fix before proceeding.

## Key Files to Know

- `src/config.py` — all parameters live here; never hardcode values inline
- `src/engine.py` — `VelocityEngine` class; `run_cycle()` is the main loop
- `src/scanner.py` — `get_candidates(data_client, screener_client)` — combined candidate pool
- `tests/test_engine.py` — includes `TestPortfolioConcentration`, `TestBreakEvenExitEnforcement`
- `tests/test_trailing_stop_scoring_screener.py` — trailing stop, scoring, and screener tests
- `logs/trading_engine.log` — rotating daily log (30 days retention)

## What NOT to Do

- Do NOT introduce IB/ib_async/ib_insync imports — the broker is Alpaca
- Do NOT remove goodAfterTime from TRAIL orders
- Do NOT introduce margin or leverage concepts
- Do NOT hardcode values that belong in `src/config.py`
- Do NOT skip tests — always run full suite after changes
- Do NOT import `MAX_POSITIONS`, `CAPITAL_SEED`, `SCAN_COUNT`, or `COMMISSION_PER_ORDER`
  from config. Live capital and commissions come from Alpaca; scanner results are not capped
  by a config constant.
- Do NOT modify the break-even floor to use order modification — the software exit in
  `check_velocity_exits` is the correct, tested enforcement mechanism
- Do NOT cancel TRAIL SELL orders inside `liquidate()` — only non-TRAIL orders are cancelled
  before placing the market sell; the TRAIL stays active as last-resort protection if the
  market sell is rejected
- Do NOT restore the unconditional `del self.state[symbol]` in `liquidate()` — state is now
  marked `pending_exit=True` and deleted by `_sync_positions_from_alpaca` once Alpaca confirms
  the position is gone; this prevents the entry timestamp being reset (which broke hold-time
  accounting and the velocity-exit clock)
- Do NOT revert `get_candidates` back to `get_most_actives` — the combined gainers+actives pool
  is intentional and catches more high-conviction setups
- Do NOT revert `_calc_max_positions` to `max(1, int(equity / MIN_BUCKET_SIZE))` — it must
  return 0 when equity is below the bucket floor
- Do NOT revert `_tod_frac` to a rate function — it is a cumulative volume CDF; the RVOL
  formula depends on it being cumulative
- Do NOT revert VIX None-handling to halt-entries — warn and continue is the correct behavior

## Fixed Bugs (reference — do not re-introduce)

### TRAIL Stop `goodAfterTime` Past-Time Rejection (May 2026)

`_audit_stop_orders` previously set `goodAfterTime = today 10:00 AM ET` unconditionally.
After a restart at e.g. 12:35 PM the time was already in the past, causing IB error 201
"Invalid effective time" for every new TRAIL. Fix: conditional — only set when `now < 10 AM`.

### `liquidate()` Cancelled TRAIL Before Sell Was Confirmed (May 2026)

`liquidate()` cancelled **all** open orders for a symbol (including the TRAIL SELL
protective stop) before placing the market sell. If the market sell was then rejected
asynchronously by IB, the position was left with no TRAIL protection.
Fix: only cancel non-TRAIL orders (`t.order.orderType != 'TRAIL'`).

### Hold Time Reset on Failed Sell / Re-sync (May 2026)

When `liquidate()` placed a market sell that IB rejected asynchronously, the call returned
without raising (no exception), so `del self.state[symbol]` executed. On the next cycle
`_sync_positions_from_ibkr` found the symbol in IBKR but not in state, re-added it with
`'time': datetime.now()`, resetting the entry timestamp. This corrupted `_count_trading_days`
(velocity-exit clock showed 0 days held) and the dashboard hold-time display.
Fix: `liquidate()` sets `pending_exit=True` instead of deleting; sync clears the flag if
the sell was rejected (position still open) or deletes state if position is gone.

### unit_price and score Lost on Restart (May 2026)

`_sync_positions_from_ibkr` "not in state" path (for positions that exist in IBKR but
not in local state — e.g. after a crash) created state entries without `fill_price`,
`peak_price`, or `score`. Dashboard showed "pending" for unit_price and score=None
permanently.
Fix:

- `engine.py`: added `'fill_price': round(avg_cost, 2)` and `'peak_price': round(avg_cost, 2)`
  to the re-sync dict. IBKR's `avgCost` already includes commission in the cost basis, so no
  separate `commission` field is needed — `fill_price` IS the all-in unit price.
- `dashboard_server.py`: `unit_price` now shows `fill_price` directly when `commission` is
  absent but `fill_price` is set (re-synced case). Only falls back to `None` ("pending") when
  neither `fill_price` nor `commission` is available. `score` remains `None` after re-sync —
  it cannot be recovered once state is lost.

### Market SELL Orders Rejected by Direct-Routing Precautionary Block — Error 10311 → 201 (May 2026)

`liquidate()` placed `MarketOrder` on `sym_pos.contract` which carries the stock's native
exchange (NASDAQ/NYSE). When IB Gateway's Precautionary Setting blocks direct-routed orders,
IB emits error 10311 then error 201 "Order discarded" — every sell (velocity exit, Friday
close, hard stop) silently fails. Positions never close.
Fix: copy the contract and set `exchange='SMART'` before placing the sell:
```python
sell_contract = copy.copy(sym_pos.contract)
sell_contract.exchange = 'SMART'
self.ib.placeOrder(sell_contract, sell_order)
```
TRAIL orders are unaffected — they use `_qualify_and_cache()` which already returns a SMART-
routed contract.

### `_preflight_order()` Crashed on List Return from `whatIfOrder()` (May 2026)

`ib.whatIfOrder()` occasionally returns a `list` of OrderState objects instead of a single
OrderState (API version-dependent). The code did `state.warningText` directly, raising
`AttributeError: 'list' object has no attribute 'warningText'`. Caught by the except block
(fail-open for stops), but preflight warnings were silently lost.
Fix: unwrap list before accessing `warningText`:
```python
if isinstance(state, list):
    state = state[0] if state else None
if state is None:
    raise ValueError("whatIfOrder returned empty result")
```

### VIX Historical Fallback Returned No Bars Before Market Open (May 2026)

`reqHistoricalData` for VIX used `useRTH=True` and `durationStr='3 D'`. During or before
the market open session boundary, this could return 0 bars, leaving `vix_price=NaN` with no
cached value — blocking ALL entries for the entire session.
Fix: changed to `useRTH=False` (fetch any available bar regardless of session) and
`durationStr='5 D'` (larger window to ensure at least one bar is always available). Also
added an explicit log when `bars` is empty to aid future diagnosis.

### Time-Sensitive Tests Failed on Friday After 3 PM (May 2026)

Five tests in `TestVelocityExit` and `TestBreakEvenExitEnforcement` used real `datetime.now()`
without mocking. When run on a Friday after 3 PM ET, `is_friday_close=True` triggered the
Friday close rule on positions with < 3% profit, causing false "liquidate was called" failures.
Fix: all affected tests now mock `src.engine.datetime` to a fixed Wednesday 10:30 AM, making
them deterministic regardless of when they run.

### `check_velocity_exits()` Placed Duplicate MKT SELL When `pending_exit=True` (May 2026)

`check_velocity_exits()` skipped positions with `pending=True` (buy pending) but NOT positions
with `pending_exit=True` (sell already submitted). On the next cycle after `liquidate()` was
called, if the MKT SELL hadn't filled yet, the position was still in IBKR and still in state
with `pending_exit=True`. `check_velocity_exits()` would re-evaluate the exit conditions, call
`liquidate()` again, cancel the in-flight sell (as a non-TRAIL order), and place a second MKT
SELL — risking a duplicate fill or a short position.
Fix: added `if data.get('pending_exit'): continue` guard immediately after the `pending` check
in `check_velocity_exits()`. Added `test_pending_exit_blocks_duplicate_sell_on_next_cycle`.

### `liquidate()` GTC Sell Set `goodAfterTime` to Past Time Post-Market (May 2026)

`liquidate()` unconditionally set `goodAfterTime = today 10:00 AM ET` on GTC sell orders.
When called post-market (e.g., Friday close rule fires at 3 PM ET), the time was already in
the past, causing IB error 201 "Invalid effective time" — the sell was silently rejected and
the position was left open over the weekend without any forced close.
Fix: wrapped `goodAfterTime` assignment in `if now_et < ten_am_today:` (same conditional
pattern already used in `_audit_stop_orders` and the entry-flow TRAIL placement). Post-market
GTC orders now omit `goodAfterTime` — IBKR routes them at the next RTH open without error.

### `run()` Exception Handler Swallowed Stack Traces (May 2026)

`run()`'s inner exception handler used `logger.error(f"RUNTIME ERROR: {e}")`. This logged
only the exception message, discarding the full traceback. Silent, hard-to-diagnose production
crashes resulted — only the exception type and message were visible in the log.
Fix: changed to `logger.exception("RUNTIME ERROR")` which automatically appends the full
traceback to the log entry.

## Survivorship Bias Warning (Backtest)
The backtest universe is current NASDAQ/NYSE listings. Bankrupt/delisted tickers from the
backtest window are absent. Momentum/breakout strategies are particularly sensitive to this
bias — reported returns may be overstated by 10-30%. Treat backtest results as directional
signal-quality indicators, not as forecasts of live performance.
