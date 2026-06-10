# VELOCITY TRADING BOT — PROJECT CONTEXT

## Project Purpose
Fully automated equity swing trading bot using the **Alpaca** brokerage API.
Optimized for **small cash accounts** with T+1 settlement. Not a margin/futures system.

## Workflow

After **every set of code edits**, run the full test suite and commit to git:

```bash
venv/bin/python -m pytest tests/ -q   # all tests must pass
git add -p                             # stage changed files selectively
git commit -m "descriptive message"
```

Never commit `.env` files, credentials, or generated artifacts (`backtest/optimizer_results*.csv`).

## Live Deployment Links

- **Dashboard**:    [alpacatradingbot-pe9m.onrender.com](https://alpacatradingbot-pe9m.onrender.com)
- **Health check**: [/health](https://alpacatradingbot-pe9m.onrender.com/health)
- **API state**:    [/api/state](https://alpacatradingbot-pe9m.onrender.com/api/state)
- **Live logs**:    [/api/logs](https://alpacatradingbot-pe9m.onrender.com/api/logs)
- **GitHub repo**:  [hb87140/AlpacaTradingBot](https://github.com/hb87140/AlpacaTradingBot)
- **Render**:       [dashboard.render.com](https://dashboard.render.com)
- **UptimeRobot**:  [dashboard.uptimerobot.com/monitors](https://dashboard.uptimerobot.com/monitors)

## Environment

- **Python**: venv at `venv/` uses Python 3.13 (symlinked to current snap release). Run via `venv/bin/python`.
- **Broker**: Alpaca (`alpaca-trade-api` / `alpaca-py`). Do NOT use IB/ib_async imports.
- **Run tests**: `venv/bin/python -m pytest tests/ -q`  *(384+ tests — all must pass)*
- **Start engine**: `venv/bin/python alpaca_auto_trader.py`
- **Run backtest**: `venv/bin/python run_backtest.py`

## Architecture
```
alpaca_auto_trader.py  ← entry point (signal handling, restart loop)
src/engine.py          ← core trading engine (VelocityEngine class)
src/config.py          ← all tunable parameters (edit here, not inline)
src/indicators.py      ← technical indicators (ATR, RSI, MA, etc.)
src/scanner.py         ← Alpaca screener (top-gainers + most-actives candidate pool)
backtest/strategy.py   ← offline backtester (yfinance data)
alpaca_dashboard.py    ← web dashboard for monitoring
run_backtest.py        ← CLI entry point for backtesting
tests/                 ← pytest test suite (384+ tests)
```

## Critical Design Decisions (Do NOT "Fix" These)

### Cash Account — No Margin
This system trades a **cash account only**. There is no leverage. Position sizing is
based on settled cash / open slots, not margin. Do not introduce leverage concepts.

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

### Backtest Strategy: Alligator Swing (Mirrors Live Engine)

`backtest/strategy.py` implements the **same** Alligator Swing strategy as the live engine.
Do NOT diverge entry logic, scoring formulas, or exit rules between them.

**Entry rules** (`_daily_scan` + `_entry_signal`):
1. Price ≥ `SCAN_MIN_PRICE`, volume ≥ `SCAN_MIN_VOLUME`, dollar vol ≥ `SCAN_MIN_DOLLAR_VOL`
2. Alligator bullish: `SMMA_FAST_ALIGNED > SMMA_SLOW_ALIGNED AND SMMA_MED_ALIGNED > SMMA_SLOW_ALIGNED`
3. Fresh crossover: `ALLIGATOR_CROSSED = True` (non-bullish bar within `ALLIGATOR_CROSS_LOOKBACK`)
4. RSI ≥ 50 AND RSI delta ≥ `RSI_MIN_DELTA`
5. RVOL ≥ `BACKTEST_RVOL_MIN` (1.2× daily close proxy)
6. Day strength: close ≥ open × (1 + `DAY_STRENGTH_OPEN_PCT`) AND close in upper half of range

**Scoring** (`_daily_scan` composite score, 0-100, gate = `SCAN_MIN_SCORE=20`):
- Alligator alignment: `min(30, (fast_aligned - slow_aligned) / slow_aligned / 0.06 * 30)` — wider SMMA mouth = stronger trend
- RVOL: `min(25, (rvol - rvol_min) / (5.0 - rvol_min) * 25)` — linear scale to 5× cap
- RSI delta: `min(25, max(0, rsi_delta / 5.0 * 25))` — mirrors live `_score_candidate`
- Liquidity: dollar-vol half (0-10); spread half unavailable in daily OHLCV

**Exit rules** (mirrors `check_velocity_exits` in live engine):
- Chandelier trailing stop: `peak_high - ATR_CHAND × CHANDELIER_MULT` (fixed dollar distance = Alpaca `trail_price`)
- Hard stop: `entry × (1 - HARD_STOP_PCT)`
- Break-even floor: once profit ≥ `BREAK_EVEN_PCT`, stop ≥ entry
- Alligator reversal: both fast+med SMMA cross below slow at day-end → exit at next-day open
- **No velocity time-exit, no forced Friday close** — these were removed in May 2026 to match the live engine

**Offset-adjusted SMMA** computed in `_apply_indicators`:
```python
df['SMMA_FAST_ALIGNED'] = df['SMMA_FAST'].shift(ALLIGATOR_FAST_OFFSET)   # 3 bars
df['SMMA_MED_ALIGNED']  = df['SMMA_MED'].shift(ALLIGATOR_MED_OFFSET)     # 5 bars
df['SMMA_SLOW_ALIGNED'] = df['SMMA_SLOW'].shift(ALLIGATOR_SLOW_OFFSET)   # 8 bars
```
`apply_all()` already produces `SMMA_FAST/MED/SLOW`; `_apply_indicators` only adds the offsets.

### Minimum Composite Score Gate (`SCAN_MIN_SCORE`)

After `_score_candidate` runs, the engine skips any candidate whose score is below
`SCAN_MIN_SCORE = 20.0` (out of 100). The gate is applied in `run_cycle` before appending to
`signals`, and in `_daily_scan` before appending to `scored`.

`SCAN_MIN_SCORE` is defined in `src/config.py`. Do NOT hardcode the threshold inline.

### Backtest Constants Must Match Config (No Hardcoded Threshold Values)

`_entry_signal` in `backtest/strategy.py` must reference config constants — never hardcode
values that exist in `src/config.py`. If threshold values change in config, both live and
backtest filters change automatically. Hardcoding creates silent divergence.

## Strategy Summary (Alligator Swing — 6 Intraday Rules)
**Entry** (6 intraday cycle rules via `src/rules.py CYCLE_RULES`):

- Spread ≤ 0.5% bid-ask (`check_spread`)
- Volume ≥ `SCAN_MIN_VOLUME` 20-day avg shares (`check_volume`, 2× on Fridays)
- Alligator bullish: offset-adjusted SMMA fast > slow AND med > slow, fresh crossover within `ALLIGATOR_CROSS_LOOKBACK` bars (`check_alligator_bullish`)
- RSI(14) ≥ 50 AND RSI rising ≥ `RSI_MIN_DELTA` pts (`check_rsi_trend`)
- RVOL ≥ `RVOL_MIN` (1.2×) intraday (`check_rvol`)
- Day strength: live price ≥ today's open × (1 + `DAY_STRENGTH_OPEN_PCT` 0.5%) (`check_day_strength`)

Plus 1 permanent daily rule:
- Dollar volume: 20-day avg ≥ `SCAN_MIN_DOLLAR_VOL` $5M (`check_dollar_vol`, 2× on Fridays)

**Exit**: Chandelier trailing stop (ATR14 × 2.5, dollar-distance = Alpaca `trail_price`) +
5% hard stop + break-even floor at 6% profit (programmatically enforced) +
Alligator reversal exit (both fast+med SMMA cross below slow)

**Universe**: NASDAQ Global Select + NYSE, price > $1, 20-day avg dollar vol > $5M

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

All 374+ tests should pass. If any fail, fix before proceeding.

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
- `liquidate()` MUST cancel ALL open orders for a symbol (including the TRAIL) before placing
  the market sell. Alpaca holds shares "for orders" — the GTC TRAIL reserves the full qty so
  a market SELL is rejected with `available=0` while the TRAIL is open. If the market sell
  fails, `_audit_stop_orders` re-places the TRAIL on the next cycle via `has_unprotected`.
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
- `alpaca_dashboard.py`: `unit_price` now shows `fill_price` directly when `commission` is
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

## Fixed Bugs (Session 15 — May 2026)

### 57. Backtest RSI Acceleration Formula 2× Divergence from Live Engine

`_daily_scan` in `backtest/strategy.py` computed RSI acceleration as `rsi_delta * 1.5`,
while the live `_score_candidate` uses `rsi_delta / 5.0 * 15.0` (= `rsi_delta * 3.0`).
At the same RSI delta, the backtest awarded half the points the live engine would, causing
the two to rank candidates differently. A stock with a 5-point RSI jump scored 7.5 in the
backtest but 15.0 in live — a 2× gap on the momentum component (25 pts max total).
Fix: changed backtest formula to `min(max(rsi_delta / 5.0 * 15.0, 0.0), 15.0)`.
Both live engine and backtest now use the same RSI acceleration scale.

### 58. Backtest Chandelier Stop Used Percentage Tracking Instead of Dollar Distance

The Alpaca `TrailingStopOrderRequest` uses `trail_price` — a **fixed dollar amount** below
the peak. As the stock rises, the stop rises by the same dollar amount, keeping a constant
dollar gap. The backtest had switched to percentage-of-peak tracking (`chand_stop = peak ×
(1 − pct)`), which widens the dollar gap as the stock rises (stop falls further below peak
in absolute terms). This made the backtest stops looser than live ones on big winners —
overstating performance by letting winning trades run longer before stopping out than Alpaca
would actually allow.
Fix: reverted to dollar tracking (`chand_stop = peak_high − chand_dist`) where `chand_dist`
is fixed at entry. Stored as `t.__dict__['_chand_dist']` (replacing the removed
`_chandelier_pct` key). Now matches Alpaca `trail_price` semantics exactly.

### 59. Dashboard P&L Panel Permanently Broken (Key Mismatch)

The engine wrote equity snapshots with key `"equity"` to `equity_history.json`, but
`alpaca_dashboard.py` read them with key `"eq"`. Every call to `_find_base()` raised
`KeyError: 'eq'`, causing the `/api/state` endpoint to 500 and the P&L panel to show
`—` for Daily / Weekly / Monthly / Overall. The equity chart still worked (it reads the
raw JSON directly, not through `_pnl()`).
Fix: changed `alpaca_dashboard.py` lines 102 and 114 from `e["eq"]` to `e["equity"]`.
Also updated the comment at line 114 from "first real IBKR reading" to "first Alpaca
account reading".

### 60. Dashboard Showed "IB GATEWAY" / "IBKR" / "IB scan" / "IB raises" (Wrong Broker)

Multiple strings in `alpaca_dashboard.py` still referenced the old Interactive Brokers
broker that was replaced with Alpaca in an earlier session:

- Status panel header: "IB GATEWAY" → "ALPACA API"
- Comment: "IBKR accountSummary API" → "Alpaca account API"
- Comment: "Re-synced from IBKR" → "Re-synced from Alpaca"
- Entry condition 4: "IB scan:" → "Alpaca scan:"
- Entry condition 10: "All IBKR scanner results" → "All Alpaca scanner results"
- Exit condition 1: "IB raises stop automatically" → "Alpaca raises stop automatically"

Fix: all six occurrences updated to Alpaca terminology.

### 61. Inline Import `compute_ma` Inside `_fetch_spy_trend` Method

`_fetch_spy_trend` in `src/engine.py` contained `from src.indicators import compute_ma`
as an in-method import. Python caches imports, so there is no runtime cost, but it hides
the module dependency, breaks linters, and misleads readers about the module's imports.
Fix: added `compute_ma` to the module-level import line `from src.indicators import
apply_all, compute_ma`. The inline import line inside the method is removed.

### 62. Optimizer CSV Result Files Committed as Source Code

`backtest/optimizer_results.csv` and `backtest/optimizer_results_partial.csv` are generated
output artifacts produced by `scripts/run_optimizer.py`. They were committed to git as if
they were source code, adding 2049 lines each of generated data to the repository.
Fix: added `backtest/optimizer_results*.csv` to `.gitignore` so future optimizer runs do
not pollute the commit history.

## Fixed Bugs (Session 16 — May 2026)

### 63. Dashboard Equity Chart Always Blank (Stale JS Key `e.eq`)

`alpaca_dashboard.py` JS `refreshChart()` function mapped history entries with
`hist.map(e => e.eq)`. The engine writes `{"ts": ..., "equity": ...}` to
`equity_history.json`, so `e.eq` was always `undefined` — every data point was
`undefined`, rendering an empty chart.
Session 15 fixed the Python `_pnl()` reads (`e["equity"]`) but missed this JS line.
Fix: changed to `hist.map(e => e.equity)`.
Test: `TestDashboardEquityChartKey.test_dashboard_js_uses_equity_key_not_eq`

### 64. Backtest `_daily_scan()` Missing `SCAN_MIN_SCORE` Gate

`_daily_scan()` computed composite scores and sorted by them, but never filtered
out candidates below `SCAN_MIN_SCORE=30`. Low-conviction stocks scoring 1–29 (passing
all 12 binary rules but weak on trend/RVOL/RSI) were returned to `_run_loop()` and
could be entered. The live engine applies `if score < SCAN_MIN_SCORE: continue` in
`run_cycle()` — backtest was silent-diverging on every such candidate.
Fix: added `SCAN_MIN_SCORE` to the `src.config` import and inserted
`if score < SCAN_MIN_SCORE: continue` inside `_daily_scan()` before `scored.append()`.
Tests: `TestDailyScanMinScore.test_low_score_candidate_excluded_from_scan_output`,
`test_scan_min_score_constant_imported_in_backtest`

### 65. Backtest Bucket Calculation Missing `BUCKET_CASH_PCT` Reserve

`_run_loop()` sized position buckets as `bucket = settled_cash / entry_slots`.
The live engine applies a 10% cash reserve: `bucket_size = settled_cash * BUCKET_CASH_PCT / open_slots`
(`BUCKET_CASH_PCT=0.90`). The backtest was deploying ~11.1% more per position than
live allows, and could enter trades when cash was between `MIN_BUCKET_SIZE` and
`MIN_BUCKET_SIZE / BUCKET_CASH_PCT` — slots the live engine would reject.
Fix: added `BUCKET_CASH_PCT` to the `src.config` import and changed to
`bucket = settled_cash * BUCKET_CASH_PCT / entry_slots`.
Tests: `TestBacktestBucketCashPct.test_bucket_cash_pct_constant_imported_in_backtest`,
`test_position_qty_reflects_bucket_cash_pct`

## Fixed Bugs (Session 17 — May 2026)

### 66. Dashboard `effective_stop` Dead Code + Stale IB Comment

`alpaca_dashboard.py` `get_state()` read `d.get("effective_stop", sl)` with
a comment "IB trail watermark if tracked". The `effective_stop` key is **never written**
to `engine_state.json` — the engine writes the break-even-floored chandelier stop value
directly to `stop_loss` via `_update_position_prices()`. The fallback always resolved to
`sl`, making the `get()` call dead code. Additionally, `effective_stop == stop_loss` in
the API response meant the `↑` break-even indicator on the dashboard table (JS line 748)
could never fire.
Fix: replaced with `effective_sl = sl` with a comment explaining why, and removed the
stale IB reference from the comment.

### 67. Stale "replaces the IB ScannerSubscription" Reference in scanner.py

`src/scanner.py` module docstring opened with
"Alpaca scanner — replaces the IB ScannerSubscription." — a historical migration note
leftover from the IB-to-Alpaca rewrite. This is the last remaining IB reference outside
of test docstrings.
Fix: updated to "Alpaca candidate scanner."

## Fixed Bugs (Session 19 — May 2026)

### 68. `_bar_cache` Never Cleared on Day Rollover — Memory Growth + Stale ORB Data

`run_cycle()` day-rollover block cleared `_daily_scan_skip` and `_insufficient_history_skip`
but not `_bar_cache`. The bar cache stores daily OHLCV bars and ORB highs keyed by symbol.
Without clearing on rollover, stale yesterday bars persisted in memory indefinitely, and any
symbol not re-fetched on the new day would still hold yesterday's ORB high — a stale breakout
level that could silently pass or block today's ORB filter incorrectly.
Fix: added `self._bar_cache.clear()` to the day-rollover block alongside the other cache clears.
Tests: `TestBarCacheClearedOnDayRollover.test_bar_cache_cleared_on_new_day`,
`test_bar_cache_not_cleared_same_day`

### 69. Dead `comm` Variable in `_write_dashboard_data()`

Line 232 of `src/engine.py` computed `comm = d.get('commission', None)` immediately after
computing `eff_stop`. `comm` was then never referenced — not in the output dict, not in any
conditional, nowhere. Alpaca is commission-free and the engine never writes a `commission`
key to state, so `d.get('commission', None)` always returned `None`. Pure dead code.
Fix: removed the line entirely.
Tests: `TestWriteDashboardDataNoDeadCommKey.test_commission_key_absent_from_output`

### 70. Risk Management Gap — Software Exits Not Checked During Multi-Signal Entry Polling

`check_velocity_exits()` ran once per cycle (step 5), before the entry loop. Each BUY
attempt then blocked up to 30 seconds polling for fill confirmation. With 2-3 signals in
one cycle, existing positions could go 60-90 seconds without software hard-stop or
break-even checks — the Alpaca trailing stop is the GTC backstop, but the 7% intraday
software hard stop would not fire during this window.
Fix: added `if placed > 0: self.check_velocity_exits()` at the top of the entry loop body
(after the `placed >= open_slots` break check). This re-runs software exits before each
second and subsequent entry attempt, capping the monitoring gap at one fill poll (≤30 s)
regardless of how many signals are queued.
Tests: `TestExitRecheckBetweenEntries.test_check_velocity_exits_called_between_signals`

### Minor. `_log_startup_summary()` Used `price` Instead of `fill_price`

Line 362 of `src/engine.py` read `ep = float(d.get('price', 0))`. Everywhere else in the
codebase (e.g., `_write_dashboard_data`) the pattern is
`float(d.get('fill_price') or d.get('price', 0))` — preferring the confirmed fill price
over the pending limit price. The startup log was showing limit prices instead of actual
fill prices when a position was loaded from persisted state after a restart.
Fix: changed to `ep = float(d.get('fill_price') or d.get('price', 0))`.
Tests: `TestLogStartupSummaryUsesFillPrice.test_uses_fill_price_when_present`,
`test_falls_back_to_price_when_no_fill_price`

## Fixed Bugs (Session 23 — May 2026)

### 78. Hardcoded Risk Thresholds Should Be Config Constants

Three inline numeric literals were used in `src/engine.py` that belong in `src/config.py`:

- `pct >= 0.85` and `pct >= 0.95` in `_check_portfolio_concentration()` — the portfolio
  concentration warn/halt thresholds. A trader adjusting risk tolerance should change these
  in `src/config.py`, not hunt through the engine.
- `if drift > 0.02` in the entry-loop re-price gate — the 2% price-drift threshold that
  skips an entry if the live price has moved too far since the scan snapshot.

Fix: added `CONCENTRATION_WARN_PCT = 0.85`, `CONCENTRATION_HALT_PCT = 0.95`, and
`REPRICE_DRIFT_MAX_PCT = 0.02` to `src/config.py`. Imported and replaced all three
hardcoded literals in `src/engine.py`.
No new tests needed — existing concentration tests still validate behavior at the correct
threshold values (unchanged).

### 77. Hardcoded "Paper Trading" in `_initialize()` and `run()` Log Messages

`_initialize()` logged `"MARKET DATA: Alpaca paper trading."` and `run()` logged
`"ENGINE DEPLOYED — Alpaca Paper Trading"` regardless of the `ALPACA_PAPER` setting.
When a trader switches to live trading (`ALPACA_PAPER=False`), the logs would misleadingly
say "paper trading" for the entire session — a serious operational confusion.
Fix: both messages now use `"PAPER" if ALPACA_PAPER else "LIVE"` so the log correctly
reflects the active trading mode.

### 76. Dashboard "Last Updated" Always Blank — Key Mismatch `ts` vs `last_updated`

`_write_dashboard_data()` wrote the current timestamp to dashboard_data.json under the
key `'ts'`. `get_state()` in `alpaca_dashboard.py` reads `dash_data.get("last_updated")`.
Due to this key mismatch, `last_updated` was always `None` in the API response, and the
"Last Updated" time shown in the dashboard JS (`if (d.last_updated) { ... }`) was always
blank for the entire lifetime of the system.

Fix: renamed `'ts': now_ny.isoformat()` → `'last_updated': now_ny.isoformat()` in the
`data` dict in `_write_dashboard_data()`.

Tests: `TestWriteDashboardDataNoDeadCommKey.test_writes_last_updated_not_ts`

## Fixed Bugs (Session 22 — May 2026)

### 74. `_check_portfolio_concentration()` Used Entry Price Instead of Current Market Price

`_check_portfolio_concentration()` computed the deployed capital value as:

```python
deployed = sum(
    float(d.get('price', 0)) * float(d.get('qty', 0))
    for d in self.state.values()
)
```

`price` is the entry limit price set at order submission — it never updates. As positions
appreciate, the engine severely underestimates true mark-to-market concentration. A position
entered at $100 that has risen to $200 is counted at cost basis ($100 × qty), not at current
value ($200 × qty) — a 2× undercount. The 85% warn and 95% halt thresholds trigger far too
late, allowing the portfolio to be fully concentrated in winning positions without any warning.

Fix: use `current_price` (updated each cycle by `_sync_positions`) with `price` as a fallback
for pending-fill positions where `current_price` has not yet been populated:

```python
deployed = sum(
    float(d.get('current_price', d.get('price', 0))) * float(d.get('qty', 0))
    for d in self.state.values()
)
```

Tests: `TestPortfolioConcentration.test_concentration_uses_current_price_not_entry_price`,
`test_concentration_falls_back_to_price_when_no_current_price`,
`test_concentration_halt_triggered_by_appreciation`

## Fixed Bugs (Session 21 — May 2026)

### 73. Dead Imports in `src/engine.py`

Two imports were present but never used:

- `OrderType` from `alpaca.trading.enums` — the engine never calls `OrderType.X`; all
  order-type comparisons use `str(o.order_type)` against hardcoded string literals like
  `'OrderType.TRAILING_STOP'`.  The enum class itself was dead weight.
- `MostActivesRequest` from `alpaca.data.requests` — this request type is used in
  `src/scanner.py` (`get_most_actives`), never in `engine.py`.  It was left behind as
  a stale import after scanner logic was extracted into its own module.

Fix: removed both unused imports from `src/engine.py`.
Tests: existing 365 tests all pass; no new tests needed (import presence is not a
behavioral property worth testing).

## Fixed Bugs (Session 20 — May 2026)

### 71. Dashboard Break-Even `↑` Indicator Never Fired

JS table column (alpaca_dashboard.py line ~750) used `p.effective_stop > p.stop_loss`
to decide whether to append the `↑` break-even indicator. `effective_stop` and
`stop_loss` are always set to the same value in the API response (the engine writes the
break-even-floored chandelier stop directly into `stop_loss` via `_update_position_prices`;
no separate `effective_stop` key exists). The comparison was always false — the `↑` never
appeared regardless of break-even status.
Fix: changed to `p.stop_loss >= p.entry_price`, which correctly fires when the break-even
floor is active (stop has been raised to at or above entry).
Also: removed the dead `?? p.stop_loss` fallback from the same expression since
`effective_stop` was passed through directly as `stop_loss` anyway.
Tests: `TestDashboardBreakEvenIndicator.test_break_even_indicator_uses_stop_vs_entry`

### 72. Dead `raw_commission` Code Path in `alpaca_dashboard.py`

`get_state()` computed `raw_commission = d.get("commission")` then checked
`if raw_commission is not None` to add commission to `unit_price`. Alpaca is
commission-free; the engine never writes a `commission` key to state. The
`if raw_commission is not None` branch was always dead — `unit_price` was always
computed from the `elif d.get("fill_price")` fallback.
Fix: removed the dead `raw_commission` variable and the unreachable branch. The
`unit_price` logic now directly checks `fill_price`, matching the actual data flow.
Tests: `TestDashboardBreakEvenIndicator.test_dashboard_no_dead_commission_key`

## Fixed Bugs (Session 24 — May 2026)

### 79. Two Time-Sensitive Tests Failed on Friday After 3 PM

Two tests in `tests/test_trailing_stop_scoring_screener.py` called
`engine.check_velocity_exits()` without mocking `src.engine.datetime`, and asserted
`not tc.submit_order.called`. On any Friday after 3 PM ET, the Friday-close rule would fire
on their positions (profit < `FRIDAY_MIN_PROFIT_PCT`) and call `liquidate()`, breaking both
assertions:

- `TestHardStop.test_hard_stop_does_not_trigger_within_threshold` — position at 6% drawdown
  (below 7% hard-stop threshold) but profit −6% < 3% Friday threshold. Friday-close fired.
- `TestExitOrders.test_velocity_exit_does_not_trigger_before_hold_bars` — fresh position at
  +1% profit (below 5% velocity-exit threshold) but profit 1% < 3% Friday threshold.
  Friday-close fired.

Fix: added `patch('src.engine.datetime')` to pin both tests to Wednesday 2024-06-05 10:30 ET
(same safe-day pattern used in all previously fixed time-sensitive tests). Friday-close rule
is now permanently inactive in both tests.

## Fixed Bugs (Session 25 — May 2026)

### 80. Backtest Entry-Price Below SCAN_MIN_PRICE Not Rejected

`_daily_scan` in `backtest/strategy.py` filtered candidates by `close >= SCAN_MIN_PRICE ($20)`,
but the actual entry proxy `max(open, prev_high) * (1 + BACKTEST_SLIPPAGE)` could resolve below
$20 even when the close passes the filter. Example: stock closes $25 but opens $5 (gap-up day or
data artefact); entry proxy = $5.005 < $20. The live engine rejects sub-$20 prices in
`get_technical_context` before any signal reaches the entry loop. The backtest had no equivalent
guard on the entry proxy — it would attempt to size and place the trade at the sub-floor price.
Fix: added `if entry_price < SCAN_MIN_PRICE: continue` immediately after computing `entry_price`
in `_daily_scan`, after the slippage multiplication.
Tests: `TestBacktestEntryPriceFloor.test_entry_below_scan_min_price_not_entered`,
`test_entry_price_floor_check_in_source`

### 81. Live Reprice Path Does Not Re-Check SCAN_MIN_PRICE After Refreshing Price

`run_cycle()` refreshes the live price for scan snapshots older than 60 seconds. The drift
check (`REPRICE_DRIFT_MAX_PCT = 2%`) validates that the price hasn't moved too far, but there
was no check that the refreshed price is still above the $20 minimum. A stock that was $22 at
scan time could fall to $15 by the time the reprice fires — the engine would proceed to place
a limit buy at $15, violating the liquidity and momentum profile that requires `price > $20`.
The live `get_technical_context` only checks the scan-snapshot price, not the reprice.
Fix: added `if new_price < SCAN_MIN_PRICE: continue` before the drift check in the reprice
block of `run_cycle()`, with a `logger.warning` matching the existing skip-log pattern.
Tests: `TestRepriceMinPriceCheck.test_skip_entry_when_reprice_below_scan_min_price`,
`test_reprice_min_price_check_in_engine_source`

### 83. Stop Audit Skipped by Early-Return Paths (VIX High, Account Failure, Circuit Breaker)

`run_cycle()` ran `_audit_stop_orders()` at step 6, after the VIX check, account-values
fetch, circuit breaker, and concentration halt. All four early-return paths called
`check_velocity_exits()` but returned before reaching step 6. During a sustained high-VIX
session or while the Alpaca API was temporarily down, any position with `stop_dist=0`
(stop placement failed silently) received zero protection for the entire period — the
audit that would place a new stop order was permanently bypassed.
Fix: moved the stop audit to step 1.5 (after `_sync_positions`, before `_get_account_values`),
using a private `_audit_today` variable so there is no name collision with `today_str`
computed later for the circuit breaker. The old step 6 block was removed. The audit now
fires unconditionally on any path that reached `_sync_positions`, which is every path that
passed the connectivity check.
Tests: `TestStopAuditUnprotectedPositions.test_audit_fires_during_vix_high_early_return`,
`test_audit_fires_when_account_values_fail`

### 82. Stop Audit Not Triggered Immediately for Unprotected Positions

`run_cycle()` ran `_audit_stop_orders()` only once per trading day (keyed on
`_last_audit_date`). If a position entered mid-day had its trailing stop silently fail
(e.g., Alpaca order rejected asynchronously), its `stop_dist` would remain `0` and no
protective stop would be in force for the rest of the day — until the next morning's audit.
Fix: added an `has_unprotected` check that scans `self.state` for confirmed positions (not
`pending` or `pending_exit`) with `stop_dist <= 0`. The audit condition is now:
`if (self._last_audit_date != today_str or has_unprotected) and self.state:`.
This triggers an immediate re-audit whenever any live position is found without a stop,
without changing the once-per-day behavior for fully-protected portfolios.
Tests: `TestStopAuditUnprotectedPositions.test_audit_fires_when_position_has_zero_stop_dist`,
`test_audit_does_not_fire_twice_when_protected`,
`test_audit_skips_pending_positions_in_unprotected_check`

## Fixed Bugs (Session 26 — May 2026)

### 84. `_audit_stop_orders` Did Not Restore `stop_dist` When Confirming an Existing TRAIL

After a crash restart, `_sync_positions` re-adds open positions from Alpaca without a
`stop_dist` key. When `_audit_stop_orders` then ran and found a valid trailing-stop order
for that position, it logged "confirmed" and skipped to the next symbol — without updating
`stop_dist` in state. This caused three cascading failures:

1. **`_has_unprotected` fired every cycle**: `stop_dist=0` is the sentinel for "unprotected",
   so the audit was called on every single `run_cycle()` call — even though the TRAIL already
   existed at Alpaca. Unnecessary API calls on every cycle.
2. **Dashboard showed `stop_loss=0.0` permanently**: `_update_position_prices` skips the
   stop_loss update when `stop_dist=0`. The dashboard position rows showed $0 stop for any
   position that survived a restart.
3. **Break-even floor never activated**: The break-even floor logic in `_update_position_prices`
   (`if sd > 0: effective_stop = max(..., ep)`) also depends on `stop_dist > 0`. Re-synced
   positions could retrace through entry with no software safety net.

Fix: when the audit confirms an existing TRAIL for a position with `stop_dist <= 0`, it
restores `stop_dist` from `kept.trail_price` and recomputes `stop_loss = fill_price - trail_dist`.
When `stop_dist` is already set (normal entry path), the audit leaves it unchanged.
Also fixed: new-stop placement path was using `pos_data.get('price', 0)` for stop_loss
computation; changed to `pos_data.get('fill_price') or pos_data.get('price', 0)` to match
the `fill_price`-first pattern used everywhere else in the codebase.
Tests: `TestAuditRestoresStopDist.test_audit_restores_stop_dist_from_existing_trail`,
`test_audit_does_not_overwrite_valid_stop_dist`,
`test_audit_restore_in_source`

## Fixed Bugs (Session 27 — May 2026)

### 85. Five `TestInitialize` Tests Hung Before 09:58 AM ET

Five tests in `tests/test_startup_init.py` called `engine._initialize()` without mocking
`_wait_for_pre_entry_sync`. When run before 09:58 AM ET, `_wait_for_pre_entry_sync` computed
a wait of several hours and entered a real `time.sleep(300)` loop — blocking the entire test
process indefinitely. The tests appeared to pass only when run during trading hours (after 09:58 AM).

Affected tests (all in `TestInitialize` and `TestInitializeAuditDateSet`):
- `test_audits_stops_when_positions_exist`
- `test_skips_audit_when_no_positions`
- `test_skips_price_update_when_no_positions`
- `test_last_audit_date_set_after_startup_audit`
- `test_last_audit_date_not_set_when_no_positions`

Fix: added `patch.object(engine, '_wait_for_pre_entry_sync')` to all five tests, matching
the pattern already used by `test_syncs_positions_twice`, `test_updates_prices_when_positions_exist`,
and `test_writes_dashboard_twice` which were already properly mocked.

### 86. Entry Point and Dashboard Renamed for Clarity

`AutoTrader.py` → `alpaca_auto_trader.py` and `dashboard_server.py` → `alpaca_dashboard.py`
to avoid naming confusion with the sibling `IBKRVelocitySwingTrader` project in the same
parent directory. All internal references, test imports, `main.py`, and `CLAUDE.md` updated.
Stale `goodAfterTime` (IB-era) and `ib_async Package` sections removed from `CLAUDE.md`.

## Fixed Bugs (Session 28 — May 2026)

### 87. No Endpoint to Download Logs from Render

Logs on Render are only visible via the `/api/logs` endpoint (last 200 lines) or the
Render dashboard UI (not searchable or downloadable). There was no way to pull the full
rotating log file for offline analysis.
Fix: added `GET /api/logs/download` to `alpaca_dashboard.py` — returns the full
`trading_engine.log` as a `text/plain` attachment with a dated filename
(`trading_engine_YYYYMMDD.log`). Returns 404 JSON when the log file doesn't exist.
Added `httpx` to `requirements.txt` (required by FastAPI's `TestClient`).
Tests: `TestDashboardLogsDownload.test_download_returns_file_response_when_log_exists`,
`test_download_returns_404_when_log_missing`,
`test_download_endpoint_in_source`

## Fixed Bugs (Session 29 — May 2026)

### 88. `bars[symbol].df` Crashed — `BarSet.__getitem__` Returns `List[Bar]`, Not a `BarSet`

`_fetch_daily_bars` and `_fetch_orb_high` called `bars[symbol].df` where `bars` is a
`BarSet` returned by `get_stock_bars`. `BarSet.__getitem__` returns `self.data[symbol]`
which is `List[Bar]` — a plain Python list. The `.df` property lives on the `BarSet` itself
(via `TimeSeriesMixin`), not on the list, causing `AttributeError: 'list' object has no
attribute 'df'` for every single symbol on every scan cycle. Zero signals were produced.
Fix: replaced `bars[symbol].df` with `pd.DataFrame([b.model_dump() for b in bars[symbol]])`
in both `_fetch_daily_bars` and `_fetch_orb_high`.

### 89. `get_stock_most_actives` Removed from `StockHistoricalDataClient`

`get_most_actives` in `src/scanner.py` called `data_client.get_stock_most_actives(request)`
on a `StockHistoricalDataClient` instance. The method no longer exists on that client —
it was moved to `ScreenerClient` as `get_most_actives`. Caused
`AttributeError: 'StockHistoricalDataClient' object has no attribute 'get_stock_most_actives'`
on every scan cycle, silently dropping the entire most-actives half of the candidate pool.
Fix: changed `get_most_actives` function signature from `data_client: StockHistoricalDataClient`
to `screener_client: ScreenerClient`, and the call from `data_client.get_stock_most_actives`
to `screener_client.get_most_actives`. Updated `get_candidates` to pass `screener_client`.

## Fixed Bugs (Session 30 — June 2026)

### 90. Backtest Strategy Mismatched Live Engine (Donchian Bounce vs Alligator Swing)

`backtest/strategy.py` implemented a **Donchian Bounce** mean-reversion strategy while the
live engine (`src/engine.py` + `src/rules.py`) runs **Alligator Swing** (Bill Williams).
Entry rules were completely different (proximity to 2-day low + RSI oversold lookback vs.
SMMA crossover + day strength). Exit rules differed (velocity time-exit + Friday forced-close
vs. Alligator reversal). Scoring used Donchian proximity component instead of Alligator SMMA
spread. Backtest results were for the wrong strategy and cannot be compared to live performance.

Fix: Full rewrite of `backtest/strategy.py`:
- Entry: Alligator bullish alignment (offset-adj SMMA fast/med > slow), fresh crossover within
  `ALLIGATOR_CROSS_LOOKBACK`, RSI ≥ 50 + delta ≥ `RSI_MIN_DELTA`, RVOL ≥ 1.2×,
  day-strength (close ≥ open × 1.005 AND close in upper half of range)
- Scoring: AlligatorAlignment(30) + RVOL(25) + RSIDelta(25) + Liquidity(20)
- Exit: Chandelier trail + hard stop + break-even floor + Alligator reversal (both
  fast+med SMMA cross below slow → exit next-day open); velocity time-exit and Friday
  forced-close removed
- Config: removed `PROFIT_MIN_THRESHOLD`, `FRIDAY_MIN_PROFIT_PCT`, `DONCHIAN_PERIOD`,
  `BACKTEST_DONCHIAN_TOL_PCT`, `RSI_OVERSOLD_*`, `RSI_BOUNCE_MAX`; added `ALLIGATOR_*`,
  `DAY_STRENGTH_OPEN_PCT`, `SCORE_ALLIGATOR_MAX`
- `_apply_indicators` now adds offset-shifted SMMA aligned columns and `ALLIGATOR_CROSSED`
  boolean after `apply_all()` (which already computes raw SMMA_FAST/MED/SLOW)
- `_apply_rsi_lookback()` method deleted (not needed without RSI oversold lookback)
- `run_backtest.py` updated: removed Donchian/velocity/Friday CLI args; added Alligator-
  aligned args

`tests/test_backtest.py` updated to match:
- `_make_df`: open=close×0.99 (day-strength), add SMMA aligned + ALLIGATOR_CROSSED columns
- `TestEntrySignal`: 16 tests for Alligator rules (was Donchian bounce)
- `TestDailyScanGainFilter` → Alligator crossover coarse filter tests
- `TestBacktestFridayClose` → `TestBacktestAlligatorExit`: confirm no friday_close exits,
  alligator_exits stat correctly tracked

374 tests pass.

## Survivorship Bias Warning (Backtest)
The backtest universe is current NASDAQ/NYSE listings. Bankrupt/delisted tickers from the
backtest window are absent. Momentum/breakout strategies are particularly sensitive to this
bias — reported returns may be overstated by 10-30%. Treat backtest results as directional
signal-quality indicators, not as forecasts of live performance.
