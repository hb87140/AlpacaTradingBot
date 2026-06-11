# ALLIGATOR ALPHA — PROJECT CONTEXT

## Project Purpose

Fully automated equity swing trading bot using the **Alpaca** brokerage API.
Optimised for **small cash accounts** with T+1 settlement. No margin, no leverage.

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

- **Python**: venv at `venv/` uses Python 3.13. Run via `venv/bin/python`.
- **Broker**: Alpaca (`alpaca-py`). Do NOT use IB/ib_async imports.
- **Run tests**: `venv/bin/python -m pytest tests/ -q`  *(380+ tests — all must pass)*
- **Start engine**: `venv/bin/python alpaca_auto_trader.py`
- **Run backtest**: `venv/bin/python run_backtest.py`

## Architecture

```text
alpaca_auto_trader.py  ← entry point (signal handling, restart loop)
src/engine.py          ← VelocityEngine — main trading loop
src/config.py          ← all tunable parameters (edit here, not inline)
src/indicators.py      ← ATR, RSI, SMMA, MA (apply_all)
src/rules.py           ← entry rules (PERMANENT_DAY_RULES + CYCLE_RULES) + scorer
src/scanner.py         ← Alpaca candidate screener (top-gainers + most-actives)
backtest/strategy.py   ← offline backtester matching live strategy exactly
alpaca_dashboard.py    ← FastAPI web dashboard
main.py                ← combined launcher for cloud deployment
run_backtest.py        ← CLI entry point for backtesting
tests/                 ← pytest suite (380+ tests)
```

## Strategy Summary — Alligator Swing (Bill Williams)

**Entry** — all rules must pass:

| Layer | Rule | Threshold |
| --- | --- | --- |
| Permanent (daily, cached) | Dollar volume | 20d avg ≥ $2.5M (`SCAN_MIN_DOLLAR_VOL`); 2× on Fridays |
| Permanent (daily, cached) | Share volume | 20d avg ≥ 500K shares (`SCAN_MIN_VOLUME`); 2× on Fridays |
| Cycle | Spread | Bid-ask ≤ 0.5% (`SPREAD_MAX_PCT`) |
| Cycle | Alligator bullish | SMMA(5) > SMMA(13) AND SMMA(8) > SMMA(13) (offset-adjusted) |
| Cycle | Fresh crossover | Within last 10 bars (`ALLIGATOR_CROSS_LOOKBACK`) |
| Cycle | RSI momentum | RSI(14) ≥ 50 AND rising ≥ 0.5 pts (`RSI_MIN_DELTA`) |
| Cycle | RVOL | Intraday ≥ 1.2× (`RVOL_MIN`) |
| Cycle | Day strength | Price ≥ open × 1.005 AND in upper 50% of intraday range |
| Score gate | Composite score | ≥ 50 / 100 (`SCAN_MIN_SCORE`) |

**Scoring** (0-100, gate = 50):

- Alligator alignment: 0-30 pts (SMMA spread width)
- RVOL: 0-25 pts (linear 1.2× → 5×)
- RSI delta: 0-25 pts (linear 0 → 10 pts rise)
- Spread + dollar vol: 0-20 pts (liquidity quality)
- Analyst consensus bonus: 0-15 pts (buy ratio 30% → 70%, capped so total ≤ 100)

**Exit** — first condition to fire wins:

1. **Tiered profit exits** — sell 25% of original qty at 0.75R profit, another 25% at 1.25R profit (R = chandelier stop distance = ATR(14) × 2.5, set at entry). Floor qty per tier. Remaining 50% rides the chandelier trailing stop. Thresholds scale with volatility — a wider stop requires a larger absolute move before locking in gains.
2. **Chandelier trailing stop** — ATR(14) × 2.5 dollar distance; placed as Alpaca `TrailingStopOrderRequest(trail_price=...)`. GTC order, active until reversal or stop.
3. **Hard stop** — 5% drawdown from entry (`HARD_STOP_PCT`). Software market sell.
4. **Break-even floor** — once profit ≥ 6% (`BREAK_EVEN_PCT`), stop rises to entry; software exit fires if price retraces back to entry.
5. **Alligator reversal** — both fast SMMA(5) AND med SMMA(8) cross below slow SMMA(13); exit at day-end confirmation.

**Universe**: NASDAQ Global Select + NYSE, price > $1, 20d avg dollar vol > $2.5M, daily gain ≥ 1% (`SCAN_MIN_GAIN_PCT`) for initial scan

## Critical Design Decisions (Do NOT "Fix" These)

### Cash Account — No Margin

This system trades a **cash account only**. No leverage. Position sizing is based on settled
cash / open slots. Do not introduce leverage concepts.

### T+1 Settlement Awareness

Exit logic and capital bucketing account for T+1 settlement. Settled cash (not total equity)
determines available capital for new entries.

### Dynamic Position Slots

`MAX_POSITIONS` is NOT a fixed constant. Computed from total equity:
`floor(NetLiquidation / MIN_BUCKET_SIZE)`, capped at `MAX_POSITIONS_CAP=8`.
New entry slots are further constrained by `floor(SettledCash / MIN_BUCKET_SIZE)`.
Import `MAX_POSITIONS_CAP` and `MIN_BUCKET_SIZE` from config — never `MAX_POSITIONS`.

### Break-Even Floor: Dual Enforcement

`BREAK_EVEN_PCT=0.06` enforced in two places:

1. **Dashboard / state tracking** (`_update_position_prices`): `stop_loss` floored at entry.
2. **Programmatic exit** (`check_velocity_exits`): if price retraces to/below entry after
   peak hit break-even, `liquidate()` fires immediately.

### Tiered Exit: Original Qty Is the Reference; Thresholds Are R-Multiples

The 25%/25%/50% tiers are computed from **`original_qty`** (saved at fill), not from the
current remaining qty. This prevents each tier shrinking as shares are sold. `tier_sold` in
state tracks how many tiers (0-2) have completed.

Tier thresholds are **R-multiples of `stop_dist`** (the chandelier ATR distance set at entry),
not fixed percentages. Tier 1 fires when `price - entry ≥ 0.75 × stop_dist`; Tier 2 at
`price - entry ≥ 1.25 × stop_dist`. This makes exits scale with volatility: a volatile stock
with a wider stop requires a larger absolute move before locking in gains. Config:
`TIER_EXIT_R_MULTIPLES = (0.75, 1.25)`, `TIER_EXIT_PCT = 0.25`.

After a tier sell:

- All open orders are cancelled (Alpaca blocks sells while TRAIL holds shares).
- `stop_order_id` is cleared from state so `_has_unprotected` fires.
- `_audit_stop_orders` re-places the TRAIL for the reduced qty on the next cycle.

Tier qty = `floor(original_qty × TIER_EXIT_PCT)`. If floor rounds to 0 (e.g. 1-share position),
the tier is skipped — no partial sell of 0 shares. If `stop_dist` is 0 (not yet set), tier
check is skipped entirely.

### Chandelier Stop: Dollar Distance (Matches Alpaca `trail_price`)

Alpaca `TrailingStopOrderRequest` uses `trail_price` — a **fixed dollar amount** below the peak.
`chandelier_dist = ATR_CHAND × CHANDELIER_MULT` is computed once at entry and stored in
`stop_dist`. The cap `fill_price × 0.24` (not 0.25) buffers against Alpaca's 25% trail_price
limit — a 1% safety margin covers intraday price movement between fill and stop submission.
If the trailing stop fails, `stop_dist` is reset to 0, triggering `_has_unprotected` so the
audit retries on the next cycle.

### Score Persistence Across Restarts

Score is saved to `engine_state.json` at entry and preserved through restarts as long as the
state file survives (persistent disk). On ephemeral-filesystem platforms (Render free tier),
the state file is lost on restart. In that case `_sync_positions` calls `_try_rescore(sym)` to
re-compute the score from current bars. `_try_rescore` returns `None` on failure — score shown
as `—` on dashboard until the next cycle re-populates it.

### `liquidate()` Must Cancel ALL Orders Before Market Sell

Alpaca holds shares "for orders" — the GTC TRAIL reserves the full qty, so a market SELL is
rejected with `available=0` while the TRAIL is open. `liquidate()` cancels all open orders for
the symbol first, then places the market sell. `_audit_stop_orders` re-places the TRAIL if the
sell fails (position still exists with `has_unprotected=True` on next cycle).

### No `del self.state[symbol]` in `liquidate()`

State is marked `pending_exit=True`; deletion is deferred until `_sync_positions` confirms the
position is flat. This prevents the hold-time clock resetting on a rejected sell.

### `_calc_max_positions` Returns 0 for Insufficient Equity

Returns `0` (not `1`) when `equity < MIN_BUCKET_SIZE`. A cash account below $500 cannot safely
size even one position. Do NOT use `max(1, int(equity / MIN_BUCKET_SIZE))`.

### SPY Regime: Slope Check Required

`_fetch_spy_trend()` requires `SPY close > SMA50 > SMA200` AND `SMA200 slope > 0`
(over `SMA200_SLOPE_LOOKBACK=5` days). Both live engine and backtest must include this check.

### Spread Filter With Live Data

With live streaming, bid/ask is real-time. If bid/ask is momentarily unavailable, `spread_pct`
falls back to 0.0 — the limit-price buffer provides cost protection.

### Combined Candidate Pool (Scanner)

`src/scanner.py` `get_candidates(data_client, screener_client)` merges:

1. `ScreenerClient.get_market_movers` → top-gainers by intraday % change (≥ `SCAN_MIN_GAIN_PCT=1.0%`)
2. `ScreenerClient.get_most_actives` → top stocks by intraday volume

Gainers first, then actives, duplicates removed.
Do NOT call `get_most_actives` or `get_top_gainers` directly from engine code; use `get_candidates`.

### `_tod_frac` — Cumulative Volume CDF (Not a Rate)

```text
elapsed ≤ 30 min  →  f = max(0.01, elapsed / 30 × 0.22)   (22% by 30 min)
elapsed > 30 min  →  f = 0.22 + (elapsed − 30) / 360 × 0.78  (100% by 390 min)
```

RVOL = `intraday_vol / avg_20d_vol / tod_frac`. Do NOT invert to a rate function.

### VIX: Day-Level Caching and Graceful Degradation

`_fetch_vix` caches for the trading day. On first miss, uses `_last_vix` with a warning.
Only when NO value has ever been fetched does `run_cycle` log a warning and continue without
the VIX filter. Do NOT revert to halt-on-None behavior.

### PERMANENT_DAY_RULES vs CYCLE_RULES

`PERMANENT_DAY_RULES` (in `src/rules.py`) are checked once per day per symbol and cached in
`_daily_scan_skip`. Rules belong here when their inputs are historical daily averages that do
not change intraday (dollar volume, share volume). `CYCLE_RULES` are checked every scan cycle
for live-data filters (spread, RVOL, day strength, Alligator state).

## Risk Parameters

- 2% equity risk per trade (ATR-based position sizing)
- Dynamic slots: `floor(equity / $500)` capped at 8; entries also require settled cash
- Daily loss circuit breaker: 3% intraday drawdown halts new entries (`MAX_DAILY_LOSS_PCT`)
- Portfolio concentration: warn at 85%, halt at 95% deployed equity
- VIX threshold: 35 blocks new entries (`VIX_THRESHOLD`)

## Running Tests

```bash
venv/bin/python -m pytest tests/ -q              # all tests
venv/bin/python -m pytest tests/test_engine.py   # engine only
```

All 380+ tests must pass. Fix failures before committing.

## Key Files

| File | Purpose |
| --- | --- |
| `src/config.py` | All tunable parameters — never hardcode inline |
| `src/engine.py` | `VelocityEngine.run_cycle()` — main loop |
| `src/rules.py` | `PERMANENT_DAY_RULES`, `CYCLE_RULES`, `score_candidate` |
| `src/scanner.py` | `get_candidates` — combined candidate pool |
| `src/indicators.py` | `apply_all` — produces SMMA, ATR, RSI, MA columns |
| `backtest/strategy.py` | `VelocityBacktest` — must mirror live strategy exactly |
| `tests/test_engine.py` | Core engine tests |
| `tests/test_trailing_stop_scoring_screener.py` | Exit, scoring, screener tests |

## What NOT to Do

- Do NOT introduce IB/ib_async/ib_insync imports — broker is Alpaca
- Do NOT hardcode values that belong in `src/config.py`
- Do NOT skip tests — always run full suite after changes
- Do NOT import `MAX_POSITIONS`, `CAPITAL_SEED`, or `SCAN_COUNT` from config
- Do NOT modify `liquidate()` to skip order cancellation before the market sell
- Do NOT restore `del self.state[symbol]` in `liquidate()` — use `pending_exit=True`
- Do NOT revert `get_candidates` back to `get_most_actives` only
- Do NOT revert `_calc_max_positions` to `max(1, int(equity / MIN_BUCKET_SIZE))`
- Do NOT revert `_tod_frac` to a rate function — it is a cumulative volume CDF
- Do NOT revert VIX None-handling to halt-entries — warn and continue is correct
- Do NOT use `original_qty × TIER_EXIT_PCT` without `int()` (floor) — Alpaca requires whole shares
- Do NOT remove the `stop_order_id` clear from `_execute_tier_sell` — it triggers the audit to re-place the TRAIL for the reduced qty
- Do NOT diverge exit logic between `src/engine.py` (`check_velocity_exits`) and `backtest/strategy.py` (`_run_loop`)
- Do NOT remove the SPY SMA200 slope check from either live engine or backtest

## Survivorship Bias Warning (Backtest)

The backtest universe is current NASDAQ/NYSE listings. Bankrupt/delisted tickers from the
backtest window are absent. Momentum/breakout strategies are particularly sensitive to this —
reported returns are overstated. Treat backtest results as signal-quality indicators, not
forecasts of live performance.
