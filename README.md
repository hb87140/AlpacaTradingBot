# AlpacaTradingBot

Fully automated equity swing trading bot using the [Alpaca](https://alpaca.markets) brokerage API.
Optimised for small cash accounts with T+1 settlement. No margin, no leverage.

## Strategy — Alligator Swing (Bill Williams)

**Entry** — all six rules must pass on the same scan cycle:

1. **Alligator bullish**: offset-adjusted SMMA(5) > SMMA(13) AND SMMA(8) > SMMA(13)
2. **Fresh crossover**: Alligator turned bullish within the last 10 bars
3. **RSI momentum**: RSI(14) ≥ 50 AND rising ≥ 2 pts from previous bar
4. **Relative volume**: intraday RVOL ≥ 1.2× 20-day average
5. **Day strength**: live price ≥ today's open × 1.005
6. **Spread**: bid-ask spread ≤ 0.5%

**Exit** — first condition to fire wins:

1. Chandelier trailing stop (ATR(22) × 2.5 dollar distance, placed as Alpaca TRAIL order)
2. Alligator reversal (both fast and mid SMMA cross below slow at day-end)
3. Hard stop: 5% drawdown from entry
4. Break-even floor: once profit ≥ 6%, stop rises to entry price or above

**Universe**: NASDAQ Global Select + NYSE, price > $20, 20-day avg dollar vol > $5M

## Architecture

```text
alpaca_auto_trader.py  ← entry point (signal handling, restart loop)
src/engine.py          ← VelocityEngine — main trading loop
src/config.py          ← all tunable parameters
src/indicators.py      ← ATR, RSI, SMMA, MA (apply_all)
src/rules.py           ← six entry rules (CYCLE_RULES)
src/scanner.py         ← Alpaca screener (top-gainers + most-actives)
backtest/strategy.py   ← offline backtester matching live strategy
alpaca_dashboard.py    ← FastAPI web dashboard
main.py                ← combined launcher for cloud deployment
tests/                 ← pytest suite (367+ tests)
```

## Risk Parameters

- 2% equity risk per trade (ATR-based position sizing)
- Dynamic slots: `floor(equity / $500)` capped at 8; new entries require settled cash
- Daily loss circuit breaker: 3% intraday drawdown halts new entries
- VIX threshold: 35 (blocks new entries above)
- Portfolio concentration: warn at 85%, halt at 95%

## Quick Start

```bash
# Install dependencies
python -m venv venv
venv/bin/pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# edit .env: set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

# Run all tests
venv/bin/python -m pytest tests/ -q

# Start the engine (standalone)
venv/bin/python alpaca_auto_trader.py

# Start engine + dashboard together
venv/bin/python main.py
# Dashboard → http://localhost:8080

# Run backtest
venv/bin/python run_backtest.py
```

## Live Deployment

See [DEPLOY.md](DEPLOY.md) for Render / Railway / Docker instructions.

- **Dashboard**: <https://alpacatradingbot-pe9m.onrender.com>
- **Health check**: <https://alpacatradingbot-pe9m.onrender.com/health>
- **Live logs**: <https://alpacatradingbot-pe9m.onrender.com/api/logs>

## Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `ALPACA_API_KEY` | Yes | — | Alpaca API key |
| `ALPACA_SECRET_KEY` | Yes | — | Alpaca secret key |
| `ALPACA_PAPER` | No | `true` | `true` = paper, `false` = live |
| `ALPACA_DATA_FEED` | No | `iex` | `iex` (free) or `sip` (paid) |
| `PORT` | No | `8080` | Set automatically by cloud platforms |
