import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE       = os.path.join(BASE_DIR, "engine_state.json")
DASHBOARD_FILE   = os.path.join(BASE_DIR, "dashboard_data.json")
EQUITY_HIST_FILE = os.path.join(BASE_DIR, "equity_history.json")
LOG_DIR          = os.path.join(BASE_DIR, "logs")
LOG_FILE         = os.path.join(LOG_DIR,  "trading_engine.log")

# ── Alpaca connectivity ───────────────────────────────────────────────────────
# Paper trading uses paper-api.alpaca.markets; live uses api.alpaca.markets.
# Set ALPACA_PAPER=false (env) to switch to live when ready.
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER      = os.getenv("ALPACA_PAPER", "true").strip().lower() not in {"0", "false", "no", "off"}
# Data feed: "iex" (free, subset of volume) or "sip" (paid, full consolidated tape).
# For paper trading IEX is sufficient; switch to "sip" when live.
ALPACA_DATA_FEED  = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()

# ── Capital, position sizing, and broker-derived values ───────────────────────
# Live capital always comes from Alpaca account.portfolio_value / account.cash.
# Backtests still need explicit assumptions.
BACKTEST_INITIAL_CAPITAL      = float(os.getenv("VELOCITY_BACKTEST_INITIAL_CAPITAL", "2000.0"))
BACKTEST_SCAN_COUNT           = int(os.getenv("VELOCITY_BACKTEST_SCAN_COUNT", "0"))   # 0 = all
BACKTEST_COMMISSION_PER_ORDER = float(os.getenv("VELOCITY_BACKTEST_COMMISSION_PER_ORDER", "0.0"))  # Alpaca is commission-free

# Dynamic position slots: floor(total_equity / MIN_BUCKET_SIZE), capped at MAX_POSITIONS_CAP.
# New entries are further constrained by settled cash so the cash account never
# spends unsettled proceeds.
MIN_BUCKET_SIZE   = float(os.getenv("VELOCITY_MIN_BUCKET_SIZE", "500.0"))
MAX_POSITIONS_CAP = int(os.getenv("VELOCITY_MAX_POSITIONS_CAP", "8"))
BUCKET_CASH_PCT   = 0.90   # deploy 90% of bucket; 10% reserve avoids overdraft

# ── Chandelier Exit trailing stop ─────────────────────────────────────────────
CHANDELIER_PERIOD = 14     # lookback for ATR and highest-high
CHANDELIER_MULT   = 2.5    # ATR multiplier — stop = peak − ATR(14) × 2.5

# ── Risk rules ────────────────────────────────────────────────────────────────
VIX_THRESHOLD        = 35
MAX_DAILY_LOSS_PCT   = 0.03    # 3% intraday equity drawdown halts new entries
RSI_MIN_DELTA        = 0.5     # minimum RSI point rise to confirm momentum turn
HARD_STOP_PCT        = 0.05    # 5% drawdown from entry → forced market exit
RISK_PER_TRADE_PCT   = 0.02    # risk 2% of equity per trade (ATR-based sizing)
BREAK_EVEN_PCT       = 0.06    # once profit ≥ 6%, floor stop at entry
LIMIT_BUF_MIN_PCT    = 0.002   # minimum limit-price buffer above ask (0.2%)
LIMIT_BUF_MAX_PCT    = 0.015   # maximum limit-price buffer (1.5%)
CONCENTRATION_WARN_PCT = 0.85  # deployed ≥ 85% of equity → warn, no new entries
CONCENTRATION_HALT_PCT = 0.95  # deployed ≥ 95% of equity → halt all order activity
REPRICE_DRIFT_MAX_PCT  = 0.02  # skip entry if price moved >2% since scan snapshot

# ── Session timing ────────────────────────────────────────────────────────────
ENTRY_START          = (9, 35)   # first valid entry (5 min after open)
EXIT_START           = (9, 32)   # earliest software exit (avoids opening-print volatility)
EXIT_END             = (16, 0)   # latest software exit — no market sells after close
ENTRY_END            = (14, 0)
VOL_MULT_FRIDAY      = 2.0       # Friday liquidity gate: 2× normal threshold
PRE_ENTRY_SYNC_TIME  = (9, 44)   # pre-entry position re-sync + stop audit
FRIDAY_CLOSE_HOUR    = 15        # ET hour after which new entries are blocked on Fridays

# ── Indicators ────────────────────────────────────────────────────────────────
RSI_PERIOD    = 14
ATR_PERIOD    = 14
MA_FAST       = 50
MA_SLOW       = 200
SMA200_SLOPE_LOOKBACK = 5

# Day-strength gate (confirms price is recovering, not fading)
DAY_STRENGTH_OPEN_PCT  = 0.005  # price must be ≥ 0.5% above today's open

# ── Alligator Swing Strategy (Bill Williams) ──────────────────────────────────
# Three Smoothed Moving Averages with Fibonacci periods and chart offsets.
# Entry: both fast and medium SMMAs cross above the slow SMMA (bullish alignment).
# Exit:  fast crosses below medium (early) OR both cross below slow (confirmed reversal).
ALLIGATOR_FAST         = 5     # green SMMA — fastest jaw
ALLIGATOR_MED          = 8     # red SMMA — medium jaw
ALLIGATOR_SLOW         = 13    # blue SMMA — slowest jaw (the trend anchor)
ALLIGATOR_FAST_OFFSET  = 3     # chart displacement: look back 3 bars for fast line
ALLIGATOR_MED_OFFSET   = 5     # chart displacement: look back 5 bars for medium
ALLIGATOR_SLOW_OFFSET  = 8     # chart displacement: look back 8 bars for slow
ALLIGATOR_CROSS_LOOKBACK = 10  # crossover must have occurred within this many bars

# ── SPY regime filter (soft — disabled by default) ────────────────────────────
# When enabled: bearish regime cuts position size and tightens RVOL threshold.
# Alligator swing works across regime types, so this is off unless explicitly re-enabled.
SPY_FILTER_ENABLED   = False     # set True to re-enable soft SPY regime
SPY_EMA_PERIOD       = 50        # EMA period used for SPY regime check
SPY_REGIME_SIZE_CUT  = 0.50      # reduce position bucket by 50% in bearish regime
SPY_REGIME_RVOL_MULT = 1.33      # multiply RVOL threshold by this in bearish regime

# ── Scoring weights ────────────────────────────────────────────────────────────
# Alligator Alignment (30) · RVOL (25) · RSI Momentum (25) · Liquidity (20)
# + Analyst Consensus (15 bonus) — total capped at 100.
SCORE_ALLIGATOR_MAX = 30.0     # SMMA spread — wider mouth = stronger trend signal
SCORE_RVOL_MAX      = 25.0
SCORE_RSI_DELTA_MAX = 25.0
SCORE_LIQUIDITY_MAX = 20.0
SCORE_ANALYST_MAX   = 15.0     # analyst consensus — majority buy rating boosts priority

# ── Historical data windows ───────────────────────────────────────────────────
# Expressed as calendar days so Alpaca's date-based API can use them directly.
DAILY_HISTORY_DAYS = 400   # enough to produce 200+ trading-day bars

# ── Scanner filters ───────────────────────────────────────────────────────────
SCAN_MIN_PRICE      = 1.0               # Universe filter: price > $1 (excludes sub-penny stocks only)
SCAN_MIN_VOLUME     = 500_000          # Universe filter: 20-day avg daily vol > 500K shares
SCAN_MIN_GAIN_PCT   = 1.0              # minimum daily % gain (backtest coarse filter)
SCAN_MIN_DOLLAR_VOL = 2_500_000        # 20-day avg dollar volume floor ($2.5M)
SCAN_MIN_SCORE      = 50.0             # minimum composite score (0-100) before entry

# ── Tiered profit exits ───────────────────────────────────────────────────────
# At each tier the engine sells TIER_EXIT_PCT of the ORIGINAL entry qty (floor).
# Remaining shares (~40%) ride the chandelier trailing stop to maximise winners.
TIER_EXIT_PROFITS   = (0.05, 0.10)  # 5 %, 10 % profit thresholds
TIER_EXIT_PCT       = 0.25          # fraction of original qty to sell at each tier (25 %)

# ── Ticker blocklist ─────────────────────────────────────────────────────────
TICKER_BLOCKLIST: set = {
    # Broad-market ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'VEA', 'VWO',
    # Sector / theme ETFs
    'XLV', 'XLK', 'XLF', 'XLE', 'XLI', 'XLU', 'XLY', 'XLP', 'XLRE', 'XLB', 'XLC',
    'IHI', 'IBB', 'IYH', 'IYW', 'IYF', 'IYE', 'IYJ', 'IYC', 'IYK', 'IYR',
    'VHT', 'VGT', 'VFH', 'VDE', 'VIS', 'VCR', 'VDC', 'VNQ', 'VAW',
    'GLD', 'SLV', 'USO', 'UNG',
    'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'AGG', 'BND',
    'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ',
    # Inverse / leveraged ETFs
    'SQQQ', 'SPXS', 'SDOW', 'SRTY', 'TZA', 'FAZ', 'SPXU',
    'SOXS', 'LABD', 'YANG', 'DRV', 'TECS', 'DUST', 'DRIP',
    'UVXY', 'SVXY',
    'TQQQ', 'UPRO', 'SPXL', 'UDOW', 'URTY', 'SOXL', 'LABU', 'TECL',
}

# ── Screener rules ────────────────────────────────────────────────────────────
MIN_CANDLES          = 210     # minimum daily bars (SMA200 + slope buffer)
RVOL_MIN             = 1.2     # minimum relative volume — aligned with backtest optimal (BACKTEST_RVOL_MIN)
BACKTEST_RVOL_MIN    = 1.2     # daily close RVOL proxy — matches live RVOL_MIN after parameter alignment
BACKTEST_SLIPPAGE    = 0.001   # 0.1% entry slippage
BACKTEST_EXIT_SLIPPAGE = 0.001
SPREAD_MAX_PCT       = 0.005   # maximum bid-ask spread (0.5%)
CORR_MAX             = 0.7     # max daily-return correlation with any open position
CORR_LOOKBACK        = 90      # calendar days for correlation calculation
MAX_SECTOR_COUNT     = 2       # max simultaneous positions in the same sector

# ── Alpaca scanner ───────────────────────────────────────────────────────────
# Combined candidate pool = top-gainers (by % change) UNION most-actives (by volume).
# Gainers are sorted first — they are the primary momentum signal.
# Most-actives supplement with high-volume movers that may not yet show large % gains.
ALPACA_SCANNER_TOP         = int(os.getenv("ALPACA_SCANNER_TOP", "100"))         # most-actives count (Alpaca API max: 100)
ALPACA_SCANNER_TOP_GAINERS = int(os.getenv("ALPACA_SCANNER_TOP_GAINERS", "50"))  # gainers count (Alpaca API max: 50)

# ── Loop timing ───────────────────────────────────────────────────────────────
SCAN_INTERVAL          = 60    # seconds between cycles
ERROR_WAIT             = 60
LOG_BACKUP_COUNT       = 30
EQUITY_RETRY_INTERVAL  = 5
EQUITY_HIST_INTERVAL   = 1800  # 30-minute dashboard history snapshots
