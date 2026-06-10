"""
Alpaca candidate scanner.

Combines four candidate sources into one pool each scan cycle:

  1. get_top_gainers        — stocks with the largest intraday % gain.
  2. get_most_actives_vol   — stocks with the highest intraday share volume.
  3. get_most_actives_trades— stocks with the most intraday transactions (different
                              ranking than volume — catches high-frequency breakout
                              activity even on stocks with moderate dollar volume).
  4. get_alligator_crossover_scan — daily pre-scan of a curated 250-stock liquid
                              universe; downloads daily bars in batch, computes
                              offset-adjusted SMMAs, and returns symbols where the
                              bullish Alligator crossover occurred within
                              ALLIGATOR_CROSS_LOOKBACK trading days. Runs once per
                              trading day (module-level cache) so the batch download
                              cost is paid only at the first scan of the session.

Sources 1-3 surface intraday momentum candidates that may or may not have active
Alligator crossovers. Source 4 ensures that liquid stocks whose Alligator signal
fired on a quiet day (below the gainers/actives threshold) are never missed.

Early filters applied at scan time:
  • Price floor   — gainers endpoint returns price; sub-$10 stocks discarded.
  • Blocklist     — ETFs, leveraged/inverse products removed from all sources.
  • Non-stocks    — warrants, rights, and other non-equity instruments removed.
  • Min % gain    — gainers below SCAN_MIN_GAIN_PCT discarded.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import MarketType, MostActivesBy

from src.config import (
    ALPACA_DATA_FEED,
    ALPACA_SCANNER_TOP,
    ALPACA_SCANNER_TOP_GAINERS,
    ALLIGATOR_FAST,
    ALLIGATOR_FAST_OFFSET,
    ALLIGATOR_MED,
    ALLIGATOR_MED_OFFSET,
    ALLIGATOR_SLOW,
    ALLIGATOR_SLOW_OFFSET,
    ALLIGATOR_CROSS_LOOKBACK,
    SCAN_MIN_GAIN_PCT,
    SCAN_MIN_PRICE,
    TICKER_BLOCKLIST,
)
from src.indicators import compute_smma

logger = logging.getLogger('VelocityEngine')

# Warrants (BASE+W), rights (BASE.RT), warrant series (BASE.WS), and other
# non-standard instruments never qualify — filter them before hitting the API.
_NON_STOCK_RE = re.compile(
    r'\.'            # contains a period  (e.g. GLED.RT, GRAF.WS)
    r'|WS$|WD$|WT$'  # warrant-series / warrant-deed suffixes
    r'|RT$|RW$'      # rights suffixes
)


def _is_non_stock(sym: str) -> bool:
    """Return True for warrants, rights, and other non-standard instruments."""
    if _NON_STOCK_RE.search(sym):
        return True
    # 5+ char symbols ending in W are almost always warrants (4-char base + W suffix)
    if len(sym) >= 5 and sym.endswith('W'):
        return True
    return False


def _keep(sym: str) -> bool:
    """True when a symbol passes the cheap scanner-level pre-filters."""
    return sym not in TICKER_BLOCKLIST and not _is_non_stock(sym)


# ── Curated liquid universe ────────────────────────────────────────────────────
# Mid-to-large cap NASDAQ and NYSE stocks across all GICS sectors.
# These are always scanned for Alligator crossovers each trading day regardless
# of whether they appear in the intraday gainers or actives lists.
# The Alligator crossover often fires on a modest-volume day (0.5-1.5% move)
# before the stock appears in momentum screeners — this universe catches those setups.
# Update quarterly when major index compositions change.
_LIQUID_UNIVERSE: List[str] = [
    # ── Technology ────────────────────────────────────────────────────────────
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'AMD', 'CRM', 'ADBE', 'NOW', 'INTU',
    'QCOM', 'TXN', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'FTNT',
    'PANW', 'CRWD', 'ZS', 'DDOG', 'SNOW', 'NET', 'OKTA', 'HUBS', 'TEAM', 'MDB',
    'INTC', 'HPQ', 'DELL', 'STX', 'WDC', 'NTAP', 'PSTG', 'SMCI', 'ARM', 'PLTR',
    # ── Consumer Discretionary ────────────────────────────────────────────────
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'BKNG', 'TJX', 'ROST', 'LULU',
    'CMG', 'YUM', 'DHI', 'LEN', 'PHM', 'TOL', 'ABNB', 'NFLX', 'DPZ', 'WYNN',
    'MGM', 'LVS', 'RCL', 'CCL', 'NCLH', 'EXPE', 'LYFT', 'UBER', 'DASH', 'RBLX',
    # ── Financials ────────────────────────────────────────────────────────────
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'V', 'MA',
    'COF', 'DFS', 'USB', 'TFC', 'PNC', 'CME', 'ICE', 'SPGI', 'MCO', 'MSCI',
    'SQ', 'PYPL', 'COIN', 'HOOD', 'AFRM', 'SOFI', 'UPST', 'LC', 'OPEN',
    # ── Healthcare ────────────────────────────────────────────────────────────
    'UNH', 'LLY', 'MRK', 'ABBV', 'TMO', 'DHR', 'AMGN', 'GILD', 'REGN', 'VRTX',
    'BIIB', 'MRNA', 'ISRG', 'DXCM', 'IDXX', 'EW', 'STE', 'HOLX', 'PODD', 'INSP',
    'NVAX', 'RXRX', 'ROIV', 'PTGX', 'SGMO', 'BEAM', 'EDIT', 'CRSP', 'NTLA',
    # ── Industrials ───────────────────────────────────────────────────────────
    'CAT', 'DE', 'HON', 'GE', 'RTX', 'LMT', 'BA', 'GD', 'NOC', 'HII',
    'UPS', 'FDX', 'CSX', 'UNP', 'NSC', 'EMR', 'ETN', 'PH', 'ROK', 'IR',
    'LDOS', 'SAIC', 'BAH', 'CACI', 'MSA', 'RRX', 'XPO', 'CHRW', 'JBHT',
    # ── Energy ────────────────────────────────────────────────────────────────
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'HES',
    'DVN', 'FANG', 'MRO', 'APA', 'HAL', 'BKR', 'NOV', 'CTRA',
    # ── Materials ─────────────────────────────────────────────────────────────
    'LIN', 'APD', 'ECL', 'NEM', 'FCX', 'NUE', 'CF', 'MOS', 'ALB', 'SQM',
    'MP', 'LTHM', 'LAC', 'SGML', 'PLL',
    # ── Communication Services ────────────────────────────────────────────────
    'GOOGL', 'META', 'DIS', 'CMCSA', 'TMUS', 'NFLX', 'EA', 'TTWO', 'RBLX',
    'SNAP', 'PINS', 'MTCH', 'ZM', 'DOCU', 'TTD', 'ROKU', 'APP', 'MGNI',
    # ── Consumer Staples ──────────────────────────────────────────────────────
    'WMT', 'COST', 'PG', 'KO', 'PEP', 'MDLZ', 'CL', 'GIS', 'MKC', 'K',
    # ── Utilities ─────────────────────────────────────────────────────────────
    'NEE', 'DUK', 'SO', 'AEP', 'EXC', 'CEG',
    # ── Real Estate ───────────────────────────────────────────────────────────
    'PLD', 'AMT', 'EQIX', 'CCI', 'SPG', 'O', 'DLR', 'WELL', 'AVB', 'EQR',
    # ── High-beta growth / mid-caps (Alligator works well here) ───────────────
    'SHOP', 'SQ', 'PYPL', 'BILL', 'GTLB', 'PATH', 'AI', 'BBAI', 'SOUN', 'IREN',
    'MSTR', 'CLSK', 'RIOT', 'MARA', 'HUT', 'CIFR', 'BTBT',
    'ENPH', 'SEDG', 'FSLR', 'RUN', 'ARRY', 'NOVA', 'SHLS',
    'LCID', 'RIVN', 'NKLA', 'GOEV', 'WKHS', 'XPEV', 'NIO', 'LI',
    'ON', 'STM', 'WOLF', 'SWKS', 'QRVO', 'MPWR', 'ALGM',
    'CELH', 'VITL', 'HIMS', 'RXST', 'PRCT', 'TMDX', 'IRTC',
]

# Remove any universe symbols that are in the blocklist
_LIQUID_UNIVERSE = [s for s in _LIQUID_UNIVERSE if _keep(s) and not _is_non_stock(s)]
# Deduplicate while preserving order
_seen_univ: set = set()
_LIQUID_UNIVERSE_DEDUP: List[str] = []
for _s in _LIQUID_UNIVERSE:
    if _s not in _seen_univ:
        _seen_univ.add(_s)
        _LIQUID_UNIVERSE_DEDUP.append(_s)
_LIQUID_UNIVERSE = _LIQUID_UNIVERSE_DEDUP

# Day-level cache: computed once per trading day and reused every scan cycle
_universe_cache: dict = {'date': None, 'symbols': []}

_UNIVERSE_BATCH_SIZE = 50   # symbols per Alpaca bars API request
_UNIVERSE_LOOKBACK   = 60   # calendar days of bars to download (≈ 42 trading days)


def get_top_gainers(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP_GAINERS) -> List[str]:
    """Return top gaining stock symbols filtered by gain %, price floor, and blocklist."""
    try:
        req  = MarketMoversRequest(top=top, market_type=MarketType.STOCKS)
        resp = screener_client.get_market_movers(req)
        raw  = resp.gainers

        symbols = []
        n_gain = n_price = n_block = 0
        for m in raw:
            if m.percent_change < SCAN_MIN_GAIN_PCT:
                n_gain += 1
                continue
            price = getattr(m, 'price', None)
            if price is not None and float(price) < SCAN_MIN_PRICE:
                n_price += 1
                continue
            if not _keep(m.symbol):
                n_block += 1
                continue
            symbols.append(m.symbol)

        logger.debug(
            f"SCANNER: top-gainers {len(symbols)} kept from {len(raw)} "
            f"(dropped: gain<{SCAN_MIN_GAIN_PCT:.0f}%={n_gain} "
            f"price<${SCAN_MIN_PRICE:.0f}={n_price} blocked={n_block})"
        )
        return symbols
    except Exception as e:
        logger.warning(f"SCANNER: top-gainers fetch failed: {e}")
        return []


def get_most_actives_vol(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP) -> List[str]:
    """Return most-active stock symbols ranked by intraday share volume."""
    try:
        request  = MostActivesRequest(top=top, by=MostActivesBy.VOLUME)
        response = screener_client.get_most_actives(request)
        raw      = response.most_actives

        symbols  = [item.symbol for item in raw if _keep(item.symbol)]
        logger.debug(f"SCANNER: most-actives(vol) {len(symbols)} kept from {len(raw)}")
        return symbols
    except Exception as e:
        logger.warning(f"SCANNER: most-actives(vol) fetch failed: {e}")
        return []


# Keep old name as alias so any code that imported it directly still works
get_most_actives = get_most_actives_vol


def get_most_actives_trades(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP) -> List[str]:
    """Return most-active stock symbols ranked by intraday transaction count.

    Trade-count ranking surfaces stocks with many small orders — often a sign of
    retail accumulation or early institutional positioning that precedes an Alligator
    crossover. Complements the volume-ranked list with different candidates.
    """
    try:
        request  = MostActivesRequest(top=top, by=MostActivesBy.TRADES)
        response = screener_client.get_most_actives(request)
        raw      = response.most_actives

        symbols  = [item.symbol for item in raw if _keep(item.symbol)]
        logger.debug(f"SCANNER: most-actives(trades) {len(symbols)} kept from {len(raw)}")
        return symbols
    except Exception as e:
        logger.warning(f"SCANNER: most-actives(trades) fetch failed: {e}")
        return []


def get_alligator_crossover_scan(data_client: StockHistoricalDataClient) -> List[str]:
    """Daily batch scan: returns universe symbols with an active Alligator crossover.

    Downloads 60 calendar days of daily bars for the full liquid universe (in
    batches of 50 to respect API request size). Computes offset-adjusted SMMAs
    and returns symbols where the bullish crossover (fast+med crossing above slow)
    occurred within ALLIGATOR_CROSS_LOOKBACK trading bars.

    Result is cached for the trading day — only the first call per session hits
    the API; subsequent calls within the same day return the cached list instantly.
    """
    today_str = date.today().isoformat()
    if _universe_cache['date'] == today_str:
        return _universe_cache['symbols']

    logger.info(
        f"SCANNER: Alligator universe scan — {len(_LIQUID_UNIVERSE)} symbols "
        f"(batches of {_UNIVERSE_BATCH_SIZE}, lookback={_UNIVERSE_LOOKBACK} days)"
    )
    start = datetime.now(timezone.utc) - timedelta(days=_UNIVERSE_LOOKBACK)
    candidates: List[str] = []
    n_batches = 0

    for i in range(0, len(_LIQUID_UNIVERSE), _UNIVERSE_BATCH_SIZE):
        batch = _LIQUID_UNIVERSE[i: i + _UNIVERSE_BATCH_SIZE]
        n_batches += 1
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                feed=ALPACA_DATA_FEED,
            )
            bars = data_client.get_stock_bars(req)
        except Exception as e:
            logger.debug(f"SCANNER: universe batch {n_batches} failed: {e}")
            continue

        for sym in batch:
            try:
                raw_bars = bars[sym]
            except (KeyError, TypeError):
                continue
            if not raw_bars or len(raw_bars) < 22:
                continue

            df = pd.DataFrame([b.model_dump() for b in raw_bars]).reset_index(drop=True)
            df.columns = [c.lower() for c in df.columns]

            smma_f = compute_smma(df['close'], ALLIGATOR_FAST)
            smma_m = compute_smma(df['close'], ALLIGATOR_MED)
            smma_s = compute_smma(df['close'], ALLIGATOR_SLOW)

            def _at(series: pd.Series, offset: int) -> float:
                idx = -1 - offset
                if len(series) > abs(idx):
                    v = float(series.iloc[idx])
                    return v if not np.isnan(v) else float('nan')
                return float('nan')

            fast_now = _at(smma_f, ALLIGATOR_FAST_OFFSET)
            med_now  = _at(smma_m, ALLIGATOR_MED_OFFSET)
            slow_now = _at(smma_s, ALLIGATOR_SLOW_OFFSET)

            if any(np.isnan(v) for v in (fast_now, med_now, slow_now)):
                continue
            # Both faster lines must be above the slow line right now
            if not (fast_now > slow_now and med_now > slow_now):
                continue

            # Check that within CROSS_LOOKBACK bars there was a bar where they were NOT above
            crossed = False
            for k in range(1, ALLIGATOR_CROSS_LOOKBACK + 1):
                pf = _at(smma_f, ALLIGATOR_FAST_OFFSET + k)
                pm = _at(smma_m, ALLIGATOR_MED_OFFSET  + k)
                ps = _at(smma_s, ALLIGATOR_SLOW_OFFSET + k)
                if not any(np.isnan(v) for v in (pf, pm, ps)):
                    if not (pf > ps and pm > ps):
                        crossed = True
                        break

            if crossed:
                candidates.append(sym)

    logger.info(
        f"SCANNER: Alligator universe scan complete — "
        f"{len(candidates)} crossover candidates from {len(_LIQUID_UNIVERSE)} symbols "
        f"({n_batches} batches)"
    )
    _universe_cache['date']    = today_str
    _universe_cache['symbols'] = candidates
    return candidates


def get_candidates(
    data_client: StockHistoricalDataClient,
    screener_client: ScreenerClient,
) -> List[str]:
    """Return the combined candidate pool from all four sources, deduplicated.

    Priority order (higher-conviction sources first):
      1. Top gainers    — strongest intraday % movers (primary momentum signal)
      2. Most-actives by volume  — highest dollar-flow stocks
      3. Most-actives by trades  — most transactions (breakout accumulation signal)
      4. Alligator universe scan — liquid stocks with active SMMA crossovers

    Deduplication preserves order so higher-priority sources are evaluated first.
    The engine's full technical context (SMMA computation, live snapshot, 12 rules)
    is only applied to symbols that pass the cheap scanner-level pre-filters.
    """
    gainers  = get_top_gainers(screener_client)
    act_vol  = get_most_actives_vol(screener_client)
    act_trds = get_most_actives_trades(screener_client)
    universe = get_alligator_crossover_scan(data_client)

    seen:   set       = set()
    result: List[str] = []
    for sym in gainers + act_vol + act_trds + universe:
        if sym not in seen and _keep(sym) and not _is_non_stock(sym):
            seen.add(sym)
            result.append(sym)

    logger.debug(
        f"SCANNER: combined pool — "
        f"{len(gainers)}g + {len(act_vol)}av + {len(act_trds)}at + {len(universe)}u "
        f"= {len(result)} unique candidates"
    )
    return result
