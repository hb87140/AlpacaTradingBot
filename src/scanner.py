"""
Alpaca candidate scanner.

Combines two Alpaca screener endpoints into one candidate pool:
  1. get_top_gainers  — stocks with the largest intraday % price gain.
     These are the primary momentum candidates a breakout strategy wants.
  2. get_most_actives — stocks with the highest intraday share volume.
     Supplements gainers with high-volume movers not yet showing large % moves.

Both lists are fetched each scan cycle and merged (gainers first, then actives).
Duplicates are removed while preserving order.  The combined list is fed to the
engine's 12-rule entry screener, which performs all quality filtering.
"""

from __future__ import annotations

import logging
from typing import List

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest
from alpaca.data.enums import MarketType

from src.config import (
    ALPACA_SCANNER_TOP,
    ALPACA_SCANNER_TOP_GAINERS,
    SCAN_MIN_GAIN_PCT,
)

logger = logging.getLogger('VelocityEngine')


def get_top_gainers(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP_GAINERS) -> List[str]:
    """Return top gaining stock symbols by intraday % change.

    Only includes gainers above SCAN_MIN_GAIN_PCT to match the backtest
    coarse filter and avoid flat/tiny-move stocks polluting the candidate pool.
    """
    try:
        req     = MarketMoversRequest(top=top, market_type=MarketType.STOCKS)
        resp    = screener_client.get_market_movers(req)
        symbols = [
            m.symbol for m in resp.gainers
            if m.percent_change >= SCAN_MIN_GAIN_PCT
        ]
        logger.debug(
            f"SCANNER: top-gainers returned {len(symbols)} symbols "
            f"(≥{SCAN_MIN_GAIN_PCT:.1f}% gain, from {len(resp.gainers)} total)"
        )
        return symbols
    except Exception as e:
        logger.warning(f"SCANNER: top-gainers fetch failed: {e}")
        return []


def get_most_actives(data_client: StockHistoricalDataClient, top: int = ALPACA_SCANNER_TOP) -> List[str]:
    """Return ticker symbols from Alpaca's most-actives screener, sorted by volume."""
    try:
        request  = MostActivesRequest(top=top, by='volume')
        response = data_client.get_stock_most_actives(request)
        symbols  = [item.symbol for item in response.most_actives]
        logger.debug(f"SCANNER: most-actives returned {len(symbols)} symbols")
        return symbols
    except Exception as e:
        logger.warning(f"SCANNER: most-actives fetch failed: {e}")
        return []


def get_candidates(
    data_client: StockHistoricalDataClient,
    screener_client: ScreenerClient,
) -> List[str]:
    """Return the union of top-gainers and most-actives, deduplicated.

    Gainers appear first — they are higher-conviction momentum candidates.
    Most-actives follow to catch high-volume setups that may not yet show
    a large percentage gain (e.g., early-session breakouts).
    """
    gainers = get_top_gainers(screener_client)
    actives = get_most_actives(data_client)

    seen:   set       = set()
    result: List[str] = []
    for sym in gainers + actives:
        if sym not in seen:
            seen.add(sym)
            result.append(sym)

    logger.debug(
        f"SCANNER: combined pool — {len(gainers)} gainers + {len(actives)} actives "
        f"= {len(result)} unique candidates"
    )
    return result
