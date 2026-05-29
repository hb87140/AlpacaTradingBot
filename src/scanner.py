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
import re
from typing import List

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


def get_most_actives(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP) -> List[str]:
    """Return ticker symbols from Alpaca's most-actives screener, sorted by volume."""
    try:
        request  = MostActivesRequest(top=top, by='volume')
        response = screener_client.get_most_actives(request)
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
    actives = get_most_actives(screener_client)

    seen:   set       = set()
    result: List[str] = []
    for sym in gainers + actives:
        if sym not in seen and not _is_non_stock(sym):
            seen.add(sym)
            result.append(sym)

    logger.debug(
        f"SCANNER: combined pool — {len(gainers)} gainers + {len(actives)} actives "
        f"= {len(result)} unique candidates"
    )
    return result
