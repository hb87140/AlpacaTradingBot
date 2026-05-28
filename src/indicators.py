import pandas as pd

from src.config import ADX_PERIOD


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low']  - df['close'].shift()),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def compute_ma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength (not direction)."""
    high, low, close = df['high'], df['low'], df['close']
    tr       = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    up_move  = high - high.shift()
    dn_move  = low.shift() - low
    plus_dm  = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)
    atr14    = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean()  / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14
    dx       = (100 * (plus_di - minus_di).abs()
                / (plus_di + minus_di).replace(0, float('nan')))
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def apply_all(df: pd.DataFrame,
              rsi_period:         int = 14,
              atr_period:         int = 14,
              ma_fast:            int = 50,
              ma_slow:            int = 200,
              slope_lookback:     int = 5,
              chandelier_period:  int = 22,
              adx_period:         int = ADX_PERIOD) -> pd.DataFrame:
    """Attach indicator columns to an OHLCV dataframe."""
    df = df.copy()
    df['MA50']         = compute_ma(df['close'], ma_fast)
    df['MA200']        = compute_ma(df['close'], ma_slow)
    df['ATR']          = compute_atr(df, atr_period)
    df['ATR_CHAND']    = compute_atr(df, chandelier_period)
    df['RSI']          = compute_rsi(df['close'], rsi_period)
    df['SMA200_SLOPE'] = df['MA200'] - df['MA200'].shift(slope_lookback)
    df['ADX']          = compute_adx(df, adx_period)
    df['HIGH200']      = df['high'].rolling(200, min_periods=1).max()
    return df
