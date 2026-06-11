import numpy as np
import pandas as pd


def compute_smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed Moving Average — less reactive than EMA; better at filtering market noise.

    SMMA[first] = SMA of the first `period` bars.
    SMMA[i]     = (SMMA[i-1] × (period − 1) + price[i]) / period

    Used by the Alligator indicator (periods 5, 8, 13 with Fibonacci-sequence values).
    """
    arr    = series.values.astype(float)
    result = np.full(len(arr), np.nan)

    for start in range(len(arr) - period + 1):
        window = arr[start: start + period]
        if not np.any(np.isnan(window)):
            result[start + period - 1] = window.mean()
            for i in range(start + period, len(arr)):
                if np.isnan(arr[i]):
                    result[i] = np.nan
                else:
                    result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
            break

    return pd.Series(result, index=series.index)


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


def apply_all(df: pd.DataFrame,
              rsi_period:         int = 14,
              atr_period:         int = 14,
              ma_fast:            int = 50,
              ma_slow:            int = 200,
              slope_lookback:     int = 5,
              chandelier_period:  int = 22,
              alligator_fast:     int = 5,
              alligator_med:      int = 8,
              alligator_slow:     int = 13) -> pd.DataFrame:
    """Attach indicator columns to an OHLCV dataframe."""
    df = df.copy()
    df['MA50']         = compute_ma(df['close'], ma_fast)
    df['MA200']        = compute_ma(df['close'], ma_slow)
    df['ATR']          = compute_atr(df, atr_period)
    df['ATR_CHAND']    = compute_atr(df, chandelier_period)
    df['RSI']          = compute_rsi(df['close'], rsi_period)
    df['SMA200_SLOPE'] = df['MA200'] - df['MA200'].shift(slope_lookback)
    # Alligator indicator — three Smoothed Moving Averages (Fibonacci periods)
    df['SMMA_FAST']    = compute_smma(df['close'], alligator_fast)
    df['SMMA_MED']     = compute_smma(df['close'], alligator_med)
    df['SMMA_SLOW']    = compute_smma(df['close'], alligator_slow)
    return df
