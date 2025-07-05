import numpy as np

nInst = 50
currentPos = np.zeros(nInst, dtype=int)

def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Compute the EMA for each row of `prices` along axis=1.
    prices.shape == (nInst, nt)
    Returns an array of same shape where out[:, t] is the EMA at time t.
    """
    alpha = 2.0 / (period + 1)
    out = np.zeros_like(prices)
    out[:, 0] = prices[:, 0]
    for t in range(1, prices.shape[1]):
        out[:, t] = alpha * prices[:, t] + (1 - alpha) * out[:, t-1]
    return out

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    prcSoFar: price history array of shape (nInst, nt)
    Returns currentPos: integer share counts for each instrument.
    Implements:
      1) Trend filter: price vs. EMA50
      2) MACD filter: MACD(12,26) & signal(9) both < 0 for longs, > 0 for shorts
      3) Crossover: MACD crosses its signal line
      4) One-shot trade to the target allocation sized on a $5k budget
    """
    global currentPos
    nins, nt = prcSoFar.shape
    if nt < 50:
        return currentPos

    # 1) Compute EMAs
    ema50  = ema(prcSoFar, 50)
    ema12  = ema(prcSoFar, 12)
    ema26  = ema(prcSoFar, 26)

    # 2) MACD line and its signal line
    macd   = ema12 - ema26
    signal = ema(macd, 9)

    # 3) Extract “today” and “yesterday” values
    price_today = prcSoFar[:, -1]
    ema50_today = ema50[:, -1]
    macd_today  = macd[:, -1]
    macd_yest   = macd[:, -2]
    sig_today   = signal[:, -1]
    sig_yest    = signal[:, -2]

    # 4) Build long/short masks
    long_mask = (
        (price_today > ema50_today)     # uptrend
        & (macd_today <  0)             # momentum below zero
        & (sig_today  <  0)
        & (macd_yest  <  sig_yest)      # yesterday MACD < signal
        & (macd_today > sig_today)      # today MACD > signal
    )
    short_mask = (
        (price_today < ema50_today)     # downtrend
        & (macd_today >  0)             # momentum above zero
        & (sig_today  >  0)
        & (macd_yest  >  sig_yest)      # yesterday MACD > signal
        & (macd_today < sig_today)      # today MACD < signal
    )

    # 5) Encode signals: +1 long, -1 short, 0 flat
    signal_dir = np.zeros(nins, dtype=float)
    signal_dir[long_mask]  =  1.0
    signal_dir[short_mask] = -1.0

    # 6) Map to a target share count vector
    if np.any(signal_dir != 0):
        weights = signal_dir / np.linalg.norm(signal_dir)
        # target shares sized on a $10,000 notional
        targetPos = np.array([int(x) for x in -3000 * weights / price_today])
    else:
        targetPos = np.zeros(nins, dtype=int)

    # 7) One-shot trade: go straight to target
    currentPos = targetPos.copy()
    return currentPos
