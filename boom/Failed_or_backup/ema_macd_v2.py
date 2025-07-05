import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst, dtype=int)

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global currentPos
    nins, nt = prcSoFar.shape
    if nt < 50:
        return currentPos

    prices_df = pd.DataFrame(prcSoFar)

    # Compute EMAs via transpose trick to run through time
    ema50 = prices_df.T.ewm(span=50, adjust=False).mean().T.to_numpy()
    ema12 = prices_df.T.ewm(span=12, adjust=False).mean().T.to_numpy()
    ema26 = prices_df.T.ewm(span=26, adjust=False).mean().T.to_numpy()

    macd = ema12 - ema26
    signal = pd.DataFrame(macd).T.ewm(span=9, adjust=False).mean().T.to_numpy()

    price_today = prcSoFar[:, -1]
    ema50_today = ema50[:, -1]
    macd_today  = macd[:, -1]
    macd_yest   = macd[:, -2]
    sig_today   = signal[:, -1]
    sig_yest    = signal[:, -2]

    long_mask = (
        (price_today > ema50_today)
        & (macd_today <  0)
        & (sig_today  <  0)
        & (macd_yest  <  sig_yest)
        & (macd_today > sig_today)
    )
    short_mask = (
        (price_today < ema50_today)
        & (macd_today >  0)
        & (sig_today  >  0)
        & (macd_yest  >  sig_yest)
        & (macd_today < sig_today)
    )

    signal_dir = np.zeros(nins, dtype=float)
    signal_dir[long_mask]  =  1.0
    signal_dir[short_mask] = -1.0

    if np.any(signal_dir):
        weights = signal_dir / np.linalg.norm(signal_dir)
        targetPos = np.array([int(x) for x in -3000 * weights / price_today])
    else:
        targetPos = np.zeros(nins, dtype=int)

    currentPos = targetPos.copy()
    return currentPos
