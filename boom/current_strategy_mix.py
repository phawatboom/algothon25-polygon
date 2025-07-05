import numpy as np
import pandas as pd

nInst = 50
dlrPosLimit  = 10000

# Your persistent state
currentPos      = np.zeros(nInst, dtype=int)
position_dir    = np.zeros(nInst, dtype=int)  # –1/0/+1 signal
last_cross      = np.zeros(nInst, dtype=int)  # last crossover direction
last_signal_dir = np.zeros(nInst, dtype=int)  # ← ADDED: remembers previous signal_dir

def compute_RSI(prices: pd.DataFrame, period: int = 14) -> np.ndarray:
    # 1) Calculate price changes
    delta = prices.diff()

    # 2) Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # 3) Wilder’s smoothing via ewm: α = 1/period → com = period−1
    avg_gain = gain.ewm(com=period-1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period, adjust=False).mean()

    # 4) Compute RSI
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Return as numpy array to match your existing code
    return rsi.to_numpy()

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global currentPos, position_dir, last_cross, last_signal_dir

    nins, nt = prcSoFar.shape
    if nt < 50:
        return currentPos

    # 1) Compute EMAs
    df     = pd.DataFrame(prcSoFar)
    ema50  = df.T.ewm(span=50, adjust=False).mean().T.to_numpy()
    ema12  = df.T.ewm(span=12, adjust=False).mean().T.to_numpy()
    ema26  = df.T.ewm(span=26, adjust=False).mean().T.to_numpy()

    macd    = ema12 - ema26
    signal  = pd.DataFrame(macd).T.ewm(span=9, adjust=False).mean().T.to_numpy()

    # 2) Compute RSI on closing prices
    rsi_all = compute_RSI(df)  # shape (nInst, nt)

    # 2) Grab “today” vs “yesterday” values
    price_t = prcSoFar[:, -1]
    macd_t  = macd[:, -1]; macd_y = macd[:, -2]
    sig_t   = signal[:, -1]; sig_y  = signal[:, -2]
    rsi_t  = rsi_all[:, -1]

    # 3) MACD crossover logic → position_dir / last_cross
    for i in range(nins):
        # long crossover?
        if (
            (macd_y[i] < sig_y[i]) 
            and (macd_t[i] > sig_t[i]) 
            and (macd_y[i] < 0) and (sig_y[i] < 0)
            and (macd_t[i] < 0) and (sig_t[i] < 0)
            and last_cross[i] != +1
            ):

            position_dir[i] = +1
            last_cross[i]   = +1
        # short crossover?
        elif (
            # detects crosses downwards
            (macd_y[i] > sig_y[i]) 
            and (macd_t[i] < sig_t[i]) 
            and (macd_y[i] > 0) and (sig_y[i] > 0)
            and (macd_t[i] > 0) and (sig_t[i] > 0)
            and last_cross[i] != -1
              ):
            position_dir[i] = -1
            last_cross[i]   = -1

#-------------------------------------------------------------
    # # 5) RSI logic for NEG_IDX instruments (vectorized)
    # bad_mask = np.zeros(nInst, dtype=bool)
    # bad_mask[NEG_IDX] = True

    # # reset any previous RSI-driven signals
    # position_dir[bad_mask] = 0

    # # long when RSI < 20, short when RSI > 80
    # rsi_long_mask  = bad_mask & (rsi_t < 20)
    # rsi_short_mask = bad_mask & (rsi_t > 80)

    # position_dir[rsi_long_mask]  = +1
    # position_dir[rsi_short_mask] = -1
#-------------------------------------------------------------
    

    # 4) Convert float to int for signal; Build the signal vector (same as position_dir, just more readable)
    signal_dir = position_dir.astype(int)

    NEG_IDX = [0, 2, 4, 5, 7, 10, 13, 15, 18, 20, 21, 25, 
               27, 28, 30, 31, 33, 34, 35, 39, 40, 42, 
               43, 46, 47, 48]
    signal_dir[NEG_IDX] = 0

    # … after building signal_dir …

    # 5) ONLY trade instruments whose signal just flipped
    if not np.array_equal(signal_dir, last_signal_dir):
        changed = np.where(signal_dir != last_signal_dir)[0]
        newPos  = currentPos.copy()

        for i in changed:
            if price_t[i] > 0:
                shares      = int(round(9000.0 / price_t[i]))
                newPos[i]   = signal_dir[i] * shares

        posLimits = (dlrPosLimit / price_t).astype(int)
        newPos    = np.clip(newPos, -posLimits, posLimits)
        
        currentPos      = newPos
        last_signal_dir = signal_dir.copy()

    return currentPos


