import numpy as np
import pandas as pd

nInst = 50
dlrPosLimit  = 10000

# Your persistent state
currentPos      = np.zeros(nInst, dtype=int)
position_dir    = np.zeros(nInst, dtype=int)  # –1/0/+1 signal
last_cross      = np.zeros(nInst, dtype=int)  # last crossover direction
last_signal_dir = np.zeros(nInst, dtype=int)  # ← ADDED: remembers previous signal_dir

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

    # 2) Grab “today” vs “yesterday” values
    price_t = prcSoFar[:, -1]
    macd_t  = macd[:, -1]; macd_y = macd[:, -2]
    sig_t   = signal[:, -1]; sig_y  = signal[:, -2]
    ema50_t = ema50[:, -1]

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


