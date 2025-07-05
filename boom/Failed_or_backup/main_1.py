
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)


# def getMyPosition(prcSoFar):
#     global currentPos
#     (nins, nt) = prcSoFar.shape
#     if (nt < 2):
#         return np.zeros(nins)
#     lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
#     lNorm = np.sqrt(lastRet.dot(lastRet))
#     lastRet /= lNorm
#     rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
#     currentPos = np.array([int(x) for x in currentPos+rpos])
#     return currentPos

import numpy as np

# main.py

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Determine portfolio positions for 50 instruments based on historical prices.
    prcSoFar: NumPy array of shape (50, t) with price history up to current day t.
    Returns: NumPy array of 50 integers, the desired position (number of shares) for each instrument.
    """
    nInst, t = prcSoFar.shape
    # Initialize desired positions to zero
    position = np.zeros(nInst, dtype=int)
    if t < 2:
        # Not enough data to compute signals (no previous day); hold nothing
        return position

    # === 1. Compute Signals ===
    # Momentum signal: 20-day return (if available, else use shorter window)
    lookback = 20
    if t <= lookback:
        # If less than 20 days of data, use all available days minus one
        lookback = t - 1
    # Calculate past `lookback` days return for each instrument
    past_prices = prcSoFar[:, -lookback-1]   # price at start of lookback window
    latest_prices = prcSoFar[:, -1]          # price today (current day)
    # Avoid division by zero or invalid prices:
    with np.errstate(divide='ignore', invalid='ignore'):
        # percent return over the lookback period
        momentum_ret = latest_prices / past_prices - 1.0

    momentum_ret = np.nan_to_num(momentum_ret)  # replace any NaN from divide-by-zero with 0
    # Z-score (mean reversion) signal: how many std dev current price is from 20-day mean
    window = 30
    if t <= window:
        window = t - 1
    if window >= 1:
        recent_slice = prcSoFar[:, -window-1:-1]  # last `window` days *before* today
        mean_recent = np.mean(recent_slice, axis=1)
        std_recent = np.std(recent_slice, axis=1)
        # Avoid division by zero:
        std_recent = np.where(std_recent == 0, 1e-9, std_recent)
        z_score = (latest_prices - mean_recent) / std_recent
    else:
        # Not enough data for z-score, use zeros (no extreme signal)
        z_score = np.zeros(nInst)
    
    # === 2. Define signal thresholds ===
    # Momentum threshold: require at least ±15% return over lookback to act
    mom_threshold = 0.10
    # Mean reversion threshold: e.g. |z| > 2 for extreme (2 standard deviations)
    rev_threshold = 4

    # === 3. Compute target dollar position for each instrument based on signals ===
    # We will accumulate target dollar exposure for each stock from both momentum and reversion signals
    target_dollar = np.zeros(nInst)
    # Momentum positions:
    for i in range(nInst):
        if momentum_ret[i] > mom_threshold:
            # Strong upward momentum -> allocate positive (long) dollar position
            target_dollar[i] += 8000.0 * (momentum_ret[i] / mom_threshold)
            # ^ allocate base $5k, scaled by how much signal exceeds threshold
        elif momentum_ret[i] < -mom_threshold:
            # Strong downward momentum -> allocate negative (short) dollar position
            target_dollar[i] -= 8000.0 * (abs(momentum_ret[i]) / mom_threshold)
    # Mean reversion positions:
    for i in range(nInst):
        if target_dollar[i] == 0:  # only apply contrarian signal if no momentum position (to avoid conflict)
            if z_score[i] < -rev_threshold:
                # Oversold (price far below recent mean) -> contrarian long
                target_dollar[i] += 5000.0 * (abs(z_score[i]) / rev_threshold)
            elif z_score[i] > rev_threshold:
                # Overbought (price far above mean) -> contrarian short
                target_dollar[i] -= 5000.0 * (abs(z_score[i]) / rev_threshold)

    # === 4. Volatility scaling ===
    # Compute 20-day volatility (standard deviation of daily returns) for each instrument
    if t > 1:
        # daily returns for last 20 days (or available period)
        ret_window = min(20, t-1)
        recent_prices = prcSoFar[:, -ret_window-1:]  # prices including today and ret_window days prior
        # Compute daily returns over this window
        daily_ret = recent_prices[:, 1:] / recent_prices[:, :-1] - 1.0
        vol = np.nanstd(daily_ret, axis=1)  # standard deviation of daily returns
    else:
        vol = np.zeros(nInst)
    # Scale target dollar positions by ratio of target volatility (e.g. 1% per day) to instrument vol
    target_vol = 0.01  # desired volatility per stock position (~1% daily)
    for i in range(nInst):
        if vol[i] > 1e-6:
            target_dollar[i] *= (target_vol / vol[i])
    # This means a more volatile stock gets its dollar position reduced, and vice versa:contentReference[oaicite:23]{index=23}

    # === 5. Convert dollar targets to integer share positions ===
    for i in range(nInst):
        # Enforce position limit of $10,000 per stock
        if target_dollar[i] > 10000:
            target_dollar[i] = 10000
        elif target_dollar[i] < -10000:
            target_dollar[i] = -10000
        price = latest_prices[i]
        if price <= 0 or np.isnan(price):
            position[i] = 0
        else:
            # Use floor division to get integer shares that do not exceed target_dollar
            shares = np.floor(abs(target_dollar[i]) / price)
            if target_dollar[i] < 0:
                shares = -shares
            position[i] = int(shares)
    return position
