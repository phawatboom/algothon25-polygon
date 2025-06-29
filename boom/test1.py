import numpy as np

# main.py

# from eval import loadPrices as loadPrices 
# from eval import calcPL as calcPL

# nInst = 50
# nt = 0
# commRate = 0.0005
# dlrPosLimit = 10000

# pricesFile="./prices.txt"
# prcAll = loadPrices(pricesFile)
# print ("Loaded %d instruments for %d days" % (nInst, nt))

# (meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll,200)
# score = meanpl - 0.1*plstd
# print ("=====")
# print ("mean(PL): %.1lf" % meanpl)
# print ("return: %.5lf" % ret)
# print ("StdDev(PL): %.2lf" % plstd)
# print ("annSharpe(PL): %.2lf " % sharpe)
# print ("totDvolume: %.0lf " % dvol)
# print ("Score: %.2lf" % score)


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Implements the 3-step price-action strategy across 50 instruments.
    1) Identify up/down trend by higher-highs / lower-lows on a lookback window
    2) Mark the supply/demand zone at the bar before the impulse
    3) Enter only when price is inside that zone with RR >= 2.5

    prcSoFar: (nInst x t) price history
    returns: integer share positions per instrument
    """
    nInst, t = prcSoFar.shape
    # initialize zero positions
    positions = np.zeros(nInst, dtype=int)
    # Strategy parameters
    lookback = 20       # bars to detect swings
    rr_min = 3       # minimum risk-reward
    risk_pct = 0.01     # 1% of portfolio per trade
    portfolio_value = 100000.0
    
    for i in range(nInst):
        prices = prcSoFar[i]
        if t < lookback + 2:
            continue   # not enough bars
        # --- 1) Determine trend via swing detection ---
        recent_highs = prices[-lookback-1:-1]
        recent_lows  = prices[-lookback-1:-1]
        swing_high = np.max(recent_highs)
        swing_low  = np.min(recent_lows)
        last_price = prices[-1]
        # check new swing high / low
        is_hh = last_price >= swing_high
        is_ll = last_price <= swing_low
        # derive trend
        trend = None
        if is_hh and not is_ll:
            trend = 'up'
        elif is_ll and not is_hh:
            trend = 'down'
        else:
            continue  # no clear directional signal
        
        # --- 2) Define zone at the bar before the impulse ---
        zone_high = prices[-2]
        zone_low  = prices[-2]
        # use tiny buffer to avoid exact-equality miss
        buff = 1e-6
        
        # --- 3) Check entry condition ---
        if trend == 'up' and zone_low - buff <= last_price <= zone_high + buff:
            # compute stop-loss and take-profit
            sl = zone_low * 0.995
            tp = last_price + (last_price - sl) * rr_min
            # dollar risk per trade
            dollar_risk = (last_price - sl) * 1.0  # per share risk
            if dollar_risk <= 0:
                continue
            # position sizing (1% risk of port)
            size = int((portfolio_value * risk_pct) / dollar_risk)
            positions[i] = size
        
        if trend == 'down' and zone_low - buff <= last_price <= zone_high + buff:
            sl = zone_high * 1.005
            tp = last_price - (sl - last_price) * rr_min
            dollar_risk = (sl - last_price)
            if dollar_risk <= 0:
                continue
            size = int((portfolio_value * risk_pct) / dollar_risk)
            positions[i] = -size

    return positions
