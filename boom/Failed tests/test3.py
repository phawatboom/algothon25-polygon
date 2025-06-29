import numpy as np

# main.py
# Stateful implementation of 3-step price-action for getMyPosition

# Global state retained across calls
global_state = {
    'initialized': False,
    'pivot_hist': {},    # per-inst list of recent pivots [('HH'/'LL', idx, price), ...]
    'zones': {},         # per-inst active zones: list of zone dicts
    'current_pos': None  # per-inst last returned positions
}

# Strategy parameters
LOOKBACK_SWING = 20        # bars for pivot detection
RR_MIN = 2.5               # minimum acceptable RR
RISK_PCT = 0.01            # risk 1% of portfolio per trade
PORTFOLIO_VALUE = 100000.0 # assumed portfolio value
MAX_DOLLAR_PER_INST = 10000.0
ZONE_BUFFER = 0.005        # ±0.5% around consolidation price


def _initialize(nInst):
    """Set up empty state arrays for each instrument."""
    global_state['pivot_hist'] = {i: [] for i in range(nInst)}
    global_state['zones'] = {i: [] for i in range(nInst)}
    global_state['current_pos'] = np.zeros(nInst, dtype=int)
    global_state['initialized'] = True


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Returns integer positions for 50 instruments based on:
      1) HH/LL pivot detection
      2) Supply/demand zone creation at pivot sequences
      3) Entry on zone revisit if RR >= RR_MIN
    """
    nInst, t = prcSoFar.shape
    if not global_state['initialized']:
        _initialize(nInst)

    last_prices = prcSoFar[:, -1]
    target_pos = np.zeros(nInst, dtype=int)

    for i in range(nInst):
        prices = prcSoFar[i]
        pivots = global_state['pivot_hist'][i]
        zones = global_state['zones'][i]

        # 1) Pivot detection at index t-2
        if t >= 3:
            idx = t - 2
            if prices[idx] > prices[idx-1] and prices[idx] > prices[idx+1]:
                pivots.append(('HH', idx, prices[idx]))
            elif prices[idx] < prices[idx-1] and prices[idx] < prices[idx+1]:
                pivots.append(('LL', idx, prices[idx]))
            # keep only last 4
            if len(pivots) > 4:
                pivots.pop(0)

        # 2) Zone creation on LL->HH or HH->LL sequences
        if len(pivots) >= 2:
            prev, curr = pivots[-2], pivots[-1]
            # Demand zone after LL->HH
            if prev[0]=='LL' and curr[0]=='HH':
                zone_price = prices[curr[1]-1]
                low = zone_price * (1 - ZONE_BUFFER)
                high = zone_price * (1 + ZONE_BUFFER)
                sl = low
                tp = curr[2] + (curr[2] - sl) * RR_MIN
                if (tp - curr[2]) / (curr[2] - sl) >= RR_MIN:
                    zones.append({'type':'demand','low':low,'high':high,'sl':sl,'tp':tp})
            # Supply zone after HH->LL
            if prev[0]=='HH' and curr[0]=='LL':
                zone_price = prices[curr[1]-1]
                low = zone_price * (1 - ZONE_BUFFER)
                high = zone_price * (1 + ZONE_BUFFER)
                sl = high
                tp = curr[2] - (sl - curr[2]) * RR_MIN
                if (sl - curr[2]) > 0 and abs(tp - curr[2]) / (sl - curr[2]) >= RR_MIN:
                    zones.append({'type':'supply','low':low,'high':high,'sl':sl,'tp':tp})
            # clear pivots to prevent duplicate zones
            pivots.clear()

        # 3) Prune invalid zones (SL hit)
        valid = []
        for z in zones:
            if z['type']=='demand' and last_prices[i] < z['sl']:
                continue
            if z['type']=='supply' and last_prices[i] > z['sl']:
                continue
            valid.append(z)
        global_state['zones'][i] = valid

        # 4) Entry logic: revisit most recent valid zone
        entered = False
        for z in reversed(valid):
            if z['low'] <= last_prices[i] <= z['high']:
                # compute risk per share
                if z['type']=='demand':
                    risk = last_prices[i] - z['sl']
                else:
                    risk = z['sl'] - last_prices[i]
                if risk > 0:
                    size = int((PORTFOLIO_VALUE * RISK_PCT) / risk)
                    target_pos[i] = size if z['type']=='demand' else -size
                    entered = True
                break
        if not entered:
            target_pos[i] = global_state['current_pos'][i]

    # 5) Enforce max $10k exposure per instrument
    for i in range(nInst):
        price = last_prices[i]
        pos = target_pos[i]
        if price > 0:
            exposure = abs(pos) * price
            if exposure > MAX_DOLLAR_PER_INST:
                target_pos[i] = int(np.sign(pos) * (MAX_DOLLAR_PER_INST // price))

    # update state
    global_state['current_pos'] = target_pos
    return target_pos
