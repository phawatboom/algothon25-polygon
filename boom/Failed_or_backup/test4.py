import numpy as np

# main.py
# Enhanced 3-step price-action strategy with proper exit logic for getMyPosition

# Global state retained across calls
global_state = {
    'initialized': False,
    'pivot_hist': {},    # per-inst list of recent pivots [('HH'/'LL', idx, price), ...]
    'zones': {},         # per-inst active zones: list of {'type','low','high','sl','tp'}
    'current_pos': None  # per-inst last returned positions
}

# Strategy parameters
LOOKBACK_SWING = 20        # bars for pivot detection window
RR_MIN = 2.5               # minimum acceptable risk-reward
RISK_PCT = 0.01            # risk 1% of portfolio per trade
PORTFOLIO_VALUE = 100000.0 # assumed total equity
MAX_DOLLAR_PER_INST = 10000.0  # exposure cap per instrument
ZONE_BUFFER = 0.005        # ±0.5% zone width


def _initialize(nInst):
    """Initialize empty state for each instrument."""
    global_state['pivot_hist'] = {i: [] for i in range(nInst)}
    global_state['zones'] = {i: [] for i in range(nInst)}
    global_state['current_pos'] = np.zeros(nInst, dtype=int)
    global_state['initialized'] = True


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Returns target positions per instrument:
      1) Detect HH/LL pivots to spot trend flips
      2) Create supply/demand zones after LL->HH or HH->LL
      3) Enter on zone revisit only if RR >= RR_MIN
      4) Exit when SL or TP are hit, then prune that zone
    """
    nInst, t = prcSoFar.shape
    if not global_state['initialized']:
        _initialize(nInst)

    last_prices = prcSoFar[:, -1]
    next_pos = np.zeros(nInst, dtype=int)

    for i in range(nInst):
        prices = prcSoFar[i]
        pivots = global_state['pivot_hist'][i]
        zones = global_state['zones'][i]
        exit_today = False

        # ---- 1) Pivot detection at bar (t-2) ----
        if t >= 3:
            idx = t - 2
            if prices[idx] > prices[idx-1] and prices[idx] > prices[idx+1]:
                pivots.append(('HH', idx, prices[idx]))
            elif prices[idx] < prices[idx-1] and prices[idx] < prices[idx+1]:
                pivots.append(('LL', idx, prices[idx]))
            if len(pivots) > 4:
                pivots.pop(0)

        # ---- 2) Zone creation on pivot sequence ----
        if len(pivots) >= 2:
            prev, curr = pivots[-2], pivots[-1]
            # Demand zone if LL -> HH
            if prev[0]=='LL' and curr[0]=='HH':
                zprice = prices[curr[1]-1]
                low = zprice*(1-ZONE_BUFFER)
                high = zprice*(1+ZONE_BUFFER)
                sl = low
                tp = curr[2] + (curr[2]-sl)*RR_MIN
                if (tp-curr[2])/(curr[2]-sl) >= RR_MIN:
                    zones.append({'type':'demand','low':low,'high':high,'sl':sl,'tp':tp})
            # Supply zone if HH -> LL
            if prev[0]=='HH' and curr[0]=='LL':
                zprice = prices[curr[1]-1]
                low = zprice*(1-ZONE_BUFFER)
                high = zprice*(1+ZONE_BUFFER)
                sl = high
                tp = curr[2] - (sl-curr[2])*RR_MIN
                if (sl-curr[2])>0 and abs(tp-curr[2])/(sl-curr[2]) >= RR_MIN:
                    zones.append({'type':'supply','low':low,'high':high,'sl':sl,'tp':tp})
            pivots.clear()

        # ---- 3) Prune zones on SL or TP ----
        new_zones = []
        for z in zones:
            # Stop-loss hit
            if z['type']=='demand' and last_prices[i] < z['sl']:
                exit_today = True
                continue
            if z['type']=='supply' and last_prices[i] > z['sl']:
                exit_today = True
                continue
            # Take-profit hit
            if z['type']=='demand' and last_prices[i] >= z['tp']:
                exit_today = True
                continue
            if z['type']=='supply' and last_prices[i] <= z['tp']:
                exit_today = True
                continue
            new_zones.append(z)
        global_state['zones'][i] = new_zones

        # If exit occurred, clear position and skip new entry
        if exit_today:
            next_pos[i] = 0
            global_state['current_pos'][i] = 0
            continue

        # ---- 4) Entry: revisit valid zone ----
        entered = False
        for z in reversed(new_zones):
            if z['low'] <= last_prices[i] <= z['high']:
                # compute per-share risk
                if z['type']=='demand':
                    risk = last_prices[i] - z['sl']
                else:
                    risk = z['sl'] - last_prices[i]
                if risk>0:
                    size = int((PORTFOLIO_VALUE * RISK_PCT)/risk)
                    next_pos[i] = size if z['type']=='demand' else -size
                    entered = True
                break
        if not entered:
            # hold prior position
            next_pos[i] = global_state['current_pos'][i]

    # ---- 5) Enforce max exposure per inst ----
    for i in range(nInst):
        price = last_prices[i]
        pos = next_pos[i]
        if price>0:
            exposure = abs(pos)*price
            if exposure>MAX_DOLLAR_PER_INST:
                next_pos[i] = int(np.sign(pos)*(MAX_DOLLAR_PER_INST//price))

    global_state['current_pos'] = next_pos
    return next_pos
