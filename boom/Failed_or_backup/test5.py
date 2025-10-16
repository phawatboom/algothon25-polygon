import numpy as np

# main.py
# Refined 3-step price-action strategy with improved pivot/zones handling

# Global state retained across calls
global_state = {
    'initialized': False,
    'pivot_hist': {},    # per-inst list of recent pivots [('HH'/'LL', idx, price), ...]
    'zones': {},         # per-inst active zones
    'current_pos': None  # per-inst last returned positions
}

# Strategy parameters
LOOKBACK_SWING = 20         # bars to detect pivots
RR_MIN = 2.5                # minimum risk-reward
RISK_PCT = 0.005            # risk 0.5% of portfolio per trade (reduced)
PORTFOLIO_VALUE = 100000.0  # assumed equity
MAX_DOLLAR_PER_INST = 10000.0
ZONE_BUFFER = 0.02          # ±2% zone width (wider)


def _initialize(nInst):
    global_state['pivot_hist'] = {i: [] for i in range(nInst)}
    global_state['zones'] = {i: [] for i in range(nInst)}
    global_state['current_pos'] = np.zeros(nInst, dtype=int)
    global_state['initialized'] = True


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
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

        # 1) Pivot detection on bar t-2
        if t >= 3:
            idx = t - 2
            if prices[idx] > prices[idx-1] and prices[idx] > prices[idx+1]:
                pivots.append(('HH', idx, prices[idx]))
            elif prices[idx] < prices[idx-1] and prices[idx] < prices[idx+1]:
                pivots.append(('LL', idx, prices[idx]))
            # keep only recent 6
            if len(pivots) > 6:
                pivots.pop(0)

        # 2) Zone creation on LL->HH or HH->LL
        if len(pivots) >= 2:
            prev, curr = pivots[-2], pivots[-1]
            # demand zone after LL->HH
            if prev[0]=='LL' and curr[0]=='HH':
                zprice = prices[curr[1]-1]
                low = zprice*(1-ZONE_BUFFER)
                high = zprice*(1+ZONE_BUFFER)
                sl = low
                tp = curr[2] + (curr[2]-sl)*RR_MIN
                if (tp-curr[2])/(curr[2]-sl) >= RR_MIN:
                    zones.append({'type':'demand','low':low,'high':high,'sl':sl,'tp':tp})
            # supply zone after HH->LL
            if prev[0]=='HH' and curr[0]=='LL':
                zprice = prices[curr[1]-1]
                low = zprice*(1-ZONE_BUFFER)
                high = zprice*(1+ZONE_BUFFER)
                sl = high
                tp = curr[2] - (sl-curr[2])*RR_MIN
                if (sl-curr[2])>0 and abs(tp-curr[2])/(sl-curr[2]) >= RR_MIN:
                    zones.append({'type':'supply','low':low,'high':high,'sl':sl,'tp':tp})
            # remove only the two used pivots
            del pivots[:2]

        # 3) Prune zones on SL/TP hit
        new_zones = []
        for z in zones:
            price = last_prices[i]
            if z['type']=='demand':
                if price < z['sl'] or price >= z['tp']:
                    exit_today = True
                    continue
            else:
                if price > z['sl'] or price <= z['tp']:
                    exit_today = True
                    continue
            new_zones.append(z)
        global_state['zones'][i] = new_zones

        if exit_today:
            next_pos[i] = 0
            global_state['current_pos'][i] = 0
            continue

        # 4) Entry on most recent valid zone revisit
        entered = False
        for z in reversed(new_zones):
            price = last_prices[i]
            if z['low'] <= price <= z['high']:
                risk = (price - z['sl']) if z['type']=='demand' else (z['sl'] - price)
                if risk > 0:
                    size = int((PORTFOLIO_VALUE * RISK_PCT) / risk)
                    next_pos[i] = size if z['type']=='demand' else -size
                    entered = True
                break
        if not entered:
            next_pos[i] = global_state['current_pos'][i]

    # 5) Cap exposure
    for i in range(nInst):
        price = last_prices[i]
        pos = next_pos[i]
        if price>0:
            if abs(pos)*price > MAX_DOLLAR_PER_INST:
                next_pos[i] = int(np.sign(pos)*(MAX_DOLLAR_PER_INST//price))

    global_state['current_pos'] = next_pos
    return next_pos
