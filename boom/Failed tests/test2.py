import numpy as np

# main.py
# Production-ready 3-step price-action strategy stateful implementation

# Global state retained across calls
global_state = {
    'initialized': False,
    'pivot': {},       # per-inst swing pivots and trend
    'zones': {},       # per-inst active supply/demand zones
    'current_pos': None
}

# Strategy parameters
LOOKBACK_SWING = 20        # bars for swing detection
RR_MIN = 2.5               # minimum risk:reward
RISK_PCT = 0.01            # 1% of portfolio per trade
PORTFOLIO_VALUE = 100000.0 # assumed portfolio value
MAX_DOLLAR_PER_INST = 10000.0


def _initialize(nInst):
    """Initialize state for each instrument"""
    global_state['pivot'] = {
        i: {'last_hh': None, 'last_ll': None, 'trend': None}
        for i in range(nInst)
    }
    global_state['zones'] = {i: [] for i in range(nInst)}
    global_state['current_pos'] = np.zeros(nInst, dtype=int)
    global_state['initialized'] = True


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Determine target positions for each instrument based on:
      1) HH/HL & LL/LH swing detection to set trend
      2) Mark supply/demand zones at pivot bars
      3) Enter only when price revisits most recent zone with RR >= RR_MIN
    Maintains state across invocations for pivots, zones, and positions.
    """
    nInst, t = prcSoFar.shape
    if not global_state['initialized']:
        _initialize(nInst)

    last_prices = prcSoFar[:, -1]
    target_positions = np.zeros(nInst, dtype=int)

    for i in range(nInst):
        prices = prcSoFar[i]
        state = global_state['pivot'][i]
        zones = global_state['zones'][i]

        # Require at least 3 bars to detect a pivot
        if t < 3:
            target_positions[i] = global_state['current_pos'][i]
            continue

        pivot_idx = t - 2  # compare bar t-2 with its neighbors
        # ---- Swing High Detection ----
        if prices[pivot_idx] > prices[pivot_idx-1] and prices[pivot_idx] > prices[pivot_idx+1]:
            # Ensure we had a prior swing low to confirm uptrend
            if state['last_ll'] is not None and pivot_idx > state['last_ll'][0]:
                state['last_hh'] = (pivot_idx, prices[pivot_idx])
                state['trend'] = 'up'
                # Define demand zone at consolidation bar before impulse
                zone_price = prices[pivot_idx-1]
                sl = zone_price * 0.995
                tp = prices[pivot_idx] + (prices[pivot_idx] - sl) * RR_MIN
                # Only add zone if RR filter passes
                if (tp - prices[pivot_idx]) / (prices[pivot_idx] - sl) >= RR_MIN:
                    zones.append({
                        'type': 'demand',
                        'low': zone_price,
                        'high': zone_price,
                        'entry_idx': pivot_idx,
                        'sl': sl,
                        'tp': tp
                    })
        # ---- Swing Low Detection ----
        elif prices[pivot_idx] < prices[pivot_idx-1] and prices[pivot_idx] < prices[pivot_idx+1]:
            # Ensure we had a prior swing high to confirm downtrend
            if state['last_hh'] is not None and pivot_idx > state['last_hh'][0]:
                state['last_ll'] = (pivot_idx, prices[pivot_idx])
                state['trend'] = 'down'
                zone_price = prices[pivot_idx-1]
                sl = zone_price * 1.005
                tp = prices[pivot_idx] - (sl - prices[pivot_idx]) * RR_MIN
                if (sl - prices[pivot_idx]) > 0 and (abs(tp - prices[pivot_idx]) / (sl - prices[pivot_idx])) >= RR_MIN:
                    zones.append({
                        'type': 'supply',
                        'low': zone_price,
                        'high': zone_price,
                        'entry_idx': pivot_idx,
                        'sl': sl,
                        'tp': tp
                    })

        # ---- Prune invalid zones ----
        valid = []
        for z in zones:
            if state['trend'] == 'up' and last_prices[i] < z['sl']:
                continue  # SL hit
            if state['trend'] == 'down' and last_prices[i] > z['sl']:
                continue
            valid.append(z)
        global_state['zones'][i] = valid

        # ---- Entry Logic: only most recent valid zone ----
        entered = False
        for z in reversed(valid):
            if z['type'] == 'demand' and state['trend'] == 'up' and z['low'] <= last_prices[i] <= z['high']:
                risk_per_share = last_prices[i] - z['sl']
                if risk_per_share > 0:
                    size = int((PORTFOLIO_VALUE * RISK_PCT) / risk_per_share)
                    target_positions[i] = size
                    entered = True
                break
            if z['type'] == 'supply' and state['trend'] == 'down' and z['low'] <= last_prices[i] <= z['high']:
                risk_per_share = z['sl'] - last_prices[i]
                if risk_per_share > 0:
                    size = int((PORTFOLIO_VALUE * RISK_PCT) / risk_per_share)
                    target_positions[i] = -size
                    entered = True
                break
        if not entered:
            target_positions[i] = global_state['current_pos'][i]

    # ---- Enforce max $10k exposure per instrument ----
    for i in range(nInst):
        price = last_prices[i]
        pos = target_positions[i]
        if price > 0:
            exposure = abs(pos) * price
            if exposure > MAX_DOLLAR_PER_INST:
                target_positions[i] = int(np.sign(pos) * (MAX_DOLLAR_PER_INST // price))

    # Update state and return
    global_state['current_pos'] = target_positions
    return target_positions
