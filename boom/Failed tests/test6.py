import numpy as np

# main.py
# Hybrid strategy: Momentum + Mean Reversion + Price-Action Zones
# Incorporating insights from multiple sources (momentum, Bollinger, supply/demand)

# Global state retained across calls
global_state = {
    'initialized': False,
    'zones': {},         # per-inst supply/demand zones
    'current_pos': None  # last positions
}

# Parameters
MOM_LOOKBACK = 10          # momentum lookback
BB_WINDOW = 20             # Bollinger band window
BB_STD = 2.0               # Bollinger band width (stddev)
RR_MIN = 2.5               # risk-reward for zones
RISK_PCT = 0.01            # 1% equity risk per zone trade
PORTFOLIO_VALUE = 100000.0
MAX_DOLLAR_PER_INST = 10000.0
MOM_WEIGHT = 0.5           # weight for momentum signal
REV_WEIGHT = 0.5           # weight for mean-reversion signal
ZONE_BUFFER = 0.01         # ±1% around consolidation


def _initialize(nInst):
    global_state['zones'] = {i: [] for i in range(nInst)}
    global_state['current_pos'] = np.zeros(nInst, dtype=int)
    global_state['initialized'] = True


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    nInst, t = prcSoFar.shape
    if not global_state['initialized']:
        _initialize(nInst)

    last_prices = prcSoFar[:, -1]
    target_pos = np.zeros(nInst, dtype=int)

    # === Factor signals ===
    mom = np.zeros(nInst)
    rev = np.zeros(nInst)
    # momentum: % return over MOM_LOOKBACK
    if t > MOM_LOOKBACK:
        mom = last_prices / prcSoFar[:, -MOM_LOOKBACK-1] - 1.0
    # mean reversion: z-score vs BB bands
    if t > BB_WINDOW:
        slice = prcSoFar[:, -BB_WINDOW:]
        m = slice.mean(axis=1)
        s = slice.std(axis=1)
        rev = (m - last_prices) / (s + 1e-9)

    # combined factor position (dollar): long on momentum & reversal
    factor_dollar = MOM_WEIGHT * mom * PORTFOLIO_VALUE * 0.1 + REV_WEIGHT * rev * PORTFOLIO_VALUE * 0.1

    # convert to shares
    for i in range(nInst):
        price = last_prices[i]
        if price <= 0: continue
        dollars = np.clip(factor_dollar[i], -MAX_DOLLAR_PER_INST, MAX_DOLLAR_PER_INST)
        shares = int(dollars / price)
        target_pos[i] = shares

    # === Price-Action Zones ===
    # very simple: mark zones on 5-bar pivot
    zones = global_state['zones']
    for i in range(nInst):
        prices = prcSoFar[i]
        if t >= 6:
            # pivot at t-3
            idx = t - 3
            if prices[idx] > prices[idx-1] and prices[idx] > prices[idx+1]:
                # supply zone
                z = prices[idx]
                low = z*(1-ZONE_BUFFER)
                high = z*(1+ZONE_BUFFER)
                sl = high
                tp = z - (high-z)*RR_MIN
                zones[i].append({'type':'supply','low':low,'high':high,'sl':sl,'tp':tp})
            if prices[idx] < prices[idx-1] and prices[idx] < prices[idx+1]:
                # demand zone
                z = prices[idx]
                low = z*(1-ZONE_BUFFER)
                high = z*(1+ZONE_BUFFER)
                sl = low
                tp = z + (z-low)*RR_MIN
                zones[i].append({'type':'demand','low':low,'high':high,'sl':sl,'tp':tp})
            # prune old
            global_state['zones'][i] = [z for z in zones[i] if t - z.get('entry_idx', idx) < 50]

        # zone entry override: if price in zone, use zone sizing
        for z in zones[i]:
            price = last_prices[i]
            if z['low'] <= price <= z['high']:
                risk = abs(price - z['sl'])
                if risk>0:
                    size = int((PORTFOLIO_VALUE*RISK_PCT)/risk)
                    target_pos[i] = size if z['type']=='demand' else -size
                break

    # update and return
    global_state['current_pos'] = target_pos
    return target_pos
