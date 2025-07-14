import numpy as np
import pandas as pd

nInst = 50
dlrPosLimit  = 10000

# Your persistent state
currentPos      = np.zeros(nInst, dtype=int)
position_dir    = np.zeros(nInst, dtype=int)  # –1/0/+1 signal
last_cross      = np.zeros(nInst, dtype=int)  # last crossover direction
last_signal_dir = np.zeros(nInst, dtype=int)  # ← ADDED: remembers previous signal_dir

# New state variables for negative instruments
entry_prices_neg = np.zeros(nInst)             # Entry price for negative instruments
days_in_trade_neg = np.zeros(nInst, dtype=int)  # Days since entry for negative instruments
days_since_tp_neg = 101 * np.ones(nInst, dtype=int)  # Days since last take-profit for negatives

best_price_neg = np.zeros(nInst)                # Best price since entry
take_profit_level = np.zeros(nInst)              # First profit target price
second_tp_level = np.zeros(nInst)                # Second profit target price
stop_loss_level = np.zeros(nInst)                # Stop-loss price
trailing_stop_level = np.zeros(nInst)            # Current trailing stop price
half_profit_taken = np.zeros(nInst, dtype=bool)  # Track if half position was taken

# Track crossover signals for delayed entry
crossover_signals = np.zeros((nInst, 3), dtype=int)  # [0]=today, [1]=yesterday, [2]=two_days_ago

NEG_IDX = [0, 2, 4, 5, 7, 10, 13, 15, 18, 20, 21, 25, 
               27, 28, 30, 31, 33, 34, 35, 39, 40, 42, 
               43, 46, 47, 48]

FIRST_TP_PERCENT = 0.15
SECOND_TP_MULTIPLIER = 2.0
STOP_LOSS_PERCENT = 0.05
TRAILING_STOP_PERCENT = 0.02

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
    global entry_prices_neg, days_in_trade_neg, days_since_tp_neg
    global best_price_neg, take_profit_level, second_tp_level, stop_loss_level
    global trailing_stop_level, half_profit_taken, crossover_signals

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
    price_t = prcSoFar[:, -1]
    price_y = prcSoFar[:, -2]
    macd_t  = macd[:, -1]; macd_y = macd[:, -2]
    sig_t   = signal[:, -1]; sig_y  = signal[:, -2]
    rsi_t  = rsi_all[:, -1]
    ema50_t = ema50[:, -1]; ema50_y = ema50[:, -2]
    ema12_t = ema12[:, -1]; ema12_y = ema12[:, -2]

# Update cool-down counters for negative instruments
    for i in NEG_IDX:
        if currentPos[i] == 0 and days_since_tp_neg[i] <= 100:
            days_since_tp_neg[i] += 1

    # Check exit conditions for negative instruments
    for i in NEG_IDX:
        if currentPos[i] != 0:
            days_in_trade_neg[i] += 1
            
            # Calculate returns based on position type
            if currentPos[i] > 0:  # Long position
                ret = (price_t[i] - entry_prices_neg[i]) / entry_prices_neg[i]
            else:  # Short position
                ret = (entry_prices_neg[i] - price_t[i]) / entry_prices_neg[i]
                
            # Check exit conditions
            if ret >= 0.8:  # Take-profit
                position_dir[i] = 0
                days_since_tp_neg[i] = 0
                entry_prices_neg[i] = 0
                days_in_trade_neg[i] = 0
            elif ret <= -0.02 or days_in_trade_neg[i] >= 50:  # Stop-loss or timeout
                position_dir[i] = 0
                entry_prices_neg[i] = 0
                days_in_trade_neg[i] = 0


    # 3) MACD crossover logic → position_dir / last_cross
    for i in range(nins):
#--------------------- EMA 50 strategy for negative returns instruments -------------------------------------
        if i in NEG_IDX:
            if currentPos[i] == 0 and days_since_tp_neg[i] > 100:

                # Long crossover with RSI < 20
                if (
                    # (macd_y[i] < sig_y[i]) 
                    # and (macd_t[i] > sig_t[i]) 
                    # and (macd_y[i] < 0) and (sig_y[i] < 0)
                    # and (macd_t[i] < 0) and (sig_t[i] < 0)
                    # and (rsi_t[i] < 30)  # RSI condition\
                    ema12_y[i] < ema50_y[i] and ema12_t[i] > ema50_t[i]
                    # (price_y[i] < ema50_y[i]) and (price_t[i] > ema50_t[i]) #ema50 condition
                    # and last_cross[i] != +1
                ):
                    position_dir[i] = +1
                    entry_prices_neg[i] = price_t[i]
                    days_in_trade_neg[i] = 0
                    # last_cross[i] = +1
                    
                # Short crossover with RSI > 80
                elif (
                    # (macd_y[i] > sig_y[i]) 
                    # and (macd_t[i] < sig_t[i]) 
                    # and (macd_y[i] > 0) and (sig_y[i] > 0)
                    # and (macd_t[i] > 0) and (sig_t[i] > 0)
                    # and (rsi_t[i] > 70)  # RSI condition
                    ema12_y[i] > ema50_y[i] and ema12_t[i] < ema50_t[i]
                    # (price_y[i] > ema50_y[i]) and (price_t[i] < ema50_t[i]) #ema50 condition
                    # and last_cross[i] != -1
                ):
                    position_dir[i] = -1
                    entry_prices_neg[i] = price_t[i]
                    days_in_trade_neg[i] = 0
                    # last_cross[i] = -1
#----------------------------------------------------------------------------------------------------
        # For positive return instruments: Original MACD logic
        else:
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

    # NEG_IDX = [0, 2, 4, 5, 7, 10, 13, 15, 18, 20, 21, 25, 
    #            27, 28, 30, 31, 33, 34, 35, 39, 40, 42, 
    #            43, 46, 47, 48]

    NEG_IDXX = [2, 7, 13, 15, 18, 21, 28, 35, 43, 47, 48]
    signal_dir[NEG_IDXX] = 0

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


