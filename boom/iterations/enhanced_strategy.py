import numpy as np
import pandas as pd

nInst = 50
dlrPosLimit  = 10000

FULL_POS_DOLLARS = 10000.0
HALF_POS_DOLLARS = 6000.0

# Your persistent state
currentPos      = np.zeros(nInst, dtype=int)
position_dir    = np.zeros(nInst, dtype=int)  # –1/0/+1 signal
last_cross      = np.zeros(nInst, dtype=int)  # last crossover direction
last_signal_dir = np.zeros(nInst, dtype=int)  # ← ADDED: remembers previous signal_dir

# New state variables for negative instruments
entry_prices_neg = np.zeros(nInst)             # Entry price for negative instruments
days_in_trade_neg = np.zeros(nInst, dtype=int)  # Days since entry for negative instruments

# CHANGED: Separate cool-down trackers for long/short
days_since_tp_long = 101 * np.ones(nInst, dtype=int)  # Days since last long take-profit
days_since_tp_short = 101 * np.ones(nInst, dtype=int)  # Days since last short take-profit


best_price_neg = np.zeros(nInst)                # Best price since entry
take_profit_level = np.zeros(nInst)              # First profit target price
second_tp_level = np.zeros(nInst)                # Second profit target price
stop_loss_level = np.zeros(nInst)                # Stop-loss price
trailing_stop_level = np.zeros(nInst)            # Current trailing stop price
half_profit_taken = np.zeros(nInst, dtype=bool)  # Track if half position was taken

# Trading parameters

FIRST_TP_PERCENT = 0.15
SECOND_TP_MULTIPLIER = 2
STOP_LOSS_PERCENT = 0.03
TRAILING_STOP_PERCENT = 0.02
COOLDOWN_DAYS = 25     # Days to wait after taking full profit
MAX_HOLD_DAYS = 60    # Maximum days to hold a position
TRAILING_UPDATE_FREQ = 5  # Frequency to update trailing stop (days)
ENTRY_DELAY = 2         # Days to wait before entering after signal

# Track crossover signals for delayed entry
crossover_signals = np.zeros((nInst, ENTRY_DELAY + 1), dtype=int)  # [0]=today, [1]=yesterday, [2]=two_days_ago

# EMA_STRATEGY_INSTS = [0, 2, 4, 5, 7, 10, 13, 15, 18, 20, 21, 25, 
#                27, 28, 30, 31, 33, 34, 35, 39, 40, 42, 
#                43, 46, 47, 48]

EMA_STRATEGY_INSTS = [0, 2, 4, 5, 10, 13, 20, 25, 
               27, 30, 33, 39, 42, 
               43, 46, 47]

RSI_STRATEGY_INSTS = [7, 15, 18, 21, 28, 31, 34, 35, 40, 48]

def compute_RSI(prices: pd.DataFrame, period: int = 30) -> np.ndarray:
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

def reset_trade_state(i):
    entry_prices_neg[i] = 0
    best_price_neg[i] = 0
    days_in_trade_neg[i] = 0
    half_profit_taken[i] = False

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global currentPos, position_dir, last_cross, last_signal_dir
    global entry_prices_neg, days_in_trade_neg, days_since_tp_long, days_since_tp_short
    global best_price_neg, take_profit_level, second_tp_level, stop_loss_level
    global trailing_stop_level, half_profit_taken, crossover_signals

    nins, nt = prcSoFar.shape
    if nt < 50:
        return currentPos

    # 1) Compute EMAs
    df     = pd.DataFrame(prcSoFar)
    ema50  = df.T.ewm(span=50, adjust=False).mean().T.to_numpy()
    ema12  = df.T.ewm(span=12, adjust=False).mean().T.to_numpy()
    ema15  = df.T.ewm(span=15, adjust=False).mean().T.to_numpy()
    ema26  = df.T.ewm(span=26, adjust=False).mean().T.to_numpy()
    ema30 = df.T.ewm(span=30, adjust=False).mean().T.to_numpy()
    ema200 = df.T.ewm(span=200, adjust=False).mean().T.to_numpy()
    macd    = ema12 - ema26
    signal  = pd.DataFrame(macd).T.ewm(span=9, adjust=False).mean().T.to_numpy()

    # 2) Compute RSI on closing prices
    rsi_all = compute_RSI(df)  # shape (nInst, nt)

    # 2) Grab “today” vs “yesterday” values
    price_t = prcSoFar[:, -1]
    price_y = prcSoFar[:, -2]
    macd_t  = macd[:, -1]; macd_y = macd[:, -2]
    sig_t   = signal[:, -1]; sig_y  = signal[:, -2]
    rsi_t  = rsi_all[:, -1]; rsi_y = rsi_all[:, -2]
    ema50_t = ema50[:, -1]; ema50_y = ema50[:, -2]
    ema12_t = ema12[:, -1]; ema12_y = ema12[:, -2]
    ema15_t = ema15[:, -1]; ema15_y = ema15[:, -2]
    ema30_t = ema30[:, -1]; ema30_y = ema30[:, -2]
    ema200_t = ema200[:, -1]; ema200_y = ema200[:, -2]

# Update cool-down counters for negative instruments
    for i in EMA_STRATEGY_INSTS + RSI_STRATEGY_INSTS:
        if currentPos[i] == 0:
            if days_since_tp_long[i] <= COOLDOWN_DAYS:
                days_since_tp_long[i] += 1
            # Update short cooldown
            if days_since_tp_short[i] <= COOLDOWN_DAYS:
                days_since_tp_short[i] += 1

    # Check exit conditions for negative instruments
    for i in EMA_STRATEGY_INSTS + RSI_STRATEGY_INSTS:
        if currentPos[i] != 0:
            days_in_trade_neg[i] += 1
            
            # Calculate returns based on position type
            if currentPos[i] > 0:  # Long position
                current_return = (price_t[i] - entry_prices_neg[i]) / entry_prices_neg[i]
                # Update best price
                if price_t[i] > best_price_neg[i]:
                    best_price_neg[i] = price_t[i]
                    
                # Update trailing stop every 10 days
                if days_in_trade_neg[i] % TRAILING_UPDATE_FREQ == 0 and half_profit_taken[i]:
                    trailing_stop_level[i] = best_price_neg[i] * (1 - TRAILING_STOP_PERCENT)

            else:  # Short position
                current_return = (entry_prices_neg[i] - price_t[i]) / entry_prices_neg[i]

                # Update best price (lowest for shorts)
                if price_t[i] < best_price_neg[i] or best_price_neg[i] == 0:
                    best_price_neg[i] = price_t[i]
                    
                # Update trailing stop every 10 days
                if days_in_trade_neg[i] % TRAILING_UPDATE_FREQ == 0 and half_profit_taken[i]:
                    trailing_stop_level[i] = best_price_neg[i] * (1 + TRAILING_STOP_PERCENT)

            # Check exit conditions
            if not half_profit_taken[i] and current_return >= FIRST_TP_PERCENT:  # First take-profit hit
                # Take half profit
                half_profit_taken[i] = True
                # Set initial trailing stop
                trailing_stop_level[i] = price_t[i] * (1 - TRAILING_STOP_PERCENT) if currentPos[i] > 0 else price_t[i] * (1 + TRAILING_STOP_PERCENT)
                # Only trigger cool-down if exited via profit target
                if (currentPos[i] > 0 and price_t[i] >= second_tp_level[i]):
                    days_since_tp_long[i] = int(COOLDOWN_DAYS / 4)  # Start long cool-down
                elif (currentPos[i] < 0 and price_t[i] <= second_tp_level[i]):
                    days_since_tp_short[i] = int(COOLDOWN_DAYS / 4)  # Start short cool-down
            # NEW CONDITION: Exit if either second TP reached OR EMA crosses opposite direction
            elif half_profit_taken[i] and (
                (currentPos[i] > 0 and price_t[i] >= second_tp_level[i]) or  # Second profit target for long 
                (currentPos[i] < 0 and price_t[i] <= second_tp_level[i])   # Second profit target for short 
                # EMA crosses opposite direction
                # (currentPos[i] > 0 and (ema12_y[i] > ema15_y[i] and ema12_t[i] < ema15_t[i])) or  # Death cross (long exit)
                # (currentPos[i] < 0 and (ema12_y[i] < ema15_y[i] and ema12_t[i] > ema15_t[i]))    # Golden cross (short exit)
            ):
                # Exit remaining position
                position_dir[i] = 0
                # Only trigger cool-down if exited via profit target
                if (currentPos[i] > 0 and price_t[i] >= second_tp_level[i]):
                    days_since_tp_long[i] = 0  # Start long cool-down
                elif (currentPos[i] < 0 and price_t[i] <= second_tp_level[i]):
                    days_since_tp_short[i] = 0  # Start short cool-down
                reset_trade_state(i)
                
            elif (half_profit_taken[i] and 
                  ((currentPos[i] > 0 and price_t[i] <= trailing_stop_level[i]) or # Trailing stop hit for long
                   (currentPos[i] < 0 and price_t[i] >= trailing_stop_level[i]))):  # Trailing stop hit for short
                # Exit remaining position
                position_dir[i] = 0
                reset_trade_state(i)
                
            elif current_return <= -STOP_LOSS_PERCENT:  # Stop-loss hit
                # Exit entire position
                position_dir[i] = 0
                reset_trade_state(i)
                
            elif days_in_trade_neg[i] >= MAX_HOLD_DAYS:  # Timeout
                # Exit entire position
                position_dir[i] = 0
                reset_trade_state(i)

                # ======== ENTRY LOGIC ========
    # Update crossover signal buffer (shift previous signals)
    crossover_signals = np.roll(crossover_signals, shift=1, axis=1)
    crossover_signals[:, 0] = 0  # Reset today's signals

    # Detect today's crossover signals for EMA_STRATEGY_INSTS to determine entry
    for i in EMA_STRATEGY_INSTS:
        # EMA crossover long signal
        if ema12_y[i] < ema50_y[i] and ema12_t[i] > ema50_t[i]:
            crossover_signals[i, 0] = +1
        # EMA crossover short signal
        elif ema12_y[i] > ema50_y[i] and ema12_t[i] < ema50_t[i]:
            crossover_signals[i, 0] = -1

    for i in RSI_STRATEGY_INSTS:
        # EMA crossover long signal
        if ema30_y[i] < ema50_y[i] and ema30_t[i] > ema50_t[i] and rsi_t[i] < 30:
            crossover_signals[i, 0] = +1
        # EMA crossover short signal
        elif ema30_y[i] > ema50_y[i] and ema30_t[i] < ema50_t[i] and rsi_t[i] > 70:
            crossover_signals[i, 0] = -1

    for i in range(nins):
#--------------------- EMA 50 strategy for negative returns instruments -------------------------------------
        if i in EMA_STRATEGY_INSTS + RSI_STRATEGY_INSTS:
            if currentPos[i] == 0:

                # Use signal from 2 days ago (delayed entry)
                delayed_signal = crossover_signals[i, ENTRY_DELAY]
                
                # Long entry based on delayed signal
                if delayed_signal == +1 and days_since_tp_long[i] > COOLDOWN_DAYS and ema50_t[i] > ema200_t[i]:
                    position_dir[i] = +1
                    entry_prices_neg[i] = price_t[i]
                    best_price_neg[i] = price_t[i]
                    take_profit_level[i] = price_t[i] * (1 + FIRST_TP_PERCENT)
                    second_tp_level[i] = price_t[i] * (1 + FIRST_TP_PERCENT * SECOND_TP_MULTIPLIER)
                    stop_loss_level[i] = price_t[i] * (1 - STOP_LOSS_PERCENT)
                    days_in_trade_neg[i] = 0
                    half_profit_taken[i] = False
                    
                # Short entry based on delayed signal
                elif delayed_signal == -1 and days_since_tp_short[i] > COOLDOWN_DAYS:
                    position_dir[i] = -1
                    entry_prices_neg[i] = price_t[i]
                    best_price_neg[i] = price_t[i]
                    take_profit_level[i] = price_t[i] * (1 - FIRST_TP_PERCENT)
                    second_tp_level[i] = price_t[i] * (1 - FIRST_TP_PERCENT * SECOND_TP_MULTIPLIER)
                    stop_loss_level[i] = price_t[i] * (1 + STOP_LOSS_PERCENT)
                    days_in_trade_neg[i] = 0
                    half_profit_taken[i] = False
            
#----------------------------------------------------------------------------------------------------
        # 3) MACD crossover logic → position_dir / last_cross
        # For positive return instruments: Original MACD logic
        else:
        # long crossover?
            if (
                (macd_y[i] < sig_y[i]) 
                and (macd_t[i] > sig_t[i]) 
                and (macd_y[i] < 0) and (sig_y[i] < 0)
                and (macd_t[i] < 0) and (sig_t[i] < 0)
                # and rsi_t[i] < 50
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
                # and rsi_t[i] > 50
                and last_cross[i] != -1
                ):
                position_dir[i] = -1
                last_cross[i]   = -1


    # 4) Convert float to int for signal; Build the signal vector (same as position_dir, just more readable)
    signal_dir = position_dir.astype(int)

    # EMA_STRATEGY_INSTS = [0, 2, 4, 5, 7, 10, 13, 15, 18, 20, 21, 25, 
    #            27, 28, 30, 31, 33, 34, 35, 39, 40, 42, 
    #            43, 46, 47, 48]

    # EMA_STRATEGY_INSTSX = [13, 15, 18, 34, 35, 43, 48, 21, 28, 31, 40]
    # signal_dir[EMA_STRATEGY_INSTSX] = 0

    # … after building signal_dir …

    # 5) ONLY trade instruments whose signal just flipped
    if not np.array_equal(signal_dir, last_signal_dir):
        changed = np.where(signal_dir != last_signal_dir)[0]
        newPos  = currentPos.copy()

        for i in changed:
            if price_t[i] > 0:
                # For negative instruments with half profit taken, adjust position size
                if i in EMA_STRATEGY_INSTS and half_profit_taken[i]:
                    shares = int(round(HALF_POS_DOLLARS / price_t[i]))  # Half position size
                else:
                    shares = int(round(FULL_POS_DOLLARS / price_t[i]))
                newPos[i]   = signal_dir[i] * shares

        posLimits = (dlrPosLimit / price_t).astype(int)
        newPos    = np.clip(newPos, -posLimits, posLimits)
        currentPos      = newPos
        last_signal_dir = signal_dir.copy()

    return currentPos


