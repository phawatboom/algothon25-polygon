import optuna
import super_enhanced_strategy
from eval_test import prcAll, calcPL

def run_backtest(params):
    # Update all your global parameters in super_enhanced_strategy
    super_enhanced_strategy.FIRST_TP_PERCENT     = params['first_tp']
    super_enhanced_strategy.SECOND_TP_MULTIPLIER = params['second_tp_mult']
    super_enhanced_strategy.STOP_LOSS_PERCENT    = params['stop_loss']
    super_enhanced_strategy.TRAILING_STOP_PERCENT= params['trailing_stop']
    super_enhanced_strategy.COOLDOWN_DAYS        = params['cooldown_days']
    super_enhanced_strategy.MAX_HOLD_DAYS        = params['max_hold_days']
    super_enhanced_strategy.ENTRY_DELAY          = params['entry_delay']
    super_enhanced_strategy.TRAILING_UPDATE_FREQ = params['trailing_update_freq']

    # Run the backtest on the last 1000 days
    meanpl, _, plstd, _, _ = calcPL(prcAll, numTestDays=1000)
    score = meanpl - 0.1 * plstd
    return score


def objective(trial):
    # Define search space
    params = {
        'first_tp':          trial.suggest_float("first_tp", 0.08, 0.2),
        'second_tp_mult':    trial.suggest_float("second_tp_mult", 1.5, 3.5),
        'stop_loss':         trial.suggest_float("stop_loss", 0.02, 0.05),
        'trailing_stop':     trial.suggest_float("trailing_stop", 0.01, 0.03),
        'cooldown_days':     trial.suggest_int("cooldown_days", 40, 120),
        'max_hold_days':     trial.suggest_int("max_hold_days", 40, 120),
        'entry_delay':       trial.suggest_int("entry_delay", 0, 5),
        'trailing_update_freq': trial.suggest_int("trailing_update_freq", 1, 10),
    }
    return run_backtest(params)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    # print("Best Sharpe:", study.best_value)
    print("Best score:", study.best_value)
    print("Best hyperparameters:", study.best_params)
