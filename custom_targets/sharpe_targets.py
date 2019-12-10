import numpy as np
import pandas as np

def get_weights(sharpe_vals, i):
    # calc norm values
    pos_norm = 0
    neg_norm = 0
    for val in sharpe_vals:
        if val > 0:
            pos_norm += val
        else:
            neg_norm += val

    # normalize values
    pos_sum = 0
    neg_sum = 0
    for i in range(len(sharpe_vals)):
        if sharpe_vals[i] > 0:
            sharpe_vals[i] = sharpe_vals[i]/pos_norm
            pos_sum += sharpe_vals[i]
        else:
            sharpe_vals[i] = sharpe_vals[i]/(-neg_norm)
            neg_sum += sharpe_vals[i]

    # scale to 1 given bias
    scale_factor = 1-BIAS
    weights = sharpe_vals*scale_factor

    # error checking
    assert weights.idxmax() == sharpe_vals.idxmax()
    assert weights.idxmin() == sharpe_vals.idxmin()
    if np.isnan(weights).any():
        print(sharpe_vals, i)
        raise Exception

    return weights

def validate_values(targets):
    for i in range(len(targets)):
        if np.isinf(targets[i]) or np.isnan(targets[i]) or np.isneginf(targets[i]):
            targets[i] = 0
    assert targets.shape == (N_STOCKS,)
    return targets

def create_sharpe_targets(rets, window, pred_length=152):
    """
    returns target weights: [50, 51, ..., 739]
    """
    targets = []

    for i in range(window, rets.shape[0]-pred_length):
        # expected returns/std
        ev = rets.iloc[i:i+pred_length, :-1].mean()
        std = rets.iloc[i:i+pred_length, :-1].std()

        # sharpe vals
        sharpe_vals = validate_values(ev / std)

        # get weights
        stock_weights = get_weights(sharpe_vals, i)

        # add to targets
        targets.append(np.hstack([stock_weights, [BIAS]]))  
    return np.array(targets)
