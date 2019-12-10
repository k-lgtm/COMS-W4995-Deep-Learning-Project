import scipy.optimize as sco

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def pos_constraint(x):
    # positive values
    pos_vals = [val for val in x if val >= 0]

    # sum(pos_vals) = 1
    return sum(pos_vals) - 1

def neg_constraint(x):
    # negative values
    neg_vals = [val for val in x if val < 0]
    
    # sum(neg_vals) = -1
    return sum(neg_vals) + 1

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    # constraints
    con1 = {'type': 'eq', 'fun': pos_constraint}
    con2 = {'type': 'eq', 'fun': neg_constraint}
    constraints = [con1, con2] 

    # bounds
    bound = (-1.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    # objective
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result
