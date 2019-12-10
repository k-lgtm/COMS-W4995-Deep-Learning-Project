from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def calc_targets(prices):
    targets = []
    for i in range(152, prices.shape[0]):
        # calc expected returns
        avg_returns = expected_returns.mean_historical_return(prices.iloc[i-152:i,:])

        # calc diagonal covariance matrix
        cov_mat = risk_models.sample_cov(prices.iloc[:150,:])
        diag_mat = cov_mat*np.identity(N_STOCKS)

        # find optimal weights
        ef = EfficientFrontier(avg_returns, cov_mat)
        weights = ef.min_volatility()

        # truncate and round values
        cleaned_weights = ef.clean_weights()
        targets.append(cleaned_weights)
    
    return targets
