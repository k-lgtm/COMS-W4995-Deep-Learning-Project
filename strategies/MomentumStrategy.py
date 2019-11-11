import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
import datetime as dt
import time

DATA_DIR = '../data'

class MomentumStrategy:
    def __init__(self, principal=1000000):
        # Setup dataframes
        self.prices = None
        self.dates = None
        self._load_data()

        # Class variables
        self.principal = principal
        self.normalize = normalize
    
    def momentum_signal(self, look_back, normalize=True):
        """
        return: momentum signal
        """
        price_data = prices.iloc[:, [2,4,6]].values
        delay_price = np.roll(price_data, look_back, axis=0)
        delay_price[:look_back] = np.nan
        mom_sig = (price_data - delay_price) / delay_price
        if normalize:
            mom_sig = mom_sig - mom_sig.mean(axis=1,keepdims=True)
            mom_sig = mom_sig / ((mom_sig > 0) * mom_sig).sum(axis=1, keepdims=True)
        return mom_sig 

    def _load_data(self):
        """
        Loads price data
        """
        self.prices = pd.read_csv(DATA_DIR, parse_dates=[0])
        self.dates = self.prices.iloc[:,0].apply(lambda x: pd.to_datetime(x))

class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(x)
        if ind >= len(self.dates) or ind < 0:
            return ''
        else:
            return self.dates[ind].strftime(self.fmt)
        
def annual_sharpe(pnl):
    mean = pnl.mean()
    var = pnl.std()
    day_sharpe = (mean / var) * np.sqrt(390)
    year_sharpe = day_sharpe * np.sqrt(252)
    return year_sharpe

def annual_return(pnl, principal=1000000):    
    ret = pnl / principal
    return np.mean(ret) * 390 * 252

def annual_volatility(pnl, principal=1000000):
    log_ret = np.log(1 + pnl / principal)
    return log_ret.std() * np.sqrt(252)

def maximum_drawdown(pnl):
    cum_pnl = np.cumsum(pnl)
    ind = np.argmax(np.maximum.accumulate(cum_pnl) - cum_pnl)
    return (np.maximum.accumulate(cum_pnl)[ind] - cum_pnl[ind]) / np.maximum.accumulate(cum_pnl)[ind]

def annual_turnover(weights):
    turnover = np.sum(np.abs(weights[1:] - weights[:-1])) / weights.shape[0]
    return turnover * 390 * 252

if __name__ == '__main__':
    # Testing class
    strat = MomentumStrategy()

    ret1 = strat.momentum_signal(lookback=1, normalize=False)

    mom = momentum_signal(prices, 2100)
    pnl_mom = np.sum(mom * np.roll(ret1, -1, axis=0), axis=1) * principal
    pnl_mom = np.nan_to_num(pnl_mom)

    plt.style.use("ggplot")
    formatter = MyFormatter(dates)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(np.arange(pnl_mom.shape[0]), np.cumsum(pnl_mom))
    fig.autofmt_xdate()
    plt.show()

    print(annual_sharpe(pnl_mom))
