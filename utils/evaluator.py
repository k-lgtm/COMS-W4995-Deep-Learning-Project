import numpy as np

"""
Evaluates models based on the following metrics:
    - annual sharpe, return, volatility
    - maximum drawdown
"""
class Evaluator:
    def __init__(self):
        pass
    
    def annual_sharpe(self, pnl):
        mean = pnl.mean()
        var = pnl.std()
        day_sharpe = (mean / var) * np.sqrt(390)
        year_sharpe = day_sharpe * np.sqrt(252)
        return year_sharpe

    def annual_return(self, pnl, principal=1000000):    
        ret = pnl / principal
        return np.mean(ret) * 390 * 252

    def annual_volatility(self, pnl, principal=1000000):
        log_ret = np.log(1 + pnl / principal)
        return log_ret.std() * np.sqrt(252)    
    
    def annual_turnover(self, weights):
        turnover = np.sum(np.abs(weights[1:] - weights[:-1])) / weights.shape[0]
        return turnover * 390 * 252
    
    def maximum_drawdown(self, pnl):
        cum_pnl = np.cumsum(pnl)
        ind = np.argmax(np.maximum.accumulate(cum_pnl) - cum_pnl)
        return (np.maximum.accumulate(cum_pnl)[ind] - cum_pnl[ind]) / np.maximum.accumulate(cum_pnl)[ind]


    def print_statistics(self, pnl, actions):
        print("Annual Sharpe: {}".format(self.annual_sharpe(pnl)))
        print("Annual Return: {}".format(self.annual_return(pnl)))
        print("Annual Volatility: {}".format(self.annual_volatility(pnl)))
        print("Annual Turnover: {}".format(self.annual_turnover(actions)))
        print("Maximum Drawdown: {}".format(self.maximum_drawdown(pnl)))   
        
    def evaluate_short(self, model, environment, num_steps=40000):
        pnl = []
        obs = environment.reset()
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=True)
            # here, action, rewards and dones are arrays, because we are using vectorized env
            obs, rewards, dones, info = environment.step(action)
            pnl.append(rewards[0])
            if dones[0]:
                break
        return sum(pnl)
        
    def evaluate(self, model, environment, num_steps=40000):
        pnl = []
        dates = []
        trans_cost = []
        action_ls = []
        obs = environment.reset()
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=True)
            # here, action, rewards and dones are arrays, because we are using vectorized env
            obs, rewards, dones, info = environment.step(action)
            date = info[0]["date"].to_pydatetime()
            cost = info[0]["transaction_cost"]
            # Stats
            pnl.append(rewards[0])
            dates.append(date)
            trans_cost.append(cost)
            action_ls.append(action)
            if dones[0]:
                break

        pnl = np.array(pnl)
        dates = np.array(dates)
        actions = np.array(action_ls)

        return pnl, dates, trans_cost, actions