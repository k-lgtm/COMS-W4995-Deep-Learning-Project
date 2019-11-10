import numpy as np
import datetime
import Formatter
from datetime import datetime
import matplotlib.pyplot as plt
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

    def annual_turnover(self, actions):
        turnover = np.sum(np.abs(actions[1:] - actions[:-1])) / actions.shape[0]
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

    def evaluate(self, model, environment, num_steps=30000):
        pnl = []
        dates = []
        trans_cost = []
        action_ls = []
        obs = environment.reset()
        
        for i in range(num_steps):
            if i % 5000 == 0:
                print('Iteration: {}'.format(i))
                
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=True)
            
            # here, action, rewards and dones are arrays, because we are using vectorized env
            obs, rewards, dones, info = environment.step(action)

            # date format: 2018-10-30 09:30:00
            date = datetime.strptime(info[0]["date"]["Dates"], '%Y-%m-%d %H:%M:%S')
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

        self.print_statistics(pnl, actions)

        plt.style.use("ggplot")
        formatter = MyFormatter(dates)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.xaxis.set_major_formatter(formatter)
        ax.plot(np.arange(pnl.shape[0]), np.cumsum(pnl))
        fig.autofmt_xdate()
        plt.show()