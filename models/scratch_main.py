import pandas as pd
import numpy as np
import gym
import random

import ray
import ray.rllib.agents.ppo as ppo

from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override, DeveloperAPI

import gym
import numpy as np
import pandas as pd
import random

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter


# All the different classes

class CustomActions(gym.spaces.Discrete):
    def __init__(self, low=-1.0, high=1.0, step=0.1, size=3):
        super(CustomActions, self).__init__(3)
        self.low = low
        self.high = high
        self.step = step
        self.size = size
        self.last_actions = None

    def sample(self): 
        actions = []
        low = self.low
        high = self.high
        for i in range(self.size):
            valid_range = np.arange(low, high, step=self.step)
            action = np.random.choice(valid_range)
            actions.append(action)
            if action > 0:
                high -= action
            elif action < 0:
                low -= action
        actions = np.around(np.array(actions), decimals=1)
        np.random.shuffle(actions)
        self.last_actions = actions
        return actions


DATA_DIR = '../data'

class EquityEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config):
        """ 
        .t: int idx of df
        .google/.amazon/.msft: [[Open, Close, High, Low, Volume]]
        .holdings: {google_pos: int, amazon_pos: int, msft_pos: int} 
        ._setup(): helper function that cleans data and initializes dfs
        .position: double = balance (double) + total of all pos. (double) 
        """  
        
        self.principal = config['principal']
        self.balance = config['principal']
        self.use_cost = config['use_cost']
        
        self.pnl = 0
        self.transaction_ratio = config['transaction_ratio']
        self.asset_num = config['asset_num']
        self.position = None
        
        # State/Action Spaces
        self.observation_space = gym.spaces.Box(low=-30, high=100, shape=(5*self.asset_num+self.asset_num,), dtype='float32')
        self.action_space = CustomActions()

        # Date idx
        self.t = None  
        self.start_idx = None
        self.end_idx = None
        
        # Dataframes
        self.states = None
        self.close_prices = None
        self.open_prices = None
        self.dates = None
        
        # Initializes dfs
        self._setup()
        
        # Training Params
        period = len(self.close_prices)-1000 #36000
        self.mode = config['mode']
        self.episode_length = config['episode_length']
        self.test_length = round(period/500)*100 #7200
        self.train_period = np.arange(0, period-self.test_length) #0 to 28800
        self.test_period = np.arange(period-self.test_length, period) # 28800 to 36000

        print('-- Environment Created --')
        
    def reset(self):
        """
        return: [position + google_t + amazon_t + msft_t]
        """
        if self.mode == 'train':
            self.start_idx = max(0, random.choice(self.train_period) - self.episode_length)
            self.end_idx = self.start_idx + self.episode_length
        elif self.mode == 'test':
            self.start_idx = max(self.test_period[0], random.choice(self.test_period) - self.episode_length)
            self.end_idx = self.start_idx + self.episode_length
        elif self.mode == 'all':
            self.start_idx = 0
            self.end_idx = 36000
        else:
            raise Exception('Invalid Mode')

        self.t = self.start_idx
        self.balance = self.principal
        self.position = np.array([0.0, 0.0, 0.0])
        self.pnl = 0
        return self._get_state(self.t)
        
    def step(self, action: list):
        """
        action: [new_google_pos, new_amazon, new_msft_pos]
        return: 
                <obs>   : [new_pos, google_t, amazon_t, msft_t]
                <reward>: double, capital_gain - transaction_cost
                <done>  : bool
                <info>  : {
                           'date': dateobj, 
                           'transaction_cost': double
                           'capital_gain': double ,
                           'previous_close': double,
                           'current_close': double
                          } 
        """
        self.t = self.t + 1

        # Done
        done = True if self.t >= self.end_idx else False
        
        # Reward 
        reward, info = self._get_reward(action)
        
        # Next State
        next_state = self._get_state(self.t)
        
        return next_state, reward, done, info
    
    def render(self, mode='human', close=False):
        return NotImplemented
    
    def _get_close_prices(self, t):
        return np.array(self.close_prices.iloc[t].tolist())
    
    def _get_state(self, t):
        return np.concatenate([self.position,
                               self.states.iloc[t].tolist()])
    
    def _get_reward(self, action):
        action = np.array(action)
        # Positions 
        old_position = self.position
        new_position = action 
        
        # Close Prices
        previous_close = self._get_close_prices(self.t-1)
        current_close = self._get_close_prices(self.t)

        # Intermediate Reward Calculations
        capital_gain = np.dot(new_position, (current_close - previous_close) / previous_close) * self.balance
        transaction_cost = (np.absolute(new_position - old_position).sum() * self.transaction_ratio * self.balance) if self.use_cost else 0

        # Reward 
        reward = capital_gain - transaction_cost
        self.pnl += reward
        self.position = new_position
        self.balance += capital_gain

        # Debugging Info
        info = {'date': self.dates.iloc[self.t],
                'transaction_cost': transaction_cost,
                'capital_gain': capital_gain,
                'previous_close': previous_close,
                'current_close': current_close,
                'action': action}

        return reward, info
    
    def _setup(self):
        states_df = pd.read_csv(DATA_DIR + "/state.csv")
        prices_df = pd.read_csv(DATA_DIR + "/price.csv")
        self.states = states_df[['open_gg', 'close_gg', 'high_gg', 'low_gg', 'volume_gg',
                       'open_am', 'close_am', 'high_am', 'low_am', 'volume_am',
                       'open_ms', 'close_ms', 'high_ms', 'low_ms', 'volume_ms']]
        self.close_prices = prices_df[['close_gg', 'close_am', 'close_ms']]
        self.open_prices = prices_df[['open_gg', 'open_am', 'open_ms']]
        print(len(self.close_prices))
        print(len(self.states))
        self.dates = states_df[['Dates']]
        print('-- Data Loaded --')

@DeveloperAPI
class CustomActionDist(ActionDistribution):
    @DeveloperAPI
    def __init__(self, inputs, model):
        super(CustomActionDist, self).__init__(inputs, model)
        assert model.num_outputs == 3
        self.low = -1.0
        self.high = 1.0
        self.step = 0.1
        self.size = 3
        self.last_actions = None

    @DeveloperAPI
    def sample(self): 
        actions = []
        low = self.low
        high = self.high
        for i in range(self.size):
            valid_range = np.arange(low, high, step=self.step)
            action = np.random.choice(valid_range)
            actions.append(action)
            if action > 0:
                high -= action
            elif action < 0:
                low -= action
        actions = np.around(np.array(actions), decimals=1)
        np.random.shuffle(actions)
        self.last_actions = actions
        return actions

    @DeveloperAPI
    def logp(self, actions): 
        return (1/3717)
    
    @DeveloperAPI
    def sampled_action_logp(self):
        return self.logp(self.last_actions)
    
    @DeveloperAPI
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 3

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


if __name__ == '__main__':

ray.init()

ModelCatalog.register_custom_action_dist("my_dist", CustomActionDist)

trainer = ppo.PPOTrainer(
	env=EquityEnv, 
	config = {
		"gamma": 0.98,
		"lr": 0.001,
		"ignore_worker_failures": False,
		"eager": True,
		"num_workers": 1,
	    "env_config": {'principal': 10,
	              'use_cost': False,
	              'transaction_ratio': 0.0002,
	              'asset_num': 3,
	              'mode': 'train',
	              'episode_length': 120},
	    },
	    "model": {
	        "custom_action_dist": "my_dist",
	    },)

while True:
	print(trainer.train())