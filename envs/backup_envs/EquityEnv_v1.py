import pandas as pd
import numpy as np
import gym
import random

DATA_DIR = 'data/'

class EquityEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, 
                 principal=1000000, 
                 use_cost=False,
                 split_data=False,
                 asset_num=3, 
                 transaction_ratio=0.0002,
                 episode_length=120):
        """ 
        .t: int idx of df
        .google/.amazon/.msft: [[Open, Close, High, Low, Volume]]
        .holdings: {google_pos: int, amazon_pos: int, msft_pos: int} 
        ._setup(): helper function that cleans data and initializes dfs
        .position: double = balance (double) + total of all pos. (double) 
        """  
        
        self.principal = principal
        self.balance = principal
        self.use_cost = use_cost
        self.position = None
        self.pnl = 0
        self.transaction_ratio = transaction_ratio
        self.asset_num = asset_num
        
        # State/Action Spaces
        self.observation_space = gym.spaces.Box(low=0, high=3, shape=(5*asset_num+asset_num,), dtype='float32')
        self.action_space = gym.spaces.Box(-2, 2, shape=(asset_num,), dtype='float32')

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
        self.split_data = split_data
        period = len(self.close_prices)
        self.test_length = round(period/500)*100 #7200
        self.train_period = np.arange(0, period-self.test_length) #0 to 28800
        self.test_period = np.arange(period-self.test_length, period) # 28800 to 36000

        print('-- Environment Created --')
        
    def reset(self):
        """
        return: [position + google_t + amazon_t + msft_t]
        """
        if self.split_data:
            self.start_idx = max(0, random.choice(TRAIN_PERIOD) - EPISODE_LENGTH)
            self.end_idx = self.start_idx + EPISODE_LENGTH
        else:
            self.start_idx = 0
            self.end_idx = 36000
        self.t = self.start_idx
        self.balance = self.principal
        self.position = np.array([0.0] * self.asset_num)
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
        
        # Positions (dollar neutral): long/short pos same
        old_position = self.position
        new_position = action - action.mean()
        
        # Clipping weights
        for i in range(self.asset_num):
            if new_position[i] > 1:
                new_position[i] = 1
            elif new_position[i] < -1:
                new_position[i] = -1
        
        # Close Prices
        previous_close = self._get_close_prices(self.t-1)
        current_close = self._get_close_prices(self.t)
        
        # Intermediate Reward Calculations
        capital_gain = np.dot(new_position, (current_close - previous_close) / previous_close) * self.principal
        transaction_cost = (np.absolute(new_position - old_position).sum() * self.transaction_ratio * self.principal) if self.use_cost else 0
        
        # Reward
        reward = capital_gain - transaction_cost
        self.pnl += reward
        self.position = new_position
        
        # Debugging Info
        info = {'date': self.dates.iloc[self.t],
                'transaction_cost': transaction_cost,
                'capital_gain': capital_gain,
                'previous_close': previous_close,
                'current_close': current_close}
        
        return reward, info
    
    def _setup(self):
        states_df = pd.read_csv(DATA_DIR + "/state.csv")
        prices_df = pd.read_csv(DATA_DIR + "/price.csv")
        self.states = states_df[['open_gg', 'close_gg', 'high_gg', 'low_gg', 'volume_gg',
                       'open_am', 'close_am', 'high_am', 'low_am', 'volume_am',
                       'open_ms', 'close_ms', 'high_ms', 'low_ms', 'volume_ms']]
        self.close_prices = prices_df[['close_gg', 'close_am', 'close_ms']]
        self.open_prices = prices_df[['open_gg', 'open_am', 'open_ms']]
        self.dates = states_df[['Dates']]
        print('-- Data Loaded --')