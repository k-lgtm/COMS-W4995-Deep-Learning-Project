import numpy as np
import pandas as pd
import gym
import random 

"""
Key Environment Features:
    - on/off transaction costs
    - discrete/continuous action space
    - train/test/all data mode
    - recurrent/non-recurrent policies
    - raw_states/processed_states (.use_trends=True/False)
        -- raw_states: direct values from .csv data files
        -- processed_states: trend values i.e. RSI, SMA, etc.
    - compatible w/ OpenAI Stable-Baselines and Ray RLlib
"""

DEFAULT_ENV_CONFIG = {
    'data_dir': '/content/gdrive/My Drive/4995_trading/data',
    'mode': 'train', #str: train, test, all
    'use_cost': False,
    'action_space_config':{
        'action_space': 'discrete', #str:'continuous','discrete'
        'bins': 50 # discrete space only
    }, 
    'recurrent_config': {
        'recurrent_policy': True,
        'lookback': 4, # non-recurrent policy only
    }, 
    'use_trends': False, # Not Implemented Yet
    'episode_length': 120, 
    'principal': 1000000, 
    'transaction_ratio': 0.0002, 
    'asset_num': 3, 
}

class EquityEnv(gym.Env):
    def __init__(self, config=DEFAULT_ENV_CONFIG):
        # Class Variables
        self.principal = config['principal']
        self.balance = config['principal']
        self.use_cost = config['use_cost'] 
        self.transaction_ratio = config['transaction_ratio']
        self.asset_num = config['asset_num']
        self.recurrent_policy = config['recurrent_config']['recurrent_policy']
        self.lookback = config['recurrent_config']['lookback']
        self.position = None
        self.pnl = 0
        
        # State Space: supports recurrent and non-recurrent policies
        if self.recurrent_policy:
            self.observation_space = gym.spaces.Box(low=-30, high=100, shape=(5*self.asset_num+self.asset_num,), dtype='float32')
        else:
            new_shape = (5*(self.lookback+1)+1, self.asset_num)
            self.observation_space = gym.spaces.Box(low=-30, high=100, shape=new_shape, dtype='float32')

        # Action Space: supports discrete and continuous spaces
        self.action_mode = config['action_space_config']['action_space']
        bins = config['action_space_config']['bins']
        self.action_space = self._set_action_space(self.action_mode, bins)
        
        if self.action_mode == 'discrete':
            self.action_mappings = self._set_action_mappings(bins)

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
        self._setup(config['data_dir'])
        
        # Training Params
        period = len(self.close_prices) #36000
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
        # set environment mode
        if self.mode == 'train':
            self.start_idx = max(0, random.choice(self.train_period) - self.episode_length)
            self.end_idx = self.start_idx + self.episode_length
        elif self.mode == 'test':
            self.start_idx = max(self.test_period[0], random.choice(self.test_period) - self.episode_length)
            self.end_idx = self.start_idx + self.episode_length
        elif self.mode == 'all':
            self.start_idx = 0
            self.end_idx = 35000
        else:
            raise Exception('reset(): Invalid Data Mode')
        
        if not self.recurrent_policy:
            self.start_idx = max(self.lookback, self.start_idx)

        # reset variables
        self.t = self.start_idx
        self.balance = self.principal
        self.position = np.array([0.0, 0.0, 0.0])
        self.pnl = 0

        # return initial state
        return self._get_state(self.t)
    
    def step(self, action):
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

    def _set_action_space(self, action_type, bins):
        if action_type == 'discrete':
            action_space = gym.spaces.MultiDiscrete([bins+1, bins+1, bins+1]) #-1, -0.9,...0,...0.9, 1
        elif action_type == 'continuous':
            action_space = gym.spaces.Box(low=-1, high=1, shape=(self.asset_num,), dtype='float32')
        else:
            raise Exception('_set_action_space(): Invalid Action Space')    
        return action_space
    
    def _set_action_mappings(self, bins):
        """
        return: Mapping of MultiDiscrete action space [0,20] to [-1,1] 
        """
        action_mappings = {}
        step_size = 2/bins
        valid_actions = np.arange(-1, 1.1, step=step_size)
        for i, action in enumerate(valid_actions):
            action_mappings[i] = action
        return action_mappings
    
    def _get_close_prices(self, t):
        """
        return: close prices for all stocks at time t
        """
        return np.array(self.close_prices.iloc[t].tolist())
    
    def _get_state(self, t):
        """
        if recurrent policy: 
            return [position, google, amazon, msft] at single timestep t
        else:
            return [goog_pos, amazon_pos, msft_pos]
                   [states at t-lookback]
                   [states at t-lookback + 1]
                   [states at t-lookback + 2]
                   ...
                   [states at t]
        """
        if self.recurrent_policy:
            return np.concatenate([self.position, self.states.iloc[t].tolist()])
        else:
            minutes = self.states.iloc[t-self.lookback:t+1]
            google = minutes[['open_gg', 'close_gg', 'high_gg', 'low_gg', 'volume_gg']].values.flatten()
            amazon = minutes[['open_am', 'close_am', 'high_am', 'low_am', 'volume_am']].values.flatten()
            msft = minutes[['open_ms', 'close_ms', 'high_ms', 'low_ms', 'volume_ms']].values.flatten()
            all_stocks = np.vstack([google,amazon,msft]).T
            return np.concatenate([[self.position], all_stocks])
    
    def _valid_action(self, action):
        """
        return: valid action if the action is invalid
        """
        if self.action_mode == 'discrete':
            action = np.array([self.action_mappings[a] for a in action])
        
        for a in action:
            if a < -1 or a > 1:
                raise Exception('_valid_action(): Invalid Action')
        return action
    
    def _get_reward(self, action):
        action = self._valid_action(action)

        # Positions 
        old_position = self.position
        new_position = action 
        
        # Close Prices
        previous_close = self._get_close_prices(self.t-1)
        current_close = self._get_close_prices(self.t)

        # Intermediate Reward Calculations
        capital_gain = np.dot(new_position, (current_close - previous_close) / previous_close) * self.balance
        transaction_cost = (np.absolute(new_position - old_position).sum() * self.transaction_ratio * self.balance) if self.use_cost else 0

        self.balance += capital_gain

        # Reward
        reward = capital_gain - transaction_cost
        self.pnl += reward
        self.position = new_position

        # Debugging Info
        info = {'date': self.dates.iloc[self.t],
                'transaction_cost': transaction_cost,
                'capital_gain': capital_gain,
                'previous_close': previous_close,
                'current_close': current_close,
                'action': action}

        return reward, info
    
    def _setup(self, data_dir):
        # Loads state and action data
        states_df = pd.read_csv(data_dir + "/state.csv")
        prices_df = pd.read_csv(data_dir + "/price.csv")
        
        # Creates dataframes
        self.states = states_df[['open_gg', 'close_gg', 'high_gg', 'low_gg', 'volume_gg',
                                 'open_am', 'close_am', 'high_am', 'low_am', 'volume_am',
                                 'open_ms', 'close_ms', 'high_ms', 'low_ms', 'volume_ms']]
        self.close_prices = prices_df[['close_gg', 'close_am', 'close_ms']]
        self.open_prices = prices_df[['open_gg', 'open_am', 'open_ms']]
        self.dates = states_df[['Dates']]
        
        if self.use_cost:
            print("USING TRANSACTION COST")
        
        print('-- Data Loaded --')
