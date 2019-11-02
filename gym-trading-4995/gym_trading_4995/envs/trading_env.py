import gym
from gym import error, spaces, utils
from gym.utils import seeding

class TradingEnv(gym.Env):
    def __init__(self):
        pass

    def step(self, action):
        raise NotImplementedError
        
        
        
        
import numpy as np
import gym
import copy
from gym import error, spaces, utils

from gym_trading_4995.envs.equity_trade import TrainAgent
from gym_trading_4995.envs.equity_trade import TestAgent


class TrainAgentEnv(gym.Env):
    def __init__(self, feature_num=5, asset_num=3, test=0, transaction_ratio=0.0002):
        self.task = TrainAgent(asset_count=asset_num, transaction_ratio=transaction_ratio)
        self.task.reset(test=test)
        self.test = test
        # Scale the data to certain magnitude
        self.observation_space = spaces.Box(low=0, high=3, shape=(feature_num * asset_num + asset_num, ),
                                            dtype='float32')
        self.action_space = spaces.Box(-2, 2, shape=(asset_num, ), dtype='float32')

    def reset(self):
        return self.task.reset(self.test)

    def step(self, action):
        return self.task.step(action)


class TestAgentEnv(gym.Env):
    def __init__(self, feature_num=5, asset_num=3, in_sample=0, transaction_ratio=0.0002):
        self.task = TestAgent(asset_count=asset_num, in_sample=in_sample, transaction_ratio=transaction_ratio)
        self.task.reset()
        # Scale the data to certain magnitude
        self.observation_space = spaces.Box(low=0, high=3, shape=(feature_num * asset_num + asset_num, ),
                                            dtype='float32')
        self.action_space = spaces.Box(-2, 2, shape=(asset_num, ), dtype='float32')

    def reset(self):
        return self.task.reset()

    def step(self, action):
        return self.task.step(action)