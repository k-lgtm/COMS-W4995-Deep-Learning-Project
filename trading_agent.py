from collections import deque
import random
import numpy as np


class TradingAgent(object):
    def __init__(self, train_data, init_invest=20000):
        self.init_invest = init_invest
        self.stock_owned = None

    def step(self, action):
        raise NotImplementedError