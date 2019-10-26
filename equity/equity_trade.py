import random
import numpy as np
import pandas as pd
import os.path
from pathlib import Path

data_folder = Path(os.path.dirname(__file__)) / "data"
states = pd.read_csv(data_folder / "state.csv", index_col=0, parse_dates=True)
prices = pd.read_csv(data_folder / "price.csv", index_col=0, parse_dates=True)
period = states.shape[0]  # around 36010
test_length = round(period / 500) * 100
episode_len = 120  # length of an episode
init_value = 1000000

train_period = np.arange(0, period - test_length)
# develop_period = np.arange(period - 2 * episode_len, period - episode_len)
test_period = np.arange(period - test_length, period)


# This environment is mainly for training purpose.
# In this environment, agent start at a random place, end with episode lenght
class TrainAgent(object):
    def __init__(self, asset_count, transaction_ratio):
        self.terminate = True
        self.date = None
        self.index = None
        self.count = None
        self.position = None
        self.asset_count = asset_count
        self.transaction_ratio = transaction_ratio

    def reset(self, test):
        """
        test=0: train mode
        test=1: develop mode
        test=2: test mode
        """
        self.terminate = False
        if test == 0:
            self.index = random.choice(train_period[: len(train_period) - episode_len + 1])
        elif test == 1:
            self.index = random.choice(develop_period[: len(develop_period) - episode_len + 1])
        else:
            self.index = random.choice(test_period[: len(test_period) - episode_len + 1])
        self.date = states.index[self.index]
        self.pnl = 0
        self.position = np.array([0.0] * self.asset_count)
        self.market = states.iloc[self.index].values
        self.close = np.array([prices.loc[self.date, 'close_'+ft] for ft in ['gg', 'am', 'ms']])
        self.count = 0
        observation = np.concatenate((self.market, self.position))
        return observation

    def step(self, action):
        self.index += 1
        self.date = states.index[self.index]
        old_position = self.position
        new_position = action - action.mean()

        for j in range(len(new_position)):
            if new_position[j] > 1:
                new_position[j] = 1
            elif new_position[j] < -1:
                new_position[j] = -1

        current_close = np.array([prices.loc[self.date, 'close_'+ft] for ft in ['gg', 'am', 'ms']])
        previous_close = self.close
        change = new_position - old_position
        # if np.absolute(change).mean() < 0.1:
        #     new_position = old_position
        #     change = np.zeros(new_position.shape)

        transaction_cost = np.absolute(change).sum() * self.transaction_ratio * init_value
        capital_gain = np.dot(new_position, (current_close - self.close) / self.close) * init_value
        reward = capital_gain - transaction_cost
        self.pnl += reward
        self.position = new_position
        self.market = states.iloc[self.index].values
        self.close = current_close
        self.count += 1
        observation = np.concatenate((self.market, self.position))
        if self.count >= episode_len:
            self.terminate = True
        else:
            self.terminate = False

        return observation, reward, self.terminate, {"date": self.date,
                                                     "transaction_cost": transaction_cost,
                                                     "capital_gain": capital_gain,
                                                     "current_close": current_close,
                                                     "previous_close": previous_close}



# This environment is mainly for the purpose of testing.
# In this environment, agent will go through the whole period
class TestAgent(object):
    def __init__(self, asset_count, in_sample, transaction_ratio):
        """
        in_sample = 0: In sample testing
        in_sample = 1: Out of sample testing
        """
        self.terminate = True
        self.date = None
        self.index = None
        self.count = None
        self.position = None
        self.period = None
        self.asset_count = asset_count
        self.transaction_ratio = transaction_ratio
        self.in_sample = in_sample

    def reset(self):
        self.terminate = False
        if self.in_sample == 0:
            self.period = train_period
        else:
            self.period = test_period
        self.index = self.period[0]
        self.date = states.index[self.index]
        self.pnl = 0
        self.position = np.array([0.0] * self.asset_count)
        self.market = states.iloc[self.index].values
        self.close = np.array([prices.loc[self.date, 'close_'+ft] for ft in ['gg', 'am', 'ms']])
        self.count = 0
        observation = np.concatenate((self.market, self.position))
        return observation

    def step(self, action):
        self.index += 1
        self.date = states.index[self.index]
        old_position = self.position
        new_position = action - action.mean()

        for j in range(len(new_position)):
            if new_position[j] > 1:
                new_position[j] = 1
            elif new_position[j] < -1:
                new_position[j] = -1

        current_close = np.array([prices.loc[self.date, 'close_'+ft] for ft in ['gg', 'am', 'ms']])
        previous_close = self.close
        change = new_position - old_position
        # if np.absolute(change).mean() < 0.1:
        #     new_position = old_position
        #     change = np.zeros(new_position.shape)

        transaction_cost = np.absolute(change).sum() * self.transaction_ratio * init_value
        capital_gain = np.dot(new_position, (current_close - self.close) / self.close) * init_value
        reward = capital_gain - transaction_cost
        self.pnl += reward
        self.position = new_position
        self.market = states.iloc[self.index].values
        self.close = current_close
        self.count += 1
        observation = np.concatenate((self.market, self.position))
        if self.count >= len(self.period) - 1:
            self.terminate = True
        else:
            self.terminate = False

        return observation, reward, self.terminate, {"date": self.date,
                                                     "transaction_cost": transaction_cost,
                                                     "capital_gain": capital_gain,
                                                     "current_close": current_close,
                                                     "previous_close": previous_close}