import numpy as np
import gym

class CustomActions(gym.spaces.Discrete):
    def __init__(self):
        self.low = -1.0
        self.high = 1.0
        self.step = 0.1
        self.size = 3
        self.last_actions = None

    def sample(self):
        """
        Action Bounds:
            -1 <= sum(actions) <= 1
            -1 <= action <= 1

        Num of valid actions: 3717

        :return: [google_action, amazon_action, msft_action]
        """
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

