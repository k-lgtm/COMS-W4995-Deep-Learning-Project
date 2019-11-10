import ray

from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override, DeveloperAPI

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

if __name__ == '__main__':

ray.init()

ModelCatalog.register_custom_action_dist("my_dist", CustomActionDist)

