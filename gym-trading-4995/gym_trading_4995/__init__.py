from gym.envs.registration import register

register(
    id='Trading-v0',
    entry_point='gym_trading_4995.envs:TradingEnv',
)


# Equity
# ----------------------------------------
register(
    id = 'Equity-train-v0',
    entry_point='gym_trading_4995.envs:TrainAgentEnv',
    max_episode_steps=120,
    reward_threshold=100000000,
    kwargs={'test': 0, 'transaction_ratio': 0},
)

register(
    id = 'Equity-develop-v0',
    entry_point='gym_trading_4995.envs:TrainAgentEnv',
    max_episode_steps=120,
    reward_threshold=100000000,
    kwargs={'test': 1},
)

register(
    id = 'Equity-test-v0',
    entry_point='gym_trading_4995.envs:TrainAgentEnv',
    max_episode_steps=1200,
    reward_threshold=100000000,
    kwargs={'test': 2},
)

register(
    id = 'Equity-insample-0bps-v0',
    entry_point='gym_trading_4995.envs:TestAgentEnv',
    max_episode_steps=36000,
    reward_threshold=100000000,
    kwargs={'in_sample': 0, 'transaction_ratio': 0},
)

register(
    id = 'Equity-insample-2bps-v0',
    entry_point='gym_trading_4995.envs:TestAgentEnv',
    max_episode_steps=36000,
    reward_threshold=100000000,
    kwargs={'in_sample': 0, 'transaction_ratio': 0.0002},
)

register(
    id = 'Equity-outsample-0bps-v0',
    entry_point='gym_trading_4995.envs:TestAgentEnv',
    max_episode_steps=36000,
    reward_threshold=100000000,
    kwargs={'in_sample': 1, 'transaction_ratio': 0},
)

register(
    id = 'Equity-outsample-2bps-v0',
    entry_point='gym_trading_4995.envs:TestAgentEnv',
    max_episode_steps=36000,
    reward_threshold=100000000,
    kwargs={'in_sample': 1, 'transaction_ratio': 0.0002},
)