import numpy as np
from spinup.envs.pointbot_const import *
class PointBotReward():
    def __init__(self):
        self.posterior = np.array([0.8, 0.15, 0.05])

    def get_reward_distribution(self, env, obs):
        OBSTACLE_COST = 10
        initial_reward = env.rewards[-1]
        if env.obstacle(obs) == 0:
            return np.array([initial_reward, initial_reward, initial_reward])
        else:
            extra_cost = -OBSTACLE_COST * env.obstacle(obs)
            return np.array([initial_reward, initial_reward + 0.15*extra_cost, initial_reward+ 0.2*extra_cost])