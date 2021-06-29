import numpy as np
from spinup.envs.pointbot_const import *
class PointBotReward():
    def __init__(self):
        self.posterior = np.array([0.4, 0.3, 0.2, .05, .05])
        self.penalties = np.array([50, 40, 0, -40, -500])
        if TRASH:
            self.penalties = np.array([0, 0, 0, 0, 0])

    def get_reward_distribution(self, env, obs):
        initial_reward = env.rewards[-1]  # NOTE: Be careful of this. If we ever want to get reward distributions from observations, this will be an issue

        if env.obstacle(obs) == 0:
            return np.array([initial_reward] * self.posterior.shape[0])
        else:
            extra_cost = self.penalties * env.obstacle(obs)
            return initial_reward + extra_cost
