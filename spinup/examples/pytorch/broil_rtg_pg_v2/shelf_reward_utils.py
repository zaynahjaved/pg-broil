import numpy as np
from spinup.envs.shelf_env import *
class ShelfReward():
    def __init__(self):
        self.posterior = np.array([1])
        self.penalties = np.array([0]) # for debugging 0 penalty always

    def get_reward_distribution(self, env):
        if not env.topple_check():
            return np.array([env.reward_fn()] * self.posterior.shape[0])
        else:
            reward_penalties = self.penalties * env.topple_check()
            return env.reward_fn() - reward_penalties
