import numpy as np
class ManipulatorReward():
    def __init__(self):
        self.posterior = np.array([1])
        self.penalties = np.array([0]) # for debugging 0 penalty always

    def get_reward_distribution(self, env):
        if not env.get_constraint():
            return np.array([env.get_reward()] * self.posterior.shape[0])
        else:
            reward_penalties = self.penalties * env.get_constraint()
            return env.get_reward() - reward_penalties
