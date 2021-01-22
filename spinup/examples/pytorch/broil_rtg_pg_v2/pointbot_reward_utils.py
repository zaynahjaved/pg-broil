import numpy as np
from spinup.envs.pointbot_const import *
class PointBotReward():
    # Below is Jerry's implemented reward. Solid explicit reward.
    """def __init__(self):
        self.posterior = np.array([0.8, 0.15, 0.05])

    def get_reward_distribution(self, env, obs):
        initial_reward = env.rewards[-1]
        if env.obstacle(obs) == 0:
            return np.array([initial_reward, initial_reward, initial_reward])
        else:
            extra_cost = COLLISION_COST * env.obstacle(obs)
            return np.array([initial_reward, initial_reward + 0.15*extra_cost, initial_reward+ 0.2*extra_cost])
    """

    # Daniel's Suggested Reward
    def __init__(self):
        self.posterior = np.array([0.4, 0.3, 0.2, .05, .05])
        # self.penalties = np.array([20, 10, 0, -10, -100])
        self.penalties = np.array([50, 40, 0, -40, -500])
        # Note: expected value of penalty is .55. So, it actually skews us to go towards obstacles
        # The idea is to train BROIL in such a way that it avoids things in the worst case...meaning it'll learn to avoid obstacles
        # despite this. Normal PPO (w/o BROIL) hopefully can't accomplish this as well.

    def get_reward_distribution(self, env, obs):
        initial_reward = env.rewards[-1]  # NOTE: Be careful of this. If we ever want to get reward distributions from observations, this will be an issue

        if env.obstacle(obs) == 0:
            return np.array([initial_reward] * self.posterior.shape[0])
        else:
            extra_cost = self.penalties * env.obstacle(obs)
            return initial_reward + extra_cost
