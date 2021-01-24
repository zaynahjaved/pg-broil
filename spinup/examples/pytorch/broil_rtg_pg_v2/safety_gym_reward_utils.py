import numpy as np
class SafetyGymReward():
    def __init__(self):
        self.posterior = np.array([1])
        self.penalties = np.array([0]) # for debugging 0 penalty always

    def get_reward_distribution(self, env):
        rew = env.reward()
        if env.goal_met():
            rew += env.reward_goal
        cost = int(bool(env.cost()['cost']))

        if not cost:
            return np.array([rew] * self.posterior.shape[0])
        else:
            reward_penalties = self.penalties * cost
            return rew - reward_penalties
