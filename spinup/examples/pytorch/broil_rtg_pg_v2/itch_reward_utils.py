import numpy as np
class ItchReward():
    def __init__(self):
    	# corresponds to probabilities of wanting to itch on [shoulder, elbow]
        self.posterior = np.array([0.7, 0.3])

    def get_reward_distribution(self, env, a):
        # return reward for each of these
        return np.array([env.get_reward(a, limb='shoulder'), env.get_reward(a, limb='elbow')])
