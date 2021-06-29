import numpy as np
class BoxingReward():
    def __init__(self):
        self.posterior = np.array([1])
        self.get_hit_penalties = np.array([-0.863]) # for debugging 0 penalty always
        self.hit_opponent_penalties = np.array([0.146])
        self.avoid_opponent_penalties = np.array([-0.484])

    def get_reward_distribution(self, reward):
        if reward < 0:
            return self.get_hit_penalties
        elif reward > 0:
            return self.hit_opponent_penalties
        else:
            return self.avoid_opponent_penalties
