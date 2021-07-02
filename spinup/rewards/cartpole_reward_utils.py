import numpy as np
class CartPoleReward():
    def __init__(self):
        self.posterior = np.array([0.95, 0.05])

    def get_reward_distribution(self, obs):

        x_pos = obs[0]
        if x_pos > 0.0:
            return np.array([+2.0, -2.0])
        else:
            return np.array([+1.0, +1.0])

if __name__ == "__main__":

    reward_dist = ToyReward()
    print(reward_dist.get_reward_distribution([0,0,0,0]))
    print(reward_dist.get_reward_distribution([1,0,0,0]))
    print(reward_dist.posterior)
