import numpy as np
from spinup.envs.pointbot_const import *
import pickle
class PointBotRewardBrex():
    # Daniel's Suggested Reward
    def __init__(self):
        with open('brex_reward_dist.pickle', 'rb') as handle:
            b = pickle.load(handle)
        #print(b)

        self.posterior = []
        self.obstacle_penalty = []
        self.non_obstacle_penalty = []
        self.trash_penalty = []
        for w, prob in b.items():
            self.posterior.append(prob)
            self.obstacle_penalty.append(w[0])
            self.non_obstacle_penalty.append(w[1])
            self.trash_penalty.append(w[2])

        self.posterior = np.array(self.posterior)
        self.obstacle_penalty = np.array(self.obstacle_penalty)
        self.non_obstacle_penalty = np.array(self.non_obstacle_penalty)
        self.trash_penalty = np.array(self.trash_penalty)

    def get_reward_distribution(self, env, obs):
        initial_reward = env.rewards[-1]  # NOTE: Be careful of this. If we ever want to get reward distributions from observations, this will be an issue

        # This must be done because pointbot rewards takes into account penalty (check spinup/envs/pointbot.py).
        # We need to subtract out to reset it to what we want
        # Add because we originally subtract this value        
        if env.trash_taken:
            initial_reward += self.trash_penalty

        if env.obstacle(obs) == 0:
            #return np.array([initial_reward] * self.posterior.shape[0])
            return initial_reward + self.non_collision_penalty
        else:
            extra_cost = self.collision_penalty * env.obstacle(obs)
            return initial_reward + extra_cost
