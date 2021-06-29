import numpy as np
from spinup.envs.pointbot_const import *
import pickle
class PointBotRewardBrex():
    def __init__(self):
        with open('brex_reward_trashbot.pkl', 'rb') as handle:
            b = pickle.load(handle)

        self.posterior = []
        self.obstacle_penalty = []
        self.non_obstacle_penalty = []
        self.trash_penalty = []
        self.weight_vectors = []
        for w, prob in b.items():
            self.posterior.append(prob)
            self.obstacle_penalty.append(w[0])
            self.non_obstacle_penalty.append(w[1])
            self.trash_penalty.append(w[2])
            self.weight_vectors.append(np.asarray(w))

        self.posterior = np.array(self.posterior)
        self.obstacle_penalty = np.array(self.obstacle_penalty)
        self.non_obstacle_penalty = np.array(self.non_obstacle_penalty)
        self.trash_penalty = np.array(self.trash_penalty)
        self.weight_vectors = np.array(self.weight_vectors)

    def get_posterior_weight_matrix(self):
        #get the matrix of hypothesis weight vectors from the posterior one per row
        return self.weight_vectors

    def get_reward_distribution(self, env, obs):
        initial_reward = env.rewards[-1]  # NOTE: Be careful of this. If we ever want to get reward distributions from observations, this will be an issue

        # This must be done because pointbot rewards takes into account penalty (check spinup/envs/pointbot.py).
        # We need to subtract out to reset it to what we want
        # Add because we originally subtract this value        
        if env.trash_taken:
            initial_reward += self.trash_penalty

        if env.obstacle(obs) == 0:
            return initial_reward + self.non_obstacle_penalty
        else:
            extra_cost = self.obstacle_penalty * env.obstacle(obs)
            return initial_reward + extra_cost
