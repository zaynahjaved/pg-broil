import numpy as np
from spinup.envs.pointbot_const import *
import pickle
class ReacherRewardBrex():
    # Daniel's Suggested Reward
    def __init__(self):
        with open('brex_reacher.pickle', 'rb') as handle:
            b = pickle.load(handle)
        #print(b)

        self.posterior = []
        self.target_penalty = []
        self.obstacle_penalty = []
        self.weight_vectors = []
        for w, prob in b.items():
            self.posterior.append(prob)
            self.target_penalty.append(w[0])
            self.obstacle_penalty.append(w[1])
            self.weight_vectors.append(np.asarray(w))

        self.posterior = np.array(self.posterior)
        self.obstacle_penalty = np.array(self.obstacle_penalty)
        self.target_penalty = np.array(self.target_penalty)
        self.weight_vectors = np.array(self.weight_vectors)

    def get_posterior_weight_matrix(self):
        #get the matrix of hypothesis weight vectors from the posterior one per row
        return self.weight_vectors

    def get_reward_distribution(self, env):
        feats = env.get_features()
        #print(feats)
        dist_rew = feats[0]*self.target_penalty
        obs_rew = feats[1]*self.obstacle_penalty
        return dist_rew+obs_rew
