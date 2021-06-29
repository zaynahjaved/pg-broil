import numpy as np
import pickle

class BoxingRewardBrex():
    def __init__(self):
        with open('/home/zaynahjaved/spinningup/spinup/algos/pytorch/rex/brex/brex_boxing_2.pkl', 'rb') as handle:
            b = pickle.load(handle)

        self.posterior = []
        self.damage_penalty = []
        self.hit_penalty = []
        self.avoid_penalty = []
        self.weight_vectors = []
        for w, prob in b.items():
            self.posterior.append(prob)
            self.damage_penalty.append(w[0])
            self.hit_penalty.append(w[1])
            self.avoid_penalty.append(w[2])
            self.weight_vectors.append(np.asarray(w))

        self.posterior = np.array(self.posterior)
        self.damage_penalty = np.array(self.damage_penalty)
        self.hit_penalty = np.array(self.hit_penalty)
        self.avoid_penalty = np.array(self.avoid_penalty)
        self.weight_vectors = np.array(self.weight_vectors)

    def get_posterior_weight_matrix(self):
        #get the matrix of hypothesis weight vectors from the posterior one per row
        return self.weight_vectors

    def get_reward_distribution(self, rew):
        if rew == 0:
            return self.avoid_penalty
        elif rew > 0:
            return self.hit_penalty
        else:
            return self.damage_penalty
