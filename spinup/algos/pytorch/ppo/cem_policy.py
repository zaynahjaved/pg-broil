"""
Code inspired by https://github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py
                 https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/MPC.py
                 https://github.com/kchua/handful-of-trials/blob/master/dmbrl/misc/optimizers/cem.py
"""

# from .policy import Policy

import pytorch_utils as ptu
# import latentsafesets.utils.spb_utils as spbu
from broil_utils import cvar_enumerate_pg, get_reward_distribution
# from latentsafesets.utils.pointbot_reward_utils import PointBotReward
# from latentsafesets.modules import VanillaVAE, PETSDynamics, ValueFunction, ConstraintEstimator, \
#     GoalIndicator

import torch
import numpy as np
import gym

import logging

log = logging.getLogger('cem')


class CEM:
    def __init__(self, env, env_name, broil_alpha, broil_lambda, reward_dist, params):
        log.info("setting up safe set and dynamics model")

        self.env = env
        self.env_name = env_name
        # self.logdir = params['logdir']
        self.reward_dist = reward_dist
        self.d_act = params['d_act']
        self.d_latent = params['d_latent']
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.plan_hor = params['plan_hor']
        self.random_percent = params['random_percent']
        self.popsize = params['num_candidates']
        self.num_elites = params['num_elites']
        self.max_iters = params['max_iters']

        self.goal_thresh = params['gi_thresh']
        self.broil_alpha = broil_alpha
        self.broil_lambda = broil_lambda


        self.mean = torch.zeros(self.d_act)
        self.std = torch.ones(self.d_act)
        self.ac_buf = np.array([]).reshape(0, self.d_act)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

    @torch.no_grad()
    def act(self, obs):
        """
        Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation. Cannot be a batch

        Returns: An action (and possibly the predicted cost)
        """

        # encode observation:
        # obs = ptu.torchify(obs).reshape(1, *self.d_obs)

        itr = 0
        reset_count = 0
        while itr < self.max_iters:
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:
                    action_samples = self._sample_actions_random()
                else:
                    num_random = int(self.random_percent * self.popsize)
                    num_dist = self.popsize - num_random
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                iter_num_elites = self.num_elites
                # predictions = self.predict(action_samples) #get states associated with action_samples
                #rollout the action_samples using the actual simulator to get the predictions
                # num_candidates, planning_hor, d_latent = predictions.shape #remove num_model
                # predictions = predictions.squeeze()
                # Sort
                values = self.compute_broil_rewards(action_samples, self.broil_alpha, self.broil_lambda)
                values = values.squeeze()
                sortid = values.argsort()
                print(values[sortid])
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]

                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)
                action_samples = self._sample_actions_normal(self.mean, self.std)

            itr += 1
        # Return the best action
        action = actions_sorted[-1][0]
        return action.detach().cpu().numpy()

    def predict(self, action_samples):
        action_samples = action_samples.detach().cpu().numpy()
        num_cand, horizon, action_dim = action_samples.shape
        predicted_states = np.zeros((num_cand, horizon + 1, action_dim + 2))
        predictor = gym.make(self.env_name)
        for i in range(num_cand):
            o = predictor.reset()
            predicted_states[i, 0, :] = o
            for s in range(1, horizon+1):
                next_o, r, d, info = predictor.step(action_samples[i][s-1].squeeze())
                predicted_states[i, s, :] = next_o
        return predicted_states


        

    def reset(self):
        # It's important to call this after each episode
        self.mean, self.std = None, None

    def _sample_actions_random(self, n=None):
        if n is None:
            n = self.popsize
        rand = torch.rand((n, self.plan_hor, self.d_act))
        scaled = rand * (self.ac_ub - self.ac_lb)
        action_samples = scaled + self.ac_lb
        return action_samples.to(ptu.TORCH_DEVICE)

    def _sample_actions_normal(self, mean, std, n=None):
        if n is None:
            n = self.popsize

        smp = torch.empty(n, self.plan_hor, self.d_act).normal_(
            mean=0, std=1).to(ptu.TORCH_DEVICE)
        mean = mean.unsqueeze(0).repeat(n, 1, 1).to(ptu.TORCH_DEVICE)
        std = std.unsqueeze(0).repeat(n, 1, 1).to(ptu.TORCH_DEVICE)

        # Sample new actions
        action_samples = smp * std + mean
        action_samples = torch.clamp(
            action_samples,
            min=self.env.action_space.low[0],
            max=self.env.action_space.high[0])
        print(self.env.action_space.low[0], self.env.action_space.high[0])
        print(action_samples)
        raise Exception
        return action_samples

    def get_reward_hypotheses(self):
        #reward_hypotheses = [num_reward_hypotheses X latent_space_size]
        #Each reward hypothesis: R(s) --> scalar value
        # #reward_hypotheses = torch.normal(0, 1, size=(10, 32))
        # reward_hypotheses_1 = torch.ones((10, 1))
        # reward_hypotheses = torch.zeros((10, 32))
        # reward_hypotheses[:, 1] = reward_hypotheses_1
        
        posterior = self.reward_dist.posterior
        reward_hypotheses = self.reward_dist.penalties
        # posterior_reward_weights = torch.full((1, len_rew_hypo), 1/len_rew_hypo).squeeze()
        return reward_hypotheses, posterior

    def compute_broil_rewards(self, actions, broil_alpha, broil_lambda):
        #for each candidate in states calculate the reward for that candidate with each reward hypothesis then calculate cvar for each candidate
        gamma=0.99
        broil_risk_metric = "cvar"
        expert_fcounts = None
        # reward_distribution = PointBotReward()
        # reward_dist = get_reward_distribution(reward_distribution) #hardcoded reward distribution
        '''batch_returns: list of numpy arrays of size num_rollouts x num_reward_fns
           weights: list of weights, e.g. advantages, rewards to go, etc by reward function over all rollouts,
            size is num_rollouts*ave_rollout_length x num_reward_fns
        '''
        #inputs are lists of numpy arrays
        #need to compute BROIL weights for policy gradient and convert to pytorch

        #first find the expected on-policy return for current policy under each reward function in the posterior
        # exp_batch_rets = np.mean(batch_rets.numpy(), axis=0)
        # posterior_reward_weights = reward_dist.posterior

        reward_hypotheses, posterior_reward_weights = self.get_reward_hypotheses() #matrix where each row is a reward hypothesis with len()=horizon
        W = reward_hypotheses #[num reward hypotheses X latent_space_size]
        len_rew_hypo = W.shape[0]
        action_samples = actions.detach().cpu().numpy()
        num_cand, horizon, action_dim = action_samples.shape
        batch_rewards = np.zeros((num_cand, len_rew_hypo)) #(num_cand, num_reward_hypotheses), omit horizon and ac_dim

        predicted_states = np.zeros((num_cand, horizon + 1, action_dim + 2))
        predictor = gym.make(self.env_name)


        #compute over the horizon
        # for i in range(num_cand):
        #     curr_cand_rew = np.zeros(len_rew_hypo)
        #     o = predictor.reset()
        #     predicted_states[i, 0, :] = o
        #     for j in range(horizon):
        #         next_o, r, d, info = predictor.step(action_samples[i][j].squeeze())
        #         predicted_states[i, j, :] = next_o
        #         rew_dist = self.reward_dist.get_reward_distribution(predictor, next_o)
        #         # rewards_curr_sample = torch.matmul(W, curr_sample_traj).squeeze() #dim = [num_reward_hypo X 1], rewards for each hypothesis for this traj
        #         curr_cand_rew += rew_dist
        #     batch_rewards[i, :] = curr_cand_rew
        # exp_batch_rets = np.mean(batch_rewards, axis=0)
        # cand_broil_values = np.zeros(num_cand)
        # for i in range(num_cand):
        #     rewards_curr_sample = batch_rewards[i]
        #     sigma, cvar = cvar_enumerate_pg(rewards_curr_sample, posterior_reward_weights, broil_alpha)
        #     # print("sigma = {}, cvar = {}".format(sigma, cvar))
        #     expected_curr_rew_sample = np.dot(rewards_curr_sample, posterior_reward_weights) #weighted average
        #     cand_broil_values[i] = broil_lambda * expected_curr_rew_sample + (1 - broil_lambda) * cvar
        # return cand_broil_values


        batch_rewards = np.zeros((num_cand, horizon))
        for i in range(num_cand):
            o = predictor.reset()
            predicted_states[i, 0, :] = o
            for j in range(horizon):
                next_o, r, d, info = predictor.step(action_samples[i][j].squeeze())
                predicted_states[i, j, :] = next_o
                batch_rewards[i, j] = r
                # rewards_curr_sample = torch.matmul(W, curr_sample_traj).squeeze() #dim = [num_reward_hypo X 1], rewards for each hypothesis for this traj
        return batch_rewards.sum(1)


