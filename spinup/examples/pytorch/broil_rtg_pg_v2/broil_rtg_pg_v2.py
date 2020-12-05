import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

from spinup.examples.pytorch.broil_rtg_pg_v2.cvar_utils import cvar_enumerate_pg
from spinup.examples.pytorch.broil_rtg_pg_v2.cartpole_reward_utils import CartPoleReward

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(reward_dist, lamda, alpha=0.95, env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()


    #### compute BROIL policy gradient loss (robust version)
    def compute_broil_weights(batch_rets, batch_rewards_to_go):
        '''batch_returns: list of numpy arrays of size num_rollouts x num_reward_fns
           batch_rewards_to_go: list of rewards to go by reward function over all rollouts,
            size is num_rollouts*ave_rollout_length x num_reward_fns
        '''
        #inputs are lists of numpy arrays
        #need to compute BROIL weights for policy gradient and convert to pytorch

        #first find the expected on-policy return for current policy under each reward function in the posterior
        exp_batch_rets = np.mean(batch_rets, axis=0)
        print(exp_batch_rets)
        posterior_reward_weights = reward_dist.posterior


        #calculate sigma and find the conditional value at risk given the current policy
        sigma, cvar = cvar_enumerate_pg(exp_batch_rets, posterior_reward_weights, alpha)
        print("sigma = {}, cvar = {}".format(sigma, cvar))

        #compute BROIL policy gradient weights

        total_rollout_steps = len(batch_rewards_to_go)
        broil_weights = np.zeros(total_rollout_steps)
        for i,prob_r in enumerate(posterior_reward_weights):
            if sigma > exp_batch_rets[i]:
                w_r_i = lamda + (1 - lamda) / (1 - alpha)
            else:
                w_r_i = lamda
            broil_weights += prob_r * w_r_i * np.array(batch_rewards_to_go)[:,i]


        return broil_weights,cvar










    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_rewards_to_go = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()
                #print(obs[0])

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, posterior over reward
            batch_acts.append(act)
            ## old code from normal policy gradient:
            ## ep_rews.append(rew)
            #### New code for BROIL
            rew_dist = reward_dist.get_reward_distribution(obs)  #S create reward
            ep_rews.append(rew_dist)
            ####

            if done:
                # if episode is over, record info about episode
                ## Old code
                ## ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                #### New code
                ep_ret_dist, ep_len = np.sum(ep_rews, axis=0), len(ep_rews)
                ####

                batch_rets.append(ep_ret_dist)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                #### we are now computing this for every element in the reward function posterior but we can use the same function
                batch_rewards_to_go.extend(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        #### take a single BROIL policy gradient update step
        broil_weights, cvar = compute_broil_weights(batch_rets, batch_rewards_to_go)
        ####
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(broil_weights, dtype=torch.float32)
                                  )

        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens, cvar

    # training loop
    cvar_list = []
    exp_ret_list = []
    wc_ret_list = []
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens, cvar = train_one_epoch()
        exp_ret = np.dot(np.mean(batch_rets,axis=0),reward_dist.posterior)
        worst_case_return = np.min(np.mean(batch_rets, axis=0))
        cvar_list.append(cvar)
        exp_ret_list.append(exp_ret)
        wc_ret_list.append(worst_case_return)
        print('epoch: %3d \t loss: %.3f \t exp return: %.3f \t cvar: %.3f \t wc return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, exp_ret, cvar, worst_case_return, np.mean(batch_lens)))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(cvar_list)
    plt.title("conditional value at risk")
    plt.figure()
    plt.plot(exp_ret_list)
    plt.title("expected return")
    plt.figure()
    plt.plot(wc_ret_list)
    plt.title("worst case return")

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--alpha', default=0.95, type=float, help="alpha for alpha CVaR")
    parser.add_argument('--lamda', default = 0.0, type=float, help='blending between exp return (lamda=1) and cvar maximization (lamda=0)')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of BROIL policy gradient.\n')
    #print('\nUsing only two reward functions in posterior')
    #print("R1(s) = +1 (if s <= 0) +2 (if s > 0)")
    #print("R2(s) = +1 (if s <= 0) -10 (if s > 0)")
    #print("Pr(R1) = 0.95")
    #print("Pr(R2) = 0.05")
    #print("Expected reward R(s) = +1 (if s <= 0) +1.4 (if s > 0)")

    #create reward function distribution
    reward_dist = CartPoleReward()

    train(reward_dist, args.lamda, args.alpha, env_name=args.env_name, epochs=args.epochs, render=args.render, lr=args.lr)
