import argparse
import gym
import os
import sys
import pickle
import time
import spinup
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from spinup.envs.pointbot_const import *
from spinup.rewards.cvar_utils import cvar_enumerate_pg
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="Shelf-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G', default=None,
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.5, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-2, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=1000, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=256, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=200, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--rollout', type=int, default=100)
parser.add_argument('--folder', type=int, default=1)
parser.add_argument('--num_demos', type=int, default=10)

# RAIL
parser.add_argument('--cvar', action="store_true")
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--lamda', type=float, default=0.9)

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
else:
    policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
value_net = Value(state_dim)
discrim_net = Discriminator(state_dim + action_dim)
discrim_criterion = nn.BCELoss()
to_device(device, policy_net, value_net, discrim_net, discrim_criterion)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 10
optim_batch_size = 64

# load trajectory

if args.expert_traj_path is not None:
    expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
    running_state.fix = True
else:
    demo_obs, demo_acs = env.get_demos(args.num_demos)
    expert_traj = []
    for i in range(len(demo_acs)):
        obs_batch = np.array(demo_obs[i][:-1])
        acs_batch = np.array(demo_acs[i])
        for j in range(len(obs_batch)):
            expert_traj.append(np.hstack([obs_batch[j], acs_batch[j]]))

expert_traj = np.array(expert_traj)
print("EXPERT TRAJ: ", expert_traj.shape)

def expert_reward(state, action):
    state_action = tensor(np.hstack([state, action]), dtype=dtype)
    with torch.no_grad():
        return -math.log(discrim_net(state_action)[0].item())


"""create agent"""
agent = Agent(env, policy_net, device, custom_reward=expert_reward, num_threads=args.num_threads)


def update_params(batch, trajs, i_iter):
    if args.cvar:
        states = []
        actions = []
        rewards = []
        masks = []
        advantages = []
        returns = []
        discrim_returns = []
        for traj in trajs:
            traj_states = torch.from_numpy(np.array([t[0] for t in traj])).to(device)
            traj_actions = torch.from_numpy(np.array([t[1] for t in traj])).to(device)
            traj_masks = torch.from_numpy(np.array([t[2] for t in traj])).to(device)
            traj_rewards = torch.from_numpy(np.array([t[-1] for t in traj])).to(device)

            states.append(traj_states)
            actions.append(traj_actions)
            masks.append(traj_masks)
            rewards.append(traj_rewards)

            traj_disc_ret = -torch.squeeze(  torch.log(discrim_net(torch.cat([traj_states, traj_actions], 1)))  )
            gamma_list = torch.from_numpy(np.array([args.gamma**t for t in range(len(traj_states))])).to(device)
            discounted_traj_disc_ret = torch.dot(traj_disc_ret, gamma_list)
            discrim_returns.append(discounted_traj_disc_ret)

        discrim_returns = torch.stack(discrim_returns)
        probs = (1/len(discrim_returns))*np.ones(len(discrim_returns))
        sigma_star, _ = cvar_enumerate_pg(discrim_returns, probs, args.alpha)
        cvar_loss = torch.sum(torch.where(discrim_returns <= sigma_star, discrim_returns, torch.zeros(len(discrim_returns)).to(device)))
        cvar_loss *= (1/len(discrim_returns))*(1/(1-args.alpha))*args.lamda

        """get advantage estimation from the trajectories"""
        for i in range(len(rewards)):
            with torch.no_grad():
                vals_i = value_net(states[i])
            advantages_traj, returns_traj = estimate_advantages(rewards[i], masks[i], vals_i, args.gamma, args.tau, device)
            if sigma_star-discrim_returns[i] > 0:
                advantages_traj += ((-args.lamda)/(1-args.alpha))*(sigma_star-discrim_returns[i])
            advantages.append(advantages_traj)
            returns.append(returns_traj)

        states = torch.stack(states).view(-1, states[0].shape[-1])
        actions = torch.stack(actions).view(-1, actions[0].shape[-1])
        rewards = torch.stack(rewards).view(-1, rewards[0].shape[-1])
        masks = torch.stack(masks).view(-1, masks[0].shape[-1])
        advantages = torch.stack(advantages).view(-1, advantages[0].shape[-1])
        returns = torch.stack(returns).view(-1, returns[0].shape[-1])

        with torch.no_grad():
            fixed_log_probs = policy_net.get_log_prob(states, actions) # TODO: add this in again
    else:
        states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)

        with torch.no_grad():
            values = value_net(states)
            fixed_log_probs = policy_net.get_log_prob(states, actions) # TODO: add this in again

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """update discriminator"""
    for _ in range(1):
        expert_state_actions = torch.from_numpy(expert_traj).to(dtype).to(device)
        g_o = discrim_net(torch.cat([states, actions], 1))
        e_o = discrim_net(expert_state_actions)
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
            discrim_criterion(e_o, zeros((expert_traj.shape[0], 1), device=device))

        if args.cvar:
            discrim_loss += cvar_loss

        discrim_loss.backward()
        optimizer_discrim.step()

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


def main_loop():
    rewards = []
    count = 0
    for i_iter in range(args.max_iter_num):
        print("Iteration: ", i_iter)
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log, count, trajs = agent.collect_samples(args.min_batch_size, render=args.render, count=count)
        discrim_net.to(device)
        t0 = time.time()
        update_params(batch, trajs, i_iter)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration)"""
        discrim_net.to(torch.device('cpu'))
        _, log_eval, _, _ = agent.collect_samples(args.eval_batch_size, mean_action=True)
        rewards.append(log_eval["avg_reward"])
        discrim_net.to(device)
        t2 = time.time()
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_discrim_R_avg {:.2f}\ttrain_R_avg {:.2f}\teval_discrim_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_c_reward'], log['avg_reward'], log_eval['avg_c_reward'], log_eval['avg_reward']))
            if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
                to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
                pickle.dump((policy_net, value_net, discrim_net), open(os.path.join(assets_dir(), 'learned_models/{}_gail.p'.format(args.env_name)), 'wb'))
                to_device(device, policy_net, value_net, discrim_net)
        if args.env_name == 'PointBot-v0': 
            visualize_policy(policy_net, env, 1, args.folder, i_iter, True)
        """clean up gpu memory"""
        torch.cuda.empty_cache()
        print(count)
    print(rewards)
    plt.figure()
    plt.title('Rewards over iterations')
    plt.plot(range(0, args.max_iter_num), rewards, label='Training Accuracy')
    plt.savefig('GAIL_viz_' + str(args.folder)+ '/rewards.png')


def visualize_policy(policy, env, num_rollouts, folder, training_i=0, training=False):
    obs_times = []
    num_trashes = []
    count = 0
    for i in range(num_rollouts):
        print("Rollout: ", i)
        o = env.reset()
        #im_list = [env.render().squeeze()]
        done = False
        j = 0
        while not done:
            j += 1
            state_var = tensor(o).unsqueeze(0)
            with torch.no_grad():
                #action = policy(state_var)[0][0].numpy() #mean_action
                action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            o, _, done, _ = env.step(action)
            done = done or j == env._max_episode_steps
            #im_list.append(env.render().squeeze())
        if args.env_name == 'PointBot-v0':
            plt.ylim((env.grid[2], env.grid[3]))
            plt.xlim((env.grid[0], env.grid[1]))
            plt.scatter([env.hist[0][0]],[env.hist[0][2]],  [8], '#00FF00', zorder=11)
            x, y = [env.hist[i][0] for i in range(1, len(env.hist))], [env.hist[i][2] for i in range(1, len(env.hist))]
            plt.scatter(x, y, [6]*len(x), zorder =9)
            if TRASH:
                for j in env.current_trash_taken:
                    plt.scatter([j[0]],[j[1]], [25], zorder = 10, color = '#000000')
                plt.scatter([env.next_trash[0]],[env.next_trash[1]], [25], zorder = 10, color = '#000000')
            x_bounds = [obstacle.boundsx for obstacle in env.obstacle.obs]
            y_bounds = [obstacle.boundsy for obstacle in env.obstacle.obs]
            for i in range(len(x_bounds)):
                plt.gca().add_patch(patches.Rectangle((x_bounds[i][0], y_bounds[i][0]), width=x_bounds[i][1] - x_bounds[i][0], height=y_bounds[i][1] - y_bounds[i][0], fill=True, alpha=.5, linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3'))
            
            if not os.path.exists('GAIL_viz_' + str(folder)):
                os.makedirs('GAIL_viz_' + str(folder))
            if not os.path.exists('GAIL_viz_' + str(folder) + '/training'):
                os.makedirs('GAIL_viz_' + str(folder) + '/training')
            count += 1
            if training:
                plt.savefig('GAIL_viz_' + str(folder)+ '/training/training_' + str(training_i) + '.png')
            else:
                plt.savefig('GAIL_viz_' + str(folder)+ '/rollout_' + str(count) + '.png')
            plt.clf()
            obs_times.append(env.feature[0])
            num_trashes.append(env.feature[2])
    if args.env_name == 'PointBot-v0' and training == False:
        f = open('GAIL_viz_' + str(folder)+ '/rolloutAverages.txt', "w")
        f.write("Average Trashes: "+ str(np.average(num_trashes[-100:])))
        f.write("\nAverage Trashes Stnd Dev: "+ str(np.std(num_trashes[-100:])))
        f.write("\nAverage Obstacle Time: "+ str(np.average(obs_times[-100:])))
        f.write("\nAverage Obstacle Time Stnd Dev: "+ str(np.std(obs_times[-100:])))
        f.close()
    # Save gif


main_loop()
if args.env_name == 'PointBot-v0':
    env = gym.make(args.env_name)
    visualize_policy(policy_net, env, args.rollout, args.folder)
