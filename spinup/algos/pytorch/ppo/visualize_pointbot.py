import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
import time
from tqdm import tqdm
import os, sys
from json import JSONEncoder
import json
import spinup.algos.pytorch.vpg.core as core
from spinup.algos.pytorch.ppo.broil_ppo import *
from spinup.utils.logx import EpochLogger
from spinup.envs.pointbot_const import *
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.examples.pytorch.broil_rtg_pg_v2.pointbot_reward_utils import PointBotReward
from spinup.examples.pytorch.broil_rtg_pg_v2.cartpole_reward_utils import CartPoleReward
# from spinup.examples.pytorch.broil_rtg_pg_v2.cheetah_reward_utils import CheetahReward
from spinup.examples.pytorch.broil_rtg_pg_v2.reacher_reward_utils import ReacherReward
from spinup.examples.pytorch.broil_rtg_pg_v2.shelf_reward_utils import ShelfReward
from spinup.examples.pytorch.broil_rtg_pg_v2.cvar_utils import cvar_enumerate_pg
import dmc2gym

def visualize_policy(env, num_rollouts, ac, num_rew_fns, std_r, mean_r, reward_dist, local_steps_per_epoch, broil_alpha, file_data, args, max_ep_len, t, broil_lambda, training=False):
    
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    cvar_list = []
    wc_ret_list = []
    bc_ret_list = []
    ret_list = []
    obstacle_list = []
    trajectories_x = []
    trajectories_y = []
    trash_trajectories = []
    num_trashes = []
    obs_times = []
    count = 0
    # Main loop: collect experience in env and update/log each epoch
    for i in range(num_rollouts):
        print("Rollout: ", i)
        total_reward_dist = np.zeros(num_rew_fns)
        running_ret = 0
        num_runs = 1
        obstacles = 0
        num_constraint_violations = 0
        num_episodes = 0
        done = False
        count += 1
        constraint_violated = False
        j = 0
        while not done:
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            # TODO: Test unnormalizing the values
            v = (v * std_r.numpy()) + mean_r.numpy()
            j += 1
            next_o, r, d, info = env.step(a)
            
            rew_dist = reward_dist.get_reward_distribution(env, next_o)
            total_reward_dist += rew_dist.flatten()
            running_ret += r
            ep_ret += r
            ep_len += 1
            if args.env == 'PointBot-v0':
                obstacles += int(env.obstacle(next_o))
            # Update obs (critical!)
            o = next_o


            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            done = done or j == env._max_episode_steps

        # Store stuff for saving data

        posterior_reward_weights = reward_dist.posterior

        #calculate sigma and find either the conditional value at risk or entropic risk measure given the current policy
            #Calculate policy gradient for conditional value at risk
        sigma, cvar = cvar_enumerate_pg(total_reward_dist, posterior_reward_weights, broil_alpha)

        ret_list.append(running_ret / float(num_runs))
        wc_ret_list.append(np.min(total_reward_dist) / float(num_runs))
        bc_ret_list.append(np.max(total_reward_dist / float(num_runs)))
        cvar_list.append(cvar)
        obstacle_list.append(obstacles / float(num_runs))
        obs_times.append(env.feature[0])
        if TRASH:
            num_trashes.append(env.feature[2])
        print('True returns:', ret_list)
        print('Cvar: ', cvar_list)
        print('Worst case:', wc_ret_list)

        if args.env == 'PointBot-v0':
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
            
            if not os.path.exists(file_data + 'Rollouts'):
                os.makedirs(file_data + 'Rollouts')
            plt.savefig(file_data + 'Rollouts'+ '/rollout_' + str(count) + '.png')
            plt.clf()
        
        o, ep_ret, ep_len = env.reset(), 0, 0
    
    experiment_name = args.env + '_alpha_' + str(broil_alpha) + '_lambda_' + str(broil_lambda) + '_rollouts'

    metrics = {"conditional value at risk": ('_cvar', cvar_list),
            "true_return": ('_true_return', ret_list),
            "worst case return": ('_worst_case_return', wc_ret_list),
            "best case return": ('_best_case_return', bc_ret_list),
            "obstacle_collision": ('_obstacles', obstacle_list)}

    for metric, result in metrics.items():
        file_metric_description, results = result
        file_path = file_data + 'Rollouts/' + experiment_name + file_metric_description + '.txt'
        #assert not os.path.isfile(file_path)  # make sure we are making a new file and not overwriting
        with open(file_path, 'w') as f:
            for item in results:
                f.write("%s\n" % item)

    f = open(file_data + '/rolloutAverages.txt', "w")
    f.write("Trash: " + str(num_trashes))
    f.write("\nAverage Trashes: "+ str(np.average(num_trashes[-100:])))
    f.write("\nAverage Trashes Stnd Dev: "+ str(np.std(num_trashes[-100:])))
    f.write("\nObs times: " + str(obs_times))
    f.write("\nAverage Obstacle Time: "+ str(np.average(obs_times[-100:])))
    f.write("\nAverage Obstacle Time Stnd Dev: "+ str(np.std(obs_times[-100:])))
    f.close()
