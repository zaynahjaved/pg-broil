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
from spinup.envs.pointbot_const import *
from spinup.algos.pytorch.ppo.visualize_pointbot import *
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from spinup.examples.pytorch.broil_rtg_pg_v2.pointbot_reward_utils import PointBotReward
from spinup.examples.pytorch.broil_rtg_pg_v2.pointbot_reward_brex import PointBotRewardBrex
from spinup.examples.pytorch.broil_rtg_pg_v2.reacher_reward_brex import ReacherRewardBrex
from spinup.examples.pytorch.broil_rtg_pg_v2.cartpole_reward_utils import CartPoleReward
from spinup.examples.pytorch.broil_rtg_pg_v2.manipulator_reward_utils import ManipulatorReward
from spinup.examples.pytorch.broil_rtg_pg_v2.safety_gym_reward_utils import SafetyGymReward
from spinup.examples.pytorch.broil_rtg_pg_v2.shelf_reward_utils import ShelfReward
from spinup.examples.pytorch.broil_rtg_pg_v2.push_reward_utils import PushReward
from spinup.examples.pytorch.broil_rtg_pg_v2.itch_reward_utils import ItchReward
from spinup.examples.pytorch.broil_rtg_pg_v2.boxing_reward_utils import BoxingReward
# from spinup.examples.pytorch.broil_rtg_pg_v2.boxing_reward_brex import BoxingRewardBrex

from spinup.examples.pytorch.broil_rtg_pg_v2.cvar_utils import cvar_enumerate_pg
import dmc2gym
from cem_policy import CEM
from arg_parser import parse_args

torchify = lambda x: torch.FloatTensor(x).to(torch.device('cpu'))

from spinup.utils.run_utils import setup_logger_kwargs
logger_kwargs = setup_logger_kwargs("mpc-broil", 0) #seed


def mpc(env_fn, reward_dist, broil_risk_metric='cvar', ac_kwargs=dict(), render=False, seed=0, max_ep_len=100, logger_kwargs=dict(),
        steps_per_epoch=1000, epochs=100, broil_lambda=1, broil_alpha=0.95):

    def get_reward_distribution(args, reward_dist, env, next_o, env_rew, action):
        if args.env == 'CartPole-v0':
            rew_dist = reward_dist.get_reward_distribution(next_o)
        elif args.env == 'PointBot-v0':
            rew_dist = reward_dist.get_reward_distribution(env, next_o)
        else:
            raise NotImplementedError("Unsupported Environment")
        return rew_dist

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create BROIL actor-critic module
    num_rew_fns = reward_dist.posterior.size
   
    # Count variables
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
   
    mean_r = torch.zeros(num_rew_fns)
    std_r = torch.zeros(num_rew_fns)
    params = parse_args()
    policy = CEM(env, args.env, broil_alpha, broil_lambda, reward_dist, params)

    # Prepare for interaction with environment
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

    file_data = 'mpc_test/'
    experiment_name = args.env + '_alpha_' + str(broil_alpha) + '_lambda_' + str(broil_lambda)
    file_folder = os.path.join(os.getcwd(), file_data[:-1])
    os.makedirs(file_folder, exist_ok=True)
    os.makedirs(os.path.join(file_folder, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(file_folder, "results"), exist_ok=True)
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in tqdm(range(epochs)):
        first_rollout = True
        running_cvar = []
        total_reward_dist = np.zeros(num_rew_fns)
        num_runs = 0
        epoch_ret = 0
        obstacles = 0
        num_episodes = 0
        for t in tqdm(range(local_steps_per_epoch)):
            a = policy.act(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, info = env.step(a)
            rew_dist = get_reward_distribution(args, reward_dist, env, next_o, r, a)
            total_reward_dist += rew_dist.flatten()
            epoch_ret += r
            ep_len += 1
            if args.env == 'PointBot-v0':
                obstacles += int(env.obstacle(next_o))

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if render and first_rollout:
                env.render()
                time.sleep(0.01)

            if terminal or epoch_ended:
                num_episodes += 1

                first_rollout = False
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    print(ep_ret)
                    logger.store(EpRet=ep_ret, EpLen=ep_len)

                num_runs += 1

                if args.env == 'PointBot-v0':
                    last_trajectory = np.array(env.hist)
                    trajectories_x.append(last_trajectory[:, 0])
                    trajectories_y.append(last_trajectory[:, 2])

                o, ep_ret, ep_len = env.reset(), 0, 0

        # Store stuff for saving data
        ret_list.append(epoch_ret)
        wc_ret_list.append(np.min(total_reward_dist) / float(num_runs))
        bc_ret_list.append(np.max(total_reward_dist / float(num_runs)))
        cvar_list.append(sum(running_cvar))
        obstacle_list.append(obstacles / float(num_runs))
        
        """print('True returns:', ret_list)
        print('Cvar: ', cvar_list)
        print('Worst case:', wc_ret_list)"""


        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('ExpectedRet', epoch_ret)
        logger.dump_tabular()

        if args.env == 'PointBot-v0':
            plt.ylim((env.grid[2], env.grid[3]))
            plt.xlim((env.grid[0], env.grid[1]))
            for i in range(1):
                x = trajectories_x[i]
                y = trajectories_y[i]
                plt.scatter([x[0]],[y[0]],  [6], '#00FF00', zorder=11)
                plt.scatter(x[1:], y[1:], len(x[1:])*[6], zorder=9)
            #Create Single PointBot Visualization for lambda
            x_bounds = [obstacle.boundsx for obstacle in env.obstacle.obs]
            y_bounds = [obstacle.boundsy for obstacle in env.obstacle.obs]
            for i in range(len(x_bounds)):
                plt.gca().add_patch(patches.Rectangle((x_bounds[i][0], y_bounds[i][0]), width=x_bounds[i][1] - x_bounds[i][0], height=y_bounds[i][1] - y_bounds[i][0], fill=True, alpha=.5, linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3'))
            plt.savefig(file_data + 'visualizations/' + experiment_name + '_' + str(epoch) + '.png')
            plt.clf()

    metrics = {"conditional value at risk": ('_cvar', cvar_list),
               "true_return": ('_true_return', ret_list),
               "worst case return": ('_worst_case_return', wc_ret_list),
               "best case return": ('_best_case_return', bc_ret_list),
               "obstacle_collision": ('_obstacles', obstacle_list)}

    for metric, result in metrics.items():
        file_metric_description, results = result
        file_path = file_data + 'results/' + experiment_name + file_metric_description + '.txt'
        os.makedirs(file_data + 'results/', exist_ok=True)
        with open(file_path, 'w') as f:
            for item in results:
                f.write("%s\n" % item)

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)
    
    #Create PointBot Trajectory Visualization with Multiple Lambdas and Input for grapher.py
    if args.env == 'PointBot-v0':
        plt.ylim((env.grid[2], env.grid[3]))
        plt.xlim((env.grid[0], env.grid[1]))
        for i in range(1):
            x = trajectories_x[i]
            y = trajectories_y[i]
            plt.scatter([x[0]],[y[0]],  [6], '#00FF00', zorder=11)
            plt.scatter(x[1:], y[1:], len(x[1:])*[6], zorder=9)
        #Create Single PointBot Visualization for lambda
        x_bounds = [obstacle.boundsx for obstacle in env.obstacle.obs]
        y_bounds = [obstacle.boundsy for obstacle in env.obstacle.obs]
        for i in range(len(x_bounds)):
            plt.gca().add_patch(patches.Rectangle((x_bounds[i][0], y_bounds[i][0]), width=x_bounds[i][1] - x_bounds[i][0], height=y_bounds[i][1] - y_bounds[i][0], fill=True, alpha=.5, linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3'))
        plt.savefig(file_data + 'visualizations/' + experiment_name + '_' + str(epoch) + '.png')
        plt.clf()

    print('Data from experiment: ', experiment_name, ' saved.')

if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PointBot-v0')
    parser.add_argument('--brex', type=bool, default=False)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--policy_lr', type=float, default=3e-4, help="learning rate for policy")
    parser.add_argument('--value_lr', type=float, default=1e-3)
    parser.add_argument('--risk_metric', type=str, default='cvar', help='choice of risk metric, options are "cvar" or "erm"' )
    parser.add_argument('--broil_lambda', type=float, default=0.1, help="blending between cvar and expret")
    parser.add_argument('--broil_alpha', type=float, default=0.95, help="risk sensitivity for cvar")
    parser.add_argument('--clone', action="store_true", help="do behavior cloning")
    parser.add_argument('--num_demos', type=int, default=0)
    parser.add_argument('--baseline_file', type=str, default='', help="file with feature counts of expert")
    parser.add_argument('--combiner', type=bool, default=False)
    parser.add_argument('--num_rollouts', type=int, default=100)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    if args.env == 'PointBot-v0':
        if args.brex:
            reward_dist = PointBotRewardBrex()
        else:
            reward_dist = PointBotReward()

    if args.env == 'reacher':
        env_fn = lambda: dmc2gym.make(domain_name='reacher', task_name='easy',visualize_reward=False,from_pixels=True,seed=args.seed, episode_length=200)
    elif args.env == 'manipulator':
        env_fn = lambda: dmc2gym.make(domain_name='manipulator', task_name='bring_ball', seed=args.seed)
    else:
        env_fn = lambda : gym.make(args.env)

    # if args.baseline_file == '':
    #     fcount_baseline = None #business as usual
    # else:
    #     #need to try and read in the file. I'll assume one line of comma separated feature counts
    #     freader = open(args.baseline_file, 'r')
    #     line = freader.read().strip()
    #     vals = []
    #     for val in line.split(','):
    #         vals.append(float(val))
    #     fcount_baseline = np.array(vals)

    mpc(env_fn, reward_dist=reward_dist, broil_risk_metric=args.risk_metric, broil_lambda=args.broil_lambda, broil_alpha=args.broil_alpha,
        render=args.render, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), logger_kwargs=logger_kwargs,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)