import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
import datetime
import spinup.algos.pytorch.vpg.core as core
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from spinup.examples.pytorch.broil_rtg_pg_v2.pointbot_reward_utils import PointBotReward
from spinup.envs.pointbot_const import GOAL_STATE


def model_eval(ac_file, parameters, seed=0):
        steps = 200
        dist_to_goal = 2

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Random seed
        np.random.seed(seed)

        # Instantiate environment
        env_fn=lambda : gym.make('PointBot-v0')
        env = env_fn()

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        # Create actor-critic module
        actor_critic=core.BROILActorCritic
        ac = actor_critic(env.observation_space, env.action_space, 5)
        ac.load_state_dict(torch.load('PointBot_Networks/'+ ac_file))
        ac.eval()

        # Sync params across processes
        sync_params(ac)

        # Prepare for interaction with environment
        o, ep_ret, ep_len = env.reset(), 0, 0

        trajectories_x = []
        trajectories_y = []
        inside_obstacle = [0, 0]

        for t in range(steps):
            a, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            #Calculate how many times bot is inside/outside obstacle
            if env.obstacle(o):
                inside_obstacle[0] += 1
            else:
                inside_obstacle[1] += 1

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == steps
            terminal = d or timeout or np.linalg.norm(np.subtract(GOAL_STATE, o)) < dist_to_goal
            epoch_ended = t==steps-1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0

                last_trajectory = np.array(env.hist)
                trajectories_x.append(last_trajectory[:, 0])
                trajectories_y.append(last_trajectory[:, 2])

                o, ep_ret, ep_len = env.reset(), 0, 0

                if terminal:
                    break

        experiment_name = 'test_' + parameters
        folder = 'brex_visuals/'

        plt.ylim((-50, 75))
        plt.xlim((-125, 25))
        for i in range(1):
            x = trajectories_x[i]
            y = trajectories_y[i]
            plt.scatter(x, y)

        x_bounds = [obstacle.boundsx for obstacle in env.obstacle.obs]
        y_bounds = [obstacle.boundsy for obstacle in env.obstacle.obs]
        for i in range(len(x_bounds)):
            plt.gca().add_patch(patches.Rectangle((x_bounds[i][0], y_bounds[i][0]), width=x_bounds[i][1] - x_bounds[i][0], height=y_bounds[i][1] - y_bounds[i][0], fill=True, alpha=.5))
        plt.savefig(folder + experiment_name + '.png')
        plt.clf()

        print(inside_obstacle)
        return inside_obstacle
