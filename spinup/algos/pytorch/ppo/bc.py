import spinup.algos.pytorch.vpg.core as core
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import moviepy.editor as mpy
import gym
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from spinup.envs.pointbot_const import *

torchify = lambda x: torch.FloatTensor(x).to(torch.device('cpu'))

def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')

# Maximize log_pi for acs in demos
def policy_BC_loss(pi, demo_obs, demo_acs):
    # Policy loss
    loss_pi = 0
    for i in range(len(demo_acs)):
        obs_batch = torchify(np.array(demo_obs[i][:-1]))
        acs_batch = torchify(np.array(demo_acs[i]))
        _, logp = pi(obs_batch, acs_batch)
        loss_pi += (-(logp).mean())
    return loss_pi

def visualize_policy(pi, env, num_rollouts, folder):
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
            policy = pi._distribution(torch.as_tensor(o, dtype=torch.float32))
            a = policy.sample()
            o, _, done, _ = env.step(a.cpu().detach().numpy())
            done = done or j == env._max_episode_steps
            #im_list.append(env.render().squeeze())
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
            
            if not os.path.exists('BC_viz_' + str(folder)):
                os.makedirs('BC_viz_' + str(folder))
            count += 1
            plt.savefig('BC_viz_' + str(folder)+ '/rollout_' + str(count) + '.png')
            plt.clf()
            obs_times.append(env.feature[0])
            num_trashes.append(env.feature[2])
    if args.env == 'PointBot-v0':
        f = open('BC_viz_' + str(folder)+ '/rolloutAverages.txt', "w")
        f.write("Trash: " + str(num_trashes))
        f.write("\nAverage Trashes: "+ str(np.average(num_trashes[-100:])))
        f.write("\nAverage Trashes Stnd Dev: "+ str(np.std(num_trashes[-100:])))
        f.write("\nObs times: " + str(obs_times))
        f.write("\nAverage Obstacle Time: "+ str(np.average(obs_times[-100:])))
        f.write("\nAverage Obstacle Time Stnd Dev: "+ str(np.std(obs_times[-100:])))
        f.close()
    # Save gif
    #npy_to_gif(im_list, "vis_" + str(i) + ".gif")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Shelf-v0')
    parser.add_argument('--seed', '-s', type=int, default=4)
    parser.add_argument('--pi_lr', type=float, default=1e-2, help="learning rate for policy")
    parser.add_argument('--clone', action="store_true", help="do behavior cloning")
    parser.add_argument('--num_demos', type=int, default=6)
    parser.add_argument('--BC_iters', type=int, default=1000)
    parser.add_argument('--folder', type=int, default=1)
    parser.add_argument('--num_rollouts', type=int, default=100)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    demo_obs, demo_acs = env.get_demos(args.num_demos)

    pi = core.MLPGaussianActor(env.observation_space.shape[0], env.action_space.shape[0], hidden_sizes=(64,64), activation=nn.Tanh)
    # Set up optimizer for policy
    pi_optimizer = Adam(pi.parameters(), lr=args.pi_lr)
    # Train BC policy
    for i in range(args.BC_iters):
        if i % 100 == 0:
            print("BC iter: ", i)
        pi_optimizer.zero_grad()
        BC_loss = policy_BC_loss(pi, demo_obs, demo_acs)
        BC_loss.backward()
        pi_optimizer.step()
    # Visualize policy
    visualize_policy(pi, env, args.num_rollouts, args.folder)
