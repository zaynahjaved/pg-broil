import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
import time
import random
import json
from tqdm import tqdm
import datetime
import os, sys
from os import listdir, rename
from os.path import isfile, join
import spinup.algos.pytorch.vpg.core as core
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
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

x_vals = []
y_vals = []
name_of_grid_search = 'broil_data_120/'

x_vals = ["1.0_x", "0.45_x", "0.4_x", "0.0_x"]
y_vals = ["1.0_y", "0.45_y", "0.4_y", "0.0_y"]

#x_vals = ["1.0_x", "0.8_x", "0.6_x", "0.45_x", "0.2_x","0.0_x"]
#y_vals = ["1.0_y", "0.8_y", "0.6_y", "0.45_y", "0.2_y","0.0_y"]

j = open(name_of_grid_search + "states_combined.txt", "r")
dedict = j.readline()
dedict = json.loads(dedict)
j.close()

experiment_name = "Maze_ppo"

plt.ylim((-50, 210))
plt.xlim((-180, 10))
# for below colors = [red, orange, yellow, green, blue, indigo, violet]
colors = ['#FF0000', '#ffa500', '#ffff00', '#008000', '#0000FF', '#4B0082', '#8F00FF']
#colors = ['#03071e', '#d00000', '#dc2f02', '#e85d04', '#faa307', '#ffba08']

env = gym.make('PointBot-v0')
fig, ax = plt.subplots()
num_obst = len(env.obstacle.obs)
for i in range(num_obst):
    xbound = env.obstacle.obs[i].boundsx
    ybound = env.obstacle.obs[i].boundsy
    rect = patches.Rectangle((xbound[0],ybound[0]),abs(xbound[1] - xbound[0]),abs(ybound[1] - ybound[0]), linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3', fill=True)
    ax.add_patch(rect)
ax.scatter([START_POS[0]],[START_POS[1]],  [20], '#00FF00', zorder=3, marker='^')
ax.scatter([END_POS[0]],[END_POS[1]],  [20], '#FF0000', zorder=3, marker='*')
size = [20, 15, 8, 18, 18, 18]
for j in range(6):
    for i in range(1):
        x = dedict[x_vals[j]][i]
        y = dedict[y_vals[j]][i]
        la = None
        if i==0:
            la = 'λ=' + x_vals[j][:len(x_vals[j])-2]
        plt.scatter(x, y, len(x)*[size[j]], colors[j], label=la, zorder=2)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.06),
          fancybox=False, shadow=True, ncol=3, prop={'size': 14})
plt.axis('off')
plt.savefig(name_of_grid_search + experiment_name + '.png')
plt.clf()

plot_comparison = [(0.95, 1.0), (0.95, 0.45), (0.95, 0.4), (0.95, 0.2), (0.95, 0.0)]
metric = ['true_return', 'cvar']
all_files = [f for f in listdir(name_of_grid_search + 'Rollouts') if isfile(join(name_of_grid_search + 'Rollouts', f))]

risks, rets = [], []
for m in metric:
    for i in range(len(plot_comparison)):
        for f in all_files:
            split_by_underscore = f.split('_')
            a, l = float(split_by_underscore[2]), float(split_by_underscore[4])
            if l == plot_comparison[i][1] and m in f:
                name = join(name_of_grid_search + 'Rollouts', f)
                with open(name, 'r') as opened_file:
                    data = opened_file.read().split('\n')
                data = [float(d) for d in data if len(d) > 0]
                avg = sum(data[:])/100
                if m == 'true_return':
                    rets.append(avg)
                elif m == 'cvar':
                    risks.append(avg)

plt.figure()
plt.plot(risks, rets, '-o')

#go through and label the points in the figure with the corresponding lambda values
unique_pts_lambdas = []
lambda_range = [1.0, 0.45, 0.4, 0.2, 0.0]

unique_pts = []

cvar_rets_array = [tuple((risks[i], rets[i])) for i in range(len(risks))]

for i,pt in enumerate(cvar_rets_array):
    unique = True
    for upt in unique_pts:
        if np.linalg.norm(upt - pt) < 0.00001:
            unique = False
            break
    if unique:
        unique_pts_lambdas.append((pt[0], pt[1], lambda_range[i]))
        unique_pts.append(np.array(pt))

#calculate offset
offsetx = (np.max(risks) - np.min(risks))/30
offsety = (np.max(rets) - np.min(rets))/17

for i,pt in enumerate(unique_pts_lambdas):
    if i in [0]:
        plt.text(pt[0] + 2.2*offsetx, pt[1] - 0.5*offsety, r"$\lambda = {}$".format('1.0, 0.8, 0.6'), fontsize=19,  fontweight='bold')
    elif i in [2]:
        plt.text(pt[0] + 0.9*offsetx, pt[1] + 0.6*offsety , r"$\lambda = {}$".format(str(pt[2])), fontsize=19,  fontweight='bold')
    elif i in [3]:
        plt.text(pt[0] + 0.6*offsetx, pt[1] - .3*offsety , r"$\lambda = {}$".format(str(pt[2])), fontsize=19,  fontweight='bold')
    elif i in [4]:
        plt.text(pt[0] - 8*offsetx, pt[1] - .55*offsety , r"$\lambda = {}$".format(str(pt[2])), fontsize=19,  fontweight='bold')
    elif i in [1]:
        plt.text(pt[0] + offsetx, pt[1] + 0.5*offsety, r"$\lambda = {}$".format(str(pt[2])), fontsize=19,  fontweight='bold')
    else:
        plt.text(pt[0]-offsetx, pt[1] - 1.5*offsety, r"$\lambda = {}$".format(str(pt[2])), fontsize=19,  fontweight='bold')

plt.xticks(np.arange(-13000, -6999, 1500), fontsize=16) 
plt.yticks(fontsize=16) 
plt.xlabel("Robustness (ERM)", fontsize=18)
plt.ylabel("Expected Return", fontsize=18)

plt.tight_layout()
plt.savefig(name_of_grid_search + 'visualizations/' + experiment_name + "_frontier" + '.png')
plt.clf()

