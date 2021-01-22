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
name_of_grid_search = 'broil_data108/'

x_vals = ["1_x", "0.8_x", "0.4_x", "0.2_x","0_x"]
y_vals = ["1_y", "0.8_y", "0.4_y", "0.2_y","0_y"]

j = open(name_of_grid_search + "states_combined.txt", "r")
dedict = j.readline()
dedict = json.loads(dedict)
j.close()

experiment_name = "Maze_ppo"

plt.ylim((-50, 210))
plt.xlim((-180, 10))
# for below colors = [red, orange, yellow, green, blue, indigo, violet]
colors = ['#FF0000', '#ffa500', '#ffff00', '#00FF00', '#0000FF', '#4B0082', '#8F00FF']
env = gym.make('PointBot-v0')
fig, ax = plt.subplots()
ax.set_title('Maze Environment with Alpha = 0.95')
num_obst = len(env.obstacle.obs)
for i in range(num_obst):
    xbound = env.obstacle.obs[i].boundsx
    ybound = env.obstacle.obs[i].boundsy
    rect = patches.Rectangle((xbound[0],ybound[0]),abs(xbound[1] - xbound[0]),abs(ybound[1] - ybound[0]), linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3', fill = True)
    ax.add_patch(rect)
ax.scatter([START_POS[0]],[START_POS[1]],  [5], '#00FF00')
ax.scatter([END_POS[0]],[END_POS[1]],  [5], '#FF0000')
size = [18, 8, 10, 18, 8]
for j in range(5):
    for i in range(2):
        x = dedict[x_vals[j]][i]
        y = dedict[y_vals[j]][i]
        la = None
        if i==0:
            la = 'Lambda = ' + x_vals[j][:len(x_vals[j])-2]
        plt.scatter(x, y, len(x)*[size[j]], colors[j], label=la)

plt.legend()
plt.savefig(name_of_grid_search + 'visualizations/' + experiment_name + '.png')
plt.clf()

plot_comparison = [(0.95, 1), (0.95, 0.8), (0.95, 0.4), (0.95, 0.2), (0.95, 0)]
metric = ['true_return', 'cvar']
all_files = [f for f in listdir(name_of_grid_search + 'results') if isfile(join(name_of_grid_search + 'results', f))]

risks, rets = [], []
for m in metric:
    for i in range(len(plot_comparison)):
        for f in all_files:
            split_by_underscore = f.split('_')
            a, l = float(split_by_underscore[3]), float(split_by_underscore[5])
            if l == plot_comparison[i][1] and m in f:
                name = join(name_of_grid_search + 'results', f)
                with open(name, 'r') as opened_file:
                    data = opened_file.read().split('\n')
                data = [float(d) for d in data if len(d) > 0]
                avg = sum(data[-15:])/15
                if m == 'true_return':
                    rets.append(avg)
                elif m == 'cvar':
                    risks.append(avg)


labels=['λ=1, λ=0.8','λ=0.4','λ=0.2','λ=0']
risks = risks[1:]
rets = rets[1:]
plt.rcParams.update({'font.size': 20})

print(risks, rets)
plt.figure(figsize=(15,8))
plt.scatter(risks,rets)
plt.plot(risks,rets)
plt.xlabel('Robustness (CVaR)')
plt.ylabel('Expected Return')
for i, txt in enumerate(labels):
    plt.annotate(txt, (risks[i]+2, rets[i]+2))

plt.savefig(name_of_grid_search + 'visualizations/' + experiment_name + "_frontier" + '.png')
plt.clf()

