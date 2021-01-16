import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
import time
import json
from tqdm import tqdm
import datetime
import os, sys
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
name_of_grid_search = 'broil_data106/'

x_vals = ["0_x", "0.1_x", "0.2_x", "0.8_x", "0.9_x", "1_x"]
y_vals = ["0_y", "0.1_y", "0.2_y", "0.8_y", "0.9_y", "1_y"]

j = open(name_of_grid_search + "states_combined.txt", "r")
dedict = j.readline()
dedict = json.loads(dedict)
j.close()

experiment_name = "Two_obstacles_narrow_corridor"

plt.ylim((-50, 210))
plt.xlim((-180, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
env = gym.make('PointBot-v0')
fig, ax = plt.subplots()
ax.set_title('Two Obstacles with Narrow Corridor with Alpha = 0.95')
num_obst = len(env.obstacle.obs)
for i in range(num_obst):
    xbound = env.obstacle.obs[i].boundsx
    ybound = env.obstacle.obs[i].boundsy
    rect = patches.Rectangle((xbound[0],ybound[0]),abs(xbound[1] - xbound[0]),abs(ybound[1] - ybound[0]), linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3', fill = True)
    ax.add_patch(rect)
ax.scatter([START_POS[0]],[START_POS[1]],  [5], '#00FF00')
ax.scatter([END_POS[0]],[END_POS[1]],  [5], '#FF0000')

for j in range(6):
    for i in range(2):
        x = dedict[x_vals[j]][i]
        y = dedict[y_vals[j]][i]
        la = None
        if i==0:
            la = 'Lambda = ' + x_vals[j][:len(x_vals[j])-2]
        plt.scatter(x, y, len(x)*[10], colors[j], label=la)

plt.legend()
plt.savefig(name_of_grid_search + 'visualizations/' + experiment_name + '.png')
plt.clf()