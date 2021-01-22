import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
import copy
import os
from gym.spaces import Discrete, Box
from spinup.envs.pointbot import *
import datetime
import os
import pickle
import math
import sys
import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from spinup.envs.pointbot_const import *

from spinup.examples.pytorch.broil_rtg_pg_v2.cvar_utils import cvar_enumerate_pg
from spinup.examples.pytorch.broil_rtg_pg_v2.pointbot_reward_utils import PointBotReward

class LineBuilder:
    def __init__(self, line, env):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.xv = [0]
        self.yv = [0]
        self.xt = []
        self.yt = []
        self.feature = [0, 1, 0] #[red region, white space, garabage collected]
        #self.feature = [0, 0] #[red region, white space]
        self.states = []
        self.env = env
        self.state = env.reset()
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.info = fig.canvas.mpl_connect('key_press_event', self.press)

    def press(self, event):
        sys.stdout.flush()
        if event.key == 'v':
            print("X_v: ", str(self.xv[-1]), " Y_v: ", str(self.yv[-1]))
        if event.key == 'p':
            print("X: ", str(self.xs[-1]), " Y: ", str(self.ys[-1]))
        if event.key == 't':
            print("Closest Trash X: ", str(self.xt[-1]), " Y: ", str(self.yt[-1]))
        if event.key == 'a':
            print("Current Feature: ", str(self.env.feature))
        if event.key == 'c':
            if math.sqrt(math.pow(self.xt[-1], 2) + math.pow(self.yt[-1], 2) * 1.0) < TRASH_RADIUS:
                idx_true = [i for i, val in enumerate(self.remaining_trash) if val]
                if len(idx_true) > 0:
                    idx_true = idx_true[0]
                    self.remaining_trash_locs.remove(self.remaining_trash_locs[idx_true])
                    self.remaining_trash.remove(True)
                    self.feature[2] += 1
            else:
                print("Too far to collect closest trash")
        if event.key == 'e':
            xmin = END_POS[0] - NOISE_SCALE * 10
            xmax = END_POS[0] + NOISE_SCALE * 10
            ymin = END_POS[1] - NOISE_SCALE * 10
            ymax = END_POS[1] + NOISE_SCALE * 10
            if xmin <= self.xs[-1] <= xmax and ymin <= self.ys[-1] <= ymax and -MAX_FORCE <= self.xv[-1] <= MAX_FORCE and -MAX_FORCE <= self.yv[-1] <= MAX_FORCE:
                plt.savefig('demonstrations/visualization' + str(args.dem_num) + '.png')
                plt.close()
            else:
                print("\nNot proper ending! X distance from END: " + str(self.xs[-1] - END_POS[0]) + " Y distance from END: " + str(self.ys[-1] - END_POS[1]))
        if event.key == 'r':
            if (os.path.exists("demonstrations/states_" + str(args.dem_num) + ".txt")):
                os.remove("demonstrations/states_" + str(args.dem_num) + ".txt")
        if event.key == 'g':
            if event.inaxes!=self.line.axes: return
            final_x = END_POS[0]
            final_y = END_POS[1]
            init_x = self.xs[-1]
            init_y = self.ys[-1]

            diff_x = final_x - init_x
            diff_y = final_y - init_y

            x_f = 0
            y_f = 0

            if diff_x >= 0 and diff_y >= 0:
                if diff_x >= diff_y:
                    x_f = abs(diff_x/diff_x) * MAX_FORCE
                    y_f = abs(diff_y/diff_x) * MAX_FORCE
                else:
                    x_f = abs(diff_x/diff_y) * MAX_FORCE
                    y_f = abs(diff_y/diff_y) * MAX_FORCE

            elif diff_x < 0 and diff_y >= 0:
                if diff_x >= diff_y:
                    x_f = -abs(diff_x/diff_x) * MAX_FORCE
                    y_f = abs(diff_y/diff_x) * MAX_FORCE
                else:
                    x_f = -abs(diff_x/diff_y) * MAX_FORCE
                    y_f = abs(diff_y/diff_y) * MAX_FORCE

            elif diff_x >= 0 and diff_y < 0:
                if diff_x >= diff_y:
                    x_f = abs(diff_x/diff_x) * MAX_FORCE
                    y_f = -abs(diff_y/diff_x) * MAX_FORCE
                else:
                    x_f = abs(diff_x/diff_y) * MAX_FORCE
                    y_f = -abs(diff_y/diff_y) * MAX_FORCE

            elif diff_x < 0 and diff_y < 0:
                if diff_x >= diff_y:
                    x_f = -abs(diff_x/diff_x) * MAX_FORCE
                    y_f = -abs(diff_y/diff_x) * MAX_FORCE
                else:
                    x_f = -abs(diff_x/diff_y) * MAX_FORCE
                    y_f = -abs(diff_y/diff_y) * MAX_FORCE

            act = tuple((x_f, y_f))
            new_state, _, _, _ = env.step(act)

            self.xs.append(new_state[0])
            self.ys.append(new_state[2])

            self.xv.append(new_state[1])
            self.yv.append(new_state[3])
            self.states.append(new_state)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

    def __call__(self, event):
        if event.inaxes!=self.line.axes: return
        
        final_x = event.xdata
        final_y = event.ydata
        init_x = self.xs[-1]
        init_y = self.ys[-1]

        diff_x = final_x - init_x
        diff_y = final_y - init_y

        x_f = 0
        y_f = 0

        if diff_x >= 0 and diff_y >= 0:
            if diff_x >= diff_y:
                x_f = abs(diff_x/diff_x) * MAX_FORCE
                y_f = abs(diff_y/diff_x) * MAX_FORCE
            else:
                x_f = abs(diff_x/diff_y) * MAX_FORCE
                y_f = abs(diff_y/diff_y) * MAX_FORCE

        elif diff_x < 0 and diff_y >= 0:
            if diff_x >= diff_y:
                x_f = -abs(diff_x/diff_x) * MAX_FORCE
                y_f = abs(diff_y/diff_x) * MAX_FORCE
            else:
                x_f = -abs(diff_x/diff_y) * MAX_FORCE
                y_f = abs(diff_y/diff_y) * MAX_FORCE

        elif diff_x >= 0 and diff_y < 0:
            if diff_x >= diff_y:
                x_f = abs(diff_x/diff_x) * MAX_FORCE
                y_f = -abs(diff_y/diff_x) * MAX_FORCE
            else:
                x_f = abs(diff_x/diff_y) * MAX_FORCE
                y_f = -abs(diff_y/diff_y) * MAX_FORCE

        elif diff_x < 0 and diff_y < 0:
            if diff_x >= diff_y:
                x_f = -abs(diff_x/diff_x) * MAX_FORCE
                y_f = -abs(diff_y/diff_x) * MAX_FORCE
            else:
                x_f = -abs(diff_x/diff_y) * MAX_FORCE
                y_f = -abs(diff_y/diff_y) * MAX_FORCE

        act = tuple((x_f, y_f))
        new_state, _, _, _ = env.step(act)

        self.xs.append(new_state[0])
        self.ys.append(new_state[2])

        self.xv.append(new_state[1])
        self.yv.append(new_state[3])
        self.states.append(new_state)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--dem_num', type=int, default=1)
    args = parser.parse_args()

    env = gym.make('PointBot-v0')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('PointBot Env Demonstrator')
    line, = ax.plot([START_POS[0]], START_POS[1])  # empty line
    linebuilder = LineBuilder(line, env)
    num_obst = len(env.obstacle.obs)
    for i in range(num_obst):
        xbound = env.obstacle.obs[i].boundsx
        ybound = env.obstacle.obs[i].boundsy
        rect = patches.Rectangle((xbound[0],ybound[0]),abs(xbound[1] - xbound[0]),abs(ybound[1] - ybound[0]),linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3', fill = True)
        ax.add_patch(rect)
    if TRASH:
        for i in env.trash_locs:
            ax.scatter([i[0]],[i[1]], [8], '#964b00')

    ax.scatter([START_POS[0]],[START_POS[1]],  [5], '#00FF00')
    ax.scatter([END_POS[0]],[END_POS[1]],  [5], '#FF0000')

    ax.set_xlim([env.grid[0], env.grid[1]])
    ax.set_ylim([env.grid[2], env.grid[3]])
    plt.show()

    for _ in range(101-len(linebuilder.states)):
        linebuilder.states.append([END_POS[0], 0, END_POS[1], 0] + NOISE_SCALE * np.random.randn(4))
    
    if not os.path.exists('demonstrations'):
        os.makedirs('demonstrations')
    
    try:
        f = open("demonstrations/states_" + str(args.dem_num) + ".txt", "a")
        assert env.feature[0] + env.feature[1] < HORIZON + 1, "ERROR: Number of states is greater than the HORIZON!"
        linebuilder.feature[1] = HORIZON + 1 - linebuilder.feature[0]
        print(linebuilder.env.feature)
        f.write("\nFeature:" + str(linebuilder.feature))
        f.write("\n\nStates:" + str(linebuilder.states))
        f.close()
        plt.clf()
    except AssertionError as msg:  
        print(msg) 
