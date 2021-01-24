import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
import copy
import os
import pickle
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
    def __init__(self, line, env, fig, typ):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.xv = [0]
        self.yv = [0]
        self.xt = []
        self.yt = []
        self.typ = typ
        self.env = env
        self.state = env.reset()
        self.states = [self.state]
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.info = fig.canvas.mpl_connect('key_press_event', self.press)

    def press(self, event):
        sys.stdout.flush()
        if event.key == 'v':
            print("X_v: ", str(self.xv[-1]), " Y_v: ", str(self.yv[-1]))
        if event.key == 'p':
            print("X: ", str(self.xs[-1]), " Y: ", str(self.ys[-1]))
        if event.key == 'w':
            print("Most recent state: ", str(self.states[-1]), " states: ", str(self.states))
        if event.key == 't':
            if TRASH:
                if len(self.env.remaining_trash) is not 0:
                    print("Closest Trash X: ", str(self.env.closest_trash(self.states[-1])[0]), " Y: ", str(self.env.closest_trash(self.states[-1])[1]))
                else:
                    print("No more TRASH on the field!")
                    plt.savefig('demonstrations/visualization_' + self.typ + "_" + str(args.dem_num) + '.png')
                    plt.close()
            else:
                print("No TRASH on the field!")
        if event.key == 'a':
            print("Current Feature: ", str(self.env.feature))
        if event.key == 'e':
            if TRASH:
                plt.savefig('demonstrations/visualization' + self.typ + "_" + str(args.dem_num) + '.png')
                plt.close()
            else:
                if np.linalg.norm(np.subtract(GOAL_STATE, self.states[-1][:4])) <= GOAL_THRESH:
                    plt.savefig('demonstrations/visualization' + self.typ + "_" + str(args.dem_num) + '.png')
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

            x_f = diff_x
            y_f = diff_y

            if abs(diff_x) >= MAX_FORCE and abs(diff_y) >= MAX_FORCE:
                if diff_x >= 0 and diff_y >= 0:
                    if diff_x >= diff_y:
                        x_f = abs(diff_x/diff_x) * MAX_FORCE/10
                        y_f = abs(diff_y/diff_x) * MAX_FORCE/10
                    else:
                        x_f = abs(diff_x/diff_y) * MAX_FORCE/10
                        y_f = abs(diff_y/diff_y) * MAX_FORCE/10

                elif diff_x < 0 and diff_y >= 0:
                    if diff_x >= diff_y:
                        x_f = -abs(diff_x/diff_x) * MAX_FORCE/10
                        y_f = abs(diff_y/diff_x) * MAX_FORCE/10
                    else:
                        x_f = -abs(diff_x/diff_y) * MAX_FORCE/10
                        y_f = abs(diff_y/diff_y) * MAX_FORCE/10

                elif diff_x >= 0 and diff_y < 0:
                    if diff_x >= diff_y:
                        x_f = abs(diff_x/diff_x) * MAX_FORCE/10
                        y_f = -abs(diff_y/diff_x) * MAX_FORCE/10
                    else:
                        x_f = abs(diff_x/diff_y) * MAX_FORCE/10
                        y_f = -abs(diff_y/diff_y) * MAX_FORCE/10

                elif diff_x < 0 and diff_y < 0:
                    if diff_x >= diff_y:
                        x_f = -abs(diff_x/diff_x) * MAX_FORCE/5
                        y_f = -abs(diff_y/diff_x) * MAX_FORCE/10
                    else:
                        x_f = -abs(diff_x/diff_y) * MAX_FORCE/10
                        y_f = -abs(diff_y/diff_y) * MAX_FORCE/10

            act = tuple((x_f, y_f))
            new_state, _, _, _ = self.env.step(act)

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

        x_f = diff_x
        y_f = diff_y

        if abs(diff_x) >= MAX_FORCE and abs(diff_y) >= MAX_FORCE:
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
        new_state, _, _, _ = self.env.step(act)

        self.xs.append(new_state[0])
        self.ys.append(new_state[2])

        self.xv.append(new_state[1])
        self.yv.append(new_state[3])
        self.states.append(new_state)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

def init(typ="Good"):
    env = gym.make('PointBot-v0')
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('PointBot Env '+ typ +' Demonstrator')
    line, = ax.plot([START_POS[0]], START_POS[1])  # empty line
    linebuilder = LineBuilder(line, env, fig, typ)
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
    if not TRASH:
        ax.scatter([END_POS[0]],[END_POS[1]],  [5], '#FF0000')

    ax.set_xlim([env.grid[0], env.grid[1]])
    ax.set_ylim([env.grid[2], env.grid[3]])
    plt.show()
    return linebuilder

def end(linebuilder, typ="Good"):
    if TRASH:
        for _ in range(101-len(linebuilder.states)):
            next_state = [END_POS[0], 0, END_POS[1], 0] + NOISE_SCALE * np.random.randn(4)
            next_state = np.concatenate((next_state, linebuilder.env.closest_trash(GOAL_STATE)))
            linebuilder.states.append(next_state)
    else:
        for _ in range(101-len(linebuilder.states)):
            linebuilder.states.append([END_POS[0], 0, END_POS[1], 0] + NOISE_SCALE * np.random.randn(4))

    if not os.path.exists('demonstrations'):
        os.makedirs('demonstrations')
    
    try:
        f = open("demonstrations/states_" + str(args.dem_num) + ".txt", "a")
        assert linebuilder.env.feature[0] + linebuilder.env.feature[1] < HORIZON + 1, "ERROR: Number of states is greater than the HORIZON!"
        linebuilder.env.feature[1] = HORIZON + 1 - linebuilder.env.feature[0]
        print(linebuilder.env.feature)
        f.write("\n" + typ)
        f.write("\nFeature: " + str(linebuilder.env.feature))
        f.write("\n\nStates: " + str(linebuilder.states))
        if TRASH:
            f.write("\n\nTrash Locations: "+ str(linebuilder.env.trash_locs))
        f.close()
        return linebuilder.env.feature
    except AssertionError as msg:  
        print(msg) 
        return None


if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--dem_num', type=int, default=1)
    args = parser.parse_args()

    linebuilder = init()
    good = end(linebuilder)

    linebuilder = init("Bad")
    bad = end(linebuilder, "Bad")

    dic = {"Good": good, "Bad": bad}
    p = open("demonstrations/features_" + str(args.dem_num) + ".pkl", "wb")
    pickle.dump(dic, p)
    p.close()

    

    
