"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise and air resistance
proportional to its velocity. State representation is (x, vx, y, vy). Action
representation is (fx, fy), and mass is assumed to be 1.
"""

import os
import pickle
import math
import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from .pointbot_const import *

def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

def lqr_gains(A, B, Q, R, T):
    Ps = [Q]
    Ks = []
    for t in range(T):
        P = Ps[-1]
        Ps.append(Q + A.T.dot(P).dot(A) - A.T.dot(P).dot(B)
            .dot(np.linalg.inv(R + B.T.dot(P).dot(B))).dot(B.T).dot(P).dot(A))
    Ps.reverse()
    for t in range(T):
        Ks.append(-np.linalg.inv(R + B.T.dot(Ps[t+1]).dot(B)).dot(B.T).dot(P).dot(A))
    return Ks, Ps


class PointBot(Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.rewards = self.done = self.time = self.state = None
        self.A = np.eye(4)
        self.A[2,3] = self.A[0,1] = 1
        self.A[1,1] = self.A[3,3] = 1 - AIR_RESIST
        self.B = np.array([[0,0], [1,0], [0,0], [0,1]])
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(2) * MAX_FORCE, np.ones(2) * MAX_FORCE)
        self.observation_space = Box(-np.ones(4) * np.float('inf'), np.ones(4) * np.float('inf'))
        self.start_state = START_STATE
        self.mode = MODE
        self.feature = [0, 0, 0] #[red region, white space, garabage collected]
        self.bonus = TRASH_BONUS
        self.obstacle = OBSTACLE[MODE]
        self.grid = [math.inf, -math.inf, math.inf, -math.inf]
        for i in range(len(self.obstacle.obs)):
            xbound = self.obstacle.obs[i].boundsx
            ybound = self.obstacle.obs[i].boundsy
            self.grid = [min(self.grid[0], xbound[0]), max(self.grid[1], xbound[1]), min(self.grid[2], ybound[0]), max(self.grid[3], ybound[1])]
        if self.mode == 1:
            self.start_state = [-100, 0, 0, 0]
        if TRASH:
            self.observation_space = Box(-np.ones(6) * np.float('inf'), np.ones(6) * np.float('inf'))
            self.trash_locs = TRASH_LOCS
            proper = True
            for i in range(NUM_TRASH_LOCS):
                if i >= len(TRASH_LOCS):
                    proper = False
                    point = 0
                    while(not proper):
                        proper = True
                        x = random.uniform(self.grid[0] + TRASH_BUFFER, self.grid[1] - TRASH_BUFFER)
                        y = random.uniform(self.grid[2] + TRASH_BUFFER, self.grid[3] - TRASH_BUFFER)
                        point = tuple((x, y))
                        for i in range(len(self.obstacle.obs)):
                            if self.obstacle.obs[i].in_obs(point, TRASH_BUFFER):
                                proper = False
                    self.trash_locs.append(point)
            self.remaining_trash_locs = self.trash_locs[:]
            self.remaining_trash = [False] * len(self.trash_locs)
            self.start_state = START_STATE + self.closest_trash(START_STATE)
    
    def closest_trash(self, state):
        closest_dist = math.inf
        curr = END_POS # if there are no more trash locations then heading to END_POS is used
        for i in range(len(self.remaining_trash_locs)):
            point = self.remaining_trash_locs[i]
            close_dist = math.sqrt(math.pow(point[0] - state[0], 2) + math.pow(point[1] - state[2], 2) * 1.0)
            if close_dist < closest_dist:
                closest_dist = close_dist
                curr = point
                self.remaining_trash = [False] * len(self.remaining_trash)
                self.remaining_trash[i] = True
        return [curr[0] - state[0], curr[1] - state[2]]

    def step(self, a):
        a = process_action(a)
        trash_bonus = self.determine_trash_bonus(self.state)
        self.augment_feature(self.state)
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a) # distance to the goal
        self.rewards.append(-cur_cost + trash_bonus)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = HORIZON <= self.time                        # where is collision cost uncertain
        return self.state, -cur_cost+trash_bonus, self.done, {}     # add information, boolean, obstacle = true or false whether collision or not, key to be in collsion

    def determine_trash_bonus(self, state):
        if TRASH and math.sqrt(math.pow(state[4], 2) + math.pow(state[5], 2) * 1.0) < TRASH_RADIUS:
            idx_true = [i for i, val in enumerate(self.remaining_trash) if val]
            if len(idx_true) > 0:
                idx_true = idx_true[0]
                self.remaining_trash_locs.remove(self.remaining_trash_locs[idx_true])
                self.feature[2] += 1
                self.remaining_trash.remove(True)
            return self.bonus
        return 0
    def augment_feature(self, state):
        point = tuple((state[0], state[2]))
        obs = False
        for i in range(len(self.obstacle.obs)):
            if self.obstacle.obs[i].in_obs(point, 0): #point within obstacle region
                self.feature[0] += 1
                obs = True
        if not obs:
            self.feature[1] += 1

    def reset(self):
        if TRASH:
            self.state = self.start_state + np.random.randn(6)
        else:
            self.state = self.start_state + np.random.randn(4)
        self.time = 0       #expectiation better to go through obstacle small number (2), worst case better around (50)
        self.rewards = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _next_state(self, s, a):
        if TRASH:
            s = s[:4]
        _next = self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(len(s))
        if TRASH:
            return np.concatenate((_next, self.closest_trash(s)))
        return _next

    def step_cost(self, s, a):
        if TRASH:
            s = s[:4]
        return np.linalg.norm(np.subtract(GOAL_STATE, s)) + self.collision_cost(s)

    def collision_cost(self, obs):
        return COLLISION_COST * self.obstacle(obs)    #put this instide the dicitonary, in collision


    def plot_trajectory(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        plt.scatter(states[:,0], states[:,2])
        plt.show()

    def plot_entire_trajectory(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        fig, ax = plt.subplots()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        #add more colors for more than 10 trajs
        for i in range(len(states)):
            ax.scatter(states[i][:,0], states[i][:,2], [10]*len(states[i][:,0]), colors[i])
        num_obst = len(self.obstacle.obs)
        for i in range(num_obst):
            xbound = self.obstacle.obs[i].boundsx
            ybound = self.obstacle.obs[i].boundsy
            rect = patches.Rectangle((xbound[0],ybound[0]),abs(xbound[1] - xbound[0]),abs(ybound[1] - ybound[0]),linewidth=1, edgecolor='r',facecolor='red', fill = True)
            ax.add_patch(rect)
        ax.set_xlim([-75, 25])
        ax.set_ylim([-50, 50])
        
    def is_stable(self, s):
        if TRASH:
            s = s[:4]
        return np.linalg.norm(np.subtract(GOAL_STATE, s)) <= GOAL_THRESH

    def teacher(self):
        return PointBotTeacher(self.mode)

class PointBotTeacher(object):

    def __init__(self, mode):
        self.mode = mode
        self.env = PointBot()
        self.env.set_mode(mode)
        self.Ks, self.Ps = lqr_gains(self.env.A, self.env.B, np.eye(4), 50 * np.eye(2), HORIZON)

    def get_rollout(self):
        obs = self.env.reset()
        O, A, rewards = [obs], [], []
        noise_std = 0.2
        for i in range(HORIZON):
            if self.mode == 1:
                noise_idx = np.random.randint(int(HORIZON * 2 / 3))
                if i < HORIZON / 2:
                    action = [0.1, 0.1]
                else:
                    action = self._expert_control(obs, i)
            else:
                noise_idx = np.random.randint(int(HORIZON))
                if i < HORIZON / 4:
                    action = [0.1, 0.25]
                elif i < HORIZON / 2:
                    action = [0.4, 0.]
                elif i < HORIZON / 3 * 2:
                    action = [0, -0.5]
                else:
                    action = self._expert_control(obs, i)

            if i < noise_idx:
                action = (np.array(action) +  np.random.normal(0, noise_std, self.env.action_space.shape[0])).tolist()

            A.append(action)
            obs, reward, done, info = self.env.step(action)
            O.append(obs)
            rewards.append(reward)
            if done:
                break
        rewards = np.array(rewards)

        if self.env.is_stable(obs):
            stabilizable_obs = O
            print("REWARDS: ", rewards)
        else:
            stabilizable_obs = []
            return self.get_rollout()

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "rewards": rewards,
        }

    def _get_gain(self, t):
        return self.Ks[t]

    def _expert_control(self, s, t):
        return self._get_gain(t).dot(s)

if __name__=="__main__":
    env = PointBot()
    obs = env.reset()
    teacher = env.teacher()
    for _ in range(10):
        teacher.get_rollout()
        print("DONE DEMOS")
