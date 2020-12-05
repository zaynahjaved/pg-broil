"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise and air resistance
proportional to its velocity. State representation is (x, vx, y, vy). Action
representation is (fx, fy), and mass is assumed to be 1.
"""

import os
import pickle

import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        self.obstacle = OBSTACLE[MODE]
        if self.mode == 1:
            self.start_state = [-100, 0, 0, 0]

    def step(self, a):
        a = process_action(a)
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a) # distance to the goal
        self.rewards.append(-cur_cost)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = HORIZON <= self.time                        # where is collision cost uncertain
        return self.state, -cur_cost, self.done, {}     # add information, boolean, obstacle = true or false whether collision or not, key to be in collsion


    def reset(self):
        self.state = self.start_state + np.random.randn(4)
        self.time = 0       #expectiation better to go through obstacle small number (2), worst case better around (50)
        self.rewards = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _next_state(self, s, a):
        return self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(len(s))

    def step_cost(self, s, a):
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
