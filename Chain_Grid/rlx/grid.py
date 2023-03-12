"""
GridWorld by Hsin-En Su

TODO:
    utilize function compute_value, _BFS
    
"""

import numpy as np
import os
import time
import sys
import gym
from gym import error, spaces, utils
import copy
from gym.utils import seeding
import matplotlib.pyplot as plt
from enum import Enum
from numpy.lib import math

import torch
import random

class Obj(Enum):
    EMPTY = 0
    AGENT = 1
    WALL = 2
    GOAL = 3

    
class Pos:
    def __init__(self, h=0, w=0):
        assert isinstance(h, int) and isinstance(w, int)
        self.h = h
        self.w = w
    
    def __eq__(self, other):
        if (isinstance(other, Pos)):
            return self.w == other.w and self.h == other.h
    
    def __str__(self):
        return f"{self.h}, {self.w}"


class GridEnv:
    def __init__(self, size=(5,5), use_sparse_reward=False, gamma=0.99, windy=0.0, goal_reward=100):
        
        self.ACTIONS = {0 : 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.ACTION_TO_IDX = {'up' : 0, 'down' : 1, 'left': 2, 'right' : 3}
        self.nA = len(self.ACTIONS)
        self.H, self.W = size
        self.backup_grid = None
        self.grid = None
        self.distance_grid = None # init by running self._BFS
        self.GOAL_REWARD = goal_reward
        self.use_sparse_reward = use_sparse_reward
        self.gamma = gamma
        self._agent_pos = None # maintained by proport agent_pos
        self.R = None # store R to avoid repeatedly calling self.calc_reward()
        self.MAX_TRAJ_LENGTH = 1000
        self.windy = windy
        self.reset() # run last
        
    @property
    def agent_pos(self):
        return self._agent_pos
    
    @agent_pos.setter
    def agent_pos(self, var):
        self._agent_pos = var

    def step(self, action):
        '''
        return Grid, reward, agent_pos after moving, done
        '''
        done = False

        # closer pos has higher reward
        if self.use_sparse_reward:
            reward = -1
        else:
            reward = -1 * self.distance_grid[self.agent_pos.h][self.agent_pos.w]

        if isinstance(action, int):
            action = self.ACTIONS[action]
        
        # moving agent
        if action == 'left':
            target_pos = Pos(self.agent_pos.h, self.agent_pos.w-1)
        elif action == 'right':
            target_pos = Pos(self.agent_pos.h, self.agent_pos.w+1)
        elif action == 'up':
            target_pos = Pos(self.agent_pos.h-1, self.agent_pos.w)
        elif action == 'down':
            target_pos = Pos(self.agent_pos.h+1, self.agent_pos.w)
        
        # judging valid target position
        if self.isValid(target_pos):
            self.move_agent(target_pos)
        
        # terminated condition
        if self.agent_pos == self.goal_pos:
            reward = self.GOAL_REWARD
            done = True

        # return self.grid, reward, self.agent_pos, done
        return self.grid, reward, self.agent_pos, done
    

    def _init_agent_pos(self):
        
        while True:
            # random select the starting position
            start_h = np.random.randint(1, self.H)
            start_w = np.random.randint(1, self.W)
            pos = Pos(start_h, start_w)
            
            # if pos is not WALL and not GOAL
            if self.isValid(pos) and not self.ObjatPos(Obj.GOAL, pos):
                self.agent_pos = pos
                self.grid[pos.h][pos.w] = Obj.AGENT.value
                break

        return self.agent_pos 

    def reset(self):
        
        # initialize grid world
        self.grid = np.zeros((self.H, self.W), dtype=np.int8)

        # goal at right bottom corner
        self.goal_pos = Pos(self.H-1-1, self.W-1-1)
        
        # generate the grid (including wall, goal) and store as backup
        if self.backup_grid is None:
            self.backup_grid = self._init_grid(self.goal_pos)
        self.grid = np.copy(self.backup_grid)
        
        # setting different for non-sparse reward case
        if self.distance_grid is None and not self.use_sparse_reward:
            self._BFS()

        # initialize agent position
        self._init_agent_pos()

        return self.grid, self.agent_pos

    def isValid(self, target_pos):
        
        if self.grid[target_pos.h][target_pos.w] == Obj.WALL.value:
            return False
        
        return True
    
    def move_agent(self, target_pos):
        
        # modified the pass position & new position
        self.grid[self.agent_pos.h][self.agent_pos.w] = Obj.EMPTY.value
        self.grid[target_pos.h][target_pos.w] = Obj.AGENT.value
        self.agent_pos = target_pos
       
    def _init_grid(self, goal_pos):
        
        def __gen_wall(self):
            """
            random generate a vertical / horizontal wall with only one place open
            e.g.
            W W W W W W
            W _ _ _ _ W
            W _ W W W W
            W _ _ _ _ W
            W _ _ _ _ W
            W W W W W W
            """
            # vertical wall
            if np.random.random() > 0.5:
                wall_index = np.random.randint(2, self.W-1-1)
                open_index = np.random.randint(1, self.H-1)
                for h in range(self.H):
                    self.grid[h][wall_index] = Obj.WALL.value
                self.grid[open_index][wall_index] = Obj.EMPTY.value
            # horizontal wall
            else:
                wall_index = np.random.randint(2, self.H-1-1)
                open_index = np.random.randint(1, self.W-1)
                for w in range(self.W):
                    self.grid[wall_index][w] = Obj.WALL.value
                self.grid[wall_index][open_index] = Obj.EMPTY.value

        # set boundary as wall
        for h in range(self.H):
            self.grid[h][0] = Obj.WALL.value
            self.grid[h][self.W-1] = Obj.WALL.value
        
        # set boundary as wall
        for w in range(self.W):
            self.grid[0][w] = Obj.WALL.value
            self.grid[self.H-1][w] = Obj.WALL.value

        # set goal
        self.grid[goal_pos.h][goal_pos.w] = Obj.GOAL.value
        
        # set wall inside the gridworld
        __gen_wall(self)

        return self.grid
        
    def get_state(self):
        return self.grid

    def render(self):
        for h in range(self.H):
            for w in range(self.W):
                if self.grid[h][w] == Obj.WALL.value:
                    print('W', end=' ')
                elif self.grid[h][w] == Obj.GOAL.value:
                    print('G', end=' ')
                elif self.grid[h][w] == Obj.AGENT.value:
                    print('A', end=' ')
                elif self.grid[h][w] == Obj.EMPTY.value:
                    print('_', end=' ')
            print()

    def calc_reward(self):
        
        # store the reward for reusablility
        if self.R is not None:
            return self.R
        nA = len(self.ACTIONS)
        R = np.zeros((self.H, self.W, nA))
        
        # calculate the reward under all (s,a)  =>  the reward dimension will be (H*W*A)
        for h in range(1, self.H-1):
            for w in range(1, self.W-1):
                for a in range(nA):
                    
                    # goal state loop back to goal state with reward 0
                    if Pos(h, w) == self.goal_pos:
                        R[h][w][a] = 0
                        continue

                    if self.ACTIONS[a] == 'left':
                        h_next = h
                        w_next = w-1
                    elif self.ACTIONS[a] == 'right':
                        h_next = h
                        w_next = w+1
                    elif self.ACTIONS[a] == 'up':
                        h_next = h-1
                        w_next = w
                    elif self.ACTIONS[a] == 'down':
                        h_next = h+1
                        w_next = w
                    
                    if self.ObjatPos(Obj.WALL, Pos(h_next, w_next)):
                        h_next = h
                        w_next = w

                    if Pos(h_next, w_next) == self.goal_pos:
                        R[h][w][a] = self.GOAL_REWARD
                    else:
                        if self.use_sparse_reward:
                            R[h][w][a] = -1
                        else:
                            R[h][w][a] = -1 * self.distance_grid[h_next][w_next]
        self.R = R
        return R

    def ObjatPos(self, object, pos):
        return self.grid[pos.h][pos.w] == object.value

    def _BFS(self):
        
        """
        breadth first search
        e.g.
        W W W W W W                   10000 10000 10000 10000 10000 10000
        W _ _ _ _ W                   10000   6     7     8     9   10000
        W _ W W W W         ==>       10000   5   10000 10000 10000 10000
        W _ _ _ _ W         ==>       10000   4     3     2     1   10000
        W _ _ _ G W                   10000   3     2     1     0   10000
        W W W W W W                   10000 10000 10000 10000 10000 10000
        """
        
        def _bfs(pos):
            nonlocal self
            self.visited_grid[pos.h][pos.w] = 1
            distance = 10000
            if pos == self.goal_pos:
                distance = 0
            else:
                steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for step in steps:
                    next_pos = Pos(pos.h + step[0], pos.w + step[1])
                    if not self.ObjatPos(Obj.WALL, next_pos) and not self.visited_grid[next_pos.h][next_pos.w]:
                        _bfs(next_pos)
                    distance = min(distance, self.distance_grid[next_pos.h][next_pos.w] + 1) 
            self.distance_grid[pos.h][pos.w] = distance
            
        self.distance_grid = np.zeros_like(self.grid) + 10000
        self.visited_grid = np.zeros_like(self.grid)

        pos = Pos(1, 1)
        _bfs(pos)
        while True:
            hasChanged = False
            for h in range(1, self.H-1):
                for w in range(1, self.W-1):
                    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    for step in steps:
                        if not self.ObjatPos(Obj.WALL, Pos(h, w)):
                            if self.distance_grid[h+step[0]][w+step[1]] + 1 < self.distance_grid[h][w]:
                                self.distance_grid[h][w] = self.distance_grid[h+step[0]][w+step[1]] + 1
                                hasChanged = True
            if not hasChanged:
                break

        return self.distance_grid
    
    def compute_value(self, action_prob, R=None, EPS=1e-9):
        '''
        compute V^pi(s) for all s
        '''
        H, W = self.H, self.W
        V0 = np.zeros((H, W))
        if R is None:
            R = self.calc_reward()
        
        while True:
            V1 = np.zeros((H, W))
            advantage = np.zeros((H, W, self.nA))
            Q = np.zeros((H, W, self.nA))
            for h in range(1, H-1):
                for w in range(1, W-1):
                    for a in range(self.nA):
                        # don't perform action at goal and wall 
                        if Pos(h, w) == self.goal_pos or self.ObjatPos(Obj.WALL, Pos(h, w)):
                        # if self.ObjatPos(Obj.WALL, Pos(h, w)):
                            continue

                        h_next = h
                        w_next = w
                        if self.ACTIONS[a] == 'left':
                            h_next = h
                            w_next = w-1
                        elif self.ACTIONS[a] == 'right':
                            h_next = h
                            w_next = w+1
                        elif self.ACTIONS[a] == 'up':
                            h_next = h-1
                            w_next = w
                        elif self.ACTIONS[a] == 'down':
                            h_next = h+1
                            w_next = w
                                           
                        if self.grid[h_next][w_next] == Obj.WALL.value:
                            # if bump into wall then no move:
                            h_next = h
                            w_next = w
                        Qsa = (R[h][w][a] + self.gamma*V0[h_next][w_next])
                        Q[h][w][a] = Qsa
                        V1[h][w] += action_prob[h][w][a] * Qsa
                        advantage[h][w][a] = Qsa - V0[h][w]
            
            if np.sum(abs(V1 - V0)) <= EPS:
                break
            else:
                V0 = V1 

        return V1, advantage, Q