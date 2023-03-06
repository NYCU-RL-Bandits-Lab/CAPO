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

import torch as th
import random
class Obj(Enum):
    EMPTY = 0
    AGENT = 1
    WALL = 2
    GOAL = 3


def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

    
class Pos:
    def __init__(self, h=0, w=0):
        assert isinstance(h, int) and isinstance(w, int)
        self.h=h
        self.w=w
    
    def __eq__(self, other):
        if (isinstance(other, Pos)):
            return self.w == other.w and self.h == other.h
    
    def __str__(self):
        return f"{self.h}, {self.w}"


class GridEnv:
    def __init__(self, size=(5,5), use_sparse_reward=True, gamma=0.99, windy=0.0, goal_reward=100):
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
        # 
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
        
        
        if action == 'left':
            target_pos = Pos(self.agent_pos.h, self.agent_pos.w-1)
        elif action == 'right':
            target_pos = Pos(self.agent_pos.h, self.agent_pos.w+1)
        elif action == 'up':
            target_pos = Pos(self.agent_pos.h-1, self.agent_pos.w)
        elif action == 'down':
            target_pos = Pos(self.agent_pos.h+1, self.agent_pos.w)
        
        if self.isValid(target_pos):
            self.move_agent(target_pos)
        
        if self.agent_pos == self.goal_pos:
            reward = self.GOAL_REWARD
            done = True

        # return self.grid, reward, self.agent_pos, done
        return self.grid, reward, self.agent_pos, done
    

    def _init_agent_pos(self):
        while True:
            start_h = np.random.randint(1, self.H)
            start_w = np.random.randint(1, self.W)
            pos = Pos(start_h, start_w)
            # if pos is not WALL and not GOAL
            if self.isValid(pos) and not self.ObjatPos(Obj.GOAL, pos):
                self.agent_pos = pos
                self.grid[pos.h][pos.w] = Obj.AGENT.value
                break

        return self.agent_pos 

        self.agent_pos = None
        while self.agent_pos is None:
            h = np.random.randint(low=1,high=self.H-1)
            w = np.random.randint(low=1,high=self.W-1)
            pos = Pos(h, w)
            if not self.ObjatPos(Obj.WALL, pos) and not self.ObjatPos(Obj.GOAL, pos):
                self.agent_pos = pos

        
    def reset(self):
        
        self.grid = np.zeros((self.H, self.W), dtype=np.int8)
        self.goal_pos = Pos(self.H-1-1, self.W-1-1)
        if self.backup_grid is None:
            self.backup_grid = self._init_grid(self.agent_pos, self.goal_pos)
        self.grid = np.copy(self.backup_grid)
        
        if self.distance_grid is None and not self.use_sparse_reward:
            self._BFS()

        self._init_agent_pos()
        return self.grid, self.agent_pos

    def isValid(self, target_pos):
        if self.grid[target_pos.h][target_pos.w] == Obj.WALL.value:
            return False
        
        return True
    
    def move_agent(self, target_pos):
        self.grid[self.agent_pos.h][self.agent_pos.w] = Obj.EMPTY.value
        self.grid[target_pos.h][target_pos.w] = Obj.AGENT.value
        self.agent_pos = target_pos
       
    

    def _init_grid(self, agent_pos, goal_pos):
        def __gen_wall(self):
            isVertical = np.random.random() > 0.5
            if isVertical:
                wall_index = np.random.randint(2, self.W-1-1)
                open_index = np.random.randint(1, self.H-1)
                for h in range(self.H):
                    self.grid[h][wall_index] = Obj.WALL.value
                self.grid[open_index][wall_index] = Obj.EMPTY.value
            else:
                wall_index = np.random.randint(2, self.H-1-1)
                open_index = np.random.randint(1, self.W-1)
                for w in range(self.W):
                    self.grid[wall_index][w] = Obj.WALL.value
                self.grid[wall_index][open_index] = Obj.EMPTY.value


        
        # self.grid[agent_pos.h][agent_pos.w] = Obj.AGENT.value
        for h in range(self.H):
            self.grid[h][0] = Obj.WALL.value
            self.grid[h][self.W-1] = Obj.WALL.value
        
        for w in range(self.W):
            self.grid[0][w] = Obj.WALL.value
            self.grid[self.H-1][w] = Obj.WALL.value

        self.grid[goal_pos.h][goal_pos.w] = Obj.GOAL.value
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
        if self.R is not None:
            return self.R
        nA = len(self.ACTIONS)
        R = np.zeros((self.H, self.W, nA))
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
        
        # action_prob = policy.get_action_prob()
        
        # for i in range(5):
            # for j in range(5):
                # for k in range(4):
                    # action_prob[i][j][k] = 0
                # action_prob[i][j][self.ACTION_TO_IDX['left']] = 1 # left
                # action_prob[i][j][1] = 0.5 # down
        # print(action_prob.shape)
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
                        
                        # goal state loop
                        # if self.ObjatPos(Obj.GOAL, Pos(h, w)):
                        #     h_next = h
                        #     w_next = w                        
                        if self.grid[h_next][w_next] == Obj.WALL.value:
                            # if bump into wall then no move:
                            h_next = h
                            w_next = w
                        Qsa = (R[h][w][a] + self.gamma*V0[h_next][w_next])
                        Q[h][w][a] = Qsa
                        V1[h][w] += action_prob[h][w][a] * Qsa
                        advantage[h][w][a] = Qsa - V0[h][w]
            # condition
            
            if np.sum(abs(V1 - V0)) <= EPS:
                # print(np.sum(abs(V1 - V0)))
                break
            else:
                V0 = V1 
        return V1, advantage, Q