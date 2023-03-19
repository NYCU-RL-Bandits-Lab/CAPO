import gym
import abc
import numpy as np
import gym.spaces as spaces
from . import grid
import torch
import random

class Environment(object):
    """
    The base class for all environments. This is a bit more flexible than 'gym.Env'.
    It supports a multi-component action-tuple and a global state of the environment.
    """

    def __init__(self):
        # Just one observation space and multi-component action-space tuple
        self.observation_space = None
        self.action_spaces = (None,)

    @abc.abstractmethod
    def reset(self, global_state=None):
        raise NotImplementedError('Implement this method in derived classes')

    @abc.abstractmethod
    def step(self, *actions):
        raise NotImplementedError('Implement this method in derived classes')

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def render(self, **kwargs):
        raise NotImplementedError('Implement this method in derived classes')

class Chain(Environment):
    """
    Chain, see ReadMe figure for more
     ┏╍╍╍╍╍╍╍╍╍╍╍╍╍╍┳╍╍╍╍╍╍╍╍╍╍╍╍> s0 <╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┓
     ┃              ┃              ┃             ...                ┃
     ┃              ┃              ┃             ...                ┃
    a0              a1             a2            ...                a{n-2}     <=     r=0.1                                          
     ┃              ┃              ┃             ...                ┃
     ┃              ┃              ┃             ...                ┃
    s1 --a{n-1}--> s2 --a{n-1}--> s3 --a{n-1}--> ... --a{n-1}--> s{n-1} --a{n-1}--> sn
           r=0            r=0            r=0             r=0               r=100
    
    a{t} is the action "go up", i.e. from s{t-1} to s0 
    a{n-1} is the action "go right"
    """

    def __init__(self, size=10, stochastic=True):
        super(Chain, self).__init__()
 
        self.observation_space = (spaces.Discrete(size), )
        self.n_action = size        # a{t} is the action "go up", i.e. from s{t-1} to s0 , a{n-1} is the action "go right"
        self.action_spaces = (spaces.Discrete(self.n_action), )
        self.size = size
        self.set_state(1)           # the agent start at state 1, see ReadMe figure
        self.reset()
        self.cyclic_state = 1
        self.cyclic_action = 0
        self.r_small = 0.1          # the reward to s0
        self.r_big = 100            # the reward to sn
        self.n_state = size+1       # s0 ~ sn
        self.stochastic = stochastic        # for stochastic chain

    def set_state(self, state):
        self._state = np.array(state)

    def _get_obs(self):
        return np.array([self._state])
    
    def _get_info(self):
        return {}
    
    def reset_cyclic(self):
        self.cyclic_state = 1
        self.cyclic_action = 0

    def reset(self, global_state=None):
        self.set_state(1)
        self.done = False
        observation = self._get_obs()
        return observation

    # TODO
    def get_cyclic_next_state_action(self):
        state, action = self.cyclic_state, self.cyclic_action
        if self.cyclic_action == (self.n_action-1):
            self.cyclic_action = 0
            self.cyclic_state += 1
        else:
            self.cyclic_action += 1 
        passed = False
        if self.cyclic_state >= self.size:
            self.cyclic_state = 1
            self.cyclic_action = 0
            passed=True
        self.set_state(state)
        return state, action, passed
    
    def step(self, *actions):
        
        # debug
        assert len(actions) == 1, 'CartPole-v0 has only one action component'
        action = actions[0]
        assert self.done == False
        assert self._state != 0
        
        # choose up (little reward)
        if action != (self.n_action-1):
            self.set_state(0)
            reward = self.r_small
            self.done = True            
        # choose right (0 reward except terminal)
        else:
            reward = 0
            if not self.stochastic:
                self._state += 1
            else:
                # stochastic chain, the next state might jump from s{t} to s{t}, s{t+1}, s{t+2}, s{t+3}
                self._state += random.choice([0, 1, 2, 3])
                self._state = min([self._state, self.size])
            
            # termination condition
            if self.size == self._state:
                reward = self.r_big
                self.done = True

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, self.done, info
    
class Grid(Environment):
    def __init__(self, size=10, use_sparse_reward=False):
        super(Grid, self).__init__()
        self.size = size+2
        self.n_state = self.size * self.size
        self.observation_space = (spaces.Discrete(self.size*self.size), )
        self.n_action = 4
        self.action_spaces = (spaces.Discrete(self.n_action), )
        
        self.env = grid.GridEnv(size=(self.size, self.size), use_sparse_reward=use_sparse_reward)
        self.reset()
        
    def _get_obs(self):
        return np.array([self.pos_to_obs(self.env.agent_pos)])
    
    def _get_info(self):
        return {}
    
    def reset_cyclic(self):
        self.cyclic_state = 1
        self.cyclic_action = 0

    def reset(self, global_state=None):
        _, pos = self.env.reset()
        obs = self.pos_to_obs(pos)
        return self._get_obs()  #TODO ness?

    # TODO
    def get_cyclic_next_state_action(self):
        assert "error"
        # state, action = self.cyclic_state, self.cyclic_action
        # if self.cyclic_action == (self.n_action-1):
        #     self.cyclic_action = 0
        #     self.cyclic_state += 1
        # else:
        #     self.cyclic_action += 1 
        # passed = False
        # if self.cyclic_state >= self.size:
        #     self.cyclic_state = 1
        #     self.cyclic_action = 0
        #     passed=True
        # self.set_state(state)
        # return state, action, passed
    

    def step(self, *actions):
        assert len(actions) == 1, 'CartPole-v0 has only one action component'
        action = actions[0][0][0]
        grid, reward, agent_pos, done = self.env.step(int(action))
        info = self._get_info()     #TODO ness?
        return self._get_obs(), reward, done, info

    def pos_to_obs(self, pos):
        """
        return the 'flatten' position of a grid world matrix
        """
        return pos.h*self.size + pos.w

    def state_to_pos(self, state):
        state = int(state)
        pos = grid.Pos(h=state//self.size, w=state%self.size)
        assert state == self.pos_to_obs(pos)
        return pos

    def translate_prob(self, probs):
        action_probs = torch.zeros((self.size, self.size, self.n_action), dtype=torch.float)
        assert probs.size() == (self.n_state, self.n_action), print(probs.size())
        for s in range(probs.size(0)):
            pos = self.state_to_pos(s)
            h, w = pos.h, pos.w 
            for a in range(probs.size(1)):
                action_probs[h][w][a] = probs[s][a] 

        return action_probs

    def compute_value(self, probs):
        """
        Input:
            probs: |S| x |A| matrix, which is the action probability
        
        After translate_prob:
            action_probs: |height of grid| x |width of grid| x |A|, which is the translated probability
        """

        action_probs = self.translate_prob(probs)
        # self.env.render() TODO better place
        _V, _ADV, _Q = self.env.compute_value(action_probs)
        
        # just input the computed Q, V, ADV into the well-form matrix
        Q = torch.zeros((self.n_state, self.n_action), dtype=torch.float)
        V = torch.zeros((self.n_state), dtype=torch.float)
        for s in range(self.n_state):
            pos = self.state_to_pos(s)
            V[s] = _V[pos.h][pos.w]
            for a in range(self.n_action):
                Q[s][a] = _Q[pos.h][pos.w][a]

        return Q, V
    
    # TODO
    # def render?

class CartPolev0(Environment):
    
    def __init__(self):
        # The original 'CartPole-v0' environment from gym
        self._gymenv = gym.make('CartPole-v0')

        self.observation_space = self._gymenv.observation_space
        self.action_spaces = (self._gymenv.action_space,)

    def reset(self, global_state=None):
        state, _ = self._gymenv.reset()
        return state

    def step(self, *actions):
        assert len(actions) == 1, 'CartPole-v0 has only one action component'
        next_state, reward, done, info, _ = self._gymenv.step(*tuple(a.item() for a in actions))
        return next_state, reward, done, info

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        self._gymenv.render(**kwargs)


class LunarLanderv2(Environment):
    
    def __init__(self):
        # The original 'CartPole-v0' environment from gym
        self._gymenv = gym.make('LunarLander-v2')

        self.observation_space = self._gymenv.observation_space
        self.action_spaces = (self._gymenv.action_space,)

    def reset(self, global_state=None):
        state, _ = self._gymenv.reset() 
        return state

    def step(self, *actions):
        assert len(actions) == 1, 'len(action)==1?'
        return self._gymenv.step(*tuple(a.item() for a in actions))[:-1]

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        self._gymenv.render(**kwargs)

class IncompleteCartPolev0(Environment):
    
    def __init__(self):
        # The original 'CartPole-v0' environment from gym
        self._gymenv = gym.make('CartPole-v0')

        low, high = self._gymenv.observation_space.low, self._gymenv.observation_space.high
        self.observation_space = gym.spaces.Box(low[:-1], high[:-1], dtype=np.float32)
        self.action_spaces = (self._gymenv.action_space,)

    def reset(self, global_state=None):
        return self._gymenv.reset()[:-1]

    def step(self, *actions):
        assert len(actions) == 1, 'CartPole-v0 has only one action component'
        next_state, rew, done, H = self._gymenv.step(*tuple(a.item() for a in actions))
        return next_state[:-1], rew, done, H

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        self._gymenv.render(**kwargs)

class CartPolev1(Environment):
    
    def __init__(self):
        # The original 'CartPole-v1' environment from gym
        self._gymenv = gym.make('CartPole-v1')

        self.observation_space = self._gymenv.observation_space
        self.action_spaces = (self._gymenv.action_space,)

    def reset(self, global_state=None):
        state, _ = self._gymenv.reset() 
        return state

    def step(self, *actions):
        assert len(actions) == 1, 'CartPole-v0 has only one action component'

        return self._gymenv.step(*tuple(a.item() for a in actions))[:-1]

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        self._gymenv.render(**kwargs)

class IncompleteCartPolev1(Environment):
    
    def __init__(self):
        # The original 'CartPole-v1' environment from gym
        self._gymenv = gym.make('CartPole-v1')

        low, high = self._gymenv.observation_space.low, self._gymenv.observation_space.high
        self.observation_space = gym.spaces.Box(low[:-1], high[:-1], dtype=np.float32)
        self.action_spaces = (self._gymenv.action_space,)

    def reset(self, global_state=None):
        return self._gymenv.reset()[:-1]

    def step(self, *actions):
        assert len(actions) == 1, 'CartPole-v0 has only one action component'
        next_state, rew, done, H = self._gymenv.step(*tuple(a.item() for a in actions))
        return next_state[:-1], rew, done, H

    def close(self):
        self._gymenv.close()

    def render(self, **kwargs):
        self._gymenv.render(**kwargs)



