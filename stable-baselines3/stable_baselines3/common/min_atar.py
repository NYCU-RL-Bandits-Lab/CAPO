import gym
from gym import spaces
from importlib import import_module
import numpy as np
class MinAtarEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, env_id, seed):
    super(MinAtarEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    N_DISCRETE_ACTIONS = 6 

    env_module = import_module('stable_baselines3.minatar.environments.'+env_id)
    self.random = np.random.RandomState(seed)
    self.env = env_module.Env(ramping = True, random_state = self.random)
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # self.observation_space = spaces.Discrete(10 * 10 * len(self.env.channels))
    self.observation_space = spaces.Box(low=0, high=1,
                                        shape=(len(self.env.channels), 10, 10), dtype=np.bool)
    


  def step(self, action):
    reward, done = self.env.act(action)
    observation = self.env.state()
    info = {}
    return np.moveaxis(observation, -1, 0), reward, done, info

  def reset(self):
    self.env.reset()
    observation = self.env.state()
    return np.moveaxis(observation, -1, 0)  # reward, done, info can't be included

  def render(self, mode='human'):
    return
  def close (self):
    return
  
  def get_action_meanings(self):
    return self.env.action_map


def make_min_atar_env(env_id, seed):
    return MinAtarEnv(env_id, seed)