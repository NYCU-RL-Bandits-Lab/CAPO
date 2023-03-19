import abc
import torch, gym
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Categorical
import traceback

class ActionDistribution(object):
    """ Encapsulates a multi-part action distribution """

    def __init__(self, *distribs: dist.Distribution):
        '''
        Constructs an object from a variable number of 'dist.Distribution's
        Arguments:
            *distribs: A set of 'dist.Distribution' instances
        '''
        super().__init__()

        self.distribs = distribs
        self.n_dist = len(self.distribs)

    def sample(self):
        ''' Sample an action (set of several constituent actions) '''
        return tuple(d.sample().unsqueeze(0) for d in self.distribs)

    def log_prob(self, *samples):
        '''
        Computes the log-probability of the action-tuple under this ActionDistribution.
        Note: Assumed to be independent, i.e., probs factorize.
        Arguments:
            samples: A tuple of actions
        '''
        assert len(samples) == self.n_dist, "Number of constituent distributions is different than number of samples"
        logprobs = []
        for d, s in zip(self.distribs, samples):
            # TODO: Hopefully okay. Need recheck
            assert len(s.shape) >= 2, 'samples must have atleast one event dim and one batch dim'
            lp = d.log_prob(s.squeeze(-1))
            if len(lp.shape) == 1:
                lp = lp.unsqueeze(-1)
            
            if len(d.event_shape) != 0:
                lp = torch.sum(lp, dim=-1, keepdim=True)
            logprobs.append(lp)
        return sum(logprobs)

    def entropy(self):
        ''' Computes entropy of (each component) the ActionDistribution '''
        return sum([d.entropy().unsqueeze(0) for d in self.distribs])

class Parametric(nn.Module):
    """
    Base class of the learnable component of an agent. It should contain Policy, Value etc.

    Required API:
        forward(states) -> ActionDistribution, others (Given states, returns ActionDistribution and other stuff)
        reset() -> None (Resets the internals of the learnable. Specifically required for RNNs)
    """

    def __init__(self, observation_space, action_spaces):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__()

        assert isinstance(action_spaces, tuple), "There must be a list (potentially singleton) of action spaces"

        self.observation_space = observation_space
        self.action_spaces = action_spaces

    @abc.abstractmethod
    def forward(self, *states):
        pass

class TabularPolicyValue(Parametric):
    """ Feed forward (policy + value) for discrete action space """

    def __init__(self, env, observation_space, action_spaces, *, n_hidden=128, is_behavior=False, is_capo=False):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__(observation_space, action_spaces)
        assert len(self.action_spaces) == 1, "Only one action component"
        assert isinstance(self.action_spaces[0], gym.spaces.Discrete), "Only discrete action allowed"
        self.is_behavior = is_behavior
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env

        # Track arguments for further use
        if type(self.observation_space) == gym.spaces.box.Box:
             self.n_state = self.observation_space.shape[0]
        else:
            self.n_state = len(self.observation_space)
        self.n_action = self.action_spaces[0].n
        self.n_hidden = n_hidden
        self.is_capo = is_capo
        

        # Layer definitions
        self.affine = torch.nn.Linear(self.n_state, self.n_hidden)
        self.hidden = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.thetas = torch.nn.Linear(self.n_hidden, self.n_action)
        self.value = torch.nn.Linear(self.n_hidden, 1)
        self.tabular_thetas = torch.nn.Parameter(torch.ones((self.env.n_state, self.n_action)), requires_grad=True)
        torch.nn.init.zeros_(self.affine.weight)
        torch.nn.init.zeros_(self.thetas.weight)
        torch.nn.init.zeros_(self.value.weight)

    def forward(self, *states):
        
        # encode state, approx value
        _, state = states
        h = F.relu(self.affine(state))      # first layer
        h = F.relu(self.hidden(h))          # second layer
        v = self.value(h)
        
        # look up theta from table
        state = state.long()
        thetas = self.tabular_thetas[state].squeeze(1)

        if self.is_behavior:
            # uniform behavior
            act = Categorical(F.softmax(torch.tensor([[1.0/self.n_action for i in range(self.n_action)]]).to(self.device), dim=-1))
        else:
            try:
                act = Categorical(F.softmax(thetas, dim=-1))
            except ValueError:
                traceback.print_exc()
                exit(-1)

        # return theta in capo
        if self.is_capo:
            return None, ActionDistribution(act), v, thetas
        else:
            return None, ActionDistribution(act), v


class DiscreteMLPPolicyValue(Parametric):
    """ Feed forward (policy + value) for discrete action space """

    def __init__(self, observation_space, action_spaces, *, n_hidden=256, is_behavior=False):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__(observation_space, action_spaces)
        assert len(self.action_spaces) == 1, "Only one action component"
        assert isinstance(self.action_spaces[0], gym.spaces.Discrete), "Only discrete action allowed"
        self.is_behavior = is_behavior
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Track arguments for further use
        if type(self.observation_space) == gym.spaces.box.Box:
             self.n_state = self.observation_space.shape[0]
        else:
            self.n_state = len(self.observation_space)
        
        self.n_action = self.action_spaces[0].n
        self.n_hidden = n_hidden
        self.n_hidden = n_hidden

        # Layer definitions
        self.affine = torch.nn.Linear(self.n_state, self.n_hidden)
        self.pi = torch.nn.Linear(self.n_hidden, self.n_action)
        self.value = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, *states):
        _, state = states
        h = F.relu(self.affine(state))
        
        if self.is_behavior:
            act = Categorical(F.softmax(torch.tensor([[1/self.n_action for i in range(self.n_action)]]).to(self.device), dim=-1))
        else:
            act = Categorical(F.softmax(self.pi(h), dim=-1))
        
        v = self.value(h)
        return None, ActionDistribution(act), v


class DiscreteRNNPolicyValue(Parametric):
    """ Recurrent (policy + value) for discrete action space """

    def __init__(self, observation_space, action_spaces, *, n_hidden=128):
        ''' Constructs every learnable item from environment specifications. '''

        super().__init__(observation_space, action_spaces)
        assert len(self.action_spaces) == 1, "Only one action component"
        assert isinstance(self.action_spaces[0], gym.spaces.Discrete), "Only discrete action allowed"

        # Track arguments for further use
        if type(self.observation_space) == gym.spaces.box.Box:
             self.n_state = self.observation_space.shape[0]
        else:
            self.n_state = len(self.observation_space)
        self.n_action = self.action_spaces[0].n
        self.n_hidden = n_hidden

        # Layer definitions
        self.cell, self.h = torch.nn.GRUCell(self.n_state, self.n_hidden), None
        self.pi = torch.nn.Linear(self.n_hidden, self.n_action)
        self.V = torch.nn.Linear(self.n_hidden, 1)
    
    def forward(self, *states):
        recur_state, state = states
        recur_state = self.cell(state, recur_state)
        act = Categorical(F.softmax(self.pi(recur_state), dim=-1))
        v = self.V(recur_state)
        return recur_state, ActionDistribution(act), v
