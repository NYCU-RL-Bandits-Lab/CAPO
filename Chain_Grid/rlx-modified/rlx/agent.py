import abc, time
import torch
import numpy as np

from .rollout import Rollout
from .env import Environment, Chain, Grid
from .policy import Parametric

class PGAgent(object):
    """ Encapsulation of an Agent """

    def __init__(self, env, device=None):
        '''
        Constructs an Agent for the given environment. This agent is almost state-less.
        Arguments:
            env: An environment respecting the 'env.Environment' API
            device: Default device of operation
        '''
        super().__init__()
        assert isinstance(env, Environment), "env object must be an instance of 'env.Environment'"

        # Track arguments
        self.env = self.environment = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    def __call__(self, network):
        ''' Augments an agent with a network. The agent cannot work without this. '''
        new_agent = PGAgent(self.environment, device=self.device)
        setattr(new_agent, 'network', network)
        return new_agent

    def cuda(self):
        self.device = torch.device('cuda')
        return self

    def reset(self, global_state=None):
        ''' Resets the environment and returns an initial state '''
        return torch.from_numpy(self.environment.reset(global_state=global_state)).float().to(self.device)

    def timestep(self, recurr_state, full_state):
        ''' Given a state-tuple, returns the action distribution and any other predicted stuff '''

        next_recurr_state, action_dist, *others = self.network(recurr_state, full_state) # invoke the policy
        return (next_recurr_state, action_dist, *others)

    def _evaluate_without_recurrence(self, rollout):
        ''' Given a rollout, evaluate it against current policy '''

        rollout_new = Rollout(device=self.device)
        for ((recur_state, full_state), action, reward, _), _, _ in rollout:
            if recur_state is not None:
                recur_state = recur_state.detach()
            _, action_dist, *others = self.timestep(recur_state, full_state)
            rollout_new << ((recur_state, full_state), action, reward, action_dist, *others)

        # Just transfer the returns
        if len(rollout._returns) != 0:
            rollout_new.returns = rollout.returns

        return rollout_new

    def _evaluate_with_recurrence(self, rollout):
        ''' Given a rollout, evaluate it against current policy (build recurrence afresh) '''

        rollout_new = Rollout(device=self.device)
        recur_state = None
        for ((_, full_state), action, reward, _), _, _ in rollout:
            recur_state, action_dist, *others = self.timestep(recur_state, full_state)
            rollout_new << ((recur_state, full_state), action, reward, action_dist, *others)

        # Just transfer the returns
        if len(rollout._returns) != 0:
            rollout_new.returns = rollout.returns

        return rollout_new

    def evaluate(self, rollout, recurrence=True):
        if recurrence:
            return self._evaluate_with_recurrence(rollout)
        else:
            return self._evaluate_without_recurrence(rollout)

    def episode(self, horizon, global_network_state=None, global_env_state=None, render=False, dry=False, eps=None, cyclic=False):
        '''
        Samples and returns an entire rollout (as 'Rollout' instance).
        Arguments:
            horizon: Maximum length of the episode.
            global_state: A global state for the whole episode. (TODO: Look into this interface)
            render: A 2-tuple (bool, float) containing whether want to render and an optional time delay
            dry: Only sample an episode, i.e., sequence of experience tuples (s, a, r), not anything else
        '''

        state = self.reset(global_state=global_env_state) # prepares for a new episode
        if cyclic:
            state, action, _ = self.environment.get_cyclic_next_state_action()
            state = torch.tensor([state], dtype=torch.float32)
            action = (torch.tensor([[action]]), )
        state = state.unsqueeze(0)
        

        # print(state)
        recurr_state = global_network_state

        rollout = Rollout(device=self.device)

        # loop for many time-steps
        for t in range(horizon):
            # Rendering
            if render:
                self.environment.render()
            
            next_recurr_state, action_dist, *others = self.timestep(recurr_state, state)
            if cyclic and t == 0 :
                pass
            else:
                if eps is None:
                    action = action_dist.sample() # sample an action
                else:
                    if np.random.random() <= eps:
                        with torch.no_grad():
                            # n_action = len(action_dist.sample())
                            # print("n_action:", n_action)
                            n_action = self.env.n_action
                            # print(torch.randint(n_action, (1, 1, 1)))
                            action = (torch.randint(self.env.n_action, (1, 1)), )
                            # print(action)
                            # exit()
                            
                    else:
                        action = action_dist.sample() # sample an action
                        # print('a:', action)
            # print(action)
            
            # Transition to new state and retrieve a reward
            next_state, reward, done, _ = self.environment.step(*[a.to(torch.device('cpu')).numpy() for a in action])
            # print(next_state)
            reward = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            
            if not dry:
                rollout << ((recurr_state, state), action, reward, action_dist, *others)
            else:
                rollout << ((recurr_state, state), action, reward, None, None)
    
            state = next_state # update current state
            state = state.unsqueeze(0)
            recurr_state = next_recurr_state # update current recurrent state
            
            if done: break

        # One last entry for the last state (sometimes required)
        _, action_dist, *others = self.timestep(recurr_state, state)
        action = action_dist.sample()

        # No need for env simulation anymore, R = 0. anyway
        reward = torch.tensor([[0.]], dtype=torch.float32, device=self.device)

        if not dry:
            rollout << ((recurr_state, state), action, reward, action_dist, *others)
        else:
            rollout << ((recurr_state, state), action, reward, None, None)

        return rollout

    def calc_QV(self):
        with torch.no_grad():
            if isinstance(self.env, Chain):
                gamma = 0.99
                n_state = self.environment.size+1
                r_small = self.environment.r_small
                r_big = self.environment.r_big
                n_action = self.environment.n_action
                Q = torch.zeros((n_state, n_action), dtype=torch.float)
                V = torch.zeros((n_state), dtype=torch.float)
                states = torch.tensor([[i] for i in range(n_state)], dtype=torch.float)
                _, action_dist, *others = self.timestep(None, states)
                probs = action_dist.distribs[0].probs
                # print(action_dist.distribs[0].probs)
                for s in reversed(range(1, n_state-1)):
                    if s == n_state-2:
                        V[s] = probs[s] @ torch.tensor([r_small]*(n_action-1) + [r_big])
                        for i in range(n_action):
                            Q[s][i] = r_small
                        Q[s][n_action-1] = r_big
                    else:
                        V[s] = probs[s] @ torch.tensor([r_small]*(n_action-1) + [gamma*V[s+1] ])

                        for i in range(n_action):
                            Q[s][i] = r_small
                        Q[s][n_action-1] = gamma * V[s+1]
            elif isinstance(self.env, Grid):
                states = torch.tensor([[i] for i in range(self.env.n_state)], dtype=torch.float)
                _, action_dist, *others = self.timestep(None, states)
                probs = action_dist.distribs[0].probs
                # print(probs)

                # for s in range(probs.size(0)):
                #     probs[s] = torch.tensor([0.0, 0.5, 0.0, 0.5])
                # print(probs)
                # exit()
                # self.env.env.render()

                Q, V = self.env.compute_value(probs)
                # for i in range(len(V)):
                #     if i % self.env.size == 0:
                #         print()
                #     print(int(V[i].item()), end=' ')
                # print()
                # exit()

            # print(Q)
            # print(V)
            # exit()
        return Q, V


