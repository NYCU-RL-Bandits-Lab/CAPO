import copy
import torch

from .pgalgo import PGAlgorithm
from ..replay import Replay

class OffPolicyActorCritic(PGAlgorithm):
    ''' Off-Policy REINFORCE with Value-baseline. '''

    def populate_replay_buffer(self, buffer, behavior, horizon, global_network_state, global_env_state,
            n_episodes, render=False):
        avg_length, avg_reward = 0., 0.
        with torch.set_grad_enabled(False):
            for b in range(n_episodes):
                bh_rollout = self.agent(behavior).episode(horizon, global_network_state, global_env_state,
                    dry=False, render=render)[:-1]
                
                # compute some metrics to track
                avg_length = ((avg_length * b) + len(bh_rollout)) / (b + 1)
                avg_reward = ((avg_reward * b) + bh_rollout.rewards.sum()) / (b + 1)

                bh_rollout.mc_returns()
                bh_logprobs = bh_rollout.logprobs
                bh_rollout = bh_rollout.make_dry()

                if not self.reccurrent:
                    bh_logprobs = bh_logprobs.view(-1, 1)
                    bh_rollout = bh_rollout.vectorize()
                buffer.populate_buffer((bh_rollout, bh_logprobs))
        return avg_reward, avg_length

    def train(self, global_network_state, global_env_state, *, behavior, horizon, batch_size=4, gamma=0.99, entropy_reg=1e-2,
            value_reg=0.5, buffer_size=100, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = None if 'grad_clip' not in kwargs.keys() else kwargs['grad_clip']
        
        self.logger = kwargs['logger']
        
        if not hasattr(self, 'replay'):
            self.replay = Replay(buffer_size)
            self.update_count = 0
        
        self.avg_reward, self.avg_length = self.populate_replay_buffer(self.replay, behavior, horizon,
                global_network_state, global_env_state, batch_size, render)

        batch = self.replay.get_batch(batch_size)

        self.zero_grad()
        ratios = []
        Q, V = self.agent(self.network).calc_QV()
        for b, (bh_roll, bh_lp) in enumerate(batch):
            rollout = self.agent(self.network).evaluate(bh_roll, recurrence=False)

            returns, logprobs, actions= rollout.returns, rollout.logprobs, rollout.actions[0]
            values, = rollout.others
            
            entropyloss = rollout.entropy
            
            # The importance ratio
            ratio = (logprobs.detach() - bh_lp.detach()).exp()
            ratios.extend(ratio)
            states = rollout._states[0][1]
            _V = V[states.long()].flatten()
            _Q = Q[states.long()].gather(-1, actions.unsqueeze(-1)).flatten()
            advantage = (_Q - _V)[: ,None]
            
            # advantage = returns - values
            
        
            if standardize and advantage.numel() != 1:
                advantage = (advantage - advantage.mean()) / advantage.std()

            policyloss = - ratio * advantage.detach() * logprobs
            # valueloss = advantage.pow(2)
            # loss = policyloss.mean() + value_reg * valueloss.mean() - entropy_reg * entropyloss.mean()
            loss = policyloss.mean() 
            loss /= batch_size
            loss.backward()
        self.step(grad_clip)
        self.update_count += 1
        self.avg_ratio = torch.max(torch.tensor(ratios)).item()

            

        # if self.update_count % 10 == 0:
            # behavior.load_state_dict(self.network.state_dict())
        
        with torch.no_grad():
            self.avg_reward, self.avg_length = self.populate_replay_buffer(Replay(buffer_size), self.network, horizon,
                global_network_state, global_env_state, batch_size, render)

        return self.avg_reward, self.avg_length, self.avg_ratio