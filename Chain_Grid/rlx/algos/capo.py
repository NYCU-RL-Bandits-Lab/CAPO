import copy
import torch

from .pgalgo import PGAlgorithm
from ..replay import Replay
from ..rollout import Rollout
from torch.distributions import Categorical, kl_divergence
import torch.nn.functional as F
import numpy as np

class CAPO(PGAlgorithm):
    ''' Off-Policy REINFORCE with Value-baseline. '''
    def __init__(self, agent, network, reccurrent=False, optimizer='adam', optim_kwargs={ }, cyclic=False, eps=0.3):
        super().__init__(agent, network, reccurrent, optimizer, optim_kwargs)
        self.cyclic = cyclic
        self.eps = eps

    def populate_replay_buffer(self, buffer, behavior, horizon, global_network_state, global_env_state,
            n_episodes, render=False, cyclic=False, eps=0.3):
        avg_length, avg_reward = 0., 0.

        with torch.set_grad_enabled(False):
            for b in range(n_episodes):
                bh_rollout = self.agent(behavior).episode(horizon, global_network_state, global_env_state,
                    dry=False, render=render, eps=eps, cyclic=cyclic)[:-1]
                
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

    def gen_full_batch(self, buffer, global_network_state, global_env_state, n_episodes):
        self.agent.environment.reset_cyclic()
        self.populate_replay_buffer(buffer, self.network, 1,
                global_network_state, global_env_state, n_episodes, False, cyclic=True, eps=0)
        # passed = False
        
        # while not passed:
        #     self.agent.environment.reset()
        #     rollout = Rollout(device=self.agent.device)
        #     state, action, passed = self.agent.environment.get_cyclic_next_state_action()
        #     state = torch.tensor([state], dtype=torch.float32)
        #     action = (torch.tensor([[action]]), )
        #     next_state, reward, done, _ = self.agent.environment.step(*[a.to(torch.device('cpu')).numpy() for a in action])
        #     reward = torch.tensor([[reward]], dtype=torch.float32, device=self.agent.device)
        #     next_state = torch.from_numpy(next_state).float().to(self.agent.device)
        #     rollout << ((None, state), action, reward, None, None)
        #     rollout.mc_returns
        #     logprobs = rollout.logprobs
        #     buffer.populate_buffer((rollout, logprobs))
        return buffer.all()
            


    def train(self, global_network_state, global_env_state, *, behavior, horizon, batch_size=4, gamma=0.99, entropy_reg=1e-2,
            value_reg=0.5, buffer_size=100, render=False, **kwargs):
        standardize = False if 'standardize_return' not in kwargs.keys() else kwargs['standardize_return']
        grad_clip = None if 'grad_clip' not in kwargs.keys() else kwargs['grad_clip']
        if not hasattr(self, 'replay'):
            self.replay = Replay(buffer_size)
            self.update_count = 0
        
        Q, V = self.agent(self.network).calc_QV()
        
        
        full_batch = kwargs.get('full_batch', False)
        if full_batch:
            full_size = (self.agent.environment.size) * self.agent.environment.n_action
            batch = self.gen_full_batch(Replay(full_size), global_network_state, global_env_state, full_size)
        else:
            if self.cyclic:
                self.avg_reward, self.avg_length = self.populate_replay_buffer(self.replay, self.network, horizon,
                    global_network_state, global_env_state, batch_size, render, cyclic=self.cyclic, eps=0.9)
            else:
                self.avg_reward, self.avg_length = self.populate_replay_buffer(self.replay, self.network, horizon,
                    global_network_state, global_env_state, batch_size, render, cyclic=self.cyclic, eps=self.eps)
            batch = self.replay.get_batch(batch_size)
            # print(self.replay.buffer[0][0]._states)

        self.zero_grad()
        for b, (bh_roll, bh_lp) in enumerate(batch):
            rollout = self.agent(self.network).evaluate(bh_roll, recurrence=False)

            returns, logprobs, actions = rollout.returns, rollout.logprobs, rollout.actions[0]
            # returns, actions = rollout.returns, rollout.actions[0]
            values, thetas = rollout.others
            states = rollout._states[0][1]
            # print([str(self.agent.env.state_to_pos(s)) + self.agent.env.env.ACTIONS[action.item()] for s, action in zip(states.flatten(), actions.flatten())])
            # exit()
            # print(states)
            # print(Q - V[:, None])
            _V = V[states.long()].flatten()
            # print(Q[states.long()])
            # print(actions)
            _Q = Q[states.long()].gather(-1, actions.unsqueeze(-1)).flatten()
            # print(_Q)
            # print(Q-V)
            # exit()
            act = Categorical(F.softmax(thetas, dim=-1))
            # print(rollout.actions)
            
            # The importance ratio
            # ratio = (logprobs.detach() - bh_lp.detach()).exp()
            # advantage = returns - values
            # print("adv: ", returns - values)
            # print(Q.shape, V.shape)
            # print(actions)
            advantage = (_Q - _V)[: ,None]
            
            # exit()
            # advantage -= 1e-7
            # print(torch.sign(advantage), advantage, actions)
            # print(advantage)
            # exit()

            if standardize and advantage.numel() != 1:
                advantage = (advantage - advantage.mean()) / advantage.std()
            # print(thetas)
            with torch.no_grad():  
                # advantage[actions == 0] = -1
                # advantage[actions == 1] = 1
                # print(advantage, actions)

                new_thetas = thetas.detach().clone()
                # print(new_thetas)
                alpha = torch.log(1.0 / logprobs.detach().exp())
                alpha = torch.clamp(alpha, max=100) 
                # print(alpha)
                # theta_a = new_thetas.gather(1, actions)
                # print(theta_a)
                # print(new_thetas, actions)
                addition = alpha * torch.sign(advantage)
                # print(addition)
                
                # check_addition = [alpha[i] * theta_a[i] * torch.sign(advantage[i]) for i in range(4)]
                # print('check_addition: ', check_addition)
                # print('addition: ', addition, actions)
                zero = torch.zeros_like(thetas)
                # print("new thetas:")
                # print(new_thetas)

                new_thetas += zero.scatter_(-1, actions, addition)
                # print(actions)
                # print(advantage)
                # print(new_thetas - thetas)
                # print(new_thetas)
                # exit()
                # print(_V)
                # for i in range(len(states)):
                #     pos = self.agent.env.state_to_pos(states[i].item())
                #     print(str(pos), self.agent.env.env.ACTIONS[actions[i].item()], advantage[i], _Q[i], "addition: ", addition[i], new_thetas[i])
            # exit()
            entropyloss = rollout.entropy
            # policyloss = - ratio * advantage.detach() * logprobs
            # valueloss = advantage.pow(2)
            # loss = policyloss.mean() + value_reg * valueloss.mean() - entropy_reg * entropyloss.mean()
            new_act = Categorical(F.softmax(new_thetas, dim=-1))
            # print(new_thetas)
            # exit()
            # print(kl_divergence(act, new_act).mean())
            loss = kl_divergence(act, new_act).mean() #+ value_reg * valueloss.mean() #- entropy_reg * entropyloss.mean()
            # loss = kl_divergence(act, new_act).mean() + value_reg * valueloss.mean() - entropy_reg * entropyloss.mean()
            loss /= batch_size
            # print('loss: ', loss)
            loss.backward()
        # print(act.probs)
        self.step(grad_clip)
        self.update_count += 1
            

        # if self.update_count % 10 == 0 and not self.cyclic:
        #     behavior.load_state_dict(self.network.state_dict())

        # eval
        with torch.no_grad():
            self.avg_reward, self.avg_length = self.populate_replay_buffer(Replay(buffer_size), self.network, horizon,
                global_network_state, global_env_state, batch_size, render, cyclic=False, eps=0)
        return self.avg_reward, self.avg_length