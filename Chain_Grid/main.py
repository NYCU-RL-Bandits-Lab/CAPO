# Package
import os
import gym
import time
import yaml
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Other .py
from rlx import PGAgent, REINFORCE, ActorCritic, PPO, A2C, OffPolicyActorCritic, CAPO
from rlx.policy import (DiscreteMLPPolicyValue,
                        DiscreteRNNPolicyValue,
                        TabularPolicyValue)
from rlx.env import (CartPolev0,
                     CartPolev1,
                     IncompleteCartPolev0,
                     IncompleteCartPolev1,
                     Chain,
                     Grid)

# setting visible GPU
os.environ['CUDA_VISIBLE_DEVICES']="-1"

PGAlgos = {
    'rf': REINFORCE,
    'ac': ActorCritic,
    'a2c': A2C,
    'ppo': PPO,
    'offpac': OffPolicyActorCritic,
    'capo': CAPO
}

GYMEnvs = {
    'CartPole-v0': CartPolev0,
    'CartPole-v1': CartPolev1,
    'IncompleteCartPole-v0': IncompleteCartPolev0,
    'IncompleteCartPole-v1': IncompleteCartPolev1,
    'Chain-v0': Chain,
    'Grid-v0': Grid 
}

MAXRewards = {
    'CartPole-v0': 193.0,
    'CartPole-v1': 480.0,
    'IncompleteCartPole-v0': 197.0,
    'IncompleteCartPole-v1': 480.0,
    'Chain-v0': 10000,
    'Grid-v0': 100
}

def main( args ):

    # Tune the state num (args.size) in Chain & GridWorld
    if args.env == "Grid-v0":
        environment = GYMEnvs[args.env](size=args.size, use_sparse_reward=args.use_sparse_reward)
    elif args.env ==  "Chain-v0":
        environment = GYMEnvs[args.env](size=args.size, stochastic=args.stochastic)
    else:
        environment = GYMEnvs[args.env]()

    # Define network
    if args.policytype == 'rnn':
        network = DiscreteRNNPolicyValue(environment.observation_space, environment.action_spaces, n_hidden=args.hidden)  
    else:
        network = TabularPolicyValue(environment, environment.observation_space, environment.action_spaces, n_hidden=args.hidden, is_capo=(args.algo == 'capo'))
        
    # Define behavior policy (in CAPO experiment, we only use policytype="mlp")
    if args.algo == 'offpac' and args.policytype == 'rnn':
        behavior = DiscreteRNNPolicyValue(environment.observation_space, environment.action_spaces, n_hidden=args.hidden)  
    elif args.algo == 'offpac' and args.policytype == 'mlp':
        behavior = TabularPolicyValue(environment, environment.observation_space, environment.action_spaces, n_hidden=args.hidden, is_behavior=True, is_capo=(args.algo == 'capo'))
    elif args.algo == 'capo':
        behavior = network

    # agent
    agent = PGAgent(environment)

    # GPU device
    if torch.cuda.is_available():
        network, agent = network.cuda(), agent.cuda()
        if args.algo == 'offpac':
            behavior = behavior.cuda()

    # optimizer
    optimizer = 'rmsprop' if args.policytype == 'rnn' else 'adam'
    
    # algorithm
    algorithm = PGAlgos[args.algo](agent, network, args.policytype == 'rnn', optimizer, {'lr': args.lr}, cyclic=args.cyclic)
    
    # logging object (TensorBoard)
    logger = None
    if len(args.tbdir) != 0:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if os.path.isdir(os.path.join(args.tbdir, f"{args.tbtag}_{timestamp}")):
            time.sleep(3)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if len(args.tbdir) != 0:
            logger = SummaryWriter(os.path.join(args.tbdir, f"{args.tbtag}-{timestamp}"), flush_secs=10)
            # save param
            with open(os.path.join(args.tbdir, f"{args.tbtag}-{timestamp}", "args.yaml"), 'w') as f:
                yaml.dump(args.__dict__, f, Dumper=yaml.CDumper)

        else:
            logger = SummaryWriter(os.path.join(args.tbdir, f"{timestamp}"), flush_secs=10)
            # save param
            with open(os.path.join(args.tbdir, f"{args.tbtag}", "args.yaml"), 'w') as f:
                yaml.dump(args.__dict__, f, Dumper=yaml.CDumper)

    train_args = {
        'horizon': args.horizon,
        'gamma': args.gamma,
        'entropy_reg': args.entropy_reg,
        'ppo_k': args.ppo_k_epochs,
        'batch_size': args.batch_size,
        'ppo_clip': args.ppo_clip,
        'render': args.render,
        'standardize_return': args.standardize_return,
        'grad_clip': None if args.grad_clip == 0. else args.grad_clip,
        'behavior': behavior if (args.algo in {'offpac', 'capo'}) else None,
        'buffer_size': args.rbsize,
        'max_buffer_usage': args.max_rbusage,
        'full_batch': args.full,
        'logger': logger
    }
    if args.algo == 'capo':
        train_args['eps'] = args.eps
    

    # TQDM Formatting
    TQDMBar = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' + \
                    'Reward: {postfix[0][r]:>3.2f}, ' + \
                    'Length: {postfix[0][l]:>3.2f}]'

    with tqdm(total=args.max_episode, bar_format=TQDMBar, disable=None, postfix=[dict(r=0.,l=0)]) as tqEpisodes:
        
        # average episodic reward
        running_reward = 0.

        # terminated condition
        N_MAX = 0

        # loop for many episodes
        for episode in range(args.max_episode):
            
            if args.policytype == 'rnn':
                global_network_state = torch.zeros(1, network.n_hidden, device=agent.device)
            else:
                global_network_state = None
            if args.algo != 'offpac':
                avg_reward, avg_length = algorithm.train(global_network_state, None, **train_args)
            else:
                avg_reward, avg_length, avg_ratio = algorithm.train(global_network_state, None, **train_args)
            running_reward = 0.05 * avg_reward + (1 - 0.05) * running_reward
            
            # terminated condition
            if running_reward > MAXRewards[args.env]:
                N_MAX += 1
                if N_MAX >= 100:
                    break
            else:
                N_MAX = 0

            # log
            if episode % args.interval == 0:
                if tqEpisodes.disable:
                    print(f'[{episode:5d}/{args.max_episode}] Running reward: {running_reward:>4.2f}, Avg. Length: {avg_length:>3.2f}')
                if len(args.tbdir) != 0:
                    logger.add_scalar('reward', running_reward, global_step=episode)
                    logger.add_scalar('length', avg_length, global_step=episode)
                    if args.algo == 'offpac':
                        logger.add_scalar('ratio', avg_ratio, global_step=episode)

            # TQDM update stuff
            if not tqEpisodes.disable:
                tqEpisodes.postfix[0]['r'] = running_reward
                tqEpisodes.postfix[0]['l'] = avg_length
                tqEpisodes.update()

if __name__ == '__main__':
    
    # args
    parser = argparse.ArgumentParser()
    
    # train
    parser.add_argument('--algo', type=str, required=True, choices=PGAlgos.keys(), help='Which algorithm to use')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='Batch size')
    parser.add_argument('--entropy_reg', type=float, required=False, default=1e-2, help='Regularizer weight for entropy')
    parser.add_argument('--env', type=str, required=True, choices=GYMEnvs.keys(), help='Gym environment name (string)')
    parser.add_argument('--eps', type=float, required=False, default=0.3, help='eps')
    parser.add_argument('--hidden', type=int, required=False, default=256, help='hidden')
    parser.add_argument('--horizon', type=int, required=False, default=500, help='Maximum no. of timesteps')
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging frequency')
    parser.add_argument('-K', '--ppo_k_epochs', type=int, required=False, default=4, help='How many iterations of trusted updates')
    parser.add_argument('--gamma', type=float, required=False, default=0.999, help='Discount factor')
    parser.add_argument('--grad_clip', type=float, required=False, default=0., help='Gradient clipping (0 means no clipping)')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--max_episode', type=int, required=False, default=1000, help='Maximum no. of episodes')
    parser.add_argument('--max_rbusage', type=int, required=False, default=5, help='Maximum usage of a replay buffer')
    parser.add_argument('--policytype', type=str, required=True, choices=['rnn', 'mlp'], help='Type of policy (MLP or RNN)')
    parser.add_argument('--ppo_clip', type=float, required=False, default=0.2, help='PPO clipping parameter (usually 0.2)')
    parser.add_argument('--rbsize', type=int, required=False, default=1, help='Size of replay buffer (if needed)')
    parser.add_argument('--standardize_return', action='store_true', help='standardize all returns/advantages')

    # root / path
    parser.add_argument('--tbdir', type=str, required=False, default='', help='folder name for TensorBoard logging (empty if no TB)')
    parser.add_argument('--tbtag', type=str, required=False, default='rltag', help='Unique identifier for experiment (for TensorBoard)')
    
    # chain & gridworld
    parser.add_argument('--stochastic', action='store_true', help='stochastic chain or not')
    parser.add_argument('--size', type=int, required=False, default=10, help='environment size')
    parser.add_argument('--use_sparse_reward', action='store_true', help='gridworld: the greater distance to the termianl, the lesser reward')

    # capo
    parser.add_argument('--cyclic', action='store_true', help='cyclic capo?')
    parser.add_argument('--full', action='store_true', help='full batch capo?')

    # utils
    parser.add_argument('--seed', type=int, required=False, default=0, help='seed')
    parser.add_argument('--render', action='store_true', help='Render environment while sampling episodes')
    args = parser.parse_args()
    
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # run
    main( args )