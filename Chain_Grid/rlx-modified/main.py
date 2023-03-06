import datetime
import os
import gym, torch
from tqdm import tqdm
import random
import numpy as np
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

N_MAX = 0

def main( args ):
    from torch.utils.tensorboard import SummaryWriter

    environment = GYMEnvs[args.env](size=args.size)
    Network = DiscreteRNNPolicyValue if args.policytype == 'rnn' else TabularPolicyValue
    is_capo = (args.algo == 'capo')
    # print(environment.observation_space, environment.action_spaces)
    # exit()
    network = Network(environment, environment.observation_space, environment.action_spaces, n_hidden=args.hidden, is_capo=is_capo)
    if args.algo == 'offpac': #or args.cyclic:
        behavior = Network(environment, environment.observation_space, environment.action_spaces, n_hidden=args.hidden, is_behavior=True, is_capo=is_capo)
    elif args.algo == 'capo':
        behavior = network


    agent = PGAgent(environment)
    if torch.cuda.is_available():
        network, agent = network.cuda(), agent.cuda()
        if args.algo == 'offpac':
            behavior = behavior.cuda()

    optimizer = 'rmsprop' if args.policytype == 'rnn' else 'adam'
    algorithm = PGAlgos[args.algo](agent, network, args.policytype == 'rnn', optimizer, {'lr': args.lr}, cyclic=args.cyclic)
    # logging object (TensorBoard)
    logger = None
    if len(args.tbdir) != 0:
        args.tbdir = os.path.join(args.tbdir, str(int(datetime.datetime.now().timestamp())) + f"_{args.size}")
        logger = SummaryWriter(os.path.join(args.base, f'{args.tbdir}/{args.tbtag}'), flush_secs=10)

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
            if running_reward > MAXRewards[args.env]:
                N_MAX += 1
                if N_MAX >= 100:
                    break
            else:
                N_MAX = 0

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='Base folder (everything is relative to this)')
    parser.add_argument('--tbdir', type=str, required=False, default='', help='folder name for TensorBoard logging (empty if no TB)')
    parser.add_argument('--tbtag', type=str, required=False, default='rltag', help='Unique identifier for experiment (for TensorBoard)')
    parser.add_argument('--algo', type=str, required=True, choices=PGAlgos.keys(), help='Which algorithm to use')
    parser.add_argument('--gamma', type=float, required=False, default=0.999, help='Discount factor')
    parser.add_argument('--render', action='store_true', help='Render environment while sampling episodes')
    parser.add_argument('--policytype', type=str, required=True, choices=['rnn', 'mlp'], help='Type of policy (MLP or RNN)')
    parser.add_argument('--interval', type=int, required=False, default=10, help='Logging frequency')
    parser.add_argument('-K', '--ppo_k_epochs', type=int, required=False, default=4, help='How many iterations of trusted updates')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='Batch size')
    parser.add_argument('--entropy_reg', type=float, required=False, default=1e-2, help='Regularizer weight for entropy')
    parser.add_argument('--ppo_clip', type=float, required=False, default=0.2, help='PPO clipping parameter (usually 0.2)')
    parser.add_argument('--max_episode', type=int, required=False, default=1000, help='Maximum no. of episodes')
    parser.add_argument('--horizon', type=int, required=False, default=500, help='Maximum no. of timesteps')
    parser.add_argument('--rbsize', type=int, required=False, default=1, help='Size of replay buffer (if needed)')
    parser.add_argument('--max_rbusage', type=int, required=False, default=5, help='Maximum usage of a replay buffer')
    parser.add_argument('--grad_clip', type=float, required=False, default=0., help='Gradient clipping (0 means no clipping)')
    parser.add_argument('--standardize_return', action='store_true', help='standardize all returns/advantages')
    parser.add_argument('--cyclic', action='store_true', help='cyclic capo?')
    parser.add_argument('--full', action='store_true', help='full batch capo?')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--eps', type=float, required=False, default=0.3, help='eps')
    parser.add_argument('--env', type=str, required=True, choices=GYMEnvs.keys(), help='Gym environment name (string)')
    parser.add_argument('--seed', type=int, required=False, default=0, help='seed')
    parser.add_argument('--size', type=int, required=False, default=10, help='size')
    parser.add_argument('--hidden', type=int, required=False, default=256, help='hidden')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main( args )