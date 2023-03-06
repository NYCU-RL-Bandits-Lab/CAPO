from typing import Any, Dict, Optional, Type, Union, Tuple
import torch as th
import gym
import os
from gym import spaces
from torch.nn import functional as F
from torch.distributions import Bernoulli, Categorical, Normal
from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.offpac.policies import OffPACPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit, Transition
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import explained_variance, polyak_update, get_linear_fn, is_vectorized_observation, get_ms
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer, TrajectoryBuffer, Trajectory, RolloutBuffer
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
import time
from geomloss import SamplesLoss
from stable_baselines3.common.distributions import EMD


class OffPAC(OffPolicyAlgorithm):

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        target_update_interval: int = 10,
        behav_update_interval: int = 100,
        tau: float = 0.9,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        train_freq: Union[int, Tuple[int, str]] = (128, 'step'),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        create_eval_env: bool = False,
        _init_setup_model: bool = True,
        KL: bool = False,
        exploration_fraction: float = 0.5,
        exploration_initial_eps: float = 0.5,
        exploration_final_eps: float = 0.01,
        support_multi_env: bool = True,
        share: bool = True,
        max_alpha: int = 10,
        reg_coef: float = 0.0,
        behav_tau: float = 1.0,
        use_rms_prop: bool = True,
        rms_prop_eps: float = 1e-5,
        use_v_net: bool=False,
        EM: bool=False,
        use_mse: bool=False,
        save_path: str=None,
        fix_after_n=-1,
        uniform_sampling=False
    ):

        if policy_kwargs is None:
            policy_kwargs = {"uniform_sampling":uniform_sampling}
        else:
            policy_kwargs.update({"uniform_sampling":uniform_sampling})


        super(OffPAC, self).__init__(
            policy,
            env,
            OffPACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
            support_multi_env=support_multi_env,
            share=share
        )
        
        # self.use_uniform_behav = use_uniform_behav
        # self.fixed_behav_policy = fixed_behav_policy
        self.uniform_sampling = uniform_sampling
        self.fix_after_n = fix_after_n
        self.save_path = save_path
        self.use_mse = use_mse
        self.use_v_net = use_v_net
        self.behav_tau = behav_tau
        self.reg_coef = reg_coef
        self.max_alpha = max_alpha
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.KL = KL
        self.EM = EM
        if self.KL:
            # self.train_mode = 'normal'
            self.train_mode = 'value'
        else:
            self.train_mode = 'normal'

        self.target_update_interval = target_update_interval
        self.behav_update_interval = behav_update_interval
        self.trajectory_buffer = None
        self.n_backward = 0
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = exploration_initial_eps
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        if self.n_envs is not None:
            self.trajectories = [Trajectory(self.device) for i in range(self.n_envs)]
        '''
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        
        '''
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)
        if _init_setup_model:
            self._setup_model()

        

    def __getstate__(self):
        state = self.__dict__.copy()

        return state
            
    def _setup_model(self) -> None:
        super(OffPAC, self)._setup_model()
        self._create_aliases()
        self.trajectory_buffer = TrajectoryBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device
        )
        self.replay_buffer = self.trajectory_buffer
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )
        self.rollout_buffer = RolloutBuffer(
            self.train_freq.frequency,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target
        self.behav_net = self.policy.behav_net
        self.action_net = self.policy.action_net
        self.v_mlp_extractor = self.policy.v_mlp_extractor
        self.v_mlp_extractor_target = self.policy.v_mlp_extractor_target
        self.a_mlp_extractor = self.policy.a_mlp_extractor
        self.a_mlp_extractor_target = self.policy.a_mlp_extractor_target
        self.value_net = self.policy.value_net
        self.logger = logger
        

    def _store_transition(
        self, 
        buffer,
        trajectory
    ) -> None:
        buffer.add(trajectory)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        use_behav: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        action_net = self.behav_net if use_behav else self.action_net
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic, use_behav)

        return action, state

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None, use_behav:bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        # action_net = self.behav_net if use_target else self.action_net
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            if self.n_envs == 1:
                unscaled_action = np.array([self.action_space.sample()])
            else:
                unscaled_action = np.array([self.action_space.sample() for i in range(self.n_envs)])

        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
          unscaled_action, _ = self.predict(self._last_obs, deterministic=False, use_behav=use_behav)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        buffer: TrajectoryBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``TrajectoryBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param trajectory_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        # assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True
        self.rollout_buffer.reset()
        


        done = np.array([False for i in range(self.n_envs)])
        episode_reward, episode_timesteps = [0.0 for i in range(self.n_envs)], [0 for i in range(self.n_envs)]
        if train_freq.unit == TrainFrequencyUnit.STEP:
            self.trajectories = [Trajectory(self.device) for i in range(self.n_envs)]
            
        while True:
            ms = [0]
            get_ms(ms)
            
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise()

            # Select action randomly or according to policy
            
            with th.no_grad():
                # action, buffer_action = self._sample_action(learning_starts, action_noise, use_behav=False)
                # log_probs = self.policy.get_action_log_probs(th.tensor(np.array(self._last_obs)).to(self.device), th.tensor(np.array([action])).T.to(self.device), use_behav=False)
                action, buffer_action = self._sample_action(learning_starts, action_noise, use_behav=True)
                log_probs = self.policy.get_action_log_probs(th.tensor(np.array(self._last_obs)).to(self.device), th.tensor(np.array([action])).T.to(self.device), use_behav=True)
                prob = th.exp(log_probs)
                prob = (1 - self.exploration_rate) * prob + (self.exploration_rate) * (1.0 / self.action_space.n)
                prob = prob.cpu().numpy()


            if (prob > 1).any():
                print("prob > 1!!! => Code in offpac.py")
                print(prob)
                print(th.tensor(log_probs))
                exit()

            new_obs, reward, done, infos = env.step(action)

            with th.no_grad():
                if self.use_v_net:
                    latent_pi, latent_vf, latent_sde = self.policy._get_latent(th.tensor(self._last_obs).to(self.device))
                    values = self.value_net(latent_vf).detach()
                else:
                    values = self.policy.compute_value(th.tensor(self._last_obs).to(self.device), use_target_v=False).detach()
                
            # self.rollout_buffer.add(self._last_obs, action.reshape(-1, 1), reward, self._last_episode_starts, values, log_probs.flatten())


            self.num_timesteps += env.num_envs
            num_collected_steps += env.num_envs


            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)
            
            episode_reward += reward
            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, done)

            for i in range(len(self.trajectories)):
                # trajectories[i].add(Transition(self._last_obs[i], action[i], reward[i], new_obs[i], done[i], prob[i]))
                if done[i]:
                    if infos[i]['terminal_observation'].dtype == np.float64:
                        self.trajectories[i].add(Transition(self._last_obs[i], action[i], reward[i], infos[i]['terminal_observation'].astype(np.float32), done[i], prob[i]))
                    else:
                        self.trajectories[i].add(Transition(self._last_obs[i], action[i], reward[i], infos[i]['terminal_observation'], done[i], prob[i]))
                else:
                    self.trajectories[i].add(Transition(self._last_obs[i], action[i], reward[i], new_obs[i], done[i], prob[i]))
            self._last_obs = new_obs
            self._last_episode_starts = done

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is done as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            '''
            if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                # even if the episdoe is not finished, we store the trajectory because no more steps can be performed
                for traj_i, traj in enumerate(trajectories):
                    self._store_transition(buffer, traj)
                    total_timesteps.append(len(traj))
                    
                    trajectories[traj_i] = Trajectory(self.device)
                    
                    episode_rewards.append(episode_reward[traj_i])
                    episode_reward[traj_i] = 0.0
                break
            '''
            



            # store transition of finished episode, but if not more steps can be collected, treat any trajectory as an episode
            if done.any():
                num_collected_episodes += np.sum(done)
                self._episode_num += np.sum(done)
                if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()


            if train_freq.unit == TrainFrequencyUnit.STEP:
                ending = not should_collect_more_steps(train_freq, num_collected_steps//self.n_envs, num_collected_episodes//self.n_envs)
                # if ending, save all trajectories, otherwise only save done episode
                if ending:
                    for traj_i, traj in enumerate(self.trajectories):
                        self._store_transition(buffer, traj)
                        # total_timesteps.append(len(traj)) # is this line affecting anything????   
                        
                        self.trajectories[traj_i] = Trajectory(self.device)
                        
                        episode_rewards.append(episode_reward[traj_i])
                        episode_reward[traj_i] = 0.0
                    break
                else:
                    if done.any():
                        traj_indexes = [i for i in np.arange(len(self.trajectories))[done]]
                        for traj_i in traj_indexes:
                            self._store_transition(buffer, self.trajectories[traj_i])
                            # total_timesteps.append(len(traj)) # is this line affecting anything????   
                            self.trajectories[traj_i] = Trajectory(self.device)
                            episode_rewards.append(episode_reward[traj_i])
                            episode_reward[traj_i] = 0.0
                            
                            

                    


            elif train_freq.unit == TrainFrequencyUnit.EPISODE:
                ending = not should_collect_more_steps(train_freq, num_collected_steps//self.n_envs, num_collected_episodes//self.n_envs)
                if done.any():
                    # if ending, save all trajectories even if not finished
                    # if not ending:
                    traj_indexes = [i for i in np.arange(len(self.trajectories))[done]]
                    for traj_i in traj_indexes:
                        self._store_transition(buffer, self.trajectories[traj_i])
                        # total_timesteps.append(len(traj)) # is this line affecting anything???? 
                        
                        self.trajectories[traj_i] = Trajectory(self.device)
                        
                        episode_rewards.append(episode_reward[traj_i])
                        episode_reward[traj_i] = 0.0
                    '''
                    else:
                        _trajectories = trajectories
                    for traj_i, traj in enumerate(_trajectories):
                        self._store_transition(buffer, traj)
                        total_timesteps.append(len(traj)) # is this line affecting anything????   
                        
                        self.trajectories[traj_i] = Trajectory(self.device)
                        
                        episode_rewards.append(episode_reward[traj_i])
                        episode_reward[traj_i] = 0.0
                    '''
                if ending:
                    break
            else:
                print(train_freq.unit)
                raise Exception("Weird train_freq.unit...")
                exit(-1)
            
        
        if done.any():
            if action_noise is not None:
                action_noise.reset()

        with th.no_grad():
            obs_tensor = th.as_tensor(new_obs).squeeze(1).to(self.device)
            if self.use_v_net:
                latent_pi, latent_vf, latent_sde = self.policy._get_latent(obs_tensor)
                values = self.value_net(latent_vf).detach()
            else:
                values = self.policy.compute_value(obs_tensor, use_target_v=False)

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)
        
        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
            
    def _on_step(self) -> None:
        """
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        pass
        # if self.num_timesteps % self.target_update_interval == 0:

        # self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)

    def _on_update(self) -> None:
        self.logger.record("train/n_updates", self._n_updates)
        if self._n_updates % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            if not self.share:
                polyak_update(self.v_mlp_extractor.parameters(), self.v_mlp_extractor_target.parameters(), self.tau)
        
        
        if self.KL:

            if self._n_updates % 2 == 0:

                self.train_mode='policy'
                '''
                careful that train() will call _on_update()
                so must clear before train
                '''
                
                self.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
                # self.train(gradient_steps=self.replay_buffer.size()*2//self.batch_size, batch_size=self.batch_size)
                self.train_mode='value'
                # print("policy updated")

        if self._n_updates % self.behav_update_interval == 0:
            # if not self.use_uniform_behav:
            
            if self.fix_after_n == -1 or self.fix_after_n >= self._n_updates:
                polyak_update(self.action_net.parameters(), self.behav_net.parameters(), tau=self.behav_tau)
                if not self.share:
                    polyak_update(self.a_mlp_extractor.parameters(), self.a_mlp_extractor_target.parameters(), tau=self.behav_tau)
            self.trajectories = [Trajectory(self.device) for i in range(self.n_envs)]
            self.trajectory_buffer.reset()
            self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
            


    def padding_tensor(self, sequences, device, max_len=None):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        if max_len is None:
            max_len = max([s.size(0) for s in sequences])
        s = sequences[0]
        if s.dim() >= 2:
            list_dims = [num, max_len]
            for d in list(s.size())[1:]:
                list_dims.append(d)
            out_dims = tuple(list_dims)
        else:
            out_dims = (num, max_len)
        out_tensor = th.zeros(out_dims)
        mask = th.zeros((num, max_len))
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if s.dim() == 2:
                out_tensor[i, -length:, :] = tensor
            else:
                out_tensor[i, -length:] = tensor
            mask[i, -length:] = 1
        return out_tensor.to(device), mask.to(device)

    def train(self, gradient_steps: int, batch_size: int=100) -> None:
        self._update_learning_rate(self.policy.optimizer)
        value_losses = []
        policy_losses = []
        '''
        if self.train_mode != 'policy': 
            gradient_steps = max(1, min(gradient_steps, self.replay_buffer.size() // batch_size // 2))
        '''

        # print(self.replay_buffer.size())
        # print(self.replay_buffer.size())
        # print(self.replay_buffer.size() //batch_size)
        # print("steps:" ,gradient_steps)
        
        ms=[0]
        get_ms(ms)

        for i_gradient_step in range(gradient_steps):
            trajectories = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)    
            
            is_last_step = i_gradient_step == (gradient_steps-1)
            
            # trajectories = []
            # trajectories.extend(self.replay_buffer.get_last(self.batch_size))

            # The following "all_{}" is for speed up by doing batched .to(device)

            all_states, all_actions, all_rewards, all_next_states, all_dones, lengths, all_probs = [], [], [], [],[], [], []
            all_next_states = []
            # we merge all the trajectories together for batch ".to(device)", later we extract the trajectories by using "lengths:list"
            ms=[0]
            get_ms(ms)
            for i, traj in enumerate(trajectories):
                states, actions, rewards, next_states, dones, probs = traj.get_tensors(device='cpu')
                lengths.append(actions.size(0))
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_next_states.append(next_states[-1].unsqueeze(0))
                all_dones.append(dones)
                all_probs.append(probs)
            # print("loop traj: ", ms[0] - get_ms(ms))
            

            all_states = th.cat(all_states).to(self.device)
            all_actions = th.cat(all_actions).to(self.device)
            all_rewards = th.cat(all_rewards).to(self.device)
            # all_next_states = th.cat(all_next_states).to(self.device)
            all_next_states = th.cat(all_next_states).to(self.device)
            all_dones = th.cat(all_dones).to(self.device)
            all_probs = th.cat(all_probs).to(self.device)
            use_behav=False
            all_Q_values, all_log_cur_probs, _  = self.policy.evaluate_actions(all_states, all_actions, use_target_v=False, use_behav=use_behav)
            with th.no_grad():
                # all_target_Q_values, _, _  = self.policy.evaluate_actions(all_states, all_actions, use_target_v=True, use_behav=True)

                all_target_Q_values, _, _  = self.policy.evaluate_actions(all_states, all_actions, use_target_v=True, use_behav=use_behav)
        

            # all_values = self.policy.compute_value(all_states, use_target_v=True, use_behav=True)
            all_values = self.policy.compute_value(all_states, use_target_v=True, use_behav=use_behav)
            # all_next_values_last = self.policy.compute_value(all_next_states, use_target_v=True, use_behav=True)
            all_next_values_last = self.policy.compute_value(all_next_states, use_target_v=True, use_behav=use_behav)
            
            traj_index_start = 0
            traj_states, traj_actions, traj_rewards, traj_dones, traj_values = [], [], [], [], []
            traj_Q_values, traj_target_Q_values, traj_rhos, traj_log_probs = [], [], [], []
            traj_latents = []
            max_len = 0
            indexes = []
            next_state_values = []
            # print('1:', ms[0] - get_ms(ms))
            get_ms(ms)
            for traj_i, traj in enumerate(trajectories):
        
                # t = [0]
                # get_ms(t)
                # ms = [0]
                # get_ms(ms)

                max_len = max(max_len, len(traj))
                # states, actions, rewards, next_states, dones, probs = traj.get_tensors()
                # _states, _actions, _rewards, _next_states, _dones = traj.get_tensors(device=None)
                # states, actions, rewards, next_states, dones = traj.get_tensors(device=None)

                traj_length = lengths[traj_i]
                traj_index_end = traj_index_start + traj_length
                states, actions, rewards, dones = all_states[traj_index_start:traj_index_end], all_actions[traj_index_start:traj_index_end], all_rewards[traj_index_start:traj_index_end], all_dones[traj_index_start:traj_index_end]
                Q_values = all_Q_values[traj_index_start:traj_index_end]
                target_Q_values = all_target_Q_values[traj_index_start:traj_index_end]
                probs = all_probs[traj_index_start:traj_index_end]
                log_cur_probs = all_log_cur_probs[traj_index_start:traj_index_end]
                values = th.cat([all_values[traj_index_start:traj_index_end], all_next_values_last[traj_i].unsqueeze(0)])
                traj_index_start += traj_length
                
        
                # KL theta
                latent, old_distribution = self.policy.get_policy_latent(states, use_behav=False)
                # latent = latent / th.max(th.abs(latent), dim=1)[0].view(-1, 1)
                # latent = latent - th.mean(latent, dim=1).view(-1,1)
                if states.dim() == 1:
                    states = states.unsqueeze(0)

                # Q_values, log_cur_probs, _  = self.policy.evaluate_actions(states, actions, use_target_v=False, use_behav=False)
                '''
                with th.no_grad():
                    target_Q_values, _, _  = self.policy.evaluate_actions(states, actions, use_target_v=True, use_behav=True)
                '''
                # print("2:")
                # print(ms[0] - get_ms(ms))
                cur_probs = th.exp(log_cur_probs)
                # compute values of states (and addition last state)
                # values = self.policy.compute_value(th.cat([states, next_states[-1].unsqueeze(0)]), use_target_v=True, use_behav=True) # checked

                next_state_value = values[-1]
                values = values[:-1]

                
                # print("b: ", ms[0] - get_ms(ms))
                next_state_values.append(next_state_value)
                # behav_probs = (1 - self.exploration_rate) * cur_probs + (self.exploration_rate) * (1.0 / self.action_space.n)
                behav_probs = probs.squeeze(1)
                rhos = cur_probs / behav_probs

                traj_states.append(states)
                traj_latents.append(latent)
                traj_actions.append(actions)
                traj_rewards.append(rewards)
                traj_dones.append(dones)
                traj_values.append(values)
                traj_Q_values.append(Q_values.squeeze(1))

                traj_target_Q_values.append(target_Q_values.squeeze(1).detach())
                traj_rhos.append(rhos)
                traj_log_probs.append(log_cur_probs)
                # print("4:")
                # print(ms[0] - get_ms(ms))
                # print(t[0] - get_ms(t))


            traj_states, masks = self.padding_tensor(traj_states, self.device, max_len)
            traj_actions, _ = self.padding_tensor(traj_actions, self.device, max_len)
            traj_rewards, _ = self.padding_tensor(traj_rewards, self.device, max_len)
            traj_dones, _ = self.padding_tensor(traj_dones, self.device, max_len)
            traj_values, _ = self.padding_tensor(traj_values, self.device, max_len)
            traj_Q_values, _ = self.padding_tensor(traj_Q_values, self.device, max_len)
            traj_target_Q_values, _ = self.padding_tensor(traj_target_Q_values, self.device, max_len)
            traj_rhos, _ = self.padding_tensor(traj_rhos, self.device, max_len)
            traj_log_probs, _ = self.padding_tensor(traj_log_probs, self.device, max_len)
            # _traj_latents = th.cat(traj_latents).to(self.device).flatten().reshape(-1, 2)
            traj_latents, _ = self.padding_tensor(traj_latents, self.device, max_len)

            
            traj_old_latents = traj_latents.clone()
            # traj_latents = traj_latents / th.max(th.abs(traj_latents), axis=2)[0].unsqueeze(-1)
            # traj_old_distributions, _ = self.padding_tensor(traj_old_distributions, self.device)

            num = traj_dones.size(0)
            Q_rets = th.zeros((num, max_len), dtype=th.float).to(self.device)
            advantages = th.zeros((num, max_len), dtype=th.float).to(self.device)
            next_state_values = th.tensor(next_state_values).to(self.device)
            alpha = th.zeros((num, max_len), dtype=th.float).to(self.device)

            with th.no_grad():
                dones = traj_dones[:, -1]
                Q_rets[:, -1] = traj_rewards[:, -1] + self.gamma * (1-dones) * next_state_values
                advantages[:, -1] =  Q_rets[:, -1] - traj_values[:, -1]
                for i in reversed(range(max_len-1)):
                    Q_rets[:, i] = traj_rewards[:, i] + self.gamma * (th.clamp(traj_rhos[:, i+1], max=1) * (Q_rets[:, i+1] - traj_target_Q_values[:, i+1]) + traj_values[:, i+1]) 
                    Q_rets = Q_rets * masks

                Q_rets = Q_rets * masks
                advantages = Q_rets - traj_values

            '''
            observations = th.tensor(self.rollout_buffer.observations).to(self.device).squeeze(1)
            returns = th.tensor(self.rollout_buffer.returns).to(self.device)
            actions = th.tensor(self.rollout_buffer.actions).to(self.device).long().flatten()
            # print(observations.size())
            # print(actions.size())
            log_probs = self.policy.get_action_log_probs(observations, actions.unsqueeze(1))

            # advantages = th.tensor(self.rollout_buffer.advantages).to(self.device)
            advantages2 = th.tensor(self.rollout_buffer.advantages).to(self.device)
            values = self.policy.compute_value(observations, use_behav=False, use_target_v=False)
            '''

            '''
            if len(traj_Q_values.flatten()) != len(traj_Q_values[th.abs(traj_Q_values) > 1e-5]):
                print(traj_Q_values)
                print(Q_rets)
                exit()
            '''

            if self.train_mode != 'policy': # if not policy only 
                value_loss = F.mse_loss(th.flatten(traj_Q_values), th.flatten(Q_rets), reduction='mean').to(self.device) * self.vf_coef
                self.logger.record("train/value_loss", value_loss.item())

            else:
                value_loss = th.tensor([0.0]).to(self.device)
            

            if self.train_mode != 'value': # if not value only 
                if not self.KL:
                    policy_loss = -(traj_rhos.detach() * advantages.detach() * traj_log_probs * masks).mean()

                    # policy_loss = -(advantages2.flatten() * log_probs.flatten()).mean()
                else:
                    # print("mean of traj_latents: ", th.mean(traj_latents))
                    with th.no_grad():

                        traj_action_probs = th.exp(traj_log_probs)
                        alpha = th.log(1.0 / traj_action_probs)
                        # alpha = 1.0 / traj_action_probs
                        alpha = th.clamp(alpha, max=self.max_alpha)

                        if i_gradient_step == 0 and False:
                            print("max alpha: {}".format(th.max(alpha)))
  

                        # th.set_printoptions(precision=6)
                        th.set_printoptions(precision=2, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
                        # addition = (th.sign(advantages) * (alpha * (1-traj_action_probs))).unsqueeze(-1)
                        addition = (th.sign(advantages) * (alpha + 0.1)).unsqueeze(-1)
                        
                        update_mask = masks.long().bool().flatten()
                        # update_mask = (addition > 0).flatten()
                        assert addition.size()  == traj_latents.gather(2, traj_actions.long()).size()

                        traj_latents = traj_latents.clamp(min=-50, max=50).detach()

                        traj_latents = traj_latents + th.zeros_like(traj_latents).scatter_(2, traj_actions.long(), addition)

                        # TODO
                        # subtract from all theta by addtion/n
                        
                        traj_latents -= addition / self.action_space.n

                        
                        traj_latents = traj_latents.clamp(min=-50, max=50).detach()

                        if th.max(traj_latents.detach()) > 50 or th.max(traj_latents.detach()) < -50:
                            print("latents abs mean before clamp: {}, max: {}, min: {}".format(th.mean(th.abs(traj_latents.detach())), th.max(traj_latents.detach()), th.min(traj_latents.detach())))

                        

                    
                    old_distribution = Categorical(probs=F.softmax(traj_old_latents.view(-1, self.action_space.n)[update_mask], dim=1))
                        
                    new_distribution = Categorical(probs=F.softmax(traj_latents.view(-1, self.action_space.n).detach()[update_mask], dim=1))
                    reg_loss = self.reg_coef * th.norm(traj_old_latents.view(-1, self.action_space.n), dim=1, p=2).mean()
                    ent_loss = self.ent_coef * old_distribution.entropy().mean()
                    
                    if self.EM:
                        KL_loss = EMD(old_distribution, new_distribution).to(self.device)
                    else:
                        # KL_loss = (0.5 * th.distributions.kl_divergence(old_distribution, new_distribution).sum() + 0.5 * th.distributions.kl_divergence( new_distribution, old_distribution).sum()) / num
                        if not self.use_mse:
                            KL_loss = th.distributions.kl_divergence(old_distribution, new_distribution).mean().to(self.device)
                        else:
                            KL_loss = th.nn.MSELoss()(traj_old_latents.view(-1, self.action_space.n)[update_mask], traj_latents.view(-1, self.action_space.n)[update_mask])

                    # KL_loss = th.nn.MSELoss()(old_distribution.probs, new_distribution.probs)
                    policy_loss = KL_loss + reg_loss
                    # max_diff_statewise = th.max(th.abs(new_distribution.probs - old_distribution.probs), dim=1)[0]
                    # max_diff, max_diff_idx = th.max(max_diff_statewise, dim=0)
                    # if i_gradient_step == 0 and self._n_updates % 8 == 0:
                    #     # print("max traj_latents: ", th.max(traj_latents.flatten(), dim=-1))
                    #     # print("max traj_latents: ", th.max(traj_latents.flatten()))
                    #     # print("min traj_latents: ", th.min(traj_latents.flatten()))
                    #     # print("Old max prob: ", th.max(old_distribution.probs))
                    #     # print("New max prob: ", th.max(new_distribution.probs))
                        
                    #     # print("Max difference: ", max_diff)
                    #     # print("Old: ", old_distribution.probs[max_diff_idx])
                    #     print(traj_old_latents.view(-1, self.action_space.n)[update_mask][0:5])
                    #     print(traj_latents.view(-1, self.action_space.n)[update_mask][0:5])
                    #     # print("New: ", new_distribution.probs[max_diff_idx])
                    #     print("regularization loss: ", reg_loss) 
                    #     print("Ent loss: ", ent_loss) 
                    #     if self.EM:
                    #         print("EM loss: ", KL_loss)
                    #     else:
                    #         print("KL loss: ", KL_loss)
                    self.logger.record("train/alpha_mean", alpha.mean().item())
                    self.logger.record("train/alpha_max", alpha.max().item())
                    self.logger.record("train/alpha_min", alpha.min().item())
                    self.logger.record("train/old_prob_max", th.max(old_distribution.probs).item())
                    self.logger.record("train/new_prob_max", th.max(new_distribution.probs).item())

                self.logger.record("train/policy_loss", policy_loss.item())

            else:
                policy_loss = th.tensor([0.0]).to(self.device)
            

            value_losses.append(value_loss.item())


            policy_losses.append(policy_loss.item())
            # value_loss = 0.0
            # policy_loss = 0.0
            # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            '''
            for rollout_data in self.rollout_buffer.get(batch_size=None):

                actions = rollout_data.actions
                actions = actions.long().flatten()
                advantages = rollout_data.advantages

                
                # values = rollout_data.old_values
                if self.use_v_net:
                    latent_pi, latent_vf, latent_sde = self.policy._get_latent(rollout_data.observations)
                    values = self.value_net(latent_vf)
                else:
                    values = self.policy.compute_value(rollout_data.observations, use_behav=False, use_target_v=False)

                values = values.flatten()

                log_probs = self.policy.get_action_log_probs(rollout_data.observations, actions.unsqueeze(1))
                assert advantages.size() == log_probs.flatten().size()
                on_policy_value_loss = F.mse_loss(rollout_data.returns, values)
                on_policy_policy_loss = -(advantages * log_probs.flatten()).mean()
            '''
            # on_policy_value_loss=0.0
            # print(on_policy_policy_loss.requires_grad)
            # print(value_loss)
            # print(on_policy_value_loss.requires_grad)
            # loss = policy_loss + on_policy_policy_loss + self.vf_coef * (value_loss + on_policy_value_loss) / 2
            
            # loss = on_policy_policy_loss + self.vf_coef * (on_policy_value_loss)  # a2c
            loss = policy_loss + value_loss
            # loss = on_policy_policy_loss
            # print(th.sum(th.isinf(loss)))
            
            if th.sum(th.isinf(loss)) > 0:
                print("min alpha: ", th.min(alpha))
                print("max alpha: ", th.max(alpha))
                print("min latent: ", th.min(traj_latents))
                print("max latent: ", th.max(traj_latents))
                print("min prob old: ", th.min(old_distribution.probs))
                print(traj_old_latents.view(-1, self.action_space.n)[update_mask])
                print(old_distribution.probs)
                print("max prob old: ", th.max(old_distribution.probs))
                print("min prob new: ", th.min(new_distribution.probs))
                print("max prob new: ", th.max(new_distribution.probs))
                print("INF detected in loss")
                print("policy_loss: ", policy_loss)
                print("value_loss: ", value_loss)
                exit(-1)
            # loss=policy_loss
            
            
            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.n_backward += 1

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.policy.optimizer.step()
            # print("step: ", ms[0] - get_ms(ms))
        
        if self.train_mode == 'policy':
            with th.no_grad():
                if self.action_space.n == 2:
                    if self.train_mode != 'value' and is_last_step and (self.KL) and (self._n_updates % 50 == 0):
                        latent, _ = self.policy.get_policy_latent(traj_states.view(-1, traj_states.size(-1)), use_behav=False)
                        backward_distribution = Categorical(probs=F.softmax(latent.view(-1, self.action_space.n)[update_mask], dim=1))

                        # print("old latent: ")
                        # print(traj_old_latents.view(-1, self.action_space.n)[update_mask][0:5])
                        # print("new latent: ")
                        # print(latent.view(-1, self.action_space.n)[update_mask][0:5])
                        # print("Old dist: ")
                        # print(old_distribution.probs[0:5] * 100)
                        # print("Update dist: ")
                        # print(backward_distribution.probs[0:5]  * 100)
                        # print("Target dist:")
                        # print(new_distribution.probs[0:5]  * 100)
                        
                        
                        correct = 0
                        incorrect = 0
                

                        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
                        for _old, _new, _target in zip((old_distribution.probs).cpu().numpy(), (new_distribution.probs).cpu().numpy(), backward_distribution.probs.cpu().numpy()):
                            _abs_update = np.sign(_target[0] - _old[0])
                            _abs_target = np.sign(_new[0] - _old[0])
                            _correct = _abs_update == _abs_target
                            hint = "+++" if _correct else "-"
                            if _correct:
                                correct += 1
                            else:
                                incorrect += 1


            return
        else:
            self._n_updates += 1
            self._on_update()
        

            # self.logger.record("train/thetas_hat_max", thetas_hat.max().item())
    
        # logger.record("train/value_loss", np.mean(value_losses))
        # logger.record("train/policy_loss", np.mean(policy_losses))
        # logger.record("rollout/epsilon", self.exploration_rate)
        
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 50,
        tb_log_name: str = "OFFPAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OFFPAC":

        rv = super(OffPAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            use_trajectory_buffer=True
        )
        return rv
