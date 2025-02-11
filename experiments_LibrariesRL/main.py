import json
import bz2
import base64
import random
import numpy as np
import torch as th
import pandas as pd
import math
from os.path import join
import sys
import os
import gymnasium as gym
from gymnasium import spaces
from typing import TypeVar, Any, Callable, Optional, Union, Tuple
from joblib import Parallel, delayed
import warnings
import time
from collections import deque
import runpy


# StableBaselines3
from stable_baselines3.ppo import MlpPolicy, PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.type_aliases import  MaybeCallback
from stable_baselines3.common.callbacks import EvalCallback
SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


# SampleFactory
sys.path.insert(0, os.path.abspath("libraries/sample-factory")) # Para usar misma version de sample-factory de GitHub (la que se instala con PyPI no tiene algunos ficheros)

from sample_factory.algo.learning.learner_worker import LearnerWorker
from sf_examples.mujoco.train_mujoco import register_mujoco_components, parse_mujoco_cfg, run_rl
from sample_factory.utils.utils import ensure_dir_exists, experiment_dir, log
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.enjoy import visualize_policy_inputs, render_frame
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.algo.learning.learner import Learner

class Commun:

    def external_run(run_script_path, run_lines):

        '''
        run_lines: [una menos que la init en los numeros del script,la misma que la ultima en los numeros del script]
        '''

        this_script = os.path.abspath(__file__)
        this_script_copy = this_script.replace('.py', '_copy.py')

        # Crear una copia de este archivo
        with open(this_script, 'r') as f_b: #Leer
            content_b = f_b.readlines()

        with open(this_script_copy, 'w') as f_b_copy: # Crear y copiar
            f_b_copy.writelines(content_b)  

        # Agregar lineas a ejecutar
        with open(run_script_path, 'r') as f_a: # Leer lineas de interes
            lines_a = f_a.readlines()
        lines_to_run = [lines_a[i] for i in run_lines]

        with open(this_script_copy, 'a') as f_b_copy: # Añadir lineas de interes
            f_b_copy.writelines(lines_to_run)

        # Ejecutar fichero y eliminar
        runpy.run_path(this_script_copy, run_name='__main__')
        os.remove(this_script_copy)

    def compress_decompress_list(my_list,compress=True):
        if compress:
            # Convertir la lista a una cadena JSON compacta
            json_str = json.dumps(my_list)

            # Comprimir la cadena
            compressed_data = bz2.compress(json_str.encode('utf-8'))

            # Convertir a base64 para guardar como texto en la base de datos
            compressed_str = base64.b64encode(compressed_data).decode('utf-8')

            return compressed_str
        else:
            # Leer la cadena comprimida de la base de datos
            compressed_data = base64.b64decode(my_list.encode('utf-8'))

            # Descomprimir la cadena
            json_str = bz2.decompress(compressed_data).decode('utf-8')

            # Convertir la cadena JSON de vuelta a lista
            my_list = json.loads(json_str)

            return my_list
    
    def training_stats(train_rewards,train_ep_ends,n_timesteps_per_iter,stats_window_size):
        ep_rw_policy=[]

        for time_steps in n_timesteps_per_iter:
            current_train_rewards=train_rewards[:time_steps]
            current_train_ep_ends=train_ep_ends[:time_steps]

            ep_rw=[]
            last_i=0
            current_i=0

            for i in current_train_ep_ends:
                current_i+=1
                if i:
                    ep_rw.append(sum(current_train_rewards[last_i:current_i]))
                    last_i=current_i
                
                    

            if len(ep_rw)<stats_window_size:
                ep_rw_policy.append(np.mean(ep_rw))
            else:
                ep_rw_policy.append(np.mean(ep_rw[-stats_window_size:]))

        return ep_rw_policy
class StableBaselines3:

    # Funciones modificadas
    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        
        ##############################MODIFICACION
        global process_dir, n_policy
        n_policy=0

        # Guardar politica inicial
        self.save(process_dir+'/policy'+str(n_policy)+'.zip')
        #################################

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

            ###############################MODIFICACION
            # Guardar politicas
            self.save(process_dir+'/policy'+str(n_policy)+'.zip')
            n_policy+=1
            ######################################

        callback.on_training_end()

        return self
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        ########################MODIFICACION
        global n_policy,df_traj
        policy_traj_rewards=[[] for _ in range(env.num_envs)]
        policy_traj_ep_end=[[] for _ in range(env.num_envs)]
        #########################



        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            ###############################MODIFICACION
            for i in range(env.num_envs):
                policy_traj_rewards[i].append(float(rewards[i]))
                policy_traj_ep_end[i].append(True if dones[i] else False)
            ################################

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        #########################MODIFICACION
        #([n_policy,n_timesteps,traj_rewards,traj_ep_end])
        df_traj.append([n_policy,self.num_timesteps,Commun.compress_decompress_list(policy_traj_rewards),Commun.compress_decompress_list(policy_traj_ep_end)])
        ##########################

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def _on_step(self) -> bool:

        #####################################
        global all_initial_states, eval_env_name, make_eval_deterministic

        def my_evaluate_policy(
            model: "type_aliases.PolicyPredictor",
            env: Union[gym.Env, VecEnv],
            n_eval_episodes: int = 10,
            deterministic: bool = True,
            render: bool = False,
            callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
            reward_threshold: Optional[float] = None,
            return_episode_rewards: bool = False,
            warn: bool = True,
        ) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
            """
            Runs policy for ``n_eval_episodes`` episodes and returns average reward.
            If a vector env is passed in, this divides the episodes to evaluate onto the
            different elements of the vector env. This static division of work is done to
            remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
            details and discussion.

            .. note::
                If environment has not been wrapped with ``Monitor`` wrapper, reward and
                episode lengths are counted as it appears with ``env.step`` calls. If
                the environment contains wrappers that modify rewards or episode lengths
                (e.g. reward scaling, early episode reset), these will affect the evaluation
                results as well. You can avoid this by wrapping environment with ``Monitor``
                wrapper before anything else.

            :param model: The RL agent you want to evaluate. This can be any object
                that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
                or policy (``BasePolicy``).
            :param env: The gym environment or ``VecEnv`` environment.
            :param n_eval_episodes: Number of episode to evaluate the agent
            :param deterministic: Whether to use deterministic or stochastic actions
            :param render: Whether to render the environment or not
            :param callback: callback function to do additional checks,
                called after each step. Gets locals() and globals() passed as parameters.
            :param reward_threshold: Minimum expected reward per episode,
                this will raise an error if the performance is not met
            :param return_episode_rewards: If True, a list of rewards and episode lengths
                per episode will be returned instead of the mean.
            :param warn: If True (default), warns user about lack of a Monitor wrapper in the
                evaluation environment.
            :return: Mean reward per episode, std of reward per episode.
                Returns ([float], [int]) when ``return_episode_rewards`` is True, first
                list containing per-episode rewards and second containing per-episode lengths
                (in number of steps).
            """
            is_monitor_wrapped = False
            # Avoid circular import
            from stable_baselines3.common.monitor import Monitor

            if make_eval_deterministic:
                env=gym.make(eval_env_name)# MODIFICADO

            if not isinstance(env, VecEnv):
                env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

            if make_eval_deterministic:
                env.seed(0)# MODIFICADO

            is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

            if not is_monitor_wrapped and warn:
                warnings.warn(
                    "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
                    "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
                    "Consider wrapping environment first with ``Monitor`` wrapper.",
                    UserWarning,
                )

            n_envs = env.num_envs
            episode_rewards = []
            episode_lengths = []
            initial_states=[]# MODIFICADO


            episode_counts = np.zeros(n_envs, dtype="int")
            # Divides episodes among different sub environments in the vector as evenly as possible
            episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

            current_rewards = np.zeros(n_envs)
            current_lengths = np.zeros(n_envs, dtype="int")
            observations = env.reset()


            initial_states.append(observations)# MODIFICADO

            states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)
            while (episode_counts < episode_count_targets).any():
                actions, states = model.predict(
                    observations,  # type: ignore[arg-type]
                    state=states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
                new_observations, rewards, dones, infos = env.step(actions)
                current_rewards += rewards
                current_lengths += 1
                for i in range(n_envs):
                    if episode_counts[i] < episode_count_targets[i]:
                        # unpack values so that the callback can access the local variables
                        reward = rewards[i]
                        done = dones[i]
                        info = infos[i]
                        episode_starts[i] = done

                        if callback is not None:
                            callback(locals(), globals())

                        if dones[i]:
                            if is_monitor_wrapped:
                                # Atari wrapper can send a "done" signal when
                                # the agent loses a life, but it does not correspond
                                # to the true end of episode
                                if "episode" in info.keys():
                                    # Do not trust "done" with episode endings.
                                    # Monitor wrapper includes "episode" key in info if environment
                                    # has been wrapped with it. Use those rewards instead.
                                    episode_rewards.append(info["episode"]["r"])
                                    episode_lengths.append(info["episode"]["l"])
                                    # Only increment at the real end of an episode
                                    episode_counts[i] += 1
                            else:
                                episode_rewards.append(current_rewards[i])
                                episode_lengths.append(current_lengths[i])
                                episode_counts[i] += 1
                            current_rewards[i] = 0
                            current_lengths[i] = 0

                            initial_states.append(new_observations)

                observations = new_observations

                if render:
                    env.render()

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            if reward_threshold is not None:
                assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
            if return_episode_rewards:
                return episode_rewards, episode_lengths, initial_states # MODIFICADO (añadido el initial_states)
            return mean_reward, std_reward

        ###################################
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, initial_states = my_evaluate_policy( #MODIFICADO
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                all_initial_states.append(initial_states)#MODIFICADO

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)


                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    initial_states=all_initial_states,#MODIFICADO
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
    
    # Mis funciones principales
    def learn_process(method,env_name,seed,total_timesteps,experiment_name,library_dir, # Parametros que determinan el proceso
                      n_steps_per_env=2048,n_workers=1, # Parametros que determinan la interaccion (aqui siempre n_envs_per_worker=1)
                      n_epoch=10,batch_size=64, # Parametros que determinan la actualizacion de politica
                      device='auto', # Parametros que determinan el tipo de ejecucion (cpu,gpu)
                      callback=None, n_eval_ep=5, eval_freq=10000, n_eval_envs=1, deterministic_eval=False,stats_window_size=100, # tecnicas de rastreo
                      ):# Añadidas como predefinidas las variables/parametros que especifican la interaccion y la actualizacion de politica
        # Variables globales
        global df_traj, process_dir, all_initial_states, eval_env_name, make_eval_deterministic
        df_traj=[]
        process_dir=library_dir+'/'+experiment_name+'/process_info'
        all_initial_states=[]
        eval_env_name=env_name
        make_eval_deterministic=deterministic_eval

        # Crear nuevos directorios.
        os.makedirs(process_dir)

        # Modificar funciones de librerias existentes
        OnPolicyAlgorithm.learn=StableBaselines3.learn
        OnPolicyAlgorithm.collect_rollouts=StableBaselines3.collect_rollouts

        if callback==True:
            EvalCallback._on_step= StableBaselines3._on_step

        # Iniciar proceso de aprendizaje fijando el metodo, el env y la semilla.
        if n_workers==1:
            env=gym.make(env_name)
        else:
            env = make_vec_env(env_name, n_envs=n_workers)

        if n_eval_envs==1:
            eval_env=gym.make(env_name)
        else:
            eval_env = make_vec_env(env_name, n_envs=n_eval_envs)

        model = PPO(MlpPolicy, env, seed=seed,n_steps=n_steps_per_env,batch_size=batch_size,n_epochs=n_epoch,verbose=1,stats_window_size=stats_window_size)# TODO: la primera linea modificarla para otros algoritmos
        model.set_random_seed(seed)

        if callback==True:
            callback=EvalCallback(eval_env,n_eval_episodes=n_eval_ep,eval_freq=eval_freq,
                                  log_path=library_dir+'/'+experiment_name,best_model_save_path=library_dir+'/'+experiment_name)

        model.learn(total_timesteps=total_timesteps,callback=callback)

        # Guardar base de datos con trajectorias.
        df_traj_csv=pd.DataFrame(df_traj,columns=['n_policy','n_timesteps','traj_rewards','traj_ep_end'])
        df_traj_csv.to_csv(join(process_dir, f"{"df_traj"}.csv"), index=False)

        # Guardar el modelo output
        model.save(library_dir+'/'+experiment_name+'/policy_output.zip')

    def eval_policy(policy_id,env,seed,n_eval_ep,process_dir,
                    n_workers=1):
        
        def eval_single_episode(args):
            env_name,policy,episode=args
        
            env=gym.make(env_name)
            if not isinstance(env, VecEnv):
                env = DummyVecEnv([lambda: env])
            env.seed(0)

            obs=[env.reset() for _ in range(episode)][-1]# La lista de estados iniciales con interaccion coincide con los estados iniciales impares sin interaccion.

            episode_rewards = 0
            episode_len=0
            done = False # Parameter that indicates after each action if the episode continues (False) or is finished (True).

            with th.no_grad():
                while not done:
                    action, _states = policy.predict(obs, deterministic=True) # The action to be taken with the model is predicted.       
                    obs, reward, done, info = env.step(action) # Action is applied in the environment.
                    episode_rewards+=reward # The reward is saved.
                    episode_len+=1

            return episode_rewards, episode_len, obs

        def parallel_eval(policy,env_name,n_eval_ep,n_workers):

            # Set up the parallel processing pool
            results=Parallel(n_jobs=n_workers, backend="loky")(
                    delayed(eval_single_episode)([env_name,policy,episode]) for episode in range(1,n_eval_ep+1))
                
            # Split the results into rewards and episode lengths
            all_episode_reward, all_episode_len, all_init_state= zip(*results)

            #return np.mean(all_episode_reward), np.std(all_episode_reward), [float(i) for i in all_episode_reward], [int(i)for i in all_episode_len], all_init_state
            return [float(i) for i in all_episode_reward], all_episode_len, np.array(all_init_state)
        # Cragar la politica.
        policy=PPO.load(process_dir+'/'+str(policy_id)) #TODO: tras llamar a process_learn se deberia crear un fichero con la info de la conf para saber que algo hemos usado

        # Evaluar la politica.
        eval_metrics=parallel_eval(policy,env,n_eval_ep,n_workers)

        # Guardar datos de evaluacion.
        #print('ep_mean: '+str(ep_mean)+';  ep_std: '+str(ep_std)) # TODO: esto se puede hacer mas sofisticado, con otras metricas, guardar en .csv.
        return eval_metrics
    
class SampleFactory:
    
    # Funciones modificadas

    def on_new_training_batch(self, batch_idx: int):

        ###########################
        def my_save(self,batch_idx):
            global df_traj, total_timesteps
            last_iter=False

            # Guardar las politicas (inspirada en funciones existentes)
            checkpoint = self.learner._get_checkpoint_dict()
            process_dir=ensure_dir_exists(join(experiment_dir(cfg=self.cfg), f"process_info"))
            policy_name = f"{"policy"}_{self.training_iteration_since_resume}.pth"
            filepath = join(process_dir, policy_name)
            th.save(checkpoint, filepath)


            # Guardar las trayectorias (usa parametros identificados de interes)
            n_policy=self.training_iteration_since_resume
            n_timesteps=math.prod(self.batcher.training_batches[batch_idx]['rewards'].shape)*(n_policy+1)
            # time_seconds=None # TODO: estaria bien añadir esta columna, ya que la freq de checkpointing se mide en segundos
            traj_rewards=Commun.compress_decompress_list(self.batcher.training_batches[batch_idx]['rewards'].tolist())
            traj_ep_end=Commun.compress_decompress_list(self.batcher.training_batches[batch_idx]['dones'].tolist())
            df_traj.append([n_policy,n_timesteps,traj_rewards,traj_ep_end])
            # Comprobar si es la ultima iteracion para guardar la base de datos
            if n_timesteps>total_timesteps:
                last_iter=True
                df_traj_csv=pd.DataFrame(df_traj,columns=['n_policy','n_timesteps','traj_rewards','traj_ep_end'])
                df_traj_csv.to_csv(join(process_dir, f"{"df_traj"}.csv"), index=False)
                return last_iter
            
            return last_iter

        last_iter=my_save(self,batch_idx)
        ########################


        stats = self.learner.train(self.batcher.training_batches[batch_idx])

        self.training_iteration_since_resume += 1
        self.training_batch_released.emit(batch_idx, self.training_iteration_since_resume)
        self.finished_training_iteration.emit(self.training_iteration_since_resume)
        if stats is not None:
            self.report_msg.emit(stats)

        ####################################MODIFICADO
        # Aunque se haya suerado el limite de steps en la interacion, en la ultima iteracion la politica se actualiza
        if last_iter:
            checkpoint = self.learner._get_checkpoint_dict()
            process_dir=ensure_dir_exists(join(experiment_dir(cfg=self.cfg), f"process_info"))
            policy_name = f"{"policy"}_{self.training_iteration_since_resume}.pth"
            filepath = join(process_dir, policy_name)
            th.save(checkpoint, filepath)
        #####################################

    def my_enjoy(cfg: Config) -> Tuple[StatusCode, float]:
        verbose = False

        cfg = load_from_checkpoint(cfg)

        eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
        assert (
            cfg.env_frameskip % eval_env_frameskip == 0
        ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
        render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
        cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
        log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

        cfg.num_envs = 1

        render_mode = "human"
        if cfg.save_video:
            render_mode = "rgb_array"
        elif cfg.no_render:
            render_mode = None

        env = make_env_func_batched(
            cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
        )
        env_info = extract_env_info(env, cfg)

        if hasattr(env.unwrapped, "reset_on_init"):
            # reset call ruins the demo recording for VizDoom
            env.unwrapped.reset_on_init = False

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        actor_critic.eval()

        device = th.device("cpu" if cfg.device == "cpu" else "cuda")
        actor_critic.model_to_device(device)

        policy_id = cfg.policy_index

        global eval_policy_from_checkpointing, eval_checkpoint_id #MODIFICADO
        if eval_policy_from_checkpointing=='True':#MODIFICADO
            if eval_checkpoint_id!=None:#MODIFICADO
                checkpoints=[Learner.checkpoint_dir(cfg, policy_id)+'/checkpoint_'+str(eval_checkpoint_id)+'.pth']
            else:#MODIFICADO
                name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
                checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
            

            
        else:#MODIFICADO
            checkpoints=[join(experiment_dir(cfg=cfg), "process_info")+'/policy_'+str(eval_policy_from_checkpointing)+'.pth']
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])


        episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
        true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
        num_frames = 0

        last_render_start = time.time()

        def max_frames_reached(frames):
            return cfg.max_num_frames is not None and frames > cfg.max_num_frames

        reward_list = []

        ######################################MODIFICADO
        global make_eval_deterministic
        if make_eval_deterministic:

            env.seed(0)
        ######################################

        obs, infos = env.reset()
        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
        rnn_states = th.zeros([env.num_agents, get_rnn_size(cfg)], dtype=th.float32, device=device)
        episode_reward = None
        finished_episode = [False for _ in range(env.num_agents)]

        video_frames = []
        num_episodes = 0

        with th.no_grad():
            while not max_frames_reached(num_frames):
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

                if not cfg.no_render:
                    visualize_policy_inputs(normalized_obs)
                policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)

                # sample actions from the distribution by default
                actions = policy_outputs["actions"]

                if cfg.eval_deterministic:
                    action_distribution = actor_critic.action_distribution()
                    actions = argmax_actions(action_distribution)

                # actions shape should be [num_agents, num_actions] even if it's [1, 1]
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(env_info, actions)

                rnn_states = policy_outputs["new_rnn_states"]

                for _ in range(render_action_repeat):
                    last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                    obs, rew, terminated, truncated, infos = env.step(actions)
                    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
                    dones = make_dones(terminated, truncated)
                    infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                    if episode_reward is None:
                        episode_reward = rew.float().clone()
                    else:
                        episode_reward += rew.float()

                    num_frames += 1
                    if num_frames % 100 == 0:
                        log.debug(f"Num frames {num_frames}...")

                    dones = dones.cpu().numpy()
                    for agent_i, done_flag in enumerate(dones):
                        if done_flag:
                            finished_episode[agent_i] = True
                            rew = episode_reward[agent_i].item()
                            episode_rewards[agent_i].append(rew)

                            true_objective = rew
                            if isinstance(infos, (list, tuple)):
                                true_objective = infos[agent_i].get("true_objective", rew)
                            true_objectives[agent_i].append(true_objective)

                            if verbose:
                                log.info(
                                    "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                    agent_i,
                                    num_frames,
                                    episode_reward[agent_i],
                                    true_objectives[agent_i][-1],
                                )
                            rnn_states[agent_i] = th.zeros([get_rnn_size(cfg)], dtype=th.float32, device=device)
                            episode_reward[agent_i] = 0

                            if cfg.use_record_episode_statistics:
                                # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                                if "episode" in infos[agent_i].keys():
                                    num_episodes += 1
                                    reward_list.append(infos[agent_i]["episode"]["r"])
                            else:
                                num_episodes += 1
                                reward_list.append(true_objective)

                    # if episode terminated synchronously for all agents, pause a bit before starting a new one
                    if all(dones):
                        render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                        time.sleep(0.05)

                    if all(finished_episode):
                        finished_episode = [False] * env.num_agents
                        avg_episode_rewards_str, avg_true_objective_str = "", ""
                        for agent_i in range(env.num_agents):
                            avg_rew = np.mean(episode_rewards[agent_i])
                            avg_true_obj = np.mean(true_objectives[agent_i])

                            if not np.isnan(avg_rew):
                                if avg_episode_rewards_str:
                                    avg_episode_rewards_str += ", "
                                avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                            if not np.isnan(avg_true_obj):
                                if avg_true_objective_str:
                                    avg_true_objective_str += ", "
                                avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                        log.info(
                            "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                        )
                        log.info(
                            "Avg episode reward: %.3f, avg true_objective: %.3f",
                            np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                            np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                        )

                    # VizDoom multiplayer stuff
                    # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                    #     key = f'PLAYER{player}_FRAGCOUNT'
                    #     if key in infos[0]:
                    #         log.debug('Score for player %d: %r', player, infos[0][key])

                if num_episodes >= cfg.max_num_episodes:
                    break

        env.close()


        if cfg.save_video:
            if cfg.fps > 0:
                fps = cfg.fps
            else:
                fps = 30
            generate_replay_video(experiment_dir(cfg=cfg), video_frames, fps, cfg)

        if cfg.push_to_hub:
            generate_model_card(
                experiment_dir(cfg=cfg),
                cfg.algo,
                cfg.env,
                cfg.hf_repository,
                reward_list,
                cfg.enjoy_script,
                cfg.train_script,
            )
            push_to_hf(experiment_dir(cfg=cfg), cfg.hf_repository)


        return np.array(episode_rewards)[0],np.mean(np.array(episode_rewards)[0])

    # Mis funciones principales
    def learn_process(method,env,seed,total_timesteps,experiment_name,library_dir, # Parametros que determinan el proceso
                      n_steps_per_env=64, n_workers=8,n_envs_per_worker=8, # Interaccion
                      n_epoch=3,batch_size=1024, n_batches_per_epoch=8, # Actualizacion de politica
                      device='cpu', # Tipo de ejecucion
                      save_every_sec=15, keep_checkpoints=2, save_best_every_sec=5,save_best_metric='reward',save_best_after=100000, stats_avg=100 # Para seleccionar politica output
                      ):
        
        # Variables globales
        global df_traj
        df_traj=[]

        # Redefinir funciones
        LearnerWorker.on_new_training_batch=SampleFactory.on_new_training_batch

        def start_learn(algo,env,seed,train_for_env_steps,experiment_name,train_dir):

            # Parametros de interes
            args = [
                # Para determinar el proceso
                f'--algo={algo}',
                f'--env={env}',
                f'--seed={seed}',
                f"--train_for_env_steps={train_for_env_steps}",

                # Directorio de almacenamiento
                f'--experiment={experiment_name}',f'--train_dir={train_dir}',

                # Para ejecutar en PC (para ejecuciones futuras en el cluster esto quitar)
                f'--device={device}',

                # Relacionados con la interaccion 
                f'--rollout={n_steps_per_env}',
                f'--num_workers={n_workers}',
                f'--num_envs_per_worker={n_envs_per_worker}',
                f'--worker_num_splits={1}',

                # Relacionados con la actualizacion de politica
                f'--batch_size={batch_size}',
                f'--num_batches_per_epoch={n_batches_per_epoch}',
                f'--num_epoch={n_epoch}',

                # Checkpointing
                f'--save_every_sec={save_every_sec}',
                f'--keep_checkpoints={keep_checkpoints}',
                f'--save_best_every_sec={save_best_every_sec}',
                f'--save_best_metric={save_best_metric}',
                f'--save_best_after={save_best_after}', 
                f'--stats_avg={stats_avg}'
                
            ]
            register_mujoco_components()
            cfg = parse_mujoco_cfg(argv=args)
            run_rl(cfg)

        if __name__ == "__main__":
            start_learn(method,env,seed,total_timesteps,experiment_name,library_dir)

    def eval_policy(env,seed,n_eval_ep,experiment_name,library_dir,
                    policy_id=False,checkpoint_id=None,load_checkpoint_kind='latest',
                    deterministic_eval=False):

        global make_eval_deterministic, eval_policy_from_checkpointing, eval_checkpoint_id
        make_eval_deterministic=deterministic_eval
        eval_policy_from_checkpointing= 'True' if str(policy_id)=='False' or checkpoint_id!=None else policy_id
        eval_checkpoint_id=checkpoint_id

        def start_eval(env,seed,experiment_name,train_dir):
            args = [
                    f'--env={env}',
                    f'--experiment={experiment_name}',f'--train_dir={train_dir}',
                    f'--max_num_episodes={n_eval_ep}',
                    '--no_render',
                    f'--load_checkpoint_kind={load_checkpoint_kind}'

                ]
            register_mujoco_components()
            cfg = parse_mujoco_cfg(argv=args,evaluation=True)
            status = SampleFactory.my_enjoy(cfg)
            return status

        if __name__ == "__main__":
            return start_eval(env,seed,experiment_name,library_dir)




