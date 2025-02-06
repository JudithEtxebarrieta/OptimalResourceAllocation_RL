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
from typing import TypeVar, Any, Callable, Optional, Union
from joblib import Parallel, delayed
import warnings

# StableBaselines3
from stable_baselines3.ppo import MlpPolicy, PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.type_aliases import  MaybeCallback
SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")



# SampleFactory
sys.path.insert(0, os.path.abspath("libraries/sample-factory")) # Para usar misma version de sample-factory de GitHub (la que se instala con PyPI no tiene algunos ficheros)

from sample_factory.algo.learning.learner_worker import LearnerWorker
from sample_factory.cfg.arguments import checkpoint_override_defaults, parse_full_cfg, parse_sf_args
from sample_factory.eval import do_eval
from sf_examples.mujoco.mujoco_params import add_mujoco_env_args, mujoco_override_defaults
from sf_examples.mujoco.train_mujoco import register_mujoco_components, parse_mujoco_cfg, run_rl
from sample_factory.utils.utils import ensure_dir_exists, experiment_dir


class Commun:

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
        
    def set_global_random_seed(seed):
        # Seed python RNG
        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        th.manual_seed(seed)


        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


from stable_baselines3.common.callbacks import EvalCallback

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
                      n_steps_per_env=2048,n_envs=1, # Parametros que determinan la interaccion
                      n_epoch=10,batch_size=64, # Parametros que determinan la actualizacion de politica
                      device='auto', # Parametros que determinan el tipo de ejecucion (cpu,gpu)
                      callback=None, n_eval_ep=5, eval_freq=10000, n_eval_envs=1, deterministic_eval=False, # tecnicas de rastreo
                      ):# TODO: añadir como predefinidas las variables/parametros que especifican la interaccion y la actualizacion de politica
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
        if n_envs==1:
            env=gym.make(env_name)
        else:
            env = make_vec_env(env_name, n_envs=n_envs)

        if n_eval_envs==1:
            eval_env=gym.make(env_name)
        else:
            eval_env = make_vec_env(env_name, n_envs=n_eval_envs)

        model = PPO(MlpPolicy, env, seed=seed,n_steps=n_steps_per_env,batch_size=batch_size,n_epochs=n_epoch,verbose=1)# TODO: la primera linea modificarla para otros algortimos
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

            # Guardar las politicas (inspurada en funciones existentes)
            checkpoint = self.learner._get_checkpoint_dict()
            process_dir=ensure_dir_exists(join(experiment_dir(cfg=self.cfg), f"process_info"))
            policy_name = f"{"policy"}_{self.training_iteration_since_resume}.pth"
            filepath = join(process_dir, policy_name)
            print(filepath)
            th.save(checkpoint, filepath)


            # Guardar las trayectorias (usa parametros identificados de interes)
            n_policy=self.training_iteration_since_resume
            n_timesteps=math.prod(self.batcher.training_batches[batch_idx]['rewards'].shape)*(n_policy+1)
            traj_rewards=Commun.compress_decompress_list(self.batcher.training_batches[batch_idx]['rewards'].tolist())
            traj_ep_end=Commun.compress_decompress_list(self.batcher.training_batches[batch_idx]['dones'].tolist())
            df_traj.append([n_policy,n_timesteps,traj_rewards,traj_ep_end])
            # Comprobar si es la ultima iteracion para guardar la base de datos
            if n_timesteps>total_timesteps:
                df_traj_csv=pd.DataFrame(df_traj,columns=['n_policy','n_timesteps','traj_rewards','traj_ep_end'])
                df_traj_csv.to_csv(join(process_dir, f"{"df_traj"}.csv"), index=False)

        my_save(self,batch_idx)
        ########################


        stats = self.learner.train(self.batcher.training_batches[batch_idx])

        self.training_iteration_since_resume += 1
        self.training_batch_released.emit(batch_idx, self.training_iteration_since_resume)
        self.finished_training_iteration.emit(self.training_iteration_since_resume)
        if stats is not None:
            self.report_msg.emit(stats)

    # Mis funciones principales
    def learn_process(method,env,seed,total_timesteps,experiment_name,library_dir):
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
                f'--device={'cpu'}',

                # Relacionados con la interaccion 
                f'--rollout={64}',
                f'--num_workers={8}',
                f'--num_envs_per_worker={8}',

                # Relacionados con la actualizacion de politica
                f'--batch_size={1024}',
                f'--num_batches_per_epoch={8}',
                f'--num_epoch={3}'#,

                # Checkpointing
                #f'--keep_checkpoints={7}'
            ]
            register_mujoco_components()
            cfg = parse_mujoco_cfg(argv=args)
            run_rl(cfg)

        if __name__ == "__main__":
            Commun.set_global_random_seed(seed)
            start_learn(method,env,seed,total_timesteps,experiment_name,library_dir)


    def eval_output(env,seed,experiment_name,library_dir):
        def start_eval(env,seed,experiment_name,train_dir):
            # Parametros para evaluacion (use `sample_env_episodes` >= `num_workers` * `num_envs_per_worker`)
            args = [
                    f'--env={env}',
                    f'--experiment={experiment_name}',f'--train_dir={train_dir}',
                    f'--sample_env_episodes={100}',
                    f'--seed={seed}'
                ]

            # Cargar parametros y modelo
            register_mujoco_components()
            parser, cfg = parse_sf_args(argv=args,evaluation=True)
            add_mujoco_env_args(cfg.env, parser)
            mujoco_override_defaults(cfg.env, parser)
            checkpoint_override_defaults(cfg, parser)
            cfg = parse_full_cfg(parser,argv=args)

            # Evaluar modelo
            status = do_eval(cfg)
            return status

        if __name__ == "__main__":
            Commun.set_global_random_seed(seed)
            start_eval(env,seed,experiment_name,library_dir)


    

    # TODO: def eval_policy_with_test(env,policy,n_eval_ep):


    # TODO: def eval_policy_with_train(env,policy_id,process_info_dir,metric='reward'):

'''
TODO: en la clase SampleFactory

De momento tengo que ejecutar las funciones de la clase SampleFactory desde aqui, porque necesitan el __main__,
y no consigo ejecutarlas de manera externa.

Además, ninguna de las dos funciones principales es determinista. La semilla que se especifica se fija para
torch, numpy y el env de gym. No se que es lo que hace que las ejecuciones no sean deterministas.
'''
method='APPO'
env='mujoco_ant'
seed=1
total_timesteps=1024*8*5 # batch_size*num_batches_per_epoch*(numero de iteraciones)
library_dir='experiments_LibrariesRL/results/samplefactory'

SampleFactory.learn_process(method,env,seed,total_timesteps,'execution1',library_dir)
SampleFactory.eval_output(env,seed,'execution1',library_dir)

SampleFactory.learn_process(method,env,1,total_timesteps,'execution2',library_dir)
SampleFactory.eval_output(env,seed,'execution2',library_dir)
