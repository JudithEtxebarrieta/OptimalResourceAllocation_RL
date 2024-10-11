import os
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy, PPO
import gymnasium as gym
import time

#--------------------------------------------------------------------------------------------------
# Con gymnasium
#--------------------------------------------------------------------------------------------------
gym_env = gym.make('Ant-v4')
model = PPO('MlpPolicy', gym_env,verbose=1,seed=1)

start_time=time.time()
model.learn(total_timesteps=2048*5)
print('Elapsed time with gymnasium: '+str(time.time()-start_time))

#--------------------------------------------------------------------------------------------------
# Con brax (usando CPUs parece que tarda mas que gym)

# Ejecutar desde entorno vitual:
# >> conda deactivate
# >> source venv/venv/bin/activate
#--------------------------------------------------------------------------------------------------
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import brax
from brax import envs
from brax.envs.wrappers.gym import GymWrapper

env_name = 'ant'  # Puedes cambiar esto por otros entornos disponibles: 
# ant, half_cheetah, hopper, humanoid, humanoidstandup, inverted_double_pendulum, inverted_pendulum, pusher, reacher, swimmer, walker2d.

env = envs.create(env_name)
gym_env = GymWrapper(env)
model = PPO('MlpPolicy', gym_env,verbose=1,seed=1)

start_time=time.time()
model.learn(total_timesteps=2048*5)
print('Elapsed time with brax: '+str(time.time()-start_time))

#--------------------------------------------------------------------------------------------------
# Con arlbench (no funciona, problema de compatibilidad con stable-baselines3)

# Ejecutar desde entorno vitual:
# >> conda deactivate
# >> conda activate arlbench
#--------------------------------------------------------------------------------------------------
from arlbench.core.environments import make_env

env=make_env(env_framework='gymnasium',env_name='Ant-v4')
model = PPO('MlpPolicy', env,verbose=1,seed=1)

start_time=time.time()
model.learn(total_timesteps=2048*5)
print('Elapsed time with brax: '+str(time.time()-start_time))



