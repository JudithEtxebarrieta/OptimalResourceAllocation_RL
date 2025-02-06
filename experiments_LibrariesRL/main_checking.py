
# Intentando ejecutar funciones de sample factory desde otro script externo.
# import subprocess

# #from main import SampleFactory


# # SampleFactory
# method='APPO'
# env='mujoco_ant'
# seed=2 
# total_timesteps=64*8*8*7 
# library_dir='experiments_LibrariesRL/results'
# experiment_name='samplefactory0'


# subprocess.run(["python", "-m", f"from experiments_LibrariesRL.main import SampleFactory; SampleFactory.learn_process('{method}','{env}','{seed}','{total_timesteps}','{experiment_name}','{library_dir}')"])


# SampleFactory.learn_process(method,env,seed,total_timesteps,experiment_name,library_dir)
# SampleFactory.eval_output(env,seed,experiment_name,library_dir)

# Entendiendo callbacks en stable baselines 3
from stable_baselines3.ppo import PPO
import gymnasium as gym
env=gym.make('Ant-v4')
a=2

