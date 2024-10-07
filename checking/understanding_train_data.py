import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import gymnasium as gym
from experiments.our_library import PPOLearner, compress_decompress_list
import pandas as pd
import numpy as np
    

# Ejecutar RL process durante 5 iteraciones con verbose
process=PPOLearner.start_learn_process(env=gym.make("InvertedDoublePendulum-v4"),
                                seed=1,
                                total_timesteps=2048*5,
                                n_test_episodes=3, 
                                path='checking/results/understanding_train_data/',
                                csv_name='.csv',
                                also_train=True,
                                verbose=1)

# Comprobar que el output del verbose coincide con lo entendido y los datos de train guardados (rewards y longitud de episodios)
df_train=pd.read_csv('checking/results/understanding_train_data/df_train_.csv')
df_train['train_rewards']=[compress_decompress_list(i,compress=False) for i in list(df_train['train_rewards'])]
df_train['train_ep_end']=[compress_decompress_list(i,compress=False) for i in list(df_train['train_ep_end'])]

stats_window_size=100 # Esta variable se define al inicializar la clase PPO

n_policies=list(df_train['n_policy'])

train_rewards=[]
train_ep_end=[]
for i in n_policies:

    train_rewards+=list(df_train[df_train['n_policy']==i]['train_rewards'])[0]
    train_ep_end+=list(df_train[df_train['n_policy']==i]['train_ep_end'])[0]


n_train_timesteps=list(df_train['n_train_timesteps'])

for time_steps in n_train_timesteps:
    current_train_rewards=train_rewards[:time_steps]
    current_train_ep_end=train_ep_end[:time_steps]

    ep_rew_mean=[]
    ep_len_mean=[]
    last_i=0
    current_i=0
    print('iteration: ')
    for i in current_train_ep_end:
        if i:

            ep_rew_mean.append(sum(current_train_rewards[last_i:current_i]))
            ep_len_mean.append(current_i-last_i)
            last_i=current_i
        current_i+=1
            

    if len(ep_len_mean)<stats_window_size:
        print('ep_len_mean: '+str(np.mean(ep_len_mean)))
        print('ep_rew_mean: '+str(np.mean(ep_rew_mean)))
    else:
        print('ep_len_mean: '+str(np.mean(ep_len_mean[-stats_window_size:])))
        print('ep_rew_mean: '+str(np.mean(ep_rew_mean[-stats_window_size:])))

    



