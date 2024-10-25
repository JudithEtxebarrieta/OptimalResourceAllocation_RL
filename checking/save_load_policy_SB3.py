'''
Probar que los cambios hechos en our_library.py para guardar las politicas funcionan. Comprobar que las politicas
guardadas al volverlas a cargar nos devuelven los mismos rewards de validacion que los guardados en la base de datos df_test.
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from experiments.our_library import PPOLearner, evaluate, UtilsDataFrame
import pandas as pd

# Primero guardar base de datos test y las politicas (ejecuto 5 iteraciones, i.e. 5 politicas)
PPOLearner.start_learn_process(env=gym.make("Ant-v4"),
                                seed=1,
                                total_timesteps=2048*5,
                                n_test_episodes=5, 
                                path='checking/results/save_load_policy_SB3/',
                                csv_name='Ant_seed1.csv',
                                also_models=True)

# Luego cargar las politicas y validarlas de nuevo sobre los mismos episodios para comprobar que los resultados salen igual que en la df guardada arriba.
df_test=pd.read_csv('checking/results/save_load_policy_SB3/df_test_Ant_seed1.csv')

correct_policy_save=True
for i in range(df_test.shape[0]):

    policy=PPOLearner.load_policy('checking/results/save_load_policy_SB3/policies_Ant_seed1/policy'+str(i+1)+'.zip')
    _,_,all_ep_reward,_=evaluate(policy,gym.make("Ant-v4"),5)

    if UtilsDataFrame.compress_decompress_list(df_test['ep_test_rewards'][i],compress=False)!=all_ep_reward:
        correct_policy_save=False

print(correct_policy_save)


