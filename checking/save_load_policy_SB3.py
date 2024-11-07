'''
Probar que los cambios hechos en our_library.py para guardar las politicas funcionan. Comprobar que las politicas
guardadas al volverlas a cargar nos devuelven los mismos rewards de validacion que los guardados en la base de datos df_test.
Ademas, probar que se obtienen los mismos rewards tanto ejecutando los episodios de validacion en secuencial como en paralelo,
y que la validacion en paralelo es mas rapida.
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from experiments.our_library import PPOLearner, PolicyValidation, UtilsDataFrame
import pandas as pd
import time as time

def model_saving_and_evaluation(env_name):
    # Primero guardar base de datos test y las politicas (ejecuto 20 iteraciones, i.e. 20 politicas)
    PPOLearner.start_learn_process(env=gym.make(env_name+"-v4"),
                                    seed=1,
                                    total_timesteps=2048*5,
                                    n_test_episodes=20, 
                                    path='checking/results/save_load_policy_SB3/',
                                    csv_name=env_name+'_seed1.csv',
                                    also_models=True)

    # Luego cargar las politicas y validarlas de nuevo sobre los mismos episodios para comprobar que los resultados salen igual que en la df guardada arriba.
    df_test=pd.read_csv('checking/results/save_load_policy_SB3/df_test_'+env_name+'_seed1.csv')

    correct_policy_save=True
    correct_parallel_evaluation=True
    time_sequential=0
    time_parallel=0
    for i in range(df_test.shape[0]):

        policy=PPOLearner.load_policy('checking/results/save_load_policy_SB3/policies_'+str(env_name)+'_seed1/policy'+str(i+1)+'.zip')

        # Validacion secuencial.
        start_time=time.time()
        _,_,all_ep_reward,_=PolicyValidation.evaluate(policy,gym.make(env_name+"-v4"),20)
        time_sequential+=time.time()-start_time
        # Validacion en paralelo.
        start_time=time.time()
        _,_,all_ep_reward_p,_=PolicyValidation.parallel_evaluate(policy,env_name+"-v4",20,4)
        time_parallel+=time.time()-start_time

        # Comprobar que la validacion en paralelo es equivalente a la secuencial en valores reward.
        if all_ep_reward!=all_ep_reward_p:
            correct_parallel_evaluation=False

        # Comprobar que la validacion tras cargar las politicas es equivalente.
        if UtilsDataFrame.compress_decompress_list(df_test['ep_test_rewards'][i],compress=False)!=all_ep_reward:
            correct_policy_save=False

    # Imprimir comprobaciones.        
    print(env_name)
    print(correct_policy_save)
    print(correct_parallel_evaluation)
    print('Sequiential time: '+str(time_sequential)+'; Parallel time: '+str(time_parallel))


# Programa principal.
model_saving_and_evaluation('Ant')
model_saving_and_evaluation('Humanoid')
model_saving_and_evaluation('InvertedDoublePendulum')
