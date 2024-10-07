'''
Comprobando que la reanudacion de procesos de aprendizaje con PPO y semillas fijas funciona bien.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import gymnasium as gym
from experiments.our_library import PPOLearner
#--------------------------------------------------------------------------------------------------------------------------------
# Primero ejecutar dos procesos de aprendizaje con PPO para semillas 1 y 2, y durante 2048*5 time steps (=5 iteraciones)
#--------------------------------------------------------------------------------------------------------------------------------
PPOLearner.start_learn_process(env=gym.make("Ant-v4"),
                                        seed=1,
                                        total_timesteps=2048*5,
                                        n_test_episodes=10, 
                                        path='checking/results/resume_learn_process/',
                                        csv_name='seed1.csv')

# La base de datos tambien se puede guardar en partes, cada cierto numero de politicas (e.g. cada 2 politicas).
PPOLearner.start_learn_process(env=gym.make("Ant-v4"),
                                        seed=1,
                                        total_timesteps=2048*5,
                                        n_test_episodes=10, 
                                        path='checking/results/resume_learn_process/',
                                        csv_name='partial_seed1.csv',
                                        partial_save=2)

PPOLearner.start_learn_process(env=gym.make("Ant-v4"),
                                        seed=2,
                                        total_timesteps=2048*5,
                                        n_test_episodes=10, 
                                        path='checking/results/resume_learn_process/',
                                        csv_name='seed2.csv')

#--------------------------------------------------------------------------------------------------------------------------------
# Para comprobar que la reanudacion funciona bien y se puede aplicar para multiples procesos que ya hayan sido inicializados 
# previamente: repetir ejecucion de los dos procesos anteriores, pero ahora dividiendolos en tres tandas.
#--------------------------------------------------------------------------------------------------------------------------------
list_seeds=[1,2] # Vamos a reanudar dos procesos.

# Inicializacion
process_buffer=[] # Para almacenar lo necesario de cada proceso para poder reanudarlo.
for seed in list_seeds:
    process=PPOLearner.start_learn_process(env=gym.make("Ant-v4"),
                                        seed=seed,
                                        total_timesteps=2048*2,
                                        n_test_episodes=10, 
                                        path='checking/results/resume_learn_process/',
                                        csv_name='seed'+str(seed)+'_1.csv')

    process_buffer.append(process)

# Primera reanudacion
process1=PPOLearner.resume_learn_process_online(process_buffer[0],
                         total_timesteps=2048*2,
                         path='checking/results/resume_learn_process/',
                         csv_name='seed1_2.csv')


process2=PPOLearner.resume_learn_process_online(process_buffer[1],
                         total_timesteps=2048*1,
                         path='checking/results/resume_learn_process/',
                         csv_name='seed2_2.csv')

# Segunda reanudacion
PPOLearner.resume_learn_process_online(process1,
                         total_timesteps=2048*1,
                         path='checking/results/resume_learn_process/',
                         csv_name='seed1_3.csv')


PPOLearner.resume_learn_process_online(process2,
                         total_timesteps=2048*2,
                         path='checking/results/resume_learn_process/',
                         csv_name='seed2_3.csv')



