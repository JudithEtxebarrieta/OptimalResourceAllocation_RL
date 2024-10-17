'''
Codigo usado para guardar los datos de train y test durante la ejecucion de varios procesos para diferentes entornos.

Disponer de estos datos nos permite no tener que ejecutar despues la heuristica que propongamos online, ya que la 
podemos ejecutar sobre los datos almacenados.

Para el estudio de un unico proceso, solo necesitamos datos hasta convergencia como mucho. Lo bueno de esto es que las ejecuciones
hasta convergencia requieren menos tiempo que las ejecuciones hasta multiples veces la convergencia. Lo malo es que los datos
guardados no serviran para aplicar heuristicos que se centran en repartir los recursos (resource allocator, RA) entre diferentes procesos. 
Aunque, hay que tener en cuneta que si ya se tienen los datos hasta convergencia el resto de datos solo pueden ser iguales o peores. Ademas, 
si que podemos conocer que entornos son validos para un RA, los que tienen procesos que convergen a diferentes optimos locales.

Se ejecutan 20 seeds hasta convergencia minimo en los entornos:

- InvertedDoublependulum
- Ant
- Humanoid

'''

import sys
import gymnasium as gym
from our_library import PPOLearner, UtilsDataFrame

# #--------------------------------------------------------------------------------------------------
# # InvertedDoublePendulum (https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)
# #--------------------------------------------------------------------------------------------------
# # Lo que he ejecutado en el cluster.
# list_seeds=[3,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
# timesteps_conv=500000 # 10.48550/ARXIV.1707.06347
# budget_conv_per=20

# for seed in list_seeds:

#     PPOLearner.start_learn_process(env=gym.make("InvertedDoublePendulum-v4"),
#                                 seed=seed,
#                                 total_timesteps=timesteps_conv*budget_conv_per,
#                                 n_test_episodes=100, 
#                                 path='results/EnvironmentProcesses/InvertedDoublePendulum/',
#                                 csv_name='InvertedDoublePendulum_seed'+str(seed)+'.csv',
#                                 also_train=True,
#                                 partial_save=250) # (total_timesteps/n_steps)//partial_save=20 particiones aprox.
    
# # Juntar csv de train y test obtenidos en las particiones por semilla.
# seed_partition={3:21,
#                 6:9,
#                 7:20,
#                 8:20,
#                 11:28,
#                 10:23,
#                 12:27,
#                 13:20,
#                 14:17,
#                 15:20,
#                 16:24,
#                 17:16,
#                 18:27,
#                 19:11,
#                 20:12,
#                 21:7,
#                 22:7,
#                 23:7
#             }# Exactamente la cantidad de particiones ejecutadas en el cluster 
#              # (algunas pasan de 20 y otras no llegan, porque el codigo ejecutado estaba con un
#              # total_timesteps superior al principio y algunas ejecuciones las he parado antes, respectivamente)

# for seed in list_seeds:
#     list_csv_name=['df_train_'+str(i)+'_InvertedDoublePendulum_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
#     UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/EnvironmentProcesses/InvertedDoublePendulum/','df_train_InvertedDoublePendulum_seed'+str(seed))

#     list_csv_name=['df_test_'+str(i)+'_InvertedDoublePendulum_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
#     UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/EnvironmentProcesses/InvertedDoublePendulum/','df_test_InvertedDoublePendulum_seed'+str(seed))

# #--------------------------------------------------------------------------------------------------
# # Ant (https://gymnasium.farama.org/environments/mujoco/ant/)
# #--------------------------------------------------------------------------------------------------
# # Lo que he ejecutado en el cluster.
# list_seeds=[1,2,3,4]
# timesteps_conv=10000000 # https://chatgpt.com/c/66edbee7-cd40-800f-b61f-f6fa198e2629

# for seed in list_seeds:

#     PPOLearner.start_learn_process(env=gym.make("Ant-v4"),
#                                 seed=seed,
#                                 total_timesteps=timesteps_conv,
#                                 n_test_episodes=100, 
#                                 path='results/EnvironmentProcesses/Ant/',
#                                 csv_name='Ant_seed'+str(seed)+'.csv',
#                                 also_train=True,
#                                 partial_save=500)# (total_timesteps/n_steps)//partial_save=10 particiones aprox.

# # Juntar csv de train y test obtenidos en las particiones por semilla.
# seed_partition={1:10,
#                 2:10,
#                 3:10,
#                 4:10
#             }

# for seed in list_seeds:
#     list_csv_name=['df_train_'+str(i)+'_Ant_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
#     UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/EnvironmentProcesses/Ant/','df_train_Ant_seed'+str(seed))

#     list_csv_name=['df_test_'+str(i)+'_Ant_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
#     UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/EnvironmentProcesses/Ant/','df_test_Ant_seed'+str(seed))

# #--------------------------------------------------------------------------------------------------
# # Humanoid (https://gymnasium.farama.org/environments/mujoco/humanoid/
# #--------------------------------------------------------------------------------------------------

# list_seeds=list(range(1,21))
# timesteps_conv=30000000 # https://chatgpt.com/c/66edbee7-cd40-800f-b61f-f6fa198e2629


# for seed in list_seeds:

#     PPOLearner.start_learn_process(env=gym.make("Humanoid-v4"),
#                                 seed=seed,
#                                 total_timesteps=timesteps_conv,
#                                 n_test_episodes=100, 
#                                 path='results/EnvironmentProcesses/Humanoid/',
#                                 csv_name='Humanoid_seed'+str(seed)+'.csv',
#                                 also_train=True,
#                                 partial_save=500)

# Juntar csv de train y test obtenidos en las particiones por semilla.
seed_partition={1:4,
                2:4,
                3:4,
                4:4
            }
list_seeds=[1,2,3,4]
for seed in list_seeds:
    list_csv_name=['df_train_'+str(i)+'_Humanoid_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
    UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/EnvironmentProcesses/Humanoid/','df_train_Humanoid_seed'+str(seed))

    list_csv_name=['df_test_'+str(i)+'_Humanoid_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
    UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/EnvironmentProcesses/Humanoid/','df_test_Humanoid_seed'+str(seed))

