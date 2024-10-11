import sys
import gymnasium as gym
from our_library import PPOLearner, UtilsDataFrame

#--------------------------------------------------------------------------------------------------
# InvertedDoublePendulum
#--------------------------------------------------------------------------------------------------

# Lo que he ejecutado en el cluster.
list_seeds=[3,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
timesteps_conv=500000 # 10.48550/ARXIV.1707.06347
budget_conv_per=20

for seed in list_seeds:

    PPOLearner.start_learn_process(env=gym.make("InvertedDoublePendulum-v4"),
                                seed=seed,
                                total_timesteps=timesteps_conv*budget_conv_per,
                                n_test_episodes=100, 
                                path='results/KnowingSingleProcess/',
                                csv_name='InvertedDoublePendulum_seed'+str(seed)+'.csv',
                                also_train=True,
                                partial_save=250)
    
# Exactamente la cantidad de particiones ejecutadas en el cluster (algunas pasan de 20 y otras no llegan, porque el codigo ejecutado estaba con un
# limite de train superior al principio y algunas ejecuciones las he parado antes, respectivamente)
seed_partition={3:21,
                6:9,
                7:20,
                8:20,
                11:28,
                10:23,
                12:27,
                13:20,
                14:17,
                15:20,
                16:24,
                17:16,
                18:27,
                19:11,
                20:12,
                21:7,
                22:7,
                23:7
            }

# Join train and test csv per seed
for seed in list_seeds:
    list_csv_name=['df_train_'+str(i)+'_InvertedDoublePendulum_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
    UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/KnowingSingleProcess/','df_train_InvertedDoublePendulum_seed'+str(seed))

    list_csv_name=['df_test_'+str(i)+'_InvertedDoublePendulum_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
    UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/KnowingSingleProcess/','df_test_InvertedDoublePendulum_seed'+str(seed))

#--------------------------------------------------------------------------------------------------
# Ant
#--------------------------------------------------------------------------------------------------
list_seeds=[1,2,3,4]
timesteps_conv=2000000 # https://github.com/DLR-RM/stable-baselines3/issues/48
budget_conv_per=5

for seed in list_seeds:

    PPOLearner.start_learn_process(env=gym.make("Ant-v4"),
                                seed=seed,
                                total_timesteps=timesteps_conv*budget_conv_per,
                                n_test_episodes=100, 
                                path='results/KnowingSingleProcess/',
                                csv_name='Ant_seed'+str(seed)+'.csv',
                                also_train=True,
                                partial_save=500)

seed_partition={1:4,
                2:4,
                3:4,
                4:2
            }
# Join train and test csv per seed
for seed in list_seeds:
    list_csv_name=['df_train_'+str(i)+'_Ant_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
    UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/KnowingSingleProcess/','df_train_Ant_seed'+str(seed))

    list_csv_name=['df_test_'+str(i)+'_Ant_seed'+str(seed)+'.csv' for i in range(1,seed_partition[seed]+1)]
    UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'results/KnowingSingleProcess/','df_test_Ant_seed'+str(seed))

#--------------------------------------------------------------------------------------------------
# AntBulletEnv
#--------------------------------------------------------------------------------------------------
# import gym
# import pybullet_envs

# list_seeds=list(range(1,11))
# timesteps_conv=2000000 # https://github.com/DLR-RM/stable-baselines3/issues/48
# budget_conv_per=5

# for seed in list_seeds:

#     PPOLearner.start_learn_process(env=gym.make("AntBulletEnv-v0"),
#                                 seed=seed,
#                                 total_timesteps=timesteps_conv*budget_conv_per,
#                                 n_test_episodes=100, 
#                                 path='results/KnowingSingleProcess/',
#                                 csv_name='Ant_seed'+str(seed)+'.csv',
#                                 also_train=True,
#                                 partial_save=500)