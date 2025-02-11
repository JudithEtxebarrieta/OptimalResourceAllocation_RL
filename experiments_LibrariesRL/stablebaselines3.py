'''
Entendiendo el output de PPO en StableBaselines3.

Se hacen ciertas comprobaciones, y se observa que:

1) Sobre funciones implementadas: 
- learn_process with stablebaselines3 is deterministic: True
- My implementation eval_policy is deterministic: True

2) Entendiendo uso de callback:
- Without callback policy_output is the policy 4/4 of the sequence
- EvalCallback does not affect the sequence of policies generated: True
- Additional steps consumed in validation are counted: False
- By default, the policies to be validated in eval_freq are validated with the same n_eval_ep initial states: False
- With modification, the policies to be validated in eval_freq are validated with the same n_eval_ep initial states:True
- The policies validated with the callback are the expected ones: True
- The policy selected with the callback is the one with the highest average reward in the n_ep_eval validation episodes: True

'''
from main import StableBaselines3, Commun
import pandas as pd
import numpy as np

# Definir parametros
method='PPO'
env='Ant-v4'
seed=1
total_timesteps=2048*5
library_dir='experiments_LibrariesRL/results/stablebaselines3'

# Ejecutar mismo proceso 2 veces
StableBaselines3.learn_process(method,env,seed,total_timesteps,'execution1',library_dir)
StableBaselines3.learn_process(method,env,seed,total_timesteps,'execution2',library_dir)

# Comprobar que learn_process es determinista-> SI
df_traj1=pd.read_csv(library_dir+'/execution1/process_info/df_traj.csv')
df_traj1['traj_rewards']=[np.array(Commun.compress_decompress_list(i,compress=False)) for i in list(df_traj1['traj_rewards'])]
df_traj1['traj_ep_end']=[np.array(Commun.compress_decompress_list(i,compress=False)) for i in list(df_traj1['traj_ep_end'])]
df_traj2=pd.read_csv(library_dir+'/execution2/process_info/df_traj.csv')
df_traj2['traj_rewards']=[np.array(Commun.compress_decompress_list(i,compress=False)) for i in list(df_traj2['traj_rewards'])]
df_traj2['traj_ep_end']=[np.array(Commun.compress_decompress_list(i,compress=False)) for i in list(df_traj2['traj_ep_end'])]
print('learn_process with stablebaselines3 is deterministic: '+str(df_traj1.equals(df_traj2)))

# Comprobar que eval_policy es determinista-> SI
eval_output1,_,_=StableBaselines3.eval_policy('policy_output',env,seed,10,library_dir+'/execution1')
eval_output2,_,_=StableBaselines3.eval_policy('policy_output',env,seed,10,library_dir+'/execution1')
print('My implementation eval_policy is deterministic: '+str(eval_output1==eval_output2))

# Entender quien es policy_output-> la ultima politica de la secuencia de politicas actualizadas
eval_policy0,_,_=StableBaselines3.eval_policy('policy'+'0',env,seed,10,library_dir+'/execution1/process_info')
eval_policy1,_,_=StableBaselines3.eval_policy('policy'+'1',env,seed,10,library_dir+'/execution1/process_info')
eval_policy2,_,_=StableBaselines3.eval_policy('policy'+'2',env,seed,10,library_dir+'/execution1/process_info')
eval_policy3,_,_=StableBaselines3.eval_policy('policy'+'3',env,seed,10,library_dir+'/execution1/process_info')
eval_policy4,_,_=StableBaselines3.eval_policy('policy'+'4',env,seed,10,library_dir+'/execution1/process_info')

eval_list= [eval_policy1,eval_policy2,eval_policy3,eval_policy4]
for i in range(len(eval_list)):
    if eval_list[i]==eval_output1:
        print('Without callback policy_output is the policy '+str(i+1)+'/'+str(len(eval_list))+' of the sequence')

# Entendiendo herramienta EvalCallback
total_timesteps=2048*10 # por defecto eval_freq=10000, de esta forma deberia de hacer 2 callbacks
StableBaselines3.learn_process(method,env,seed,total_timesteps,'execution3',library_dir)
StableBaselines3.learn_process(method,env,seed,total_timesteps,'execution4',library_dir,callback=True)
StableBaselines3.learn_process(method,env,seed,total_timesteps,'execution5',library_dir,callback=True,deterministic_eval=True)

# El EvalCallback afecta en la secuencia de politicas, i.e., modifica la actualizacion de politicas por defecto? NO
equal_seq=True
for i in range(10):
    eval_policy1,_,_=StableBaselines3.eval_policy('policy'+str(i),env,seed,5,library_dir+'/execution3/process_info')
    eval_policy2,_,_=StableBaselines3.eval_policy('policy'+str(i),env,seed,5,library_dir+'/execution4/process_info')
    if eval_policy1!=eval_policy2:
        equal_seq=False
print('EvalCallback does not affect the sequence of policies generated: '+str(equal_seq))

# Se cuentan los steps adicionales de validacion? NO
df_traj1=pd.read_csv('experiments_LibrariesRL/results/stablebaselines3/execution3/process_info/df_traj.csv')
df_traj2=pd.read_csv('experiments_LibrariesRL/results/stablebaselines3/execution4/process_info/df_traj.csv')
print('Additional steps consumed in validation are counted: '+str(list(df_traj1['n_timesteps'])!=list(df_traj2['n_timesteps'])))

# La validacion de las politicas se hace con los mismos n_eval_ep? NO
evaluations=np.load('experiments_LibrariesRL/results/stablebaselines3/execution4/evaluations.npz')
print('By default, the policies to be validated in eval_freq are validated with the same n_eval_ep initial states: '+
      str(np.all(evaluations['initial_states'] == evaluations['initial_states'][0])))

evaluations=np.load('experiments_LibrariesRL/results/stablebaselines3/execution5/evaluations.npz')
print('With modification, the policies to be validated in eval_freq are validated with the same n_eval_ep initial states: '+
      str(np.all(evaluations['initial_states'] == evaluations['initial_states'][0])))

# Que politica se selecciona como mejor? (con los parametros definidos se validan la politica 3 y 8, despues de gastar 10000 y 20000 steps)
all_ep_reward1,_,_=StableBaselines3.eval_policy('policy3',env,seed,5,library_dir+'/execution5/process_info')
all_ep_reward2,_,_=StableBaselines3.eval_policy('policy8',env,seed,5,library_dir+'/execution5/process_info')
print('The policies validated with the callback are the expected ones: '+
      str(np.all([[int(j) for j in i] for i in evaluations['results']]
                 ==[[int(i) for i in all_ep_reward1],[int(i) for i in all_ep_reward2]])))

all_ep_reward3,_,_=StableBaselines3.eval_policy('best_model',env,seed,5,library_dir+'/execution5')
mean_rewards=[np.mean(all_ep_reward1),np.mean(all_ep_reward2)]
idx_max=mean_rewards.index(max(mean_rewards))
idx_best=mean_rewards.index(np.mean(all_ep_reward3))
print('The policy selected with the callback is the one with the highest average reward in the n_ep_eval validation episodes: '+
      str(idx_max==idx_best))








