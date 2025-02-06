'''
TODO
- conseguir importar las funciones en main, para poderlas ejecutar desde otro script
- conseguir hacer deterministas learn_process y eval_output
- definir funcion para evaluar cualquier politica cargandola desde la carpeta process_info

Parametros de proceso:
- algo
- env
- seed
- train_for_env_steps

Parametros para Interaction:
- rollout
- num_workers
- num_envs_per_worker

Parametros para Update policy:
- batch_size
- num_batches_per_epoch
- num_epoch

Comentarios de parametros:
- rollout*num_workers*num_envs_per_worker tiene que ser un multiplo de batch_size*num_batches_per_epoch, i.e., los datos generados durante la
interaccion con el entorno deben ser exactamente los necesarios para definir un epoch en la actualizacion de la politica.

'''
import pandas as pd
import numpy as np
from main import Commun


# Ejecutar mismo proceso 2 veces (ejecutando el fichero main.py)

# Comprobar que learn_process es determinista-> NO
df_traj1=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution1/process_info/df_traj.csv')
df_traj1['traj_rewards']=[np.array(Commun.compress_decompress_list(i,compress=False)) for i in list(df_traj1['traj_rewards'])]
df_traj1['traj_ep_end']=[np.array(Commun.compress_decompress_list(i,compress=False)) for i in list(df_traj1['traj_ep_end'])]
df_traj2=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution2/process_info/df_traj.csv')
df_traj2['traj_rewards']=[np.array(Commun.compress_decompress_list(i,compress=False)) for i in list(df_traj2['traj_rewards'])]
df_traj2['traj_ep_end']=[np.array(Commun.compress_decompress_list(i,compress=False)) for i in list(df_traj2['traj_ep_end'])]

print('learn_process with samplefactory is deterministic: '+str(df_traj1.equals(df_traj2)))

# Comprobar que eval_output es determinista-> NO
df_eval1=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution1/eval_p0.csv')
df_eval2=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution2/eval_p0.csv')

print('eval_output with samplefactory is deterministic: '+str(df_eval1.equals(df_eval2)))