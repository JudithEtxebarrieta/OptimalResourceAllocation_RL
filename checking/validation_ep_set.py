'''
Encontrando el tamaño del conjunto de validacion que representa la "validacion ideal" (proporciona la validacion de maxima precision)
para los diferentes environments: InvertedDoublePendulum, Ant y Humanoid.

Observaciones en ejecucion del cluster:
- El proceso de guardar las politicas parece que lleva mas tiempo que guardar la validacion de la politica.
- La politica ocupa mas espacio (memoria) que la validacion (al menos el vector encriptado de 100 episodios).
- Como la carpeta de todas las politcas ocupa mucho espacio, para copiarla de Hipatia lleva mucho tiempo. Por eso, primero solo copio los df_test,
luego identifico con df_test las politicas que me interesan (good, mean, bad, deterministic and stochastic) usando los 100 datos de validacion que 
he guardado para cada una, y finalmente solo copio de Hipatia esas politicas identificadas.

'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.our_library import UtilsDataFrame, UtilsFigure, PPOLearner, evaluate
import pandas as pd
import numpy as np
import subprocess
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm


# Cargar csv-s de cluster y juntarlos para formar el df_test
def form_cluster_to_df_test(env_name,seed,n_partitions):

    # Cargar de cluster csv-s relacionados.
    path = f"/home/jechevarrieta/results_fakeHome/df_test_*_"+env_name+'_seed'+str(seed)+'.csv'
    command = ["rsync", "-r", "-avHPe", "sshpass -p R5ShrSjIxLHX ssh -p6556", f"jechevarrieta@hpc.bcamath.org:{path}", "_hipatia/"]
    subprocess.run(command)

    # Juntar csv-s y guardarlos en parquet en el directorio deseado.
    list_csv_name=['df_test_'+str(i)+'_'+env_name+'_seed'+str(seed)+'.csv' for i in range(1,n_partitions+1)]
    UtilsDataFrame.join_csv_and_save_parquet(list_csv_name,'_hipatia/','checking/results/validation_ep_set/'+env_name+'/','df_test_'+env_name+'_seed'+str(seed))

# Identificar con el df_test las politicas de interes y cargarlas desde el cluster.
def load_from_cluster_interesting_policies(env_name,seed):

    # Leer bases de datos.
    df_test=pd.read_parquet('checking/results/validation_ep_set/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_test['std_rewards']=[np.std(UtilsDataFrame.compress_decompress_list(i,compress=False)) for i in list(df_test['ep_test_rewards'])]

    # Encontrar posiciones de politicas de los 5 tipos: mala, regular, buena, determinista y estocastica.
    indx_good=df_test['mean_reward'].idxmax()
    indx_mean=np.argmin(abs(np.array(df_test['mean_reward'])- df_test['mean_reward'].mean()))
    indx_bad=df_test['mean_reward'].idxmin()
    indx_det=df_test['std_rewards'].idxmin()
    indx_stoc=df_test['std_rewards'].idxmax()

    os.makedirs('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed))
    id_policies={'policy_type':['Good','Mean','Bad','Deterministic','Stochastic'],'id_policy':[indx_good+1,indx_mean+1,indx_bad+1,indx_det+1,indx_stoc+1]}
    pd.DataFrame(id_policies).to_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/id_policies.csv')

    # Cargar del cluster unicamente las politicas de interes.
    for id_policy in id_policies['id_policy']:
        cluster_path = f"/home/jechevarrieta/results_fakeHome/policies_{env_name}_seed{seed}/policy{id_policy}.zip"
        pc_path="checking/results/validation_ep_set/"+env_name+"/policies_seed"+str(seed)+'/'
        command = ["rsync", "-r", "-avHPe", "sshpass -p R5ShrSjIxLHX ssh -p6556", f"jechevarrieta@hpc.bcamath.org:{cluster_path}", pc_path]
        subprocess.run(command)

# Conseguir conjunto de validacion de las politicas de interes.
def validate_policies(env_name,seed,n_eval_episodes):

    '''
    Para ejecutar esta funcion hay que hacerlo desde el entorno vistual py39venv, porque al guardar la politica en el cluster 
    se ha usado un entorno virtual con la misma version de python de py39venv. De lo contrario, da error por no considerar 
    mismas versiones de ciertas librerias para guardar y cargar la politica.
    '''

    id_policies=pd.read_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/id_policies.csv')
    new_column=[]

    for i in range(id_policies.shape[0]):
        id_policy=id_policies['id_policy'][i]

        # Cargar politica actual
        policy=PPOLearner.load_policy('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/policy'+str(id_policy)+'.zip')

        # Lista de rewards por episodio (considerar un maximo de episodios)
        _,_,all_ep_reward,_=evaluate(policy,gym.make(env_name+"-v4"),n_eval_episodes)

        new_column.append(UtilsDataFrame.compress_decompress_list(all_ep_reward))
    
    id_policies['ep_test_rewards']=new_column
    id_policies.to_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/id_policies.csv')

# Repetir experimento de la semana pasada: dibujar intervalo de confianza de reward de validacion usando boostrap.
def boostrap_test_reward(ep_test_rewards,policy_type,pos,sample_sizes):

    # Boostrap para cada tamaño de consjunto de validacion.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    for i in sample_sizes:

        mean,q05,q95=UtilsFigure.bootstrap_mean_and_confidence_interval(ep_test_rewards[:i])

        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    # Grafica 1: intervalos de confianza.
    ax=plt.subplot(2,5,pos)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    ax.fill_between(sample_sizes,all_q05,all_q95, alpha=.2, linewidth=0)
    plt.plot(sample_sizes, all_mean, linewidth=1)

    ax.set_title(policy_type+' policy')
    if pos==1:
        ax.set_ylabel('Reward confidence interval with boostrap')
    # Grafica 2: rangos de intervalos de confianza. 
    ax=plt.subplot(2,5,5+pos)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    plt.plot(sample_sizes, abs(np.array(all_q95)-np.array(all_q05)), linewidth=1)

    ax.set_xlabel('Sample size (number of episodes)')
    if pos==1:
        ax.set_ylabel('Boostrap confidence interval range')

def plot_per_policy_type(env_name,seed,list_n_eval_ep):

    fig=plt.figure(figsize=[20,8])
    plt.subplots_adjust(left=0.06,bottom=0.11,right=0.98,top=0.88,wspace=0.33,hspace=0.2)

    id_policies=pd.read_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/id_policies.csv')

    for i in tqdm(range(id_policies.shape[0])):
        policy_type=id_policies['policy_type'][i]
        ep_test_rewards=UtilsDataFrame.compress_decompress_list(id_policies['ep_test_rewards'][i],compress=False)
        boostrap_test_reward(ep_test_rewards,policy_type,i+1,list_n_eval_ep)

    plt.savefig('checking/results/validation_ep_set/'+str(env_name)+str(seed)+'.pdf')
    plt.show()


#==================================================================================================
# Programa principal
#==================================================================================================

# Humanoid
#--------------------------------------------------------------------------------------------------
form_cluster_to_df_test('Humanoid',1,30)
load_from_cluster_interesting_policies('Humanoid',1)
validate_policies('Humanoid',1,10000) # tiempos 12:20;13:48;3:24;3:18;13:42
plot_per_policy_type('Humanoid',1,list(range(10,10100,100)))

form_cluster_to_df_test('Humanoid',2,30)
load_from_cluster_interesting_policies('Humanoid',2)
validate_policies('Humanoid',2,10000)
plot_per_policy_type('Humanoid',2,list(range(10,10100,100)))

# InvertedDoublePendulum
#--------------------------------------------------------------------------------------------------
form_cluster_to_df_test('InvertedDoublePendulum',1,1)
load_from_cluster_interesting_policies('InvertedDoublePendulum',1)
validate_policies('InvertedDoublePendulum',1,10000) 
plot_per_policy_type('InvertedDoublePendulum',1,list(range(100,10100,100)))

form_cluster_to_df_test('InvertedDoublePendulum',2,1)
load_from_cluster_interesting_policies('InvertedDoublePendulum',2)
validate_policies('InvertedDoublePendulum',2,10000)#times 39:47;32:47;0:27;39:48;25:21
plot_per_policy_type('InvertedDoublePendulum',2,list(range(10,10100,100)))

# Ant
#--------------------------------------------------------------------------------------------------
form_cluster_to_df_test('Ant',1,4)
load_from_cluster_interesting_policies('Ant',1)
validate_policies('Ant',1,10000) 
plot_per_policy_type('Ant',1,list(range(100,10100,100)))

form_cluster_to_df_test('Ant',2,4)
load_from_cluster_interesting_policies('Ant',2)
validate_policies('Ant',2,10000)
plot_per_policy_type('Ant',2,list(range(10,10100,100)))

