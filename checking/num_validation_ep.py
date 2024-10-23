'''
El numero de episodios que usamos para la validacion debe ser lo suficientemente grande como para garantizar que el mean_reward
nos proporciona una medida de calidad precisa/real de cada politica. Aqui analizamos si 100 episodios son suficientes para
garantizar la maxima precision de validacion de las politicas en cada environment. 

Concretamente, se analiza el numero de episodios necesarios para diferentes tipos de politicas, que definimos como sigue:
- Good: politica visitada con maximo mean_reward.
- Poor: politica visitada con mean_reward mas cercano la media de los mean_rewards de todas las politicas.
- Bad: politica visitada con minimo mean_reward.
- Deterministic: politica visitada con minima varianza en los rewards de los 100 episodios registrados.
- Sthocastic: politica visitada con maxima varianza en los rewards de los 100 episodios registrados.

Para que 100 sea suficiente, las curvas y el rango del intervalos de confianza deben converger antes de 100.

Nota.- realmente el numero de episodios de validacion que deberiamos guardar tiene que ser mayor que el numero de episodios 
necesarios para maxima precision, ya que los episodios que usemos para identificar la mejor politica tienen que se diferentes
a los que usemos para calcular el reward real de la politica seleccionada (asociado con el experimento KnowingSingleProcess).
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.our_library import UtilsDataFrame, UtilsFigure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Estimar test reward con boostrap usando diferente numero de episodios
def n_test_ep_for_max_acc(ep_test_rewards,policy_type,pos):
    sample_sizes=list(range(10,101,1))
    all_mean=[]
    all_q05=[]
    all_q95=[]

    for i in sample_sizes:

        mean,q05,q95=UtilsFigure.bootstrap_mean_and_confidence_interval(ep_test_rewards[:i])

        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    # Grafica
    ax=plt.subplot(1,5,pos)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    ax.fill_between(sample_sizes,all_q05,all_q95, alpha=.2, linewidth=0)
    plt.plot(sample_sizes, all_mean, linewidth=1)

    ax.set_title(policy_type+' policy')
    ax.set_xlabel('Sample size (number of episodes)')
    if pos==1:
        ax.set_ylabel('Reward confidence interval with boostrap')

# Dibujar curvas para identificar numero de episodios de validacion en que el test reward converge para diferentes tipos de politicas
def plot_per_policy_type(env_name,seed):

    # Leer bases de datos
    df_test=pd.read_parquet('results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_test['std_rewards']=[np.std(UtilsDataFrame.compress_decompress_list(i,compress=False)) for i in list(df_test['ep_test_rewards'])]

    # Encontrar posiciones de politicas de los tres tipos: mala, regular y buena
    indx_good=df_test['mean_reward'].idxmax()
    indx_poor=np.argmin(abs(np.array(df_test['mean_reward'])- df_test['mean_reward'].mean()))
    indx_bad=df_test['mean_reward'].idxmin()
    indx_det=df_test['std_rewards'].idxmin()
    indx_stoc=df_test['std_rewards'].idxmax()

    # Usar los 100 test rewards almacenados para cada politica y dibujar curva para observar convergencia
    fig=plt.figure(figsize=[20,4])
    plt.subplots_adjust(left=0.06,bottom=0.11,right=0.98,top=0.88,wspace=0.33,hspace=0.2)
    n_test_ep_for_max_acc(UtilsDataFrame.compress_decompress_list(list(df_test['ep_test_rewards'])[indx_good],compress=False),'Good',1)
    n_test_ep_for_max_acc(UtilsDataFrame.compress_decompress_list(list(df_test['ep_test_rewards'])[indx_poor],compress=False),'Poor',2)
    n_test_ep_for_max_acc(UtilsDataFrame.compress_decompress_list(list(df_test['ep_test_rewards'])[indx_bad],compress=False),'Bad',3)
    n_test_ep_for_max_acc(UtilsDataFrame.compress_decompress_list(list(df_test['ep_test_rewards'])[indx_det],compress=False),'Deterministic',4)
    n_test_ep_for_max_acc(UtilsDataFrame.compress_decompress_list(list(df_test['ep_test_rewards'])[indx_stoc],compress=False),'Stochastic',5)

    plt.savefig('checking/results/num_validation_ep/'+str(env_name)+str(seed)+'.pdf')
    plt.show()

# Programa principal
plot_per_policy_type('InvertedDoublePendulum',10)
plot_per_policy_type('InvertedDoublePendulum',11)
plot_per_policy_type('Ant',1)
plot_per_policy_type('Ant',2)
plot_per_policy_type('Humanoid',1)
plot_per_policy_type('Ant',2)



