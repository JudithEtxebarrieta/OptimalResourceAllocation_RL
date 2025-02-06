'''
Encontrando el tamaño del conjunto de validacion que representa la "situacion ideal" o "ground truth" (proporciona la validacion de 
maxima precision) para los diferentes environments: InvertedDoublePendulum, Ant y Humanoid.

Observaciones en ejecucion del cluster:
- El proceso de guardar las politicas parece que lleva mas tiempo que guardar la validacion de la politica.
- La politica ocupa mas espacio (memoria) que la validacion (al menos el vector encriptado de 100 episodios).
- Como la carpeta de todas las politicas ocupa mucho espacio, para copiarla de Hipatia lleva mucho tiempo. Por eso, primero solo copio los df_test,
luego identifico con df_test las politicas que me interesan (e.g., good, mean, bad, deterministic and stochastic) usando los 100 datos de validacion que 
he guardado para cada una, y finalmente solo copio de Hipatia esas politicas identificadas.

Experimentos realizados:
- Precision de estimacion de reward: al principio hablamos de encontrar el numero de episodios necesarios para estimar de manera precisa la
calidad de cada politica, i.e. mean rewards realista. Por eso, como dependiendo de la politica este numero de episodios puede variar, primero
seleccionamos un conjunto de 5 politicas por proceso (good, mena, bad, deterministic y stochastic) basandonos en los 100 ep de validacion
almacenados, y luego unicamente validamos esas politicas en 10000 ep para estimar los intervalos de confianza de los reward medios 
aplicando boostrap a diferentes tamaños de muestra.
- Precision de comparacion: despues nos dimos cuenta que realmente para representar el "ground truth" no nos interesa tanto conocer el mean_reward
real de cada politica, sino que es suficiente con dar con el numero de episodios que nos permite ordenar correctamente la secuencia de politicas
que visitamos en un proceso. Por ello, seria mas interesante comparar la estabilidad de los ranking de multiples curvas boostrap como las de arriba.
Para escoger las politicas que representaremos en esa grafica, tiene que ser un conjunto que represente apropiadamente toda la secuencia de politicas
del proceso. Ahora las escojo con 5 calidades diferentes y 5 estocasticidades diferentes, pero es un conjunto muy variado, y no representa correctamente
la secuencia, por eso son tan faciles de distinguir las politicas y tan estables los rankings. Para mejorar esto, deberiamos probar escogiendo:
    1) 10 politicas "equidistantes" (en cuanto a orden de visita) en la secuencia de politicas visitadas.
    2) 10 politicas o todas de la subsecuencia formada por la mejor observada hasta ahora.
Despues, en lugar de dibujar las curvas de diferente color solo para distinguirlas, dibujarlas en color degradado para distinguir de la curva 
asociada a la politica "mas reciente" (oscuro) a la "mas antigua" (claro).

'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.our_library import UtilsDataFrame, UtilsFigure, PPOLearner, PolicyValidation
import pandas as pd
import numpy as np
import subprocess
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm


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
    df_test['std_reward']=[np.std(UtilsDataFrame.compress_decompress_list(i,compress=False)) for i in list(df_test['ep_test_rewards'])]

    # Encontrar y guardar posiciones de politicas de los 5 tipos: mala, regular, buena, determinista y estocastica.
    indx_good=df_test['mean_reward'].idxmax()
    indx_mean=np.argmin(abs(np.array(df_test['mean_reward'])- df_test['mean_reward'].mean()))
    indx_bad=df_test['mean_reward'].idxmin()
    indx_det=df_test['std_reward'].idxmin()
    indx_stoc=df_test['std_reward'].idxmax()

    os.makedirs('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed))
    id_policies=pd.DataFrame({'policy_type':['Good','Mean','Bad','Deterministic','Stochastic'],'id_policy':[indx_good+1,indx_mean+1,indx_bad+1,indx_det+1,indx_stoc+1],'ep_test_rewards':[None]*5})

    # Encontrar y guardar posiciones de politicas mas cercanas a los 5 cuartiles de los vectores mean_reward y std_reward
    mean_perc=[np.quantile(list(df_test['mean_reward']),i) for i in [0,.25,.5,.75,1]]
    std_perc=[np.quantile(list(df_test['std_reward']),i) for i in [0,.25,.5,.75,1]]

    for i in range(5):
        id_policies.loc[len(id_policies)]=['Quality'+str(i),np.argmin(abs(np.array(df_test['mean_reward'])- mean_perc[i]))+1,None]
        id_policies.loc[len(id_policies)]=['Stochasticity'+str(i),np.argmin(abs(np.array(df_test['std_reward'])- std_perc[i]))+1,None]

    pd.DataFrame(id_policies).to_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/id_policies.csv')

    # Cargar del cluster unicamente las politicas de interes.
    for id_policy in id_policies['id_policy']:
        cluster_path = f"/home/jechevarrieta/results_fakeHome/policies_{env_name}_seed{seed}/policy{id_policy}.zip"
        pc_path="checking/results/validation_ep_set/"+env_name+"/policies_seed"+str(seed)+'/'
        command = ["rsync", "-r", "-avHPe", "sshpass -p R5ShrSjIxLHX ssh -p6556", f"jechevarrieta@hpc.bcamath.org:{cluster_path}", pc_path]
        subprocess.run(command)

def load_from_cluster_representative_policy_subsequence(env_name,seed):

    # Leer bases de datos.
    df_test=pd.read_parquet('checking/results/validation_ep_set/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_test['std_reward']=[np.std(UtilsDataFrame.compress_decompress_list(i,compress=False)) for i in list(df_test['ep_test_rewards'])]

    # Subsecuencia 1: 10 politicas equidistantes de la secuencia completa.
    indx_seq1=[round(i) for i in np.arange(1,df_test['n_policy'].max(),(df_test['n_policy'].max()-1)/10)]

    id_policies=pd.DataFrame({'policy_type':np.arange(1,11),'id_policy':indx_seq1,'ep_test_rewards':[None]*10})
    pd.DataFrame(id_policies).to_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/id_policies_subseq1.csv')

    # Subsecuencia 2: 10 politicas equidistantes de la subsecuencia formada por la mejor polica "on the fly".
    all_indx_seq=[]
    for i in range(df_test.shape[0]):
        indx_max=df_test['mean_reward'][df_test['n_policy']<=i+1].idxmax()
        if df_test['n_policy'][indx_max] not in all_indx_seq:
            all_indx_seq.append(df_test['n_policy'][indx_max])

    if len(all_indx_seq)>10:
        indx_seq2=[all_indx_seq[round(i)] for i in np.arange(0,len(all_indx_seq),len(all_indx_seq)/10)]

    id_policies=pd.DataFrame({'policy_type':np.arange(1,11),'id_policy':indx_seq2,'ep_test_rewards':[None]*10})
    pd.DataFrame(id_policies).to_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/id_policies_subseq2.csv')

    # Cargar del cluster unicamente las politicas de interes.
    for id_policy in indx_seq1+indx_seq2:
        cluster_path = f"/home/jechevarrieta/results_fakeHome/policies_{env_name}_seed{seed}/policy{id_policy}.zip"
        pc_path="checking/results/validation_ep_set/"+env_name+"/policies_seed"+str(seed)+'/'
        command = ["rsync", "-r", "-avHPe", "sshpass -p R5ShrSjIxLHX ssh -p6556", f"jechevarrieta@hpc.bcamath.org:{cluster_path}", pc_path]
        subprocess.run(command)
        
# Conseguir conjunto de validacion de las politicas de interes.
def validate_policies(env_name,seed,id_policies_csv,n_eval_episodes,n_processes):

    '''
    Para ejecutar esta funcion hay que hacerlo desde el entorno vistual py39venv, porque al guardar la politica en el cluster 
    se ha usado un entorno virtual con la misma version de python de py39venv. De lo contrario, da error por no considerar 
    mismas versiones de ciertas librerias para guardar y cargar la politica.
    '''

    id_policies=pd.read_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/'+id_policies_csv+'.csv')

    for i in range(id_policies.shape[0]):
        id_policy=id_policies['id_policy'][i]

        # Cargar politica actual
        policy=PPOLearner.load_policy('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/policy'+str(id_policy)+'.zip')

        # Lista de rewards por episodio (considerar un maximo de episodios)
        _,_,all_ep_reward,_=PolicyValidation.parallel_evaluate(policy,env_name+"-v4",n_eval_episodes,n_processes)

        id_policies['ep_test_rewards'][i]=UtilsDataFrame.compress_decompress_list(all_ep_reward)
    
    id_policies.to_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/'+id_policies_csv+'.csv')

# Grafica 1: analizando numero de episodios necesarios para maxima precison de validacion.
def boostrap_validation_accuracy(ep_test_rewards,policy_type,pos,sample_sizes):

    # Boostrap para cada tamaño de conjunto de validacion.
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

def plot_validation_accuracy(env_name,seed,list_n_eval_ep):

    # Leer datos necesarios para dibujar las graficas.
    id_policies=pd.read_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/id_policies.csv')

    fig=plt.figure(figsize=[20,8])
    plt.subplots_adjust(left=0.06,bottom=0.11,right=0.98,top=0.88,wspace=0.33,hspace=0.2)

    for i in tqdm(range(0,5)):
        ep_test_rewards=UtilsDataFrame.compress_decompress_list(id_policies['ep_test_rewards'][i],compress=False)
        boostrap_validation_accuracy(ep_test_rewards,id_policies['policy_type'][i],i+1,list_n_eval_ep)

    plt.savefig('checking/results/validation_ep_set/'+str(env_name)+str(seed)+'_validation.pdf')
    plt.show()

# Grafica 2: anaizando numero de episodios necesarios para maxima precison de comparacion.
def boostrap_comparison_accuracy(ep_test_rewards,sample_sizes):

    # Boostrap para cada tamaño de conjunto de validacion.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    for i in sample_sizes:

        mean,q05,q95=UtilsFigure.bootstrap_mean_and_confidence_interval(ep_test_rewards[:i])

        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean, all_q05, all_q95

def extract_data(env_name,seed,id_policies_csv,list_n_eval_ep):
    # Leer datos necesarios para dibujar las graficas.
    id_policies=pd.read_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/'+id_policies_csv+'.csv')

    labels=[]
    list_all_mean=[]
    list_all_q05=[]
    list_all_q95=[]

    if id_policies_csv=='id_policies':
        rows=range(5,15)
        output_csv='data_plot'
    elif id_policies_csv=='id_policies_subseq1':
        rows=range(id_policies.shape[0])
        output_csv='data_plot_subseq1'
    elif id_policies_csv=='id_policies_subseq2':
        rows=range(id_policies.shape[0])
        output_csv='data_plot_subseq2'
    for i in tqdm(rows):
        ep_test_rewards=UtilsDataFrame.compress_decompress_list(id_policies['ep_test_rewards'][i],compress=False)
        all_mean,all_q05,all_q95=boostrap_comparison_accuracy(ep_test_rewards,list_n_eval_ep)
        labels.append(id_policies['policy_type'][i])
        list_all_mean.append(UtilsDataFrame.compress_decompress_list(all_mean))
        list_all_q05.append(UtilsDataFrame.compress_decompress_list(all_q05))
        list_all_q95.append(UtilsDataFrame.compress_decompress_list(all_q95))

    data_plot={'labels':labels,'all_mean':list_all_mean,'all_q05':list_all_q05,'all_q95':list_all_q95}
    data_plot=pd.DataFrame(data_plot)
    data_plot=data_plot.sort_values(by='labels', ascending=True)
    data_plot.to_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/'+output_csv+'.csv')


def plot_comparison_accuracy(env_name,seed,data_plot_csv,list_n_eval_ep):

    if data_plot_csv=='data_plot':
        default_colors=list(mcolors.TABLEAU_COLORS.keys())
    else:
        default_colors = [cm.get_cmap('Greens')(i / 9) for i in range(10)]

    data_plot=pd.read_csv('checking/results/validation_ep_set/'+env_name+'/policies_seed'+str(seed)+'/'+data_plot_csv+'.csv')
    
    fig=plt.figure(figsize=[10,7])
    plt.subplots_adjust(left=0.07,bottom=0.09,right=0.96,top=0.97,wspace=0.43,hspace=0.2)

    mean_matrix=[]
    colors=[]
    labels=[]
    for i in range(len(data_plot)):

        all_q05=UtilsDataFrame.compress_decompress_list(data_plot['all_q05'][i],compress=False)
        all_q95=UtilsDataFrame.compress_decompress_list(data_plot['all_q95'][i],compress=False)
        all_mean=UtilsDataFrame.compress_decompress_list(data_plot['all_mean'][i],compress=False)
        label=str(data_plot['labels'][i])

        mean_matrix.append(all_mean)
        colors.append(default_colors[i])
        labels.append(label)

        ax=plt.subplot(221)
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)
        ax.fill_between(list_n_eval_ep,all_q05,all_q95, alpha=.2, linewidth=0,color=default_colors[i])
        plt.plot(list_n_eval_ep, all_mean, linewidth=1,color=default_colors[i])
        ax.set_xlabel('Sample size (number of episodes)')
        ax.set_ylabel('Boostrap confidence interval range')

        ax=plt.subplot(223)
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)
        plt.plot(list_n_eval_ep, abs(np.array(all_q95)-np.array(all_q05)), linewidth=1,label=label,color=default_colors[i])
        ax.set_xlabel('Sample size (number of episodes)')
        ax.set_ylabel('Boostrap confidence interval range')
        ax.legend(title="Policy",fontsize=8,bbox_to_anchor=(1.5, 1, 0, 0))



    ax=plt.subplot(222)
    mean_matrix=np.array(mean_matrix).T
    ranking_matrix=np.zeros(mean_matrix.shape, dtype='<U100')
    for i in range(len(mean_matrix)):
        argsort=np.argsort(mean_matrix[i],)[::-1]
        for j in range(len(argsort)):
            ranking_matrix[i][j]=labels[argsort[j]]

    
    ranking_matrix=ranking_matrix.T

    # Convertir los valores categóricos a números para visualización
    category_map = {k: v for v, k in enumerate(labels)}
    numerical_data = np.vectorize(category_map.get)(ranking_matrix)

    # Dibujar el mapa de calor con color bar discreta
    sns.heatmap(numerical_data, cmap=colors, cbar=False, linewidths=0.5, cbar_kws={"ticks": range(len(labels))})

    ax.set_xlabel('Sample size')
    ax.set_ylabel('From best (top) to worts (bottom)')
    ax.set_title('')
    ax.set_xticks([1,21,41,61,81])
    ax.set_xticklabels([100,2100,4100,6100,8100],rotation=0)


    if data_plot_csv=='data_plot':
        pdf_name='comparison'
    elif data_plot_csv=='data_plot_subseq1':
        pdf_name='comparison_subseq1'
    elif data_plot_csv=='data_plot_subseq2':
        pdf_name='comparison_subseq2'
    plt.savefig('checking/results/validation_ep_set/'+str(env_name)+str(seed)+'_'+pdf_name+'.pdf')
    plt.show()


#==================================================================================================
# Programa principal
#==================================================================================================

'''
Las lineas de validacion las estoy ejecutando en el cluster Hipatia
'''

# InvertedDoublePendulum
#--------------------------------------------------------------------------------------------------
# Cargar del cluster df_test.
form_cluster_to_df_test('InvertedDoublePendulum',1,1)

# Cargar y validar politicas heterogeneas.
load_from_cluster_interesting_policies('InvertedDoublePendulum',1)
validate_policies('InvertedDoublePendulum',1,'id_policies',10000,6) 

# Cargar y validar subsecuencia de politicas.
load_from_cluster_representative_policy_subsequence('InvertedDoublePendulum',1)
validate_policies('InvertedDoublePendulum',1,'id_policies_subseq1',10000,6) 
validate_policies('InvertedDoublePendulum',1,'id_policies_subseq2',10000,6) 

# Analisis grafico.
plot_validation_accuracy('InvertedDoublePendulum',1,list(range(100,10100,100)))

extract_data('InvertedDoublePendulum',1,'id_policies',list(range(100,10100,100)))
plot_comparison_accuracy('InvertedDoublePendulum',1,'data_plot',list(range(100,10100,100)))

extract_data('InvertedDoublePendulum',1,'id_policies_subseq1',list(range(100,10100,100)))
plot_comparison_accuracy('InvertedDoublePendulum',1,'data_plot_subseq1',list(range(100,10100,100)))

extract_data('InvertedDoublePendulum',1,'id_policies_subseq2',list(range(100,10100,100)))
plot_comparison_accuracy('InvertedDoublePendulum',1,'data_plot_subseq2',list(range(100,10100,100)))

# Ant
#--------------------------------------------------------------------------------------------------
# Cargar del cluster df_test.
form_cluster_to_df_test('Ant',1,4)

# Cargar y validar politicas heterogeneas.
load_from_cluster_interesting_policies('Ant',1)
validate_policies('Ant',1,'id_policies',10000,6) 

# Cargar y validar subsecuencia de politicas.
load_from_cluster_representative_policy_subsequence('Ant',1)
validate_policies('Ant',1,'id_policies_subseq1',10000,6) 
validate_policies('Ant',1,'id_policies_subseq2',10000,6) 

# Analisis grafico.
plot_validation_accuracy('Ant',1,list(range(100,10100,100)))

extract_data('Ant',1,'id_policies',list(range(100,10100,100)))
plot_comparison_accuracy('Ant',1,'data_plot',list(range(100,10100,100)))

extract_data('Ant',1,'id_policies_subseq1',list(range(100,10100,100)))
plot_comparison_accuracy('Ant',1,'data_plot_subseq1',list(range(100,10100,100)))

extract_data('Ant',1,'id_policies_subseq2',list(range(100,10100,100)))
plot_comparison_accuracy('Ant',1,'data_plot_subseq2',list(range(100,10100,100)))

# Humanoid
#--------------------------------------------------------------------------------------------------
# Cargar del cluster df_test.
form_cluster_to_df_test('Humanoid',1,30)

# Cargar y validar politicas heterogeneas.
load_from_cluster_interesting_policies('Humanoid',1)
validate_policies('Humanoid',1,'id_policies',10000,6) 

# Cargar y validar subsecuencia de politicas.
load_from_cluster_representative_policy_subsequence('Humanoid',1)
validate_policies('Humanoid',1,'id_policies_subseq1',10000,6)
validate_policies('Humanoid',1,'id_policies_subseq2',10000,6)

# Analisis grafico.
plot_validation_accuracy('Humanoid',1,list(range(100,10100,100)))

extract_data('Humanoid',1,'id_policies',list(range(100,10100,100)))
plot_comparison_accuracy('Humanoid',1,'data_plot',list(range(100,10100,100)))

extract_data('Humanoid',1,'id_policies_subseq2',list(range(100,10100,100)))
plot_comparison_accuracy('Humanoid',1,'data_plot_subseq2',list(range(100,10100,100)))










