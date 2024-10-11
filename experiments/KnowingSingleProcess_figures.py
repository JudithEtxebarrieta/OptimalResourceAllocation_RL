'''
OBSERVACIONES:

- InvertedDoublePendulun no parece que tiene multiples optimos locales, y en general en cualquier environmet?
Yo creo que la segunda condicion deberia de ser que diferentes procesos tengan diferentes ritmos de convergencia. El PPO tiene
integrado un mecanismo que si ve que la politica no mejora introduce mas aleatoriedad, por eso, tarde o temprano siempre se
alcanza el mismo optimo.

- Quedarnos con la mejor politica observada en lugar de la ultima implica una ligera mejora (amarilla mejor que negra). Pero 
el tiempo extra de test que hay que invertir hace que esa mejora no componse (resto de curvas a la amarilla peoares que la negra).

- Entonces, podemos obtener la informacion de cual es la mejor politica observada reciclando el train reward?



'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.colors import PowerNorm

from our_library import  UtilsDataFrame

#==================================================================================================
# GRAFICA 1: Comprobar graficamente que se cumplen los requisitos del environment
#==================================================================================================
def plot_environment_requirements(list_seeds,path,env_name,n_policies,conv_policies,reward_threshold):

    fig=plt.figure(figsize=[10,4])
    plt.subplots_adjust(left=0.08,bottom=0.27,right=0.97,top=0.84,wspace=0.39,hspace=0.2)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: learning-curves (cortadas en n_policy relevantes)
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(131)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    all_x_conv=[]
    all_y_max=[]

    for seed in list_seeds:
        df=pd.read_parquet(path+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')

        x=list(range(1,n_policies))
        y=[]
        y_max=-np.Inf
        for i in x:
            new_y=df[df['n_policy']<=i]['mean_reward'].max()
            y.append(new_y)
            if new_y>y_max:
                y_max=new_y
                if new_y<=reward_threshold:
                    x_conv=i

        plt.plot(x, y, linewidth=1,color='grey')
        all_x_conv.append(x_conv)
        all_y_max.append(y_max)

    plt.axhline(y=reward_threshold,color='black', linestyle='--')
    plt.axvline(x=np.mean(all_x_conv),color='red', linestyle='--')
    plt.xlim(0,conv_policies)

    ax.set_xlabel("n_policies",fontsize=10)
    ax.set_ylabel("Best test reward",fontsize=10)
    ax.set_title('Learning-curve: Best policy found\n',fontsize=10)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 2: condicion 1 (tiempo suficiente como para ejecutar multiples procesos hasta "convergencia promedio")
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(132)
    ax.grid(True, axis='y',linestyle='--', linewidth=0.8,alpha=0.2)
    ax.bar(range(1,len(list_seeds)+1),all_x_conv,width=0.7,color='grey')
    plt.axhline(y=n_policies,color='black', linestyle='--')
    plt.axhline(y=np.mean(all_x_conv),color='red', linestyle='--')
    ax.set_xlabel("process",fontsize=10) #\nReformulated condition 1:\nbudget is significantly longer\nthan average convergence time
    ax.set_ylabel("convergence n_policies",fontsize=10)
    y_perc=[round(i/n_policies*100)for i in all_x_conv]
    for i, total in enumerate(y_perc):
        ax.text(i+1, all_x_conv[i] + n_policies*0.05, str(total)+' %',
                ha = 'center', weight = 'bold', color = 'black')
    ax.set_title('Condition 1:\nLarge enough budget\n',fontsize=10)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 3: condicion 2 (comprobar si tenemos multiples optimos locales)
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(133)
    ax.grid(True, axis='y',linestyle='--', linewidth=0.8,alpha=0.2)
    ax.bar(range(1,len(list_seeds)+1),all_y_max,width=0.7,color='grey')
    plt.ylim(min(all_y_max)-np.std(all_y_max), max(all_y_max)+np.std(all_y_max))
    ax.set_xlabel("process",fontsize=10) # \n Reformulated condition 2:\nprocesses converge at\nsignificantly different speeds
    ax.set_ylabel("convergence test reward",fontsize=10)
    ax.set_title('Condition 2:\nConvergence does not imply optimality',fontsize=10,color='red')

    plt.savefig(path+'/conditions_'+str(env_name)+'.pdf')
    plt.show()
    plt.close()

#==================================================================================================
# GRAFICA 2: validar todas las politicas durante el entrenamiento con numero de episodios test constante
#==================================================================================================
def plot_leraning_curves_constant_n_test_episodes(x,list_n_test_ep,path,seed,list_eval_freq=[1]):

    random.seed(0)
    eval_freq_with_train=False

    df_test=pd.read_parquet(path+'/df_test_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    df_train=pd.read_parquet(path+'/df_train_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    df_test=df_test[df_test['n_policy']<=max(x)//2048]
    df_train=df_train[df_train['n_policy']<=max(x)//2048-1]

    n_policies=list(df_test['n_policy'])

    fig=plt.figure(figsize=[10,5])
    plt.subplots_adjust(left=0.08,bottom=0.152,right=0.7,top=0.9,wspace=0.39,hspace=0.2)

    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    # Dibujar curva sin test durante el proceso (ultima politica observada)
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(df_test[df_test['n_policy']==last_policy]['mean_reward'])
    plt.plot(x, y, linewidth=1,color='black',label='n_test_ep=0 (last visited policy)')

    # Dibujar curva con mejor test pero sin contar el extra de tiempo
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(df_test[df_test['n_policy']<=last_policy]['mean_reward'].max())
    plt.plot(x, y, linewidth=1,color='yellow',label='eval_freq=1  n_test_ep=100 (counted as 0)')

    # Resto de curvas
    if len(list_n_test_ep)>1:
        for_list=list_n_test_ep
    else:
        for_list=list_eval_freq
    
    for i in for_list:

        
        if len(list_n_test_ep)>1:
            n_test_ep=i
            eval_freq=list_eval_freq[0]
            label=n_test_ep
        else:
            eval_freq=i
            n_test_ep=list_n_test_ep[0]
            label=eval_freq


        current_mean_reward=[]
        current_n_test_timesteps=[]
        cumulative_n_test_timesteps=0


        for policy in n_policies:
            
            if type(eval_freq) is int:
                condition=policy%eval_freq==0
            else:
                label,change,threshold=eval_freq
                condition=change[policy-1]>threshold
                eval_freq_with_train=True


            if condition:
                ep_test_rewards_compressed=list(df_test[df_test['n_policy']==policy]['ep_test_rewards'])
                ep_test_len_compressed=list(df_test[df_test['n_policy']==policy]['ep_test_len'])

                ep_test_rewards=UtilsDataFrame.compress_decompress_list(ep_test_rewards_compressed[0],compress=False)
                ep_test_len=UtilsDataFrame.compress_decompress_list(ep_test_len_compressed[0],compress=False)

                current_mean_reward.append(np.mean([np.mean(random.sample(ep_test_rewards,n_test_ep)) for i in range(100)]))
                cumulative_n_test_timesteps+=sum(random.sample(ep_test_len,n_test_ep))
            else:
                current_mean_reward.append(0)
            current_n_test_timesteps.append(cumulative_n_test_timesteps)

        df_test['current_mean_reward']=current_mean_reward
        df_test['current_n_test_timesteps']=current_n_test_timesteps

        total_timesteps=np.array(df_test['current_n_test_timesteps'])+np.array(df_train['n_train_timesteps'])
        df_test['total_timesteps']=total_timesteps
        y=[]
        for timesteps in x:
            indx_max=df_test[df_test['total_timesteps']<=timesteps]['current_mean_reward'].idxmax()
            y.append(df_test['mean_reward'][indx_max])


        plt.plot(x, y, linewidth=1,label=label)

    ax.legend(title="",fontsize=8,bbox_to_anchor=(1, 1, 0, 0))

    ax.set_xlabel("Total time steps (train+test)\n \n(2048 train timesteps = training 1 policy = 1 trajectory)",fontsize=10)
    ax.set_ylabel("Best policy reward (in terms of max_n_test_ep)",fontsize=10)

    if len(list_n_test_ep)>1:
        ax.set_title('Learning-curve: best policy found\n Constant eval_freq: '+str(eval_freq)+' policy; Variable n_test_ep (legend)',fontsize=10)
        plt.savefig(path+'/learning_curve_variable_n_test_ep_constant_eval_freq'+str(eval_freq)+'_'+str(seed)+'.pdf')
    else:
        ax.set_title('Learning-curve: best policy found\n Variable eval_freq (legend); Constant n_test_ep: '+str(n_test_ep),fontsize=10)
        if eval_freq_with_train:
            plt.savefig(path+'/learning_curve_constant_n_test_ep'+str(n_test_ep)+'_train_eval_freq_'+str(seed)+'.pdf')
        else:
            plt.savefig(path+'/learning_curve_constant_n_test_ep'+str(n_test_ep)+'_variable_eval_freq_'+str(seed)+'.pdf')



    plt.show()
    plt.close()


#==================================================================================================
# GRAFICA 3: validar todas las politicas durante el entrenamiento con numero de episodios train constante
#==================================================================================================
# Funciones auxiliares
def ranking_from_argsort(argsort):
    ranking=[0]*len(argsort)
    ranking_pos=1
    for i in argsort:
        ranking[i]=ranking_pos
        ranking_pos+=1
    return ranking

def rank_labels_from_argsort(labels,argsort):
    new_labels=[]
    for i in argsort:
        new_labels.append(labels[i])
    return new_labels

def form_df_train_to_matrix_train_rewards(list_window_sizes,path,seed,max_train_timesteps,batch_size=None):

    df_train=pd.read_parquet(path+'/df_train_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    df_train['train_rewards']=[UtilsDataFrame.compress_decompress_list(i,compress=False) for i in list(df_train['train_rewards'])]
    df_train['train_ep_end']=[UtilsDataFrame.compress_decompress_list(i,compress=False) for i in list(df_train['train_ep_end'])]


    df_train=df_train[df_train['n_train_timesteps']<=max_train_timesteps]
    n_policies=list(df_train['n_policy'])

    train_rewards=[]
    train_ep_end=[]
    for i in n_policies:
        train_rewards+=list(df_train[df_train['n_policy']==i]['train_rewards'])[0]
        train_ep_end+=list(df_train[df_train['n_policy']==i]['train_ep_end'])[0]

    n_train_timesteps=list(df_train['n_train_timesteps'])
    reward_matrix=[]

    # Añadido para guardar por batches de iteraciones los rewards de los episodes de train
    batch_ep_reward_matrix=[]
    num_ep_start_batch=[]
    first_iteration=True
    ##
    
    for stats_window_size in list_window_sizes:
        rewards=[]

        ep_rew_mean=[]
        last_i=0# Last episode end
        current_i=0# Current step

        for i in train_ep_end:
            if i:
                ep_rew_mean.append(sum(train_rewards[last_i:current_i]))
                last_i=current_i
            current_i+=1

            if current_i in n_train_timesteps:
                
                if len(ep_rew_mean)<stats_window_size:
                    rewards.append(np.mean(ep_rew_mean))
                else:
                    rewards.append(np.mean(ep_rew_mean[-stats_window_size:]))

                # Añadido para guardar por batches de iteraciones los rewards de los episodes de train
                if batch_size is not None and first_iteration:
                    if (current_i//min(n_train_timesteps))%batch_size==0:
                        num_ep_start_batch.append(len(ep_rew_mean))
                ##
        first_iteration=False
                    
        reward_matrix.append(rewards)

     # Añadido para guardar por batches de iteraciones los rewards de los episodes de train
    for i in range(len(num_ep_start_batch)):
        if i==0:
            batch_ep_reward_matrix.append(ep_rew_mean[:num_ep_start_batch[i]])
        else:
            batch_ep_reward_matrix.append(ep_rew_mean[num_ep_start_batch[i-1]:num_ep_start_batch[i]])
    ##

    if batch_size is None: 
        return reward_matrix
    else:       

        return reward_matrix,batch_ep_reward_matrix

def plot_leraning_curves_constant_n_train_episodes(x,list_window_sizes,path,seed):

    fig=plt.figure(figsize=[10,5])
    plt.subplots_adjust(left=0.08,bottom=0.152,right=0.73,top=0.9,wspace=0.39,hspace=0.2)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: learning-curves
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    train_reward_matrix=form_df_train_to_matrix_train_rewards(list_window_sizes,path,seed,max(x))

    df_test=pd.read_parquet(path+'/df_test_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    df_train=pd.read_parquet(path+'/df_train_InvertedDoublePendulum_seed'+str(seed)+'.parquet')

    y_matrix=[]
    labels=[]
    colors=[]
    default_colors=list(mcolors.TABLEAU_COLORS.keys())

    # Dibujar curva sin test durante el proceso (ultima politica observada)
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(float(df_test[df_test['n_policy']==last_policy]['mean_reward']))
    plt.plot(x, y, linewidth=1,color='black',label='0 (last visited policy)')
    y_matrix.append(y)
    labels.append(str(0))
    colors.append('black')

    # Resto de curvas
    for i in range(len(list_window_sizes)):
        y=[]
        for train_timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=train_timesteps]['n_policy'].max()
            best_policy=train_reward_matrix[i].index(max(train_reward_matrix[i][:last_policy]))
            y.append(float(df_test[df_test['n_policy']==best_policy+1]['mean_reward']))

        plt.plot(x, y, linewidth=1,label=list_window_sizes[i])
        y_matrix.append(y)
        labels.append(str(list_window_sizes[i]))
        colors.append(default_colors[i])

    ax.legend(title="n_train_ep",fontsize=8,bbox_to_anchor=(1.3, 1, 0, 0))
    ax.set_xlabel("Total train steps\n(no extra consumption of test steps in validation here)",fontsize=10)
    ax.set_ylabel("Best policy reward (in terms of max_n_test_ep)",fontsize=10)
    ax.set_title('Learning-curve: best policy found\n(iteratively validating policies with n_train_ep test episodes)',fontsize=10)

    plt.savefig(path+'/learning_curve_constant_n_train_ep'+str(seed)+'.pdf')
    plt.show()
    plt.close()

    #----------------------------------------------------------------------------------------------
    # GRAFICA 2: rankings de las learning-curves durante el entrenamiento
    #----------------------------------------------------------------------------------------------
    fig=plt.figure(figsize=[15,4])
    plt.subplots_adjust(left=0.05,bottom=0.3,right=0.97,top=0.9,wspace=0.39,hspace=0.35)

    ax=plt.subplot(111)
    y_matrix=np.array(y_matrix)
    y_matrix=y_matrix.T
    data=[]
    for i in y_matrix:
        data.append(rank_labels_from_argsort(labels,np.argsort(-np.array(i))))

    data=np.array(data)
    data= data.T

    # Convertir los valores categóricos a números para visualización
    category_map = {k: v for v, k in enumerate(labels)}
    numerical_data = np.vectorize(category_map.get)(data)

    # Dibujar el mapa de calor con color bar discreta
    sns.heatmap(numerical_data, cmap=colors, cbar=True, linewidths=0.5, cbar_kws={"ticks": range(len(labels))})

    # Ajustar la barra de color con las categorías
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(range(len(labels)))
    colorbar.set_ticklabels(labels)
    colorbar.set_label('n_train_ep')

    ax.set_xlabel('Train timesteps')
    ax.set_xticks(range(0,len(x),10))
    ax.set_xticklabels(range(min(x),max(x),int((max(x)-min(x))/10)))
    ax.set_ylabel('From best (top)\nto worts (bottom)')
    ax.set_title('The best n_train_ep during training')
    ax.set_yticklabels([],rotation=0)

    plt.savefig(path+'/learning_curve_constant_n_train_ep_rankings'+str(seed)+'.pdf')
    plt.show()
    plt.close()

#==================================================================================================
# GRAFICA 4: entendiendo relacion entre train y test reward
#==================================================================================================
def relationship_test_and_train_reward(list_window_sizes,path,seed):

    fig=plt.figure(figsize=[10,3])
    plt.subplots_adjust(left=0.14,bottom=0.152,right=0.97,top=0.9,wspace=0.39,hspace=0.2)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: comparacion de ranking de todas las politicas visitadas durante el entrenamiento,
    # definiendo los ranking a partir de las diferentes formas de validar las politicas (ya sea
    # usando el maximo numero de episodios test, como usando diferentes tamaños de ventana train).
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    df_train=pd.read_parquet(path+'/df_train_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    n_steps=int(df_train['n_train_timesteps'].min())

    df_test=pd.read_parquet(path+'/df_test_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    df_test=df_test[df_test['n_policy']<=500000//n_steps]

    ranking_matrix=[]
    y=[]

    # Test reward por politica con maximo n_test_ep
    ranking_matrix.append(ranking_from_argsort(np.argsort(list(df_test['mean_reward']))[::-1]))
    y.append('100 (test)')

    # Train reward por politica con diferentes n_train_ep constantes
    batch_size=25# para la siguiente grafica
    matrix_train_rewards,batch_ep_reward_matrix=form_df_train_to_matrix_train_rewards(list_window_sizes,path,seed,500000,batch_size)

    for i in range(len(matrix_train_rewards)):
        ranking_matrix.append(ranking_from_argsort(np.argsort(matrix_train_rewards[i])[::-1]))
        y.append(list_window_sizes[i])

    ranking_matrix=np.matrix(ranking_matrix)

    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    color=cm.get_cmap(color)
    color=color(np.linspace(0,1,ranking_matrix.shape[1]))
    color[:1, :]=np.array([14/256, 241/256, 249/256, 1])# Rojo (codigo rgb)
    color = ListedColormap(color)

    ax = sns.heatmap(ranking_matrix, cmap=color,linewidths=.5, linecolor='lightgray')

    colorbar=ax.collections[0].colorbar
    colorbar.set_label('Ranking position')
    colorbar.set_ticks(range(0,ranking_matrix.shape[1],25))
    colorbar.set_ticklabels(range(1,ranking_matrix.shape[1]+1,25))

    ax.set_xlabel('n_policy')
    ax.set_xticks(range(0,ranking_matrix.shape[1],10))
    ax.set_xticklabels(range(1,ranking_matrix.shape[1]+1,10),rotation=0)
    ax.set_ylabel('n_ep')
    ax.set_title('Comparing rankings depending on accuracy')
    ax.set_yticks(np.arange(0.5,len(y)+0.5,1))
    ax.set_yticklabels(y,rotation=0)

    plt.savefig(path+'/comparison_test_train_reward_overall_ranking'+str(seed)+'.pdf')
    plt.show()
    plt.close()

    #----------------------------------------------------------------------------------------------
    # GRAFICA 2: misma grafica anterior pero dividida en batches de politicas, para ver si 
    # el ranking mas parecido al primero se da con diferenetes tamaños de ventana en diferentes
    # etapas del entrenamiento. Además se añaden por batches los trozos correspondientes a 
    # la learning-curve por defecto y la curva de train reward por episodio train recolectado.
    #----------------------------------------------------------------------------------------------
    fig=plt.figure(figsize=[15,5])
    plt.subplots_adjust(left=0.08,bottom=0.152,right=0.96,top=0.9,wspace=0.16,hspace=0.2)

    def ranking_error(ranking,best_argsort):
        ranking_error=[]
        for i in best_argsort:
            ranking_error.append(abs((i+1)-ranking[i]))
        return ranking_error
    
    def reorder_matrix_by_first_row(matrix):
        order=np.argsort(matrix[0])
        new_matrix=[]
        for i in matrix:
            new_matrix.append(rank_labels_from_argsort(i,order))
        return new_matrix

    matrix_train_rewards=np.array(matrix_train_rewards)


    # LINEA 3: comparacion de errores de ordenacion de politicas por batch
    #----------------------------------------------------------------------------------------------
    n_batch=1
    matrix_batch_mean_rewards=[]
    for i in range(1,df_test.shape[0]+1):
        if i%batch_size==0:

            ranking_matrix=[]
            y=[]

            # Test reward por politica con maximo n_test_ep
            mean_rewards=list(df_test.iloc[:i]['mean_reward'])
            ranking_matrix.append(ranking_from_argsort(np.argsort(mean_rewards[-batch_size:])[::-1]))
            y.append('100 (test)')


            # Train reward por politica con diferentes n_train_ep constantes
            batch_matrix_train_rewards=matrix_train_rewards[:,(i-batch_size+1):(i+1)]
            for i in range(len(batch_matrix_train_rewards)):
                ranking_matrix.append(ranking_from_argsort(np.argsort(batch_matrix_train_rewards[i])[::-1]))
                y.append(list_window_sizes[i])

            # Reordenar matriz de acuerdo a orden de primera fila
            ranking_matrix=reorder_matrix_by_first_row(ranking_matrix)

            # Definir matriz de error de posiciones
            pos_err_matrix=[]
            best_argsort=np.argsort(ranking_matrix[0])

            for ranking in ranking_matrix:
                new_ranking_err=ranking_error(ranking,best_argsort)
                norm_ranking_err=[i/max(new_ranking_err) for i in new_ranking_err]
                pos_err_matrix.append(norm_ranking_err)
                
            # Para dibujar despues los trozos de learning-curve asociados a las politicas de cada batch
            mean_rewards=list(mean_rewards[-batch_size:])
            matrix_batch_mean_rewards.append(mean_rewards)
            
            # Dibujar ahora
            ax=plt.subplot(3,df_test.shape[0]//batch_size,2*(df_test.shape[0]//batch_size)+n_batch)
            pos_err_matrix=np.matrix(pos_err_matrix)

            color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)

            if n_batch==df_test.shape[0]//batch_size:
                ax = sns.heatmap(pos_err_matrix, cmap=color,linewidths=.5, linecolor='lightgray')
                colorbar=ax.collections[0].colorbar
                colorbar.set_label('Normaliced ranking\n position error')
                ax.set_yticklabels([])


            elif n_batch==1:
                ax = sns.heatmap(pos_err_matrix, cmap=color,linewidths=.5, linecolor='lightgray',cbar=False)
                ax.set_ylabel('n_train_ep')
                ax.set_yticks(np.arange(0.5,len(y)+0.5,1))
                ax.set_yticklabels(y,rotation=0)
            
            else:
                ax = sns.heatmap(pos_err_matrix, cmap=color,linewidths=.5, linecolor='lightgray',cbar=False)
                ax.set_yticklabels([])

            ax.set_xlabel('Batch'+str(n_batch)+'\nof '+str(batch_size)+'policies')
            ax.set_xticks(range(0,pos_err_matrix.shape[1],10))
            ax.set_xticklabels(range(batch_size*(n_batch-1),batch_size*n_batch,10),rotation=0)
            ax.set_xticklabels([])
            

            n_batch+=1

    # LINEA 2: trozos de learning-curve por defecto (test reward de ultima politica visitada) por batch
    #----------------------------------------------------------------------------------------------
    matrix_batch_mean_rewards=np.array(matrix_batch_mean_rewards)
    matrix_batch_mean_rewards=matrix_batch_mean_rewards/np.max(matrix_batch_mean_rewards)
    for i in range(1,n_batch):
        ax=plt.subplot(3,df_test.shape[0]//batch_size,df_test.shape[0]//batch_size+i)
        plt.plot(range(batch_size),matrix_batch_mean_rewards[i-1],color='grey')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(-0.1,1.1)
        if i==1:
            ax.set_ylabel('Learning-curve\nlast visited policy\n(normalized)')

    # LINEA 1: trozo de "curva de rewards por episodio de entrenamiento" por batch
    #----------------------------------------------------------------------------------------------
    max_value=-np.Inf
    for i in batch_ep_reward_matrix:
        max_row=max(i)
        if max_row>max_value:
            max_value=max_row
    norm_batch_ep_reward_matrix=[]
    for i in batch_ep_reward_matrix:
        norm_batch_ep_reward_matrix.append(np.array(i)/max_value)

    for i in range(1,n_batch):
        ax=plt.subplot(3,df_test.shape[0]//batch_size,i)
        plt.plot(range(len(norm_batch_ep_reward_matrix[i-1])),norm_batch_ep_reward_matrix[i-1],color='grey')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(-0.1,1.1)
        if i==1:
            ax.set_ylabel('Train rewards\nper episode\n(normalized)')

    plt.savefig(path+'/comparison_test_train_reward_per_batches'+str(seed)+'.pdf')
    plt.show()
    plt.close()

    #----------------------------------------------------------------------------------------------
    # GRAFICA 3: 
    #----------------------------------------------------------------------------------------------

    # Test reward por politica con maximo n_test_ep
    test_rewards=list(df_test['mean_reward'])
    max_test_reward_change,_=max_change_in_list(test_rewards)

    # Train reward por episodio con cada politica (basado en trayectorias propias)
    matrix_train_rewards=form_df_train_to_per_policy_ep_train_rewards(path,seed,500000)

    # Representar curva por batch
    fig=plt.figure(figsize=[10,10])
    plt.subplots_adjust(left=0.08,bottom=0.07,right=0.96,top=0.9,wspace=0.16,hspace=0.55)

    batch_size=25
    n_batches=0
    for i in range(len(test_rewards)):
        if (i+1)%batch_size==0:

            ax=plt.subplot(df_test.shape[0]//batch_size,1,n_batches+1)

            # Curva de test rewards
            x=range(n_batches*batch_size*n_steps+1,(n_batches+1)*batch_size*n_steps,n_steps)
            y=test_rewards[(n_batches*batch_size):((n_batches+1)*batch_size)]
            max_change=max_test_reward_change[(n_batches*batch_size):((n_batches+1)*batch_size)]

            
            change_True=[i for i in range(len(max_change)) if max_change[i]]
            change_False=[i for i in range(len(max_change)) if not max_change[i]]
    
            plt.scatter([x[i] for i in change_True],[y[i] for i in change_True],color='green')
            plt.scatter([x[i] for i in change_False],[y[i] for i in change_False],color='red')

            #Curva de train rewrads
            current_rows=matrix_train_rewards[(n_batches*batch_size):((n_batches+1)*batch_size)]
            x=[]
            for i in range(batch_size):
                x+=list(np.linspace(n_batches*batch_size*n_steps+i*n_steps,n_batches*batch_size*n_steps+i*n_steps+n_steps,len(current_rows[i]),dtype=int))
            y=sum(current_rows,[])
            plt.plot(x,y,color='black')

            if n_batches==0:
                ax.set_title('Train rewards per episode (in black) and\n test reward per policy (dots, green when max changes)')
            elif n_batches==df_test.shape[0]//batch_size-1:
                ax.set_xlabel('Train steps')            

            n_batches+=1

    plt.savefig(path+'/comparison_test_train_reward_on_the_fly'+str(seed)+'.pdf')
    plt.show()
    plt.close()

################
'''
- Test reward: cuando cambia la mejor politica, y como de importante es el cambio (incremento de test reward)
- Train reward: definir diferentes medidas para representar el train reward obtenido con cada politica (trayectoria),
y segun esa medida, mirar cuando cambia la mejor politica.

Con esos datos, construir una grafica tile. El eje OX son las politicas visitadas por orden de visita. La primera fila (arriba)
sera degradada, blanco= no hay cambio de maximo, negro= hay cambio y es grande. Las demas filas, cada una correspondera a una 
diferente medida usada para resumir el train reward de la interaccion de cada politica, y sera una tile rojo=no hay cambio de maximo
y verde=si hay cambio.

Esta grafica permitira ver que criterio detecta mejor los cambios importantes (los que nos interesan) en test reward. Ademas,
aunque no detecte todos los cambios importantes, podremos ver si detecta los importantes (en los que el cambio de reward es grande).

'''
def form_df_train_to_per_policy_ep_train_rewards(path,seed,max_train_timesteps):

    df_train=pd.read_parquet(path+'/df_train_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    df_train['train_rewards']=[UtilsDataFrame.compress_decompress_list(i,compress=False) for i in list(df_train['train_rewards'])]
    df_train['train_ep_end']=[UtilsDataFrame.compress_decompress_list(i,compress=False) for i in list(df_train['train_ep_end'])]

    # Obtener listas con rewards train por step e inicio de cada episodio train.
    df_train=df_train[df_train['n_train_timesteps']<=max_train_timesteps]
    n_policies=list(df_train['n_policy'])

    train_rewards=[]
    train_ep_end=[]
    for i in n_policies:
        train_rewards+=list(df_train[df_train['n_policy']==i]['train_rewards'])[0]
        train_ep_end+=list(df_train[df_train['n_policy']==i]['train_ep_end'])[0]

    n_train_timesteps=list(df_train['n_train_timesteps'])

    # Obtener lista con el numero de episocios train evaluados por cada politica durante el entrenamiento.
    num_ep_start_policy=[]
    ep_rew_mean=[]
    last_i=0# Last episode end
    current_i=0# Current step

    for i in train_ep_end:
        if i:
            ep_rew_mean.append(sum(train_rewards[last_i:current_i]))
            last_i=current_i
        current_i+=1

        if current_i in n_train_timesteps:
            num_ep_start_policy.append(len(ep_rew_mean))

    # Obtener matriz con los train rewards por episodio asociados a cada politica por fila.
    policy_ep_reward_matrix=[]
    for i in range(len(num_ep_start_policy)):
        if i==0:
            policy_ep_reward_matrix.append(ep_rew_mean[:num_ep_start_policy[i]])
        else:
            policy_ep_reward_matrix.append(ep_rew_mean[num_ep_start_policy[i-1]:num_ep_start_policy[i]])
     

    return policy_ep_reward_matrix

def max_change_in_list(value_list):
    max_change_bool=[]
    change_values=[]
    max_value=0
    for i in value_list:
        if i>max_value:
            max_change_bool.append(True)
            change_values.append(i-max_value)
            max_value=i
        else:
            max_change_bool.append(False)
            change_values.append(0)

    # Normalizar
    change_values=[(i-min(change_values))/(max(change_values)-min(change_values))  if i!= 0 else i for i in change_values]

    return max_change_bool,change_values

def train_reward_metrics_per_policy(matrix_train_rewards):
    metrics_mean=[]
    metrics_max=[]
    metrics_percentile=[]

    for i in matrix_train_rewards:
        metrics_mean.append(np.mean(i))
        metrics_max.append(max(i))
        metrics_percentile.append(np.percentile(i,0.75))

    return metrics_mean,metrics_max,metrics_percentile

def test_freq_from_train_reward(path,seed):

    df_train=pd.read_parquet(path+'/df_train_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    n_steps=int(df_train['n_train_timesteps'].min())

    df_test=pd.read_parquet(path+'/df_test_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    df_test=df_test[df_test['n_policy']<=500000//n_steps]

    # Test reward por politica con maximo n_test_ep
    test_rewards=list(df_test['mean_reward'])
    _,test_reward_changes=max_change_in_list(test_rewards)


    # Train reward por episodio con cada politica (basado en trayectorias propias)
    trajec_train_rewards=form_df_train_to_per_policy_ep_train_rewards(path,seed,500000)
    metrics_mean,metrics_max,metrics_percentile=train_reward_metrics_per_policy(trajec_train_rewards)

    _,metrics_mean_changes=max_change_in_list(metrics_mean)
    _,metrics_max_changes=max_change_in_list(metrics_max)
    _,metrics_perc_changes=max_change_in_list(metrics_percentile)

    # Train rewards usando ventanas deslizantes
    window_train_rewaerds=form_df_train_to_matrix_train_rewards([100,50,20,10,5,1],path,seed,500000)
    window_train_rewaerds_changes=[]
    for i in window_train_rewaerds:
        _,changes=max_change_in_list(i)
        window_train_rewaerds_changes.append(changes)


    # Comparar eficacia de cada metrica a la hora de detectar cambios en el maximo test reward
    fig=plt.figure(figsize=[10,4])
    plt.subplots_adjust(left=0.13,bottom=0.152,right=0.96,top=0.9,wspace=0.16,hspace=0.2)
    ax=plt.subplot(111)

    plot_matrix=np.array([test_reward_changes,metrics_mean_changes,metrics_max_changes,metrics_perc_changes]+window_train_rewaerds_changes)


    cmap_grises = sns.color_palette("Greys", as_cmap=True)
    sns.heatmap(plot_matrix, cmap=cmap_grises, cbar=True, ax=ax, linewidths=0.5, vmin=0, vmax=1,norm=PowerNorm(gamma=0.3))

    ax.set_xticks(list(range(1,plot_matrix.shape[1]+1,20)))
    ax.set_yticks(np.arange(plot_matrix.shape[0]) + 0.5)
    ax.set_yticklabels(['Test reward', 'Trajec mean', 'Trajec max', 'Trajec perc','Window 100','Window 50','Window 20','Window 10','Window 5','Window 1'],rotation=0) 
    ax.set_xticklabels(list(range(1,plot_matrix.shape[1]+1,20)),rotation=0)
    colorbar=ax.collections[0].colorbar
    colorbar.set_label('Normalized change in max metric value ')

    ax.set_xlabel('n_policy')
    ax.set_ylabel('Metric to detect change')
    plt.title("Maximum change detection in test reward")
    plt.savefig(path+'/maximum_change_detection'+str(seed)+'.pdf')
    plt.show()
    plt.close()



    # learning-curves determinando el cuando validar con las trayectorias del train
    plot_leraning_curves_constant_n_test_episodes(list(range(10000, 500001, 5000)),[1],'results/KnowingSingleProcess',seed,
                                                  list_eval_freq=[1,
                                                                  ['Trajec max 0.01',metrics_max_changes,0.01],
                                                                  ['Trajec max 0.05',metrics_max_changes,0.05],
                                                                  ['Trajec max 0.08',metrics_max_changes,0.08],
                                                                  ['Trajec max 0.1',metrics_max_changes,0.1]
                                                                  ])






#==================================================================================================
# Programa principal
#==================================================================================================

# Inverted Double pendulum
#--------------------------------------------------------------------------------------------------
# GRAFICA 1: conditions
plot_environment_requirements([3,7,8,10,11,12,13,15],'results/KnowingSingleProcess','InvertedDoublePendulum',250*20,200,9350)

# GRAFICA 2: validacion en todas las iteraciones con test ep constante
plot_leraning_curves_constant_n_test_episodes(list(range(10000, 500001, 5000)),[100,50,25,12,6,3,1],'results/KnowingSingleProcess',3)

plot_leraning_curves_constant_n_test_episodes(list(range(10000, 500001, 5000)),[100,50,25,12,6,3,1],'results/KnowingSingleProcess',11)


# GRAFICA 3: validacion en todas las iteraciones con train ep (tamaño ventana) constante
plot_leraning_curves_constant_n_train_episodes(list(range(10000, 500001, 5000)),[100,50,25,12,6,3,1],'results/KnowingSingleProcess',3)

# GRAFICA 4: entendiendo relacion entre test y train reward
relationship_test_and_train_reward([100,50,25,12,6,3,1],'results/KnowingSingleProcess',3)

# GRAFICA 5: determinando frecuencia de validacion test
plot_leraning_curves_constant_n_test_episodes(list(range(10000, 500001, 5000)),[100,50,25,12,6,3,1],'results/KnowingSingleProcess',3,[5])
plot_leraning_curves_constant_n_test_episodes(list(range(10000, 500001, 5000)),[100,50,25,12,6,3,1],'results/KnowingSingleProcess',3,[10])
plot_leraning_curves_constant_n_test_episodes(list(range(10000, 500001, 5000)),[100,50,25,12,6,3,1],'results/KnowingSingleProcess',3,[50])
plot_leraning_curves_constant_n_test_episodes(list(range(10000, 500001, 5000)),[1],'results/KnowingSingleProcess',3,[1,5,10,50])
test_freq_from_train_reward('results/KnowingSingleProcess',3)

test_freq_from_train_reward('results/KnowingSingleProcess',11)

# Ant
#--------------------------------------------------------------------------------------------------
plot_environment_requirements([1,2,3],'results/KnowingSingleProcess','Ant',500*4,500*4,6000)