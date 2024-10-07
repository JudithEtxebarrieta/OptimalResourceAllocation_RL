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

from our_library import  UtilsDataFrame

#==================================================================================================
# GRAFICA 1: Comprobar graficamente que se cumplen los requisitos del environment
#==================================================================================================
def plot_environment_requirements(list_seeds,path):

    fig=plt.figure(figsize=[10,4])
    plt.subplots_adjust(left=0.08,bottom=0.27,right=0.97,top=0.84,wspace=0.39,hspace=0.2)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: learning-curves (cortadas en n_policy relevantes)
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(131)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    n_policies=250*20
    all_x_conv=[]
    all_y_max=[]

    for seed in list_seeds:
        df=pd.read_parquet(path+'/df_test_InvertedDoublePendulum_seed'+str(seed)+'.parquet')

        x=list(range(1,n_policies))
        y=[]
        y_max=-np.Inf
        for i in x:
            new_y=df[df['n_policy']<=i]['mean_reward'].max()
            y.append(new_y)
            if new_y>y_max:
                y_max=new_y
                if new_y<=9350:
                    x_conv=i

        plt.plot(x, y, linewidth=1,color='grey')
        all_x_conv.append(x_conv)
        all_y_max.append(y_max)

    plt.axhline(y=9350,color='black', linestyle='--')
    plt.axvline(x=np.mean(all_x_conv),color='red', linestyle='--')
    plt.xlim(0,200)

    ax.set_xlabel("n_policies\n(cut in representative values,\nbut executed until 5000)",fontsize=10)
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
    ax.set_xlabel("process\n\nReformulated condition 1:\nbudget is significantly longer\nthan average convergence time",fontsize=10)
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
    ax.set_xlabel("process\n\n Reformulated condition 2:\nprocesses converge at\nsignificantly different speeds",fontsize=10)
    ax.set_ylabel("convergence test reward",fontsize=10)
    ax.set_title('Condition 2:\nConvergence does not imply optimality',fontsize=10,color='red')

    plt.savefig(path+'/conditions.pdf')
    plt.show()
    plt.close()

#==================================================================================================
# GRAFICA 2: validar todas las politicas durante el entrenamiento con numero de episodios test constante
#==================================================================================================
def plot_leraning_curves_constant_n_test_episodes(x,list_n_test_ep,path,seed):

    random.seed(0)

    df_test=pd.read_parquet(path+'/df_test_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    df_train=pd.read_parquet(path+'/df_train_InvertedDoublePendulum_seed'+str(seed)+'.parquet')

    n_policies=list(df_test['n_policy'])

    fig=plt.figure(figsize=[10,5])
    plt.subplots_adjust(left=0.08,bottom=0.152,right=0.73,top=0.9,wspace=0.39,hspace=0.2)

    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    # Dibujar curva sin test durante el proceso (ultima politica observada)
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(df_test[df_test['n_policy']==last_policy]['mean_reward'])
    plt.plot(x, y, linewidth=1,color='black',label='0 (last visited policy)')

    # Dibujar curva con mejor test pero sin contar el extra de tiempo
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(df_test[df_test['n_policy']<=last_policy]['mean_reward'].max())
    plt.plot(x, y, linewidth=1,color='yellow',label='100 (counted as 0)')

    # Resto de curvas
    for n_test_ep in list_n_test_ep:
        print(n_test_ep)

        if n_test_ep==100:
            df_test['current_mean_reward']=df_test['mean_reward']
            df_test['current_n_test_timesteps']=df_test['n_test_timesteps']

        else:
            current_mean_reward=[]
            current_n_test_timesteps=[]
            cumulative_n_test_timesteps=0
            for policy in n_policies:
                ep_test_rewards_compressed=list(df_test[df_test['n_policy']==policy]['ep_test_rewards'])
                ep_test_len_compressed=list(df_test[df_test['n_policy']==policy]['ep_test_len'])

                ep_test_rewards=UtilsDataFrame.compress_decompress_list(ep_test_rewards_compressed[0],compress=False)
                ep_test_len=UtilsDataFrame.compress_decompress_list(ep_test_len_compressed[0],compress=False)

                current_mean_reward.append(np.mean(random.sample(ep_test_rewards,n_test_ep)))
                cumulative_n_test_timesteps+=sum(random.sample(ep_test_len,n_test_ep))
                current_n_test_timesteps.append(cumulative_n_test_timesteps)

            df_test['current_mean_reward']=current_mean_reward
            df_test['current_n_test_timesteps']=current_n_test_timesteps

        total_timesteps=np.array(df_test['current_n_test_timesteps'])+np.array(df_train['n_train_timesteps'])
        df_test['total_timesteps']=total_timesteps
        y=[]
        for timesteps in x:
            indx_max=df_test[df_test['total_timesteps']<=timesteps]['current_mean_reward'].idxmax()
            y.append(df_test['mean_reward'][indx_max])


        plt.plot(x, y, linewidth=1,label=n_test_ep)

    ax.legend(title="n_test_ep",fontsize=8,bbox_to_anchor=(1.3, 1, 0, 0))
    ax.set_xlabel("Total time steps (train+test)\n \n(2048 train timesteps = training 1 policy = 1 trajectory)",fontsize=10)
    ax.set_ylabel("Best policy reward (in terms of max_n_test_ep)",fontsize=10)
    ax.set_title('Learning-curve: best policy found\n(iteratively validating policies with n_test_ep test episodes) ',fontsize=10)

    plt.savefig(path+'/learning_curve_constant_n_test_ep'+str(seed)+'.pdf')
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
        print('iteration: ')
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

    plt.savefig(path+'/best_n_train_ep_during_training'+str(seed)+'.pdf')
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

    df_test=pd.read_parquet(path+'/df_test_InvertedDoublePendulum_seed'+str(seed)+'.parquet')
    df_test=df_test[df_test['n_policy']<=500000//2048]

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

    plt.savefig(path+'/comparison_test_train_reword'+str(seed)+'.pdf')
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

    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
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

    plt.savefig(path+'/comparison_test_train_reword_per_batches'+str(seed)+'.pdf')
    plt.show()
    plt.close()

#==================================================================================================
# Programa principal
#==================================================================================================
# GRAFICA 1: conditions
plot_environment_requirements([3,7,8,10,11,12,13,15],'results/KnowingSingleProcess')

# GRAFICA 2: validacion en todas las iteraciones con test ep constante
plot_leraning_curves_constant_n_test_episodes(list(range(10000, 500001, 5000)),[100,50,25,12,6,3,1],'results/KnowingSingleProcess',3)

# GRAFICA 3: validacion en todas las iteraciones con train ep (tamaño ventana) constante
plot_leraning_curves_constant_n_train_episodes(list(range(10000, 500001, 5000)),[100,50,25,12,6,3,1],'results/KnowingSingleProcess',3)

# GRAFICA 4: entendiendo relacion entre test y train reward
relationship_test_and_train_reward([100,50,25,12,6,3,1],'results/KnowingSingleProcess',3)
