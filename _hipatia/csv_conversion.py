import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from experiments.our_library import compress_decompress_list

def from_SavedListColumn_to_ReadListColumn(column,elem_type='float'):
    new_column=[]
    for elem_list in column:
        if elem_type=='float':
            new_column.append([float(i) for i in elem_list.replace('], dtype=float32), array([',',').replace('[array([','').replace('], dtype=float32)]','').split(',')])
        if elem_type=='bool':
            new_column.append([True if i==' True' else False for i in elem_list.replace(']), array([',',').replace('[array([','').replace('])]','').split(',') ])
        if elem_type=='int':
            new_column.append([int(i) for i in elem_list.replace(']','').replace('[','').replace(' ','').split(',') ])


    return new_column


# Covertir a comprimido

def compress_cluster_csv(csv_name,train=True):
    df=pd.read_csv('_hipatia/results_hipatia/'+csv_name)
    #os.remove('_hipatia/results_hipatia/'+csv_name)

    if train:
        df['train_rewards']=from_SavedListColumn_to_ReadListColumn(df['train_rewards'])
        df['train_ep_end']=from_SavedListColumn_to_ReadListColumn(df['train_ep_end'],elem_type='bool')

        a=list(df['train_rewards'])
        b=list(df['train_ep_end'])
    else:
        df['ep_test_len']=from_SavedListColumn_to_ReadListColumn(df['ep_test_len'],elem_type='int')
        df['ep_test_rewards']=from_SavedListColumn_to_ReadListColumn(df['ep_test_rewards'],elem_type='float')

        a=list(df['ep_test_len'])
        b=list(df['ep_test_rewards'])


    new_a=[]
    new_b=[]
    for i in range(len(a)):
        new_a.append(compress_decompress_list(a[i]))
        new_b.append(compress_decompress_list(b[i]))

    if train:    
        df['train_rewards']=new_a
        df['train_ep_end']=new_b
    else:
        df['ep_test_len']=new_a
        df['ep_test_rewards']=new_b
    df.to_csv('results/cluster_experiment/'+csv_name)

# Unir bases de datos de misma semilla
def join_csv_and_save_parquet(list_csv_name,path,joined_name):
    df=pd.read_csv(path+list_csv_name[0],index_col=0)
    os.remove(path+list_csv_name[0])
    for i in range(1,len(list_csv_name)):
        df_new=pd.read_csv(path+list_csv_name[i],index_col=0)
        df = pd.concat([df, df_new], ignore_index=False)
        os.remove(path+list_csv_name[i])

    df.to_parquet(path+joined_name, engine='pyarrow', compression='gzip',index=False)

def join_csv_and_save_parquetNEW(list_csv_name,path,joined_name):
    df=pd.read_csv(path+list_csv_name[0])
    #os.remove(path+list_csv_name[0])
    for i in range(1,len(list_csv_name)):
        df_new=pd.read_csv(path+list_csv_name[i])
        df = pd.concat([df, df_new], ignore_index=False)
        #os.remove(path+list_csv_name[i])

    df.to_parquet('results/cluster_experiment/'+joined_name, engine='pyarrow', compression='gzip',index=False)

def joint_parquet(path,csv_name):
    df=pd.read_parquet(path+csv_name+'.parquet')
    #df = df.drop(columns=['Unnamed: 0'])
    df_new=pd.read_parquet(path+csv_name+'NEW.parquet')
    os.remove(path+csv_name+'NEW.parquet')
    df = pd.concat([df, df_new], ignore_index=False)
    df.to_parquet(path+csv_name+'.parquet', engine='pyarrow', compression='gzip',index=False)

def from_csv_to_parquet(path,csv_name):
    df = pd.read_csv(path+csv_name+'.csv')
    df.to_parquet(path+csv_name+'.parquet', engine='pyarrow', compression='gzip')


#######
'''
3:21#
6:9//
7:20#
8:20#
11:28#
10:23#
12:27#
13:20#
14:17
15:20#
16:24#
17:16
18:27
19:11
20:12
21:7
22:7
23:7
'''


def create_list_csv_name(seed,tasks,train_test):
    list_csv_name=[]

    for i in tasks:
        list_csv_name.append('df_'+str(train_test)+'_'+str(i)+'_InvertedDoublePendulum_seed'+str(seed)+'.csv')
    
    funcion1='df_'+str(train_test)+'_InvertedDoublePendulum_seed'+str(seed)+'.parquet'
    funcion2='df_'+str(train_test)+'_InvertedDoublePendulum_seed'+str(seed)

    return list_csv_name,funcion1,funcion2

list_csv_name,funcion1,funcion2=create_list_csv_name(23,list(range(1,8)),'train')

# for i in list_csv_name:
#     compress_cluster_csv(i,train=False)

# join_csv_and_save_parquet(list_csv_name,'results/cluster_experiment/','df_test_InvertedDoublePendulum_seed11NEW.parquet')
# joint_parquet('results/cluster_experiment/','df_test_InvertedDoublePendulum_seed11')

####
join_csv_and_save_parquetNEW(list_csv_name,'_hipatia/results_hipatia/',funcion1)
# joint_parquet('results/cluster_experiment/',funcion2)



