import json
import bz2
import base64
import numpy as np
import os
import runpy


def external_run(run_script_path, run_lines):

    '''
    run_lines: [una menos que la init en los numeros del script,la misma que la ultima en los numeros del script]
    '''
    
    this_script = os.path.dirname(os.path.abspath(__file__))+'/'+os.path.splitext(os.path.basename(run_script_path))[0]+'.py'
    this_script_copy = this_script.replace('.py', '_copy.py')

    # Crear una copia de este archivo
    with open(this_script, 'r') as f_b: #Leer
        content_b = f_b.readlines()

    with open(this_script_copy, 'w') as f_b_copy: # Crear y copiar
        f_b_copy.writelines(content_b)  

    # Agregar lineas a ejecutar
    with open(run_script_path, 'r') as f_a: # Leer lineas de interes
        lines_a = f_a.readlines()
    lines_to_run = [lines_a[i] for i in run_lines]

    with open(this_script_copy, 'a') as f_b_copy: # Añadir lineas de interes
        f_b_copy.writelines(lines_to_run)

    # Ejecutar fichero y eliminar
    runpy.run_path(this_script_copy, run_name='__main__')
    os.remove(this_script_copy)

def compress_decompress_list(my_list,compress=True):
    if compress:
        # Convertir la lista a una cadena JSON compacta
        json_str = json.dumps(my_list)

        # Comprimir la cadena
        compressed_data = bz2.compress(json_str.encode('utf-8'))

        # Convertir a base64 para guardar como texto en la base de datos
        compressed_str = base64.b64encode(compressed_data).decode('utf-8')

        return compressed_str
    else:
        # Leer la cadena comprimida de la base de datos
        compressed_data = base64.b64decode(my_list.encode('utf-8'))

        # Descomprimir la cadena
        json_str = bz2.decompress(compressed_data).decode('utf-8')

        # Convertir la cadena JSON de vuelta a lista
        my_list = json.loads(json_str)

        return my_list

def training_stats(train_rewards,train_ep_ends,n_timesteps_per_iter,stats_window_size):
    ep_rw_policy=[]

    for time_steps in n_timesteps_per_iter:
        current_train_rewards=train_rewards[:time_steps]
        current_train_ep_ends=train_ep_ends[:time_steps]

        ep_rw=[]
        last_i=0
        current_i=0

        for i in current_train_ep_ends:
            current_i+=1
            if i:
                ep_rw.append(sum(current_train_rewards[last_i:current_i]))
                last_i=current_i
            
                

        if len(ep_rw)<stats_window_size:
            ep_rw_policy.append(np.mean(ep_rw))
        else:
            ep_rw_policy.append(np.mean(ep_rw[-stats_window_size:]))

    return ep_rw_policy
