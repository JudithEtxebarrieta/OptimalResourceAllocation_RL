
''' 
En este script se ejecuta un proceso de RL con el algoritmo PPO implementado en Sample Factory sobre el entorno Ant de MuJoCo. 
Basado en: https://github.com/alex-petrenko/sample-factory/tree/master/sf_examples/mujoco

El objetivo era poder implementar estas dos funciones:
1) learn
    input: entorno, semilla aleatoria, y tiempo total de entrenamiento.
    output: guardar en una base de datos la informacion relevante durante en entrenamiento.

2) resume_learn
    input: informacion de proceso de entrenamiento anterior, nueva posible semilla aleatoria, y tiempo total de entrenamiento.
    output: guardar en una base de datos la informacion relevante durante en entrenamiento.

Los objetivos quedan pendientes por ciertas complicaciones:

- Muchos parametros con documentacion pobre: https://github.com/alex-petrenko/sample-factory/blob/master/docs/02-configuration/cfg-params.md.
La implementacion de PPO en Sample Factory es muy sofisticada, va mas halla de lo que nosotros necesitamos (incluye PBT, multi-task, 
multi-environment, ejecuciones asincronas, ejecuciones en multiples GPU o CPU,...). Por eso, algunos parametrso hay que modificarlos de 
su valor por defecto para que encajen con nuestro caso, y no estoy del todo segura si los estoy modificando bien, por posibles dependencias
que pueda haber entre ellos que no se indican en la documentacion.

- Parametros conflictivos: rollout, batch_size, num_epochs y num_batches_per_epoch.
Cuando se inicializa la clase RolloutWorker al ejecutar run_mujoco_experiment, en la version no-asincrona (la que nos interesa a nosotros) 
se hacen unas comprobaciones de la definicion de los parametros. Concretamente en la linea 121 de rollout_worker.py se quiere comprobar 
que durante cada ciclo del entrenamiento (epoch) como minimo se recorre todo el rollout buffer una vez (una trayectoria mas o igual de larga 
que la que representa el rollout buffer, i.e. como minimo una trayectoria). Pero para calcular el numero de trayectorias consideradas
durante una epoch se hace: num_batches_per_epoch* (batch_size // rollout). Al coger la parte entera en la division, como en teoria batch_size
debe ser menor que rollout, el factor se anula y el assert se activa dando error. He intentado modificar esa linea sustituyendo // por / 
(asi la condicion si se cumple), pero luego me sale otro error.

'''
    
from sf_examples.mujoco import train_mujoco  

def run_mujoco_experiment(seed,train_for_env_steps,env, experiment_name,train_dir,
                          algo, async_rl,device,env_gpu_observations,serial_mode,num_workers,num_envs_per_worker,worker_num_splits):
    args = [
        
        # Estos son los parametros que nos interesan realmente.
        f'--seed={seed}',
        f"--train_for_env_steps={train_for_env_steps}",
        f'--env={env}',
        f'--experiment={experiment_name}',f'--train_dir={train_dir}',

        # Lo ideal seria definir estos argumentos directamente aqui sin que el usuario tenga que meterlos manualmente en __main__
        f'--algo={algo}',f'--async_rl={async_rl}',
        f'--device={device}',
        f'--env_gpu_observations={env_gpu_observations}',
        f'--serial_mode={serial_mode}',
        f'--num_workers={num_workers}',
        f'--num_envs_per_worker={num_envs_per_worker}',
        f'--worker_num_splits={worker_num_splits}'
        
    ]
    train_mujoco.register_mujoco_components()
    final_cfg = train_mujoco.parse_mujoco_cfg(argv=args)
    train_mujoco.run_rl(final_cfg)


if __name__ == "__main__":

    #----------------------------------------------------------------------------------------------
    # Parametros que nos interesan a nosotros
    #----------------------------------------------------------------------------------------------
    seed=1 # Random seed
    train_for_env_steps=100000 # Learning time in time steps
    env='mujoco_ant' # Environment
    train_dir, experiment_name=['checking/results','getstarted_sampleFactory'] # Directories
    
    #----------------------------------------------------------------------------------------------
    # Parametros que deberiamos dejar predefinidos
    #----------------------------------------------------------------------------------------------
    # RL algorithm (queremos usar version original del PPO, i.e. no asincrona)
    algo='APPO'
    async_rl=False

    # Machine (asumimos que tenemos un unico GPU o CPU, i.e. ejecuciones secuenciales)
    device='cpu'
    env_gpu_observations=False
    serial_mode=True
    num_workers=1
    num_envs_per_worker=1
    worker_num_splits=1

    #----------------------------------------------------------------------------------------------
    # Run experiment
    run_mujoco_experiment(seed,train_for_env_steps,env, experiment_name,train_dir,
                          algo, async_rl,device,env_gpu_observations,serial_mode,num_workers,num_envs_per_worker,worker_num_splits)


