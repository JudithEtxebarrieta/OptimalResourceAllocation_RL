'''
Copiado de runner.py. Hacer como en sample-factory para poner a mi gusto este fichero.


poetry install -E brax
poetry run pip install --upgrade "jax[cuda]==0.3.13" -f https://storage.googleapis.com/jax-releases/jax_releases.html
poetry run python runner.py --train --file rl_games/configs/brax/ppo_ant.yaml
poetry run python runner.py --play --file rl_games/configs/brax/ppo_ant.yaml --checkpoint runs/Ant_brax/nn/Ant_brax.pth
'''

from distutils.util import strtobool
import argparse, os, yaml
import torch

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def learn_process(file):

    # Setting de los parametros
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, required=False, 
                    help="random seed, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-p", "--play", required=False, help="play(test) network", action='store_true')
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-f", "--file", required=True, help="path to config")
    ap.add_argument("-na", "--num_actors", type=int, default=0, required=False,
                    help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-s", "--sigma", type=float, required=False, help="sets new sigma value in case if 'fixed_sigma: True' in yaml config")
    ap.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    ap.add_argument("--wandb-project-name", type=str, default="rl_games",
        help="the wandb's project name")
    ap.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Para añadir los indicados explicitamente
    my_args = [  
                '--train', # Para entrenar/aprender
                f'--file={file}'
    ]

    args = vars(ap.parse_args(args=my_args))
    config_name = args['file']

    # Cargar parametros del fichero
    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)

        if args['num_actors'] > 0:
            config['params']['config']['num_actors'] = args['num_actors']

        if args['seed'] > 0:
            config['params']['seed'] = args['seed']
            config['params']['config']['env_config']['seed'] = args['seed']

        from rl_games.torch_runner import Runner

        try:
            import ray
        except ImportError:
            pass
        else:
            ray.init(object_store_memory=1024*1024*1000)

        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    global_rank = int(os.getenv("RANK", "0"))
    if args["track"] and global_rank == 0:
        import wandb
        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )

    runner.run(args)

    try:
        import ray
    except ImportError:
        pass
    else:
        ray.shutdown()

    if args["track"] and global_rank == 0:
        wandb.finish()


if __name__ == '__main__':
    #torch.device("cpu")
    file='/home/jesusangel/Dropbox/PhD/Mi trabajo/Codigo/OptimalResourceAllocation_RL/experiments_LibrariesRL/results/rlgames/ppo_pong.yaml'
    learn_process(file)


