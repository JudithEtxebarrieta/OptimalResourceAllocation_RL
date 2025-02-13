# gymnasium environments
pip install "gymnasium[mujoco]"

# Stable Baselines3
pip install 'stable-baselines3[extra]'

# Sample Factory from PyPI
pip install sample-factory
pip install sample-factory[mujoco]

# Sample Factory from sources
cd libraries
git clone https://github.com/JudithEtxebarrieta/sample-factory.git
cd sample-factory
pip install -e .

# rl-games from PyPI (con esto no funciona, incompatibilidad con las versiones de las librerias)
pip3 install torch torchvision
pip install rl-games
pip install ray
pip install brax

# rl-games from sources (he tenido que instalar versiones diferentes a las indicadas para hacerlas compatibles)
cd libraries
git clone https://github.com/JudithEtxebarrieta/rl_games.git
cd rl_games

poetry install -E brax
poetry run pip3 install torch torchvision
poetry run pip install ray
poetry run pip install numpy==1.25.0
poetry run pip install jax==0.4.13
poetry run pip install jaxlib==0.4.13
poetry install -E atari





