# To install MuJoCo environments to work with Sample-Factory v2
pip install sample-factory[mujoco]
sudo apt-get install libglew-dev libosmesa6-dev

# PPO with stable_baselines3
pip install stable-baselines3[extra]

# PPO with Gymnasium-Robotics environments
pip install gymnasium-robotics[mujoco-py]

# highway-env (does not work)
sudo apt-get update -y
sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev
    libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev
    ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
pip install highway-env