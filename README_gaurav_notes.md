# Setup

# Norms on Running Experiments

1. Make sure to use --track flag to use wandb. Best way to share results 
2. Always run experiment like this `python algos/maTT/run_<algo>.py <args>` so that results are stored in parent directory. Make this the norm.

# Instructions for Docker

- docker build -t ma_dyna_TT .
- docker run -it --gpus all --net host --ipc host ma_dyna_TT bash
- conda create -n ma_dyna_TT_venv python tensorflow-gpu=1.14 -y
- conda activate ma_dyna_TT_venv
- pip install pyyaml==5.4.1 scipy numpy tabulate matplotlib "gym[atari,classic_control]==0.24.1" tqdm joblib zmq dill progressbar2 mpi4py cloudpickle click opencv-python wandb filterpy scikit-image
- pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
- git clone https://github.com/gauravkuppa/scalableMARL.git
- cd scalableMARL/
- git checkout ppo_works
- source setup 