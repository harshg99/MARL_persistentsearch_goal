# MARL for persistent target search

## File Structure

```
scalableMARL
    |___algos
        |___maTT                          #RL alg folder for the target tracking environment
            |___core                      #Self-Attention-based Model Architecture
            |___core_behavior             #Used for further evaluation (Ablation D.2.)
            |___dql                       #Soft Double Q-Learning
            |___evaluation                #Evaluation for Main Results
            |___evaluation_behavior       #Used for further evaluation (Ablation D.2.)
            |___modules                   #Self-Attention blocks
            |___replay_buffer             #RL replay buffer for sets
            |___run_script                #**Main run script to do training and evaluation
    |___envs
        |___maTTenv                       #multi-agent target tracking
            |___env
                |___setTracking_v0        #Standard environment (i.e. 4a4t tasks)
                |___setTracking_vGreedy   #Baseline Greedy Heuristic
                |___setTracking_vGru      #Experiment with Gru (Ablation D.3)
                |___setTracking_vkGreedy  #Experiment with Scalability and Heuristic Mask k=4 (Ablation D.1)
        |___run_ma_tracking               #Example scipt to run environment
    |___setup                             #set PYTHONPATH ($source setup)
```

## Acknowledgements
Extension of work done in
[Scalable Reinforcement Learning Policies for Multi-Agent Control](https://arxiv.org/abs/2011.08055)

CD. Hsu, H. Jeong, GJ. Pappas, P. Chaudhari. "Scalable Reinforcement Learning Policies for Multi-Agent Control". IEEE International Conference on Intelligent Robots and Systems (IROS), Prague, Czech Republic, 2021.

Multi-Agent Reinforcement Learning method to learn scalable control polices for multi-agent target tracking.


## Setup

+ To setup scalableMARL, follow the instruction below.

## Set up python environment for the scalableMARL repository

### Install python3.8 (if it is not already installed)
```
#to check python version
python3 -V

sudo apt-get update
sudo apt-get install python3.8-dev
```

### Set up virtualenv
Python virtual environments are used to isolate package installation from the system

Replace 'virtualenv name' with your choice of folder name
```
sudo apt-get install python3-venv 

python3 -m venv --system-site-packages ./'virtualenv name'
# Activate the environment for use, any commands now will be done within this venv
source ./'virtualenv name'/bin/activate

# To deactivate (in terminal, exit out of venv), do not use during setup
deactivate
```
Now that the virtualenv is activated, you can install packages that are isolated from your system

When the venv is activated, you can now install packages and run scripts

### Install isolated packages in your venv
```
sudo apt-get install -y eog python3-tk python3-yaml python3-pip ssh git

#This command will auto install packages from requirements.txt
pip3 install --trusted-host pypi.python.org -r requirements.txt
```

## Current workflow

scalableMARL repo is ready to go

### Running an algorithm
```
# its best to run from the scalableMARL folder so that logging and saving is consistent
cd ./scalableMARL
# run the alg
python3 algos/maTT/run_ppo.py

# you can run the alg with different argument parameters. See within run_script for more options.
# for example
python3 algos/maTT/run_ppo.py --seed 0 --logdir ./results/maPredPrey --epochs 40
```
### To test, evaluate, and render()
```
# for a general example 
python3 algos/maTT/run_ppo.py --mode test --render 1 --log_dir ./results/maTT/setTracking-v0_123456789/seed_0/ --nb_test_eps 50
# for a saved policy in saved_results
python3 algos/maTT/run_ppo.py --mode test --render 1 --log_dir ./saved_results/maTT/setTracking-v0_123456789/seed_0/
```
### To see training curves
Enter wandb api account and view training curves in your wandb profile

