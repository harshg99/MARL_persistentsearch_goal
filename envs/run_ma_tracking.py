import maTTenv
import numpy as np
import argparse
import gym
from stable_baselines.common.cmd_util import make_vec_env

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', type=str, default='setTracking-v0')
parser.add_argument('--render', help='whether to render', type=int, default=0)
parser.add_argument('--record', help='whether to record', type=int, default=0)
parser.add_argument('--nb_agents', help='the number of agents', type=int, default=4)
parser.add_argument('--nb_targets', help='the number of targets', type=int, default=4)
parser.add_argument('--num_envs', help='the number of envs', type=int, default=4)
parser.add_argument('--log_dir', help='a path to a directory to log your data', type=str, default='.')
parser.add_argument('--map', type=str, default="emptyMed")

args = parser.parse_args()

# @profile

def main():
    envs = maTTenv.make(args.env,
                    render=args.render,
                    record=args.record,
                    directory=args.log_dir,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=False,
                    num_envs=args.num_envs
                    )
    
    nlogdetcov = []
    action_dict = [{} for _ in range(args.num_envs)]
    done = np.array([False for _ in range(args.num_envs)])

    obs = envs.reset()
    # from IPython import embed; embed()
    # See below why this check is needed for training or eval loop
    i = 1
    while np.sum(done) != args.num_envs:
        if args.render:
            envs.envs[0].render()
            #for env_i in range(args.num_envs):
            #    envs.envs[env_i].render()

        for agent_id, o in obs.items():
            
            action = envs.action_space.sample()
            for j in range(action.shape[0]):
                action_dict[j][agent_id] = action[j].item()

        print(envs)
        obs, rew, done, info = envs.step(action_dict)
        
        # done = np.array(done)
        print(f"type {type(done)} next done {done}")
        print(f"step num {i} every 200 {i % 200 == 0}")
        # if i % 200 == 0:
        #    from IPython import embed; embed()
        nlogdetcov.append(info['mean_nlogdetcov'])
        i += 1

    print("Sum of negative logdet of the target belief covariances : %.2f"%np.sum(nlogdetcov))

"""
def main():
    env = maTTenv.make(args.env,
                    render=args.render,
                    record=args.record,
                    directory=args.log_dir,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=False,
                    )
    nlogdetcov = []
    action_dict = {}
    done = False #{'__all__':False}

    obs = env.reset()
    # See below why this check is needed for training or eval loop
    while not done: # ['__all__']:
        if args.render:
            env.render()

        for agent_id, o in obs.items():
            action_dict[agent_id] = env.action_space.sample()

        obs, rew, done, info = env.step(action_dict)
        print(done)
        nlogdetcov.append(info['mean_nlogdetcov'])

    print("Sum of negative logdet of the target belief covariances : %.2f"%np.sum(nlogdetcov))
"""

if __name__ == "__main__":
    main()
    """
    To use line_profiler
    add @profile before a function to profile
    kernprof -l run_ma_example.py --env setTracking-v3 --nb_agents 4 --nb_targets 4 --render 0
    python -m line_profiler run_ma_example.py.lprof 

    Examples:
        >>> env = MyMultiAgentEnv()
        >>> obs = env.reset()
        >>> print(obs)
        {
            "agent_0": [2.4, 1.6],
            "agent_1": [3.4, -3.2],
        }
        >>> obs, rewards, dones, infos = env.step(
            action_dict={
                "agent_0": 1, "agent_1": 0,
            })
        >>> print(rew)
        {
            "agent_0": 3,
            "agent_1": -1,
            "__all__": 2,
        }
        >>> print(done)
        #Due to gym wrapper, done at TimeLimit is bool, True.
        #During episode, it is a dict so..
        #While done is a dict keep running
        {
            "agent_0": False,  # agent_0 is still running
            "agent_1": True,   # agent_1 is done
            "__all__": False,  # the env is not done
        }
        >>> print(info)
        {
            "agent_0": {},  # info for agent_0
            "agent_1": {},  # info for agent_1
        }
    """