import numpy as np
import envs
import torch

class GreedyAgent:
    def __init__(self):
        pass

    def act(self, agent, avail_actions):
        # Gets the nearest expected relative position of the target and selects the action which takes it the closest
        # Avail actions is a dictionary with a list of available action and what they do : goal setting , velocity setting#
        # Nearest target
        nearest_pos = np.array([np.inf,np.inf])
        nearest_dist = np.inf
        for i in range(len(agent.belief)):
            target_rel_pos = agent.belief[i].state[:2] - agent.state[:2]
            if nearest_dist > np.linalg.norm(target_rel_pos):
                nearest_dist = np.linalg.norm(target_rel_pos)
                nearest_pos = target_rel_pos

        nearest_dist_action = np.inf
        action_index = -1
        selected_action = -1
        global_yaw = np.arctan2(nearest_pos[1], nearest_pos[0]) + np.pi
        global_yaw_dis = int(global_yaw / (np.pi/2))*np.pi/2
        for action in avail_actions:
            if np.linalg.norm(nearest_pos - avail_actions[action][:2]) <= nearest_dist_action:
                action_index = action
                nearest_dist_action = np.linalg.norm(nearest_pos - avail_actions[action][:2])
                if abs(avail_actions[action][2] - global_yaw_dis) < 0.02:
                    selected_action = action

        if selected_action != -1:
            return selected_action
        elif action_index != -1:
            return action_index
        else:
            # Return greefy if nothing works
            action = np.random.choice(list(avail_actions.keys()), size=1).item()
            return action

class GreedyAssignedAgent:
    def __init__(self, num_targets):
        self.assigned_targets = {j: False for j in range(num_targets)}

    def reset(self):
        self.assigned_targets = {j: False for j in range(len(self.assigned_targets))}

    def act(self, agent, avail_actions):
        # Gets the nearest expected relative position of the target and selects the action which takes it the closest
        # Avail actions is a dictionary with a list of available action and what they do : goal setting , velocity setting#
        # Nearest target
        nearest_pos = np.array([np.inf,np.inf])
        nearest_dist = np.inf
        target_selected =  -1
        for i in range(len(agent.belief)):
            target_rel_pos = agent.belief[i].state[:2] - agent.state[:2]
            if nearest_dist > np.linalg.norm(target_rel_pos) and not self.assigned_targets[i]:
                nearest_dist = np.linalg.norm(target_rel_pos)
                nearest_pos = target_rel_pos
                target_selected = i

        self.assigned_targets[target_selected] = True

        nearest_dist_action = np.inf
        action_index = -1
        selected_action = -1
        global_yaw = np.arctan2(nearest_pos[1], nearest_pos[0]) + np.pi
        global_yaw_dis = int(global_yaw / (np.pi/2))*np.pi/2
        for action in avail_actions:
            if np.linalg.norm(nearest_pos - avail_actions[action][:2]) <= nearest_dist_action:
                action_index = action
                nearest_dist_action = np.linalg.norm(nearest_pos - avail_actions[action][:2])
                if abs(avail_actions[action][2] - global_yaw_dis) < 0.02:
                    selected_action = action

        if selected_action != -1:
            return selected_action
        elif action_index != -1:
            return action_index
        else:
            # Return greefy if nothing works
            action = np.random.choice(list(avail_actions.keys()), size=1).item()
            return action

class MinMaxAgent:
    def __init__(self):
        pass

    def act(self, agent, avail_actions):
        # Gets the nearest expected relative position of the target and selects the action which takes it the closest
        # Avail actions is a dictionary with a list of available action and what they do : goal setting , velocity setting#
        # Nearest target
        nearest_pos = np.array([np.inf,np.inf])
        nearest_dist = np.inf
        for i in range(len(agent.belief)):
            target_rel_pos = agent.belief[i].state[:2] - agent.state[:2]
            if nearest_dist > np.linalg.norm(target_rel_pos):
                nearest_dist = np.linalg.norm(target_rel_pos)
                nearest_pos = target_rel_pos

        nearest_dist_action = np.inf
        action_index = -1
        selected_action = -1
        global_yaw = np.arctan2(nearest_pos[1], nearest_pos[0]) + np.pi
        global_yaw_dis = int(global_yaw / (np.pi/2))*np.pi/2
        for action in avail_actions:
            if np.linalg.norm(nearest_pos - avail_actions[action][:2]) <= nearest_dist_action:
                action_index = action
                nearest_dist_action = np.linalg.norm(nearest_pos - avail_actions[action][:2])
                if abs(avail_actions[action][2] - global_yaw_dis) < 0.02:
                    selected_action = action

        if selected_action != -1:
            return selected_action
        elif action_index != -1:
            return action_index
        else:
            # Return greefy if nothing works
            action = np.random.choice(list(avail_actions.keys()), size=1).item()
            return action

class RandomAgent:
    def __init__(self):
        pass

    def act(self, agent, avail_actions):
        # Avail actions is a dictionary with a list of available action and what they do : goal setting , velocity setting
        # Action map in the environment
        action = np.random.choice(list(avail_actions.keys()),size = 1).item()
        return action



def parse_args():
    import argparse
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--policy", type=str, default="random", help="type of agent: random, greedy")

    parser.add_argument("--num_steps", type=int, default=250, help="maximum episode length")

    parser.add_argument('--env', help='environment ID', default='setTracking-v2')
    parser.add_argument('--map', type=str, default="emptyMed")
    parser.add_argument('--nb_agents', type=int, default=4)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--nb_targets', type=int, default=4)
    parser.add_argument('--mode', choices=['train', 'test', 'test-behavior'], default='train')
    parser.add_argument('--record',type=int, default=1)
    parser.add_argument('--render', type=int, default=1)
    parser.add_argument('--nb_test_eps',type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='./results/runs')
    parser.add_argument('--log_fname', type=str, default='last_model.pt')
    parser.add_argument('--repeat', type=int, default=1)


    parser.add_argument("--reward_type", type=str , default='Mean',
                        help="type of reward structure")

    return parser.parse_args()

def run_episode(env, args, policy):

    obs = env.reset()
    global_step = 0
    rewards = np.zeros(shape = (args.num_steps,args.nb_agents))
    dones = np.zeros(shape = (args.num_steps))
    ep_length = 0
    next_done = False
    for step in range(0, args.num_steps):
        global_step += 1

        dones[step] = next_done  # dones[step] = next_done
        # ALGO LOGIC: action logic

        action_dict = [{} for _ in range(args.num_envs)]
        for i in range(args.num_envs):
            agents = env.envs[i].agents
            action_map = env.envs[i].action_map
            agent_keys = env.envs[i].observation_space.keys()
            for j, key in enumerate(agent_keys):
                action_dict[i][key] = policy.act(agents[j], action_map)
        if hasattr(policy, "reset"):
            policy.reset()
        if args.render:
            env.envs[0].render()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = env.step(action_dict)
        rewards[step, :] = reward
        dones[step] = done
        next_done = done
        print("Progress {}".format(step))

    print( "Episode Rewards : {}".format(np.sum(rewards)))

if __name__ =="__main__":
    args = parse_args()
    env = envs.make(args.env,
                    'ma_target_tracking',
                    render = bool(args.render),
                    record = bool(args.record),
                    directory = args.log_dir +"/"+args.policy,
                    map_name = args.map,
                    num_agents = args.nb_agents,
                    num_targets = args.nb_targets,
                    is_training = True,
                    num_envs = args.num_envs,
                    scaled = False,
                    reward_type = args.reward_type
                    )


    policy = None
    # set policy ot greedy or randon based on agent-type
    if args.policy =='Greedy':
        policy = GreedyAgent()
    elif args.policy == 'Random':
        policy = RandomAgent()
    elif args.policy == 'GreedyAssigned':
        policy = GreedyAssignedAgent(num_targets = args.nb_targets)

    run_episode(env, args, policy)

