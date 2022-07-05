import os, copy, pdb
import numpy as np
from numpy import linalg as LA
import torch
from gym import spaces, logger
from envs.maTTenv.maps import map_utils
import envs.maTTenv.util as util 
from envs.maTTenv.agent_models import *
from envs.maTTenv.belief_tracker import KFbelief
from envs.maTTenv.metadata import METADATA
from envs.maTTenv.env.maTracking_Base import maTrackingBase

"""
Target Tracking Environments for Reinforcement Learning.
[Variables]

d: radial coordinate of a belief target in the learner frame
alpha : angular coordinate of a belief target in the learner frame
ddot : radial velocity of a belief target in the learner frame
alphadot : angular velocity of a belief target in the learner frame
Sigma : Covariance of a belief target

[Environment Description]
Varying number of agents, varying number of randomly moving targets
No obstacles

setTrackingEnv0 : Double Integrator Target model with KF belief tracker
    obs state: [d, alpha, ddot, alphadot, logdet(Sigma), observed] *nb_targets
            where nb_targets and nb_agents vary between a range
            num_targets describes the upperbound on possible number of targets in env
            num_agents describes the upperbound on possible number of agents in env
    Target : Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model

"""

class setTrackingEnv2(maTrackingBase):

    def __init__(self, num_agents=1, num_targets=2, map_name='empty', 
                        is_training=True, known_noise=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets,
                        map_name=map_name, is_training=is_training)

        self.id = 'setTracking-v2'
        self.metadata = self.id
        self.nb_agents = num_agents #only for init, will change with reset()
        self.nb_targets = num_targets #only for init, will change with reset()
        self.agent_dim = 3
        self.target_dim = 4
        self.target_init_vel = METADATA['target_init_vel']*np.ones((2,))
        # LIMIT
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-METADATA['target_vel_limit'], -METADATA['target_vel_limit']])),
                                np.concatenate((self.MAP.mapmax, [METADATA['target_vel_limit'], METADATA['target_vel_limit']]))]
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0] # Maximum relative speed
        self.limit['state'] = [np.array(([0.0, -np.pi, -rel_vel_limit, -10*np.pi, -50.0, 0.0])),
                               np.array(([600.0, np.pi, rel_vel_limit, 10*np.pi, 50.0, 2.0]))]
        
        observation_space = spaces.Box(np.tile(self.limit['state'][0], (self.num_targets, 1)), np.tile(self.limit['state'][1], (self.num_targets, 1)), dtype=np.float32)
        self.observation_space = {f"agent-{i}":observation_space for i in range(self.nb_agents)}
        self.observation_space = spaces.Dict(self.observation_space)
        self.targetA = np.concatenate((np.concatenate((np.eye(2), self.sampling_period*np.eye(2)), axis=1), 
                                        [[0,0,1,0],[0,0,0,1]]))
        self.target_noise_cov = METADATA['const_q'] * np.concatenate((
                        np.concatenate((self.sampling_period**3/3*np.eye(2), self.sampling_period**2/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.concatenate((
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))

        # Build a robot 
        self.setup_agents()
        # Build a target
        self.setup_targets()
        self.setup_belief_targets()
        # Use custom reward
        self.get_reward()

    def setup_agents(self):
        self.agents = [AgentSE2(agent_id = 'agent-' + str(i), 
                        dim=self.agent_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['agent'], 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_agents)]

    def setup_targets(self):
        self.targets = [AgentDoubleInt2D(agent_id = 'target-' + str(i),
                        dim=self.target_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['target'],
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                        A=self.targetA, W=self.target_true_noise_sd) 
                        for i in range(self.num_targets)]

    def setup_belief_targets(self):
        self.belief_targets = [[KFbelief(agent_id = f"agent-{i}_target-{j}",
                        dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                        W=self.target_noise_cov, obs_noise_func=self.observation_noise, 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for j in range(self.num_targets)]
                        for i in range(self.num_agents)]

    def get_reward(self, observed=None, is_training=True):
        return reward_fun(self.nb_targets, self.belief_targets, is_training)

    def get_init_pose_random(self,
                            lin_dist_range_target=(METADATA['init_distance_min'], METADATA['init_distance_max']),
                            ang_dist_range_target=(-np.pi, np.pi),
                            lin_dist_range_belief=(METADATA['init_belief_distance_min'], METADATA['init_belief_distance_max']),
                            ang_dist_range_belief=(-np.pi, np.pi),
                            blocked=False,
                            **kwargs):
        is_agent_valid = False
        init_pose = {}
        init_pose['agents'] = []
        init_pose['belief_targets'] = [[] for _ in range(self.nb_agents)]
        for ii in range(self.nb_agents):
            is_agent_valid = False
            if self.MAP.map is None and ii==0:
                if blocked:
                    raise ValueError('Unable to find a blocked initial condition. There is no obstacle in this map.')
                a_init = self.agent_init_pos[:2]
            else:
                while(not is_agent_valid):
                    a_init = np.random.random((2,)) * (self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin
                    is_agent_valid = not(map_utils.is_collision(self.MAP, a_init))
            init_pose_agent = [a_init[0], a_init[1], np.random.random() * 2 * np.pi - np.pi]
            init_pose['agents'].append(init_pose_agent)

        init_pose['targets'] = []
        for jj in range(self.nb_targets):
            is_target_valid = False
            while(not is_target_valid):
                rand_agent = np.random.randint(self.nb_agents)
                is_target_valid, init_pose_target = self.gen_rand_pose(
                    init_pose['agents'][rand_agent][:2], init_pose['agents'][rand_agent][2],
                    lin_dist_range_target[0], lin_dist_range_target[1],
                    ang_dist_range_target[0], ang_dist_range_target[1])
            init_pose['targets'].append(init_pose_target)
            for kk in range(self.nb_agents):
                is_belief_valid, init_pose_belief = False, np.zeros((2,))
                while((not is_belief_valid) and is_target_valid):
                    is_belief_valid, init_pose_belief = self.gen_rand_pose(
                        init_pose['targets'][jj][:2], init_pose['targets'][jj][2],
                        lin_dist_range_belief[0], lin_dist_range_belief[1],
                        ang_dist_range_belief[0], ang_dist_range_belief[1])
                init_pose['belief_targets'][kk].append(init_pose_belief)
        return init_pose
    
    def reset(self,**kwargs):
        """
        Random initialization a number of agents and targets at the reset of the env epsiode.
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return an observation state dict with agent ids (keys) that refer to their observation
        """
        self.rng = np.random.default_rng()
        obs_dict = {}
        init_pose = self.get_init_pose(**kwargs)
        # Initialize agents
        for ii in range(self.nb_agents):
            self.agents[ii].reset(init_pose['agents'][ii])
            obs_dict[self.agents[ii].agent_id] = []
        #from IPython import embed; embed()
        # Initialize targets and beliefs
        for i in range(self.nb_targets):
            # reset target
            self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))
            for j in range(self.nb_agents):
                # reset belief target
                self.belief_targets[j][i].reset(
                            init_state=np.concatenate((init_pose['belief_targets'][j][i][:2], np.zeros(2))),
                            init_cov=self.target_init_cov)
                

        # For nb agents calculate belief of targets assigned
        for jj in range(self.nb_targets):
            for kk in range(self.nb_agents):
                r, alpha = util.relative_distance_polar(self.belief_targets[kk][jj].state[:2],
                                            xy_base=self.agents[kk].state[:2], 
                                            theta_base=self.agents[kk].state[2])
                logdetcov = np.log(LA.det(self.belief_targets[kk][jj].cov))
                obs_dict[self.agents[kk].agent_id].append([r, alpha, 0.0, 0.0, logdetcov, 0.0])
        for agent_id in obs_dict:
            obs_dict[agent_id] = torch.Tensor(obs_dict[agent_id])
        return obs_dict

    def step(self, action_dict):
        obs_dict = {}
        reward_dict = {}
        done_dict = {'__all__':False}
        info_dict = {}

        # Targets move (t -> t+1)
        for i in range(self.nb_targets):
            # update target
            self.targets[i].update() # self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))
            for j in range(self.nb_agents):
                self.belief_targets[j][i].predict() # Belief state at t+1
        # Target and map observations
        observed = np.zeros((self.nb_agents, self.nb_targets), dtype=bool)
            
        # Agents move (t -> t+1) and observe the targets
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            action_vw = self.action_map[action_dict[agent_id]]

            # Locations of all targets and agents in order to maintain a margin between them
            margin_pos = [t.state[:2] for t in self.targets[:self.nb_targets]]
            for p, ids in enumerate(action_dict):
                if agent_id != ids:
                    margin_pos.append(np.array(self.agents[p].state[:2]))
            _ = self.agents[ii].update(action_vw, margin_pos)
            
            

            # Update beliefs of targets
            for jj in range(self.nb_targets):
                # Observe
                obs, z_t = self.observation(self.targets[jj], self.agents[ii])
                observed[ii][jj] = obs
                if obs: # if observed, update the target belief.
                    self.belief_targets[ii][jj].update(z_t, self.agents[ii].state)

                r_b, alpha_b = util.relative_distance_polar(self.belief_targets[ii][jj].state[:2],
                                        xy_base=self.agents[ii].state[:2], 
                                        theta_base=self.agents[ii].state[-1])
                r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                                        self.belief_targets[ii][jj].state[:2],
                                        self.belief_targets[ii][jj].state[2:],
                                        self.agents[ii].state[:2], self.agents[ii].state[-1],
                                        action_vw[0], action_vw[1])
                obs_dict[agent_id].append([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                        np.log(LA.det(self.belief_targets[ii][jj].cov)), float(obs)])
            obs_dict[agent_id] = torch.Tensor(obs_dict[agent_id])
            # shuffle obs to promote permutation invariance
            self.rng.shuffle(obs_dict[agent_id])
        # Get all rewards after all agents and targets move (t -> t+1)
        reward, done, mean_nlogdetcov = self.get_reward(observed, self.is_training)
        done_dict['__all__'], info_dict['mean_nlogdetcov'] = done, mean_nlogdetcov
        return obs_dict, reward, done_dict, info_dict

def reward_fun(nb_targets, belief_targets, is_training=True, c_mean=0.1):
    detcov = [[LA.det(b_target.cov) for b_target in belief_target] for belief_target in belief_targets]
    detcov = np.ravel(detcov)
    r_detcov_mean = - np.mean(np.log(detcov))
    reward = c_mean * r_detcov_mean

    mean_nlogdetcov = None
    if not(is_training):
        logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets[:nb_targets]]
        mean_nlogdetcov = -np.mean(logdetcov)
    return reward, False, mean_nlogdetcov