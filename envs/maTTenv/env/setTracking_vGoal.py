import os, copy, pdb
import math
import random

import numpy as np
from numpy import linalg as LA
import torch
from gym import spaces, logger
from envs.maTTenv.maps import map_utils
import envs.maTTenv.util as util
from envs.maTTenv.agent_models import *
from envs.maTTenv.belief_tracker import KFbelief
from envs.maTTenv.metadata import METADATA
from envs.maTTenv.env.setTracking_v2 import setTrackingEnv2

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

class setTrackingEnvGoal(setTrackingEnv2):

    def __init__(self, num_agents=1, num_targets=2, map_name='empty',
                        is_training=True, known_noise=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets,
                        map_name=map_name, is_training=is_training)

        self.steps = 0
        self.id = 'setTracking-vGoal'
        self.metadata = self.id
        self.scaled = kwargs["scaled"]
        self.reward_type = kwargs["reward_type"]
        self.nb_agents = num_agents  # only for init, will change with reset()
        self.nb_targets = num_targets  # only for init, will change with reset()


        self.target_init_vel = METADATA['target_init_vel'] * np.ones((2,))
        
        # Setting simulation time
        self.dT = METADATA['dT']
        self.sampling_period = METADATA['sampling_period']


        # LIMIT
        self.limit = {}  # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [
            np.concatenate((self.MAP.mapmin, [-METADATA['target_vel_limit'], -METADATA['target_vel_limit']])),
            np.concatenate((self.MAP.mapmax, [METADATA['target_vel_limit'], METADATA['target_vel_limit']]))]
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0]  # Maximum relative speed

        # normalize
        self.limit['state'] = [np.array(([0.0, 0.0, 0.0, -np.pi, -rel_vel_limit, -10 * np.pi, -50.0, 0.0])),
                               np.array(([1.0, 1.0, 600.0, np.pi, rel_vel_limit, 10 * np.pi, 50.0, 2.0]))]

        self.communication_range = METADATA['comms_r']

        observation_space = spaces.Box(np.tile(self.limit['state'][0], (self.num_targets, 1)),
                                       np.tile(self.limit['state'][1], (self.num_targets, 1)), dtype=np.float32)
        self.observation_space = {f"agent-{i}": observation_space for i in range(self.nb_agents)}
        self.observation_space = spaces.Dict(self.observation_space)
        self.targetA = np.concatenate((np.concatenate((np.eye(2), self.sampling_period * np.eye(2)), axis=1),
                                       [[0, 0, 1, 0], [0, 0, 0, 1]]))
        self.target_noise_cov = METADATA['const_q'] * np.concatenate((
            np.concatenate((self.sampling_period ** 3 / 3 * np.eye(2), self.sampling_period ** 2 / 2 * np.eye(2)),
                           axis=1),
            np.concatenate((self.sampling_period ** 2 / 2 * np.eye(2), self.sampling_period * np.eye(2)), axis=1)))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.concatenate((
                np.concatenate((self.sampling_period ** 2 / 2 * np.eye(2), self.sampling_period / 2 * np.eye(2)),
                               axis=1),
                np.concatenate((self.sampling_period / 2 * np.eye(2), self.sampling_period * np.eye(2)), axis=1)))

        # Coverage rewards in the environment
        self.coverage_reward_factor = METADATA['coverage_reward']
        self.global_coverage_map = np.zeros(shape=self.MAP.mapmax)

        # Build a robot
        self.setup_agents()

        # Build a target
        self.setup_targets()
        self.setup_belief_targets()

        self.action_map = {}
        actions_pos = METADATA['actions_pos']
        actions_yaw = METADATA['actions_yaw']
        self.num_actions = 0

        # Setting the action map configurations
        for x in range(start=actions_pos[0],end = actions_pos[1],step=actions_pos[2]):
            for y in range(start=actions_pos[0], end=actions_pos[1], step=actions_pos[2]):
                for z in range(start=actions_yaw[0], end=actions_yaw[1], step=actions_yaw[2]):
                    self.num_actions+=1
                    self.action_map[self.num_actions] = (x,y,z)


        self.action_space = spaces.Discrete(self.num_actions)


    def setup_agents(self):
        self.agents = [AgentSE2Goal(agent_id='agent-' + str(i),
                                dim=self.agent_dim, sampling_period=self.sampling_period,
                                limit=self.limit['agent'],
                                collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                                horizon = self.dT)
                       for i in range(self.num_agents)]


    def step(self, action_dict):
        obs_dict = {}
        reward_dict = {}
        done_dict = {'__all__': False}
        info_dict = {}

        planners_dict = {}
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []
            action_vw = self.action_map[action_dict[agent_id]]

            margin_pos = [t.state[:2] for t in self.targets[:self.nb_targets]]
            for p, ids in enumerate(action_dict):
                if agent_id != ids:
                    margin_pos.append(np.array(self.agents[p].state[:2]))
            planners_dict = self.agents[ii].set_goals(action_vw, margin_pos)

            obs_dict[self.agents[ii].agent_id] = self.observe_single(ii, action_vw=action_vw, isObserved=observed[ii])

        # Get all rewards after all agents and targets move (t -> t+1)
        reward, reward_dict, done, mean_nlogdetcov = self.get_reward(observed, self.is_training)
        done_dict['__all__'], info_dict['mean_nlogdetcov'] = done, mean_nlogdetcov

        info_dict['reward_all'] = reward_dict
        info_dict['metrics'] = [self.calculate_total_uncertainity(), self.calculate_max_uncertainity()]
        self.steps += 1
        return obs_dict, reward, done, info_dict

    def step_single(self,planners_dict):
        '''
        :param planners_dict: Goal based planners for each agent
        :return: return whether all agents have reached the goals
        '''

        # Time increments to step low level planner
        collision = [False for _ in range(len(planners_dict))]

        for j in range((int(self.dT/self.sampling_period))):
            for i in range(self.nb_targets):
                # update target
                self.targets[i].update()  # self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))
                for j in range(self.nb_agents):
                    self.agents[j].belief[i].predict()

                self.belief_targets[i].predict()  # Belief state at t+1
            # Target and map observations
            observed = np.zeros((self.nb_agents, self.nb_targets), dtype=bool)

            # Communication
            agent_graph = self.communicate_graph()

            update_comm_beliefs = []

            for id in agent_graph.keys():
                comm_recv_beliefs = [self.agents[ID].belief for ID in agent_graph[id]]
                update_comm_beliefs.append(self.agents[id].updateCommBelief(comm_recv_beliefs))

            for agentid, updatedCommBelief in enumerate(update_comm_beliefs):
                self.agents[agentid].setupBelief(updatedCommBelief)

            # Agents move (t -> t+1) and observe the targets

            for ii, agent_id in enumerate(planners_dict):

                # returns an action

                state, action_vw = planners_dict[agent_id].get_controller(j*self.sampling_period)

                # Locations of all targets and agents in order to maintain a margin between them
                margin_pos = [t.state[:2] for t in self.targets[:self.nb_targets]]
                for p, ids in enumerate(planners_dict):
                    if agent_id != ids:
                        margin_pos.append(np.array(self.agents[p].state[:2]))

                if not collision[ii]:
                    collision[ii] = self.agents[ii].update((state,action_vw), margin_pos)

                # Update beliefs of targets
                for jj in range(self.nb_targets):
                    # Observe
                    obs, z_t = self.observation(self.targets[jj], self.agents[ii])
                    observed[ii][jj] = obs
                    if obs:  # if observed, update the target belief.
                        # Update agents indivuudla beliefs based on observation
                        self.agents[ii].updateBelief(targetID=jj, z_t=z_t)

                        # Update global belief
                        self.belief_targets[jj].update(z_t, self.agents[ii].state)



