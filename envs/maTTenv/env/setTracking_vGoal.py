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

DEBUG = False
class setTrackingEnvGoal(setTrackingEnv2):

    def __init__(self, num_agents=1, num_targets=2, map_name='empty',
                        is_training=True, known_noise=True, **kwargs):
        super(setTrackingEnv2,self).__init__(num_agents=num_agents, num_targets=num_targets,
                        map_name=map_name, is_training=is_training)

        self.steps = 0
        self.id = 'setTracking-vGoal'
        self.metadata = self.id
        self.metadata = self.id
        self.scaled = kwargs["scaled"]
        self.reward_type = kwargs["reward_type"]
        self.nb_agents = num_agents  # only for init, will change with reset()
        self.nb_targets = num_targets  # only for init, will change with reset()
        self.agent_dim = 3
        self.target_dim = 4

        self.target_init_vel = METADATA['target_init_vel'] * np.ones((2,))
        
        # Setting simulation time
        self.step_goal = METADATA['step_goal']


        self.dT = METADATA['dT']

        if not self.step_goal:
            self.sampling_period = METADATA['sampling_period']
        else:
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
        for x in np.arange(actions_pos[0],actions_pos[1],actions_pos[2]):
            for y in np.arange(actions_pos[0],actions_pos[1],actions_pos[2]):
                for z in np.arange(actions_yaw[0],actions_yaw[1],actions_yaw[2]):
                    self.action_map[self.num_actions] = (x,y,z)
                    self.num_actions += 1


        self.action_space = spaces.Discrete(self.num_actions)


    def setup_agents(self):
        self.agents = [AgentSE2Goal(agent_id='agent-' + str(i),
                                dim=self.agent_dim, sampling_period=self.sampling_period,
                                limit=self.limit['agent'],
                                collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                                horizon = self.dT)
                       for i in range(self.num_agents)]

    def reset(self):
        obs = super(setTrackingEnvGoal, self).reset()

        for i in range(self.num_agents):
            self.agents[i].state[-1] += np.pi
            self.agents[i].state[-1] = int(self.agents[i].state[-1] / (np.pi / 2.0)) * np.pi

        obs_dict = {}
        for kk in range(self.nb_agents):
            obs_dict[self.agents[kk].agent_id] = self.observe_single(kk)

        return obs_dict

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
            if DEBUG:
                for k,v in self.action_map.items():
                    self.agents[ii].set_goals(v,margin_pos,DEBUG)


            planners = self.agents[ii].set_goals(control_goal = action_vw,
                                                 margin_pos = margin_pos,
                                                 step_goal = self.step_goal)
            planners_dict[ii] = planners

        observed, reward, reward_dict, mean_logdetcov = self.step_single(planners_dict)

        # Get all rewards after all agents and targets move (t -> t+1)
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = self.observe_single(
                ii,  action_vw=self.action_map[action_dict[agent_id]],
                isObserved=observed[ii]
            )
        # Get all rewards after all agents and targets move (t -> t+1)

        done = False
        done_dict['__all__'], info_dict['mean_nlogdetcov'] = done, mean_logdetcov

        info_dict['reward_all'] = reward_dict
        info_dict['metrics'] = [self.calculate_total_uncertainity(), self.calculate_max_uncertainity()]
        self.steps += 1
        return obs_dict, reward, done, info_dict

    def observe_single(self,agentID,action_vw = None,isObserved = None):
        observation = []
        for jj in range(self.nb_targets):
            r, alpha = util.relative_distance_polar(self.agents[agentID].belief[jj].state[:2],
                                                    xy_base=self.agents[agentID].state[:2],
                                                    theta_base=self.agents[agentID].state[2])
            if action_vw is None:
                r_dot_b,alpha_dot_b = 0.0,0.0
            else:
                r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                    self.agents[agentID].belief[jj].state[:2],
                    self.agents[agentID].belief[jj].state[2:],
                    self.agents[agentID].state[:2], self.agents[agentID].state[-1],
                    action_vw[0], action_vw[1])

            logdetcov = np.log(LA.det(self.agents[agentID].belief[jj].cov))
            if action_vw is None:
                observed = 0.0
            else:
                observed = float(isObserved[jj])

            observation.append([self.agents[agentID].state[0]/self.MAP.mapmax[0],
                                self.agents[agentID].state[1]/self.MAP.mapmax[0],
                                r, alpha, r_dot_b, alpha_dot_b, logdetcov, observed])

        return torch.tensor(observation,dtype=torch.float32)

    def step_single(self,planners_dict):
        '''
        :param planners_dict: Goal based planners for each agent or just the ooal for rach agent
        :return: return whether all agents have reached the goals
        '''

        # Time increments to step low level planner
        collision = [False for _ in range(len(planners_dict))]
        observed = np.zeros((self.nb_agents, self.nb_targets), dtype=bool)
        mean_reward = 0.0
        mean_reward_dict = np.array([0.0 for _ in range(len(planners_dict.keys()))])
        mean_mean_logdetcov = None if self.is_training else 0.0


        # Low level control
        if not self.step_goal:
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

                coverage_reward = np.array(self.update_global_coverage())
                reward, reward_dict, done, mean_nlogdetcov = self.get_reward(observed, self.is_training)

                mean_reward += reward
                mean_reward_dict += (reward_dict + self.coverage_reward_factor*coverage_reward)


                if not self.is_training:
                    mean_mean_logdetcov += mean_nlogdetcov
        else:
            # update target

            for j in range((int(self.dT / self.sampling_period))):
                for i in range(self.nb_targets):
                    # update target
                    self.targets[
                        i].update()  # self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))
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
                state = np.zeros(len(self.agents[agent_id].state))
                state[:2] = self.agents[agent_id].state[:2] + planners_dict[agent_id][:2]
                state[-1] = planners_dict[agent_id][-1]
                action_vw = np.array([0,0])
                # Locations of all targets and agents in order to maintain a margin between them
                margin_pos = [t.state[:2] for t in self.targets[:self.nb_targets]]
                for p, ids in enumerate(planners_dict):
                    if agent_id != ids:
                        margin_pos.append(np.array(self.agents[p].state[:2]))

                if not collision[ii]:
                    collision[ii] = self.agents[ii].update((state, action_vw), margin_pos, step_goal = self.step_goal)

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

            coverage_reward = np.array(self.update_global_coverage())
            reward, reward_dict, done, mean_nlogdetcov = self.get_reward(observed, self.is_training)

            mean_reward += reward
            mean_reward_dict += (reward_dict + self.coverage_reward_factor * coverage_reward)

            if not self.is_training:
                mean_mean_logdetcov += mean_nlogdetcov


        mean_reward /= (int(self.dT/self.sampling_period))
        mean_reward_dict = mean_reward_dict / (int(self.dT/self.sampling_period))

        if not self.is_training:
            mean_mean_logdetcov /= (int(self.dT/self.sampling_period))

        return observed, reward, mean_reward_dict,  mean_mean_logdetcov

        # reward, reward_dict, done, mean_nlogdetcov = self.get_reward(observed, self.is_training)
    def update_global_coverage(self):
        agent_list_id = np.arange(len(self.agents))
        random.shuffle(agent_list_id)
        decay = np.exp(np.array([-1 / 40]))
        self.global_coverage_map = np.copy(self.global_coverage_map) * decay
        square_side_divided_by_2 = METADATA['sensor_r'] / 2 * np.sqrt(np.pi)
        rectangles = []
        coverage_rew_dict = np.zeros(shape=len(self.agents))

        for j in agent_list_id:
            # import pdb; pdb.set_trace()
            x, y = self.agents[j].state[0], self.agents[j].state[1]
            r1 = int(x - square_side_divided_by_2)
            c1 = int(y - square_side_divided_by_2)
            r2 = int(x + square_side_divided_by_2)
            c2 = int(y + square_side_divided_by_2)
            r1 = r1 if r1 > 0 else 0
            c1 = c1 if c1 > 0 else 0
            r2 = r2 if r2 < self.MAP.mapmax[0] else self.MAP.mapmax[0]
            c2 = c2 if c2 < self.MAP.mapmax[1] else self.MAP.mapmax[1]

            rectangles_x, rectangles_y = np.meshgrid(np.arange(self.MAP.mapmax[0]), np.arange(self.MAP.mapmax[1]))
            rectangles = np.logical_and(np.logical_and(rectangles_x >= r1, rectangles_x <= r2)
                                        , np.logical_and(rectangles_y >= c1, rectangles_y <= c2))
            coverage_reward = self.update_global_coverage_map(rectangles, np.exp(np.array([-1 / 40])))
            coverage_rew_dict[j] = coverage_reward / (np.prod(self.MAP.mapmax))
            # self.draw_circle(grid, agent.state[0], agent.state[1], METADATA['sensor_r'])
            # sensor_footprint = union_rectangles(rectangles)

        return coverage_rew_dict

    def get_reward(self, observed=None, is_training=True):
        return self.reward_fun(self.agents, self.nb_targets, self.belief_targets, is_training, c_mean=0.1,
                               scaled=self.scaled)

    def reward_fun(self, agents, nb_targets, belief_targets, is_training=True, c_mean=0.05, scaled=False):
        # TODO: reward should be per agent
        globaldetcov = [LA.det(b_target.cov) for b_target in belief_targets]

        globaldetcov = np.ravel(globaldetcov)

        r_detcov_sum = - np.sum(np.log(globaldetcov))
        reward = c_mean * r_detcov_sum

        ## discretize grid
        # grid = torch.zeros(self.MAP.mapmax[0], self.MAP.mapmax[1])
        ## find occupied cells by all agent's sensor radius


        # randomising the coverage so that no agent is given priority
        coverage_rew_dict = np.zeros(shape=len(self.agents))

        reward_dict = []
        if self.reward_type == "Max":
            for id, agent in enumerate(self.agents):
                detcov = [LA.det(b.cov) for b in agent.belief]
                detcov = np.ravel(detcov)
                if is_training:
                    detcov_max = - c_mean*np.log(np.max(detcov))
                    # print("centralized")
                else:
                    detcov_max = - c_mean*np.log(np.max(detcov))
                    # print("individual")
                reward_dict.append(detcov_max)
        elif self.reward_type == "Mean":
            for id, agent in enumerate(self.agents):
                detcov = [LA.det(b.cov) for b in agent.belief]
                detcov = np.ravel(detcov)
                detcov_max = - c_mean * np.log(detcov).mean()
                reward_dict.append( detcov_max)

        mean_nlogdetcov = None
        if not (is_training):
            logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets[:nb_targets]]
            mean_nlogdetcov = -np.mean(logdetcov)
        return reward, np.array(reward_dict), False, mean_nlogdetcov