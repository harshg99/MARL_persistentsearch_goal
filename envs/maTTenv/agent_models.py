"""Dynamic Object Models

AgentDoubleInt2D : Double Integrator Model in 2D
                   state: x,y,xdot,ydot
AgentSE2 : SE2 Model 
           state x,y,theta

Agent2DFixedPath : Model with a pre-defined path
Agent_InfoPlanner : Model from the InfoPlanner repository

SE2Dynamics : update dynamics function with a control input -- linear, angular velocities
SEDynamicsVel : update dynamics function for contant linear and angular velocities

edited by christopher-hsu from coco66 for multi_agent
"""

import numpy as np
from envs.maTTenv.metadata import METADATA
import envs.maTTenv.util as util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from copy import deepcopy
from numpy import linalg as LA
from trajax.integrators import rk4
from trajax import optimizers
import functools
import jax
import jax.numpy as jnp

class Agent(object):
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func, margin=METADATA['margin'],KFBelief = None):
        self.agent_id = agent_id
        self.dim = dim
        self.sampling_period = sampling_period
        self.limit = limit
        self.collision_func = collision_func
        self.margin = margin
        self.belief = KFBelief
        self.coverage_map = None

    def setupBelief(self,belief):
        #List of belief over all targets
        self.belief = deepcopy(belief)


    def updateBelief(self,targetID = None,z_t=None):
        if targetID is not None and z_t is not None:
            self.belief[targetID].update(z_t,self.state)

    def update_coverage_map(self,rectangle_map,decay):
        if self.coverage_map is not None:
            coverage_map2= np.copy(self.coverage_map)*decay
            coverage_map2[rectangle_map] = 1.0
            coverage_reward = np.sum(coverage_map2) - np.sum(self.coverage_map)
            self.coverage_map = deepcopy(coverage_map2)
        else:
            self.coverage_map = deepcopy(rectangle_map)
            coverage_reward = np.sum(self.coverage_map)

        return coverage_reward

    def updateCommBelief(self,comms_recv_beliefs):
        '''

        :param comms_recv_beliefs: List of KF/UKF belifs from communicated agents
        :return: intermediate update beliefs
        '''

        intermediateBelief = deepcopy(self.belief)
        if len(comms_recv_beliefs)==0:
            return intermediateBelief

        # for each belief target
        for target_id in range(len(intermediateBelief)):
            # for each neighbor, use min cov
            for neigh in range(len(comms_recv_beliefs)):
                logdetCov = LA.det(intermediateBelief[target_id].cov)
                commslogdetCov = LA.det(comms_recv_beliefs[neigh][target_id].cov)
                if commslogdetCov<logdetCov:
                    # update list
                    intermediateBelief[target_id].state = deepcopy(comms_recv_beliefs[neigh][target_id].state)
                    intermediateBelief[target_id].cov = deepcopy(comms_recv_beliefs[neigh][target_id].cov)

        return intermediateBelief

    def range_check(self):
        self.state = np.clip(self.state, self.limit[0], self.limit[1])

    def collision_check(self, pos):
        return self.collision_func(pos[:2])

    def margin_check(self, pos, target_pos):
        return any(np.sqrt(np.sum((pos - target_pos)**2, axis=1)) < self.margin) # no update

    def reset(self, init_state):
        self.state = init_state

class AgentSE2(Agent):
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func, 
                        margin=METADATA['margin'], policy=None):
        super().__init__(agent_id, dim, sampling_period, limit, collision_func, margin=margin)
        self.policy = policy

    def reset(self, init_state):
        super().reset(init_state)
        self.vw = [0.0, 0.0]
        if self.policy:
            self.policy.reset(init_state)

    def update(self, control_input=None, margin_pos=None, col=False):
        """
        control_input : [linear_velocity, angular_velocity]
        margin_pos : a minimum distance to a target
        """
        if control_input is None:
            control_input = self.policy.get_control(self.state)

        new_state = SE2Dynamics(self.state, self.sampling_period, control_input) 
        is_col = 0 
        if self.collision_check(new_state[:2]):
            is_col = 1
            new_state[:2] = self.state[:2]
            control_input = self.vw
            if self.policy is not None:
                # self.policy.collision(new_state)
                corrected_policy = self.policy.collision(new_state)
                if corrected_policy is not None:
                    new_state = SE2DynamicsVel(self.state,
                                        self.sampling_period, corrected_policy)

        elif margin_pos is not None:
            if self.margin_check(new_state[:2], margin_pos):
                new_state[:2] = self.state[:2]
                control_input = self.vw
       
        self.state = new_state
        self.vw = control_input
        self.range_check()        

        return is_col

def SE2Dynamics(x, dt, u):  
    assert(len(x)==3)          
    tw = dt * u[1] 
    # Update the agent state 
    if abs(tw) < 0.001:
        diff = np.array([dt*u[0]*np.cos(x[2]+tw/2),
                dt*u[0]*np.sin(x[2]+tw/2),
                tw])
    else:
        diff = np.array([u[0]/u[1]*(np.sin(x[2]+tw) - np.sin(x[2])),
                u[0]/u[1]*(np.cos(x[2]) - np.cos(x[2]+tw)),
                tw])
    new_x = x + diff
    new_x[2] = util.wrap_around(new_x[2])
    return new_x


def SE2DynamicsRK4(x, u, t):

    del t
    return jnp.array([u[0]*jnp.cos(x[2]),u[0]*jnp.sin(x[2]),u[1]])

def SE2DynamicsVel(x, dt, u=None):  
    assert(len(x)==5) # x = [x,y,theta,v,w]
    if u is None:
        u = x[-2:]
    odom = SE2Dynamics(x[:3], dt, u)
    return np.concatenate((odom, u))

class AgentDoubleInt2D(Agent):
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func, 
                    margin=METADATA['margin'], A=None, W=None):
        super().__init__(agent_id, dim, sampling_period, limit, collision_func, margin=margin)
        self.A = np.eye(self.dim) if A is None else A
        self.W = W

    def update(self, margin_pos=None):
        new_state = np.matmul(self.A, self.state)
        if self.W is not None:
            noise_sample = np.random.multivariate_normal(np.zeros(self.dim,), self.W)
            new_state += noise_sample
        if self.collision_check(new_state[:2]):
            new_state = self.collision_control(new_state)

        self.state = new_state
        self.range_check()

    def collision_control(self, new_state):
        new_state[0] = self.state[0]
        new_state[1] = self.state[1]
        if self.dim > 2:
            new_state[2] = -2 * .2 * new_state[2] + np.random.normal(0.0, 0.2)#-0.001*np.sign(new_state[2])
            new_state[3] = -2 * .2 * new_state[3] + np.random.normal(0.0, 0.2)#-0.001*np.sign(new_state[3])
        return new_state

class AgentDoubleInt2D_Avoidance(AgentDoubleInt2D):
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func,
                    margin=METADATA['margin'], A=None, W=None, obs_check_func=None):
        super().__init__(agent_id, dim, sampling_period, limit,
            collision_func, margin=margin, A=A, W=W)
        self.obs_check_func = obs_check_func

    def update(self, margin_pos=None):
        new_state = np.matmul(self.A, self.state[:self.dim])
        if self.W is not None:
            noise_sample = np.random.multivariate_normal(np.zeros(self.dim,), self.W)
            new_state += noise_sample

        is_col = 0
        if self.collision_check(new_state[:2]):
            new_state = self.collision_control()
            is_col = 1

        if self.obs_check_func is not None:
            del_vx, del_vy = self.obstacle_detour_maneuver(
                    r_margin=METADATA['target_speed_limit']*self.sampling_period*2)
            new_state[2] += del_vx
            new_state[3] += del_vy

        self.state = new_state
        self.range_check()
        return is_col

    def range_check(self):
        """
        Limit the position and the velocity.
        self.limit[:][2] = self.limit[:][3] = speed limit. The velocity components
        are clipped proportional to the original values.
        """
        self.state[:2] = np.clip(self.state[:2], self.limit[0][:2], self.limit[1][:2])
        v_square = self.state[2:]**2
        del_v = np.sum(v_square) - self.limit[1][2]**2
        if del_v > 0.0:
            self.state[2] = np.sign(self.state[2]) * np.sqrt(max(0.0,
                v_square[0] - del_v * v_square[0] / (v_square[0] + v_square[1])))
            self.state[3] = np.sign(self.state[3]) * np.sqrt(max(0.0,
                v_square[1] - del_v * v_square[1] / (v_square[0] + v_square[1])))

    def collision_control(self):
        """
        Assigns a new velocity deviating the agent with an angle (pi/2, pi) from
        the closest obstacle point.
        """
        odom = [self.state[0], self.state[1], np.arctan2(self.state[3], self.state[2])]
        # obs_pos = self.obs_check_func(odom)
        v = np.sqrt(np.sum(np.square(self.state[2:]))) + np.random.normal(0.0,1.0)
        if obs_pos[1] >= 0:
            th = obs_pos[1] - (1 + np.random.random()) * np.pi/2
        else:
            th = obs_pos[1] + (1 + np.random.random()) * np.pi/2

        state = np.array([self.state[0], self.state[1], v * np.cos(th + odom[2]), v * np.sin(th + odom[2])])
        return state

    def obstacle_detour_maneuver(self, r_margin=1.0):
        """
        Returns del_vx, del_vy which will be added to the new state.
        This provides a repultive force from the closest obstacle point based
        on the current velocity, a linear distance, and an angular distance.

        Parameters:
        ----------
        r_margin : float. A margin from an obstalce that you want to consider
        as the minimum distance the target can get close to the obstacle.
        """
        odom = [self.state[0], self.state[1], np.arctan2(self.state[3], self.state[2])]
        # obs_pos = self.obs_check_func(odom)
        speed = np.sqrt(np.sum(self.state[2:]**2))
        rot_ang = np.pi/2 * (1. + 1./(1. + np.exp(-(speed-0.5*METADATA['target_speed_limit']))))
        if obs_pos is not None:
            acc = max(0.0, speed * np.cos(obs_pos[1])) / max(METADATA['margin2wall'], obs_pos[0] - r_margin)
            th = obs_pos[1] - rot_ang if obs_pos[1] >= 0 else obs_pos[1] + rot_ang
            del_vx = acc * np.cos(th + odom[2]) * self.sampling_period
            del_vy = acc * np.sin(th + odom[2]) * self.sampling_period
            return del_vx, del_vy
        else:
            return 0., 0.


class Agent2DFixedPath(Agent):
    def __init__(self, dim, sampling_period, limit, collision_func, path, margin=METADATA['margin']):
        Agent.__init__(self, dim, sampling_period, limit, collision_func, margin=margin)
        self.path = path

    def update(self, margin_pos=None):
        # fixed policy for now
        self.t += 1
        new_state = np.concatenate((self.path[self.t][:2], self.path[self.t][-2:]))
        if margin_pos is not None:
            if type(margin_pos) != list:
                margin_pos = [margin_pos]
            in_margin = False
            while(True):
                in_margin = self.margin_check(new_state[:2], margin_pos)
                if in_margin:
                    new_state[:2] = new_state[:2] + 0.01*(np.random.random((2,))-0.5)
                else:
                    break
        self.state = new_state

    def reset(self, init_state):
        self.t = 0
        self.state = np.concatenate((self.path[self.t][:2], self.path[self.t][-2:]))


class AgentDoubleInt2D_Nonlinear(AgentDoubleInt2D):
    def __init__(self, dim, sampling_period, limit, collision_func,
                    margin=METADATA['margin'], A=None, W=None, obs_check_func=None):
        AgentDoubleInt2D.__init__(self, dim, sampling_period, limit,
            collision_func, margin=margin, A=A, W=W)
        self.obs_check_func = obs_check_func

    def update(self, margin_pos=None):
        new_state = np.matmul(self.A, self.state[:self.dim])
        if self.W is not None:
            noise_sample = np.random.multivariate_normal(np.zeros(self.dim,), self.W)
            new_state += noise_sample

        is_col = 0
        if self.collision_check(new_state[:2]):
            new_state = self.collision_control()
            is_col = 1

        if self.obs_check_func is not None:
            del_vx, del_vy = self.obstacle_detour_maneuver(
                    r_margin=METADATA['target_speed_limit']*self.sampling_period*2)
            new_state[2] += del_vx
            new_state[3] += del_vy

        self.state = new_state
        self.range_check()
        return is_col

    def range_check(self):
        """
        Limit the position and the velocity.
        self.limit[:][2] = self.limit[:][3] = speed limit. The velocity components
        are clipped proportional to the original values.
        """
        self.state[:2] = np.clip(self.state[:2], self.limit[0][:2], self.limit[1][:2])
        v_square = self.state[2:]**2
        del_v = np.sum(v_square) - self.limit[1][2]**2
        if del_v > 0.0:
            self.state[2] = np.sign(self.state[2]) * np.sqrt(max(0.0,
                v_square[0] - del_v * v_square[0] / (v_square[0] + v_square[1])))
            self.state[3] = np.sign(self.state[3]) * np.sqrt(max(0.0,
                v_square[1] - del_v * v_square[1] / (v_square[0] + v_square[1])))

    def collision_control(self):
        """
        Assigns a new velocity deviating the agent with an angle (pi/2, pi) from
        the closest obstacle point.
        """
        odom = [self.state[0], self.state[1], np.arctan2(self.state[3], self.state[2])]
        obs_pos = self.obs_check_func(odom)
        v = np.sqrt(np.sum(np.square(self.state[2:]))) + np.random.normal(0.0,1.0)
        if obs_pos[1] >= 0:
            th = obs_pos[1] - (1 + np.random.random()) * np.pi/2
        else:
            th = obs_pos[1] + (1 + np.random.random()) * np.pi/2

        state = np.array([self.state[0], self.state[1], v * np.cos(th + odom[2]), v * np.sin(th + odom[2])])
        return state

    def obstacle_detour_maneuver(self, r_margin=1.0):
        """
        Returns del_vx, del_vy which will be added to the new state.
        This provides a repultive force from the closest obstacle point based
        on the current velocity, a linear distance, and an angular distance.

        Parameters:
        ----------
        r_margin : float. A margin from an obstalce that you want to consider
        as the minimum distance the target can get close to the obstacle.
        """
        odom = [self.state[0], self.state[1], np.arctan2(self.state[3], self.state[2])]
        obs_pos = self.obs_check_func(odom)
        speed = np.sqrt(np.sum(self.state[2:]**2))
        rot_ang = np.pi/2 * (1. + 1./(1. + np.exp(-(speed-0.5*METADATA['target_speed_limit']))))
        if obs_pos is not None:
            acc = max(0.0, speed * np.cos(obs_pos[1])) / max(METADATA['margin2wall'], obs_pos[0] - r_margin)
            th = obs_pos[1] - rot_ang if obs_pos[1] >= 0 else obs_pos[1] + rot_ang
            del_vx = acc * np.cos(th + odom[2]) * self.sampling_period
            del_vy = acc * np.sin(th + odom[2]) * self.sampling_period
            return del_vx, del_vy
        else:
            return 0., 0.


class AgentSE2Goal(Agent):
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func,
                 margin=METADATA['margin'], policy=None,horizon= 2.0):
        super().__init__(agent_id, dim, sampling_period, limit, collision_func, margin=margin)
        self.policy = policy
        self.horizon = horizon

    def reset(self, init_state):
        super().reset(init_state)
        self.vw= [0.,0.]
        if self.policy:
            self.policy.reset(init_state)

    def set_goals(self,control_goal=None,margin_pos=None,DEBUG=False):
        '''

        :param control_goal:
        :param marigin_pos:
        :return: A planner which achieves the goals
        '''

        self.control_goal = control_goal
        self.margin_pos = margin_pos


        plan = self.plan(DEBUG)
        return plan

    def plan(self,DEBUG=False):
        '''
        :param self: 
        :return: returns a planner object to reach the desired control goal
        '''

        return SE2Planner(state = self.state,
                          goal = self.control_goal,
                          time = self.horizon,
                          dynamics = SE2DynamicsRK4,
                          dT=self.sampling_period,
                          DEBUG=DEBUG)

    def update(self, input=None, margin_pos=None, col=False):
        """
        control_input : (state: (x,y,theta), u (v,omega))
        margin_pos : a minimum distance to a target
        """


        new_state = input[0]
        control_input = input[1]
        is_col = 0
        if self.collision_check(new_state[:2]):
            is_col = 1
            new_state[:2] = self.state[:2]
            control_input = self.vw
            if self.policy is not None:
                # self.policy.collision(new_state)
                corrected_policy = self.policy.collision(new_state)
                if corrected_policy is not None:
                    new_state = SE2DynamicsVel(self.state,
                                               self.sampling_period, corrected_policy)

        elif margin_pos is not None:
            if self.margin_check(new_state[:2], margin_pos):
                new_state[:2] = self.state[:2]
                control_input = self.vw

        self.state = new_state
        self.vw = control_input
        self.range_check()

        return is_col


class SE2Planner:

    def __init__(self,state,goal,time,dynamics,dT,threshold = 1e-3,control_dim = 2,DEBUG = True):
        '''.
        @params:
        goal: set a goal
        time: set a time to reach the goal
        dynamics: the dynamics function to containerise
        dT: the timme discretization
        '''

        self.goal = list(goal)
        self.goal[-1] = self.goal[-1]/360*2*np.pi
        self.goal = jnp.asarray(self.goal)
        self.goal += jnp.array(state)
        self.time = time
        self.dim = len(goal)
        self.dynamic_fn = rk4(dynamics,dt = dT)
        self.threshold = 1e-3
        self.dT = dT
        self.cost_params = {
            'x_stage_cost' : 2.0,
            'u_stage_cost' : 1.0
        }

        self.state  = jnp.asarray(state)
        self.control_dim = control_dim
        self.u_lower = -6.0 * jnp.ones(self.control_dim)
        self.u_upper = 6.0 * jnp.ones(self.control_dim)

        U = jnp.zeros((int(self.time/self.dT),control_dim))


        self.solution =  optimizers.constrained_ilqr(
        functools.partial(self.cost, params = self.cost_params), self.dynamic_fn, self.state, U,
        equality_constraint=self.equality_constraint,
        inequality_constraint=self.inequality_constraint,
        constraints_threshold=self.threshold,
        maxiter_al=10)


        # print(" State Current {}".format(self.state))
        # print(" State Goal {}".format(self.goal))
        # print("States: {}".format(self.solution[0]))
        # print("Controls: {}".format(self.solution[1]))
        # print("Cost: {}".format(self.solution[2]))
        # print("Equality constraint: {}".format(self.solution[3]))
        # print("Inequality constraint: {}".format(self.solution[4]))

        if DEBUG:
            from matplotlib import pyplot as plt

            #plot all states
            plt.figure()
            plt.plot(self.state[0], self.state[1], 'g*')
            plt.plot(self.solution[0][:,0], self.solution[0][:,1], 'b')
            plt.plot(self.goal[0], self.goal[1], 'r*')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('States')
            #plt.show()

            # plot all scontrols
            plt.figure()
            plt.plot(self.solution[1][:,0], 'b')
            plt.plot(self.solution[1][:,1], 'b')
            plt.xlabel('t')
            plt.ylabel('u')
            plt.title('Controls')
            plt.show()

        return

    def cost(self,x,u,t,params):
        delta = x - self.goal

        stagewise_cost = 0.5 * params['x_stage_cost'] * jnp.dot( delta, delta) \
                         + 0.5 * params['u_stage_cost'] * jnp.dot(u, u)
        terminal_cost = params['x_stage_cost'] * jnp.dot(
            delta, delta)
        return jnp.where(t == int(self.time/self.dT), terminal_cost, stagewise_cost)

    def get_controller(self,time):
        '''
        :param time: get control input associated with the time
        :return:
        tuple(x,u): x is the state at time t and u is the control at time t
        '''

        x = np.array(self.solution[0][int(time/self.dT),:])
        u = np.array(self.solution[1][int(time/self.dT),:])

        return (x,u)

    def equality_constraint(self,x,u,t):
        del u

        def goal_constraint(x):
            err = x - self.goal
            return err

        return jnp.where(t == int(self.time/self.dT), goal_constraint(x), np.zeros(self.dim))

    def inequality_constraint(self,x, u, t):

        del x
        def control_limits(u):
            return jnp.concatenate((-u + self.u_lower, u - self.u_upper))


        return jnp.where(t == int(self.time/self.dT), jnp.zeros(2 * self.control_dim),control_limits(u))