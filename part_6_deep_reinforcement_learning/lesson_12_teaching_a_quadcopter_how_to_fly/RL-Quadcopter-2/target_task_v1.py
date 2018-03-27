import numpy as np
from physics_sim_v1 import PhysicsSimV1

class TargetTaskV1():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSimV1(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        
#         self.state_size = 6 * self.action_repeat
        self.state_size = len(self._get_status()) * self.action_repeat

        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
    
    def _get_status(self):
        status = list(self.sim.pose) + list(self.sim.v) + list(self.sim.angular_v)
#         print('status', len(status), status)
        return status

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
#         state = np.concatenate([self.sim.pose] * self.action_repeat) 
        state = self._get_status() * self.action_repeat
        return state
    
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 1. - .3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        # has improve 
        # reward = 100. - .3*(abs(self.sim.pose[:3] - self.target_pos)).sum()   

        # bad
        # reward = 300 - (abs(self.sim.pose[:3] - self.target_pos)).sum()           

        # ok
        p1 = self.sim.pose[:3]
        p2 = self.target_pos
        env_bounds = 300.0
        bound = np.array([env_bounds, env_bounds, env_bounds])
        reward = (0.5 - np.mean(np.square((p1 - p2) / bound))) * 2

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        next_state = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()                         
            next_state.extend(self._get_status())
        
        return next_state, reward, done    