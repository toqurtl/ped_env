"""This module tracks the state odf scene and scen elements like pedestrians, groups and obstacles"""
from typing import List
import numpy as np
from pysfrl.sim.utils import stateutils


"""계산기로 변경"""

class PedState:
    """Tracks the state of pedstrains and social groups"""

    def __init__(self, config):        
        self.default_tau = config["tau"]
        self.step_width = config["step_width"]
        self.agent_radius = config["agent_radius"]
        self.max_speed_multiplier = config["max_speed_multiplier"]
        self.max_speed = 2.1
        self.initial_speeds = None
        self.current_state = None        
        self.group_states = []        
    
    @property
    def state(self):
        return self.current_state

    # def get_states(self):
    #     return np.stack(self.ped_states), self.group_states

    def size(self) -> int:
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        return self.state[:, 4:6]

    def visible(self):
        return self.state[7:8]

    def tau(self):
        return self.state[:, 9:10]

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def set_state(self, state, groups, visible_max_speeds):
        self.current_state = state        
        self.max_speeds = visible_max_speeds
        self.groups = groups

    def step(self, force, visible_state, group_state=None):
        # desired velocity
        desired_velocity = self.vel() + self.step_width * force                
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)        
        visible_state[:, 0:2] += desired_velocity * self.step_width        
        visible_state[:, 2:4] = desired_velocity
        if group_state is None:
            next_group_state = []
        else:
            next_groups_state = []             
        return visible_state, next_group_state
        
    def desired_directions(self):
        return stateutils.desired_directions(self.state)[0]

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""        
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    @property
    def groups(self) -> List[List]:
        return self._groups

    @groups.setter
    def groups(self, groups: List[List]):
        if groups is None:
            self._groups = []
        else:
            self._groups = groups
        self.group_states.append(self._groups.copy())

    def has_group(self):
        return self.groups is not None

    def which_group(self, index: int) -> int:
        """find group index from ped index"""
        for i, group in enumerate(self.groups):
            if index in group:
                return i
        return -1
