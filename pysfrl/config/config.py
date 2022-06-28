from typing import Dict
import numpy as np
import json


class SimulationConfig(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.ped_info: Dict = self.cfg["ped_info"]
        self.obstacles_info: Dict = self.cfg["obstacles"]
        self.scene_config: Dict = self.cfg["scene"]
        self.force_config: Dict = self.cfg["forces"]
        self.condition: Dict = self.cfg["condition"]
        self.initial_state_info: Dict = self.cfg["initial_state"]

    @property
    def simul_time_threshold(self):
        return self.cfg["condition"]["simul_time_threshold"]

    @property
    def num_ped(self):
        return len(self.ped_info)

    @property
    def max_speed_multiplier(self):
        return self.cfg["scene"]["max_speed_multiplier"]

    @property
    def initial_speeds(self):
        speed_vecs = self.initial_state_arr[:, 2:4]
        return np.array([np.linalg.norm(s) for s in speed_vecs])

    @property
    def max_speeds(self):
        return self.max_speed_multiplier * self.initial_speeds

    @property
    def initial_state_arr(self):
        state_list = []
        for ped_info in self.initial_state_info.values():
            state = []
            for idx, value in ped_info.items():
                if idx != "id":
                    state.append(value)
            state_list.append(state)
        return np.array(state_list)

    @property
    def desired_force_params(self):
        return self.force_config["desired_force"]

    @property
    def obstacle_force_params(self):
        return self.force_config["obstacle_force"]
    
    @property
    def repulsive_force_name(self):
        return self.force_config["repulsive_force"]["name"]

    @property
    def repulsive_force_params(self):
        return self.force_config["repulsive_force"]["params"]

    def save(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.cfg)
        return