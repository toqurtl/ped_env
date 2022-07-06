from typing import Dict
import numpy as np
import json


class SimulationConfig(object):
    def __init__(self):
        self.config_id = None
        self.ped_info: Dict = {}
        self.obstacles_info: Dict = {}
        self.scene_config: Dict = {}
        self.force_config: Dict = {}
        self.condition: Dict = {}
        self.initial_state_info: Dict = {}
        return

    def set_config(self, cfg):        
        self.ped_info = cfg["ped_info"]
        self.obstacles_info = cfg["obstacles"]
        self.scene_config = cfg["scene"]
        self.force_config = cfg["forces"]
        self.condition = cfg["condition"]
        self.initial_state_info = cfg["initial_state"]
        self.config_id = cfg["config_id"]
        return

    def set_force_config(self, force_config):
        self.force_config = force_config
        return

    def set_ped_info(self, ped_info):
        self.ped_info = ped_info
        return

    def set_initial_state_info(self, initial_state_info):
        self.initial_state_info = initial_state_info
        return

    def set_config_id(self, config_id):
        self.config_id = config_id

    @property
    def simul_time_threshold(self):
        return self.condition["simul_time_threshold"]

    @property
    def num_ped(self):
        return len(self.ped_info)

    @property
    def max_speed_multiplier(self):
        return self.scene_config["max_speed_multiplier"]

    @property
    def initial_speeds(self):
        speed_vecs = self.initial_state_arr[:, 2:4]
        return np.array([np.linalg.norm(s) for s in speed_vecs])

    @property
    def max_speeds(self):
        return self.max_speed_multiplier * self.initial_speeds

    @property
    def max_speed(self):        
        return self.scene_config["max_speed"]

    @property
    def step_width(self):
        return self.scene_config["step_width"]

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

    def get_config_dict(self):
        return {
            "config_id": self.config_id,
            "ped_info": self.ped_info,
            "initial_state": self.initial_state_info,
            "group_info": [],
            "obstacles": self.obstacles_info,
            "scene": self.scene_config,
            "forces": self.force_config,
            "condition": self.condition
        }

    def save(self, file_path):
        cfg = self.get_config_dict()        
        with open(file_path, "w") as f:
            json.dump(cfg, f, indent=4)
        return