import os
import json

class PedConfig(object):
    def __init__(self, config_path):
        with open(config_path, 'r', encoding="UTF-8") as f:
            self.cfg = json.load(f)

    @property
    def setting_id(self):
        return self.cfg["setting_id"]

    @property
    def repulsive_force(self):
        return self.cfg["repulsive_force"]

    @property
    def vid_folder_path(self):
        return self.cfg["path"]["vid_folder_path"]

    @property
    def result_folder_path(self):
        return self.cfg["path"]["result_folder_path"]
    
    @property
    def simul_time_threshold(self):
        return int(self.cfg["condition"]["simul_time_threshold"])
    
    @property
    def obstacles(self):
        return self.cfg["obstacles"]

    @property
    def scene_config(self):
        return self.cfg["scene"]

    @property
    def force_config(self):
        return self.cfg["forces"]

    @property
    def simul_config(self):
        return self.cfg["condition"]

    @property
    def force_name(self):
        return self.cfg["repulsive_force"]

    