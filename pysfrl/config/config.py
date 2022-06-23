class SimulationConfig(object):
    def __init__(self, cfg):
        self.cfg = cfg        

    @property
    def obstacle_info(self):
        return self.cfg["obstacles"]
    
    @property
    def scene_config(self):
        return self.cfg["scene"]

    @property
    def force_config(self):
        return self.cfg["forces"]

    @property
    def simul_time_threshold(self):
        return self.cfg["condition"]["simul_time_threshold"]
    
    @property
    def initial_state(self):
        return

    @property
    def initial_speeds(self):
        return

    @property
    def max_speeds(self):
        return

    @property
    def ped_info(self):
        return self.cfg["ped_info"]
