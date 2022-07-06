from typing import Dict, List
import numpy as np

from pysfrl.sim.utils import stateutils
from pysfrl.sim.update_manager import UpdateManager
from pysfrl.config.sim_config import SimulationConfig
from .ped import PedAgent
from .parameters import DataIndex as Index


class Pedestrians(object):
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.peds: Dict[int, PedAgent] = {}
        self.states: List[np.ndarray] = []
        self.group_states = []
        self.generate_ped_agents()
        self.reset()
        return
    
    def generate_ped_agents(self):
        for key, agent_data in self.cfg.initial_state_info.items():            
            info = self.cfg.ped_info[key]
            ped = PedAgent(agent_data, info)
            self.peds[key] = ped
        return

    def reset(self):        
        self.states.clear()
        self.group_states.clear()
        for ped in self.peds.values():
            ped.reset()

        initial_state_arr = [ped.current_state for ped in self.peds.values()]       
        self.states.append(np.array(initial_state_arr))        
        return

    @property
    def time_step(self):
        return len(self.states) - 1  

    @property
    def num_peds(self):
        return len(self.peds)  

    @property
    def current_state(self):
        return self.states[-1]

    @property
    def max_speeds(self):
        return np.ones((self.num_peds)) * self.cfg.max_speed

    def visible_info(self):
        whole_state = self.current_state.copy()
        visible_state = UpdateManager.get_visible(whole_state)
        visible_idx = UpdateManager.get_visible_idx(whole_state)
        visible_max_speeds = self.max_speeds[visible_idx]
        return visible_state, visible_idx, visible_max_speeds

    # def pos(self):
    #     return self.current_state[:, 0:2]
    
    # def vel(self):
    #     return self.current_state[:, 2:4]
    
    # def goal(self):
    #     return self.current_state[:, 4:6]

    # def size(self):
    #     return self.current_state.shape[0]

    # def speeds(self):
    #     return stateutils.speeds(self.current_state)

    def state_at(self, time_step):
        return self.states[time_step]

    def update(self, new_peds_state, next_group_state, time_step):
        new_state_list = []
        for ped in self.peds.values():
            new_state = ped.update(new_peds_state)          
            new_state_list.append(new_state)                        
        if next_group_state is None:
            self.group_states.append([])
        else:    
            self.group_states.append(next_group_state)
        self.states.append(np.array(new_state_list))            
        return

    def check_finished(self):
        finish_state = self.current_state[:, Index.finished.index]    
        return np.sum(finish_state) == len(self.peds)

    def target_pos(self):
        gx_list, gy_list = [], []
        for idx, ped in self.peds.items():
            gx_list.append(ped.gx)
            gy_list.append(ped.gy)
        return np.array(gx_list), np.array(gy_list)

    def update_target_pos(self):
        gx_array, gy_array = self.target_pos()        
        self.current_state[:, Index.gx.index] = gx_array
        self.current_state[:, Index.gy.index] = gy_array
        return
