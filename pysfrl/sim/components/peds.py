from typing import Dict, List
from pysfrl.sim.update_manager import UpdateManager
from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.components.ped import PedAgent
from pysfrl.sim.parameters import DataIndex as Index
import numpy as np


class Pedestrians(object):
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.peds: Dict[int, PedAgent] = {}
        self.states: List[np.ndarray] = []        
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
        visible_state, visible_idx = UpdateManager.get_visible(whole_state)                
        return visible_state, visible_idx

    def update(self, new_peds_state, next_group_state=None):
        new_state_list = []
        for ped in self.peds.values():
            new_state = ped.update(new_peds_state)          
            new_state_list.append(new_state)
        
        self.states.append(np.array(new_state_list))  
        self.update_target_pos()          
        return

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
