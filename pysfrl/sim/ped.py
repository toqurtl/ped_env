import numpy as np
from .parameters import DataIndex as Index

index_list = sorted([index for index in Index], key=lambda data: data.index)


# -1: id, -2: visible, -3: tau
class PedAgent(object):
    def __init__(self, base_data, ped_info):
        self.base_data = base_data
        self.id = base_data.get(Index.id.str_name)
        self.start_time = base_data.get(Index.start_time.str_name)
        self.states = []
        self.phase = 0        
        self.final_phase = ped_info["final_phase"]        
        self.goal_schedule = ped_info["goal_schedule"]
        self._initialize()

    @property
    def current_state(self):
        return self.states[-1]

    def _initialize(self):
        state = [self.base_data.get(index.str_name) for index in index_list]        
        self.states.append(np.array(state))
        return

    def basic_state(self):
        return np.array([[self.base_data.get(index.str_name) for index in index_list]])

    """ state 검색"""
    def state_at(self, time_step):
        return self.states[time_step]
    
    """ state 추가"""
    def update(self, new_whole_state):           
        new_state = new_whole_state[self.id]
        return np.squeeze(new_state)

    """property들"""
    @property
    def distancing(self):
        return self.current_state.get(Index.distancing.str_name)

    @property
    def tau(self):
        return self.current_state.get(Index.tau.str_name)

    @property
    def px(self):
        return self.current_state[Index.px.index]
    @property
    def py(self):
        return self.current_state[Index.py.index]

    @property
    def vx(self):
        return self.current_state[Index.vx.index]

    @property
    def vy(self):
        return self.current_state[Index.vy.index]

    @property
    def current_phase(self):
        return str(int(self.current_state[Index.phase.index]))

    @property
    def gx(self):         
        if self.current_phase in self.goal_schedule.keys():      
            return self.goal_schedule[self.current_phase]["tx"]
        else:            
            return self.goal_schedule[self.final_phase]["tx"]
    
    @property
    def gy(self):
        if self.current_phase in self.goal_schedule.keys():
            return self.goal_schedule[self.current_phase]["ty"]
        else:
            return self.goal_schedule[self.final_phase]["ty"]

    @property
    def finished(self):
        return self.current_state[Index.finished.index]
