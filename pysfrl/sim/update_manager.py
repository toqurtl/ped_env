import numpy as np
from dis_ped.video.parameters import DataIndex as Index


class UpdateManager(object):
    # check functions
    @classmethod
    def is_started(cls, state: np.ndarray, time_step):
        return state[:, Index.start_time.index] <= time_step

    # target point에 도달했는지 확인(gx - px)
    @classmethod
    def is_arrived(cls, state: np.ndarray):
        vecs = state[:,4:6] - state[:, 0:2]        
        distance_to_target = np.array([np.linalg.norm(line) for line in vecs])
        return distance_to_target < 0.5

    @classmethod
    def is_final_phase(cls, state: np.ndarray):
        return state[:, Index.final_phase.index] < state[:, Index.phase.index]

    # TODO - 여러 목적지일 때는 변경해야 함
    @classmethod
    def is_finished(cls, state: np.ndarray):        
        return np.logical_and(cls.is_final_phase(state), cls.is_arrived(state))        

    @classmethod
    def simul_finished(cls, state:np.ndarray):
        return np.sum(state[:, Index.finished.index]) == len(state)

    @classmethod
    def is_visible(cls, state: np.ndarray, time_step):
        time_cond = cls.is_started(state, time_step)
        not_finish_cond = np.logical_not(cls.is_finished(state))                
        return np.logical_and(time_cond, not_finish_cond)

    # get_idx functions
    @classmethod
    def update_finished(cls, state: np.ndarray):
        # finished된 agent들을 체크해서, finished/visible 상태를 변경
        finish_cond = cls.is_finished(state)        
        state[:, Index.finished.index] = finish_cond * 1
        for idx, data in enumerate(state):  
            if data[Index.id.index] in cls.finished_idx(state):
                state[idx][Index.visible.index] = 0        
        return state

    @classmethod
    def update_phase(cls, state: np.ndarray):
        arrived_cond = cls.is_arrived(state) * 1
        final_phase_cond = np.logical_not(cls.is_final_phase(state)) * 1
        state[:, Index.phase.index] = state[:, Index.phase.index] + arrived_cond * final_phase_cond
        return state        

    @classmethod
    def update_visible(cls, state: np.ndarray, time_step):
        visible_cond = cls.is_visible(state, time_step)
        state[:, Index.visible.index] = visible_cond * 1
        return state

    @classmethod
    def update_new_peds(cls, state:np.ndarray, time_step):
        for idx, data in enumerate(state):
            if data[Index.id.index] in cls.start_idx(state, time_step):
                state[idx][Index.visible.index] = 1
        return state

    # 업데이트된 visible state를 whole state에 반영
    @classmethod
    def new_state(cls, whole_state, next_state):
        id_index = next_state[:, Index.id.index].astype(np.int64)        
        whole_state[id_index] = next_state
        return whole_state

    @classmethod
    def finished_idx(cls, state: np.ndarray):
        finished_peds = state[state[:, Index.finished.index] == 1]
        return finished_peds[:, Index.id.index].astype(np.int64)

    @classmethod
    def start_idx(cls, state: np.ndarray, time_step):
        start_peds = state[state[:, Index.start_time.index] == time_step]
        return start_peds[:, Index.id.index].astype(np.int64)

    # result_functions
    @classmethod
    def get_visible(cls, state: np.ndarray):
        return state[state[:, Index.visible.index] == 1]

    @classmethod
    def get_visible_idx(cls, state: np.ndarray):
        visible_peds = cls.get_visible(state)
        return visible_peds[:, Index.id.index].astype(np.int64)

    @classmethod
    def update_target(cls, state:np.ndarray):
        pass