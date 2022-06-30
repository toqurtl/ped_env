from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.peds import Pedestrians
from pysfrl.sim.pedstate import PedState
from pysfrl.sim.obstaclestate import ObstacleState
from pysfrl.sim.update_manager import UpdateManager
from pysfrl.sim.force.force import Force
import numpy as np
import json

repulsive_force_dict ={
    "my_force": Force.social_sfm_1,
    "my_force_2": Force.social_sfm_2,
    "my_force_3": Force.basic_sfm
}


class NewSimulator(object):
    def __init__(self, config: SimulationConfig):
        # configuration
        self.cfg: SimulationConfig = config
        # initialization(config 정보 바탕으로 초기정보 생성)
        # components: Ped, pedstate 계산기, Obstacle 계산기
        # Pedestrian 기본 정보 + 시뮬레이션동안 발생하는 정보 반환
        self.peds = Pedestrians(self.cfg.ped_info, self.cfg.initial_state_info)
        self.ped_state = PedState(config.scene_config)        
        # 시뮬레이터 안에서 인식하기 위함(힘 계산을 위해)
        self.obstacle_state = ObstacleState(config.obstacles_info)
        self.repulsive_force_name = self.cfg.force_config["repulsive_force"]["name"]

        # Simulation 
        self.time_step = 0
        self.time_table = None
        self.step_width_list = []
        pass

    @property
    def max_speeds(self):
        return self.cfg.max_speeds
    
    @property
    def initial_speeds(self):
        return self.cfg.initial_speeds
    
    # obstacle_info, [[1,2,3,4],[1,2,3,4]] 이런식으로 된 친구들
    @property
    def obstacle_info(self):        
        return np.array(self.cfg.obstacles_info)
    
    @property
    def peds_states(self):
        return np.array(self.peds.states)
    
    def num_peds(self):
        return self.peds.num_peds
    
    def get_obstacles(self):
        return self.obstacle_state.obstacles

    def set_step_width(self):
        new_step_width = 0
        if self.time_table is None:
            new_step_width = self.cfg.step_width       
        else:# time_table setting 부분 완성 후
            try: 
                new_step_width = self.time_table[self.time_step]                
            except IndexError:
                new_step_width = 0.133            
        self.ped_state.step_width = new_step_width
        self.step_width_list.append(new_step_width)
        return

    def check_finish(self):        
        return np.sum(self.peds.check_finished())

    def compute_forces(self):
        desired_force = Force.desired_force(self)
        obstacle_force = Force.obstacle_force(self)
        repulsive_force = repulsive_force_dict[self.repulsive_force_name](self)
        return desired_force + obstacle_force + repulsive_force

    def simulate(self):
        while True:            
            is_finished = self.step_once()             
            if is_finished: 
                break

            if self.time_step>1000:
                break
        return

    def do_step(self, visible_state, visible_max_speeds, visible_group):
        self.ped_state.set_state(visible_state, visible_group, visible_max_speeds)        
        force = self.compute_forces()        
        next_state, next_group_state = self.ped_state.step(force, visible_state)                
        return next_state, next_group_state
            
    def after_step(self, next_state, next_group_state):
        # Pedestrian에 결과를 업데이트
        self.peds.update(next_state, next_group_state, self.time_step)        
        self.peds.update_target_pos()
        # target_phase 변경
        self.time_step += 1
        return True
    
    # 이전 whole_state -> 시뮬레이션 끝났는지 확인 -> visible state 가져옴 -> 다음 visible state -> whole state 만듦 -> 
    def step_once(self):
        # update_visible
        # states중 마지막 것 가져옴
        
        whole_state = self.peds.current_state.copy()
        
        # whole_state = UpdateManager.update_finished(whole_state)
        
        # 모든 agent가 끝나면 종료
        # if self.check_finish():
        if UpdateManager.simul_finished(whole_state):
            return True
        # goal schedule을 확인해서 update
        
        # finish 여부에 따라 visible한 agent들의 state 가져옴
        visible_state = UpdateManager.get_visible(whole_state)
        # finish 여부에 따라 visible한 agent를의 idx를 가져옴
        visible_idx = UpdateManager.get_visible_idx(whole_state)
        visible_max_speeds = self.max_speeds[visible_idx]
        
        # step_width update
        self.set_step_width()

        # group 행동할 때 설정
        next_group_state = None
        # 힘 계산 수행해서 다음 state를 얻어냄
        if len(visible_state) > 0:
            # visible state들에 대해서 힘 계산해서 변경
            next_state, next_group_state = self.do_step(visible_state, visible_max_speeds, None)                         
            # 변경된 visbile state를 whole state에 반영            
            whole_state = UpdateManager.new_state(whole_state, next_state)
        
        # 계산안하고 등장해야 하는 애들 반영 -> whole_state
        whole_state = UpdateManager.update_new_peds(whole_state, self.time_step)                        
        
        # finish 여부를 확인        
        whole_state = UpdateManager.update_phase(whole_state)
        whole_state = UpdateManager.update_finished(whole_state)
        
        # whole_state를 pedestrians에 저장
        self.after_step(whole_state, next_group_state)
        return False        

    def SDVR(self):
        pass

    def save(self, file_path):
        result_data = {}        
        time = 0
        result_data[0] = {
            "step_width": time,
            "states": self.peds.states[0].tolist()
        }
        
        for i in range(0, len(self.step_width_list)):
            time += self.step_width_list[i]
            result_data[i+1] = {
                "step_width": time,
                "states": self.peds.states[i+1].tolist()
            }
        
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4)        
        return