# coding=utf-8
from pysfrl.sim.obstaclestate import EnvState
from pysfrl.sim.pedstate import PedState
from pysfrl.sim.force import forces
from pysfrl.sim.video.peds import Pedestrians
from pysfrl.sim.update_manager import UpdateManager
from pysfrl.sim.config.exp_setting import ExperimentSetting
import numpy as np
import json

force_dict={
    "desired_force": forces.DesiredForce(),
    "obstacle_force": forces.ObstacleForce(),
    "ped_repulsive_force": forces.PedRepulsiveForce(),
    "social_force": forces.SocialForce(),
    "my_force": forces.Myforce(),
    "my_force_2": forces.MyforceSecond(),
    "my_force_3": forces.MyforceThird()
}


class Simulator(object):
    """초기값 설정"""
    def __init__(self, exp: ExperimentSetting, groups=None):
        # Config 읽어보는 부분        
        self.scene_config = exp.scene_config
        self.force_config = exp.force_config

        # 시뮬레이션 전체를 관장하는 것        
        self.pedestrians = exp.peds
        
        # PedState. 다음 스텝을 계산해주는 계산기        
        self.peds = PedState(self.scene_config)        
        self.env = EnvState(exp.obstacle, self.scene_config["resolution"])
                
        # 시뮬레이션 time 관리
        self.time_step = 0  
        self.time_table = exp.video.time_table
        self.step_width_list = []
        
        self._initialize_force()
        self._initialize()
        
        return

    # 초기 속도를 설정
    def _initialize(self):
        speed_vecs = self.pedestrians.current_state[:,2:4]                
        self.initial_speeds = np.array([np.linalg.norm(s) for s in speed_vecs])        
        self.max_speeds = self.peds.max_speed_multiplier * self.initial_speeds

    # 힘 설정
    def _initialize_force(self):
        force_list = []
        for force_name in self.force_config["set"].keys():
            force_list.append(force_dict[force_name])
        
        group_forces = []
        if self.scene_config["enable_group"]:
            force_list += group_forces

        for force in force_list:
            force.init(self, self.force_config["set"])        
        self.forces = force_list
        return

    def get_obstacles(self):
        return self.env.obstacles

    """시뮬레이션 함수"""
    def simulate(self):
        while True:            
            is_finished = self.step_once()            
            if is_finished: 
                break

            if self.time_step>1000:
                break
        return

    def step_once(self):
        # update_visible
        # states중 마지막 것 가져옴
        
        whole_state = self.pedestrians.current_state.copy()
        
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
        
        #finish 여부를 확인
        
        whole_state = UpdateManager.update_phase(whole_state)
        whole_state = UpdateManager.update_finished(whole_state)        
        # print(whole_state)
        # whole_state를 pedestrians에 저장
        self.after_step(whole_state, next_group_state)
        return False        

    def before_step(self):
        return

    # 힘을 계산하고, 힘에 의한 위치 변화 및 속도 변화를 다음 state로 결과를 만듦
    # visible state + force -> new_state
    def do_step(self, visible_state, visible_max_speeds, visible_group=None):        
        # 계산기 설정
        self.peds.set_state(visible_state, visible_group, visible_max_speeds)        
        force = self.compute_forces()        
        next_state, next_group_state = self.peds.step(force, visible_state)                
        return next_state, next_group_state

    def after_step(self, next_state, next_group_state):
        # Pedestrian에 결과를 업데이트
        self.pedestrians.update(next_state, next_group_state, self.time_step)        
        self.pedestrians.update_target_pos()
        # target_phase 변경
        self.time_step += 1
        return True

    def check_finish(self):        
        return np.sum(self.pedestrians.check_finished())

    def set_step_width(self):
        new_step_width = 0
        if self.time_table is None:
            new_step_width = 0.133            
        else:
            try: 
                new_step_width = self.time_table[self.time_step]                
            except IndexError:
                new_step_width = 0.133            
        self.peds.step_width = new_step_width
        self.step_width_list.append(new_step_width)
        return

    def compute_forces(self):        
        return sum(map(lambda x: x.get_force(), self.forces))

    """결과값 저장"""
    def result_to_json(self, file_path):
        result_data = {}        
        time = 0
        result_data[0] = {
            "step_width": time,
            "states": self.pedestrians.states[0].tolist()
        }
        
        for i in range(0, len(self.step_width_list)):
            time += self.step_width_list[i]
            result_data[i+1] = {
                "step_width": time,
                "states": self.pedestrians.states[i+1].tolist()
            }
        
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4)        
        return

    def summary_to_json(self, file_path, success):
        data = {}
        data["success"] = success
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return

        
        
        

        
    

