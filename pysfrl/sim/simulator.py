from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.peds import Pedestrians
from pysfrl.sim.pedstate import PedState
from pysfrl.sim.obstaclestate import ObstacleState
from pysfrl.sim.update_manager import UpdateManager
from pysfrl.sim.force.force import Force
import numpy as np
import json

from pysfrl.sim.utils import stateutils

repulsive_force_dict ={
    "my_force": Force.social_sfm_1,
    "my_force_2": Force.social_sfm_2,
    "my_force_3": Force.basic_sfm
}

STEP_WIDTH = 0.067

class Simulator(object):
    def __init__(self, config: SimulationConfig):
        # configuration
        self.cfg: SimulationConfig = config
        # initialization(config 정보 바탕으로 초기정보 생성)
        
        self.peds = Pedestrians(self.cfg)
        # self.ped_state = PedState(config.scene_config)        
        # 시뮬레이터 안에서 인식하기 위함(힘 계산을 위해)
        self.obstacle_state = ObstacleState(config.obstacles_info)
        self.repulsive_force_name = self.cfg.force_config["repulsive_force"]["name"]

        # Simulation 
        self.time_step = 0
        self.time_table = None        
        pass

    @property
    def initial_state(self):
        return self.cfg.initial_state_arr

    @property
    def current_state(self):
        return self.peds.current_state

    @property
    def max_speeds(self):
        return np.ones((self.num_peds())) * self.cfg.max_speed
        # return self.cfg.max_speeds
    
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

    @property
    def step_width(self):
        return self.cfg.step_width 
    
    def num_peds(self):
        return self.peds.num_peds
    
    def get_obstacles(self):
        return self.obstacle_state.obstacles

    def compute_forces(self, external_force=None):
        desired_force = Force.desired_force(self)
        obstacle_force = Force.obstacle_force(self)
        if external_force is None:
            repulsive_force = repulsive_force_dict[self.repulsive_force_name](self)
        else:
            repulsive_force = external_force
        return desired_force + obstacle_force + repulsive_force

    def simulate(self):
        success = True
        while True:            
            is_finished = self.step_once()             
            if is_finished:
                break

            if self.time_step>1000:
                success = False
                break
        return success

    def capped_velocity(self, desired_velocity, max_velocity):
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)


    def do_step(self, visible_state, visible_max_speeds, visible_group, external_force=None):
        # self.ped_state.set_state(visible_state, visible_group, visible_max_speeds)        
        force = self.compute_forces(external_force=external_force)
        vel = visible_state[:,2:4]
        tau = visible_state[:, 6:7]
        desired_velocity = vel + 0.067 * force
        # desired_velocity = self.capped_velocity(desired_velocity, 2.1)
        desired_velocity = self.capped_velocity(desired_velocity, visible_max_speeds)
        
        visible_state[:, 0:2] += desired_velocity * STEP_WIDTH  
        visible_state[:, 2:4] = desired_velocity
        if visible_group is None:
            next_group_state = []
        else:
            next_groups_state = []
        
        return visible_state, next_group_state
            
    def after_step(self, next_state, next_group_state):
        # Pedestrian에 결과를 업데이트
        self.peds.update(next_state, next_group_state, self.time_step)        
        self.peds.update_target_pos()
        # target_phase 변경
        self.time_step += 1
        return True
    
    def get_visible_info(self):
        return self.peds.visible_info()

    def step_once(self, external_force=None):
        whole_state = self.peds.current_state.copy()
        # 시뮬레이션 종료 여부 판단(모든 agent 끝났을 때)
        if UpdateManager.simul_finished(whole_state):
            return True
        
        # 힘 계산에 필요한 visible한 agent들의 state, idx 가져옴
        visible_state = UpdateManager.get_visible(whole_state)
        visible_idx = UpdateManager.get_visible_idx(whole_state)
        visible_max_speeds = self.max_speeds[visible_idx]
        
        # group 행동할 때 설정
        next_group_state = None

        # 힘 계산 수행해서 다음 state를 얻어냄
        if len(visible_state) > 0:
            # visible state들에 대해서 힘 계산해서 변경
            next_state, next_group_state = self.do_step(
                visible_state, 
                visible_max_speeds, 
                next_group_state, 
                external_force=None
            )
            # 변경된 visbile state를 whole state에 반영            
            whole_state = UpdateManager.new_state(whole_state, next_state)
        
        ## 위치 계산 이후 state들을 변경(visible/phase/finished)
        # start_time 된 친구들 visible 변경
        whole_state = UpdateManager.update_new_peds(whole_state, self.time_step)
        # phase update
        whole_state = UpdateManager.update_phase(whole_state)
        # finished
        whole_state = UpdateManager.update_finished(whole_state)
        # whole_state를 pedestrians에 저장
        self.after_step(whole_state, next_group_state)
        return False        

    def reset(self):
        return

