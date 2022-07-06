from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.components.peds import Pedestrians
from pysfrl.sim.components.obstaclestate import ObstacleState
from pysfrl.sim.update_manager import UpdateManager
from pysfrl.sim.force.force import Force
from pysfrl.sim.utils.custom_utils import CustomUtils
import numpy as np


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
        self.obstacle_state = ObstacleState(config.obstacles_info)
        self.repulsive_force_name = self.cfg.force_config["repulsive_force"]["name"]

        # Simulation 
        self.peds = Pedestrians(self.cfg)
        self.time_step = 0
        self.time_table = None        
        pass

    # 고정값
    @property
    def initial_state(self):
        return self.cfg.initial_state_arr

    @property
    def max_speeds(self):
        return np.ones((self.num_peds())) * self.cfg.max_speed

    @property
    def current_state(self):
        return self.peds.current_state

    @property
    def initial_speeds(self):
        return self.cfg.initial_speeds

    @property
    def obstacle_info(self):
        return np.array(self.cfg.obstacles_info)

    def num_peds(self):
        return self.peds.num_peds
    
    def get_obstacles(self):
        return self.obstacle_state.obstacles

    # 시뮬레이션 중 변하는 것들    
    @property
    def peds_states(self):
        return np.array(self.peds.states)

    @property
    def step_width(self):
        return self.cfg.step_width 

    def get_visible_info(self):
        return self.peds.visible_info()

    # 시뮬레이션에 필요한 함수
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

    def check_finished(self):
        return UpdateManager.simul_finished(self.current_state)

    def step_once(self, external_force=None):
        print("chekc")
        # 시뮬레이션 종료 여부 판단(모든 agent 끝났을 때)
        if self.check_finished():
            return True

        whole_state = self.current_state.copy()
        whole_state = self.next_state(whole_state, external_force)
        whole_state = self.update_new_state(whole_state)  
        
        self.peds.update(whole_state)
        self.time_step += 1
        return False

    # 포지션 변한 것들 반영
    def next_state(self, whole_state, external_force=None):        
        # 힘 계산에 필요한 visible한 agent들의 state, idx 가져옴
        visible_state, visible_idx = self.get_visible_info()
        # 힘 계산 수행해서 다음 state를 얻어냄
        if len(visible_state) > 0:
            # visible state들에 대해서 힘 계산해서 변경
            # next_state = self.do_step(visible_state, external_force)

            vel = visible_state[:,2:4]
            force = self.compute_forces(visible_state, external_force=external_force)        

            desired_velocity = vel + self.step_width * force
            visible_max_speeds = CustomUtils.max_speeds(len(visible_state), self.cfg.max_speed)
            desired_velocity = CustomUtils.capped_velocity(desired_velocity, visible_max_speeds)
            
            visible_state[:, 0:2] += desired_velocity * self.step_width
            visible_state[:, 2:4] = desired_velocity
            # 변경된 visbile state를 whole state에 반영            
            whole_state = UpdateManager.new_state(whole_state, visible_state)
        
        return whole_state
    
    ## 위치 계산 이후 visible/phase/finished
    def update_new_state(self, whole_state):        
        # start_time 된 친구들 visible 변경
        whole_state = UpdateManager.update_new_peds(whole_state, self.time_step)
        # phase update
        whole_state = UpdateManager.update_phase(whole_state)
        # finished
        whole_state = UpdateManager.update_finished(whole_state)
        return whole_state

    def extra_forces(self, visible_state):
        desired_force = Force.desired_force(self.cfg, visible_state)
        obstacle_force = Force.obstacle_force(self.cfg, visible_state, self.get_obstacles())
        return desired_force + obstacle_force

    def compute_forces(self, visible_state, external_force=None):        
        if external_force is None:
            repulsive_force = repulsive_force_dict[self.repulsive_force_name](self.cfg, visible_state)
        else:
            repulsive_force = external_force
        return self.extra_forces(visible_state) + repulsive_force

    def reset(self):
        self.peds.reset()
        self.time_step = 0
        return

