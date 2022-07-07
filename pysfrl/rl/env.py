from typing import List
import gym
from gym import spaces
from pysfrl.rl.utils import neighbor_distance
from pysfrl.sim.simulator import Simulator
from pysfrl.sim.parameters import DataIndex as Index
import random
import numpy as np

from pysfrl.sim.utils.custom_utils import CustomUtils


class PysfrlEnv(gym.Env):
    #렌더링 모드
    metadata = {'render_modes': ['human']}
    
    def __init__(self, simulator_list: List[Simulator], learned_idx=0):
        self.simulator_list = simulator_list                
        self.simulator: Simulator = None
        self.observation_space = spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-3,high=3, shape=(2,), dtype=np.float32)
        self.learned_idx = learned_idx

    @property
    def initial_state(self):
        return self.simulator.initial_state

    @property
    def sim_time_step(self):
        return self.simulator.time_step

    @property
    def num_ped(self):
        return self.simulator.num_peds()

    # target_idx가 끝나면 시뮬레이션 전체가 안 끝나도 종료
    def is_finished(self):
        return self.simulator.current_state[self.learned_idx, Index.finished.index] == 1        

    def step(self, action):            
        while True:
            is_skip_step, finished = self.skip_step()
            if not is_skip_step:
                break

            if finished:
                print(self.simulator.cfg.config_id, "finished")
                return self.dummpy_output()

        pre_state = self.simulator.current_state.copy()        
        next_state = self.simulator.next_state(pre_state, action)
        next_state = self.simulator.update_new_state(next_state)
        self.simulator.peds.update(next_state)
        self.simulator.time_step += 1
        
        done = self.is_finished() or self.simulator.time_step > 1000

        if done:
            print(self.simulator.cfg.config_id, self.learned_idx, "finished")
            obs = None
            reward = 5
            return obs, reward, done, self.info()        
        else:                        
            visible_state, _ = self.simulator.get_visible_info()            
            if len(visible_state) == 1:
                dummy_obs = np.zeros(self.observation_space.sample().shape)         
                return dummy_obs, 0, False, self.info()
            obs = self.observation(self.simulator, visible_state)
            reward = self.reward(self.simulator, pre_state, next_state, visible_state)            
        return obs, reward, done, self.info()
    
    def observation(self, sim: Simulator, visible_state):
        visible_idx = CustomUtils.find_visible_idx(visible_state, self.learned_idx)       
        extra_force = sim.extra_forces(visible_state)[visible_idx]        
        neighbor_info = CustomUtils.neighbor_info(visible_state, visible_idx)        
        return np.concatenate((extra_force, neighbor_info))

    def reward(self, sim: Simulator, pre_state, next_state, visible_state):
        # idx를 주면, 그 idx와 가장 가까운 거리에 있는 거리를 구하는 함수
        reward = 0
        delta = CustomUtils.goal_distance_delta(pre_state, next_state, self.learned_idx)
        if delta > 1.5:
            reward += 1

        # idx를 주면 목적지와의 거리를 계산하는 함수
        obs = self.observation(sim, visible_state)
        neighbor_distance = np.linalg.norm(obs[2:4])
        if neighbor_distance < 2:
            reward -= 5
        return reward

    def info(self):
        return {
            "simulator": self.simulator
        }


    def reset(self):
        self.simulator = random.choice(self.simulator_list)
        self.learned_idx = random.choice(range(0, self.num_ped))        
        self.simulator.reset()
        print(self.simulator.cfg.config_id, self.learned_idx, "start")
        while True:
            is_skip_step, finished = self.skip_step()
            if not is_skip_step:
                break

            if finished:                
                obs, _, _, _ = self.dummpy_output()
                return obs
        visible_state, _ = self.simulator.get_visible_info()
        return self.observation(self.simulator, visible_state)

    # step을 스킵해야 하는지 아닌 지 판단
    # target_idx가 혼자 있거나, visible하지 않은 경우 그냥 스킵
    # 그렇지 않더라도 시뮬레이션이 끝난경우 dummy 반환하면서 episode 종료
    def skip_step(self):                
        visible_state, visible_idx = self.simulator.get_visible_info()
        is_skip_step = self.learned_idx not in visible_idx or len(visible_idx) < 2
        finished = False
        if is_skip_step:
            finished = self.simulator.step_once()
        
        if self.simulator.time_step > 1000:
            finished = True

        return is_skip_step, finished

    def dummpy_output(self):
        return np.zeros(self.observation_space.sample().shape), 0, True, self.info()