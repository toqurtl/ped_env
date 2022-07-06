import gym
from gym import spaces
from pysfrl.rl.utils import neighbor_distance
from pysfrl.sim.simulator import Simulator
from pysfrl.sim.parameters import DataIndex as Index
import numpy as np

from pysfrl.sim.utils.custom_utils import CustomUtils


class PysfrlEnv(gym.Env):
    #렌더링 모드
    metadata = {'render_modes': ['human']}
    
    def __init__(self, sim: Simulator, learned_idx):                
        self.simulator: Simulator = sim
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

    def is_finished(self):
        return self.simulator.current_state[self.learned_idx, Index.finished.index] == 1        

    def step(self, action):        
        while True:      
            _, visible_idx = self.simulator.get_visible_info()
            if not self.learned_idx in visible_idx:
                self.simulator.step_once()                
            elif len(visible_idx) < 2:
                self.simulator.step_once()                 
            else:
                break
            if self.simulator.time_step > 10000:
                print("inifitie")
                break
        
        visible_state, _ = self.simulator.get_visible_info()
        pre_state = self.simulator.current_state.copy()
        next_state = self.simulator.next_state(pre_state, action)
        next_state = self.simulator.update_new_state(next_state)

        self.simulator.peds.update(next_state)
        self.simulator.time_step += 1        
        obs = self.observation(self.simulator, visible_state)
        reward = self.reward(self.simulator, pre_state, next_state, visible_state)
        done = self.is_finished() or self.simulator.time_step > 1000
        return obs, reward, done, self.info()
    
    def observation(self, sim: Simulator, visible_state):        
        extra_force = sim.extra_forces(visible_state)[self.learned_idx]
        neighbor_info = CustomUtils.neighbor_info(visible_state, self.learned_idx)
        return np.concatenate((extra_force, neighbor_info))

    def reward(self, sim: Simulator, pre_state, next_state, visible_state):
        # idx를 주면, 그 idx와 가장 가까운 거리에 있는 거리를 구하는 함수
        reward = 0
        delta = CustomUtils.goal_distance_delta(pre_state, next_state, self.learned_idx)
        if delta > 0.2:
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
        self.simulator.reset()
        return

