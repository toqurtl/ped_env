import gym
from gym import spaces
from pysfrl.sim.simulator import Simulator
import numpy as np


class PysfrlEnv(gym.Env):
    #렌더링 모드
    metadata = {'render_modes': ['human']}
    
    def __init__(self, sim: Simulator):                
        self.simulator: Simulator = sim
        self.observation_space = spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-3,high=3, shape=(2,), dtype=np.float32)
        self.learned_agent_idx = 0

    @property
    def initial_state(self):
        return self.simulator.initial_state

    @property
    def time_step(self):
        return self.simulator.time_step

    @property
    def num_ped(self):
        return self.simulator.num_peds()

    def step(self, action):
        whole_state = self.simulator.peds.current_state.copy()
        _, visible_idx, _ = self.simulator.get_visible_states(whole_state)
        check = self.learned_agent_idx in visible_idx
        if check:
            self.simulator.step_once(action)
        else:
            reward = 0
        self.simulator.step_once(action)
        # learned_agent 있는지 확인
        # learned_agent 등장할 때부터 learn 시작
        
        # learned_agent 등장 끝나면 시뮬레이션 종료(Done=True)

        # 혹은 learned_agent 등장 이후 너무 시간 오래걸려도 종료(Done=False)
        pass
        # return obs, reward, done, self._get_info()

    def done(self, action):
        pass

    def reset(self):
        pass

