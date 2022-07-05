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

    @property
    def initial_state(self):
        return self.simulator.initial_state

    @property
    def time_step(self):
        return self.simulator.time_step

    @property
    def num_ped(self):
        return self.simulator.num_peds()

    def test(self):
        pass

