# from pysfrl.data.exp_setting import ExperimentSetting
# from pysfrl.sim.simulator import Simulator
from pysfrl.config.config import SimulationConfig
from pysfrl.sim.new_simulator import NewSimulator
from pysfrl.sim.result.sim_result import SimResult
import numpy as np
import json

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

# 정리
with open("test\\config\\simulation_config_sample.json") as f:
    data = json.load(f)

cfg = SimulationConfig(data)
s = NewSimulator(cfg)
s.simulate()
SimResult.sim_result_to_json(s, "test.json")
