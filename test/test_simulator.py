from ..pysfrl.sim.simulator import Simulator
from ..pysfrl.data.exp_setting import ExperimentSetting
import numpy as np
import sys

# print할 때
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

config_path = sys.argv[0]

exp = ExperimentSetting('config/sample_1.json', 1)
s = Simulator()