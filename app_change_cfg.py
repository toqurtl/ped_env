from pysfrl.experiment.exp_setting import ExpSetting
from pysfrl.data.video_data import VideoData
from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.simulator import Simulator
from pysfrl.sim.utils.sim_result import SimResult
from pysfrl.visualize.plots import PlotGenerator
from pysfrl.experiment import utils
import os
import sys
import json

env_name = sys.argv[1]
cfg_path = sys.argv[2]

entry_path = os.path.abspath(".")

onedrive_path = os.environ['onedrive']

exp_path = "연구\\pandemic\\experiment\\0717_fixed"
vid_path = "연구\\pandemic\\data\\ped_texas\\new\\whole"

# 강화학습 힘
exp_folder_path = os.path.join(onedrive_path, exp_path, env_name)
# 비디오
video_folder_path = os.path.join(onedrive_path, vid_path)

exp = ExpSetting(exp_folder_path=exp_folder_path)

exp_cfg_path = os.path.join(exp_folder_path, "sim_cfg.json")

with open(cfg_path, "r") as f:
    data = json.load(f)

exp.change_cfg(data)
