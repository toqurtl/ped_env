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
exp.set_default_cfg_path(cfg_path)

exp_cfg_path = os.path.join(exp_folder_path, "sim_cfg.json")


# 비디오에서 scene 추가
for vid_id in os.listdir(video_folder_path):
    print("add_environment", vid_id)
    vid_path = os.path.join(video_folder_path, vid_id)
    video_data = VideoData(scene_folder=vid_path)
    exp.add_scene_from_video(video_data, vid_id)


with open(cfg_path, "r") as f:
    data = json.load(f)
    
with open(exp_cfg_path, "w") as f:
    json.dump(data, f, indent=4)    

