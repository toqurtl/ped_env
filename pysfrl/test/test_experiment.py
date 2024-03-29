from pysfrl.test import test_simulator
from pysfrl.experiment.exp_setting import ExpSetting
from pysfrl.data.video_data import VideoData
from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.simulator import Simulator
from pysfrl.sim.utils.sim_result import SimResult
from pysfrl.visualize.plots import PlotGenerator
from pysfrl.experiment import utils
import os
import json

onedrive_path = os.environ['onedrive']
exp_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\experiment\\0704")
video_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\data\\ped_texas\\new\\new_opposite")


def generate_scene_from_video():
    exp = ExpSetting(exp_folder_path=exp_folder_path)
    # 비디오에서 scene 추가
    for vid_id in os.listdir(video_folder_path):
        vid_path = os.path.join(video_folder_path, vid_id)
        video_data = VideoData(scene_folder=vid_path)
        exp.add_scene_from_video(video_data, vid_id)  


def test_simulation_in_experiment_environment():
    exp = ExpSetting(exp_folder_path=exp_folder_path)
    exp.simulate_every_scene()

