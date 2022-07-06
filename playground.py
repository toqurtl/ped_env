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



exp = ExpSetting(exp_folder_path=exp_folder_path)

# for scene_folder_path in exp.scene_folder_path_list():
#     a = utils.gt_trajectory_to_numpy(scene_folder_path)
#     xy_range = (-5, 5, -10, 10)
#     sub_plots = PlotGenerator.generate_sub_plots(xy_range)
#     sub_plots = PlotGenerator.plot_trajectory(sub_plots, a)
#     fig, ax = sub_plots
#     fig_path = os.path.join(scene_folder_path, "data", "gt_trajectory.png")
#     fig.savefig(fig_path)

# for vid_id in os.listdir(video_folder_path):
#     print(vid_id)
#     vid_path = os.path.join(video_folder_path, vid_id)
#     video_data = VideoData(scene_folder=vid_path)
#     exp.add_scene_from_video(video_data, vid_id)    


exp.simulate_scene("118")

