from pysfrl.data.video_data import VideoData
from pysfrl.experiment.exp_setting import ExpSetting
from pysfrl.rl.env import PysfrlEnv
from pysfrl.sim.utils.sim_result import SimResult
from pysfrl.experiment.file_finder import FileFinder
from pysfrl.visualize.plots import PlotGenerator
import os
import json
import numpy as np
import sys
# np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
env_name = sys.argv[1]

entry_path = os.path.abspath(".")
onedrive_path = os.environ['onedrive']
exp_path = "연구\\pandemic\\experiment\\0717_fixed"
vid_path = "연구\\pandemic\\data\\ped_texas\\new\\whole"

exp_folder_path = os.path.join(onedrive_path, exp_path, env_name)
video_folder_path = os.path.join(onedrive_path, vid_path)

exp = ExpSetting(exp_folder_path=exp_folder_path)
cfg_path = os.path.join(exp_folder_path, "sim_cfg.json")
exp.set_default_cfg_path(cfg_path)


FileFinder.exp_folder_path = exp_folder_path
FileFinder.video_folder_path = video_folder_path

simulator_list = exp.get_simulator_list()

ade, dtw, sdvr, sdvr_error, simulation_time_error, fde = 0, 0, 0, 0, 0, 0
ctn = len(simulator_list)
for sim in simulator_list:
    cfg_idx = sim.cfg.config_id
    valid_path = FileFinder.valid_result(cfg_idx)
    with open(valid_path, "r") as f:
        data = json.load(f)
    ade += data["ade"]
    dtw += data["dtw"]
    sdvr += data["sdvr"]
    simulation_time_error += data["simulation_time_error"]
    sdvr_error += data["sdvr_error"]
    fde += data["fde"]
    

data = {
    "ade": ade/ctn,
    "dtw": dtw/ctn,
    "sdvr": sdvr/ctn,
    "sdvr_error": sdvr_error/ctn,
    "simulation_time_error": simulation_time_error/ctn,
    "fde": fde/ctn
}

file_path = FileFinder.exp_result()
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)