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
import pickle


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

folder_path_list = exp.scene_folder_path_list()

ade_list = []
dtw_list = []

high_sdvr_ade_list = []
low_sdvr_ade_list = []

high_sdvr_dtw_list = []
low_sdvr_dtw_list = []

for path in folder_path_list:
    if os.path.isdir(path):
        print(path)      
        valid_path = os.path.join(path, "valid.json")
        with open(valid_path, "r") as f:
            data = json.load(f)
        
        sdvr_sim, sdvr_error = data["sdvr"], data["sdvr_error"]
        try:
            sdvr = sdvr_sim / (1+sdvr_error)
        except:
            sdvr = 0
        ade_list.append(data["ade"])
        dtw_list.append(data["dtw"])
        if sdvr > 0.247:
            high_sdvr_ade_list.append(data["ade"])
            high_sdvr_dtw_list.append(data["dtw"])
        else:
            low_sdvr_ade_list.append(data["ade"])
            low_sdvr_dtw_list.append(data["dtw"])

def avg(temp_list):
    return sum(temp_list) / len(temp_list)

ade_avg, dtw_avg = avg(ade_list), avg(dtw_list)
high_ade_avg, high_dtw_avg = avg(high_sdvr_ade_list), avg(high_sdvr_dtw_list)
low_ade_avg, low_dtw_avg = avg(low_sdvr_ade_list), avg(low_sdvr_dtw_list)

data ={
    "ade": ade_avg,
    "dtw": dtw_avg,
    "high_ade": high_ade_avg,
    "high_dtw": high_dtw_avg,
    "low_ade": low_ade_avg,
    "low_dtw": low_dtw_avg,
}

with open("test.json", "w") as f:
    json.dump(data, f, indent=4)
        

        