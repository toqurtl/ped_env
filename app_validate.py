from pysfrl.data.video_data import VideoData
from pysfrl.experiment.exp_setting import ExpSetting
from pysfrl.rl.env import PysfrlEnv
from pysfrl.sim.utils.sim_result import SimResult
from pysfrl.experiment.file_finder import FileFinder
from pysfrl.visualize.plots import PlotGenerator
import os
import json
import numpy as np
# np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})


entry_path = os.path.abspath(".")
RL_CFG_PATH = os.path.join(entry_path, "pysfrl", "test", "data", "simulation_nn_config.json")

onedrive_path = os.environ['onedrive']
# 강화학습 힘
exp_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\experiment\\0708")
# 일반(Basic-SFM) 힘
exp_folder_path_2 = os.path.join(onedrive_path, "연구\\pandemic\\experiment\\0708_2")
# 비디오
video_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\data\\ped_texas\\new\\whole")


exp = ExpSetting(exp_folder_path=exp_folder_path)
exp.set_default_cfg_path(RL_CFG_PATH)

exp_2 = ExpSetting(exp_folder_path=exp_folder_path_2)


# FileFinder.exp_folder_path = exp_folder_path_2
FileFinder.exp_folder_path = exp_folder_path
FileFinder.video_folder_path = video_folder_path

for sim in exp.get_simulator_list():
    cfg_idx = sim.cfg.config_id
    if cfg_idx=="29":
        print(FileFinder.valid_result(cfg_idx))        
        video_path = os.path.join(video_folder_path, cfg_idx)
        video_data = VideoData(scene_folder=video_path)
        
        sim.set_time_table(video_data.time_table)
        sim.simulate()

        sim_result_path = FileFinder.sim_result(cfg_idx)
        summary_path = FileFinder.sim_summary(cfg_idx)
        fig_path = FileFinder.sim_trajectory(cfg_idx)
        SimResult.sim_result_to_json(sim, sim_result_path)
        SimResult.summary_to_json(sim, summary_path)
        fig, ax = PlotGenerator.generate_sim_result_plot((-5,5,-10,10), sim)    
        fig.savefig(fig_path)

        origin_states = sim.peds_states
        gt_states = video_data.ground_truth_state()
        valid_path = FileFinder.valid_result(cfg_idx)
        SimResult.validate_to_json(origin_states, gt_states, FileFinder.valid_result(cfg_idx))
    

# ade = 0
# dtw = 0
# ctn = len(exp_2.get_simulator_list())
# for sim in exp_2.get_simulator_list():
#     cfg_idx = sim.cfg.config_id
#     valid_path = FileFinder.valid_result(cfg_idx)
#     with open(valid_path, "r") as f:
#         data = json.load(f)
#     ade += data["ade"]
#     dtw += data["dtw"]

# print(ade/ctn)
# print(dtw/ctn)