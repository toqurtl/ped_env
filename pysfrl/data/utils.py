from pysfrl.data.video_data import VideoData
from pysfrl.config.sim_config import SimulationConfig
from pysfrl.visualize.plots import PlotGenerator
import os
import json


def generate_trajectory_fig(v: VideoData, fig_path):
    xy_range = (-5, 5, -10, 10)
    sub_plots = PlotGenerator.generate_sub_plots(xy_range)    
    sub_plots = PlotGenerator.plot_trajectory(sub_plots, v.ground_truth_state())
    fig, ax = sub_plots
    fig.savefig(fig_path)
    return


def save_info_from_video(v: VideoData, save_path):    
    v.to_json(save_path)
    v.ped_info_to_json(save_path)
    v.trajectory_to_json(save_path)
    v.save(save_path)
    fig_path = os.path.join(save_path, "gt_trajectory.png")
    generate_trajectory_fig(v, fig_path)
    return
    
def generate_config(v: VideoData, cfg_id, default_cfg_path) -> SimulationConfig:        
    with open(default_cfg_path, "r") as f:
        default_cfg = json.load(f)

    sim_cfg = SimulationConfig()
    sim_cfg.set_config(default_cfg)
    sim_cfg.set_config_id(cfg_id)
    sim_cfg.set_initial_state_info(v.initial_state())
    sim_cfg.set_ped_info(v.ped_info())
    return sim_cfg
