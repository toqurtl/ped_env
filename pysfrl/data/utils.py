from pysfrl.data.video_data import VideoData
from pysfrl.config.sim_config import SimulationConfig
import os
import json

def save_info_from_video(v: VideoData, save_path):    
    v.to_json(save_path)
    v.ped_info_to_json(save_path)
    v.trajectory_to_json(save_path)
    v.save(save_path)
    return
    
def generate_config(v: VideoData, cfg_id) -> SimulationConfig:    
    entry_path = os.path.abspath(".")
    default_cfg_path = os.path.join(entry_path, "pysfrl", "test", "data", "simulation_config_sample.json")
    with open(default_cfg_path, "r") as f:
        default_cfg = json.load(f)

    sim_cfg = SimulationConfig()
    sim_cfg.set_config(default_cfg)
    sim_cfg.set_config_id(cfg_id)
    sim_cfg.set_initial_state_info(v.initial_state())
    sim_cfg.set_ped_info(v.ped_info())
    return sim_cfg
