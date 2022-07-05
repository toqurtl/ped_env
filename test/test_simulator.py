# from pysfrl.data.exp_setting import ExperimentSetting
# from pysfrl.sim.simulator import Simulator
from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.simulator import Simulator
from pysfrl.sim.result.sim_result import SimResult
from pysfrl.data.video_data import VideoData
from pysfrl.visualize.plots import PlotGenerator
import numpy as np
import json
import os


np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})


def sim_cfg_to_simulate():
    # 정리
    with open("test\\config\\simulation_config_sample.json") as f:
        data = json.load(f)
    config_id = "test"
    cfg = SimulationConfig()
    cfg.set_config(data)
    cfg.set_config_id = config_id
    s = Simulator(cfg)
    s.simulate()
    SimResult.sim_result_to_json(s, "test.json")

def video_to_simulate():
    # 정리
    config_path = "test\\config\\simulation_config_sample.json"
    with open(config_path, "r") as f:
        default_conifg = json.load(f)


    scene_folder = "C:\\Users\\yoon9\\data\\pandemic\\new_opposite\\15"
    save_folder = "C:\\Users\\yoon9\\data\\pandemic\\new_opposite\\15"
    v = VideoData(scene_folder)
    sim_result_folder = "C:\\Users\\yoon9\\workspace\\ped_result\\0629"
    sim_result_path = os.path.join(sim_result_folder, "sim_result.json")
    sim_cfg = SimulationConfig()
    sim_cfg.set_config(default_conifg)
    sim_cfg.set_ped_info(v.ped_info())
    s = Simulator(sim_cfg)
    s.simulate()    
    SimResult.sim_result_to_json(s, sim_result_path)
    fig, ax = PlotGenerator.generate_sim_result_plot((-5,5,-10,10), s)
    fig.save("test.png")
    return


if "__name__" == "__main__":
    pass