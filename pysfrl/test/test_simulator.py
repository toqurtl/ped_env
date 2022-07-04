# from pysfrl.data.exp_setting import ExperimentSetting
# from pysfrl.sim.simulator import Simulator
from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.new_simulator import NewSimulator
from pysfrl.sim.result.sim_result import SimResult
from pysfrl.data.video_data import VideoData
from pysfrl.visualize.plots import PlotGenerator
import numpy as np
import json
import os


TEST_CFG_PATH = "test\\config\\simulation_config_sample.json"
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})


def sim_cfg_to_simulate():
    # 정리
    with open(TEST_CFG_PATH) as f:
        data = json.load(f)
    config_id = "test"
    cfg = SimulationConfig()
    cfg.set_config(data)
    cfg.set_config_id(config_id)
    s = NewSimulator(cfg)
    s.simulate()
    SimResult.sim_result_to_json(s, "test.json")


def video_to_simulate(scene_folder, sim_result_folder):
    sim_result_path = os.path.join(sim_result_folder, "sim_result.json")
    fig_path = os.path.join(sim_result_folder, "trajectory.png")

    with open(TEST_CFG_PATH, "r") as f:
        default_conifg = json.load(f)

    # video 불러옴
    v = VideoData(scene_folder)    

    # video로부터 config 생성
    sim_cfg = SimulationConfig()
    sim_cfg.set_config(default_conifg)
    sim_cfg.set_ped_info(v.ped_info())

    # cfg로부터 시뮬레이션 수행
    s = NewSimulator(sim_cfg)
    s.simulate()

    # 결과 저장
    SimResult.sim_result_to_json(s, sim_result_path)    
    fig, ax = PlotGenerator.generate_sim_result_plot((-5,5,-10,10), s)    
    fig.savefig(fig_path)
    return


if "__name__" == "__main__":
    pass