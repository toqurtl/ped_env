from pysfrl.config.sim_config import SimulationConfig
from pysfrl.data.video_data import VideoData
from pysfrl.data import utils
from pysfrl.sim.simulator import Simulator
from pysfrl.sim.utils.sim_result import SimResult
from pysfrl.visualize.plots import PlotGenerator
from pysfrl.experiment.file_finder import FileFinder
import json
import os


entry_path = os.path.abspath(".")
DEFAULT_CFG_PATH = os.path.join(entry_path, "pysfrl", "test", "data", "simulation_config_sample.json")

# Experiment 폴더 관리(만들고, State 보고 등등)
class ExpSetting(object):
    def __init__(self, exp_folder_path):
        self.exp_folder_path = exp_folder_path
        self.default_cfg_path = DEFAULT_CFG_PATH
        self.valid_folder_path = None

    def set_default_cfg_path(self, cfg_path):
        self.default_cfg_path = cfg_path

    def get_simulator_list(self):
        return [self.get_simulator(cfg_id) for cfg_id in self.cfg_id_list()]            

    def scene_folder_path_list(self):
        folder_list = []
        for scene_path in os.listdir(self.exp_folder_path):
            folder_list.append(os.path.join(self.exp_folder_path, scene_path))
        return folder_list

    def cfg_id_list(self):
        return [cfg_id for cfg_id in os.listdir(self.exp_folder_path)]
        
    def scene_folder_path(self, cfg_id):
        return os.path.join(self.exp_folder_path, cfg_id)
    
    def ground_truth_folder_path(self, cfg_id):
        return os.path.join(self.scene_folder_path(cfg_id), "data")

    def compare_folder_path(self, cfg_id):
        return os.path.join(self.scene_folder_path(cfg_id), "compare")

    def add_scene(self, sim_cfg: SimulationConfig):        
        cfg_folder_path = self.scene_folder_path(sim_cfg.config_id)
        sim_cfg_file_path = os.path.join(cfg_folder_path, "sim_cfg.json")
        ground_truth_data_path = self.ground_truth_folder_path(sim_cfg.config_id)
        # 입력한 sim_cfg의 폴더가 있는지 확인
        if not os.path.exists(cfg_folder_path):
            os.makedirs(cfg_folder_path)
            os.makedirs(ground_truth_data_path)            
        sim_cfg.save(sim_cfg_file_path)
        return

    def add_scene_from_video(self, v: VideoData, cfg_id):        
        sim_config: SimulationConfig = utils.generate_config(v, cfg_id, default_cfg_path=self.default_cfg_path)        
        self.add_scene(sim_config)
        config_id = sim_config.config_id        
        utils.save_info_from_video(v, self.ground_truth_folder_path(config_id))
        return

    def simulate_scene(self, cfg_id):
        s = self.get_simulator(cfg_id)
        folder_path = self.scene_folder_path(cfg_id)
        success = s.simulate()        
        sim_result_path = os.path.join(folder_path, "sim_result.json")
        summary_path = os.path.join(folder_path, "summary.json")
        fig_path = os.path.join(folder_path, "trajectory.png")
        SimResult.sim_result_to_json(s, sim_result_path)
        SimResult.summary_to_json(s, summary_path)
        fig, ax = PlotGenerator.generate_sim_result_plot((-5,5,-10,10), s)    
        fig.savefig(fig_path)
        return success

    def get_simulator(self, cfg_id):
        sim_cfg = SimulationConfig()
        folder_path = self.scene_folder_path(cfg_id)
        sim_cfg_path = os.path.join(folder_path, "sim_cfg.json")        
        with open(sim_cfg_path, "r") as f:
            data = json.load(f)
        sim_cfg.set_config(data)        
        return Simulator(sim_cfg)

    def simulate_every_scene(self):
        for cfg_id in self.cfg_id_list():     
            success = self.simulate_scene(cfg_id)
