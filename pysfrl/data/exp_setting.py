import json
import os
from .video_data import VideoData
from dis_ped.video.peds import Pedestrians
from dis_ped.config.filefinder import FileFinder
from dis_ped.config.config import PedConfig

class ExperimentSetting(object):
    def __init__(self, config_path, idx):
        self.cfg = PedConfig(config_path)
        self.file_finder = FileFinder(config_path)
        self.idx = idx
        self._set_experiment()
        self._save_vid_data()

    @property
    def video(self) -> VideoData:
        try:
            hp_path = self.file_finder.hp_path(self.idx)
            vp_path = self.file_finder.vp_path(self.idx)        
            v = VideoData(hp_path, vp_path, self.idx)
        except FileNotFoundError:            
            hp_path = self.file_finder.hp_path_2(self.idx)
            vp_path = self.file_finder.vp_path_2(self.idx)        
            v = VideoData(hp_path, vp_path, self.idx)                         
        return v
        
    @property
    def peds(self):
        video_info_path = self.file_finder.video_info_path(self.idx)
        with open(video_info_path, 'r') as f:
            json_data = json.load(f)
        return Pedestrians(json_data)

    @property
    def obstacle(self):
        return self.cfg.obstacles

    @property
    def simul_result_path(self):
        return self.file_finder.simul_result_path(self.idx, self.force_idx)

    @property
    def animation_path(self):
        return self.file_finder.animation_path(self.idx, self.force_idx)

    @property
    def plot_path(self):
        return self.file_finder.plot_path(self.idx, self.force_idx)

    @property
    def force_config(self):
        return self.cfg.force_config

    @property
    def scene_config(self):
        return self.cfg.scene_config

    @property
    def simul_config(self):
        return self.cfg.simul_config
    
    def force_list(self):
        return self.force_config["set"].keys()

    def _set_experiment(self):
        # set folder
        result_path = self.file_finder.result_path
        env_path = self.file_finder.env_path(self.idx) 
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        if not os.path.exists(env_path):
            os.mkdir(env_path)

    def _save_vid_data(self):
        video_info_path = self.file_finder.video_info_path(self.idx)
        gt_path = self.file_finder.gt_path(self.idx)
        self.video.to_json(video_info_path)        
        self.video.trajectory_to_json(gt_path)
        return


