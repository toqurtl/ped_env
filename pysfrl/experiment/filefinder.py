import os
import json
from dis_ped.config.config import PedConfig


class FileFinder(object):
    def __init__(self, config_path):
        self.cfg = PedConfig(config_path)          

    # 비디오 path 관련
    @property
    def vid_path(self):
        return os.path.abspath(self.cfg.vid_folder_path)

    def hp_path(self, idx):
        return os.path.join(self.vid_path, idx, "hp.csv")

    def hp_path_2(self, idx):
        return os.path.join(self.vid_path, idx, idx+"_hp.csv")
    
    def vp_path(self, idx):
        return os.path.join(self.vid_path, idx, "vp.csv")

    def vp_path_2(self, idx):
        return os.path.join(self.vid_path, idx, idx+"_vp.csv")

    # 결과 폴더 관련(force setting)
    @property
    def setting_id(self):
        return self.cfg.setting_id

    ## setting의 결과폴더(force setting을 의미)
    @property
    def result_path(self):
        return os.path.join(self.cfg.result_folder_path, self.setting_id)

    def result_csv_path(self, name):
        return os.path.join(self.result_path, "result_"+name+".csv")

    def result_xlsx_path(self, name):
        return os.path.join(self.result_path, "result_"+name+".xlsx")

    @property
    def simul_time_threshold(self):
        return self.cfg.simul_time_threshold
    
    # Scene 결과폴더
    def env_path(self, idx):
        return os.path.join(self.result_path, str(idx))

    # Scene 결과파일
    def video_info_path(self, idx):
        return os.path.join(self.env_path(idx), "video_info.json")
    
    def gt_path(self, idx):
        return os.path.join(self.env_path(idx),"gt.json")

    def compare_path(self):
        return os.path.join(self.result_path, "compare.json")

    def compare_path_idx(self):
        return os.path.join(self.result_path, "compare_idx.json")
    
    def summary_path(self, idx):
        return os.path.join(self.env_path(idx), "summary.json")

    def plot_path(self, idx):
        return os.path.join(self.env_path(idx), "plot")

    def animation_path(self, idx):
        return os.path.join(self.env_path(idx), "animation")

    # 결과시계열데이터
    def simul_result_path(self, idx):
        return os.path.join(self.env_path(idx), "simul_result.json")
    
    # valid값 시계열
    def valid_path(self, idx):
        return os.path.join(self.env_path(idx), "valid.json")
    
    def get_idx_folder_list(self):        
        idx_folder_list = []
        for element in os.listdir(self.result_path):
            element_path = os.path.join(self.result_path, element)
            if os.path.isdir(element_path):
                idx_folder_list.append(element)
        return idx_folder_list

    # def get_exp_folder_list(self):
    #     folder_list = []
    #     for element in os.listdir(self.result_path):
    #         element_path = os.path.join(self.result_path, element)
    #         if os.path.isdir(element_path):
    #             folder_list.append(element_path)
    #     return folder_list

    def get_vid_folder_list(self):
        folder_list = []        
        for element in os.listdir(self.vid_path):
            element_path = os.path.join(self.vid_path, element)
            if os.path.isdir(element_path):                
                folder_list.append(element_path)                
        return folder_list

    def summary_to_json(self, idx, success):
        data = {}
        data["success"] = success
        
        with open(self.summary_path(idx), 'w') as f:
            json.dump(data, f, indent=4)
        return

    def is_comparable(self, idx):        
        with open(self.summary_path(idx),"r",encoding="UTF-8") as f:
            success = bool(json.load(f)["success"])
        if not success:
            return False
        return True

    def is_success(self, idx):        
        if not os.path.exists(self.compare_path()):
            return False
        with open(self.compare_path(), "r", encoding="UTF-8") as f:
            success = bool(json.load(f)["state"]["success"])
        return success
    