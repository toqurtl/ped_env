import os


class FileFinder(object):
    exp_folder_path = "."
    video_folder_path = "."

    @classmethod
    def exp_result(cls):
        return os.path.join(cls.exp_folder_path, "summary.json")

    @classmethod
    def video_trajectory(cls, cfg_id):
        return os.path.join(cls.exp_folder_path, cfg_id, "data", "trajectory.json")
    
    @classmethod
    def ped_info(cls, cfg_id):
        return os.path.join(cls.exp_folder_path, cfg_id, "data", "ped_info.json")
    
    @classmethod
    def basic_info(cls, cfg_id):
        return os.path.join(cls.exp_folder_path, cfg_id, "data", "basic_info.json")

    @classmethod
    def sim_cfg(cls, cfg_id):
        return os.path.join(cls.exp_folder_path, cfg_id, "sim_cfg.json")

    @classmethod
    def exp_cfg(cls):
        return os.path.join(cls.exp_folder_path, "sim_cfg.json")
    
    @classmethod
    def sim_result(cls, cfg_id):
        return os.path.join(cls.exp_folder_path, cfg_id, "sim_result.json")

    @classmethod
    def sim_summary(cls, cfg_id):
        return os.path.join(cls.exp_folder_path, cfg_id, "summary.json")

    @classmethod
    def sim_trajectory(cls, cfg_id):
        return os.path.join(cls.exp_folder_path, cfg_id, "trajectory.png")


    @classmethod
    def valid_result(cls, cfg_id):
        return os.path.join(cls.exp_folder_path, cfg_id, "valid.json")

    @classmethod
    def video_path(cls, vid_id):
        return os.path.join(cls.video_folder_path, vid_id)
    
    
    



