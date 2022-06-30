from pysfrl.data.video_data import VideoData
import json

def raw_data_info():
    # 정리
    config_path = "test\\config\\simulation_config_sample.json"
    with open(config_path, "r") as f:
        default_conifg = json.load(f)


    scene_folder = "C:\\Users\\yoon9\\data\\pandemic\\new_opposite\\15"
    save_folder = "C:\\Users\\yoon9\\data\\pandemic\\new_opposite\\15"
    v = VideoData(scene_folder)
    # v.trajectory_to_json(save_folder)
    # v.ped_info_to_json(save_folder)


raw_data_info()

# if __name__ == "__main__":
#     raw_data_info()