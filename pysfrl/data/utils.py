from pysfrl.data.video_data import VideoData


# video 저장된 폴더 -> basic_info, ped_info, trajectory_info json으로 변경
# ped_info -> simulation config로 변경할 때 활용
# 
def get_ped_info_from_video(video_folder_path):
    v = VideoData(video_folder_path)
    v.to_json(video_folder_path)
    v.ped_info_to_json(video_folder_path)
    v.trajectory_to_json(video_folder_path)
    return
    
    