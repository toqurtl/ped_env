import os
import numpy as np
import json

# exp 폴더구조에서 원하는 데이터를 얻는 함수들

# trajectroy.json 데이터에서 3차원 numpy 형태의 경로파일 얻기
# for plotting
def gt_trajectory_to_numpy(scene_folder):
    gt_path_file = os.path.join(scene_folder, "data", "trajectory.json")
    with open(gt_path_file, "r") as f:
        traj_data = json.load(f)
    
    state_array = []
    for value in traj_data.values():
        state_array.append(value["states"])
    return np.array(state_array)