import json
import numpy as np
from pysfrl.sim.parameters import DataIndex as Index
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
# only video and 

def get_distance(data, p_idx_1, p_idx_2):
    px_1, py_1 = data[p_idx_1][Index.px.index], data[p_idx_1][Index.py.index]
    px_2, py_2 = data[p_idx_2][Index.px.index], data[p_idx_2][Index.py.index]
    return ((px_1-px_2)**2 + (py_1-py_2)**2)**0.5


class ResultData(object):
    def __init__(self, origin_path, gt_path=None):
        with open(origin_path, 'r') as f:
            self.origin_data = json.load(f)
        
        if gt_path is not None:
            with open(gt_path, 'r') as f:
                self.gt_data = json.load(f)

            self.gt_states = \
                np.array([data["states"] for data in self.gt_data.values()])

        self.origin_states = \
            np.array([data["states"] for data in self.origin_data.values()])
        
        self.gt_states = None
        

        return

    @property
    def num_person(self):
        return len(self.origin_states[0])

    @property
    def simulation_time(self):
        return len(self.origin_states)

    @property
    def gt_time(self):
        return len(self.gt_states)
        
    def ade_range(self, person_idx):
        origin_person_data = self.origin_states[:, person_idx]
        gt_person_data = self.gt_states[:, person_idx]
        start = origin_person_data[0][Index.start_time.index]
        origin_finish = len(origin_person_data)
        gt_finish = len(gt_person_data)

        for idx, data in enumerate(origin_person_data):                
            if data[Index.finished.index] == 1:               
                origin_finish = idx-1
                break
        
        for idx, data in enumerate(gt_person_data):                
            if data[Index.finished.index] == 1:               
                gt_finish = idx-1
                break
        finish = min(origin_finish, gt_finish)
        return int(start), int(finish)

    def ade_of_person(self, person_idx):
        origin_person_data = self.origin_states[:, person_idx]
        gt_person_data = self.gt_states[:, person_idx]
        start, finish = self.ade_range(person_idx)
        traj_x = origin_person_data[start:finish+1, Index.px.index] 
        traj_y = origin_person_data[start:finish+1, Index.py.index]
        
        gt_x = gt_person_data[start:finish+1, Index.px.index]
        gt_y = gt_person_data[start:finish+1, Index.py.index]
        
        try:        
            d_x = np.sum(traj_x - gt_x) / (finish-start + 1)
            d_y = np.sum(traj_y - gt_y) / (finish-start + 1)
        except ValueError:
            traj_x, traj_y = traj_x[:-1], traj_y[:-1]
            d_x = np.sum(traj_x - gt_x) / (finish-start + 1)
            d_y = np.sum(traj_y - gt_y) / (finish-start + 1)

        return (d_x**2 + d_y**2)**0.5
    
    def fde_of_person(self, person_idx):
        origin_person_data = self.origin_states[:, person_idx]
        gt_person_data = self.gt_states[:, person_idx]
        
        origin_finish = len(origin_person_data)
        gt_finish = len(gt_person_data)

        for idx, data in enumerate(origin_person_data):                
            if data[Index.finished.index] == 1:               
                origin_finish = idx-1
                break
        
        for idx, data in enumerate(gt_person_data):                
            if data[Index.finished.index] == 1:               
                gt_finish = idx-1
                break


        last_traj_x = origin_person_data[origin_finish, Index.px.index]
        last_traj_y = origin_person_data[origin_finish, Index.py.index]    
        gt_traj_x = gt_person_data[gt_finish-1, Index.px.index]
        gt_traj_y = gt_person_data[gt_finish-1, Index.py.index]
        
        return ((last_traj_x - gt_traj_x)**2 + (last_traj_y - gt_traj_y)**2)**0.5

    def ade_of_scene(self):
        ade_sum = 0
        for person_idx in range(0, self.num_person):
            ade_sum += self.ade_of_person(person_idx)
        return ade_sum / self.num_person

    def fde_of_scene(self):
        fde_sum = 0        
        for person_idx in range(0, self.num_person):
            fde_sum += self.fde_of_person(person_idx)
        return fde_sum / self.num_person

    def risk_index_of_person(self, person_idx, distance):
        start, finish = self.ade_range(person_idx)
        ctn = 0
        check_data = []
        for time in range(start, finish+1):
            try:
                data = self.origin_states[time]         
            except IndexError as e:
                continue
            for idx, person_data in enumerate(data):
                is_visible = person_data[Index.visible.index] == 1
                if idx is not person_idx and is_visible:
                    dis = get_distance(data, person_idx, idx)
                    if dis < distance:
                        ctn += 1
                        check_data.append(time)
        return ctn, finish-start + 1, check_data

    def risk_index_of_scene(self, distance):
        result_data = {}
        avg = 0
        for person_idx in range(0, self.num_person):        
            ctn, total_time, check_data = self.risk_index_of_person(person_idx, distance)
            avg += ctn / total_time
        return avg / self.num_person

    def risk_index_of_video_person(self, person_idx, distance):
        start, finish = self.ade_range(person_idx)
        ctn = 0
        check_data = []
        for time in range(start, finish+1):
            try:
                data = self.gt_states[time]         
            except IndexError as e:
                continue
            for idx, person_data in enumerate(data):
                is_visible = person_data[Index.visible.index] == 1
                if idx is not person_idx and is_visible:
                    dis = get_distance(data, person_idx, idx)
                    if dis < distance:
                        ctn += 1
                        check_data.append(time)
                        break
        return ctn, finish-start+1, check_data
    
    def risk_index_of_video_scene(self, distance):
        result_data = {}
        avg = 0
        for person_idx in range(0, self.num_person):        
            ctn, total_time, check_data = self.risk_index_of_video_person(person_idx, distance)
            avg += ctn / total_time
        return avg / self.num_person

    def dtw_of_person(self, person_idx):
        origin_person_data = self.origin_states[:, person_idx]
        gt_person_data = self.gt_states[:, person_idx]
        start, finish = self.ade_range(person_idx)
        traj_x = origin_person_data[start:finish+1, Index.px.index] 
        traj_y = origin_person_data[start:finish+1, Index.py.index]
        
        gt_x = gt_person_data[start:finish+1, Index.px.index]
        gt_y = gt_person_data[start:finish+1, Index.py.index]        
        
        try:        
            traj = np.column_stack((traj_x, traj_y))
            gt = np.column_stack((gt_x, gt_y))
            distance, path = fastdtw(traj, gt, dist=euclidean)

        except ValueError:
            traj_x, traj_y = traj_x[:-1], traj_y[:-1]
            traj = np.column_stack((traj_x, traj_y))
            gt = np.column_stack((gt_x, gt_y))
            distance, path = fastdtw(traj, gt, dist=euclidean)

        return distance

    def dtw_of_scene(self):
        dtw_sum = 0
        for person_idx in range(0, self.num_person):
            dtw_sum += self.dtw_of_person(person_idx)
        return dtw_sum / self.num_person

    def result(self, vid_id, force_name, success):
        data = {}
        data["basic"] = {}
        data["result"] = {}
        data["basic"]["vid_id"] = vid_id
        data["basic"]["num_person"] = self.num_person
        data["basic"]["gt_time"] = len(self.gt_data)  
        data["basic"]["social"] = self.risk_index_of_video_scene(2) 
        data["result"]["force_name"] = force_name
        data["result"]["success"] = success
        data["result"]["simulation_time"] = self.simulation_time_error()
        data["result"]["sdcr_error"] = self.sdcr_error(2)
        data["result"]["ade"] = self.ade_of_scene()
        data["result"]["fde"] = 0
        data["result"]["dtw"] = self.dtw_of_scene()
        # TODO - 추후 목표 distance 설정

        data["result"]["social"] = self.risk_index_of_scene(2)
        return data
                    
    def to_json(self, file_path, vid_id, force_name, success):
        state = self.result(vid_id, force_name, success)
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)
        return

    def simulation_time_error(self):
        return (len(self.origin_data) - len(self.gt_data)) / len(self.gt_data)

    def sdcr_error(self, distance):
        sdcr_gt = self.risk_index_of_video_scene(distance)
        sdcr_simul = self.risk_index_of_scene(distance)
        if sdcr_gt == 0:
            return 0
        else:
            return (sdcr_simul-sdcr_gt)/sdcr_gt
        
    def minmax(self, person_idx):
        px_origin_min = np.min(self.origin_states[:, person_idx, Index.px.index])
        py_origin_min = np.min(self.origin_states[:, person_idx, Index.py.index])
        px_gt_min = np.min(self.gt_states[:, person_idx, Index.px.index])
        py_gt_min = np.min(self.gt_states[:, person_idx, Index.py.index])
        px_origin_max = np.max(self.origin_states[:, person_idx, Index.px.index])
        py_origin_max = np.max(self.origin_states[:, person_idx, Index.py.index])
        px_gt_max = np.max(self.gt_states[:, person_idx, Index.px.index])
        py_gt_max = np.max(self.gt_states[:, person_idx, Index.py.index])
        px_min = min(px_origin_min, px_gt_min)
        py_min = min(py_origin_min, py_gt_min)
        px_max = min(px_origin_max, px_gt_max)
        py_max = min(py_origin_max, py_gt_max)
        return (px_min, py_min, px_max, py_max)
        
