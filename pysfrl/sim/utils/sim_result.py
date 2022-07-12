from pysfrl.sim.simulator import Simulator
from pysfrl.sim.parameters import DataIndex as Index
from scipy.spatial.distance import euclidean
import json
import numpy as np
from fastdtw import fastdtw


class SimResult(object):

    @staticmethod
    def validate_to_json(origin_states, gt_states, file_path, distance=2):
        data ={}        
        data["ade"] = SimResult.ade_of_scene(origin_states, gt_states)
        data["dtw"] = SimResult.dtw_of_scene(origin_states, gt_states)
        data["fde"] = SimResult.fde_of_scene(origin_states, gt_states)
        data["sdvr"] = SimResult.risk_index_of_scene(origin_states, distance=2)
        data["simulation_time_error"] = SimResult.simulation_time_error(origin_states, gt_states)
        data["sdvr_error"] = SimResult.sdvr_error(origin_states, gt_states, distance)       
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)        
        return

    @staticmethod
    def sim_result_to_json(sim: Simulator, file_path):    
        result_data = {}        
        for i in range(0, sim.time_step+1):            
            result_data[i] = {
                "step_width": sim.step_width,
                "states": sim.peds.states[i].tolist()
            }
        
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4)        
        return

    @staticmethod
    def summary_to_json(sim: Simulator, file_path):
        data = {}
        data["num_person"] = sim.num_peds()
        data["sdvr"] = SimResult.risk_index_of_scene(sim.peds_states, 2)
        data["simulation_time"] = len(sim.peds_states)
        data["success"] = sim.success
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return

    @staticmethod
    def ade_of_scene(origin_states, gt_states):
        num_person = len(origin_states[0])
        ade_sum = 0
        for person_idx in range(0, num_person):
            ade_sum += SimResult.ade_of_person(origin_states, gt_states, person_idx)
        return ade_sum / num_person

    @staticmethod
    def ade_of_person(origin_states, gt_states, person_idx):
        origin_person_data = origin_states[:, person_idx]
        gt_person_data = gt_states[:, person_idx]
        start, finish = SimResult.range_of_person(person_idx, origin_states, gt_states=gt_states)
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

    @staticmethod
    def fde_of_scene(origin_states, gt_states):
        num_person = len(origin_states[0])
        fde_sum = 0        
        for person_idx in range(0, num_person):
            fde_sum += SimResult.fde_of_person(origin_states, gt_states, person_idx)
        return fde_sum / num_person

    @staticmethod
    def fde_of_person(origin_states, gt_states, person_idx):
        origin_person_data = origin_states[:, person_idx]
        gt_person_data = gt_states[:, person_idx]
        
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
        try:
            last_traj_x = origin_person_data[origin_finish, Index.px.index]
            last_traj_y = origin_person_data[origin_finish, Index.py.index]    
        except:
            last_traj_x = origin_person_data[origin_finish-1, Index.px.index]
            last_traj_y = origin_person_data[origin_finish-1, Index.py.index]    

        gt_traj_x = gt_person_data[gt_finish-1, Index.px.index]
        gt_traj_y = gt_person_data[gt_finish-1, Index.py.index]
        
        return ((last_traj_x - gt_traj_x)**2 + (last_traj_y - gt_traj_y)**2)**0.5

    @staticmethod
    def dtw_of_scene(origin_states, gt_states):
        num_person = len(origin_states[0])
        dtw_sum = 0
        for person_idx in range(0, num_person):
            dtw_sum += SimResult.dtw_of_person(origin_states, gt_states, person_idx)
        return dtw_sum / num_person

    @staticmethod
    def dtw_of_person(origin_states, gt_states, person_idx):
        origin_person_data = origin_states[:, person_idx]
        gt_person_data = gt_states[:, person_idx]
        start, finish = SimResult.range_of_person(person_idx, origin_states, gt_states=gt_states)
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

    @staticmethod
    def risk_index_of_scene(origin_states, distance):
        num_person = len(origin_states[0])
        avg = 0
        for person_idx in range(0, num_person):        
            ctn, total_time, check_data = SimResult.risk_index_of_person(origin_states, person_idx, distance)
            try:
                avg += ctn / total_time
            except:
                avg += 0
        return avg / num_person

    @staticmethod
    def risk_index_of_person(origin_states, person_idx, distance):
        start, finish = SimResult.range_of_person(person_idx, origin_states)
        ctn = 0
        check_data = []
        for time in range(start, finish+1):
            try:
                state = origin_states[time]
            except IndexError as e:
                continue
            for idx, person_data in enumerate(state):
                is_visible = person_data[Index.visible.index] == 1
                if idx is not person_idx and is_visible:
                    dis = SimResult.get_distance(state, person_idx, idx)
                    if dis < distance:
                        ctn += 1
                        check_data.append(time)
        return ctn, finish-start + 1, check_data

    @staticmethod
    def sdvr_error(origin_states, gt_states, distance):
        sdvr_gt = SimResult.risk_index_of_scene(gt_states, distance)
        sdvr_simul = SimResult.risk_index_of_scene(origin_states, distance)
        if sdvr_gt == 0:
            return 0
        else:
            return (sdvr_simul-sdvr_gt)/sdvr_gt

    @staticmethod
    def simulation_time_error(origin_states, gt_states):
        return (len(origin_states) - len(gt_states)) / len(gt_states)

    @staticmethod
    def range_of_person(person_idx, origin_states, gt_states=None):
        origin_person_data = origin_states[:, person_idx]        
        start = origin_person_data[0][Index.start_time.index]
        origin_finish = len(origin_person_data)

        for idx, data in enumerate(origin_person_data):                
            if data[Index.finished.index] == 1:               
                origin_finish = idx-1
                break
        
        if gt_states is None:
            return int(start), int(origin_finish)
        
        gt_person_data = gt_states[:, person_idx]     
        gt_finish = len(gt_person_data)
        
        for idx, data in enumerate(gt_person_data):                
            if data[Index.finished.index] == 1:               
                gt_finish = idx-1
                break

        finish = min(origin_finish, gt_finish)
        return int(start), int(finish)

    @staticmethod
    def get_distance(state, p_idx_1, p_idx_2):
        px_1, py_1 = state[p_idx_1][Index.px.index], state[p_idx_1][Index.py.index]
        px_2, py_2 = state[p_idx_2][Index.px.index], state[p_idx_2][Index.py.index]
        return ((px_1-px_2)**2 + (py_1-py_2)**2)**0.5