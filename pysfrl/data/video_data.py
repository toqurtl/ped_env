import pandas as pd
import numpy as np
import json
import os
import pickle


# Case의 데이터를 다루기 위한 class
class VideoData(object):
    # x_path: hp.csv, y_path: vp.csv
    def __init__(self, scene_folder):
        x_path = os.path.join(scene_folder, "hp.csv")
        y_path = os.path.join(scene_folder, "vp.csv")
        self.x_origin: np.ndarray = pd.read_csv(x_path).to_numpy()
        self.y_origin: np.ndarray = pd.read_csv(y_path).to_numpy()        
        self.origin_data = {
            "x": self.x_origin,
            "y": self.y_origin
        }

    @property
    def num_person(self):
        _, num_col = self.x_origin.shape
        return num_col - 1

    # 총 time_step
    @property
    def num_data(self):
        return len(self.x_origin)        

    @property
    def time_table(self):
        return np.diff(self.x_origin[:,0])/1000

    @property
    def pos_x(self):
        return self.x_origin[:, 1:]

    @property
    def pos_y(self):
        return self.y_origin[:, 1:]

    def pos_of_person(self, person_idx):
        return self.pos_x[:, person_idx], self.pos_y[:, person_idx] 

    @property
    def position_vec(self):
        return np.stack((self.pos_x, self.pos_y), axis=-1)

    @property
    def velocity_vec(self):
        time = np.stack((self.time_table, self.time_table), axis=-1)
        time = np.expand_dims(time, axis=1)
        return np.diff(self.position_vec, axis=0) / time

    # 한 장면만 나온 경우 문제 생김(데이터 검사할 때 필요)
    @property
    def initial_direction(self):
        directions = []    
        for person_idx in range(0, self.num_person):
            start_index = self.start_time_array[person_idx]
            start_pos = self.position_vec[start_index][person_idx]
            next_pos = self.position_vec[start_index+1][person_idx]
            directions.append(next_pos - start_pos)

        return directions / np.expand_dims(np.linalg.norm(directions, axis=1), axis=1)

    @property
    def average_velocity(self):                
        return np.nanmean(self.velocity_vec, axis=0)

    @property
    def initial_speed(self):
        average_speed = np.linalg.norm(self.average_velocity, axis=1)        
        return np.expand_dims(average_speed, axis=1) * self.initial_direction

    @property
    def start_time_array(self):
        start_time_list = []
        for person_idx in range(0, self.num_person):
            # start_time, _ = self.represent_time(self.pos_x[:, person_idx])
            start_time, _ = self.represent_time(person_idx)
            start_time_list.append(start_time)
        return np.array(start_time_list)
        
    def initial_state(self):
        state = {}
        for idx in range(0, self.num_person): 
            state[str(idx)] = self.initial_state_of_person(idx)           
        return state

    def initial_state_of_person(self, person_idx):
        x_data, y_data = self.pos_of_person(person_idx)        
        state = {}
        state["id"] = person_idx        
        state["px"], state["py"] = self.initial_pos(person_idx)
        state["vx"] = self.initial_speed[person_idx][0]
        state["vy"] = self.initial_speed[person_idx][1]
        state["gx"] = self.goal_schedule(person_idx)[0]["tx"]
        state["gy"] = self.goal_schedule(person_idx)[0]["ty"]
        state["distancing"] = 2
        start, _ = self.represent_time(person_idx)
        state["start_time"] = start
        state["visible"] = int(start == 0)
        state["tau"] = 0.5
        state["finished"] = 0
        state["phase"] = 0
        state["final_phase"] = self.final_phase(person_idx)
        return state
            
    def ground_truth(self):
        states = []
        for x, y in zip(self.pos_x, self.pos_y):
            state = []
            for idx in range(0, self.num_person):
                state.append([x[idx], y[idx]])
            states.append(state)        
        return np.array(states)

    # start_time 구하는 것임
    def represent_time(self, person_idx):
        represent_idx_list = []
        x_data, y_data = self.pos_of_person(person_idx)        
        for idx, data in enumerate(x_data):
            if ~np.isnan(data):
                represent_idx_list.append(idx)
        return represent_idx_list[0], represent_idx_list[-1]

    def initial_pos(self, person_idx):
        start_idx, finish_idx = self.represent_time(person_idx)
        x_data, y_data = self.pos_of_person(person_idx)
        return x_data[start_idx], y_data[start_idx]
    
    def final_pos(self, person_idx):
        start_idx, finish_idx = self.represent_time(person_idx)
        x_data, y_data = self.pos_of_person(person_idx)
        return x_data[finish_idx], y_data[finish_idx]

    def ped_info(self):
        info = {}
        for idx in range(0, self.num_person): 
            info[idx] = self.ped_info_of_person(idx)
        return info

    def ped_info_of_person(self, person_idx):
        start, _ = self.represent_time(person_idx)        
        return {            
                "goal_schedule": self.goal_schedule(person_idx),
                "num_phase": self.num_phase(person_idx),
                "final_phase": self.final_phase(person_idx),
                "start_time": start                
            }
        
    # TODO - 현재 case 국한된 것
    def num_phase(self, person_idx):
        if self.is_special_case(person_idx):
            return 2
        else:
            return 1
    
    # TODO - 현재 case 국한된 것
    def final_phase(self, person_idx):
        return self.num_phase(person_idx) - 1

    # TODO - 이 case에 한정된 함수
    def is_special_case(self, person_idx):
        px, py = self.initial_pos(person_idx)
        gx, gy = self.final_pos(person_idx)        
        if -1.5 < px < 0 and -1.2 < py < 0  and gy > 0:
            return True
        elif -1.5 < gx < 0 and -1.2 < gy < 0 and py > 0:
            return True
        elif -1.5 < px < 0 and 6 < py < 7.2 and gy > 7.5:            
            return True
        elif -1.5 < gx < 0 and 6 < gy < 7.2 and py > 7.5:
            return True
        else:
            return False

    # goal schedule 형태 뽑아내기 위함임
    def goal_schedule(self, person_idx):
        px, py = self.initial_pos(person_idx)
        gx, gy = self.final_pos(person_idx)
        if -1.5 < px < 0 and -1.2 < py < 0  and gy > 0:
            return {
                0:{
                    "tx": 0,
                    "ty": -0.6
                },
                1:{
                    "tx": gx,
                    "ty": gy
                }
            }
        elif -1.5 < gx < 0 and -1.2 < gy < 0 and py > 0:
            return {
                0:{
                    "tx": 0,
                    "ty": -0.6
                },
                1:{
                    "tx": gx,
                    "ty": gy
                }
            }
        elif -1.5 < px < 0 and 6 < py < 7.2 and gy > 7.2:
            return {
                0:{
                    "tx": 0,
                    "ty": 6.6
                },
                1:{
                    "tx": gx,
                    "ty": gy
                }
            }
        elif -1.5 < gx < 0 and 6 < gy < 7.2 and py > 7.2:
            return {
                0:{
                    "tx": 0,
                    "ty": 6.6
                },
                1:{
                    "tx": gx,
                    "ty": gy
                }
            }
        else:            
            return {
                0:{
                    "tx": gx,
                    "ty": gy
                }
            }

    def ground_truth_state(self):        
        for time_idx, step_width in enumerate(self.time_table):            
            states = []
            for ped_idx in range(0, self.num_person):
                state = []                 
                start, finish= self.represent_time(ped_idx)
                px = self.x_origin[time_idx][ped_idx+1]
                py = self.y_origin[time_idx][ped_idx+1]
                if np.isnan(px):
                    visible = 0
                else:
                    visible = 1
                
                if finish > time_idx:
                    finish_value = 0
                else:
                    finish_value = 1
                if visible == 1:
                    state.append(px)
                    state.append(py)
                else:
                    state.append(0)
                    state.append(0)
                for i in range(0, 6):
                    state.append(0)
                state.append(visible)
                state.append(start)
                state.append(ped_idx)
                state.append(finish_value)
                states.append(state)
        return np.array(states)

    def save(self, folder_path):
        save_path = os.path.join(folder_path, "video.vdt")
        with open(save_path, "wb") as fw:
            pickle.dump(self, fw)
        return

    def to_json(self, folder_path):  
        file_path = os.path.join(folder_path, "basic_info.json")
        state = self.initial_state()
        data = {
            "basic":{                
                "num_person": self.num_person,
                "gt_time": self.num_data
            },
            "ped_info": self.ped_info(),
            "initial_state": state
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return
        
    def trajectory_to_json(self, folder_path):
        file_path = os.path.join(folder_path, "trajectory.json")
        result_data = {}        
        for time_idx, step_width in enumerate(self.time_table):
            result_data[time_idx] = {}
            states = []
            for ped_idx in range(0, self.num_person):
                state = []                 
                start, finish= self.represent_time(ped_idx)
                px = self.x_origin[time_idx][ped_idx+1]
                py = self.y_origin[time_idx][ped_idx+1]
                if np.isnan(px):
                    visible = 0
                else:
                    visible = 1
                
                if finish > time_idx:
                    finish_value = 0
                else:
                    finish_value = 1
                if visible == 1:
                    state.append(px)
                    state.append(py)
                else:
                    state.append(0)
                    state.append(0)
                for i in range(0, 6):
                    state.append(0)
                state.append(visible)
                state.append(start)
                state.append(ped_idx)
                state.append(finish_value)
                states.append(state)                
            
            result_data[time_idx] ={
                "step_width": step_width,
                "states": states
            }

        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        return

    def ped_info_to_json(self, folder_path):
        file_path = os.path.join(folder_path, "ped_info.json")
        with open(file_path, 'w') as f:
            json.dump(self.ped_info(), f, indent=4)
        return