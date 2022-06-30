from ..force.potentials import PedPedPotential
from . import stateutils
import numpy as np


class CustomUtils(object):
    @classmethod
    def get_distance_of_state(cls, state):
        return stateutils.desired_directions(state)[1]

    @classmethod
    def get_distance_matrix(cls, peds):
        # 위치벡터 리스트
        diff_list = stateutils.vec_diff(peds.state[:,:2])        
        # 거리계산
        person_list = []
        for person_diff in diff_list:
            detail_list = []
            for a in person_diff:
                detail_list.append(np.linalg.norm(a))
            person_list.append(detail_list)
        return np.array(person_list)

    @classmethod
    def get_angle_matrix(cls, peds):
        potential_func = PedPedPotential(peds.step_width, v0=2.1, sigma=0.3)
        f_ab = -1.0 * potential_func.grad_r_ab(peds.state)
        
        forces_direction = -f_ab
        desired_direction = peds.desired_directions()        
        dot_matrix = np.einsum("aj,abj->ab", desired_direction, forces_direction)
        norm_norm_matrix = np.linalg.norm(forces_direction, axis=-1)      
        np.fill_diagonal(norm_norm_matrix, 1)
        
        angle_matrix = dot_matrix / norm_norm_matrix
        # force가 0일 때 처리하기 위함(norm_norm_matrix가 0이 됨)
        angle_matrix[np.isnan(angle_matrix)] = -1        
        angle_matrix[np.isinf(angle_matrix)] = -1
        
        # Runtime Error, diagonal이 다 0이라서 그럼. 아래서 처리하므로 괜찮       
        np.fill_diagonal(angle_matrix, 0)
        return angle_matrix

    @classmethod
    def field_of_view(cls, peds, env):
        # angle
        angle_matrix = cls.get_angle_matrix(peds)    
        cosphi = np.cos(30 / 180.0 * np.pi)
        
        # cos is decrease function
        in_angle = (
            angle_matrix > cosphi        
        )
        np.fill_diagonal(in_angle, False)
        # distance
        distance_matrix = cls.get_distance_matrix(peds)
        in_distance = distance_matrix < 15   
        np.fill_diagonal(in_distance, False)
        
        fov = np.logical_and(in_angle, in_distance)
        
        # obstacle    
        # pos = peds.pos()
        # min, max = env.obstacles_min, env.obstacles_max    
        
        # length = np.linalg.norm(max-min, axis=1)
        # cal_min = np.repeat(np.expand_dims(min, axis=1), len(pos), axis=1)
        # cal_max = np.repeat(np.expand_dims(max, axis=1), len(pos), axis=1)
        # vec_1, vec_2 = np.swapaxes(cal_max-pos, 0, 1), np.swapaxes(cal_min-pos, 0, 1)
        # distance = np.abs(np.cross(vec_2, vec_1)/length)
        # print(np.cross(vec_2-vec_1)/length)
        return fov
        
    @classmethod
    def num_person_in_vision(cls, peds, env):
        fov = cls.field_of_view(peds, env)
        return np.sum(fov, axis=1)

    @classmethod 
    def ped_directions(cls, peds):
        # num_ped X num_ped array
        pos = peds.state[:,:2]        
        person_list = []
        for a in pos:
            diff_list = []
            for b in pos:
                d = b-a
                dis = np.linalg.norm(np.array(d), axis=-1)                
                if dis == 0:
                    diff_list.append([0,0])
                else:
                    diff_list.append(d/dis)
            person_list.append(diff_list)
        t = np.array(person_list)        
        return t