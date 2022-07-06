from pysfrl.sim.force.potentials import PedPedPotential
from pysfrl.sim.utils import stateutils
import numpy as np


class CustomUtils(object):
    @classmethod
    def get_distance_of_state(cls, state):
        return stateutils.desired_directions(state)[1]

    @classmethod
    def get_distance_matrix(cls, state):
        # 위치벡터 리스트
        diff_list = stateutils.vec_diff(state[:,:2])        
        # 거리계산
        person_list = []
        for person_diff in diff_list:
            detail_list = []
            for a in person_diff:
                detail_list.append(np.linalg.norm(a))
            person_list.append(detail_list)
        return np.array(person_list)

    @classmethod
    def get_angle_matrix(cls, state, step_width=0.067):
        potential_func = PedPedPotential(step_width, v0=2.1, sigma=0.3)
        f_ab = -1.0 * potential_func.grad_r_ab(state)
        
        forces_direction = -f_ab
        desired_direction = stateutils.desired_directions(state)[0]       
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
    def ped_directions(cls, state):
        # num_ped X num_ped array
        pos = state[:,:2]        
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

    @classmethod
    def capped_velocity(cls, desired_velocity, max_velocity):        
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    @classmethod
    def max_speeds(cls, num_peds, max_speed):
        return np.ones((num_peds)) * max_speed