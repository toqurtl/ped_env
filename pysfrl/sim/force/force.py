from pysfrl.sim.utils import stateutils
from pysfrl.sim.utils.custom_utils import CustomUtils
from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.parameters import DataIndex as Index
import numpy as np

# sim: Simulator

class Force(object):

    @classmethod
    def extra_force(cls, sim_cfg: SimulationConfig, state, obstacles):
        return cls.desired_force(sim_cfg, state) + cls.obstacle_force(sim_cfg, state, obstacles)


    @classmethod
    def desired_force(cls, sim_cfg: SimulationConfig, state, obstacles=None, model=None):
        cfg = sim_cfg.force_config["desired_force"]   
        relaxation_time = cfg["relaxation_time"]
        goal_threshold = cfg["goal_threshold"]
               
        max_speeds = CustomUtils.max_speeds(len(state), sim_cfg.max_speed)
        # visible_state, visible_idx, visible_max_speeds = sim.get_visible_info()
        pos, vel, goal = state[:, 0:2], state[:, 2:4], state[:, 4:6]       
        direction, dist = stateutils.normalize(goal - pos)
        force = np.zeros((len(state), 2))
        force[dist > goal_threshold] = (
            direction * max_speeds.reshape((-1, 1)) - vel.reshape((-1, 2))
        )[dist > goal_threshold, :]
        force[dist <= goal_threshold] = -1.0 * vel[dist <= goal_threshold]        
        force /= relaxation_time        
        return force * cfg["factor"]

    @classmethod
    def obstacle_force(cls, sim_cfg: SimulationConfig, state, obstacles, model=None):
        cfg = sim_cfg.force_config["obstacle_force"]
        
        sigma = cfg["sigma"]
        threshold = cfg["threshold"] + sim_cfg.agent_radius
        force = np.zeros((len(state), 2))
        if len(obstacles) == 0:
            return force

        obstacles = np.vstack(obstacles)        
        pos = state[:, 0:2]

        for i, p in enumerate(pos):
            diff = p - obstacles
            directions, dist = stateutils.normalize(diff)
            dist = dist - sim_cfg.agent_radius
            if np.all(dist >= threshold):
                continue
            dist_mask = dist < threshold
            directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / sigma)
            force[i] = np.sum(directions[dist_mask], axis=0)
        
        # TODO- obstacle force가 너무 세면 목적지에 잘 가지 못하는 문제 발생
        return force * cfg["factor"]

    @classmethod
    def basic_sfm(cls, sim_cfg: SimulationConfig, state, obstacles=None, model=None):        
        cfg = sim_cfg.force_config["repulsive_force"]["params"]
        distance_mat = CustomUtils.get_distance_matrix(state) 
        alpha, beta, lamb = cfg["alpha"], cfg["beta"], cfg["lambda"]        
        angle_matrix = CustomUtils.get_angle_matrix(state)
        term_1 = np.exp((0.6 - distance_mat) / beta)
        term_2 = lamb + (1-lamb)*(1 + angle_matrix)/2
        term = alpha * term_1 * term_2
        term = np.repeat(np.expand_dims(term, axis=2), 2, axis=2)
        e_ij = CustomUtils.ped_directions(state)
        return -np.sum(e_ij * term, axis=1)

    @classmethod
    def social_sfm_1(cls, sim_cfg: SimulationConfig, state, obstacles=None, model=None):        
        cfg = sim_cfg.force_config["repulsive_force"]["params"]        
        distance_mat = CustomUtils.get_distance_matrix(state)        
        desired_social_distance = cfg["desired_distance"]        
        alpha, beta, lamb = cfg["alpha"], cfg["beta"], cfg["lambda"]
        in_desired_distance = distance_mat < desired_social_distance
        np.fill_diagonal(in_desired_distance, False)
        in_desired_distance = in_desired_distance.astype(int)
        
        angle_matrix = CustomUtils.get_angle_matrix(state)        
        term_1 = 0.5 * (distance_mat - desired_social_distance)        
        term_2 = 0.5 + (1-0.5)*(1 + angle_matrix)/2 
        term_1 = np.exp((distance_mat - desired_social_distance) / beta)
        term_2 = lamb + (1-lamb)*(1 + angle_matrix)/2
        term = alpha * term_1 * term_2 * in_desired_distance        
        term = np.repeat(np.expand_dims(term, axis=2), 2, axis=2)
        e_ij = CustomUtils.ped_directions(state)                
        return -np.sum(e_ij * term, axis=1)

    @classmethod
    def social_sfm_2(cls, sim_cfg: SimulationConfig, state, obstacles=None, model=None):        
        cfg = sim_cfg.force_config["repulsive_force"]["params"]
        alpha, beta, lamb = cfg["alpha"], cfg["beta"], cfg["lambda"]
        small_alpha = cfg["small_alpha"]
        distance_mat = CustomUtils.get_distance_matrix(state) 
        angle_matrix = CustomUtils.get_angle_matrix(state)
        angle_term = 0.5 + (1-0.5)*(1 + angle_matrix)/2                        
        if len(distance_mat) > 1:            
            desired_social_distance = cfg["desired_distance"]        
            Dij = desired_social_distance * 1         
            sort_idx = np.argsort(np.argsort(distance_mat))
            dijmin = distance_mat[sort_idx == 1]
            angmin = angle_term[sort_idx == 1]
            force_type_1 = np.expand_dims((Dij>dijmin)*1, axis=1)
            force_type_2 = np.expand_dims((Dij<=dijmin)*1, axis=1)

            term_1 = np.expand_dims(small_alpha * (Dij - dijmin)/Dij * angmin, axis=1) 
            vec_diff = CustomUtils.ped_directions(state)           
            
            nij = vec_diff[sort_idx==1]

            term_1_force = term_1 * nij            
            term_2 = alpha*np.exp((0.5-distance_mat)/beta) * angle_term
            np.fill_diagonal(term_2, 0)            
            not_contain = np.expand_dims(term_2[sort_idx==1], axis=1) * nij            
            term_2 = np.expand_dims(term_2, axis=2) 
            term_3_force = np.sum(term_2 * vec_diff, axis=1)            
            term_2_force = term_3_force - not_contain
                
            force_type_1_value = term_1_force + term_2_force            
            force = force_type_1 * force_type_1_value + force_type_2 * term_3_force
            normalized, norm_factors = stateutils.normalize(force)
            # 너무 큰거 threshold                     
            force[norm_factors>10] = 5 * normalized[norm_factors >10]            
            
            return - force

    @classmethod
    def nn_repulsive(cls, sim_cfg: SimulationConfig, state, obstacles, model):
        force_list = []
        for s in state:
            idx = s[Index.id.index]
            action = cls.nn_repulsive_of_idx(sim_cfg, state, obstacles, idx, model)
            force_list.append(action)
        return np.array(force_list)
            
    @classmethod
    def nn_repulsive_of_idx(cls, sim_cfg: SimulationConfig, state, obstacles, idx, model):
        if len(state) < 2:
            return 0
        else:
            obs = cls.observation_of_idx(sim_cfg, state, obstacles, idx)
            action, states = model.predict(obs)
            return action

    @classmethod
    def observation_of_idx(cls, sim_cfg: SimulationConfig, visible_state, obstacles, idx):
        visible_idx = CustomUtils.find_visible_idx(visible_state, idx)
        extra_force = Force.extra_force(sim_cfg, visible_state, obstacles)[visible_idx]        
        neighbor_info = CustomUtils.neighbor_info(visible_state, visible_idx)        
        return np.concatenate((extra_force, neighbor_info))