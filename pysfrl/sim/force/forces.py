"""Calculate forces for individuals and groups"""
import re
from abc import ABC, abstractmethod

import numpy as np

from pysfrl.sim.force.potentials import PedPedPotential, PedSpacePotential
from pysfrl.sim.fieldofview import FieldOfView
from pysfrl.sim.utils import stateutils, logger
# from dis_ped.custom.utils import CustomUtils
from pysfrl.sim.utils.custom_utils import CustomUtils


def camel_to_snake(camel_case_string):
    """Convert CamelCase to snake_case"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_string).lower()


class Force(ABC):
    """Force base class"""

    def __init__(self):
        super().__init__()
        self.scene = None
        self.peds = None
        self.factor = 1.0        
        self.config = None

    def init(self, scene, config):    
        """Load config and scene"""
        self.config = config[self.name()]  
        self.factor = self.config["factor"]      
        self.scene = scene
        self.peds = self.scene.peds

    @abstractmethod
    def _get_force(self) -> np.ndarray:
        """Abstract class to get social forces
            return: an array of force vectors for each pedestrians
        """
        raise NotImplementedError

    @abstractmethod
    def _name(self):
        """Abstract class to get social forces
            return: an array of force vectors for each pedestrians
        """
        raise NotImplementedError

    def get_force(self, debug=False):
        force = self._get_force()
        if debug:
            logger.debug(f"{camel_to_snake(type(self).__name__)}:\n {repr(force)}")
        return force

    def name(self):
        name = self._name()
        return name


class GoalAttractiveForce(Force):
    """accelerate to desired velocity"""

    def _get_force(self):        
        F0 = (
            1.0 / self.peds.tau()
            * (
                np.expand_dims(self.peds.initial_speeds, -1) * self.peds.desired_directions()
                - self.peds.vel()
            )
        )        
        return F0 * self.factor
    
    def _name(self):
        return "goal_attractive_force"


class PedRepulsiveForce(Force):
    """Ped to ped repulsive force"""

    def _get_force(self):
        potential_func = PedPedPotential(            
            self.peds.step_width, v0=self.config["v0"], sigma=self.config["sigma"]
        )
        
       
        f_ab = -1.0 * potential_func.grad_r_ab(self.peds.state)
        fov = FieldOfView(phi=self.config["fov_phi"], out_of_view_factor=self.config["fov_factor"],)
        
        w = np.expand_dims(fov(self.peds.desired_directions(), -f_ab), -1)        
        F_ab = w * f_ab
        return np.sum(F_ab, axis=1) * self.factor * 10

    def _name(self):
        return "ped_repulsive_force"


class SpaceRepulsiveForce(Force):
    """obstacles to ped repulsive force"""

    def _get_force(self):
        if self.scene.get_obstacles() is None:
            F_aB = np.zeros((self.peds.size(), 0, 2))
        else:
            potential_func = PedSpacePotential(
                self.scene.get_obstacles(), u0=self.config["u0"], r=self.config["r"]
            )
            F_aB = -1.0 * potential_func.grad_r_aB(self.peds.state)
        return np.sum(F_aB, axis=1) * self.factor
        
    def _name(self):
        return "space_repulsive_force"


class GroupCoherenceForce(Force):
    """Group coherence force, paper version"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                vectors, norms = stateutils.normalize(force_vec)
                vectors[norms < threshold] = [0, 0]
                forces[group, :] += vectors
        return forces * self.factor

    def _name(self):
        return "group_coherence_force"

class GroupCoherenceForceAlt(Force):
    """ Alternative group coherence force as specified in pedsim_ros"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                norms = stateutils.speeds(force_vec)
                softened_factor = (np.tanh(norms - threshold) + 1) / 2
                forces[group, :] += (force_vec.T * softened_factor).T
        return forces * self.factor
    
    def _name(self):
        return "group_coherence_force_alt"


class GroupRepulsiveForce(Force):
    """Group repulsive force"""

    def _get_force(self):
        threshold = self.config["threshold"]
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                size = len(group)
                member_pos = self.peds.pos()[group, :]
                diff = stateutils.each_diff(member_pos)  # others - self
                _, norms = stateutils.normalize(diff)
                diff[norms > threshold, :] = 0
                # forces[group, :] += np.sum(diff, axis=0)
                forces[group, :] += np.sum(diff.reshape((size, -1, 2)), axis=1)

        return forces * self.factor

    def _name(self):
        return "group_repulsive_force"

class GroupGazeForce(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        vision_angle = self.config("fov_phi", 100.0)
        directions, _ = stateutils.desired_directions(self.peds.state)
        if self.peds.has_group():
            for group in self.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.peds.pos()[group, :]
                member_directions = directions[group, :]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.center_of_mass(member_pos[np.arange(group_size) != i, :2])
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, _ = stateutils.normalize(relative_com)
                # angle between walking direction and center of mass
                element_prod = np.array(
                    [np.dot(d, c) for d, c in zip(member_directions, com_directions)]
                )
                com_angles = np.degrees(np.arccos(element_prod))
                rotation = np.radians(
                    [a - vision_angle if a > vision_angle else 0.0 for a in com_angles]
                )
                force = -rotation.reshape(-1, 1) * member_directions
                forces[group, :] += force

        return forces * self.factor

    def _name(self):
        return "group_gaze_force"

class GroupGazeForceAlt(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        directions, dist = stateutils.desired_directions(self.peds.state)
        if self.peds.has_group():
            for group in self.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.peds.pos()[group, :]
                member_directions = directions[group, :]
                member_dist = dist[group]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.center_of_mass(member_pos[np.arange(group_size) != i, :2])
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, com_dist = stateutils.normalize(relative_com)
                # angle between walking direction and center of mass
                element_prod = np.array(
                    [np.dot(d, c) for d, c in zip(member_directions, com_directions)]
                )
                force = (
                    com_dist.reshape(-1, 1)
                    * element_prod.reshape(-1, 1)
                    / member_dist.reshape(-1, 1)
                    * member_directions
                )
                forces[group, :] += force

        return forces * self.factor

    def _name(self):
        return "group_gaze_force_alt"


class DesiredForce(Force):
    """Calculates the force between this agent and the next assigned waypoint.
    If the waypoint has been reached, the next waypoint in the list will be
    selected.
    :return: the calculated force
    """

    def _get_force(self):
        relexation_time = self.config["relaxation_time"]
        goal_threshold = self.config["goal_threshold"]
        pos = self.peds.pos()
        vel = self.peds.vel()
        goal = self.peds.goal()
        direction, dist = stateutils.normalize(goal - pos)
        force = np.zeros((self.peds.size(), 2))
        
        force[dist > goal_threshold] = (
            direction * self.peds.max_speeds.reshape((-1, 1)) - vel.reshape((-1, 2))
        )[dist > goal_threshold, :]
        force[dist <= goal_threshold] = -1.0 * vel[dist <= goal_threshold]        
        force /= relexation_time        
        return force * self.factor

    def _name(self):
        return "desired_force"

class SocialForce(Force):
    ## Moussaid et al. 2009
    """Calculates the social force between this agent and all the other agents
    belonging to the same scene.
    It iterates over all agents inside the scene, has therefore the complexity
    O(N^2). A better
    agent storing structure in Tscene would fix this. But for small (less than
    10000 agents) scenarios, this is just
    fine.
    :return:  nx2 ndarray the calculated force
    """

    def _get_force(self):
        lambda_importance = self.config["lambda_importance"]
        gamma = self.config["gamma"]
        n = self.config["n"]
        n_prime = self.config["n_prime"]

        pos_diff = stateutils.each_diff(self.peds.pos())  # n*(n-1)x2 other - self        
        diff_direction, diff_length = stateutils.normalize(pos_diff)
        vel_diff = -1.0 * stateutils.each_diff(self.peds.vel())  # n*(n-1)x2 self - other
        
        # compute interaction direction t_ij
        interaction_vec = lambda_importance * vel_diff + diff_direction
        
        interaction_direction, interaction_length = stateutils.normalize(interaction_vec)

        # compute angle theta (between interaction and position difference vector)
        theta = stateutils.vector_angles(interaction_direction) - stateutils.vector_angles(
            diff_direction
        )
        # compute model parameter B = gamma * ||D||
        B = gamma * interaction_length
        

        force_velocity_amount = np.exp(-1.0 * diff_length / B - np.square(n_prime * B * theta))
        force_angle_amount = -np.sign(theta) * np.exp(
            -1.0 * diff_length / B - np.square(n * B * theta)
        )
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * stateutils.left_normal(
            interaction_direction
        )

        force = force_velocity + force_angle  # n*(n-1) x 2
        force = np.sum(force.reshape((self.peds.size(), -1, 2)), axis=1)                
        return force * self.factor

    def _name(self):
        return "social_force"

class ObstacleForce(Force):
    """Calculates the force between this agent and the nearest obstacle in this
    scene.
    :return:  the calculated force
    """

    def _get_force(self):
        sigma = self.config["sigma"]
        threshold = self.config["threshold"] + self.peds.agent_radius
        force = np.zeros((self.peds.size(), 2))
        if len(self.scene.get_obstacles()) == 0:
            return force
        obstacles = np.vstack(self.scene.get_obstacles())        
        pos = self.peds.pos()

        for i, p in enumerate(pos):
            diff = p - obstacles
            directions, dist = stateutils.normalize(diff)
            dist = dist - self.peds.agent_radius
            if np.all(dist >= threshold):
                continue
            dist_mask = dist < threshold
            directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / sigma)
            force[i] = np.sum(directions[dist_mask], axis=0)
        
        # TODO- obstacle force가 너무 세면 목적지에 잘 가지 못하는 문제 발생
        return force * 1.5

    def _name(self):
        return "obstacle_force"

class Myforce(Force):
    def _get_force(self):
        # fov = CustomUtils.field_of_view(self.peds, self.scene.env)
        # desired_direction = self.peds.desired_directions()
        distance_mat = CustomUtils.get_distance_matrix(self.peds)        
        # desired_social_distance = self.peds.state[:, -1:]
        # desired_social_distance = self.peds.state[:, Index.distancing.index]        
        desired_social_distance = self.config["desired_distance"]        
        alpha, beta, lamb = self.config["alpha"], self.config["beta"], self.config["lambda"]
        in_desired_distance = distance_mat < desired_social_distance
        np.fill_diagonal(in_desired_distance, False)
        in_desired_distance = in_desired_distance.astype(int)
        
        angle_matrix = CustomUtils.get_angle_matrix(self.peds)        
        term_1 = 0.5 * (distance_mat - desired_social_distance)        
        term_2 = 0.5 + (1-0.5)*(1 + angle_matrix)/2 
        term_1 = np.exp((distance_mat - desired_social_distance) / beta)
        term_2 = lamb + (1-lamb)*(1 + angle_matrix)/2
        term = alpha * term_1 * term_2 * in_desired_distance        
        term = np.repeat(np.expand_dims(term, axis=2), 2, axis=2)
        e_ij = CustomUtils.ped_directions(self.peds)                
        return -np.sum(e_ij * term, axis=1)

    def _name(self):
        return "my_force"


class MyforceSecond(Force):
    def _get_force(self):
        # force_1
        
        alpha, beta, lamb = self.config["alpha"], self.config["beta"], self.config["lambda"]
        small_alpha = self.config["small_alpha"]
        distance_mat = CustomUtils.get_distance_matrix(self.peds) 
        angle_matrix = CustomUtils.get_angle_matrix(self.peds)
        angle_term = 0.5 + (1-0.5)*(1 + angle_matrix)/2                        
        if len(distance_mat) > 1:            
            desired_social_distance = self.config["desired_distance"]        
            Dij = desired_social_distance * 1         
            sort_idx = np.argsort(np.argsort(distance_mat))
            dijmin = distance_mat[sort_idx == 1]
            angmin = angle_term[sort_idx == 1]
            force_type_1 = np.expand_dims((Dij>dijmin)*1, axis=1)
            force_type_2 = np.expand_dims((Dij<=dijmin)*1, axis=1)

            term_1 = np.expand_dims(small_alpha * (Dij - dijmin)/Dij * angmin, axis=1) 
            vec_diff = CustomUtils.ped_directions(self.peds)           
            
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
        else:
            return np.array([[0,0]])

    def _name(self):
        return "my_force_2"
        

class MyforceThird(Force):
    def _get_force(self):               
        distance_mat = CustomUtils.get_distance_matrix(self.peds) 
        # if len(distance_mat) >2:
        #     print(self.peds.pos())
        #     nij = stateutils.vec_diff(self.peds.pos())               
        #     tij = np.flip(nij, axis=2)*np.array([-1, 1])          
        #     delta_v = stateutils.vec_diff(self.peds.vel())            
        #     friction_force = np.expand_dims(np.sum(delta_v * tij, axis=2), axis=2) * tij

        alpha, beta, lamb = self.config["alpha"], self.config["beta"], self.config["lambda"]                
        
        angle_matrix = CustomUtils.get_angle_matrix(self.peds)                
        term_1 = np.exp((0.6 - distance_mat) / beta)
        term_2 = lamb + (1-lamb)*(1 + angle_matrix)/2
        term = alpha * term_1 * term_2
        term = np.repeat(np.expand_dims(term, axis=2), 2, axis=2)
        e_ij = CustomUtils.ped_directions(self.peds)        
        return -np.sum(e_ij * term, axis=1)

    def _name(self):
        return "my_force_3"