from typing import List
import numpy as np


class ObstacleState(object):
    """State of the environment obstacles"""

    def __init__(self, obstacles, resolution=10):
        self.resolution = resolution
        self.obstacles = obstacles
        self.obstacles_line = obstacles
        obstacles_min, obstacles_max = [], []
        
        for obstacle in obstacles:
            obstacles_min.append([obstacle[0], obstacle[2]])
            obstacles_max.append([obstacle[1], obstacle[3]])
        self.obstacles_min = np.array(obstacles_min)
        self.obstacles_max = np.array(obstacles_max)
                        

    @property
    def obstacles(self) -> List[np.ndarray]:
        """obstacles is a list of np.ndarray"""
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacles):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        if obstacles is None:
            self._obstacles = []
        else:
            self._obstacles = []
            for startx, endx, starty, endy in obstacles:
                samples = int(np.linalg.norm((startx - endx, starty - endy)) * self.resolution)
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples))
                    )
                )
                self._obstacles.append(line)