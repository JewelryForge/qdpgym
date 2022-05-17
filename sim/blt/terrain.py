import abc

import numpy as np

from qdpgym.sim.abc import Terrain


class TerrainBt(Terrain, metaclass=abc.ABCMeta):
    def __init__(self):
        self._id: int = -1

    @property
    def id(self):
        return self._id

    def spawn(self, sim_env, random_state):
        pass


class PlainBt(TerrainBt):
    def spawn(self, sim_env, random_state):
        self._id = sim_env.loadURDF("plane.urdf")
        sim_env.changeDynamics(self._id, -1, lateralFriction=1.0)

    def get_height(self, x, y):
        return 0.0

    def get_normal(self, x, y):
        return np.array((0, 0, 1))

    def out_of_range(self, x, y):
        return False
