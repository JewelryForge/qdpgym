import abc
import dataclasses
import enum
from typing import Union, Any

import numpy as np

ARRAY_LIKE = Union[np.ndarray, list, tuple]
NUMERIC = Union[int, float]


class QuadrupedHandle(metaclass=abc.ABCMeta):
    @property
    def obs_history(self):
        raise NotImplementedError

    @property
    def cmd_history(self):
        raise NotImplementedError

    def get_base_pos(self) -> np.ndarray:
        raise NotImplementedError

    def get_base_orn(self) -> np.ndarray:
        raise NotImplementedError

    def get_base_rot(self) -> np.ndarray:
        raise NotImplementedError

    def get_base_rpy(self) -> np.ndarray:
        raise NotImplementedError

    def get_base_rpy_rate(self) -> np.ndarray:
        raise NotImplementedError

    def get_base_lin(self) -> np.ndarray:
        raise NotImplementedError

    def get_base_ang(self) -> np.ndarray:
        raise NotImplementedError

    def get_velocimeter(self) -> np.ndarray:
        raise NotImplementedError

    def get_gyro(self) -> np.ndarray:
        raise NotImplementedError

    def get_accelerometer(self) -> np.ndarray:
        raise NotImplementedError

    def get_state_history(self, latency):
        raise NotImplementedError

    def get_cmd_history(self, latency):
        raise NotImplementedError

    def get_torso_contact(self):
        raise NotImplementedError

    def get_leg_contacts(self):
        raise NotImplementedError

    def get_foot_contacts(self):
        raise NotImplementedError

    def get_contact_forces(self):
        raise NotImplementedError

    def get_force_sensor(self):
        raise NotImplementedError

    def get_joint_pos(self) -> np.ndarray:
        raise NotImplementedError

    def get_joint_vel(self) -> np.ndarray:
        raise NotImplementedError

    def get_last_command(self) -> np.ndarray:
        raise NotImplementedError

    def get_last_torque(self) -> np.ndarray:
        raise NotImplementedError


class Quadruped(QuadrupedHandle, metaclass=abc.ABCMeta):
    STANCE_CONFIG: tuple

    @property
    def noisy(self) -> QuadrupedHandle:
        return self

    def set_random_dynamics(self, flag: bool = True):
        raise NotImplementedError

    def set_latency(self, lower: float, upper=None):
        raise NotImplementedError

    @classmethod
    def inverse_kinematics(cls, leg: int, pos: ARRAY_LIKE):
        raise NotImplementedError

    @classmethod
    def forward_kinematics(cls, leg: int, angles: ARRAY_LIKE):
        raise NotImplementedError


class Terrain(metaclass=abc.ABCMeta):
    def get_height(self, x, y):
        raise NotImplementedError

    def get_normal(self, x, y):
        raise NotImplementedError

    def out_of_range(self, x, y) -> bool:
        raise NotImplementedError


class StepType(enum.IntEnum):
    INIT = 0
    REGULAR = 1
    FAIL = 2
    SUCCESS = 3


@dataclasses.dataclass
class TimeStep:
    status: StepType
    observation: Any
    reward: Any = None
    reward_info: Any = None
    info: Any = None


@dataclasses.dataclass
class Snapshot(object):
    position: np.ndarray = None
    orientation: np.ndarray = None
    rotation: np.ndarray = None
    rpy: np.ndarray = None
    linear_vel: np.ndarray = None
    angular_vel: np.ndarray = None
    joint_pos: np.ndarray = None
    joint_vel: np.ndarray = None
    foot_pos: np.ndarray = None
    velocimeter: np.ndarray = None
    gyro: np.ndarray = None
    accelerometer: np.ndarray = None
    torso_contact: int = None
    leg_contacts: np.ndarray = None
    contact_forces: np.ndarray = None
    force_sensor: np.ndarray = None
    rpy_rate: np.ndarray = None


@dataclasses.dataclass
class Command:
    command: np.ndarray = None
    torque: np.ndarray = None


class Environment(metaclass=abc.ABCMeta):
    @property
    def robot(self) -> QuadrupedHandle:
        raise NotImplementedError

    @property
    def arena(self) -> Terrain:
        raise NotImplementedError

    @property
    def action_history(self):
        raise NotImplementedError

    @property
    def sim_time(self):
        raise NotImplementedError

    @property
    def num_substeps(self):
        raise NotImplementedError

    @property
    def timestep(self):
        raise NotImplementedError

    def init_episode(self) -> TimeStep:
        raise NotImplementedError

    def step(self, action) -> TimeStep:
        raise NotImplementedError

    def get_perturbation(self, in_robot_frame=False):
        raise NotImplementedError

    def set_perturbation(self, value=None):
        raise NotImplementedError


class Hook(metaclass=abc.ABCMeta):
    def initialize_episode(self, robot, env):
        pass

    def before_step(self, robot, env):
        pass

    def before_substep(self, robot, env):
        pass

    def after_step(self, robot, env):
        pass

    def after_substep(self, robot, env):
        pass

    def on_success(self, robot, env):
        pass

    def on_termination(self, robot, env):
        pass


class Task(metaclass=abc.ABCMeta):
    def initialize_episode(self):
        pass

    def before_step(self, action):
        pass

    def before_substep(self):
        pass

    def after_step(self):
        pass

    def after_substep(self):
        pass

    def on_success(self):
        pass

    def on_fail(self):
        pass

    def register_env(self, robot: Quadruped, env: Environment, random_state):
        raise NotImplementedError

    def add_hook(self, hook: Hook):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_reward(self, detailed=True):
        raise NotImplementedError

    def is_succeeded(self):
        raise NotImplementedError

    def is_failed(self):
        raise NotImplementedError
