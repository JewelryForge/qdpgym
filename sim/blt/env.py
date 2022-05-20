import collections

import numpy as np
import pybullet as pyb
import pybullet_data as pyb_data
from pybullet_utils.bullet_client import BulletClient

from qdpgym.sim.abc import Task, Environment, ARRAY_LIKE, TimeStep, StepType
from qdpgym.sim.blt.quadruped import AliengoBt
from qdpgym.sim.blt.terrain import TerrainBt
from qdpgym.utils import PadWrapper


class QuadrupedEnvBt(Environment):
    def __init__(self, robot: AliengoBt, arena: TerrainBt, task: Task,
                 timestep: float = 2e-3, time_limit: float = None,
                 num_sub_steps=10, seed=None):
        self._robot = robot
        self._arena = arena
        self._task = task
        self._timestep = timestep
        self._time_limit = time_limit
        self._num_substeps = num_sub_steps
        self._render = False

        self._init = False
        self._num_sim_steps = 0
        self._random = np.random.RandomState(seed)

        self._sim_env = None if True else pyb
        self._reset_times, self._debug_reset_param = 0, -1
        self._task.register_env(self._robot, self, self._random)
        self._action_history = collections.deque(maxlen=10)
        self._perturbation = None

    def set_render(self, flag=True):
        self._render = flag

    def init_episode(self, init_yaw=0.):
        self._init = True
        self._num_sim_steps = 0
        self._action_history.clear()

        if self._sim_env is None:
            if self._render:
                fps = int(1 / (self._timestep * self._num_substeps))
                self._sim_env = BulletClient(pyb.GUI, options=f'--width=1024 --height=768 --mp4fps={fps}')
                self._debug_reset_param = self._sim_env.addUserDebugParameter('reset', 1, 0, 0)
            else:
                self._sim_env = BulletClient(pyb.DIRECT)
            self._sim_env.setAdditionalSearchPath(pyb_data.getDataPath())
            self._sim_env.setTimeStep(self._timestep)
            self._sim_env.setGravity(0, 0, -9.8)

        self._task.initialize_episode()
        if not self._arena.spawned:
            self._arena.spawn(self._sim_env)
        self._robot.add_to(self._arena, yaw=init_yaw)
        self._robot.spawn(self._sim_env, self._random)

        for i in range(50):
            self._robot.update_observation(None, minimal=True)
            self._robot.apply_command(self._robot.STANCE_CONFIG)
            self._sim_env.stepSimulation()

        self._action_history.append(np.array(self._robot.STANCE_CONFIG))
        self._robot.update_observation(self._random)
        return TimeStep(
            StepType.INIT,
            self._task.get_observation()
        )

    @property
    def robot(self):
        return self._robot

    @property
    def sim_env(self):
        return self._sim_env

    @property
    def render_mode(self):
        return self._render

    @property
    def arena(self):
        return self._arena

    @arena.setter
    def arena(self, value: TerrainBt):
        self._arena.hand_over_to(value)
        self._arena = value

    @property
    def action_history(self):
        return PadWrapper(self._action_history)

    @property
    def sim_time(self):
        return self._num_sim_steps * self._timestep

    @property
    def timestep(self):
        return self._timestep

    @property
    def num_substeps(self):
        return self._num_substeps

    def step(self, action: ARRAY_LIKE):
        if self._check_debug_reset_param():
            return self.init_episode()

        assert self._init, 'Call `init_episode` before `step`!'
        action = self._task.before_step(action)
        action = np.asarray(action)
        prev_action = self._action_history[-1]
        self._action_history.append(action)

        for i in range(self._num_substeps):
            weight = (i + 1) / self._num_substeps
            current_action = action * weight + prev_action * (1 - weight)
            self._robot.apply_command(current_action)
            self._task.before_substep()

            self._apply_perturbation()
            self._sim_env.stepSimulation()
            self._num_sim_steps += 1
            self._update_observation()
            self._robot.update_observation(self._random)

            self._task.after_substep()
        self._task.after_step()
        if self._task.is_failed():
            status = StepType.FAIL
            self._task.on_fail()
        elif ((self._time_limit is not None and self.sim_time > self._time_limit)
              or self._task.is_succeeded()):
            status = StepType.SUCCESS
            self._task.on_success()
        else:
            status = StepType.REGULAR
        reward, reward_info = self._task.get_reward(True)
        return TimeStep(
            status,
            self._task.get_observation(),
            reward,
            reward_info
        )

    def _update_observation(self):
        pass

    def get_perturbation(self, in_robot_frame=False):
        if self._perturbation is None:
            return None
        elif in_robot_frame:
            rotation = self._robot.get_base_rot()
            perturbation = np.concatenate(
                [rotation.T @ self._perturbation[i * 3:i * 3 + 3] for i in range(2)]
            )
        else:
            perturbation = self._perturbation
        return perturbation

    def set_perturbation(self, value=None):
        if value is None:
            self._perturbation = None
        else:
            self._perturbation = np.array(value)

    def _apply_perturbation(self):
        if self._perturbation is not None:
            self._applied_link_id = 0
            self._sim_env.applyExternalForce(objectUniqueId=self._robot.id,
                                             linkIndex=self._applied_link_id,
                                             forceObj=self._perturbation[:3],
                                             posObj=self._robot.get_base_pos(),
                                             flags=pyb.WORLD_FRAME)
            self._sim_env.applyExternalTorque(objectUniqueId=self._robot.id,
                                              linkIndex=self._applied_link_id,
                                              torqueObj=self._perturbation[3:],
                                              flags=pyb.WORLD_FRAME)

    def _check_debug_reset_param(self):
        if self._debug_reset_param != -1:
            reset_times = self._sim_env.readUserDebugParameter(self._debug_reset_param)
            if reset_times != self._reset_times:
                self._reset_times = reset_times
                return True
        return False
