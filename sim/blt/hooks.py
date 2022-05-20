import collections
import copy
import math
import time

import numpy as np
import pybullet as pyb

from qdpgym.sim.abc import Hook
from qdpgym.sim.blt.terrain import HillsBt, SlopesBt, StepsBt, PlainHfBt
from qdpgym.utils import Angle, tf


class ViewerBtHook(Hook):
    def __init__(self):
        self._pre_vis = False
        self._init_vis = False
        self._sleep_on = True

        self._robot_yaw_buffer = collections.deque(maxlen=10)
        self._cam_state = 0

        self._last_frame_time = time.time()

    def initialize(self, robot, env, random_state):
        env.set_render()

    def init_episode(self, robot, env, random_state):
        self._robot_yaw_buffer.clear()
        if self._pre_vis:
            return
        self._pre_vis = True
        sim_env = env.sim_env
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_GUI, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_TINY_RENDERER, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)

    def before_step(self, robot, env, random_state):
        if not self._init_vis:
            self._init_vis = True
            env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, True)
            env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_GUI, True)

    def after_step(self, robot, env, random_state):
        sim_env = env.sim_env
        period = env.timestep * env.num_substeps
        time_spent = time.time() - self._last_frame_time
        if self._sleep_on:
            sleep_time = period - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._last_frame_time = time.time()
        kbd_events = pyb.getKeyboardEvents()

        switch_cam_state = self.is_triggered(ord('`'), kbd_events)
        if switch_cam_state:
            self._cam_state = (self._cam_state + 1) % 5

        x, y, _ = robot.get_base_pos()
        z = robot.STANCE_HEIGHT + env.arena.get_height(x, y)
        if self._cam_state == 0:
            if switch_cam_state:
                yaw = Angle.to_deg(Angle.mean(self._robot_yaw_buffer))
                sim_env.resetDebugVisualizerCamera(1.5, yaw, -30., (x, y, z))
            else:
                yaw, pitch, dist = sim_env.getDebugVisualizerCamera()[8:11]
                sim_env.resetDebugVisualizerCamera(dist, yaw, pitch, (x, y, z))
        else:
            self._robot_yaw_buffer.append(robot.get_base_rpy()[2] - math.pi / 2)
            # To avoid carsick :)
            yaw = Angle.mean(self._robot_yaw_buffer)
            degree = -30.
            if self._cam_state == 2:  # around robot
                yaw = Angle.norm(yaw + math.pi / 2)
                degree = 0.
            elif self._cam_state == 3:
                yaw = Angle.norm(yaw + math.pi)
            elif self._cam_state == 4:
                yaw = Angle.norm(yaw - math.pi / 2)
                degree = 0.
            sim_env.resetDebugVisualizerCamera(1.5, Angle.to_deg(yaw), degree, (x, y, z))
        env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SINGLE_STEP_RENDERING, True)

        KEY_SPACE = ord(' ')
        if self.is_triggered(KEY_SPACE, kbd_events):
            while True:  # PAUSE
                env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SINGLE_STEP_RENDERING, True)
                time.sleep(0.01)
                if self.is_triggered(KEY_SPACE, pyb.getKeyboardEvents()):
                    self._cam_state = 0
                    break

    @staticmethod
    def is_triggered(key, keyboard_events):
        return key in keyboard_events and keyboard_events[key] & pyb.KEY_WAS_TRIGGERED


class _TorqueVisualizerHelper(object):
    def __init__(self):
        self._axis_x = None
        self._axis_y = None
        self._marker_start = None
        self._marker_phase = 0

    def update(self, torque):
        if (torque != 0.).any():
            self._axis_x, self._axis_y, _ = tf.Rotation.from_zaxis(tf.vunit(torque)).T
            magnitude = tf.vnorm(torque)
            if self._marker_start is None:
                self._marker_start = self._axis_x * magnitude / 20
            self._marker_phase += math.pi / 36
            marker_end = (self._axis_x * math.cos(self._marker_phase) +
                          self._axis_y * math.sin(self._marker_phase)) * magnitude / 20

            marker_info = dict(lineFromXYZ=copy.deepcopy(self._marker_start),
                               lineToXYZ=marker_end)
            self._marker_start = marker_end
            return marker_info


class ExtraViewerBtHook(ViewerBtHook):
    def __init__(self, perturb=True):
        super().__init__()
        self._show_perturb = perturb
        self._last_perturb = None

        self._force_marker = -1
        self._torque_vis = _TorqueVisualizerHelper()

    def after_step(self, robot, env, random_state):
        super().after_step(robot, env, random_state)
        sim_env = env.sim_env
        if self._show_perturb:
            perturb = env.get_perturbation(in_robot_frame=True)
            if perturb is not None and (self._last_perturb != perturb).any():
                self._force_marker = sim_env.addUserDebugLine(
                    lineFromXYZ=(0., 0., 0.),
                    lineToXYZ=perturb[:3] / 50,
                    lineColorRGB=(1., 0., 0.),
                    lineWidth=3, lifeTime=1,
                    parentObjectUniqueId=robot.id,
                    replaceItemUniqueId=self._force_marker
                )
                sim_env.addUserDebugLine(
                    **self._torque_vis.update(perturb[3:]),
                    lineColorRGB=(0., 0., 1.),
                    lineWidth=5, lifeTime=0.1,
                    parentObjectUniqueId=robot.id)

                self._last_perturb = perturb


class RandomTerrainBtHook(Hook):
    def __init__(self):
        self.max_roughness = 0.2
        self.max_slope = 10 / 180 * math.pi
        self.max_step_height = 0.1

    def generate_terrain(self, random_state):
        """
        If no terrain has been spawned, create and spawn it.
        Otherwise, update its height field.
        """
        size, resolution = 30, 0.1
        terrain_type = random_state.randint(4)
        difficulty = random_state.random()
        terrain = None
        if terrain_type == 0:
            roughness = self.max_roughness * difficulty
            terrain = HillsBt.make(size, resolution, (roughness, 20),
                                   random_state=random_state)
        elif terrain_type == 1:
            slope = self.max_slope * difficulty
            axis = random_state.choice(('x', 'y'))
            terrain = SlopesBt.make(size, resolution, slope, 3., axis)
        elif terrain_type == 2:
            step_height = self.max_step_height * difficulty
            terrain = StepsBt.make(size, resolution, 1., step_height, random_state)
        elif terrain_type == 3:
            terrain = PlainHfBt.make(size, resolution)
        return terrain

    def init_episode(self, robot, env, random_state):
        env.arena = self.generate_terrain(random_state)


class RandomPerturbBtHook(Hook):
    def __init__(self):
        self.force_magnitude = np.array((20., 20.))
        self.torque_magnitude = np.array((2.5, 5., 5.))
        self.interval_range = (0.5, 2.0)
        self.update_interval = 0
        self.last_update = 0

    def init_episode(self, robot, env, random_state):
        env.set_perturbation(np.concatenate(self.get_random_perturb(random_state)))
        self.update_interval = random_state.uniform(*self.interval_range)

    def get_random_perturb(self, random_state):
        horizontal_force = random_state.uniform(0, self.force_magnitude[0])
        vertical_force = random_state.uniform(0, self.force_magnitude[1])
        yaw = random_state.uniform(0, 2 * math.pi)
        external_force = np.array((
            horizontal_force * np.cos(yaw),
            horizontal_force * np.sin(yaw),
            vertical_force * random_state.choice((-1, 1))
        ))

        external_torque = random_state.uniform(-self.torque_magnitude, self.torque_magnitude)
        return external_force, external_torque

    def before_substep(self, robot, env, random_state):
        if env.sim_time >= self.last_update + self.update_interval:
            env.set_perturbation(np.concatenate(self.get_random_perturb(random_state)))
            self.update_interval = random_state.uniform(*self.interval_range)
            self.last_update = env.sim_time
