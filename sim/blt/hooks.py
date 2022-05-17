import collections
import math
import time

import pybullet as pyb

from qdpgym.sim.abc import Hook
from qdpgym.utils import Angle


class ViewerBtHook(Hook):
    def __init__(self, moving_cam=True):
        self._disabled = False
        self._pre_vis = False
        self._init_vis = False
        self._sleep_on = True
        self._moving_cam = moving_cam

        if self._moving_cam:
            self._robot_yaw_filter = collections.deque(maxlen=10)
            self._cam_state = 0

        self._last_frame_time = time.time()

    def initialize_episode(self, robot, env):
        if self._pre_vis:
            return
        self._pre_vis = True
        if not env.render_mode:
            self._disabled = True
        else:
            sim_env = env.sim_env
            sim_env.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, False)
            sim_env.configureDebugVisualizer(pyb.COV_ENABLE_GUI, False)
            sim_env.configureDebugVisualizer(pyb.COV_ENABLE_TINY_RENDERER, False)
            sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, False)
            sim_env.configureDebugVisualizer(pyb.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
            sim_env.configureDebugVisualizer(pyb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
            sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)

    def before_step(self, robot, env):
        if self._disabled:
            return
        if not self._init_vis:
            self._init_vis = True
            env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, True)
            env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_GUI, True)
            # self._dbg_reset = sim_env.addUserDebugParameter('reset', 1, 0, 0)

    def after_step(self, robot, env):
        if self._disabled:
            return
        sim_env = env.sim_env
        period = env.timestep * env.num_substeps
        time_spent = time.time() - self._last_frame_time
        if self._sleep_on:
            sleep_time = period - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._last_frame_time = time.time()

        if self._moving_cam:
            keys = pyb.getKeyboardEvents()
            switch_cam = ord('`')
            if switch_cam in keys and keys[switch_cam] & pyb.KEY_WAS_TRIGGERED:
                self._cam_state = (self._cam_state + 1) % 6

            x, y, _ = robot.get_base_pos()
            z = robot.STANCE_HEIGHT + env.arena.get_height(x, y)
            if self._cam_state == 0:
                yaw, pitch, dist = sim_env.getDebugVisualizerCamera()[8:11]
                sim_env.resetDebugVisualizerCamera(dist, yaw, pitch, (x, y, z))
            else:
                self._robot_yaw_filter.append(robot.get_base_rpy()[2] - math.pi / 2)
                # To avoid carsick :)
                mean = Angle.mean(self._robot_yaw_filter)
                if self._cam_state == 2:
                    mean = Angle.norm(mean + math.pi / 2)
                elif self._cam_state == 3:
                    mean = Angle.norm(mean + math.pi)
                elif self._cam_state == 4:
                    mean = Angle.norm(mean - math.pi / 2)
                sim_env.resetDebugVisualizerCamera(1.5, Angle.to_deg(mean), -20., (x, y, z))
            env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SINGLE_STEP_RENDERING, True)
