import time

import pybullet as pyb

from qdpgym.sim.abc import Hook


class ViewerBtHook(Hook):
    def __init__(self):
        self._init_vis = False
        self._sleep_on = True
        self._last_frame_time = time.time()

    def initialize_episode(self, robot, env):
        sim_env = env.sim_env
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_GUI, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_TINY_RENDERER, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
        sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)

    def before_step(self, robot, env):
        if not self._init_vis:
            env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, True)
            env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_GUI, True)
            self._init_vis = True
            # self._dbg_reset = sim_env.addUserDebugParameter('reset', 1, 0, 0)

    def after_step(self, robot, env):
        period = env.timestep * env.num_substeps
        time_spent = time.time() - self._last_frame_time
        if self._sleep_on:
            sleep_time = period - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time/2)
            time_spent = time.time() - self._last_frame_time
        self._last_frame_time = time.time()
        env.sim_env.configureDebugVisualizer(pyb.COV_ENABLE_SINGLE_STEP_RENDERING, True)
