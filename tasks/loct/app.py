import sys
import time

import qdpgym
import qdpgym.sim as sim
from qdpgym.tasks.loct import LocomotionV0
from qdpgym.utils import log

__all__ = ['LocomotionApp']


class GamepadCommander(object):
    def __init__(self, gamepad_type='Xbox'):
        from qdpgym.thirdparty.gamepad import gamepad, controllers
        if not gamepad.available():
            log.warn('Please connect your gamepad...')
            while not gamepad.available():
                time.sleep(1.0)
        try:
            self.gamepad: gamepad.Gamepad = getattr(controllers, gamepad_type)()
            self.gamepad_type = gamepad_type
        except AttributeError:
            raise RuntimeError(f'`{gamepad_type}` is not supported,'
                               f'all {controllers.all_controllers}')
        self.gamepad.startBackgroundUpdates()
        log.info('Gamepad connected')

    @classmethod
    def is_available(cls):
        from qdpgym.thirdparty.gamepad import gamepad
        return gamepad.available()

    def callback(self, task, robot, env):
        if self.gamepad.isConnected():
            x_speed = -self.gamepad.axis('LAS -Y')
            y_speed = -self.gamepad.axis('LAS -X')
            steering = -self.gamepad.axis('RAS -X')
            task.cmd = (x_speed, y_speed, steering)
        else:
            sys.exit(1)

    def __del__(self):
        print('disconnect')
        self.gamepad.disconnect()


class LocomotionApp(sim.Application):
    def __init__(self, policy, gamepad='Xbox'):
        robot = sim.Aliengo(500, 'actuator_net', noisy=False)
        task = LocomotionV0()
        if qdpgym.sim_engine == qdpgym.Sim.BULLET:
            arena = sim.TerrainBase()
            task.add_hook(sim.ExtraViewerHook())
            task.add_hook(sim.RandomTerrainHook())
            task.add_hook(sim.RandomPerturbHook())
            # task.add_hook(hooks.VideoRecorderBtHook())
        else:
            raise NotImplementedError
        env = sim.QuadrupedEnv(robot, arena, task)
        super().__init__(robot, env, task, policy)
        if gamepad and GamepadCommander.is_available():
            self.add_callback(GamepadCommander(gamepad).callback)
