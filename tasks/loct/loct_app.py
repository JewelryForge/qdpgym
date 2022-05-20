import sys
import time

import numpy as np
import torch

from qdpgym.sim.app import Application
from qdpgym.tasks.loct.loct import LocomotionBase
from qdpgym.tasks.loct.loct_utils import Actor
from qdpgym.utils import log

__all__ = ['LocomotionAppMj', 'LocomotionAppBt']


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


class LocomotionAppMj(Application):
    def __init__(self, model_path, gamepad='Xbox'):
        from qdpgym.sim.mjc.env import QuadrupedEnvMj
        from qdpgym.sim.mjc.hooks import ViewerMjHook
        from qdpgym.sim.mjc.quadruped import AliengoMj
        from qdpgym.sim.mjc.terrain import PlainMj, HillsMj

        robot = AliengoMj(500, 'actuator_net', noisy=False)
        # arena = PlainMj(10)
        arena = HillsMj(10, 0.1, (0.2, 20))
        task = LocomotionBase().add_hook(ViewerMjHook())
        env = QuadrupedEnvMj(robot, arena, task)
        network = Actor(78, 133, 12, (72, 64), (), (512, 256, 128)).to('cuda')
        log.info(f'Loading model {model_path}')
        model_info = torch.load(model_path)
        network.load_state_dict(model_info['actor_state_dict'])
        policy = lambda obs: network.policy(obs) * np.array((0.2, 0.2, 0.1) * 4)
        super().__init__(robot, env, task, policy)
        if gamepad and GamepadCommander.is_available():
            self.add_callback(GamepadCommander(gamepad).callback)


class LocomotionAppBt(Application):
    def __init__(self, model_path, gamepad='Xbox'):
        from qdpgym.sim.blt.env import QuadrupedEnvBt
        from qdpgym.sim.blt.quadruped import AliengoBt
        from qdpgym.sim.blt import terrain as t, hooks as h

        robot = AliengoBt(500, 'actuator_net', noisy=False)
        arena = t.TerrainBt()
        # arena = HillsBt.make(20, 0.1, (0.2, 10), random_state=np.random)
        task = LocomotionBase().add_hook(h.ExtraViewerBtHook())
        task.add_hook(h.RandomTerrainBtHook())
        task.add_hook(h.RandomPerturbBtHook())
        env = QuadrupedEnvBt(robot, arena, task)
        network = Actor(78, 133, 12, (72, 64), (), (512, 256, 128)).to('cuda')
        log.info(f'Loading model {model_path}')
        model_info = torch.load(model_path)
        network.load_state_dict(model_info['actor_state_dict'])
        policy = lambda obs: network.policy(obs) * np.array((0.2, 0.2, 0.1) * 4)
        super().__init__(robot, env, task, policy)
        if gamepad and GamepadCommander.is_available():
            self.add_callback(GamepadCommander(gamepad).callback)
