import unittest

import numpy as np
import pybullet as pyb
import pybullet_data

from qdpgym.sim.blt.env import QuadrupedEnvBt
from qdpgym.sim.blt.quadruped import AliengoBt
from qdpgym.sim.blt.terrain import PlainBt
from qdpgym.sim.task import NullTask


class BulletTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_robot(self):
        pyb.connect(pyb.GUI)
        pyb.setTimeStep(2e-3)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        PlainBt().spawn(pyb)
        rob = AliengoBt(500, 'pd')
        rs = np.random.RandomState()
        rob.spawn(pyb, rs)
        pyb.setGravity(0, 0, -9.8)

        for _ in range(100000):
            # with MfTimer() as t:
            pyb.stepSimulation()
            rob.update_observation(rs)
            tq = rob.apply_command(rob.STANCE_CONFIG)

    def test_env(self):
        rob = AliengoBt(500, 'pd')
        arena = PlainBt()
        task = NullTask()
        env = QuadrupedEnvBt(rob, arena, task)
        env.init_episode(True)
        for _ in range(100000):
            env.step(rob.STANCE_CONFIG)


if __name__ == '__main__':
    unittest.main()
