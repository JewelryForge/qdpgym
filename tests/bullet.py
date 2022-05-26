import itertools
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pyb
import pybullet_data
import torch

from qdpgym.sim.blt.env import QuadrupedEnv
from qdpgym.sim.blt.hooks import ViewerHook
from qdpgym.sim.blt.quadruped import Aliengo
from qdpgym.sim.blt.terrain import Plain, Hills, Slopes, Steps, PlainHf
from qdpgym.sim.task import NullTask
from qdpgym.tasks.loct import LocomotionV0, LocomotionV0Raw
from qdpgym.utils import tf
from qdpgym.utils.parallel import ParallelWrapper


class BulletTestCase(unittest.TestCase):
    def test_robot(self):
        pyb.connect(pyb.GUI)
        pyb.setTimeStep(2e-3)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        Plain().spawn(pyb)
        rob = Aliengo(500, 'pd')
        rs = np.random.RandomState()
        rob.spawn(pyb, rs)
        pyb.setGravity(0, 0, -9.8)

        for _ in range(1000):
            # with MfTimer() as t:
            pyb.stepSimulation()
            rob.update_observation(rs)
            rob.apply_command(rob.STANCE_CONFIG)

        pyb.disconnect()

    def test_env(self):
        rob = Aliengo(500, 'pd')
        arena = Plain()
        task = NullTask()
        task.add_hook(ViewerHook())
        env = QuadrupedEnv(rob, arena, task)
        env.init_episode()
        for _ in range(1000):
            env.step(rob.STANCE_CONFIG)

    def test_replaceHeightfield(self):
        pyb.connect(pyb.GUI)
        pyb.setRealTimeSimulation(True)
        terrains = [PlainHf.default(), Hills.default(),
                    Steps.default(), Slopes.default()]
        current = None
        for i in range(5):
            for terrain in terrains:
                if current is None:
                    terrain.spawn(pyb)
                else:
                    terrain.replace(pyb, current)
                current = terrain
                time.sleep(0.5)
        current.remove(pyb)
        time.sleep(1.0)
        pyb.disconnect()

    def test_terrainApi(self):
        pyb.connect(pyb.GUI)
        pyb.setRealTimeSimulation(True)
        terrain = Hills.make(20, 0.1, (0.5, 10), random_state=np.random)
        # terrain = StepsBt.make(20, 0.1, 1.0, 0.5, random_state=np.random)
        # terrain = SlopesBt.make(20, 0.1, np.pi / 6, 0.5)
        terrain.spawn(pyb)

        sphere_shape = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE, radius=0.01, rgbaColor=(0., 0.8, 0., 0.6)
        )
        ray_hit_shape = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE, radius=0.01, rgbaColor=(0.8, 0., 0., 0.6)
        )
        cylinder_shape = pyb.createVisualShape(
            shapeType=pyb.GEOM_CYLINDER, radius=0.005, length=0.11,
            rgbaColor=(0., 0, 0.8, 0.6)
        )
        box_shape = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX, halfExtents=(0.03, 0.03, 0.03),
            rgbaColor=(0.8, 0., 0., 0.6)
        )

        points, vectors, ray_hits = [], [], []
        box_id = -1

        for i in range(10):
            peak = terrain.get_peak((-1, 1), (-1, 1))
            if i == 0:
                box_id = pyb.createMultiBody(
                    baseVisualShapeIndex=box_shape,
                    basePosition=peak
                )
            else:
                pyb.resetBasePositionAndOrientation(
                    box_id, peak, (0., 0., 0., 1.)
                )
            for idx, (x, y) in enumerate(
                    itertools.product(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
            ):
                h = terrain.get_height(x, y)
                vec_orn = tf.Quaternion.from_rotation(
                    tf.Rotation.from_zaxis(terrain.get_normal(x, y))
                )
                ray_pos = pyb.rayTest((x, y, 2), (x, y, -1))[0][3]
                if i == 0:
                    points.append(pyb.createMultiBody(
                        baseVisualShapeIndex=sphere_shape,
                        basePosition=(x, y, h)
                    ))
                    vectors.append(pyb.createMultiBody(
                        baseVisualShapeIndex=cylinder_shape,
                        basePosition=(x, y, h),
                        baseOrientation=vec_orn
                    ))
                    ray_hits.append(pyb.createMultiBody(
                        baseVisualShapeIndex=ray_hit_shape,
                        basePosition=ray_pos
                    ))
                else:
                    pyb.resetBasePositionAndOrientation(
                        points[idx], (x, y, h), (0., 0., 0., 1.)
                    )
                    pyb.resetBasePositionAndOrientation(
                        ray_hits[idx], ray_pos, (0., 0., 0., 1.)
                    )
                    pyb.resetBasePositionAndOrientation(
                        vectors[idx], (x, y, h), vec_orn
                    )
            time.sleep(3)
            new = Hills.make(20, 0.1, (np.random.random(), 10), random_state=np.random)
            new.replace(pyb, terrain)
            terrain = new

        pyb.disconnect()

    def test_parallel(self):
        def make_env():
            rob = Aliengo(500, 'pd')
            arena = Plain()
            task = LocomotionV0()
            return QuadrupedEnv(rob, arena, task)

        env = ParallelWrapper(make_env, 4)
        env.init_episode()
        for _ in range(10):
            obs = env.step(torch.zeros(4, 12))
            print(obs)

    def test_parallel_composed_obs(self):
        def make_env():
            rob = Aliengo(500, 'pd')
            arena = Plain()
            task = LocomotionV0Raw()
            return QuadrupedEnv(rob, arena, task)

        env = ParallelWrapper(make_env, 4)
        env.init_episode()
        for _ in range(10):
            step = env.step(torch.zeros(4, 12))
            obs, rew, done, info = step
            print(rew, done, info)

    def test_reward_reshape(self):
        x = np.arange(-2, 2, 0.01)
        plt.plot(x, np.tanh(x))
        plt.show()


if __name__ == '__main__':
    unittest.main()
