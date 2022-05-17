from typing import Callable

import torch

from qdpgym.sim.abc import Quadruped, Environment, Task, StepType
from qdpgym.utils import MfTimer


class Application(object):
    def __init__(self, robot, env, task, policy):
        self.robot: Quadruped = robot
        self.env: Environment = env
        self.task: Task = task
        self.policy = policy
        self.callbacks = []

    def add_callback(self, obj: Callable):
        self.callbacks.append(obj)

    def launch(self, allow_reset=True):
        with torch.inference_mode():
            obs = self.env.init_episode()

            for _ in range(20000):
                actions = self.policy(obs.observation)
                for callback in self.callbacks:
                    callback(self.task, self.robot, self.env)
                obs = self.env.step(actions)
                if obs.status == StepType.FAIL and allow_reset:
                    self.env.init_episode()
