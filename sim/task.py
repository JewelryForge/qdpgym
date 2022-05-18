import collections
from typing import Optional, Callable

from .abc import Task, Hook, Quadruped, Environment


class NullTask(Task):
    """The Simplest Task Structure"""

    def __init__(self):
        self._robot: Optional[Quadruped] = None
        self._env: Optional[Environment] = None
        self._random = None
        self._hooks = []
        self._hook_names = set()
        self._options = []

    def register_env(self, robot, env, random_state):
        self._robot = robot
        self._env = env
        self._random = random_state
        for hook in self._hooks:
            hook.initialize(self._robot, self._env)

    def add_hook(self, hook: Hook, name=None):
        self._hooks.append(hook)
        if name is None:
            name = hook.__class__.__name__
        if name in self._hook_names:
            raise ValueError(f'Duplicated Hook `{name}`')
        self._hook_names.add(name)
        return self

    def get_observation(self):
        return None

    def get_reward(self, detailed=True):
        if detailed:
            return 0., {}
        else:
            return 0.

    def is_succeeded(self):
        return False

    def is_failed(self):
        return False

    def initialize_episode(self):
        for hook in self._hooks:
            hook.initialize_episode(self._robot, self._env)

    def before_step(self, action):
        for hook in self._hooks:
            hook.before_step(self._robot, self._env)
        return action

    def before_substep(self):
        for hook in self._hooks:
            hook.before_substep(self._robot, self._env)

    def after_step(self):
        for hook in self._hooks:
            hook.after_step(self._robot, self._env)

    def after_substep(self):
        for hook in self._hooks:
            hook.after_substep(self._robot, self._env)

    def on_success(self):
        for hook in self._hooks:
            hook.on_success(self._robot, self._env)

    def on_fail(self):
        for hook in self._hooks:
            hook.on_fail()


class RewardRegistry(object):
    def __init__(self, robot, env, task):
        self._robot = robot
        self._env = env
        self._task = task

        self._rewards_set = set()
        self._rewards_weights = []
        self._weight_sum = 0.0
        self._coefficient = 1.0
        self._reward_details = {}

    def add_reward(self, name: str, obj: Callable, weight: float):
        if name in self._rewards_set:
            raise RuntimeError(f'Duplicated Reward Type {name}')
        self._rewards_set.add(name)
        self._weight_sum += weight
        self._rewards_weights.append((obj, weight))

    def set_coeff(self, coeff):
        self._coefficient = coeff

    def report(self):
        from qdpgym.utils import colored_str
        print(colored_str(f'Got {len(self._rewards_weights)} types of rewards:\n', 'white'),
              f"{'Reward Type':<28}Weight * {self._coefficient:.3f}", sep='')
        for reward, weight in self._rewards_weights:
            reward_name: str = reward.__class__.__name__
            length = len(reward_name)
            if reward_name.endswith('Reward'):
                reward_name = colored_str(reward_name, 'green')
            elif reward_name.endswith('Penalty'):
                reward_name = colored_str(reward_name, 'magenta')
            print(f'{reward_name}{" " * (28 - length)}{weight:.3f}')
        print()

    def calc_reward(self, detailed=True):
        self._reward_details.clear()
        reward_value = 0.0
        for reward, weight in self._rewards_weights:
            rew = reward(self._robot, self._env)
            self._reward_details[reward.__class__.__name__] = rew
            reward_value += rew * weight
        reward_value /= self._coefficient
        if detailed:
            return reward_value, self._reward_details
        else:
            return reward_value


class BasicTask(NullTask):
    def __init__(self, reward_coeff=1.0, substep_reward_on=True):
        self._reward_registry: Optional[RewardRegistry] = None
        self._reward_coeff = reward_coeff
        self._substep_reward_on = substep_reward_on
        self._reward = 0.
        self._reward_details = collections.defaultdict(float)
        self._substep_cnt = 0

        self._terminate = False
        super().__init__()

    def add_reward(self, name: str, weight: float):
        reward_class = eval(name)
        self._reward_registry.add_reward(name, reward_class(), weight)

    def register_env(self, robot, env, random_state):
        super().register_env(robot, env, random_state)
        self._reward_registry = RewardRegistry(robot, env, self)
        self._reward_registry.set_coeff(self._reward_coeff)

    def after_substep(self):
        if self._substep_reward_on:
            reward, reward_details = self._reward_registry.calc_reward(detailed=True)
            self._reward += reward
            for k, v in reward_details.items():
                self._reward_details[k] += v
            self._substep_cnt += 1
        super().after_substep()

    def after_step(self):
        if self._substep_reward_on:
            self._reward /= self._substep_cnt
            for k in self._reward_details:
                self._reward_details[k] /= self._substep_cnt
            self._substep_cnt = 0
        else:
            self._reward, self._reward_details = self._reward_registry.calc_reward(detailed=True)
        super().after_step()

    def get_reward(self, detailed=True):
        if detailed:
            return self._reward, self._reward_details
        else:
            return self._reward

    def set_terminate(self, flag=True):
        self._terminate = flag

    def is_failed(self):
        return self._terminate
