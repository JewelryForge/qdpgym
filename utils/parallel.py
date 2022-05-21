import dataclasses
import enum
from typing import Callable, Any, Optional, List, Iterable

import numpy as np
import torch
import torch.multiprocessing as mp

from qdpgym.sim.abc import Environment, TimeStep


class _ActionType(enum.IntEnum):
    REGULAR = 0
    RESET = 1


@dataclasses.dataclass
class _Action:
    type: _ActionType
    action: Any = None


def _merge_dicts_as_numpy(dicts: List[dict]):
    merged = {}
    for k, v in dicts[0].items():
        if isinstance(v, dict):
            merged[k] = _merge_dicts_as_numpy([d[k] for d in dicts])
        else:
            merged[k] = np.array([d[k] for d in dicts])
    return merged


def _merge_dicts_as_tensor(dicts: List[dict]):
    merged = {}
    for k, v in dicts[0].items():
        if isinstance(v, dict):
            merged[k] = _merge_dicts_as_tensor([d[k] for d in dicts])
        else:
            merged[k] = torch.Tensor(np.array([d[k] for d in dicts]))
    return merged


class ParallelWrapper(object):
    def __init__(self, make_env: Callable[[], Environment], num_envs: int, as_tensor=False):
        self._proc_conn, self._main_conn = [], []
        self._processes = []
        for _ in range(num_envs):
            conn1, conn2 = mp.Pipe(duplex=True)
            proc = mp.Process(target=self._run, args=(make_env, conn1))
            proc.start()
            self._proc_conn.append(conn1)
            self._main_conn.append(conn2)
            self._processes.append(proc)
        if as_tensor:
            self.merge_results = self.merge_results_as_tensor
        else:
            self.merge_results = self.merge_results_as_numpy

    def __del__(self):
        for conn in self._main_conn:
            conn.send(None)
        for proc in self._processes:
            proc.join()

    def init_episode(self, ids: Iterable[int] = None):
        action_reset = _Action(_ActionType.RESET)
        if ids is None:
            for conn in self._main_conn:
                conn.send(action_reset)
            return self.merge_results([conn.recv() for conn in self._main_conn])
        else:
            for i in ids:
                self._main_conn[i].send(action_reset)
            return self.merge_results([self._main_conn[i].recv() for i in ids])

    def step(self, actions: torch.Tensor):
        for action, conn in zip(actions.detach().cpu().numpy(), self._main_conn):
            conn.send(_Action(_ActionType.REGULAR, action))
        results = [conn.recv() for conn in self._main_conn]
        return self.merge_results(results)

    @classmethod
    def _run(cls, make_env, conn):
        env: Environment = make_env()
        while True:
            action: Optional[_Action] = conn.recv()
            if action is None:
                return
            obs = None
            if action.type == _ActionType.REGULAR:
                obs = env.step(action.action)
            elif action.type == _ActionType.RESET:
                obs = env.init_episode()
            conn.send(obs)

    @staticmethod
    def merge_results_as_tensor(results: List[TimeStep]):
        status, observation, reward = [], [], []
        reward_info, info = [], []
        for res in results:
            status.append(res.status)
            observation.append(res.observation)
            reward.append(res.reward)
            if res.reward_info is not None:
                reward_info.append(res.reward_info)
            if res.info is not None:
                info.append(res.info)
        return TimeStep(
            torch.Tensor(status),
            torch.Tensor(np.array(observation)),
            torch.Tensor(reward),
            _merge_dicts_as_tensor(reward_info) if reward_info else {},
            _merge_dicts_as_tensor(info) if info else {}
        )

    @staticmethod
    def merge_results_as_numpy(results: List[TimeStep]):
        status, observation, reward = [], [], []
        reward_info, info = [], []
        for res in results:
            status.append(res.status)
            observation.append(res.observation)
            reward.append(res.reward)
            if res.reward_info is not None:
                reward_info.append(res.reward_info)
            if res.info is not None:
                info.append(res.info)
        return TimeStep(
            np.array(status),
            np.array(observation),
            np.array(reward),
            _merge_dicts_as_numpy(reward_info) if reward_info else {},
            _merge_dicts_as_numpy(info) if info else {}
        )
