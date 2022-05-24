import dataclasses
import enum
import functools
from typing import Callable, Any, Optional, List, Iterable, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp

from qdpgym.sim.abc import Environment, TimeStep, ComposedObs


class _ActionType(enum.IntEnum):
    REGULAR = 0
    RESET = 1


@dataclasses.dataclass
class _Action:
    type: _ActionType
    action: Any = None


def _merge_dicts(dicts: List[dict],
                 datatype: Callable[[Sequence], Any]):
    merged = {}
    for k, v in dicts[0].items():
        if isinstance(v, dict):
            merged[k] = _merge_dicts([d[k] for d in dicts], datatype)
        else:
            merged[k] = datatype([d[k] for d in dicts])
    return merged


class ParallelWrapper(object):
    def __init__(self, make_env: Callable[[], Environment], num_envs: int, as_tensor=False):
        self._proc_conn, self._main_conn = [], []
        self._processes = []
        assert num_envs > 0, '`num_envs` must be a positive int'
        for _ in range(num_envs):
            conn1, conn2 = mp.Pipe(duplex=True)
            proc = mp.Process(target=self._run, args=(make_env, conn1))
            proc.start()
            self._proc_conn.append(conn1)
            self._main_conn.append(conn2)
            self._processes.append(proc)

        self.merge_results = functools.partial(
            self._merge_results,
            datatype=np.array if not as_tensor else lambda data: torch.Tensor(np.array(data))
        )

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
    def _merge_results(results: List[TimeStep],
                       datatype: Callable[[Sequence], Any]):
        status, observation, reward = [], [], []
        reward_info, info = [], []
        composed = isinstance(results[0].observation, ComposedObs)
        for res in results:
            status.append(res.status)
            observation.append(res.observation)
            reward.append(res.reward)
            if res.reward_info is not None:
                reward_info.append(res.reward_info)
            if res.info is not None:
                info.append(res.info)
        merged_obs = ComposedObs(
            [datatype(obs) for obs in zip(*observation)]
        ) if composed else datatype(observation)

        return TimeStep(
            datatype(status),
            merged_obs,
            datatype(reward),
            _merge_dicts(reward_info, datatype) if reward_info else {},
            _merge_dicts(info, datatype) if info else {}
        )
