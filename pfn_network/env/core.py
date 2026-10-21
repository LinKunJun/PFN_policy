from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Protocol

import numpy as np

StateKey = tuple[int, tuple[int, ...]]


class Dynamics(str, Enum):
    # Python 3.10 equivalent of StrEnum, including str() and repr().
    __str__ = str.__str__
    NONE = "NONE"
    FC = "FC"
    CD = "CD"


def exposures(adjacency: np.ndarray, allocations: np.ndarray):
    """Rows are allocations; ties are NOT majority exposure."""
    degree = adjacency.sum(1)
    counts = np.asarray(allocations, dtype=np.int32) @ adjacency.T
    exact = np.divide(counts, degree, out=np.zeros_like(counts, dtype=float),
                      where=degree > 0)
    majority = counts > (degree // 2)
    return exact, majority


@dataclass(frozen=True)
class PublicTask:
    adjacency: np.ndarray
    features: np.ndarray
    catalog: np.ndarray
    budget: int
    horizon: int
    dynamics: Dynamics
    fatigue_cost: float = 1.0
    budget_mode: str = "at_most"

    def __post_init__(self):
        n = len(self.features)
        a, c = self.adjacency, self.catalog
        if a.shape != (n, n) or not np.array_equal(a, a.T) or np.any(a.diagonal()):
            raise ValueError("Expected a simple undirected graph.")
        if not np.isin(a, [0, 1]).all() or c.ndim != 2 or c.shape[1] != n:
            raise ValueError("Invalid graph or catalog shape.")
        if not np.isin(c, [0, 1]).all() or len(np.unique(c, axis=0)) != len(c):
            raise ValueError("Catalog must contain distinct binary allocations.")
        if self.budget_mode not in {"at_most", "exact"}:
            raise ValueError("Unknown budget mode.")
        counts = c.sum(1)
        feasible = counts <= self.budget if self.budget_mode == "at_most" else counts == self.budget
        if not len(c) or not feasible.all() or not 0 <= self.budget <= n or self.horizon < 1:
            raise ValueError("Invalid budget, horizon, or candidate allocation.")
        if self.dynamics == Dynamics.CD and (self.budget_mode != "at_most" or not np.any(counts == 0)):
            raise ValueError("Cooldown requires an at-most budget and the waiting action.")
        for array in (a, self.features, c):
            array.setflags(write=False)

        # ------------------ 方案 A：针对 Zen 4 SIMD 预计算 ------------------
        if n <= 64:
            # 预计算每个动作的 64 位无符号整型掩码 (C-contiguous，命中 L1 Cache)
            powers = (1 << np.arange(n, dtype=np.uint64))
            catalog_bits = (c.astype(np.uint64) * powers).sum(axis=1, dtype=np.uint64)
            catalog_bits.setflags(write=False)
            object.__setattr__(self, "_catalog_bits", catalog_bits)
        else:
            object.__setattr__(self, "_catalog_bits", None)

        # 预计算激活节点下标与动作元组，加速 transition
        object.__setattr__(self, "_action_active_nodes", tuple(
            tuple(int(idx) for idx in np.flatnonzero(c[i])) for i in range(len(c))
        ))
        object.__setattr__(self, "_catalog_tuples", tuple(
            tuple(int(x) for x in row) for row in c
        ))
        # -------------------------------------------------------------------

    @property
    def n(self):
        return len(self.features)

    @property
    def m(self):
        return len(self.catalog)

    def fingerprint(self):
        h = sha256()
        for a in (self.adjacency, self.features, self.catalog):
            h.update(str((a.shape, a.dtype)).encode())
            h.update(a.tobytes())
        h.update(repr((self.budget, self.horizon, self.dynamics,
                       self.fatigue_cost, self.budget_mode)).encode())
        return h.hexdigest()


def initial_key(task: PublicTask) -> StateKey:
    return task.horizon, (0,) * task.n


def valid_mask(task: PublicTask, key: StateKey, out: np.ndarray | None = None) -> np.ndarray:
    remaining, memory = key
    if remaining == 0:
        if out is not None:
            out.fill(False)
            return out
        return np.zeros(task.m, dtype=bool)

    if task.dynamics == Dynamics.CD:
        # 快速短路：若没有任何节点处于冷却中，全动作直接合法
        if not any(memory):
            if out is not None:
                out.fill(True)
                return out
            return np.ones(task.m, dtype=bool)

        catalog_bits = getattr(task, "_catalog_bits", None)
        if catalog_bits is not None:
            # 将 memory 中的冷却状态转化为一个 64 位整型掩码
            busy_mask = 0
            for i, m in enumerate(memory):
                if m:
                    busy_mask |= (1 << i)

            # AMD Zen 4 下硬件 SIMD 向量化位运算：单周期高通量吞吐
            if out is not None:
                np.equal(catalog_bits & np.uint64(busy_mask), 0, out=out)
                return out
            return (catalog_bits & np.uint64(busy_mask)) == 0

        # 回退通用路径
        res = ~np.any(task.catalog & (np.asarray(memory) > 0), axis=1)
        if out is not None:
            np.copyto(out, res)
            return out
        return res

    if out is not None:
        out.fill(True)
        return out
    return np.ones(task.m, dtype=bool)


def transition(task: PublicTask, key: StateKey, action: int) -> StateKey:
    remaining, memory = key
    action = int(action)

    if task.dynamics == Dynamics.FC:
        catalog_tuples = getattr(task, "_catalog_tuples", None)
        new_memory = catalog_tuples[action] if catalog_tuples is not None else tuple(int(x) for x in task.catalog[action])
    elif task.dynamics == Dynamics.CD:
        active_nodes = getattr(task, "_action_active_nodes", None)
        if active_nodes is not None:
            # 状态步进规则：未选中的 2->1, 1->0, 0->0；选中的置为 2
            mem = [1 if m == 2 else 0 for m in memory]
            for node in active_nodes[action]:
                mem[node] = 2
            new_memory = tuple(mem)
        else:
            z = task.catalog[action]
            new_memory = tuple(2 if z_i else (m_i - 1 if m_i > 0 else 0) for z_i, m_i in zip(z, memory))
    else:
        new_memory = (0,) * task.n

    return remaining - 1, new_memory


def fatigue_penalty(task: PublicTask, key: StateKey):
    return task.fatigue_cost * sum(key[1]) / task.n if task.dynamics == Dynamics.FC else 0.0


class RewardSource(Protocol):
    def prepare(self, task: PublicTask, exact: np.ndarray) -> np.ndarray: ...
    def sample_noise(self, rng: np.random.Generator, n: int) -> float: ...


class NetworkEnv:
    """Only observations and executed scalar feedback cross the learner boundary.

    Reward/exposure lookup is O(1) in M. Constructing the public mask remains
    O(MN); constructing node state remains O(N). step itself is not O(1).
    """

    def __init__(self, task: PublicTask, reward_source: RewardSource,
                 seed: int, prediction: np.ndarray | None = None):
        self.task = task
        self.__source = reward_source
        self.__rng = np.random.default_rng(seed)
        self.prediction = prediction
        self.invalid_attempts = 0
        self._key = initial_key(task)
        self._observation_buffer = None
        self._public_prepared = False
        self._mask = np.ones(task.m, dtype=bool)

    def reset(self, seed: int | None = None):
        self.invalid_attempts = 0
        if seed is not None:
            self.__rng = np.random.default_rng(seed)
        if not self._public_prepared:
            self._exact, self._majority = exposures(self.task.adjacency, self.task.catalog)
            self._edge_index = np.asarray(np.nonzero(self.task.adjacency), dtype=np.int64)
            for array in (self._exact, self._majority, self._edge_index):
                array.setflags(write=False)
            self._public_prepared = True
        self.__base_rewards = self.__source.prepare(self.task, self._exact)
        self._key = initial_key(self.task)
        # 原来是: self._mask = valid_mask(self.task, self._key)
        self._mask = valid_mask(self.task, self._key, out=self._mask)
        for array in (self._exact, self._majority, self._edge_index):
            array.setflags(write=False)
        # Public features are constant within an episode; observations own copies.
        task, d = self.task, self.task.features.shape[1]
        nodes = np.zeros((task.n, d + 11), dtype=np.float32)
        nodes[:, :d] = task.features
        nodes[:, d] = task.adjacency.sum(1) / max(1, task.n - 1)
        nodes[:, d + 3] = 1
        nodes[:, d + 4] = task.budget / task.n
        if self.prediction is not None:
            nodes[:, d + 6:d + 10] = np.nan_to_num(self.prediction.reshape(task.n, 4))
            nodes[:, d + 10] = 1
        self._node_template = nodes
        self._global_template = np.array([task.dynamics == x for x in Dynamics] +
                                         [task.budget_mode == "exact", task.fatigue_cost,
                                          task.horizon / 20, 0], dtype=np.float32)
        return self._observation(), {}

    def _observation(self):
        task = self.task
        remaining, memory = self._key
        d = task.features.shape[1]
        buffer = self._observation_buffer
        nodes = self._node_template.copy() if buffer is None else buffer.nodes
        if buffer is not None:
            np.copyto(nodes, self._node_template)
        if task.dynamics == Dynamics.FC:
            nodes[:, d + 1] = memory
        elif task.dynamics == Dynamics.CD:
            h = np.asarray(memory)
            nodes[:, d + 2] = h / 2
            nodes[:, d + 3] = h == 0
        nodes[:, d + 5] = remaining / task.horizon
        global_state = self._global_template.copy() if buffer is None else buffer.global_state
        if buffer is not None:
            np.copyto(global_state, self._global_template)
        global_state[-1] = remaining / task.horizon
        mask = self._mask.copy() if buffer is None else buffer.mask
        if buffer is not None:
            np.copyto(mask, self._mask)
        return {"pfn_state": nodes, "global_state": global_state,
                "task_budget": int(task.budget),
                "tabular_state_key": self._key, "valid_action_mask": mask,
                "catalog": task.catalog, "edge_index": self._edge_index,
                "catalog_exact_exposure": self._exact,
                "catalog_majority_exposure": self._majority,
                "episode_time": task.horizon - remaining}

    def step(self, action_id: int):
        if self._key[0] == 0:
            raise RuntimeError("Terminal episode: call reset().")
        if not isinstance(action_id, (int, np.integer)) or not 0 <= action_id < self.task.m or not self._mask[action_id]:
            self.invalid_attempts += 1
            raise ValueError("Infeasible action; state and noise stream were not advanced.")
        # No exposure computation or counterfactual feedback occurs here.
        reward = float(self.__base_rewards[action_id] - fatigue_penalty(self.task, self._key)
                       + self.__source.sample_noise(self.__rng, self.task.n))
        self._key = transition(self.task, self._key, int(action_id))
        self._mask = valid_mask(self.task, self._key, out=self._mask)
        done = self._key[0] == 0
        return self._observation(), reward, done, False, {}


_STATIC_OBSERVATION = ('catalog', 'edge_index', 'catalog_exact_exposure',
                       'catalog_majority_exposure', 'task_budget')


class _SharedObservation:
    """One worker writes; its completion message precedes the parent's snapshot."""

    def __init__(self, task, name=None):
        from multiprocessing.shared_memory import SharedMemory
        shape = (task.n, task.features.shape[1] + 11)
        node_bytes = int(np.prod(shape)) * 4
        self.memory = SharedMemory(name=name, create=name is None, size=node_bytes + 28 + task.m)
        self.nodes = np.ndarray(shape, np.float32, buffer=self.memory.buf)
        self.global_state = np.ndarray((7,), np.float32, buffer=self.memory.buf, offset=node_bytes)
        self.mask = np.ndarray((task.m,), bool, buffer=self.memory.buf, offset=node_bytes + 28)

    def snapshot(self):
        return {'pfn_state': self.nodes.copy(), 'global_state': self.global_state.copy(),
                'valid_action_mask': self.mask.copy()}

    def close(self, unlink=False):
        self.memory.close()
        if unlink:
            self.memory.unlink()


def _environment_worker(connection):
    """Only simulator observations and executed scalar rewards leave the worker."""
    import traceback
    envs, buffers, sent_static = {}, {}, set()
    try:
        while True:
            command, jobs = connection.recv()
            if command == 'close':
                break
            try:
                if command == 'prepare_tasks':
                    from .dgp import training_task
                    result = {index: training_task(spec) for index, spec in jobs}
                elif command == 'load':
                    for buffer in buffers.values():
                        buffer.close()
                    envs, buffers, sent_static = {}, {}, set()
                    for index, (env, name) in jobs:
                        buffer = _SharedObservation(env.task, name)
                        env._observation_buffer = buffer
                        envs[index], buffers[index] = env, buffer
                    result = {}
                else:
                    result = {}
                    for index, value in jobs:
                        if command == 'reset':
                            obs, info = envs[index].reset(seed=value)
                            tail = (info,)
                        else:
                            obs, reward, done, truncated, info = envs[index].step(value)
                            tail = (reward, done, truncated, info)
                        obs = {k: v for k, v in obs.items()
                               if k not in ('pfn_state', 'global_state', 'valid_action_mask')}
                        if index in sent_static:
                            obs = {k: v for k, v in obs.items() if k not in _STATIC_OBSERVATION}
                        else:
                            sent_static.add(index)
                        result[index] = (obs, *tail)
                connection.send((True, result))
            except Exception:
                connection.send((False, traceback.format_exc()))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        for buffer in buffers.values():
            buffer.close()
        connection.close()


class ParallelEnvs:
    """Persistent CPU simulators; the policy and accelerator context stay in the parent."""

    def __init__(self, workers=8):
        import multiprocessing as mp
        import os
        self.connections, self.processes, self.static, self.buffers = [], [], {}, {}
        context = mp.get_context('spawn')
        # Prevent each simulator from starting its own multi-threaded BLAS pool.
        names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ[name] = '1'
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_environment_worker, args=(child,), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
        except BaseException:
            self.close()
            raise
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _exchange(self, command, jobs):
        groups = [[] for _ in self.connections]
        for index, value in jobs:
            groups[index % len(groups)].append((index, value))
        pending = []
        for connection, group in zip(self.connections, groups):
            if group or command == 'load':
                connection.send((command, group))
                pending.append((connection, self.processes[self.connections.index(connection)]))
        result, errors = {}, []
        from multiprocessing.connection import wait
        for connection, process in pending:
            ready = wait([connection, process.sentinel])
            if connection not in ready:
                raise RuntimeError(f'Environment worker exited with code {process.exitcode}.')
            try:
                ok, payload = connection.recv()
            except EOFError as error:
                raise RuntimeError('Environment worker closed its response pipe.') from error
            if ok:
                result.update(payload)
            else:
                errors.append(payload)
        if errors:
            raise RuntimeError('Environment worker failed:\n' + '\n'.join(errors))
        if command not in ('load', 'prepare_tasks'):
            for index, record in result.items():
                obs = record[0]
                obs.update(self.buffers[index].snapshot())
                if index not in self.static:
                    self.static[index] = {k: obs[k] for k in _STATIC_OBSERVATION}
                    for value in self.static[index].values():
                        if isinstance(value, np.ndarray):
                            value.setflags(write=False)
                obs.update(self.static[index])
        return result

    def prepare_tasks(self, specs):
        """Parallel task/prior generation, including factual static estimation."""
        result = self._exchange('prepare_tasks', list(enumerate(specs)))
        return [result[i] for i in range(len(specs))]

    def load(self, tasks):
        self.static.clear()
        previous = self.buffers
        self.buffers = {}
        try:
            for i, (env, _) in enumerate(tasks):
                self.buffers[i] = _SharedObservation(env.task)
            self._exchange('load', [(i, (env, self.buffers[i].memory.name))
                                    for i, (env, _) in enumerate(tasks)])
        finally:
            for buffer in previous.values():
                buffer.close(unlink=True)

    def reset(self, seeds):
        result = self._exchange('reset', list(enumerate(seeds)))
        return [result[i][0] for i in range(len(seeds))]

    def step(self, actions):
        return self._exchange('step', actions)

    def __enter__(self):
        return self

    def close(self):
        from time import monotonic
        # Closing the parent pipes lets idle workers exit via EOF, without send().
        for connection in self.connections:
            connection.close()
        deadline = monotonic() + 2
        for process in self.processes:
            process.join(timeout=max(0, deadline - monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        deadline = monotonic() + 1
        for process in self.processes:
            process.join(timeout=max(0, deadline - monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.kill()
        deadline = monotonic() + 1
        for process in self.processes:
            process.join(timeout=max(0, deadline - monotonic()))
        for buffer in self.buffers.values():
            buffer.close(unlink=True)
        self.buffers.clear()

    def __exit__(self, exc_type, exc, tb):
        self.close()
