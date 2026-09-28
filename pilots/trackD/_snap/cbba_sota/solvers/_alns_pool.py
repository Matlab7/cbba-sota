"""Shared-memory plan stores of the coalition ALNS (driver in ``cbba_sota.solvers.alns``).

Both are created before the workers fork and work unchanged inside one process. A plan is stored as its global
order, coalitions and ``(makespan, objective)``; loading one into a kernel solution recomputes its schedule.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from cbba_sota.solvers import _alns_kernels as K


class _Slots:
    """``n`` plan slots in shared memory (``seq``, ``mem``, ``cnt``, makespan, objective)."""

    def __init__(self, n: int, T: int, W: int):
        ctx = mp.get_context("fork")
        self.lock = ctx.Lock()
        self.n, self.T, self.W = n, T, W
        self._seq, self._cnt = ctx.RawArray("q", n * T), ctx.RawArray("q", n * T)
        self._mem = ctx.RawArray("q", n * T * W)
        self._val = ctx.RawArray("d", 2 * n)
        self.seq = np.frombuffer(self._seq, np.int64).reshape(n, T)
        self.cnt = np.frombuffer(self._cnt, np.int64).reshape(n, T)
        self.mem = np.frombuffer(self._mem, np.int64).reshape(n, T, W)
        self.val = np.frombuffer(self._val, np.float64).reshape(n, 2)
        self.val[:] = np.inf

    def put(self, k: int, sol: tuple) -> None:
        self.seq[k], self.mem[k], self.cnt[k] = sol[0], sol[1], sol[2]
        self.val[k] = sol[10][0], sol[10][2]

    def get(self, k: int, sol: tuple, inst: tuple, lam: float) -> None:
        sol[0][:], sol[1][...], sol[2][:], sol[3][0] = self.seq[k], self.mem[k], self.cnt[k], self.T
        K.schedule(inst, sol, lam)


class Exchange(_Slots):
    """Shared incumbent (v1): offer a better best, adopt a strictly better shared one."""

    def __init__(self, T: int, W: int):
        super().__init__(1, T, W)

    def sync(self, best: tuple, into: tuple, inst: tuple, lam: float) -> bool:
        """Publish ``best`` if it beats the shared plan, else load a strictly better shared plan into ``into``;
        returns True in the latter case."""
        ms, obj = best[10][0], best[10][2]
        with self.lock:
            if K.better(ms, obj, *self.val[0]):
                self.put(0, best)
                return False
            if not K.better(*self.val[0], ms, obj):
                return False
            self.get(0, into, inst, lam)
        return True


class ElitePool(_Slots):
    """Up to ``size`` distinct good plans shared by all workers. An offered plan is dropped if one with the same
    makespan and objective is stored, and otherwise replaces the worst stored plan when the pool is full and it is
    better."""

    def __init__(self, size: int, T: int, W: int):
        super().__init__(size, T, W)

    def offer(self, sol: tuple) -> bool:
        ms, obj = sol[10][0], sol[10][2]
        with self.lock:
            if (np.abs(self.val - (ms, obj)) <= K.EPS).all(axis=1).any():
                return False
            k = int(np.lexsort((self.val[:, 1], self.val[:, 0]))[-1])  # worst (empty slots are inf)
            if not K.better(ms, obj, *self.val[k]):
                return False
            self.put(k, sol)
            return True

    def filled(self) -> np.ndarray:
        return np.flatnonzero(np.isfinite(self.val[:, 0]))

    def load(self, k: int, sol: tuple, inst: tuple, lam: float) -> None:
        with self.lock:
            self.get(k, sol, inst, lam)
