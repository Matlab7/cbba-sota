"""Shared-memory incumbent of the coalition ALNS workers (driver in ``cbba_sota.solvers.alns``).

Created before the workers fork. The plan is stored as its global order, coalitions and ``(makespan, objective)``;
loading it into a kernel solution recomputes its schedule.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from cbba_sota.solvers import _alns_kernels as K


class Exchange:
    """Shared incumbent: offer a better best, adopt a strictly better shared one."""

    def __init__(self, T: int, W: int):
        ctx = mp.get_context("fork")
        self.lock = ctx.Lock()
        self.T = T
        self._seq, self._cnt, self._mem = ctx.RawArray("q", T), ctx.RawArray("q", T), ctx.RawArray("q", T * W)
        self._val = ctx.RawArray("d", 2)
        self.seq = np.frombuffer(self._seq, np.int64)
        self.cnt = np.frombuffer(self._cnt, np.int64)
        self.mem = np.frombuffer(self._mem, np.int64).reshape(T, W)
        self.val = np.frombuffer(self._val, np.float64)
        self.val[:] = np.inf

    def sync(self, best: tuple, into: tuple, inst: tuple, lam: float) -> bool:
        """Publish ``best`` if it beats the shared plan, else load a strictly better shared plan into ``into``;
        returns True in the latter case."""
        ms, obj = best[10][0], best[10][2]
        with self.lock:
            if K.better(ms, obj, *self.val):
                self.seq[:], self.mem[...], self.cnt[:] = best[0], best[1], best[2]
                self.val[:] = ms, obj
                return False
            if not K.better(*self.val, ms, obj):
                return False
            into[0][:], into[1][...], into[2][:], into[3][0] = self.seq, self.mem, self.cnt, self.T
            K.schedule(inst, into, lam)
        return True
