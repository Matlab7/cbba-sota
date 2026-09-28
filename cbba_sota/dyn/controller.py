"""Run a rolling planner (SPARC or a Section 5.2 baseline) inside the ground-truth env, ``DynTaskEnvX.run_plan``.

``PlanController`` is the ``Controller`` of ``cbba_sota.dyn.env``: at every epoch with events it maps the env's
events to trigger kinds (release -> ``release``; failure, abandon -> ``orphan``; idle -> ``idle`` when open tasks
exist, i.e. released tasks that are neither committed, started nor finished, the spec's repair region (the env's
own ``open_tasks`` count also includes committed tasks); finish, wasted, window_end, wakeup -> noise), asks the policy whether to re-plan (spec 4.2), builds the belief
(``PlanState``) from the env's causal snapshot ``env.state()`` with the declared predictors (spec 4.1), and writes
the new routes back (``env.set_route``).

Commitment bookkeeping lives here, because the env only knows which members *departed*: a task is committed once
any member departed to it, and its coalition (the adopted plan's, plus residual extras) and key are then frozen.
Members that abandon (a partner's failure detection, spec 4.3), fail, or arrive late or redundant (D2, a wasted
trip) leave the frozen coalition. The env cannot
redirect a travelling robot; if a plan releases a commitment (G4 fallback) a travelling member is abandoned from it
when it arrives (``request_wakeup`` at its ETA).
"""
from __future__ import annotations

import numpy as np

from cbba_sota.dyn.planner import AT_DEPOT, DynPlan, PlanState
from cbba_sota.hetero.instance import Instance

__all__ = ["PlanController", "run_env"]

EVENT_KIND = {"release": "release", "failure": "orphan", "abandon": "orphan", "finish": "finish",
              "wasted": "wasted", "window_end": "horizon", "wakeup": "wakeup"}


class PlanController:
    """Adapter from ``RollingPolicy`` (``cbba_sota.dyn.sparc``) to ``DynTaskEnvX.run_plan``."""

    def __init__(self, policy, inst: Instance, kappa: float = 1.0):
        self.policy, self.inst, self.kappa = policy, inst, float(kappa)
        T = inst.n_tasks
        self.frozen: list[tuple[int, ...]] = [()] * T
        self.keys = np.full(T, np.nan)
        self.key_floor = -1.0
        self.pending: dict[int, int] = {}  # robot -> released task it must leave on arrival
        self.cpu: list[float] = []

    # ---- events -> trigger kinds ---------------------------------------------------------------------------------
    def kinds(self, events, n_open: int = 1) -> frozenset[str]:
        out = set()
        for e in events:
            if e.kind == "idle":
                if n_open > 0:
                    out.add("idle")
            else:
                out.add(EVENT_KIND.get(e.kind, e.kind))
        return frozenset(out)

    def n_open(self, sv) -> int:
        """Released tasks that are neither committed (by this controller's bookkeeping), started nor finished."""
        return sum(1 for j in range(len(self.frozen)) if sv.released[j] and not sv.finished[j]
                   and not sv.started[j] and not self.frozen[j])

    def on_events(self, env, t: float, events) -> bool:
        for e in events:  # members that left a coalition are no longer frozen in it
            if e.kind == "failure":
                gone = set(e.agents) | set(e.get("abandoned", ()) or ())
                self.frozen = [tuple(m for m in f if m not in gone) for f in self.frozen]
            elif e.kind in ("abandon", "wasted"):  # left the coalition (a wasted trip: late or redundant arrival)
                for j in e.tasks:
                    self.frozen[j] = tuple(m for m in self.frozen[j] if m not in e.agents)
        sv = env.state()
        if self._leave_released(env, sv):
            sv = env.state()  # the abandons changed the coalitions
        self._freeze(sv)
        kinds = self.kinds(events, self.n_open(sv))
        if not self.policy.triggered(kinds):
            self.policy.decisions.append({"t": t, "kinds": sorted(kinds), "replan": False})
            return False
        state = self.belief(env, sv)
        plan = self.policy.replan(state)
        self.policy.decisions.append({"t": t, "kinds": sorted(kinds), "replan": True, "seed": plan.seed,
                                      "iters": plan.iterations, "cpu_s": plan.cpu_s, "pred_ms": plan.makespan,
                                      "n_open": plan.n_open})
        self.cpu.append(plan.cpu_s)
        self.adopt(env, sv, plan)
        return True

    # ---- commitments ----------------------------------------------------------------------------------------------
    def _freeze(self, sv) -> None:
        inc = self.policy.incumbent
        for j in range(len(self.frozen)):
            if sv.finished[j] or sv.started[j]:
                self.frozen[j] = ()
                continue
            dep = tuple(m for m in sv.committed[j] if sv.alive[m] and self.pending.get(m) != j)
            if dep and not self.frozen[j]:  # first departure since the last plan: freeze its coalition
                planned = tuple(inc.members[j]) if inc is not None else ()
                self.frozen[j] = tuple(sorted(set(planned) | set(dep)))
            elif dep:
                self.frozen[j] = tuple(sorted(set(self.frozen[j]) | set(dep)))
            self.frozen[j] = tuple(m for m in self.frozen[j] if sv.alive[m])

    def _leave_released(self, env, sv) -> bool:
        """Robots that arrived at a task whose commitment was released leave it; returns True if any did."""
        left = False
        for i, j in list(self.pending.items()):
            if sv.mode[i] == "wait" and sv.target[i] == j:
                left |= bool(env.abandon(i))
                del self.pending[i]
            elif sv.mode[i] in ("failed", "idle", "home", "to_home") or sv.target[i] != j:
                del self.pending[i]
        return left

    # ---- belief ---------------------------------------------------------------------------------------------------
    def belief(self, env, sv) -> PlanState:
        """Good-communication belief from the env's causal snapshot (spec 4.1 predictors)."""
        inst, t, kap = self.inst, float(sv.t), self.kappa
        T, A = inst.n_tasks, inst.n_agents
        committed = np.array([bool(self.frozen[j]) and not sv.finished[j] and not sv.started[j] for j in range(T)])
        ready, pos = np.zeros(A), np.array(inst.depot, float)
        pos_task = np.full(A, AT_DEPOT, np.int64)
        head = np.full(A, -1, np.int64)
        for i in range(A):
            m, c = sv.mode[i], int(sv.target[i])
            if m == "failed":
                continue
            if m in ("travel", "to_home"):
                moved = (t - sv.dep[i]) - sv.stall[i]
                ready[i] = max(sv.dep[i] + sv.stall[i] + sv.tau[i], t) if kap == 1.0 else \
                    t + max(sv.tau[i] - moved, 0.0) * kap
                if m == "travel":  # locked to its target unless it leaves on arrival (released commitment)
                    pos[i], pos_task[i] = inst.loc[c], c
                    head[i] = c if committed[c] and i in self.frozen[c] and self.pending.get(i) != c else -1
            elif m == "wait":
                ready[i], pos[i], pos_task[i] = sv.eta[i], inst.loc[c], c
                head[i] = c if committed[c] and i in self.frozen[c] and self.pending.get(i) != c else -1
            elif m == "work":
                ready[i] = max(sv.start[c] + inst.dur[c], t)
                pos[i], pos_task[i] = inst.loc[c], c
            elif m == "idle":
                ready[i], pos[i], pos_task[i] = t, inst.loc[c], c
            else:  # home
                ready[i] = t
        keys = np.nan_to_num(self.keys, nan=0.0)
        return PlanState(inst=inst, now=t, released=sv.released.copy(), done=sv.finished.copy(),
                         started=sv.started.copy(), committed=committed, members=list(self.frozen), keys=keys,
                         alive=np.asarray(sv.alive, bool).copy(), ready=ready, pos=pos, pos_task=pos_task, head=head,
                         kappa=kap, key_floor=self.key_floor)

    # ---- adoption ------------------------------------------------------------------------------------------------
    def adopt(self, env, sv, plan: DynPlan) -> None:
        T = self.inst.n_tasks
        live = sv.released & ~sv.finished & ~sv.started
        released = set(plan.released)
        for j in range(T):
            if not live[j]:
                continue
            if plan.members[j]:
                self.keys[j] = plan.keys[j]
            if j in released:
                self.frozen[j] = ()
            elif self.frozen[j] and plan.members[j]:
                self.frozen[j] = tuple(plan.members[j])
        if np.isfinite(plan.keys).any():
            self.key_floor = max(self.key_floor, float(np.nanmax(plan.keys)))
        routes = plan.routes_for(self.inst.n_agents)
        for i in range(self.inst.n_agents):
            if not sv.alive[i]:
                continue
            r = [j for j in routes[i] if live[j]]
            m, c = sv.mode[i], int(sv.target[i])
            if m in ("travel", "wait") and c >= 0:
                locked = self.pending.get(i) != c and c not in released
                if r and r[0] == c:
                    r = r[1:]
                    self.pending.pop(i, None)
                elif c in r and locked:
                    raise AssertionError(f"robot {i} is committed to task {c} but the plan puts it later: {r}")
                elif m == "wait":  # leaves now (it may come back later if ``c`` is further down its route)
                    env.abandon(i)
                else:  # leaves on arrival
                    self.pending[i] = c
                    env.request_wakeup(float(sv.eta[i]))
            env.set_route(i, r)


def run_env(policy, inst: Instance, real, **env_options):
    """One episode of ``policy`` in ``DynTaskEnvX`` (ground truth) on realization ``real`` (``perturb``).

    Returns ``(Episode, PlanController)``; ``Episode.world`` names the env."""
    from cbba_sota.dyn.env import make_env

    env = make_env(inst.source or inst, real, **env_options)
    ctl = PlanController(policy, inst, real.kappa())
    ep = env.run_plan(ctl)
    return ep, ctl
