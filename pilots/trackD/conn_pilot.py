"""Track D connectivity pilot: is the network a bottleneck at HeteroMRTA geometry and time scales?

Replays the cached nominal ALNS plans (out/alns_plans.json, 20 dev instances x 4 settings), reconstructs every
agent's piecewise-linear trajectory (depart at previous finish, straight leg at speed 0.2, wait/work at the task,
return to depot, stay there), and measures a unit-disk radio graph over agents + one station at (0.5, 0.5):

- in_station: share of agent-ticks inside the station's connected component (instant multi-hop relay);
- comp_frac: mean size of an agent's component / number of agents;
- flooding latency (store-carry-forward, instant relay inside a component, one component update per tick) from
  (a) an agent to the station (uplink), (b) the station to an agent (downlink), (c) an agent to the partners of its
  next coalition (the robots a local change touches first), (d) an agent to all agents.
Origins every 1.0 time unit in [0, 0.8 * makespan]. Radii: absolute, and multiples of the random-geometric-graph
connectivity radius r_c(n) = sqrt(ln n / (pi n)) (Penrose 2003; Gupta & Kumar 1998).

Also reports the commitment lead time: departure -> start of the task a robot travels to (how long a plan change
for that task can still be delivered before the robot is physically committed).
Usage: .venv/bin/python pilots/trackD/conn_pilot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
SNAP = HERE / "_snap"
sys.path.insert(0, str(SNAP if (SNAP / "cbba_sota").exists() else HERE.parents[1]))
from cbba_sota.hetero import Instance, Plan, evaluate  # noqa: E402

DT = 0.1
STATION = np.array([0.5, 0.5])


def trajectories(inst: Instance, plan: Plan):
    sch = evaluate(inst, plan)
    routes = plan.routes()
    T_end = sch.makespan
    ticks = np.arange(0.0, T_end + DT, DT)
    A = inst.n_agents
    pos = np.empty((len(ticks), A, 2))
    lead = []  # (start - departure) of every visit
    nxt = np.full((len(ticks), A), -1)  # task the agent is heading to / at (-1 depot)
    for i in range(A):
        pts_t, pts_x = [0.0], [inst.depot[i]]
        free, here = 0.0, inst.depot[i]
        seg_task = []  # (t_from, t_to, task)
        for j in routes[i]:
            arr = free + np.linalg.norm(inst.loc[j] - here) / inst.speed
            pts_t += [arr, sch.finish[j]]
            pts_x += [inst.loc[j], inst.loc[j]]
            lead.append(sch.start[j] - free)
            seg_task.append((free, sch.finish[j], j))
            free, here = sch.finish[j], inst.loc[j]
        arr = free + np.linalg.norm(inst.depot[i] - here) / inst.speed
        pts_t += [arr, max(arr, T_end) + 1]
        pts_x += [inst.depot[i], inst.depot[i]]
        pts_t = np.array(pts_t)
        pts_x = np.array(pts_x)
        pos[:, i, 0] = np.interp(ticks, pts_t, pts_x[:, 0])
        pos[:, i, 1] = np.interp(ticks, pts_t, pts_x[:, 1])
        for a, b, j in seg_task:
            nxt[(ticks >= a) & (ticks < b), i] = j
    return ticks, pos, nxt, sch, routes, np.array(lead)


def components(p: np.ndarray, R: float) -> np.ndarray:
    nodes = np.vstack([p, STATION])  # station = last node
    d = np.linalg.norm(nodes[:, None] - nodes[None], axis=-1)
    adj = csr_matrix(d <= R)
    return connected_components(adj, directed=False)[1]


def analyse(inst: Instance, plan: Plan, radii: list[float]) -> dict:
    ticks, pos, nxt, sch, routes, lead = trajectories(inst, plan)
    A = inst.n_agents
    members = plan.members
    out = {"lead": lead}
    for R in radii:
        labs = np.array([components(pos[k], R) for k in range(len(ticks))])  # [K, A+1]
        in_st = (labs[:, :A] == labs[:, [A]]).mean()
        comp = np.array([np.bincount(l[:A], minlength=A + 1)[l[:A]].mean() / A for l in labs]).mean()
        # flooding from origins every 1.0 time unit
        o_ticks = np.arange(0, int(0.8 * len(ticks)), int(1.0 / DT))
        up, down, partner, allr = [], [], [], []
        for k0 in o_ticks:
            N = A + 1
            inf = np.eye(N, dtype=bool)  # origin o = node o (agents and the station)
            t_reach = np.full((N, N), np.inf)
            t_reach[np.arange(N), np.arange(N)] = 0.0
            for k in range(k0, len(ticks)):
                lab = labs[k]
                for c in np.unique(lab):
                    idx = np.flatnonzero(lab == c)
                    got = inf[:, idx].any(axis=1)
                    new = got[:, None] & ~inf[:, idx]
                    if new.any():
                        r, cc = np.nonzero(new)
                        t_reach[r, idx[cc]] = (k - k0) * DT
                        inf[np.ix_(got, idx)] = True
                if inf.all():
                    break
            up += list(t_reach[:A, A])
            down += list(t_reach[A, :A])
            allr += list(t_reach[:A, :A].max(axis=1))
            for i in range(A):
                j = nxt[k0, i]
                if j >= 0:
                    others = [m for m in members[j] if m != i]
                    if others:
                        partner.append(t_reach[i, others].max())
        out[R] = dict(in_station=in_st, comp_frac=comp, up=np.array(up), down=np.array(down),
                      partner=np.array(partner), all=np.array(allr))
    return out


def q(x, p):
    x = np.asarray(x, float)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(np.where(np.isfinite(x), x, 1e9), p))


def main():
    plans = json.loads((HERE / "out" / "alns_plans.json").read_text())
    by_setting: dict[str, list[str]] = {}
    for p in plans:
        by_setting.setdefault(Path(p).parts[-3], []).append(p)
    lines = []
    for setting, paths in sorted(by_setting.items()):
        A = int(setting.split("-")[2])
        rc = float(np.sqrt(np.log(A) / (np.pi * A)))
        radii = [0.5, 0.3, 0.2, 0.15, 0.1]
        acc = {R: {k: [] for k in ("in_station", "comp_frac", "up", "down", "partner", "all")} for R in radii}
        leads, mks, durs, legs = [], [], [], []
        for p in sorted(paths)[:10]:
            inst = Instance.from_pickle(p)
            plan = Plan.from_routes(plans[p]["routes"], inst.n_tasks, one_based=True)
            res = analyse(inst, plan, radii)
            leads.append(res["lead"])
            mks.append(evaluate(inst, plan).makespan)
            durs.append(inst.dur)
            for R in radii:
                for k in acc[R]:
                    v = res[R][k]
                    acc[R][k].append(v if np.ndim(v) else [v])
        lead = np.concatenate(leads)
        lines.append(f"=== {setting}  (10 dev inst, ALNS nominal plans)  r_c(n={A})={rc:.3f}  "
                     f"makespan {np.mean(mks):.1f}  task dur mean {np.concatenate(durs).mean():.2f}  "
                     f"lead(dep->start) median {np.median(lead):.2f} p10 {np.quantile(lead, .1):.2f} "
                     f"p90 {np.quantile(lead, .9):.2f}")
        lines.append(f"  R     R/r_c  in_station  comp_frac | latency median/p90 (inf=never within episode): "
                     f"uplink         downlink       ->partners     ->all")
        for R in radii:
            a = {k: np.concatenate([np.atleast_1d(np.asarray(x, float)) for x in v]) for k, v in acc[R].items()}
            def f(k):
                x = a[k]
                never = np.mean(~np.isfinite(x))
                return f"{q(x, .5):5.2f}/{q(x, .9):5.2f} n{never:.2f}".replace("1000000000.00", "  inf")
            lines.append(f"  {R:4.2f}  {R / rc:5.2f}  {a['in_station'].mean():9.3f}  {a['comp_frac'].mean():9.3f} | "
                         f"{f('up'):>14} {f('down'):>14} {f('partner'):>14} {f('all'):>14}")
        print("\n".join(lines[-len(radii) - 2:]), flush=True)
    (HERE / "out" / "conn.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
