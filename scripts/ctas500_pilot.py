"""CTAS-D (CP_SAT backend, 8 threads) at B1 on dev instance 0 of a 500-task setting: does it find an incumbent, and
how much memory does the solve hold? One JSON line (docs/results/ctas/ctas500_dev_pilot.jsonl).

Usage: ctas500_pilot.py SETTING CPUS        (e.g. MA-AT-150-5-500 0-7; run with OMP_NUM_THREADS=1)
"""
import json
import resource
import sys
import time

from cbba_sota.bench import configs, runtime
from cbba_sota.hetero import Instance, replay
from cbba_sota.solvers import ctas

name = sys.argv[1]
s = configs.get(name)
inst = Instance.from_pickle(s.instance_path("dev", 0))
inst.tt, inst.da  # noqa: B018
runtime.pin(runtime.parse_cpus(sys.argv[2]))
b1 = s.paper["RL(s.10)"].time_s
t0 = time.perf_counter()
res = ctas.solve(inst, b1, backend="CP_SAT", threads=8, t0=t0)
out = {"setting": name, "status": res.status, "makespan": res.makespan, "wall_s": res.wall_s, "cpu_s": res.cpu_s,
       "build_s": res.build_s, "qmax": res.qmax, "objective": res.objective, "bound": res.bound,
       "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
if res.plan is not None:
    out["replay"] = replay(inst, res.plan)["makespan"]
print(json.dumps(out), flush=True)
