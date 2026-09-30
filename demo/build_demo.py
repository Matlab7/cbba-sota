"""Data of the interactive demo (demo/site): two showcase instances of the test campaign with the plan of every method.

Usage: .venv/bin/python demo/build_demo.py
For each case, every method's plan at budget B1 on 8 cores (the routes in the rows of runs/anytime_test) is replayed in
the benchmark's own simulator (pre_set_route + execute_by_route, as the campaign scored it), and the simulator's
records give the per-robot timelines: arrival at every task, the task's start and finish, and the return to the
depot. The replayed makespan is checked against the row. Writes demo/site/data/<case>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import contextlib
import io

from cbba_sota.bench import configs
from cbba_sota.hetero import load_instance
from cbba_sota.hetero.replay import make_env
import anytime  # noqa: E402  (scripts/: the harness's definition of a disturbed row)

CASES = [  # (id, setting, split, instance, title)
    ("large", "MA-AT-150-5-500", "test", 5, "One of the largest wins: 150 robots, 500 tasks"),
    ("small", "MA-AT-25-5-50", "test", 0, "Easy to follow: 25 robots, 50 tasks"),
]
METHODS = [  # (row key, name, colour, description)
    ("ALNS2", "ALNS (ours)", "#0072B2", "adaptive large neighbourhood search over a deadlock-free coalition plan"),
    ("PCPSAT", "Parallel CP-LNS", "#E69F00", "8 parallel large neighbourhood searches that repair with CP-SAT"),
    ("CPSAT", "CP-LNS", "#D55E00", "large neighbourhood search that repairs with CP-SAT (8 threads)"),
    ("CPFULL", "CP-SAT full model", "#009E73", "CP-SAT's own search on the complete model"),
    ("CONSTRUCT", "Greedy restarts", "#7F7F7F", "8 streams of randomized greedy constructions"),
    ("RL", "RL policy (published)", "#CC79A7", "the published reinforcement-learning policy, best sampled rollout"),
    ("CTAS", "CTAS-D (exact MILP)", "#56B4E9", "the exact mixed-integer program used by the benchmark paper"),
]


def rows_of(setting: str, split: str, i: int) -> dict[str, dict]:
    runs = ROOT / "runs" / ("anytime_test" if split == "test" else "anytime_c1")
    out = {}
    for line in (runs / f"{setting}.jsonl").open():
        r = json.loads(line)
        # the rows the paper uses: this campaign's code, not disturbed (a disturbed row was re-run later)
        if (r["instance"] == i and r["cores"] == 8 and r["budget"] == "B1" and not anytime.disturbed(r)
                and r.get("split") == split):
            out[r["method"]] = r
    return out


def timeline(inst, routes) -> dict:
    """Replay 0-based routes in the simulator; per robot [task, arrival, start, finish] and the return time."""
    env = make_env(inst)
    for agent_id, route in enumerate(routes):
        env.pre_set_route([t + 1 for t in route], agent_id)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route()
    tasks = [env.task_dic[j] for j in range(inst.n_tasks)]
    robots = []
    for i in range(inst.n_agents):
        a = env.agent_dic[i]
        events, ret = [], 0.0
        for tid, arr in zip(a["route"][1:], a["arrival_time"][1:]):
            if tid >= 0:
                t = tasks[tid]
                events.append([int(tid), round(float(arr), 4), round(float(t["time_start"]), 4),
                               round(float(t["time_finish"]), 4)])
            else:
                ret = round(float(arr), 4)
        robots.append({"events": events, "ret": ret})
    done = [t for t in tasks if t["feasible_assignment"]]
    return {"robots": robots, "makespan_eval": round(float(env.current_time), 4),
            "start": [round(float(t["time_start"]), 4) if t["feasible_assignment"] else None for t in tasks],
            "finish": [round(float(t["time_finish"]), 4) if t["feasible_assignment"] else None for t in tasks],
            "completion": sorted(round(float(t["time_finish"]), 4) for t in done)}


def main() -> None:
    out_dir = ROOT / "demo" / "site" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for cid, setting, split, i, title in CASES:
        s = configs.get(setting)
        inst = load_instance(s.instance_path(split, i))
        rows = rows_of(setting, split, i)
        data = {"id": cid, "title": title, "setting": setting, "split": split, "instance": i,
                "robots_n": inst.n_agents, "tasks_n": inst.n_tasks, "species_n": inst.n_species,
                "budget_s": s.paper["RL(s.10)"].time_s, "speed": inst.speed,
                "tasks": [{"x": round(float(x), 5), "y": round(float(y), 5), "d": round(float(d), 4),
                           "req": [int(v) for v in r]} for (x, y), d, r in zip(inst.loc, inst.dur, inst.req)],
                "robots": [{"species": int(sp), "depot": [round(float(a), 5), round(float(b), 5)],
                            "traits": [int(v) for v in t]} for sp, (a, b), t in zip(inst.species, inst.depot, inst.ab)],
                "methods": []}
        for key, name, colour, desc in METHODS:
            r = rows.get(key)
            entry = {"key": key, "name": name, "colour": colour, "desc": desc}
            if r is None:
                entry.update(status="not run", makespan=None)
            elif not r["success"] or not r.get("routes"):
                entry.update(status="no plan within the budget", makespan=None)
            else:
                tl = timeline(inst, r["routes"])
                official = r["makespan"]
                if abs(tl["makespan_eval"] - official) > 0.05:
                    raise SystemExit(f"{cid} {key}: evaluator {tl['makespan_eval']} vs simulator {official}")
                entry.update(status="ok", makespan=round(float(official), 4), **tl)
            data["methods"].append(entry)
            print(f"{cid:6s} {name:24s} {entry['status']:26s} {entry['makespan']}")
        (out_dir / f"{cid}.json").write_text(json.dumps(data, separators=(",", ":")))
        index.append({"id": cid, "title": title, "setting": setting, "instance": i})
    (out_dir / "index.json").write_text(json.dumps(index))
    write_index(ROOT / "demo" / "site")
    print("wrote", out_dir)


def write_index(site: Path) -> None:
    """index.html for GitHub Pages: the page content (page.html) inside a full document, not indexed by search
    engines while the paper is under anonymous review."""
    page = (site / "page.html").read_text()
    cut = page.index("</style>") + len("</style>")
    head, body = page[:cut], page[cut:]
    (site / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="robots" content="noindex, nofollow">\n' + head + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")


if __name__ == "__main__":
    main()
