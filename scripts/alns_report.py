"""Summaries of ALNS runs against per-instance references (all ALNS makespans are env replays).

Usage: alns_report.py TAG [TAG ...]
References (per instance), every one an env run or env replay:
- RALTestSet: RL(g.)/RL(s.10) rollouts (pilots/heteromrta/rl_eval.json), CTAS-D (shipped results.yaml, timeCost /
  100, which equals the replay of its solutions), pilot CP-SAT 30 s x 8 workers (redteam/cpsat_30s_8w.json routes,
  replayed here).
- ScaleSet50: RL(g.)/RL(s.10) rollouts (redteam/rl_eval_scale50.json), pilot LNS 13 s / 30 s
  (redteam/scale50_lns_*.json routes, replayed here on the instances rebuilt by run_alns.instances_from_json).
- Generated dev settings: the RL campaign's runs/rl/*/dev.jsonl, scored by the best rollout's own env run. The rows
  of runs/alns/rl_ref (scripts/alns_rl_reference.py) store only the replay of the best rollout's routes, which is not
  the RL score, so they are not used.
A failed run (not every task finished, or makespan >= 200) counts as 200. Ratios are paired means of ALNS /
reference with a bootstrap 95% CI (10,000 resamples, seed 0). Rows computed on a superseded instance (fingerprint
differs from the current one) are dropped; of several rows for one instance the last counts.
"""
from __future__ import annotations

import json
import re
import sys
from functools import cache

import numpy as np

from cbba_sota.bench.configs import HETEROMRTA_DIR, MAX_TIME, ROOT, RUNS_DIR
from cbba_sota.bench.runtime import fingerprint, is_stale, read_rows
from cbba_sota.hetero.replay import replay_routes, succeeded

PILOT = ROOT / "pilots" / "heteromrta"
sys.path.insert(0, str(ROOT / "scripts"))


def _score(makespan: float, success: bool) -> float:
    return makespan if succeeded(success, makespan) else MAX_TIME


def _replayed(source, routes) -> float:
    """Env replay score of stored 1-based routes."""
    rep = replay_routes(source, routes)
    return _score(rep["makespan"], rep["success"])


def _ral_refs() -> dict[str, dict[str, float]]:
    rl = json.loads((PILOT / "rl_eval.json").read_text())
    cp = {r[0]: r[-1] for r in json.loads((PILOT / "redteam" / "cpsat_30s_8w.json").read_text())}  # routes
    out = {}
    for i in range(50):
        yaml = (HETEROMRTA_DIR / "RALTestSet" / f"env_{i}" / "results.yaml").read_text()
        out[f"RALTestSet/test/{i}"] = {
            "RL(g.)": rl[f"RALTestSet/env_{i}.pkl"]["g"], "RL(s.10)": rl[f"RALTestSet/env_{i}.pkl"]["s10"],
            "CTAS-D 600s": float(re.search(r"timeCost: ([\d.]+)", yaml).group(1)) / 100,
            "CP-SAT 30s x8": _replayed(HETEROMRTA_DIR / "RALTestSet" / f"env_{i}.pkl",
                                       cp[f"HeteroMRTA/RALTestSet/env_{i}.pkl"])}
    return out


@cache
def _json_fingerprints(stem: str) -> dict[str, str]:
    """Fingerprints of the instances of a red-team JSON export (run_alns --json), by name."""
    from run_alns import instances_from_json

    return {name: fingerprint(inst) for name, inst in instances_from_json(PILOT / "redteam" / f"{stem}.json").items()}


def _stale(row: dict) -> bool:
    if row.get("split") != "json":
        return is_stale(row)
    return "fingerprint" in row and row["fingerprint"] != _json_fingerprints(row["setting"]).get(row["name"])


def _rows(path) -> list[dict]:
    """Rows of one ALNS output file on the current instances, the last one per instance."""
    return list({(r["setting"], r["split"], r["index"]): r for r in read_rows(path) if not _stale(r)}.values())


def _scale50_refs() -> dict[str, dict[str, float]]:
    from run_alns import instances_from_json

    rt = PILOT / "redteam"
    insts = instances_from_json(rt / "instances_scale50.json")
    rl = json.loads((rt / "rl_eval_scale50.json").read_text())
    lns13 = json.loads((rt / "scale50_lns_13s.json").read_text())
    lns30 = json.loads((rt / "scale50_lns_30s.json").read_text())
    out = {}
    for i in range(30):
        inst, key = insts[f"ScaleSet50/env_{i}"], f"HeteroMRTA/ScaleSet50/env_{i}.pkl"
        out[f"ScaleSet50/env_{i}"] = {
            "RL(g.)": rl[f"ScaleSet50/env_{i}.pkl"]["g"], "RL(s.10)": rl[f"ScaleSet50/env_{i}.pkl"]["s10"],
            "pilot LNS 13s": _replayed(inst, lns13[key][1]), "pilot LNS 30s": _replayed(inst, lns30[key][1])}
    return out


def _rl_ref_refs() -> dict[str, dict[str, float]]:
    """The RL campaign's dev rows (runs/rl/<setting>/dev.jsonl), scored by the best rollout's own env run."""
    out: dict[str, dict[str, float]] = {}
    for path in (RUNS_DIR / "rl").glob("*/dev.jsonl"):
        for r in read_rows(path):
            if is_stale(r):
                continue
            name = f"{r['setting']}/dev/{r['instance']}"
            out.setdefault(name, {})[f"{r['method']} {r['workers']}cpu"] = _score(r["makespan"], r["success"])
    return out


def _ci(ratios: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = ratios[rng.integers(0, len(ratios), (10_000, len(ratios)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    refs = {**_ral_refs(), **_scale50_refs(), **_rl_ref_refs()}
    for tag in sys.argv[1:]:
        for path in sorted((RUNS_DIR / "alns" / tag).glob("*.jsonl")):
            rows = _rows(path)
            if not rows:
                continue
            ms = np.array([_score(r["makespan"], r["success"]) for r in rows])
            exact = all(r["makespan"] == r["eval_makespan"] == r["search_makespan"] and r["skipped"] == 0
                        for r in rows)
            print(f"{tag}/{path.stem}: n={len(rows)} budget {rows[0]['time_limit']:g}s x {rows[0]['workers']} worker(s)"
                  f"  ALNS mean {ms.mean():.3f} (sd {ms.std(ddof=1) if len(ms) > 1 else 0:.3f})"
                  f"  success {np.mean([succeeded(r['success'], r['makespan']) for r in rows]):.2f}"
                  f"  replay==evaluator {exact}"
                  f"  init {np.mean([r['init_makespan'] for r in rows]):.3f}"
                  f"  it/s {np.mean([r['it_per_s'] for r in rows]):.0f}")
            methods = sorted({m for r in rows for m in refs.get(r["name"], {})})
            for m in methods:
                pairs = [(_score(r["makespan"], r["success"]), refs[r["name"]][m]) for r in rows
                         if m in refs.get(r["name"], {}) and np.isfinite(refs[r["name"]][m])]
                if not pairs:
                    continue
                a, b = np.array(pairs).T
                ratio = a / b
                lo, hi = _ci(ratio)
                print(f"    vs {m:14s} n={len(a):2d} ref mean {b.mean():8.3f}  ALNS/ref {ratio.mean():.3f} "
                      f"[{lo:.3f}, {hi:.3f}]  ALNS better {(a < b - 1e-9).sum()}/{len(a)}")


if __name__ == "__main__":
    main()
