"""Validation split, BKS collection and anytime-curve helpers (scripts/gen_instances.py, bks.py, anytime.py)."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cbba_sota.bench import configs, runtime
from cbba_sota.bench.configs import SETTINGS, get

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name: str):
    sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)


def _need(setting: str, split: str, i: int = 0) -> None:
    if not get(setting).instance_path(split, i).exists():
        pytest.skip(f"{setting}/{split} not generated (scripts/gen_instances.py)")


def test_val_split_seeds():
    """val: seed base 900000, 20 instances per generated setting, disjoint from test and dev seeds."""
    s = get("MA-AT-150-10-500")
    assert s.seed("val", 3) == 900_000 + 1000 * s.index + 3 and s.n_instances("val") == 20
    seeds = [t.seed(split, i) for t in SETTINGS[:16] for split in configs.SPLITS for i in range(t.n_instances(split))]
    assert len(seeds) == len(set(seeds)) == 16 * (50 + 20 + 20)
    assert get("RALTestSet").splits() == ("test",)
    with pytest.raises(IndexError):
        s.seed("val", 20)


def test_val_instances_are_fresh_and_match_their_seeds():
    """Each val pickle equals a fresh generation from its seed and shares no data with dev or test."""
    gen = _script("gen_instances")
    for name in ("SA-BT-25-5-20", "MA-AT-50-5-50"):
        _need(name, "val")
        _, _, _, fp, same = gen._fingerprint((name, "val", 0))
        assert same
        others = {runtime.instance_fingerprint(name, split, i) for split in ("test", "dev") for i in range(3)}
        assert fp not in others


def test_bks_takes_the_best_successful_plan_and_the_best_bound():
    bks = _script("bks")
    base = {"setting": "S", "split": "val", "instance": 0, "routes": [[0]]}
    rows = [base | {"kind": "ALNS2", "makespan": 10.0, "success": True},
            base | {"kind": "CPLNS", "makespan": 9.5, "success": False},  # failed: never a BKS
            base | {"method": "ALNS2", "cores": 8, "budget": "B1", "makespan": 9.8, "success": True},
            base | {"kind": "CPFULL", "makespan": 9.9, "success": True, "lb": 9.0},
            base | {"kind": "CPFULL", "makespan": 9.9, "success": True, "lb": 9.2, "routes": None},
            base | {"kind": "ALNS2", "makespan": 9.0, "success": True, "check": True},  # compared, not included
            base | {"kind": "ALNS2i", "makespan": 9.1, "success": True},  # a subset check as well
            base | {"instance": 1, "kind": "ALNS2", "makespan": 5.0, "success": True}]
    best = bks.best_known(rows)
    assert best[0]["bks"] == 9.8 and best[0]["row"]["method"] == "ALNS2" and best[0]["lb"] == 9.2
    assert best[1]["bks"] == 5.0 and best[1]["lb"] is None


def test_cpfull_bound_is_below_every_replayed_plan():
    """The certified bound (CP bound - (T + 1) / SCALE) never exceeds an env-replayed makespan, and every BKS row is
    an env replay of its own plan."""
    bks = _script("bks")
    name = "SA-BT-25-5-20"
    _need(name, "val")
    alns = bks.run_job(name, "val", 0, "ALNS2", 2, 1.0)
    full = bks.run_job(name, "val", 0, "CPFULL", 4, 20.0, alns["routes"])
    for row in (alns, full):
        assert row["success"] and row["makespan"] == row["eval_makespan"] and row["skipped"] == 0
    T = get(name).n_tasks
    assert full["lb"] == pytest.approx(full["bound"] - (T + 1) / 1000)
    assert full["lb"] <= min(alns["makespan"], full["makespan"]) + 1e-9
    assert full["makespan"] <= alns["makespan"] + 1e-9  # hinted with the ALNS plan


def test_independent_references_use_no_alns_plan(tmp_path, monkeypatch):
    """PLNS and CPFULLc start from constructions only; they enter the BKS and are reported on their own."""
    bks = _script("bks")
    name = "SA-BT-25-5-20"
    _need(name, "val")
    plns = bks.run_job(name, "val", 0, "PLNS", 2, 2.0)
    full = bks.run_job(name, "val", 0, "CPFULLc", 2, 4.0)
    for row in (plns, full):
        assert row["success"] and row["makespan"] == row["eval_makespan"] and row["routes_base"] == 0
    assert plns["makespan"] <= plns["init_makespan"] + 1e-9 and full["makespan"] <= full["hint_makespan"] + 1e-9
    assert full["lb"] <= min(plns["makespan"], full["makespan"]) + 1e-9
    fp = runtime.instance_fingerprint(name, "val", 0)
    rows = [{"setting": name, "split": "val", "instance": 0, "kind": k, "workers": 2, "time_s": 2.0} | r
            | {"fingerprint": fp} for k, r in (("PLNS", plns), ("CPFULLc", full))]
    rows.append(rows[0] | {"kind": "ALNS2", "makespan": plns["makespan"] + 1.0})
    monkeypatch.setattr(bks, "OUT", tmp_path)
    monkeypatch.setattr(bks, "ANYTIME", ())
    (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    best = bks.best_known(bks.rows_of(name))[0]
    assert best["bks"] == min(plns["makespan"], full["makespan"]) and best["row"]["kind"] in bks.INDEPENDENT


def test_anytime_grid():
    at = _script("anytime")
    small, mid, large = (at.grid(get(n)) for n in ("SA-AT-25-5-20", "MA-AT-50-5-200", "MA-AT-150-5-500"))
    for method in ("ALNS2", "ALNS1", "CPSAT", "RL"):  # every method on 1 core at every budget
        assert {b for m, c, b in small if m == method and c == 1} == set(at.budgets(get("SA-AT-25-5-20")))
    assert ("ALNS1", 8, "B1") in small and ("ALNS1", 8, "2B1") not in small and ("RL", 8, "2B1") not in small
    assert ("ALNS2", 8, "2B1") in mid and ("CPSAT", 8, "2B1") not in mid
    assert {b for m, c, b in small if c == 8 and m != "CONSTRUCT"} == {"B1"}  # 20 tasks: 8 cores at B1 only
    later = at.grid(get("MA-AT-25-5-50"), at.EXTENDED)  # core points only
    assert {(c, b) for m, c, b in later if m == "ALNS2"} == {(1, "0.5"), (1, "1"), (1, "2"), (1, "B1"), (8, "B1"),
                                                             (8, "2B1")}
    assert set(later) < set(at.grid(get("MA-AT-25-5-50"), 0))
    assert {(m, c, b) for m, c, b in large if c == 8} == {(m, 8, "B1") for m in ("ALNS2", "CPSAT", "RL")}
    assert ("CONSTRUCT", 1, "stream") in large and at.stream_budget(get("MA-AT-150-5-500")) == 560.5


def test_c1_grid_and_tune_variants():
    """The same-host campaign: every competitor on 8 cores at B1 on every instance, ALNS2 on 1 core at the C2
    budgets, the 500-task settings included (8 restart streams up to B1), no 20-task settings; tune variants."""
    at = _script("anytime")
    mid, large = at.grid_c1(get("MA-AT-25-5-50"), 0), at.grid_c1(get("MA-AT-150-5-500"))
    later = at.grid_c1(get("MA-AT-25-5-50"), at.EXTENDED)
    for jobs in (mid, later, large):
        assert {(m, 8, "B1") for m in ("ALNS2", "CPSAT", "PCPSAT", "CPFULL", "RL")} <= set(jobs)
        assert ("CONSTRUCT", 8, "stream") in jobs and {("ALNS2", 1, b) for b in ("0.5", "1", "2", "B1")} <= set(jobs)
    assert ("CTAS", 8, "B1") in mid and ("CTAS", 8, "B1") not in large  # CTAS-D fails at 500 tasks (dev pilot)
    assert set(later) < set(mid) and ("PCPSAT", 8, "2B1") in later and ("RL", 8, "1") in mid
    assert at.grid_c1(get("SA-AT-25-5-20")) == [] and {b for _, c, b in large if c == 8} == {"B1", "stream"}
    assert at.parse_variant("PCPSAT:sub_time=0.5,q0=8") == ("PCPSAT", "sub_time=0.5,q0=8", {"sub_time": 0.5, "q0": 8})
    with pytest.raises(SystemExit):
        at.parse_variant("ALNS2:lam=0.1")


def test_construct_streams_are_read_off_at_every_budget():
    """Stream rows: best plan finished by each budget, env-replayed; the 8-stream row is the best of the streams."""
    at = _script("anytime")
    name = "SA-AT-25-5-20"
    _need(name, "val")
    checkpoints = [0.02, 0.1, 0.3]
    parts = [at.run_stream(name, "val", 0, k, checkpoints) for k in range(2)]
    for p in parts:
        ms = [p["checkpoints"][b]["makespan"] for b in checkpoints]
        assert ms == sorted(ms, reverse=True)
        for b in checkpoints:
            c = p["checkpoints"][b]
            assert c["makespan"] == c["eval_makespan"] and c["success"]
            assert c["t_found"] <= b or c["constructions"] == 0
    job = {"setting": name, "instance": 0}
    stream_rows = [r for r in at._stream_rows(job, parts) if r["budget_s"] in checkpoints]
    assert stream_rows == []  # the checkpoints are not budgets of the setting; rows exist only at its budgets
    budgets = at.budgets(get(name))
    parts = [at.run_stream(name, "val", 0, k, [budgets["0.5"], budgets["1"]]) for k in range(2)]
    rows = at._stream_rows(job | {"budget_s": 1.0}, parts)
    by = {(r["cores"], r["budget"]): r for r in rows}
    assert set(by) == {(1, "0.5"), (2, "0.5"), (1, "1"), (2, "1")}
    for label in ("0.5", "1"):
        assert by[2, label]["makespan"] <= by[1, label]["makespan"]
        assert by[1, label]["stream"] == 0


def test_gap_table_and_ratios(tmp_path):
    """Gap to BKS, share of the constructor-to-BKS gap closed (ratio of means) and the late rule."""
    at = _script("anytime")
    name = "SA-BT-25-5-20"

    def row(method, cores, label, i, ms, wall=None):
        b = at.budgets(get(name))[label]
        return {"method": method, "cores": cores, "budget": label, "budget_s": b, "instance": i, "makespan": ms,
                "success": True, "wall_s": b if wall is None else wall}

    rows = {}
    for r in [row("ALNS2", 8, "B1", 0, 10.5), row("ALNS2", 8, "B1", 1, 21.0),
              row("CONSTRUCT", 8, "B1", 0, 12.0), row("CONSTRUCT", 8, "B1", 1, 24.0),
              row("CPSAT", 8, "B1", 0, 11.0), row("CPSAT", 8, "B1", 1, 22.0, wall=100.0)]:  # instance 1 late
        rows.setdefault((r["method"], r["cores"], r["budget"]), {})[r["instance"]] = r
    g = {(x["method"], x["budget"]): x for x in at.gap_table(name, rows, {0: 10.0, 1: 20.0})}
    assert g["ALNS2", "B1"]["gap_pct"] == pytest.approx(5.0)
    assert g["ALNS2", "B1"]["closed"] == pytest.approx((18.0 - 15.75) / (18.0 - 15.0))
    assert g["CPSAT", "B1"]["on_time"] == 1 and g["CPSAT", "B1"]["n"] == 2
    assert g["CPSAT", "B1"]["gap_pct"] == pytest.approx(10.0)
    rat = {r["other"]: r for r in at.ratios(name, rows)}
    assert rat["CONSTRUCT-8@B1"]["ratio"] == pytest.approx(0.875) and rat["CONSTRUCT-8@B1"]["wins"] == 2
    assert "CPSAT-8@B1" not in rat  # one on-time pair only


def test_bks_table_roundtrip(tmp_path, monkeypatch):
    bks = _script("bks")
    monkeypatch.setattr(bks, "OUT", tmp_path)
    (tmp_path / "val_bks.json").write_text(json.dumps({"S": {"3": {"bks": 1.5}}}))
    assert bks.load_bks() == {"S": {3: 1.5}}
    assert np.isfinite(bks.SECONDS[500]) and bks.sub_time("MA-AT-150-10-500") == 10.0


def test_disturbed_rows_are_left_out(tmp_path, monkeypatch):
    """Rows whose job saw more than 40% throttled CPU periods are re-run and left out of the report tables."""
    at = _script("anytime")
    assert not at.disturbed({"nr_periods": 100, "nr_throttled": 40}) and at.disturbed({"nr_periods": 100,
                                                                                        "nr_throttled": 41})
    assert not at.disturbed({"wall_s": 1.0})  # rows without the counters
    fp = runtime.instance_fingerprint("SA-BT-25-5-20", "val", 0)
    if fp is None:
        pytest.skip("val not generated")
    base = {"setting": "SA-BT-25-5-20", "split": "val", "instance": 0, "fingerprint": fp, "method": "ALNS2",
            "cores": 1, "budget": "1", "budget_s": 1.0, "success": True, "wall_s": 1.0, "nr_periods": 10}
    rows = [base | {"makespan": 10.0, "nr_throttled": 0}, base | {"makespan": 9.0, "nr_throttled": 9}]
    monkeypatch.setattr(at, "OUT", tmp_path)
    (tmp_path / "SA-BT-25-5-20.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert at.load_rows("SA-BT-25-5-20")[("ALNS2", 1, "1")][0]["makespan"] == 10.0
    assert ("SA-BT-25-5-20", 0, "ALNS2", 1, "1") in at._done()
