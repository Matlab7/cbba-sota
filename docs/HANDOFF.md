# Handoff (2026-09-28): continue on a new server

Work paused here on purpose; the next phase starts on a new server. Everything needed is in this repository except
the raw run rows (`runs/`, gitignored) and the generated instances (`data/`, gitignored, regenerable bit for bit).

## 1. Where things stand

| Track | Status | Key evidence |
| --- | --- | --- |
| **S, static planner (the paper's contribution)** | ALNS v2 done; C1/C2 hold on the untuned `val` split for 6 of 8 H1 settings; not frozen | `docs/headroom-2026-09.md`, `docs/results/phase1b/`, `docs/results/alns-v2/` |
| **D, dynamic / degraded networks** | Week 1 done. K1 and K6 fired, so there is no dynamic-method claim and C3 is descriptive; Track D is an analysis appendix | `docs/trackD-week1-memo.md`, `docs/results/trackD-week1/` |

Headline numbers (dev/val, not yet the confirmatory test run):
- **ALNS v2 vs competitors, 8 cores at B1 (val):**
  - ALNS v2 is 0.4-1.4% above the best of our own runs.
  - Paired ratios: vs CP-SAT-LNS 0.882-0.953, vs the published RL 0.716-0.787, vs constructor restarts 0.882-0.933.
  - Every pair won; Holm p <= 0.014.
- **C2, ALNS v2 on 1 core at 2 s vs CP-SAT-LNS on 8 cores at B1:** 0.930-0.982.
  - It ties on SA-AT-50-5-50.
  - It loses there to 8-core constructor restarts (1.015).
- **Track D (dev):**
  - SPARC beats every learned or optimizer baseline: RL online 0.914, RL-MPC 0.906, rolling CP-SAT-LNS Heavy 0.919.
  - It is only 1.5% better than a rolling constructor at equal CPU (K1 fires).
  - On partitioned networks the one-line rule INF-r is the most robust (K6 fires).

Decision records: `docs/research/decision-2026-09-27.md` (arena choice), `docs/prereg-phase1.md` (draft, not frozen), `docs/trackD-spec.md`, `docs/plan-phase1b.md` (must-do list before freezing).

## 2. Setting up the new server

Reference versions from the old server:
- Python 3.12.3, uv 0.12.19.
- Packages: torch 2.8.0, numpy 2.5.3, numba 0.67.0, ortools 9.15.6755, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, pyyaml 6.0.3, natsort 8.4.0, pytest 9.1.1, ruff 0.16.9.
- GPUs: 4x H100. A GPU is only used for the RL baseline's lockstep sampler.
- Host: 96-CPU cgroup quota.

```bash
git clone https://github.com/Matlab7/cbba-sota && cd cbba-sota
git config user.name Matlab7 && git config user.email crdhsh@gmail.com
# push auth: gh CLI logged in as Matlab7, then
git config credential.helper "" && git config --add credential.helper '!gh auth git-credential'

uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -e '.[dev]' "torch==2.8.0" numba ortools matplotlib pandas pyyaml natsort
scripts/setup_third_party.sh   # HeteroMRTA @ db51e29 + numpy-2 patch; Sadcher (no licence: run only, never redistribute)

# instances: test/dev/val are regenerated from their seeds; --check verifies fingerprints and seed regeneration
.venv/bin/python scripts/gen_instances.py --splits test dev val --procs 32 --check

# optional: restore the raw run rows from the archive made on the old server (see Section 4)
tar -xzf cbba-sota-runs-2026-09-28.tar.gz

OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m pytest tests/ -q    # 333 passed on the old server
.venv/bin/ruff check cbba_sota scripts tests
```

Notes:
- The first test run creates `~/.cache/cbba-sota/pinned-4cf5e04`, a git archive of the kernel that Track D pins.
- The released HeteroMRTA checkpoint loads with `weights_only=False` under numpy 2. It is the official file, so it is trusted.
- If any `data/hetero/**` file differs after regeneration, stop: every stored row carries an instance fingerprint, and the report scripts drop rows whose fingerprint does not match.

## 3. Next steps, in order

1. **Harden the static claims** (`docs/plan-phase1b.md`, section "Before freezing the static claims"):
   - Add a properly parallel CP-SAT-LNS: 8 concurrent single-thread sub-solves, or CP-SAT's own LNS workers on the full model, hinted with the best of 8 constructions. Report the CPU seconds each method used.
   - Add one external method: CTAS-D with an open MIP solver, or a published synchronization or coalition metaheuristic.
   - Add pinning to `scripts/run_alns.py`, `scripts/alns_ablation.py` and `scripts/cpsat_lns_ref.py` (reuse `cbba_sota.bench.runtime`).
   - Run the two 500-task settings through `scripts/anytime.py` on `val`. They are the missing C1/C2 evidence.
   - Minor items:
     - Store the 1-based routes convention explicitly.
     - Give RL a first batch of 1 at low budgets.
     - Refuse timed campaigns on a dirty tree.
     - Add an independent reference on 50-task instances.
2. **Decide Track D's final form.**
   - Default: an analysis appendix, as the memo says.
   - Optional rescue: re-pin `cbba_sota/dyn` to the committed ALNS v2 (0e449fb), prespecify a new T1 (SPARC-Heavy or relaxed keys), and confirm it on `val`. This takes 1-2 days and has low odds.
3. **Freeze the pre-registration.**
   - Finalize `docs/prereg-phase1.md`:
     - budgets, competitors and the exclusion threshold for throttled rows;
     - disclose that the constructor's test-split rows were seen in Phase 1;
     - disclose that the RALTestSet is non-blind.
   - Write the solver commit hash into it. **The user commits the freeze.**
   - The runners refuse the test split until then (e.g. `compare_dev.py --split` excludes `test`). Enable it only in the frozen commit.
4. **Run the test split once** on a quiet, pinned host. Check load and throttling first, and keep nothing else running.
   - 8 H1 settings x 50 instances x competitors x {B1, low budgets}.
   - The 500-task cells dominate the cost (560 s x 8 cores per job). Plan about 1-2 days, or reduce the 500-task n in the prereg.
5. **Analysis, figures and the RA-L draft.**

## 4. Raw results archive

The archive is `/home/jovyan/dev/cbba-sota-runs-2026-09-28.tar.gz` on the old server, outside the repo (85 MB, sha256 `4d9d8cea6c9fe099b311e2f9052bfe71d4419e9be4b3f8e217807cb619836d55`). It contains `runs/` (raw JSONL rows, BKS and anytime campaigns, Track D rows). Copy it to the new server if you want to re-analyse old rows; nothing in Sections 3.1-3.5 requires it. The summary tables in `docs/results/` are already versioned.

## 5. Process lessons (keep)

- Never message a subagent that is running inside a Workflow. The message starts a second copy of the agent, and stopping it kills both. To change instructions, stop the workflow, edit the script and relaunch.
- Stop old workflow runs before relaunching; runs from an ended session can auto-resume.
- Kill the orphaned process groups of stopped agents.
- On a shared host, timed numbers are only valid with pinned lanes and a throttling filter (`cbba_sota/bench/runtime.py`, `scripts/anytime.py`). Budgets inside episodes must be deterministic (iterations), never wall-clock.
- Tune on `dev` only. `val` is for untuned dry runs; `test` is touched once, after the freeze.
