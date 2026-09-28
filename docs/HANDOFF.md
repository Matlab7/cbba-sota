# Handoff (updated 2026-09-28 evening): step 3.1 done on the new server

The first handoff (commit `00af86a`) moved the work to a new server. Step 3.1 ("harden the static claims") is now
done there: `docs/baselines-2026-09.md`. Everything needed is in this repository except the raw run rows (`runs/`,
gitignored; archived, Section 4) and the generated instances (`data/`, gitignored, regenerable bit for bit).

## 1. Where things stand

| Track | Status | Key evidence |
| --- | --- | --- |
| **S, static planner (the paper's contribution)** | Step 3.1 done: stronger competitors (parallel CP-SAT LNS, CP-SAT full-model portfolio, CTAS-D on an open solver), same-host val dry run of all 8 H1 settings incl. the 500-task ones, independent references. C1 holds on 8/8; not frozen | `docs/baselines-2026-09.md`, `docs/results/val-c1/`, `docs/results/ctas/` |
| **D, dynamic / degraded networks** | Week 1 done. K1 and K6 fired, so there is no dynamic-method claim and C3 is descriptive; Track D is an analysis appendix | `docs/trackD-week1-memo.md`, `docs/results/trackD-week1/` |

Headline numbers (val dry run on `dhcho-dev-2gpus-0`, not yet the confirmatory test run):
- **C1, ALNS v2 on 8 cores at B1 vs each competitor on 8 cores** (val 0-9; 500 tasks val 0-4):
  - vs CPSAT 0.885-0.968, PCPSAT 0.889-0.951, CPFULL 0.867-0.930, constructor restarts 0.885-0.934, RL 0.680-0.799.
  - CTAS-D solves 12 of 60 instances within B1 (not run at 500 tasks: no incumbent and 55-87 GB on the dev pilots).
  - Every pair won; all 46 tests Holm-significant (largest p 0.013). ALNS v2 ends 0.0-1.5% above the best known.
- **C2, ALNS v2 on 1 core at 2 s vs the competitors on 8 cores at B1:** holds on 4 of 8 settings; fails on
  SA-AT-50-5-50 and SA-BT-50-5-50 (0.977-0.995) and, at n = 5, on the 500-task settings (39 of 40 pairs won).
- **Parallel CP-SAT is not stronger:** PCPSAT uses all 8 cores and is within 0.02 of CPSAT everywhere.
- **Best known vs independent runs:** no run without an ALNS component matches the best ALNS-derived plan on any
  50- to 500-task val instance (2.8-12.0% worse on average); certified bounds 12-58% below.
- **Track D (dev):** SPARC only 1.5% better than a rolling constructor (K1 fires); INF-r most robust on partitioned
  networks (K6 fires).

Decision records: `docs/research/decision-2026-09-27.md` (arena choice), `docs/prereg-phase1.md` (draft, not frozen), `docs/trackD-spec.md`, `docs/plan-phase1b.md` (must-do list before freezing, with its status).

## 1a. Servers

The user's pods share `/home/jovyan` (repo, runs, data, venv) but not processes:
- `dhcho-dev-2gpus-0` (2 x H100, 57.6-CPU quota, 330 GB): the "new server". Quiet except the user's GPU keep-alive
  (`use-gpu.py`, ~2 CPUs). Every timed row of step 3.1 ran here; run the test split here too, at most 6 lanes.
- `dhcho-4gpu-0` (4 x H100, 96-CPU quota): the old server, busy with other projects. The last part of this session ran
  here (analysis and an untimed check only), after the controlling session moved pods; a campaign on the other pod kept
  running and was followed through its log in runs/logs.

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

1. ~~**Harden the static claims**~~: done, `docs/baselines-2026-09.md`. Left open from the review: the disclosure that
   constructor rows on the test split were seen in Phase 1 (goes into the prereg).
2. **Decide Track D's final form.**
   - Default: an analysis appendix, as the memo says.
   - Optional rescue: re-pin `cbba_sota/dyn` to the committed ALNS v2 (0e449fb), prespecify a new T1 (SPARC-Heavy or relaxed keys), and confirm it on `val`. This takes 1-2 days and has low odds.
3. **Freeze the pre-registration.** Proposed edits from step 3.1 are in `docs/baselines-2026-09.md`, Section 8
   (six competitors, the late-run rule, the host, the 500-task n, disclosures, Holm family size).
   - Finalize `docs/prereg-phase1.md`:
     - budgets, competitors and the exclusion threshold for throttled rows;
     - disclose that the constructor's test-split rows were seen in Phase 1;
     - disclose that the RALTestSet is non-blind.
   - Write the solver commit hash into it. **The user commits the freeze.**
   - The runners refuse the test split until then (e.g. `compare_dev.py --split` excludes `test`). Enable it only in the frozen commit.
4. **Run the test split once** on `dhcho-dev-2gpus-0`, pinned, from a clean worktree of the frozen commit
   (`scripts/campaign_worktree.sh`, `anytime.py run --grid c1 --split test` once the runners accept test). Check load
   and throttling first, and keep nothing else running.
   - 8 H1 settings x 50 instances x competitors x {B1, low budgets}.
   - The 500-task cells dominate the cost (560 s x 8 cores per job). Plan about 1-2 days, or reduce the 500-task n in the prereg.
5. **Analysis, figures and the RA-L draft.**

## 4. Raw results archive

Both archives are in `/home/jovyan/dev/`, outside the repo, on the volume both pods share:
- `cbba-sota-runs-2026-09-28.tar.gz` (85 MB, sha256 `4d9d8cea6c9fe099b311e2f9052bfe71d4419e9be4b3f8e217807cb619836d55`):
  `runs/` as of the first handoff (raw JSONL rows, BKS and anytime campaigns, Track D rows).
- `cbba-sota-runs-2026-09-28b.tar.gz` (1.9 MB, sha256 `8c022466097d32b9fb87bca985a40113b4eddc545c8bf58314cea9726bdc2ae2`):
  step 3.1's rows (`runs/anytime_c1`, `runs/tune_dev`, the updated `runs/bks` and `runs/logs`). Extract it over the
  first one.
The summary tables in `docs/results/` are versioned; `docs/results/val-c1/analyze.sh` regenerates step 3.1's from the rows.

## 5. Process lessons (keep)

- Never message a subagent that is running inside a Workflow. The message starts a second copy of the agent, and stopping it kills both. To change instructions, stop the workflow, edit the script and relaunch.
- Stop old workflow runs before relaunching; runs from an ended session can auto-resume.
- Kill the orphaned process groups of stopped agents.
- On a shared host, timed numbers are only valid with pinned lanes and a throttling filter (`cbba_sota/bench/runtime.py`, `scripts/anytime.py`). Budgets inside episodes must be deterministic (iterations), never wall-clock.
- Tune on `dev` only. `val` is for untuned dry runs; `test` is touched once, after the freeze.
- Run timed campaigns from a clean worktree of a fixed commit (`scripts/campaign_worktree.sh`, `PYTHONPATH` set to it),
  launched with `nohup setsid`; the runners refuse a dirty tree and a mismatched checkout.
- Keep lanes x 8 CPUs plus other load below the cgroup quota (57.6 CPUs on `dhcho-dev-2gpus-0`: 6 lanes at most), or
  whole campaigns get throttled.
- Stop a campaign by killing its setsid session (`pkill -s <SID>`), found from the setsid'd `bash` itself: a
  `ps | grep <script>` can return the launching shell's session instead, and two runs then overlap (happened once;
  the throttled rows were dropped and re-run).
- Never mix hosts in one comparison. Check `hostname` at the start of a session: the controlling session can move to
  another pod while a campaign keeps running on the first; follow it through its log in runs/logs.
- CTAS-D at 500 tasks holds 55-87 GB per solve; do not run several at once in the 330 GB container.
