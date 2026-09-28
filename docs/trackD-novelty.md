# Track D novelty check (gate K9)

Date: 2026-09-28. Scope: `docs/trackD-spec.md` Section 8, gate K9, and the day-1 reading list (DODRL-HMRTA, the
CSCWD 2026 fault-recovery paper, the RA-L 2026 LLM-MILP paper, Demiray et al., Calvo & Capitán).

**K9 question.** Does any HeteroMRTA citer already evaluate *dynamic HeteroMRTA* (its instances, generator or released
policy under online release, execution noise or failures) against a rolling optimizer? If so, drop the "first dynamic
HeteroMRTA study" framing, and add that paper as a baseline if its code exists.

## Decision

**K9 passes, but only for a narrow claim, with one open residual.**

- No HeteroMRTA citer found evaluates HeteroMRTA's benchmark (instances, generator or released RL policy) under
  online release, execution noise or failures, against a rolling optimizer or at all. This covers all 62 citers, and
  the five named papers are read below.
- The *problem class* is not new. DODRL-HMRTA (Neurocomputing 2026) is a dynamic heterogeneous-MRTA study on its own
  instances. It covers robot failures, task insertions, task cancellations and execution-time overruns, uses partial
  rescheduling, and compares with three metaheuristics.
  - Demiray et al. (C&IE 2026) is event-triggered ALNS re-optimization with synchronization (multi-team tasks).
  - Calvo & Capitán (T-RO 2025) repair or fully recompute heterogeneous coalition plans online, with MIT code.
  - The RA-L 2026 LLM-MILP paper does event-triggered MILP rescheduling after robot failures and urgent arrivals.
  - Any wording like "first dynamic heterogeneous / coalition MRTA study" or "first to re-plan coalitions online"
    is therefore false and must not be used.
- **Wording that survives.** "The first evaluation of the HeteroMRTA benchmark (RA-L 2025: its instance generator,
  settings and released policy) under online task release, execution delays and robot failures. The released
  policy runs online against rolling optimizers at equal per-event compute." Everything else is empirical (spec
  Section 0, risk 1).
- **Residual.** DODRL-HMRTA's full text could not be read (paywalled). Its abstract is in no open index (Crossref,
  OpenAlex, Semantic Scholar, Elsevier's API without a key, ScienceDirect: HTTP 403). The verdict rests on Google
  Scholar full-text snippets (quoted below).
  - The snippets show its own instances, named "HMRTA-TO". A Google Scholar full-text search for "HeteroMRTA"
    returns only two documents, and DODRL-HMRTA is not one of them.
  - What remains unknown is whether HMRTA-TO was generated with HeteroMRTA's generator.
  - This does not change K9: its baselines are GA, GA-PSO and ACO, not a rolling ALNS, CP-SAT or MILP on HeteroMRTA
    instances.
  - Before the freeze, library access or an email to the corresponding author should confirm this. The paper must
    not claim that no dynamic HMRTA learning work exists.
- **Baseline consequence.** None. No public code for DODRL-HMRTA was found (GitHub search, 2026-09-28: 0 results for
  "DODRL-HMRTA"), and its problem (HMRTA-TO) is not HeteroMRTA's. Calvo & Capitán's planner is MIT-licensed code, but
  it solves a different problem class (batteries, recharges, fragmentable and relayable tasks, deadlines). A port would
  be a reimplementation, so it stays related work, not a baseline. None of the five papers changes the Section 5.2
  competitor list.

## How the check was done

- **Citers of HeteroMRTA** (Dai et al., RA-L 2025, doi [10.1109/LRA.2025.3534682](https://doi.org/10.1109/LRA.2025.3534682)):
  - Semantic Scholar `citations` API: 42 citers, with citation contexts.
  - OpenAlex `cites:W4406857017`: 51 citers.
  - Union after de-duplication: 62 documents. Abstracts came from Semantic Scholar, OpenAlex and Crossref where
    available: 48 of 62 have one.
  - The 14 without an abstract were screened by title and venue, and by Google Scholar snippets where the title
    suggested dynamics (EJOR wildfire, ASOC amphibious, RA-L TALB-MAPPO, AuRo macroscopic ensembles, DODRL-HMRTA).
- **Benchmark usage.** Google Scholar full-text search for `"HeteroMRTA"` (2026-09-28) returns 2 documents: SADCHER
  (MRS 2025, static, uses HeteroMRTA as a baseline) and the HeteroMRTA paper itself. Semantic Scholar keyword search
  for "HeteroMRTA" (titles and abstracts): 0.
- **Reading.**
  - Full text of Demiray et al. (arXiv v1) and Calvo & Capitán (arXiv v3).
  - Google Scholar full-text snippets and the Crossref reference list of DODRL-HMRTA (no abstract is accessible).
  - The abstract and Google Scholar full-text snippets of the CSCWD paper.
  - The full abstract of the RA-L LLM-MILP paper.
  - Metadata of all five checked on Crossref.
- **Not accessible.** ScienceDirect, Springer and IEEE Xplore landing pages (403, login redirect or JS challenge from
  this host). Scholar and arXiv search rate-limited us intermittently, and we backed off.

## The five named papers

### 1. DODRL-HMRTA (Zeng, Fang, Gao, Wang, Yan, Wang; Neurocomputing, Dec 2026)

- **Record.** doi [10.1016/j.neucom.2026.135060](https://doi.org/10.1016/j.neucom.2026.135060) (Crossref: 39
  references; OpenAlex online 2026-09-08). Not open access. Crossref, OpenAlex and Semantic Scholar carry no abstract.
- **Content from Google Scholar full-text snippets** (verbatim fragments, retrieved 2026-09-28):
  - "We named this method as Dynamic Optimization based on DRL for HMRTA, ie DODRL-HMRTA."
  - "… uncertainties related to tasks and robots: robot failures, task insertions, task cancellations, and task
    execution time [overruns] …"
  - "… we also propose a partial rescheduling algorithm based on dynamic scheduling points …"
  - "… the proposed MDP transforms the HMRTA-TO problem into a sequence of robot-centered allocation decisions."
  - "… heuristic actions that constitute the action space. These heuristics include three types of priority
    rules …"
  - "… compare the performance of the proposed algorithm with three metaheuristic algorithms across three problem
    scales …" and "GA, GA-PSO, and ACO were selected as representative metaheuristic baselines …"
  - "Experimental results show that our framework reduces runtime by 2–3 orders of magnitude compared to
    metaheuristic algorithms while achieving 92% of their optimal solution quality."
  - "Average performance Comparison on the same HMRTA-TO test instances."
- **Reference list (Crossref).** It cites HeteroMRTA (Dai 2025), CBBA, sequential single-item auctions, DQN / double
  DQN / dueling DQN, GA / PSO / ACO MRTA papers, and a 2025 Neurocomputing DRL paper on dynamic heterogeneous robot
  scheduling. It cites no ALNS, LNS, CP-SAT, rolling-horizon or D-ITAGS work.
- **Reading.** A DQN-family hyper-heuristic that picks priority rules, with partial rescheduling on its own
  HMRTA-TO instances. Its quality is below the metaheuristics (92%), and its selling point is runtime.
  - It is dynamic heterogeneous MRTA prior art, so it must be cited, and "first dynamic HMRTA" is dead.
  - It does not use HeteroMRTA instances or policy per the available evidence.
  - Its baselines are metaheuristics, not a rolling ALNS, CP-SAT or MILP on HeteroMRTA. It does not satisfy K9's
    condition.
  - No code was found.

### 2. CSCWD 2026 fault-recovery paper (Hu & Ma)

- **Record.** "A Proactive Fault-Recovery Framework for Multi-Robot Task Allocation Using Successor Pre-Assignment",
  CSCWD 2026, doi [10.1109/CSCWD68734.2026.11582118](https://doi.org/10.1109/CSCWD68734.2026.11582118). Abstract via
  Semantic Scholar.
- **Content.**
  - It contrasts "Native fault-tolerant" and "Integrated" architectures, and proposes SPA-CBPA: consensus-based
    allocation that pre-assigns successor robots during consensus, so a failed robot's tasks transfer without
    re-auctioning.
  - Its evaluation is warehouse simulations. The metrics are recovery latency, recovery message overhead, task
    success and makespan ("near-zero recovery latency … at the cost of a modest increase in makespan").
  - Scholar snippet: "We consider a multi-robot task allocation (MRTA) problem in which a team of robots would execute
    a set of spatially distributed tasks in a warehouse environment."
  - Its HeteroMRTA citation (Semantic Scholar context) supports a metric definition only.
- **Verdict.** It concerns failures, but uses consensus auctions in a warehouse. There is no HeteroMRTA benchmark,
  no coalition makespan and no rolling optimizer. It does not trigger K9.
- **Useful.** It is related work for F3 (failure recovery) and for the CBBA-family baselines (B6).

### 3. RA-L 2026 LLM-MILP paper (Chen, Peng, Zhang, Huang, Li, Gao)

- **Record.** "An LLM Based Framework for Automated MILP Modeling in Dynamic Multi-Robot Task Scheduling", RA-L,
  Oct 2026, doi [10.1109/LRA.2026.3723332](https://doi.org/10.1109/LRA.2026.3723332).
- **Content** (abstract).
  - Aircraft-skin fabrication. The fleet must react to "robot failures and urgent task arrivals".
  - LLMs regenerate MILP constraints, and "an event-triggered mechanism regenerates only the affected constraints".
  - The case study "matches the schedule quality of baselines".
  - HeteroMRTA is cited only as a learning-based approximate method (context: "genetic algorithms [12] and
    learning-based methods [13], [14] offer faster solutions").
- **Verdict.** It is an event-triggered rolling MILP, but on an industrial scheduling case study, not HeteroMRTA. It
  does not trigger K9.
- **Useful.** It is the second 2026 precedent (with Demiray) for event-triggered re-optimization under failures and
  arrivals. Structural-event triggers (spec Section 4.2) are therefore not novel.

### 4. Demiray, Tolga, Yücel: dynamic multi-skill workforce scheduling and routing with synchronization

- **Record.**
  - arXiv [2309.09321](https://arxiv.org/abs/2309.09321) (v1, 2023-09-17; full text read).
  - Journal version: C&IE, Sept 2026, doi [10.1016/j.cie.2026.112203](https://doi.org/10.1016/j.cie.2026.112203).
    Its title adds "and anticipated tasks". It was not read (paywalled), so the anticipation component is unverified.
  - It does not cite HeteroMRTA (Crossref reference list, 43 entries).
- **Content** (arXiv full text).
  - A centralized "real-time optimization framework triggered upon the arrival of new tasks or the elapse of a set
    time". It keeps a frozen period, and synchronization is needed "when task demands exceed a team's capabilities".
  - Re-optimization uses a MILP (CPLEX, 15 min limit) or an ALNS.
  - Experiments vary the re-optimization trigger β_task (re-optimize after 1, 3 or 5 new tasks), the frozen-period
    length and the degree of dynamism, on 30-75 tasks.
  - β_task = 1 (re-optimize on every arrival) gives the lowest weighted throughput time in 39 of the 40
    super-instance rows of their Tables 5-6, at a higher CPU cost (e.g. I(30,2,0.2): 14445 vs 15369 vs 17001). The
    exception is loose I(40,3,0.6): 13839 vs 12247 vs 19372.
  - Shorter frozen periods are better.
  - CPU per simulated day runs from seconds to hours, so it is not real-time at our scale.
- **Verdict.** This is the closest OR precedent for "event-triggered ALNS with synchronized multi-resource tasks"
  (the spec already lists it as risk 1). It is not HeteroMRTA and not K9.
- **Consequences for the spec.**
  - The degree-of-dynamism citation the spec asks to verify (Section 3.3, R2/R3) is **Lund et al. (1996)**,
    δ = n_dynamic / n_total. Larsen (2001) defined the effective degree of dynamism, δ_e = mean(a_i / τ_max). Both
    are verified as quoted in Demiray et al. Section 2.
  - Our R2/R3 use Lund's δ (0.5 and 0.8). Their Larsen δ_e is about δ/2 for uniform arrivals on [0, H], taking
    τ_max = H.
  - Their β_task = 1 result agrees with our S8 (re-plan on every structural event beats batching). It is a
    supporting citation, not a novelty threat.

### 5. Calvo & Capitán, T-RO 2025

- **Record.** "Heterogeneous Multirobot Task Allocation for Long-Endurance Missions in Dynamic Scenarios", IEEE
  T-RO 2025, doi [10.1109/TRO.2025.3626651](https://doi.org/10.1109/TRO.2025.3626651); arXiv
  [2411.02062](https://arxiv.org/abs/2411.02062) (v3, 2025-11-26; full text read). Code:
  [github.com/multirobot-use/mrta_heuristic_planner](https://github.com/multirobot-use/mrta_heuristic_planner) (MIT,
  last push 2025-11-22). It does not cite HeteroMRTA (Crossref reference list, 55 entries).
- **Content.**
  - A MILP and a heuristic for heterogeneous MRTA with coalitions of fixed, variable or unspecified size.
  - It also covers recharges, fragmentable and relayable tasks, and deadlines. The objective is makespan plus
    deadline delay, synchronization waiting and coalition-size deviation.
  - Online, it repairs on robot delays (absorbing the delay through waiting slack), and recomputes the full plan on
    failures, new tasks or repair infeasibility.
  - Table III compares Repair, Replanning and Combined under short and long delays. Success rates are 89.5 / 23 %
    (repair), 100 / 45.5 % (replanning) and 100 / 45.5 % (combined). The objective increase is 3.5-84%.
  - Instances are its own solar-plant UAV inspection scenarios, with a MILP optimum on small instances.
- **Verdict.** It is centralized online repair and recompute of heterogeneous coalition plans, i.e. prior art for
  SPARC's plan-then-repair structure. It is not HeteroMRTA and not K9.
- **Useful.** It supports the spec's decision not to claim a new re-planning architecture, and D-ITAGS-style repair
  (B5) plus rolling re-solve as baselines.

## Screen of all 62 HeteroMRTA citers (Semantic Scholar ∪ OpenAlex, 2026-09-28)

- **Legend.**
  - **dyn**: the paper has online arrivals, failures or execution uncertainty.
  - **HB**: it uses HeteroMRTA's instances, generator or policy.
  - **RO**: it compares with a rolling or re-planning optimizer.
  - **K9 hit** needs dyn, HB and RO.
- **Source codes.** abs = abstract read; ctx = Semantic Scholar citation context; t = title and venue only;
  gs = Google Scholar snippet.

| Date | Title (short) | DOI / id | Read | dyn | HB | RO | K9 |
|---|---|---|---|---|---|---|---|
| 2025 | LiDAR-based exploration via DRL (TIM) | 10.1109/tim.2025.3584117 | abs | – | – | – | no |
| 2025 | Multiagent Reinforcement Learning (encyclopedia entry) | 10.1007/978-3-642-41610-1_240-1 | t | – | – | – | no |
| 2025-03 | HIPPO-MAT (arXiv) | 10.48550/arxiv.2503.07662 | abs | yes | – | – | no |
| 2025-03 | STALC (T-RO) | 10.1109/tro.2026.3686241 | abs | – | – | – | no |
| 2025-05 | Hyper-SAMARL (ICRA) | 10.1109/icra55743.2025.11128092 | abs | yes | – | – | no |
| 2025-06 | TinyML workload distribution (Cluster Comput.) | 10.1007/s10586-025-05289-x | t | – | – | – | no |
| 2025-06 | LGTC-IPPO resource allocation (RA-L) | 10.1109/lra.2025.3581126 | abs | yes | – | – | no |
| 2025-06 | AI-driven chemistry lab (Appl. Sci.) | 10.3390/app15137387 | abs | – | – | – | no |
| 2025-07 | Unified decentralized framework (IJASRET) | 10.65521/ijasret.v9i7.1550 | abs | yes | – | – | no |
| 2025-07 | CNN+DDQN human-robot allocation (Systems) | 10.3390/systems13080631 | abs | yes | – | – | no |
| 2025-08 | AMOEAD-HRM dynamic MO-MRTA (SWEVO) | 10.1016/j.swevo.2025.102123 | t, search summary | yes | – | vs DNSGA-II | no |
| 2025-09 | Multi-robot assembly planning (RAS) | 10.1016/j.robot.2025.105179 | abs | – | – | – | no |
| 2025-10 | DP-MACN diffusion policy (AIAC) | 10.1109/aiac68175.2025.11332435 | abs | yes | – | – | no |
| 2025-10 | HATA human-aware allocation (IROS) | 10.1109/iros60139.2025.11246701 | abs | yes | – | – | no |
| 2025-10 | PDTA search and rescue (TNSE) | 10.1109/tnse.2025.3624250 | abs | yes | – | – | no |
| 2025-11 | Multi-view clustered TSP (RA-L) | 10.1109/lra.2025.3632724 | abs | – | – | – | no |
| 2025-11 | Human motion symmetry survey (JoCC) | 10.1186/s13677-025-00810-4 | abs | – | – | – | no |
| 2025-12 | SADCHER (MRS) | 10.1109/mrs66243.2025.11357250 | abs, ctx | – (static, "real-time") | yes (baseline) | – | no |
| 2025-12 | MRTA/MAPF review (ASOC) | 10.1016/j.asoc.2025.114498 | t | – | – | – | no |
| 2025-12 | MPPI-GA UGV-UAV (Access) | 10.1109/access.2025.3649891 | abs | yes | – | – | no |
| 2026 | HierTask UAV routing (TNSE) | 10.1109/tnse.2026.3697704 | abs | – | – | – | no |
| 2026 | Dual event-triggered apprentice game (TASE) | 10.1109/tase.2026.3702993 | abs | – | – | – | no |
| 2026 | Hydraulic fracturing SRL (SSRN) | 10.2139/ssrn.6960220 | abs | – | – | – | no |
| 2026 | Swarm embodied cognition survey (TASE) | 10.1109/tase.2026.3708974 | abs | – | – | – | no |
| 2026 | Primate-inspired swarm CE&SG (Research) | 10.34133/research.1412 | abs | yes | – | – | no |
| 2026 | Air-ground-human disinfection (MAV) | 10.48130/mav-0026-0002 | abs | yes | – | – | no |
| 2026 | LGMTA learning-guided bidding, dynamic HMRTA (IET CSY) | 10.1049/csy2.70067 | abs | yes | – (own simulator) | – (market, CBBA) | no |
| 2026-01 | Human-like decentralised assignment (AIAA SciTech) | 10.2514/6.2026-0326 | abs | yes | – | – | no |
| 2026-01 | Super-agile satellite strip imaging RL (AIAA) | 10.2514/6.2026-0132 | abs | – | – | – | no |
| 2026-01 | CHORAL routing (arXiv) | 10.48550/arxiv.2601.10340 | abs | – | – | – | no |
| 2026-01 | Resilient MAS control (NNICE) | 10.1109/nnice68970.2026.11465656 | abs | – | – | – | no |
| 2026-03 | Collaboration taxonomy survey (arXiv) | 10.48550/arxiv.2603.23898 | abs | – | – | – | no |
| 2026-03 | IMD-TAPP multi-drone (arXiv) | 10.48550/arxiv.2603.24908 | abs | – | – | – | no |
| 2026-04 | Macroscopic ensemble allocation (AuRo) | 10.1007/s10514-025-10238-z | gs | yes | – | – | no |
| 2026-04 | Multi-task edge AI networks (JoCS) | 10.1016/j.jocs.2026.102877 | t | – | – | – | no |
| 2026-04 | DQL assembly lines (DEA) | 10.64972/dea.2026.v5i2.1734d:44-57 | abs | yes | – | – | no |
| 2026-04 | Energy-balanced IGA + local greedy rescheduling (Appl. Sci.) | 10.3390/app16094311 | abs | yes | – (MTSP) | – | no |
| 2026-05 | ARMATA (arXiv) | 10.48550/arxiv.2605.04225 | abs, ctx | – | – | – | no |
| 2026-05 | Dual-UGV carrying RL (Symmetry) | 10.3390/sym18050833 | abs | – | – | – | no |
| 2026-05 | HMOVIG flowshop (CSMS) | 10.23919/csms.2026.0004 | abs | – | – | – | no |
| 2026-05 | **SPA-CBPA fault recovery (CSCWD)** | 10.1109/cscwd68734.2026.11582118 | abs, gs, ctx | yes | – | – | no |
| 2026-05 | HADRL-VCS crowdsensing (TMC) | 10.1109/tmc.2026.3693470 | abs | – | – | – | no |
| 2026-05 | Multi-agent TAMP survey (AIR) | 10.1007/s10462-026-11588-5 | abs | – | – | – | no |
| 2026-05 | SAR image interpretation scheduling (Sensors) | 10.3390/s26113311 | abs | yes | – | – | no |
| 2026-05 | HierTask FANETs (ICC) | 10.1109/icc59461.2026.11588259 | abs | – | – | – | no |
| 2026-06 | Wildfire aircraft DRL (EJOR) | 10.1016/j.ejor.2026.06.040 | gs | yes | – (fire spread model) | unknown | no |
| 2026-06 | IAOM dynamic cooperative allocation (RA-L) | 10.1109/lra.2026.3703220 | abs | yes | – (grid rescue benchmarks) | "optimization-based" baselines | no |
| 2026-06 | Warehouse WMHS (Discover Comput.) | 10.1007/s10791-026-10236-4 | abs | yes | – | – | no |
| 2026-07 | Auction-consensus learned bidding (UR) | 10.1109/ur69537.2026.11626710 | t | ? | – | – | no |
| 2026-07 | Formation control PINN (SPIE) | 10.1117/12.3117220 | abs | – | – | – | no |
| 2026-07 | UAV battery DDQN (Electronics) | 10.3390/electronics15142984 | abs | – | – | – | no |
| 2026-07 | DRLCC humanoid concurrency (Appl. Sci.) | 10.3390/app16157388 | abs | yes | – | – | no |
| 2026-07 | AUV target capture (Ocean Eng.) | 10.1016/j.oceaneng.2026.127180 | abs | – | – | – | no |
| 2026-07 | TALB-MAPPO grinding (RA-L) | 10.1109/lra.2026.3719201 | gs | – | – | – | no |
| 2026-08 | **LLM-MILP dynamic scheduling (RA-L)** | 10.1109/lra.2026.3723332 | abs, ctx | yes | – | yes (event-triggered MILP) | no |
| 2026-08 | Swarm DGAT allocation (Springer chapter) | 10.1007/978-981-92-3082-2_25 | t | ? | – | – | no |
| 2026-08 | AUV search HMARL (EAAI) | 10.1016/j.engappai.2026.115891 | t | yes | – | – | no |
| 2026-08 | MBRL rollout, hospital requests (arXiv) | arXiv 2608.21554 | abs, ctx | yes | – | – (reactive, token-passing, greedy) | no |
| 2026-08 | Balance-Aware Pointerformer (RA-L) | 10.1109/lra.2026.3728333 | abs, ctx | – | – | – | no |
| 2026-09 | Amphibious robot teams HRL (ASOC) | 10.1016/j.asoc.2026.116413 | gs | yes | – | – | no |
| 2026-09 | **DODRL-HMRTA (Neurocomputing)** | 10.1016/j.neucom.2026.135060 | gs (full-text snippets), refs | yes | – (own HMRTA-TO) | metaheuristics (GA, GA-PSO, ACO) | no, residual |
| 2026-09 | Spatio-temporal team allocation (Computing) | 10.1007/s00607-026-01743-9 | t | – | – | – | no |

## Consequences for the paper and the spec

1. **Framing.**
   - Keep: "first evaluation of the HeteroMRTA benchmark and its released policy under online release, execution
     delays and failures, against rolling optimizers at equal per-event compute".
   - Drop any claim that dynamic heterogeneous or coalition MRTA, online coalition repair, or event-triggered
     re-optimization is new.
   - Cite DODRL-HMRTA, Calvo & Capitán, Demiray et al., the LLM-MILP paper and D-ITAGS as that prior art.
2. **Baselines.** No change to spec Section 5.2. DODRL-HMRTA has no public code and a different problem.
   Calvo & Capitán is MIT code for another problem class. Demiray has no released code (none linked in the arXiv
   version).
3. **Citation fix.** For R2/R3 in spec 3.3, cite Lund et al. (1996) for the degree of dynamism and Larsen (2001) for
   the effective degree of dynamism, both via Demiray et al.
4. **Open before freeze.** Get DODRL-HMRTA's full text (library or email to the authors) and confirm that HMRTA-TO is
   not HeteroMRTA's generator. If it is, the "first on HeteroMRTA" wording must be narrowed further to "first against
   rolling optimizers". The Section 5.2 competitor list is unaffected either way.
