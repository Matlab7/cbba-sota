"""State-start coalition ALNS kernels for rolling-horizon planning (Track D planner, SPARC).

Pinned copy: this file starts from ``cbba_sota/solvers/_alns_kernels.py`` at commit **4cf5e04** (ALNS v2, obtained
with ``git show 4cf5e04:cbba_sota/solvers/_alns_kernels.py``; docs/trackD-spec.md Section 8 "pin the v2 commit") and
from the pilot patches ``pilots/trackD/rh_kernels.py`` and ``pilots/trackD_robust/robust_rh_kernels.py``. Another
workflow edits ``cbba_sota/solvers`` concurrently; nothing here imports it.

Changes to the pinned kernel (each one is a no-op on a static instance, so the static case is the pinned kernel bit
for bit; ``scripts/trackD_kernel_check.py`` compares the two):

1. State start. Robot ``i`` is free at ``ready[i]``; its first leg to task ``t`` takes ``dout[i, t]`` (travel from
   its current position, predictor-scaled); a robot without tasks returns home at ``ready[i] + dhome[i]``
   (``dhome = -inf``: not counted, e.g. a failed robot). Static: ``ready = 0``, ``dout = da``, ``dhome = 0``.
2. Earliest start ``est[t]`` (task start >= est; releases for the hindsight reference). Static: ``-inf``.
3. Anchors. ``mode[t]``: 0 free, 1 anchored (a committed coalition: fixed members, fixed place in the key order,
   never removed), 2 anchored residual (a committed task whose fixed members no longer cover it after a failure
   or abandon: removable, reinserted only at its rank, its cover starts from the fixed members and only the extra
   members are pruned to a minimal cover). Anchored tasks form the prefix ``seq[:npre]`` in ``rank`` order; free
   tasks are inserted only after it, which is the key monotonicity of the spec (G1).
4. Head locks. ``head[i]`` is the anchored task robot ``i`` travels to or waits at; the insertion sweep treats the
   robot as unavailable for any task placed before its head.
5. ``removable[t]``: destroy and swap moves touch only these tasks (targeted repair restricts the set further).
6. Unplaced tasks. An insertion without a finite slot leaves the task out; ``fl[3]`` counts tasks outside the
   plan and every comparison is lexicographic (tasks placed, makespan, objective).
7. Dropped: ``crossover`` and ``kick`` (the elite pool is off, ``pool = 0``, in the pinned v2 default).

A solution is a tuple of arrays (see ``SOL``): the global task order ``seq[:n]`` (tasks outside it are removed and
have ``cnt == 0``), the coalitions ``mem[j, :cnt[j]]``, and its schedule, filled by ``schedule``. Every coalition is
built by ``cover`` as a minimal cover (on top of fixed members) and every robot visits its tasks in ``seq`` order.
``schedule`` also fills ``tail[t]``, the longest path from the start of ``t`` to the end.

Insertion sweeps the slots of ``seq`` once with incrementally updated arrivals and picks a greedy minimal cover by
(noisy) earliest arrival for each candidate slot. Its makespan follows in O(cover) from the tails (the schedule is
a longest-path problem; inserting ``j`` only adds paths through ``j``, and releases and state starts are extra
source edges), which bounds the objective; slots are then evaluated best-first by that bound with an incremental
suffix pass.
"""
from __future__ import annotations

import numpy as np
from numba import njit

EPS = 1e-9
INF = np.inf
PEN = 1e6  # objective penalty per task left outside the plan (lexicographic: placed tasks first)

# Instance tuple layout (see ``cbba_sota.dyn.planner.KernelInstance``).
INST = ("req", "ab", "dur", "tt", "da", "capl", "ncap", "near", "partners", "pstart", "group", "capf", "ready", "dout",
        "dhome", "est", "fmem", "fcnt", "mode", "rank", "removable", "head", "flags")
(I_REQ, I_AB, I_DUR, I_TT, I_DA, I_CAPL, I_NCAP, I_NEAR, I_PARTNERS, I_PSTART, I_GROUP, I_CAPF, I_READY, I_DOUT,
 I_DHOME, I_EST, I_FMEM, I_FCNT, I_MODE, I_RANK, I_REM, I_HEAD, I_FLAGS) = range(23)
M_FREE, M_ANCHOR, M_RESIDUAL = 0, 1, 2
F_RELAXED = 0  # flags[F_RELAXED] = 1: free tasks may interleave with anchored ones (key order of committed tasks and
#                head-first are kept); 0: anchored prefix, free tasks behind it (the spec's key monotonicity)

# Solution tuple layout.
SOL = ("seq", "mem", "cnt", "ni", "start", "finish", "barr", "pred", "ret", "last", "fl", "succ", "first", "lastk",
       "tail")
N_FL = 4  # fl = [makespan, sum of finish times, objective (penalised), tasks outside the plan]

# Destroy operators (the last two are moves without repair) and repair operators.
DESTROY = ("random", "shaw", "worst_wait", "critical", "route_segment", "tail_swap", "member_swap")
REPAIR = ("random", "random_noisy", "largest_first", "regret2")
D_SWAP, D_MSWAP = 5, 6
R_NOISY, R_LARGEST, R_REGRET = 1, 2, 3

# Parameter vector layout (float64).
PARAMS = ("lam", "noise", "q_lo", "q_hi", "max_slots", "shaw_p", "worst_p", "regret_max_q", "rho", "seg_len",
          "sigma_best", "sigma_better", "sigma_accept", "restart_iters", "q_stag", "q_big", "stop_on_stag")
(P_LAM, P_NOISE, P_QLO, P_QHI, P_SLOTS, P_SHAW, P_WORST, P_REGQ, P_RHO, P_SEG, P_S1, P_S2, P_S3, P_RESTART, P_QSTAG,
 P_QBIG, P_STOP) = range(17)

# Insertion counters (ws[7], int64): stamp id, exact slot evaluations, suffix steps, insertions, candidate slots.
COUNTERS = ("stamp", "exact_evals", "suffix_steps", "insertions", "candidate_slots")

# Stats vector layout (int64).
STATS = ("iterations", "improvements", "accepted", "restarts", "since_best", "seg_iter")
S_IT, S_IMP, S_ACC, S_RESTART, S_SINCE, S_SEG = range(6)


@njit(cache=True)
def seed(s):
    np.random.seed(s)


@njit(cache=True)
def schedule(inst, sol, lam):
    """Forward pass of ``sol`` in place (times, per-member arrival and route predecessor/successor); returns the
    makespan."""
    dur, tt, da = inst[2], inst[3], inst[4]
    ready, dout, dhome, est = inst[12], inst[13], inst[14], inst[15]
    seq, mem, cnt, ni, start, finish, barr, pred, ret, last, fl, succ, first, lastk, tail = sol
    T, A = dur.shape[0], da.shape[0]
    for i in range(A):
        ret[i] = ready[i]  # the robots' free time during the pass
        last[i] = -1
        first[i] = -1
    total = 0.0
    for p in range(ni[0]):
        t = seq[p]
        s = est[t]
        for k in range(cnt[t]):
            i = mem[t, k]
            lt = last[i]
            a = ret[i] + (dout[i, t] if lt < 0 else tt[lt, t])
            barr[t, k] = a
            pred[t, k] = lt
            if lt < 0:
                first[i] = t
            else:
                succ[lt, lastk[i]] = t
            s = max(s, a)
        f = s + dur[t]
        start[t] = s
        finish[t] = f
        total += f
        for k in range(cnt[t]):
            i = mem[t, k]
            ret[i] = f
            last[i] = t
            lastk[i] = k
    ms = 0.0
    for i in range(A):
        if last[i] >= 0:
            succ[last[i], lastk[i]] = -1
        ret[i] = ret[i] + da[i, last[i]] if last[i] >= 0 else ready[i] + dhome[i]
        ms = max(ms, ret[i])
    out = T - ni[0]
    fl[0] = ms
    fl[1] = total
    fl[2] = ms + lam * total / T + PEN * out
    fl[3] = out
    for p in range(ni[0] - 1, -1, -1):
        t = seq[p]
        b = 0.0
        for k in range(cnt[t]):
            u = succ[t, k]
            b = max(b, tt[t, u] + tail[u] if u >= 0 else da[mem[t, k], t])
        tail[t] = dur[t] + b
    return ms


@njit(cache=True)
def copy_sol(src, dst):
    dst[0][:] = src[0]
    dst[1][...] = src[1]
    dst[2][:] = src[2]
    dst[3][0] = src[3][0]
    dst[4][:] = src[4]
    dst[5][:] = src[5]
    dst[6][...] = src[6]
    dst[7][...] = src[7]
    dst[8][:] = src[8]
    dst[9][:] = src[9]
    dst[10][:] = src[10]
    dst[11][...] = src[11]
    dst[12][:] = src[12]
    dst[13][:] = src[13]
    dst[14][:] = src[14]


@njit(cache=True)
def cover(j, req, ab, capl, ncap, arr, key, need, chosen, noise, fmem, fcnt):
    """Greedy minimal cover of task ``j`` by (noisy) earliest arrival ``arr``; members in ``chosen[:c]``.

    The fixed members ``fmem[j, :fcnt[j]]`` come first and are always kept. Then contributing candidates are added
    in key order until the requirement is met, and redundant extras are dropped, latest arrival first (one pass
    suffices since dropping only shrinks the surplus). Returns ``c`` (-1 if infeasible). Candidates with an
    infinite arrival are never chosen.
    """
    K = req.shape[1]
    for k in range(K):
        need[k] = req[j, k]
    fc = fcnt[j]
    for a in range(fc):
        i = fmem[j, a]
        chosen[a] = i
        for k in range(K):
            need[k] -= ab[i, k]
    nc = ncap[j]
    for x in range(nc):
        i = capl[j, x]
        key[x] = arr[i] * (1.0 + noise * np.random.random()) if noise > 0 else arr[i]
    c = fc
    while True:
        done = True
        for k in range(K):
            if need[k] > 0:
                done = False
                break
        if done:
            break
        bx = -1
        bk = INF
        for x in range(nc):
            if key[x] < bk:
                i = capl[j, x]
                useful = False
                for k in range(K):
                    if need[k] > 0 and ab[i, k] > 0:
                        useful = True
                        break
                if useful:
                    bk = key[x]
                    bx = x
                else:
                    key[x] = INF  # needs only shrink, so it never becomes useful again
        if bx < 0:
            return -1
        i = capl[j, bx]
        key[bx] = INF
        chosen[c] = i
        c += 1
        for k in range(K):
            need[k] -= ab[i, k]
    for a in range(fc + 1, c):  # sort the extras by arrival, latest first
        v = chosen[a]
        b = a - 1
        while b >= fc and arr[chosen[b]] < arr[v]:
            chosen[b + 1] = chosen[b]
            b -= 1
        chosen[b + 1] = v
    m = fc
    for a in range(fc, c):
        i = chosen[a]
        removable = True
        for k in range(K):
            if ab[i, k] > -need[k]:  # contribution exceeds the surplus
                removable = False
                break
        if removable:
            for k in range(K):
                need[k] += ab[i, k]
        else:
            chosen[m] = i
            m += 1
    return m


@njit(cache=True)
def insertion(j, inst, sol, ws, slot_ok, noise, lam, need_second, best_c):
    """Best slot and coalition for removed task ``j`` in ``sol`` (whose schedule must be current).

    Phase 1 sweeps the order once, updating the robots' arrivals at ``j`` incrementally, and builds each marked
    slot's cover and a lower bound on its objective from the delays the cover causes at its members' next tasks.
    Phase 2 evaluates the slots best-first by that bound with the incremental suffix pass and stops once the bound
    reaches the incumbent (the second best for regret).
    Returns ``(p, obj, ms, c, second_obj)``: insert before ``seq[p]`` with members ``best_c[:c]``; ``p = -1`` if no
    slot is feasible.
    """
    req, ab, dur, tt, da, capl, ncap, capf = inst[0], inst[1], inst[2], inst[3], inst[4], inst[5], inst[6], inst[11]
    ready, dout, est, fmem, fcnt, head = inst[12], inst[13], inst[15], inst[16], inst[17], inst[21]
    seq, mem, cnt, ni, start, finish, barr, ret, fl, succ, first, tail = (sol[0], sol[1], sol[2], sol[3], sol[4],
                                                                           sol[5], sol[6], sol[8], sol[10], sol[11],
                                                                           sol[12], sol[14])
    arr, key, need, chosen, stamp, ofree, olast, ctr, cp, cf, clb, cms, ccnt, cmem, lastp, lastkp, nxt = ws
    T, A = dur.shape[0], da.shape[0]
    n = ni[0]
    base_ms, base_sum = fl[0], fl[1]
    scale = lam / T
    for x in range(ncap[j]):
        i = capl[j, x]
        arr[i] = INF if head[i] >= 0 and head[i] != j else ready[i] + dout[i, j]
        lastp[i] = -1
    for x in range(fcnt[j]):
        i = fmem[j, x]
        arr[i] = INF if head[i] >= 0 and head[i] != j else ready[i] + dout[i, j]
        lastp[i] = -1
    ctr[3] += 1  # insertions
    nc = 0
    for p in range(n + 1):
        if slot_ok[p]:
            c = cover(j, req, ab, capl, ncap, arr, key, need, chosen, noise, fmem, fcnt)
            if c > 0:
                s = est[j]
                for a in range(c):
                    s = max(s, arr[chosen[a]])
                if s < INF:
                    f = s + dur[j]
                    lb = base_ms  # the new makespan, exact up to rounding
                    for a in range(c):
                        i = chosen[a]
                        t = first[i] if lastp[i] < 0 else succ[lastp[i], lastkp[i]]
                        nxt[a] = t
                        lb = max(lb, f + (da[i, j] if t < 0 else tt[j, t] + tail[t]))
                    delay = 0.0
                    for a in range(c):
                        t = nxt[a]
                        if t < 0:
                            continue
                        dup = False
                        for b in range(a):
                            if nxt[b] == t:
                                dup = True
                        if dup:
                            continue
                        d = 0.0
                        for b in range(a, c):
                            if nxt[b] == t:
                                d = max(d, f + tt[j, t] - start[t])
                        delay += d
                    cp[nc], cf[nc], cms[nc], ccnt[nc] = p, f, lb, c
                    clb[nc] = lb + scale * (base_sum + f + delay)
                    cmem[nc, :c] = chosen[:c]
                    nc += 1
        if p < n:
            t = seq[p]
            for k in range(cnt[t]):
                i = mem[t, k]
                if capf[j, i]:
                    if head[i] == j:  # a robot locked to j must visit j first
                        arr[i] = INF
                    else:
                        arr[i] = finish[t] + tt[t, j]
                        lastp[i] = t
                        lastkp[i] = k
    ctr[4] += nc  # candidate slots
    best_obj = INF
    second = INF
    best_p = -1
    best_ms = INF
    best_n = 0
    for o in np.argsort(clb[:nc]):
        thr = (second if need_second else best_obj) + EPS
        if clb[o] >= thr:
            break
        p, f, lb, c = cp[o], cf[o], cms[o], ccnt[o]
        dsum = f
        ctr[1] += 1  # slots evaluated
        ctr[0] += 1
        sid = ctr[0]
        for a in range(c):
            i = cmem[o, a]
            stamp[i] = sid
            ofree[i] = f
            olast[i] = j
        npert = c
        pruned = False
        for q in range(p, n):
            t = seq[q]
            ctr[2] += 1  # suffix steps
            hit = False
            s2 = est[t]
            for k in range(cnt[t]):
                i = mem[t, k]
                if stamp[i] == sid:
                    hit = True
                    s2 = max(s2, ofree[i] + tt[olast[i], t])
                else:
                    s2 = max(s2, barr[t, k])
            if not hit:
                continue
            if s2 == start[t]:  # members leave t exactly as in the baseline
                for k in range(cnt[t]):
                    i = mem[t, k]
                    if stamp[i] == sid:
                        stamp[i] = 0
                        npert -= 1
                if npert == 0:
                    break
            else:
                f2 = s2 + dur[t]
                dsum += f2 - finish[t]
                for k in range(cnt[t]):
                    i = mem[t, k]
                    if stamp[i] != sid:
                        stamp[i] = sid
                        npert += 1
                    ofree[i] = f2
                    olast[i] = t
                if lb + scale * (base_sum + dsum) >= thr:
                    pruned = True
                    break
        if pruned:
            continue
        ms = base_ms
        if npert > 0:
            ms = 0.0
            for i in range(A):
                ms = max(ms, ofree[i] + da[i, olast[i]] if stamp[i] == sid else ret[i])
        obj = ms + scale * (base_sum + dsum)
        if obj < best_obj:
            second = best_obj
            best_obj, best_p, best_ms, best_n = obj, p, ms, c
            best_c[:c] = cmem[o, :c]
        elif obj < second:
            second = obj
    return best_p, best_obj, best_ms, best_n, second


@njit(cache=True)
def apply_insertion(j, p, members, c, sol):
    seq, mem, cnt, ni = sol[0], sol[1], sol[2], sol[3]
    n = ni[0]
    for q in range(n, p, -1):
        seq[q] = seq[q - 1]
    seq[p] = j
    ni[0] = n + 1
    mem[j, :c] = members[:c]
    cnt[j] = c


@njit(cache=True)
def prefix_len(inst, sol):
    """Number of anchored tasks at the front of ``seq`` (contiguous there in monotone mode); 0 in relaxed mode."""
    if inst[22][F_RELAXED]:
        return 0
    mode = inst[18]
    seq, ni = sol[0], sol[3]
    p = 0
    while p < ni[0] and mode[seq[p]] != M_FREE:
        p += 1
    return p


@njit(cache=True)
def mark_slots(j, inst, sol, slot_ok, posn, max_slots):
    """The slots task ``j`` may take: for an anchored residual, between the anchored tasks of the neighbouring
    ranks (its rank position in monotone mode); otherwise every slot after the anchored prefix (every slot in
    relaxed mode) or, when there are more than ``max_slots``, the slots around the nearest present tasks."""
    near, mode, rank = inst[7], inst[18], inst[19]
    seq, ni = sol[0], sol[3]
    n = ni[0]
    lo = prefix_len(inst, sol)
    slot_ok[:n + 1] = False
    if mode[j] != M_FREE:
        a, b = 0, -1
        for p in range(n):
            t = seq[p]
            if mode[t] != M_FREE:
                if rank[t] < rank[j]:
                    a = p + 1
                elif b < 0:
                    b = p
        if b < 0:
            b = n if inst[22][F_RELAXED] else max(a, lo)
        slot_ok[a:b + 1] = True
        return
    if n + 1 - lo <= max_slots:
        slot_ok[lo:n + 1] = True
        return
    for p in range(lo, n):
        posn[seq[p]] = p
    slot_ok[lo] = True
    slot_ok[n] = True
    marked = 2
    for x in range(near.shape[1]):
        t = near[j, x]
        p = posn[t]
        if p < 0:
            continue
        for pp in (p, p + 1):
            if not slot_ok[pp]:
                slot_ok[pp] = True
                marked += 1
        if marked >= max_slots:
            break
    for p in range(lo, n):
        posn[seq[p]] = -1


@njit(cache=True)
def insert_one(j, inst, sol, ws, slot_ok, posn, best_c, noise, lam, max_slots):
    """Insert ``j`` at its best slot; returns False (and leaves ``j`` out) if no slot is feasible."""
    mark_slots(j, inst, sol, slot_ok, posn, max_slots)
    p, _, _, c, _ = insertion(j, inst, sol, ws, slot_ok, noise, lam, False, best_c)
    if p < 0:
        return False
    apply_insertion(j, p, best_c, c, sol)
    schedule(inst, sol, lam)
    return True


@njit(cache=True)
def regret_insert(removed, nrem, inst, sol, ws, slot_ok, posn, best_c, regret_c, lam, max_slots):
    """Regret-2 insertion of ``removed[:nrem]``: repeatedly insert the task with the largest gap between its best
    and second-best slot. Tasks without a feasible slot stay out; returns how many were inserted."""
    left = nrem
    placed = 0
    while left > 0:
        bx, breg, bobj = -1, -INF, INF
        bp, bc = -1, 0
        for x in range(left):
            j = removed[x]
            mark_slots(j, inst, sol, slot_ok, posn, max_slots)
            p, obj, _, c, second = insertion(j, inst, sol, ws, slot_ok, 0.0, lam, True, regret_c)
            if p < 0:
                continue
            reg = second - obj
            if reg > breg or (reg == breg and obj < bobj):
                bx, breg, bobj, bp, bc = x, reg, obj, p, c
                best_c[:c] = regret_c[:c]
        if bx < 0:
            break
        apply_insertion(removed[bx], bp, best_c, bc, sol)
        schedule(inst, sol, lam)
        left -= 1
        placed += 1
        removed[bx] = removed[left]
    return placed


@njit(cache=True)
def repair(op, removed, nrem, inst, sol, ws, slot_ok, posn, best_c, regret_c, par):
    """Reinsert ``removed[:nrem]`` (``sol`` schedule current, removed tasks absent)."""
    req = inst[0]
    lam, max_slots = par[P_LAM], int(par[P_SLOTS])
    noise = par[P_NOISE] if op == R_NOISY else 0.0
    if op == R_REGRET and nrem <= par[P_REGQ]:
        regret_insert(removed, nrem, inst, sol, ws, slot_ok, posn, best_c, regret_c, lam, max_slots)
        return
    if op == R_LARGEST:
        size = np.empty(nrem)
        for x in range(nrem):
            size[x] = req[removed[x]].sum() + 0.5 * np.random.random()
        order = removed[:nrem][np.argsort(-size)]
    else:
        order = removed[:nrem][np.random.permutation(nrem)]
    for x in range(nrem):
        insert_one(order[x], inst, sol, ws, slot_ok, posn, best_c, noise, lam, max_slots)


@njit(cache=True)
def _pick_ranked(score, n_pick, power, out):
    """Pick ``n_pick`` indices into ``out`` by ascending ``score``, each at rank ``floor(u**power * left)`` among
    those left (Ropke & Pisinger randomization); returns how many were picked."""
    order = np.argsort(score)
    left = order.shape[0]
    got = 0
    while got < n_pick and left > 0:
        r = int(np.random.random() ** power * left)
        t = order[r]
        out[got] = t
        got += 1
        order[r:left - 1] = order[r + 1:left]
        left -= 1
    return got


@njit(cache=True)
def destroy(op, q, inst, sol, removed, flag, buf, par):
    """Choose up to ``q`` present removable tasks with operator ``op``, take them out of ``sol`` and reschedule.

    ``sol``'s schedule must be current on entry. Returns the number of removed tasks."""
    req, tt, near, rem = inst[0], inst[3], inst[7], inst[20]
    seq, mem, cnt, ni, start, _, barr, pred, ret, last, fl = sol[:11]
    n = ni[0]
    cpos = np.empty(n, np.int64)  # positions of removable tasks
    nr = 0
    for p in range(n):
        if rem[seq[p]]:
            cpos[nr] = p
            nr += 1
    q = min(q, nr)
    flag[:] = False
    got = 0
    if nr == 0:
        return 0
    if op == 0:  # random
        perm = np.random.permutation(nr)
        for x in range(q):
            removed[x] = seq[cpos[perm[x]]]
            flag[removed[x]] = True
        got = q
    elif op == 1:  # Shaw relatedness to a random seed task
        r = seq[cpos[np.random.randint(0, nr)]]
        w_d, w_t, w_r = np.random.random(), np.random.random(), np.random.random()
        span = max(fl[0], EPS)
        tmax = max(tt.max(), EPS)
        score = np.empty(nr)
        for x in range(nr):
            t = seq[cpos[x]]
            dreq = 0.0
            for k in range(req.shape[1]):
                dreq += abs(req[r, k] - req[t, k])
            score[x] = w_d * tt[r, t] / tmax + w_t * abs(start[r] - start[t]) / span + w_r * dreq / req.shape[1]
        got = _pick_ranked(score, q, par[P_SHAW], buf)
        for x in range(got):
            removed[x] = seq[cpos[buf[x]]]
            flag[removed[x]] = True
    elif op == 2:  # tasks whose members wait the longest
        score = np.empty(nr)
        for x in range(nr):
            t = seq[cpos[x]]
            w = 0.0
            for k in range(cnt[t]):
                w += start[t] - barr[t, k]
            score[x] = -w
        got = _pick_ranked(score, q, par[P_WORST], buf)
        for x in range(got):
            removed[x] = seq[cpos[buf[x]]]
            flag[removed[x]] = True
    elif op == 3:  # critical chain behind the makespan, padded with its spatial neighbours
        a = np.argmax(ret)
        t = last[a]
        L = 0
        while t >= 0:
            if rem[t]:
                buf[L] = t
                L += 1
            kb = 0
            for k in range(1, cnt[t]):
                if barr[t, k] > barr[t, kb]:
                    kb = k
            t = pred[t, kb]
        chain = buf[:L][np.random.permutation(L)]
        for x in range(min(q, L)):
            removed[got] = chain[x]
            flag[chain[x]] = True
            got += 1
        tries = 0
        while got < q and L > 0 and near.shape[1] > 0 and tries < 4 * q:
            tries += 1
            c0 = chain[np.random.randint(0, L)]
            t = near[c0, min(np.random.randint(0, 4 + q), near.shape[1] - 1)]
            if cnt[t] > 0 and not flag[t] and rem[t]:
                removed[got] = t
                flag[t] = True
                got += 1
    elif op == 4:  # contiguous segment of one robot's route (the makespan robot half of the time)
        A = ret.shape[0]
        a = np.argmax(ret) if np.random.random() < 0.5 else np.random.randint(0, A)
        L = 0
        for p in range(n):
            t = seq[p]
            if not rem[t]:
                continue
            for k in range(cnt[t]):
                if mem[t, k] == a:
                    buf[L] = t
                    L += 1
                    break
        if L > 0:
            s = np.random.randint(0, max(1, L - q + 1))
            for x in range(s, min(L, s + q)):
                removed[got] = buf[x]
                flag[buf[x]] = True
                got += 1
    m = 0
    for p in range(n):
        t = seq[p]
        if flag[t]:
            cnt[t] = 0
        else:
            seq[m] = t
            m += 1
    ni[0] = m
    schedule(inst, sol, par[P_LAM])
    return got


@njit(cache=True)
def _is_fixed(inst, t, i):
    fmem, fcnt = inst[16], inst[17]
    for x in range(fcnt[t]):
        if fmem[t, x] == i:
            return True
    return False


@njit(cache=True)
def _pos_of(sol, t):
    seq, ni = sol[0], sol[3]
    for p in range(ni[0]):
        if seq[p] == t:
            return p
    return -1


@njit(cache=True)
def _gains_before_head(inst, sol, a, b, cut):
    """Would robot ``a`` get a visit before its head if it took over ``b``'s swappable visits from ``cut`` on?"""
    h = inst[21][a]
    if h < 0:
        return False
    ph = _pos_of(sol, h)
    if ph < cut:
        return False
    seq, mem, cnt, rem = sol[0], sol[1], sol[2], inst[20]
    for p in range(cut, ph):
        t = seq[p]
        if not rem[t]:
            continue
        ha, hb = False, False
        for k in range(cnt[t]):
            if mem[t, k] == a:
                ha = True
            elif mem[t, k] == b:
                hb = True
        if hb and not ha and not _is_fixed(inst, t, b):
            return True
    return False


@njit(cache=True)
def tail_swap(inst, sol, par):
    """Exchange the route tails (from a random cut in the free part of the order) of two robots with identical
    traits, on removable tasks only (fixed members stay).

    Coalitions keep their trait multisets, so covers stay minimal. Returns False when no partner exists."""
    partners, pstart, group, rem = inst[8], inst[9], inst[10], inst[20]
    seq, mem, cnt, ni, ret = sol[0], sol[1], sol[2], sol[3], sol[8]
    A = ret.shape[0]
    a1 = np.argmax(ret) if np.random.random() < 0.5 else np.random.randint(0, A)
    g = group[a1]
    size = pstart[g + 1] - pstart[g]
    if size < 2:
        return False
    a2 = partners[pstart[g] + np.random.randint(0, size - 1)]
    if a2 == a1:
        a2 = partners[pstart[g + 1] - 1]
    n = ni[0]
    lo = prefix_len(inst, sol)
    cut = lo + np.random.randint(0, n - lo + 1)
    if _gains_before_head(inst, sol, a1, a2, cut) or _gains_before_head(inst, sol, a2, a1, cut):
        return False
    for p in range(cut, n):
        t = seq[p]
        if not rem[t]:
            continue
        k1, k2 = -1, -1
        for k in range(cnt[t]):
            if mem[t, k] == a1:
                k1 = k
            elif mem[t, k] == a2:
                k2 = k
        if k1 >= 0 and k2 < 0 and not _is_fixed(inst, t, a1):
            mem[t, k1] = a2
        elif k2 >= 0 and k1 < 0 and not _is_fixed(inst, t, a2):
            mem[t, k2] = a1
    schedule(inst, sol, par[P_LAM])
    return True


@njit(cache=True)
def _critical_chain(sol, buf):
    """Tasks of the chain behind the makespan (latest-arriving member's predecessor, back to the start) into
    ``buf``; returns its length."""
    cnt, barr, pred, ret, last = sol[2], sol[6], sol[7], sol[8], sol[9]
    t = last[np.argmax(ret)]
    L = 0
    while t >= 0:
        buf[L] = t
        L += 1
        kb = 0
        for k in range(1, cnt[t]):
            if barr[t, k] > barr[t, kb]:
                kb = k
        t = pred[t, kb]
    return L


@njit(cache=True)
def member_swap(inst, sol, buf, par):
    """Hand one visit of robot ``a1`` (on the critical chain half of the time) to an identical robot ``a2`` that is
    not in that coalition or, with probability 1/2, exchange it for a visit of ``a2`` to a task without ``a1``.
    Only free, removable tasks and non-fixed members move. Returns False when no such move exists."""
    partners, pstart, group, mode, rem = inst[8], inst[9], inst[10], inst[18], inst[20]
    seq, mem, cnt, ni = sol[0], sol[1], sol[2], sol[3]
    n = ni[0]
    if n == 0:
        return False
    if np.random.random() < 0.5:
        L = _critical_chain(sol, buf)
        if L == 0:
            return False
        t = buf[np.random.randint(0, L)]
    else:
        t = seq[np.random.randint(0, n)]
    if not rem[t] or mode[t] != M_FREE:
        return False
    k1 = np.random.randint(0, cnt[t])
    a1 = mem[t, k1]
    if _is_fixed(inst, t, a1):
        return False
    g = group[a1]
    size = pstart[g + 1] - pstart[g]
    if size < 2:
        return False
    a2 = partners[pstart[g] + np.random.randint(0, size - 1)]
    if a2 == a1:
        a2 = partners[pstart[g + 1] - 1]
    for k in range(cnt[t]):
        if mem[t, k] == a2:
            return False
    head = inst[21]
    pt = _pos_of(sol, t)
    if head[a2] >= 0 and _pos_of(sol, head[a2]) > pt:  # a2 would visit t before its head
        return False
    ph1 = _pos_of(sol, head[a1]) if head[a1] >= 0 else -1
    if np.random.random() < 0.5:
        seen, t2, k2 = 0, -1, -1
        for p in range(n):
            u = seq[p]
            if not rem[u] or mode[u] != M_FREE or p < ph1:
                continue
            ka, kb = -1, -1
            for k in range(cnt[u]):
                if mem[u, k] == a1:
                    ka = k
                elif mem[u, k] == a2:
                    kb = k
            if kb >= 0 and ka < 0 and not _is_fixed(inst, u, a2):
                seen += 1
                if np.random.random() * seen < 1.0:  # reservoir sampling
                    t2, k2 = u, kb
        if t2 >= 0:
            mem[t2, k2] = a1
    mem[t, k1] = a2
    schedule(inst, sol, par[P_LAM])
    return True


@njit(cache=True)
def _roulette(w):
    r = np.random.random() * w.sum()
    acc = 0.0
    for k in range(w.shape[0]):
        acc += w[k]
        if r < acc:
            return k
    return w.shape[0] - 1


@njit(cache=True)
def better(fa, fb):
    """Lexicographic (tasks outside the plan, makespan, objective) comparison of two ``fl`` vectors with tolerance:
    is ``a`` strictly better than ``b``?"""
    if fa[3] < fb[3] - 0.5:
        return True
    if fa[3] > fb[3] + 0.5:
        return False
    return fa[0] < fb[0] - EPS or (fa[0] <= fb[0] + EPS and fa[2] < fb[2] - EPS)


@njit(cache=True)
def run_batch(n_iter, temp_a, temp_b, inst, cur, cand, best, ws, slot_ok, posn, best_c, regret_c, removed, flag,
              buf, wd, wr, sd, sr, ud, ur, par, stats):
    """``n_iter`` ALNS iterations with simulated-annealing acceptance, the temperature going geometrically from
    ``temp_a`` to ``temp_b`` over the batch. Returns the number of improvements of ``best``."""
    improved = 0
    q_lo, q_hi = int(par[P_QLO]), int(par[P_QHI])
    q_stag, q_big = int(par[P_QSTAG]), int(par[P_QBIG])
    for it in range(n_iter):
        temp = temp_a * (temp_b / temp_a) ** (it / n_iter) if temp_a > 0 else 0.0
        copy_sol(cur, cand)
        d = _roulette(wd)
        r = -1
        if (d == D_SWAP and not tail_swap(inst, cand, par)) or (d == D_MSWAP and not member_swap(inst, cand, buf,
                                                                                                    par)):
            d = 0
        if d != D_SWAP and d != D_MSWAP:
            r = _roulette(wr)
            qh = q_hi if q_stag <= 0 else max(q_hi, min(q_big, q_hi + stats[S_SINCE] // q_stag))
            q = np.random.randint(q_lo, qh + 1)
            nrem = destroy(d, q, inst, cand, removed, flag, buf, par)
            repair(r, removed, nrem, inst, cand, ws, slot_ok, posn, best_c, regret_c, par)
        stats[S_IT] += 1
        stats[S_SINCE] += 1
        delta = cand[10][2] - cur[10][2]
        score = 0.0
        accept = delta <= EPS or (temp > 0 and np.random.random() < np.exp(-delta / temp))
        if accept:  # equal-objective moves (often no-ops) earn nothing
            score = par[P_S2] if delta < -EPS else (par[P_S3] if delta > EPS else 0.0)
            copy_sol(cand, cur)
            stats[S_ACC] += 1
            if better(cur[10], best[10]):
                copy_sol(cur, best)
                score = par[P_S1]
                improved += 1
                stats[S_IMP] += 1
                stats[S_SINCE] = 0
        sd[d] += score
        ud[d] += 1
        if r >= 0:
            sr[r] += score
            ur[r] += 1
        stats[S_SEG] += 1
        if stats[S_SEG] >= par[P_SEG]:
            rho = par[P_RHO]
            for k in range(wd.shape[0]):
                if ud[k] > 0:
                    wd[k] = max(0.05, (1 - rho) * wd[k] + rho * sd[k] / ud[k])
            for k in range(wr.shape[0]):
                if ur[k] > 0:
                    wr[k] = max(0.05, (1 - rho) * wr[k] + rho * sr[k] / ur[k])
            sd[:] = 0
            sr[:] = 0
            ud[:] = 0
            ur[:] = 0
            stats[S_SEG] = 0
        if stats[S_SINCE] >= par[P_RESTART]:
            if par[P_STOP] > 0:  # the driver restarts
                return improved
            copy_sol(best, cur)  # a large random kick of the best plan here did not help (dev, v1)
            stats[S_SINCE] = 0
            stats[S_RESTART] += 1
    return improved


@njit(cache=True)
def construct(order, inst, sol, ws, slot_ok, posn, best_c, noise, lam, max_slots):
    """Sequential best insertion of ``order`` into ``sol`` (which holds the other tasks, schedule current); returns
    how many were inserted (the others stay out)."""
    placed = 0
    for x in range(order.shape[0]):
        if insert_one(order[x], inst, sol, ws, slot_ok, posn, best_c, noise, lam, max_slots):
            placed += 1
    return placed
