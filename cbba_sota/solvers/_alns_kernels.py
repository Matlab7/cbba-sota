"""Numba kernels of the coalition ALNS (driver in ``cbba_sota.solvers.alns``).

A solution is a tuple of arrays (see ``SOL``): the global task order ``seq[:n]`` (tasks outside it are removed and
have ``cnt == 0``), the coalitions ``mem[j, :cnt[j]]``, and its schedule, filled by ``schedule`` with exactly the
operations of ``cbba_sota.hetero.forward_pass`` (so makespans are bit-identical). Every coalition is built by
``cover`` as a minimal cover and every agent visits its tasks in ``seq`` order, which keeps plans exact w.r.t. the
env and deadlock-free. ``schedule`` also fills ``tail[t]``, the longest path from the start of ``t`` to the end
(its duration, then the members' travel to their next tasks and those tasks' tails, or home).

Insertion (``insertion``) sweeps the slots of ``seq`` once with incrementally updated arrivals and picks a greedy
minimal cover by (noisy) earliest arrival for each candidate slot (skipping slots where no capable agent's arrival
changed, and keeping the agents sorted by arrival when most slots are candidates); a slot whose cover delays other
tasks also gets a delay-aware alternative cover. Each candidate's makespan follows in O(cover) from the tails:
the schedule is a longest-path problem, and inserting ``j`` only adds paths through ``j`` (a replaced edge
``prev -> next`` is never longer than ``prev -> j -> next``, by the triangle inequality), so the new makespan is
``max(old makespan, start_j + dur_j + max over members of travel to their next task + its tail, or home)``. With
the first-order delays of the members' next tasks this bounds the objective; slots are then evaluated best-first by
that bound, re-running only the agents whose state differs from the baseline schedule. That suffix pass stops as
soon as no agent is perturbed any more (a later task still starts at its baseline time) and is pruned by the
monotone sum of finish times. The resulting makespan is bit-identical to a full forward pass (tested).
"""
from __future__ import annotations

import numpy as np
from numba import njit

EPS = 1e-9
INF = np.inf

# Solution tuple layout.
SOL = ("seq", "mem", "cnt", "ni", "start", "finish", "barr", "pred", "ret", "last", "fl", "succ", "first", "lastk",
       "tail")
N_FL = 3  # fl = [makespan, sum of finish times, objective]

# Destroy operators (the last one is a move without repair) and repair operators.
DESTROY = ("random", "shaw", "worst_wait", "critical", "route_segment", "tail_swap")
REPAIR = ("random", "random_noisy", "largest_first", "regret2")
D_SWAP = 5
R_NOISY, R_LARGEST, R_REGRET = 1, 2, 3

# Parameter vector layout (float64).
PARAMS = ("lam", "noise", "q_lo", "q_hi", "max_slots", "shaw_p", "worst_p", "regret_max_q", "rho", "seg_len",
          "sigma_best", "sigma_better", "sigma_accept", "restart_iters", "alt_cover", "resort")
(P_LAM, P_NOISE, P_QLO, P_QHI, P_SLOTS, P_SHAW, P_WORST, P_REGQ, P_RHO, P_SEG, P_S1, P_S2, P_S3, P_RESTART, P_ALT,
 P_RESORT) = range(16)

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
    seq, mem, cnt, ni, start, finish, barr, pred, ret, last, fl, succ, first, lastk, tail = sol
    T, A = dur.shape[0], da.shape[0]
    for i in range(A):
        ret[i] = 0.0  # used as the agents' free time during the pass
        last[i] = -1
        first[i] = -1
    total = 0.0
    for p in range(ni[0]):
        t = seq[p]
        s = -INF
        for k in range(cnt[t]):
            i = mem[t, k]
            lt = last[i]
            a = ret[i] + (da[i, t] if lt < 0 else tt[lt, t])
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
        ret[i] = ret[i] + da[i, last[i]] if last[i] >= 0 else 0.0
        ms = max(ms, ret[i])
    fl[0] = ms
    fl[1] = total
    fl[2] = ms + lam * total / T
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
def copy_plan(src, dst):
    """Order and coalitions only; ``dst``'s schedule is stale until ``schedule`` runs."""
    n = src[3][0]
    dst[0][:n] = src[0][:n]
    dst[1][...] = src[1]
    dst[2][:] = src[2]
    dst[3][0] = n


@njit(cache=True)
def cover(j, req, ab, capl, ncap, arr, key, need, chosen, fac):
    """Greedy minimal cover of task ``j`` by earliest key ``arr * fac`` (``fac``: noise factor per agent); members
    in ``chosen[:c]``.

    Adds contributing agents in key order until the requirement is met, then drops redundant members, latest
    arrival first (``_prune``). Returns ``c`` (-1 if infeasible).
    """
    K = req.shape[1]
    for k in range(K):
        need[k] = req[j, k]
    nc = ncap[j]
    for x in range(nc):
        i = capl[j, x]
        key[x] = arr[i] * fac[i]
    c = 0
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
    return _prune(chosen, c, arr, ab, need)


@njit(cache=True)
def _prune(chosen, c, arr, ab, need):
    """Drop redundant members of ``chosen[:c]`` (surplus in ``need`` <= 0), latest arrival first (one pass suffices
    since dropping only shrinks the surplus); returns the size of the minimal cover left in ``chosen``."""
    K = ab.shape[1]
    for a in range(1, c):  # sort by arrival, latest first
        v = chosen[a]
        b = a - 1
        while b >= 0 and arr[chosen[b]] < arr[v]:
            chosen[b + 1] = chosen[b]
            b -= 1
        chosen[b + 1] = v
    m = 0
    for a in range(c):
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
def cover_sorted(j, req, ab, order, nc, arr, need, chosen):
    """``cover`` for agents already sorted by key (``order[:nc]``): takes each contributing agent in turn."""
    K = req.shape[1]
    left = 0
    for k in range(K):
        need[k] = req[j, k]
        left += need[k] > 0
    c = 0
    for x in range(nc):
        if left == 0:
            break
        i = order[x]
        useful = False
        for k in range(K):
            if need[k] > 0 and ab[i, k] > 0:
                useful = True
                break
        if useful:
            chosen[c] = i
            c += 1
            for k in range(K):
                if need[k] > 0 and need[k] - ab[i, k] <= 0:
                    left -= 1
                need[k] -= ab[i, k]
    if left > 0:
        return -1
    return _prune(chosen, c, arr, ab, need)


@njit(cache=True)
def _same_set(a, na, b, nb):
    """``a[:na]`` and ``b[:nb]`` hold the same agents."""
    if na != nb:
        return False
    for x in range(na):
        hit = False
        for y in range(nb):
            hit = hit or a[x] == b[y]
        if not hit:
            return False
    return True


@njit(cache=True)
def _add_slot(j, p, chosen, c, arr, inst, sol, ws, nc, base_ms, base_sum, scale):
    """Store slot ``p`` with cover ``chosen[:c]`` as candidate ``nc``: its finish, exact new makespan and a lower
    bound on its objective from the first-order delays of the members' next tasks. Returns the summed delay plus the
    makespan increase (0 when ``j`` fits without pushing anything)."""
    dur, tt, da = inst[2], inst[3], inst[4]
    start, succ, first, tail = sol[4], sol[11], sol[12], sol[14]
    cp, cf, clb, cms, ccnt, cmem, lastp, lastkp, nxt = ws[8:17]
    s = -INF
    for a in range(c):
        s = max(s, arr[chosen[a]])
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
    return delay + (lb - base_ms)


@njit(cache=True)
def insertion(j, inst, sol, ws, slot_ok, noise, lam, need_second, best_c, alt):
    """Best slot and coalition for removed task ``j`` in ``sol`` (whose schedule must be current).

    Phase 1 sweeps the order once, updating the agents' arrivals at ``j`` incrementally, and builds each marked
    slot's cover and a lower bound on its objective from the delays the cover causes at its members' next tasks.
    With ``alt``, a slot whose cover delays other tasks also gets the cover by arrival plus the delay each agent
    would cause at its next task (or at its return beyond the makespan) if ``j`` started with the earliest cover.
    Phase 2 evaluates the candidates best-first by that bound with the incremental suffix pass and stops once the
    bound reaches the incumbent (the second best for regret). Ranking by the bound alone and evaluating only the top
    few was tried and is much worse: early slots have small bounds but long delay cascades.
    Returns ``(p, obj, ms, c, second_obj)``: insert before ``seq[p]`` with members ``best_c[:c]``.
    """
    req, ab, dur, tt, da, capl, ncap, capf = inst[0], inst[1], inst[2], inst[3], inst[4], inst[5], inst[6], inst[11]
    seq, mem, cnt, ni, start, finish, barr, ret, fl, succ, first = (sol[0], sol[1], sol[2], sol[3], sol[4], sol[5],
                                                                     sol[6], sol[8], sol[10], sol[11], sol[12])
    (arr, key, need, chosen, stamp, ofree, olast, ctr, cp, cf, clb, cms, ccnt, cmem, lastp, lastkp, _, arr2, order,
     rank, skey, fac) = ws
    T, A = dur.shape[0], da.shape[0]
    n = ni[0]
    base_ms, base_sum = fl[0], fl[1]
    scale = lam / T
    nca = ncap[j]
    for x in range(nca):  # key of an agent: arrival x a noise factor drawn once per call
        i = capl[j, x]
        arr[i] = da[i, j]
        lastp[i] = -1
        fac[i] = 1.0 + noise * np.random.random() if noise > 0 else 1.0
    marked = 0
    for p in range(n + 1):
        marked += slot_ok[p]
    dense = 2 * marked > n + 1  # then keep the capable agents sorted by key in the sweep
    if dense:
        for x in range(nca):
            i = capl[j, x]
            skey[i] = arr[i] * fac[i]
            key[x] = skey[i]
        o = np.argsort(key[:nca], kind="mergesort")
        for x in range(nca):
            order[x] = capl[j, o[x]]
            rank[order[x]] = x
    ctr[3] += 1  # insertions
    nc = 0
    fresh = True  # arrivals changed since the last candidate slot (else that slot's plan is the same)
    for p in range(n + 1):
        if slot_ok[p] and fresh:
            fresh = False
            if dense:
                c = cover_sorted(j, req, ab, order, nca, arr, need, chosen)
            else:
                c = cover(j, req, ab, capl, ncap, arr, key, need, chosen, fac)
            cost = _add_slot(j, p, chosen, c, arr, inst, sol, ws, nc, base_ms, base_sum, scale) if c > 0 else 0.0
            nc += c > 0
            if alt and cost > 0:
                s1 = cf[nc - 1] - dur[j]  # start with the earliest cover
                for x in range(nca):
                    i = capl[j, x]
                    t = first[i] if lastp[i] < 0 else succ[lastp[i], lastkp[i]]
                    late = max(arr[i], s1) + dur[j] + (tt[j, t] - start[t] if t >= 0 else da[i, j] - base_ms)
                    arr2[i] = arr[i] + max(late, 0.0)
                c2 = cover(j, req, ab, capl, ncap, arr2, key, need, chosen, fac)
                if c2 > 0 and not _same_set(chosen, c2, cmem[nc - 1], c):
                    _add_slot(j, p, chosen, c2, arr, inst, sol, ws, nc, base_ms, base_sum, scale)
                    nc += 1
        if p < n:
            t = seq[p]
            for k in range(cnt[t]):
                i = mem[t, k]
                if capf[j, i]:
                    arr[i] = finish[t] + tt[t, j]
                    lastp[i] = t
                    lastkp[i] = k
                    fresh = True
                    if not dense:
                        continue
                    skey[i] = arr[i] * fac[i]  # arrivals grow along the order (triangle inequality): move i back
                    x = rank[i]
                    while x + 1 < nca and skey[order[x + 1]] < skey[i]:
                        order[x] = order[x + 1]
                        rank[order[x]] = x
                        x += 1
                    while x > 0 and skey[order[x - 1]] > skey[i]:  # rounding
                        order[x] = order[x - 1]
                        rank[order[x]] = x
                        x -= 1
                    order[x] = i
                    rank[i] = x
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
            s2 = -INF
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
def mark_slots(j, inst, sol, slot_ok, posn, max_slots):
    """All slots, or (when there are more than ``max_slots``) the slots around the nearest present tasks."""
    near = inst[7]
    seq, ni = sol[0], sol[3]
    n = ni[0]
    if n + 1 <= max_slots:
        slot_ok[:n + 1] = True
        return
    slot_ok[:n + 1] = False
    for p in range(n):
        posn[seq[p]] = p
    slot_ok[0] = True
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
    for p in range(n):
        posn[seq[p]] = -1


@njit(cache=True)
def insert_one(j, inst, sol, ws, slot_ok, posn, best_c, noise, lam, max_slots, alt):
    mark_slots(j, inst, sol, slot_ok, posn, max_slots)
    p, _, _, c, _ = insertion(j, inst, sol, ws, slot_ok, noise, lam, False, best_c, alt)
    apply_insertion(j, p, best_c, c, sol)
    schedule(inst, sol, lam)


@njit(cache=True)
def repair(op, removed, nrem, inst, sol, ws, slot_ok, posn, best_c, regret_c, par):
    """Reinsert ``removed[:nrem]`` (``sol`` schedule current, removed tasks absent)."""
    req = inst[0]
    lam, max_slots, alt = par[P_LAM], int(par[P_SLOTS]), par[P_ALT] > 0
    noise = par[P_NOISE] if op == R_NOISY else 0.0
    if op == R_REGRET and nrem <= par[P_REGQ]:
        left = nrem
        while left > 0:
            bx, breg, bobj = -1, -INF, INF
            bp, bc = -1, 0
            for x in range(left):
                j = removed[x]
                mark_slots(j, inst, sol, slot_ok, posn, max_slots)
                p, obj, _, c, second = insertion(j, inst, sol, ws, slot_ok, 0.0, lam, True, regret_c, alt)
                reg = second - obj
                if reg > breg or (reg == breg and obj < bobj):
                    bx, breg, bobj, bp, bc = x, reg, obj, p, c
                    best_c[:c] = regret_c[:c]
            apply_insertion(removed[bx], bp, best_c, bc, sol)
            schedule(inst, sol, lam)
            left -= 1
            removed[bx] = removed[left]
        return
    if op == R_LARGEST:
        size = np.empty(nrem)
        for x in range(nrem):
            size[x] = req[removed[x]].sum() + 0.5 * np.random.random()
        order = removed[:nrem][np.argsort(-size)]
    else:
        order = removed[:nrem][np.random.permutation(nrem)]
    for x in range(nrem):
        insert_one(order[x], inst, sol, ws, slot_ok, posn, best_c, noise, lam, max_slots, alt)


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
def destroy(op, q, inst, src, sol, removed, flag, buf, par):
    """Choose up to ``q`` present tasks with operator ``op``, take them out of ``sol`` and reschedule it.

    ``sol`` holds the plan of ``src`` (it may be ``src`` itself), whose schedule must be current. Returns the number
    of removed tasks."""
    req, tt, near = inst[0], inst[3], inst[7]
    seq, mem, cnt, ni = sol[:4]
    start, barr, ret, fl = src[4], src[6], src[8], src[10]
    n = ni[0]
    q = min(q, n)
    flag[:] = False
    got = 0
    if op == 0:  # random
        perm = np.random.permutation(n)
        for x in range(q):
            removed[x] = seq[perm[x]]
            flag[removed[x]] = True
        got = q
    elif op == 1:  # Shaw relatedness to a random seed task
        r = seq[np.random.randint(0, n)]
        w_d, w_t, w_r = np.random.random(), np.random.random(), np.random.random()
        span = max(fl[0], EPS)
        tmax = EPS
        for t in range(near.shape[0] if near.shape[1] > 0 else 0):  # tt.max() from each farthest neighbour
            tmax = max(tmax, tt[t, near[t, near.shape[1] - 1]])
        score = np.empty(n)
        for x in range(n):
            t = seq[x]
            dreq = 0.0
            for k in range(req.shape[1]):
                dreq += abs(req[r, k] - req[t, k])
            score[x] = w_d * tt[r, t] / tmax + w_t * abs(start[r] - start[t]) / span + w_r * dreq / req.shape[1]
        got = _pick_ranked(score, q, par[P_SHAW], buf)
        for x in range(got):
            removed[x] = seq[buf[x]]
            flag[removed[x]] = True
    elif op == 2:  # tasks whose members wait the longest
        score = np.empty(n)
        for x in range(n):
            t = seq[x]
            w = 0.0
            for k in range(cnt[t]):
                w += start[t] - barr[t, k]
            score[x] = -w
        got = _pick_ranked(score, q, par[P_WORST], buf)
        for x in range(got):
            removed[x] = seq[buf[x]]
            flag[removed[x]] = True
    elif op == 3:  # critical chain behind the makespan, padded with its spatial neighbours
        L = _critical_chain(src, buf)
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
            if cnt[t] > 0 and not flag[t]:
                removed[got] = t
                flag[t] = True
                got += 1
    elif op == 4:  # contiguous segment of one agent's route (the makespan agent half of the time)
        A = ret.shape[0]
        a = np.argmax(ret) if np.random.random() < 0.5 else np.random.randint(0, A)
        L = 0
        for p in range(n):
            t = seq[p]
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
def tail_swap(inst, src, sol, par):
    """Exchange the route tails (from a random cut in the global order) of two agents with identical traits in
    ``sol``, which holds the plan of ``src`` (schedule current).

    Coalitions keep their trait multisets, so covers stay minimal. Returns False when no partner exists."""
    partners, pstart, group = inst[8], inst[9], inst[10]
    seq, mem, cnt, ni, ret = sol[0], sol[1], sol[2], sol[3], src[8]
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
    cut = np.random.randint(0, n + 1)
    for p in range(cut, n):
        t = seq[p]
        k1, k2 = -1, -1
        for k in range(cnt[t]):
            if mem[t, k] == a1:
                k1 = k
            elif mem[t, k] == a2:
                k2 = k
        if k1 >= 0 and k2 < 0:
            mem[t, k1] = a2
        elif k2 >= 0 and k1 < 0:
            mem[t, k2] = a1
    schedule(inst, sol, par[P_LAM])
    return True


@njit(cache=True)
def _critical_chain(sol, buf):
    """Tasks of the chain behind the makespan (latest-arriving member's predecessor, back to the depot) into
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
def sort_by_start(sol):
    """Reorder ``seq`` by start time (stable): another topological order of the same plan, so the schedule is
    unchanged but later insertions see the slots of a time line."""
    seq, n, start = sol[0], sol[3][0], sol[4]
    keys = np.empty(n)
    for p in range(n):
        keys[p] = start[seq[p]]
    seq[:n] = seq[:n][np.argsort(keys, kind="mergesort")]


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
def better(ms_a, obj_a, ms_b, obj_b):
    """Lexicographic (makespan, objective) comparison with tolerance: is ``a`` strictly better than ``b``?"""
    return ms_a < ms_b - EPS or (ms_a <= ms_b + EPS and obj_a < obj_b - EPS)


@njit(cache=True)
def run_batch(n_iter, temp_a, temp_b, inst, cur, cand, best, ws, slot_ok, posn, best_c, regret_c, removed, flag,
              buf, wd, wr, sd, sr, ud, ur, par, stats):
    """``n_iter`` ALNS iterations with simulated-annealing acceptance, the temperature going geometrically from
    ``temp_a`` to ``temp_b`` over the batch. Returns the number of improvements of ``best``."""
    improved = 0
    q_lo, q_hi = int(par[P_QLO]), int(par[P_QHI])
    a, b = cur, cand  # current and candidate plan; an accepted candidate swaps them instead of being copied
    swapped = False
    for it in range(n_iter):
        temp = temp_a * (temp_b / temp_a) ** (it / n_iter) if temp_a > 0 else 0.0
        copy_plan(a, b)
        d = _roulette(wd)
        r = -1
        if d == D_SWAP and not tail_swap(inst, a, b, par):
            d = 0
        if d != D_SWAP:
            r = _roulette(wr)
            q = np.random.randint(q_lo, q_hi + 1)
            nrem = destroy(d, q, inst, a, b, removed, flag, buf, par)
            repair(r, removed, nrem, inst, b, ws, slot_ok, posn, best_c, regret_c, par)
        stats[S_IT] += 1
        stats[S_SINCE] += 1
        delta = b[10][2] - a[10][2]
        score = 0.0
        accept = delta <= EPS or (temp > 0 and np.random.random() < np.exp(-delta / temp))
        if accept:  # equal-objective moves (often no-ops) earn nothing
            score = par[P_S2] if delta < -EPS else (par[P_S3] if delta > EPS else 0.0)
            a, b = b, a
            swapped = not swapped
            if par[P_RESORT] > 0 and np.random.random() < par[P_RESORT]:
                sort_by_start(a)
            stats[S_ACC] += 1
            if better(a[10][0], a[10][2], best[10][0], best[10][2]):
                copy_sol(a, best)
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
        if stats[S_SINCE] >= par[P_RESTART]:  # restart from the cycle's best (kicks, elite pools and crossover
            copy_sol(best, a)                   # children instead did not help on dev)
            stats[S_SINCE] = 0
            stats[S_RESTART] += 1
    if swapped:
        copy_sol(a, cur)
    return improved


@njit(cache=True)
def construct(order, inst, sol, ws, slot_ok, posn, best_c, noise, lam, max_slots, alt):
    """Sequential best insertion of ``order`` into ``sol`` (which holds the other tasks, schedule current)."""
    for x in range(order.shape[0]):
        insert_one(order[x], inst, sol, ws, slot_ok, posn, best_c, noise, lam, max_slots, alt)
