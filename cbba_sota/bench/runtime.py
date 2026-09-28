"""Provenance and host conditions of campaign rows.

- ``fingerprint``: sha1 of an instance's arrays. Every row stores it, and resumable campaigns only skip rows whose
  fingerprint matches the current instance, so rows computed on a regenerated instance are recomputed.
- ``code_version``: the git commit of the source, or ``<commit|nocommit>-dirty-<hash>`` when it has uncommitted
  changes.
- CPU pinning: ``choose_cpus`` picks idle physical cores, ``pin`` restricts every thread of the calling process to
  a CPU set (threads and forked children created later inherit it); call it in pool initializers or at job start.
- ``Probe``: CPU affinity, load average and the cgroup's throttling counters over one job.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

import numpy as np

from cbba_sota.bench.configs import ROOT

_DATA = ("req", "loc", "dur", "ab", "depot", "species")
SOURCE = ("cbba_sota", "scripts", "pyproject.toml")  # what code_version hashes
_CPU_STAT = ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat")


# --- provenance ------------------------------------------------------------------------------------------------


def fingerprint(inst) -> str:
    """sha1 over the arrays (dtype, shape and bytes) and speed of an ``Instance``; name, source and the pickle
    layout do not enter."""
    h = hashlib.sha1()
    for name in _DATA:
        a = np.ascontiguousarray(getattr(inst, name))
        h.update(f"{name}:{a.dtype.str}:{a.shape};".encode())
        h.update(a.tobytes())
    h.update(repr(float(inst.speed)).encode())
    return h.hexdigest()


@cache
def instance_fingerprint(setting: str, split: str, i: int) -> str | None:
    """Fingerprint of the current pickle of a benchmark instance (cached per process), None if it is missing."""
    from cbba_sota.bench import configs
    from cbba_sota.hetero.instance import Instance

    path = configs.get(setting).instance_path(split, i)
    return fingerprint(Instance.from_pickle(path)) if path.exists() else None


def is_current(row: dict) -> bool:
    """The row carries the fingerprint of the current pickle of its (setting, split, instance)."""
    try:
        fp = instance_fingerprint(row["setting"], row["split"], row.get("instance", row.get("index")))
    except (KeyError, IndexError, TypeError):
        return False
    return fp is not None and row.get("fingerprint") == fp


@cache
def code_version(root: Path = ROOT) -> str:
    """HEAD if ``SOURCE`` is clean, else ``<HEAD or nocommit>-dirty-<sha1[:12]>`` over the diff to HEAD and the
    untracked source files; ``unknown`` outside a git checkout."""

    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout

    try:
        git("rev-parse", "--git-dir")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    try:
        head = git("rev-parse", "--verify", "-q", "HEAD").decode().strip()
    except subprocess.CalledProcessError:  # no commit yet
        head = ""
    h = hashlib.sha1(git("diff", "HEAD", "--", *SOURCE) if head else b"")
    listed = git("ls-files", "--others", "--exclude-standard", "-z", *([] if head else ["--cached"]), "--", *SOURCE)
    files = sorted(p for p in listed.split(b"\0") if p)
    for path in files:
        if (root / path.decode()).is_file():
            h.update(path + b"\0" + (root / path.decode()).read_bytes())
    if head and h.digest() == hashlib.sha1().digest():
        return head
    return f"{head or 'nocommit'}-dirty-{h.hexdigest()[:12]}"


def read_rows(path: Path) -> Iterator[dict]:
    """Rows of a JSONL file, skipping a torn last line after a crash."""
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# --- CPU sets --------------------------------------------------------------------------------------------------


def parse_cpus(text: str) -> list[int]:
    """``"0-3,8"`` -> ``[0, 1, 2, 3, 8]``."""
    out: set[int] = set()
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def format_cpus(cpus: Iterable[int]) -> str:
    """``[0, 1, 2, 3, 8]`` -> ``"0-3,8"``."""
    runs: list[list[int]] = []
    for c in sorted(cpus):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _jiffies() -> dict[int, tuple[int, int]]:
    """Per CPU (busy, total) jiffies from /proc/stat."""
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        name, *values = line.split()
        if name.startswith("cpu") and name != "cpu":
            v = list(map(int, values))
            idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
            out[int(name[3:])] = (sum(v[:8]) - idle, sum(v[:8]))
    return out


def _core(cpu: int) -> tuple[int, ...]:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    return tuple(parse_cpus(path.read_text())) if path.exists() else (cpu,)


def choose_cpus(n: int, sample_s: float = 0.5) -> list[int]:
    """``n`` CPUs this process may use, one per physical core, on the cores that were least busy (summed over
    their hyperthreads) during ``sample_s`` seconds."""
    allowed = sorted(os.sched_getaffinity(0))
    before = _jiffies()
    time.sleep(sample_s)
    after = _jiffies()
    load = {c: (after[c][0] - before[c][0]) / max(1, after[c][1] - before[c][1]) for c in after if c in before}
    cores: dict[tuple[int, ...], list[int]] = {}
    for c in allowed:
        cores.setdefault(_core(c), []).append(c)
    if len(cores) < n:
        raise ValueError(f"{n} CPUs requested, {len(cores)} physical cores available")
    ranked = sorted(cores, key=lambda core: (sum(load.get(c, 1.0) for c in core), core))
    return sorted(cores[core][0] for core in ranked[:n])


def blocks(cpus: list[int], size: int) -> list[list[int]]:
    """Disjoint consecutive blocks of ``size`` CPUs (the remainder is dropped)."""
    return [cpus[k:k + size] for k in range(0, len(cpus) - size + 1, size)]


def pin(cpus: Iterable[int]) -> None:
    """Restrict every thread of this process to ``cpus``; later threads and forked children inherit it."""
    cpus = set(cpus)
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except ProcessLookupError:  # the thread ended meanwhile
            pass


def affinity() -> str:
    return format_cpus(os.sched_getaffinity(0))


# --- host conditions -------------------------------------------------------------------------------------------


def cgroup_cpu_stat() -> dict[str, int]:
    """Counters of the container's cgroup ``cpu.stat`` (v1 ``throttled_time`` in ns becomes ``throttled_usec``)."""
    for path in _CPU_STAT:
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        stat = {k: int(v) for k, v in (line.split() for line in text.splitlines()) if v.isdigit()}
        if "throttled_time" in stat:
            stat["throttled_usec"] = stat.pop("throttled_time") // 1000
        return stat
    return {}


class Probe:
    """Host conditions over one job: CPU affinity of the process running it (``cpus`` if given), load average at
    the start and end, and the change in the cgroup's throttling counters (the whole container, not only this
    job)."""

    def __init__(self, cpus: Iterable[int] | None = None):
        self.cpus = None if cpus is None else format_cpus(cpus)
        self.load1, self.stat = os.getloadavg()[0], cgroup_cpu_stat()

    def fields(self) -> dict:
        end = cgroup_cpu_stat()
        out = {"affinity": self.cpus or affinity(), "load1": self.load1, "load1_end": os.getloadavg()[0]}
        for key in ("nr_throttled", "throttled_usec"):
            if key in self.stat and key in end:
                out[key] = end[key] - self.stat[key]
        return out
