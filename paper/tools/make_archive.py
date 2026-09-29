"""Anonymised supplementary archive for the AAMAS 2027 submission: paper/supplementary.zip (at most 25 MB).

Usage: make_archive.py [--rows runs/anytime_test] [--out paper/supplementary.zip]
Contents: the supplementary PDF, the solver and harness code with its tests, the pre-registration, the result reports
(test, validation, development and CTAS-D checks), the rows of the test campaign (gzipped JSON lines) and the scripts
that make every table and figure. Every text file is anonymised (host names, user names, paths, account names) and the
archive is refused if an identifying string survives. The benchmark itself is not included: it is public (Apache-2.0)
and is fetched at a fixed commit; our small compatibility patch is included.
"""
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 25 * 1024 * 1024
REDACT = [  # (pattern, replacement), applied in order to every text file
    (r"/home/jovyan/dev/cbba-sota-run", "<campaign-worktree>"),
    (r"/home/jovyan/dev/cbba-sota", "<repo>"),
    (r"/home/jovyan/dev/cbja", "<other-project>"),
    (r"/home/jovyan", "<home>"),
    (r"dhcho-dev-2gpus-0", "host-A"),
    (r"dhcho-4gpu-0", "host-B"),
    (r"dhcho[\w.-]*", "host"),
    (r"crdhsh@gmail\.com", "anonymous@example.org"),
    (r"github\.com/Matlab7/[\w.-]+", "<anonymous-repository>"),
    (r"Matlab7", "anonymous"),
]
FORBIDDEN = re.compile(r"dhcho|jovyan|Matlab7|crdhsh", re.IGNORECASE)
TEXT = {".py", ".md", ".txt", ".csv", ".json", ".jsonl", ".sh", ".toml", ".patch", ".tex", ".yaml", ".yml", ".cfg"}
CODE = ["cbba_sota", "scripts", "tests", "pyproject.toml"]
RESULTS = ["test", "val-c1", "ctas", "alns-v2", "phase1b", "trackD-week1"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
README = """# Supplementary archive (anonymous submission)

- `supplement.pdf`: the supplementary material.
- `code/`: the solver (ALNS), the competitors, the exact evaluator and simulator replay, the campaign harness and the
  test suite. Python 3.12; `pip install -e code/` (dependencies in `code/pyproject.toml`).
- `code/third_party/`: put the public HeteroMRTA benchmark here (github.com/marmotlab/HeteroMRTA at commit db51e29,
  Apache-2.0) and apply `heteromrta_numpy2.patch` (NumPy 2 compatibility only).
- `prereg/prereg-phase1.md`: the pre-registration of the confirmatory run (frozen before the test instances were run;
  the git identifiers of the frozen source trees are recorded in it; one dated correction of its AI-assistance note).
- `results/`: reports and CSV files of every campaign: `test` (confirmatory), `val-c1` (dry run and references),
  `alns-v2` (development ablation), `ctas` (checks of the CTAS-D port), `phase1b` (20-task settings), `trackD-week1`
  (exploratory dynamic study).
- `rows/anytime_test/*.jsonl.gz`: one JSON row per run of the test campaign (plan as routes, simulator makespan, wall
  and CPU time, trace, pinned CPUs, host load and throttling counters).
- `paper_tools/`: the scripts that turn the reports and rows into every table, number and figure of the paper.

Instances are regenerated from their seeds with `scripts/gen_instances.py` (`--check` verifies them). The test
campaign is `scripts/anytime.py run --split test --grid test --n 50 --n-large 50 --lanes 6`; its reports come from
`results/test/analyze.sh`. Host names, user names and local paths in this archive are replaced by placeholders.
"""


def redact(text: str) -> str:
    for pat, rep in REDACT:
        text = re.sub(pat, rep, text)
    return text


def copy_tree(src: Path, dst: Path) -> None:
    for path in sorted(src.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts) or path.suffix in {".nbi", ".nbc", ".pyc"}:
            continue
        if path.is_dir():
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in TEXT:
            target.write_text(redact(path.read_text(errors="replace")))
        else:
            shutil.copy2(path, target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, default=ROOT / "runs" / "anytime_test")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "supplementary.zip")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "supplementary"
        stage.mkdir()
        (stage / "README.md").write_text(README)
        shutil.copy2(ROOT / "paper" / "supplement.pdf", stage / "supplement.pdf")
        for item in CODE:
            src = ROOT / item
            if src.is_dir():
                copy_tree(src, stage / "code" / item)
            else:
                (stage / "code").mkdir(exist_ok=True)
                (stage / "code" / item).write_text(redact(src.read_text()))
        patch = ROOT / "third_party" / "heteromrta_numpy2.patch"
        (stage / "code" / "third_party").mkdir(parents=True, exist_ok=True)
        (stage / "code" / "third_party" / patch.name).write_text(redact(patch.read_text()))
        (stage / "prereg").mkdir()
        (stage / "prereg" / "prereg-phase1.md").write_text(redact((ROOT / "docs" / "prereg-phase1.md").read_text()))
        for name in RESULTS:
            copy_tree(ROOT / "docs" / "results" / name, stage / "results" / name)
        for name in ("make_tables.py", "make_supplement.py", "make_figures.py", "sensitivity.py", "abstract_txt.py",
                     "time_constructor.py", "make_overview.py", "post_campaign.sh"):
            (stage / "paper_tools").mkdir(exist_ok=True)
            (stage / "paper_tools" / name).write_text(redact((ROOT / "paper" / "tools" / name).read_text()))
        rows = stage / "rows" / args.rows.name
        rows.mkdir(parents=True)
        for path in sorted(args.rows.glob("*.jsonl")):
            with gzip.open(rows / (path.name + ".gz"), "wt") as f:
                f.write(redact(path.read_text()))
        bad = []
        for path in stage.rglob("*"):
            if path.is_file():
                if path.name.endswith(".jsonl.gz"):
                    with gzip.open(path, "rt") as f:
                        data = f.read()
                else:
                    data = path.read_text(errors="ignore") if path.suffix in TEXT else ""
                if FORBIDDEN.search(data) or FORBIDDEN.search(str(path.relative_to(stage))):
                    bad.append(str(path.relative_to(stage)))
        if bad:
            raise SystemExit(f"identifying strings left in: {bad[:10]}")
        args.out.unlink(missing_ok=True)
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(stage.parent))
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 2**20:.1f} MB)")
    if size > LIMIT:
        raise SystemExit("archive exceeds the 25 MB limit of AAMAS 2027")


if __name__ == "__main__":
    main()
