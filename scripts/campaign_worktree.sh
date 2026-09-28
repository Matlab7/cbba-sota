#!/usr/bin/env bash
# Check out COMMIT (default HEAD) into a separate git worktree for timed campaigns, so that edits in this checkout
# cannot change the code a running campaign imports (lane processes import solver modules lazily). data/, runs/ and
# third_party/<repo>/ are shared with this checkout by symlink. Campaigns must run with PYTHONPATH set to the
# worktree, so that cbba_sota is imported from it rather than from the editable install; the rows then record the
# worktree's clean commit as their code version.
#
# Usage: scripts/campaign_worktree.sh [COMMIT] [DIR]        (default DIR: <repo>-run; an existing DIR is moved to COMMIT)
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
commit="$(git -C "$repo" rev-parse --verify "${1:-HEAD}^{commit}")"
dest="${2:-$repo-run}"
if [ -e "$dest/.git" ]; then
  git -C "$dest" checkout -q --detach "$commit"
else
  git -C "$repo" worktree add -q --detach "$dest" "$commit"
fi
for link in data runs; do
  [ -e "$dest/$link" ] || ln -s "$repo/$link" "$dest/$link"
done
for sub in "$repo"/third_party/*/; do
  name="$(basename "$sub")"
  [ -e "$dest/third_party/$name" ] || ln -s "$repo/third_party/$name" "$dest/third_party/$name"
done
# compile (or load) the numba kernels once, before any lane starts
(cd "$dest" && PYTHONPATH="$dest" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 "$repo/.venv/bin/python" -c "
import cbba_sota, numpy as np
from cbba_sota.bench import runtime
from cbba_sota.hetero import Instance, evaluate
from cbba_sota.solvers import alns, greedy
assert cbba_sota.__file__.startswith('$dest'), cbba_sota.__file__
alns._warmup()
rng = np.random.default_rng(0)
toy = Instance(req=np.eye(2)[[0, 1, 0, 1]], loc=rng.random((4, 2)), dur=rng.random(4), ab=np.eye(2)[[0, 0, 1, 1]],
               depot=np.zeros((4, 2)), species=[0, 0, 1, 1])
evaluate(toy, greedy.construct(toy, restarts=2))
print('worktree $dest at', runtime.code_version())
")
echo "run campaigns as: cd $dest && PYTHONPATH=$dest $repo/.venv/bin/python scripts/<script>.py ..."
