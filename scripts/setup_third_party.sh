#!/usr/bin/env bash
# Clone pinned external benchmark code into third_party/ (not committed) and apply local patches.
set -euo pipefail
cd "$(dirname "$0")/../third_party"
if [ ! -d HeteroMRTA ]; then
  git clone -q https://github.com/marmotlab/HeteroMRTA.git
  git -C HeteroMRTA checkout -q db51e29
  git -C HeteroMRTA apply ../third_party/heteromrta_numpy2.patch 2>/dev/null || git -C HeteroMRTA apply "$PWD/heteromrta_numpy2.patch"
fi
if [ ! -d Sadcher ]; then
  # No license: run unmodified, never redistribute.
  git clone -q https://github.com/jakbichler/Sadcher.git
fi
git -C HeteroMRTA log --oneline -1
git -C Sadcher log --oneline -1
