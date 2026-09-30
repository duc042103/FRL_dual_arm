#!/usr/bin/env bash
# Run the full training queue used for the README results (4 parallel CPU workers).
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-4000000}
JOBS=${JOBS:-4}
OUT=${OUT:-results/runs}
mkdir -p "$OUT" results/logs
{
  for seed in 0 1 2; do
    for method in frl filter_only filter_rp soft_penalty; do echo "$method $seed"; done
  done
  for seed in 0 1 2; do
    for method in frl_noRP frl_direct frl_meanproj; do echo "$method $seed"; done
  done
  for seed in 3 4; do
    for method in frl filter_only filter_rp soft_penalty; do echo "$method $seed"; done
  done
} | xargs -P "$JOBS" -L 1 bash -c \
  'if [ ! -f '"$OUT"'/"$0"_s"$1".pt ]; then python3 experiments/train.py --method "$0" --seed "$1" --steps '"$STEPS"' --out '"$OUT"' > results/logs/"$0"_s"$1".log 2>&1; fi'
