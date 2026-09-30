#!/usr/bin/env bash
# Seeds 5-9 for the remaining methods, so that every main method has 10 seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-4000000}
JOBS=${JOBS:-4}
OUT=${OUT:-results/runs}
mkdir -p "$OUT" results/logs
{
  for seed in 5 6 7 8 9; do echo "frl_noRP $seed"; done
  for seed in 5 6 7 8 9; do
    for method in filter_only soft_penalty; do echo "$method $seed"; done
  done
} | xargs -P "$JOBS" -L 1 bash -c \
  'if [ ! -f '"$OUT"'/"$0"_s"$1".pt ]; then python3 experiments/train.py --method "$0" --seed "$1" --steps '"$STEPS"' --out '"$OUT"' > results/logs/"$0"_s"$1".log 2>&1; fi'
