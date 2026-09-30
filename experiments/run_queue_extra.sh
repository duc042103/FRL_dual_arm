#!/usr/bin/env bash
# Extra seeds for the head-to-head comparison (FRL vs. hard filter + reward penalty)
# and for the 2x2 factorial (FRL without reward penalty).
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-4000000}
JOBS=${JOBS:-4}
OUT=${OUT:-results/runs}
mkdir -p "$OUT" results/logs
{
  for seed in 5 6 7 8 9; do
    for method in frl filter_rp; do echo "$method $seed"; done
  done
  for seed in 3 4; do echo "frl_noRP $seed"; done
} | xargs -P "$JOBS" -L 1 bash -c \
  'if [ ! -f '"$OUT"'/"$0"_s"$1".pt ]; then python3 experiments/train.py --method "$0" --seed "$1" --steps '"$STEPS"' --out '"$OUT"' > results/logs/"$0"_s"$1".log 2>&1; fi'
