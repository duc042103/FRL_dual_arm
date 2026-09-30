#!/usr/bin/env bash
# Reproduce every result in the README (CPU only).
#   STEPS  - environment steps per training run
#   SEEDS  - seeds per method
#   JOBS   - parallel training processes (one CPU thread each)
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-4000000}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-4}
OUT=${OUT:-results/runs}
mkdir -p "$OUT" results/logs

# Result 3a: controlled study of the Eq. (5) target (seconds)
python3 experiments/toy_bias.py

# Results 1 and 3b: training runs
jobs_list=()
for seed in $SEEDS; do
  for method in frl filter_only soft_penalty filter_rp frl_direct frl_meanproj; do
    jobs_list+=("$method $seed")
  done
done
printf '%s\n' "${jobs_list[@]}" | xargs -P "$JOBS" -L 1 bash -c \
  'python3 experiments/train.py --method "$0" --seed "$1" --steps '"$STEPS"' --out '"$OUT"' > results/logs/"$0"_s"$1".log 2>&1'

# Result 2: deployment robustness of the frozen policies
python3 experiments/deploy_eval.py --runs "$OUT" --episodes 1000

# Figures and tables
python3 experiments/plot_results.py --runs "$OUT"
