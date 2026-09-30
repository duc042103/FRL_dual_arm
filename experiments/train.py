"""Train one method with one seed and write a JSON log + checkpoint.

Example:
    python experiments/train.py --method frl --seed 0 --steps 3000000
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frl.ppo import METHODS, TrainConfig, train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=sorted(METHODS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--lambda-feas", type=float, default=1.0)
    p.add_argument("--w-interv", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--out", default="results/runs")
    p.add_argument("--tag", default="")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.method}{a.tag}_s{a.seed}"
    tc = TrainConfig(method=a.method, seed=a.seed, total_steps=a.steps, num_envs=a.num_envs,
                     lambda_feas=a.lambda_feas, w_interv=a.w_interv, eval_every=a.eval_every,
                     log_path=os.path.join(a.out, name + ".json"), ckpt_path=os.path.join(a.out, name + ".pt"))
    train(tc)


if __name__ == "__main__":
    main()
