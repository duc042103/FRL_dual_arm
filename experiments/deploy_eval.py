"""Result 2 - deployment robustness of frozen policies (Stage C of FIG. 4).

Every trained checkpoint is evaluated noise-free (mean action) for one episode
in each of ``--episodes`` environments under several runtime-filter conditions:

  filter_on        nominal runtime filter (as in training)
  filter_off       runtime filter removed
  reduced_rate     filter only executes every 3rd control cycle (slow / delayed checker)
  approx_model     filter uses capsules 2 cm thinner than the real links (approximate model)
  noisy_filter_on  nominal filter, 5x sensor and actuation noise (sim-to-real gap)

Collisions are always judged with the TRUE capsule model at the end point and
mid-point of every control step.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frl.env import DualPiperBoxLift, EnvConfig  # noqa: E402
from frl.ppo import ActorCritic, TrainConfig, make_env_cfg  # noqa: E402

CONDITIONS = ["filter_on", "filter_off", "reduced_rate", "approx_model", "noisy_filter_on"]


def condition_cfg(base: EnvConfig, cond: str) -> tuple[EnvConfig, bool]:
    ec = copy.deepcopy(base)
    ec.filt.enabled = True  # a runtime filter is available to every policy at deployment
    ec.filt.period = 1
    filter_on = True
    if cond == "filter_off":
        filter_on = False
    elif cond == "reduced_rate":
        ec.filt.period = 3
    elif cond == "approx_model":
        ec.filt.radius_offset = -0.02
    elif cond == "noisy_filter_on":
        ec.obs_joint_noise *= 5
        ec.act_joint_noise *= 5
    return ec, filter_on


@torch.no_grad()
def evaluate_deploy(model: ActorCritic, env_cfg: EnvConfig, seed: int, filter_on: bool, episodes: int) -> dict:
    """One episode per environment with the frozen noise-free policy.

    Besides success / collision / intervention it records how deep the policy's
    OWN proposals (before the filter) go into the safety margin:
    depth = max(0, d_safe - d_min(q_des)), evaluated with the true capsule model.
    """
    ec = copy.deepcopy(env_cfg)
    ec.num_envs = episodes
    ec.collision_terminates = False
    env = DualPiperBoxLift(ec, seed=seed)
    env.filter_on = filter_on
    N = episodes
    obs = env.observe()
    alive = torch.ones(N, dtype=torch.bool)
    acc = {k: torch.zeros(N) for k in ["steps", "coll", "margin", "interv", "corr", "depth", "prop_viol"]}
    success = torch.zeros(N, dtype=torch.bool)
    collided = torch.zeros(N, dtype=torch.bool)
    min_clear = torch.full((N,), 1e9)
    d_safe = ec.filt.d_safe
    for _ in range(ec.max_steps):
        a = model.mean(model.norm(obs))
        obs, r, done, info = env.step(a)
        al = alive.float()
        d_prop = env.true_dist.dmin(info["q_des"])
        depth = (d_safe - d_prop).clamp_min(0.0)
        acc["steps"] += al
        acc["coll"] += info["collision"].float() * al
        acc["margin"] += info["margin_violation"].float() * al
        acc["interv"] += info["m"].float() * al
        acc["corr"] += info["corr"] * info["m"].float() * al
        acc["depth"] += depth * al
        acc["prop_viol"] += (d_prop < d_safe).float() * al
        min_clear = torch.where(alive, torch.minimum(min_clear, info["d_path"]), min_clear)
        collided |= info["collision"] & alive
        success |= info["success"] & alive
        alive &= ~done
        if not bool(alive.any()):
            break
    S = acc["steps"].sum()
    return {
        "success_rate": success.float().mean().item(),
        "collision_episode_rate": collided.float().mean().item(),
        "collision_step_rate": (acc["coll"].sum() / S).item(),
        "margin_violation_rate": (acc["margin"].sum() / S).item(),
        "intervention_rate": (acc["interv"].sum() / S).item(),
        "mean_correction_when_intervened": (acc["corr"].sum() / acc["interv"].sum().clamp_min(1)).item(),
        "proposal_violation_rate": (acc["prop_viol"].sum() / S).item(),
        "proposal_depth_mean_mm": (1000 * acc["depth"].sum() / S).item(),
        "min_clearance_mean": min_clear.mean().item(),
        "min_clearance_p05": min_clear.quantile(0.05).item(),
    }


def load(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    tc = TrainConfig(method=cfg["method"], seed=cfg["seed"])
    model = ActorCritic(62, 12)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--out", default="results/deploy_eval.json")
    p.add_argument("--methods", nargs="*", default=None)
    a = p.parse_args()
    torch.set_num_threads(1)
    results = {}
    if os.path.exists(a.out):
        with open(a.out) as fh:
            results = json.load(fh)
    for ck in sorted(glob.glob(os.path.join(a.runs, "*.pt"))):
        name = os.path.basename(ck)[:-3]
        model, tc = load(ck)
        if a.methods and tc.method not in a.methods:
            continue
        if name in results:
            continue
        base = make_env_cfg(tc)
        res = {"method": tc.method, "seed": tc.seed}
        for cond in CONDITIONS:
            ec, on = condition_cfg(base, cond)
            r = evaluate_deploy(model, ec, seed=20_000 + tc.seed, filter_on=on, episodes=a.episodes)
            res[cond] = r
            print(f"{name:22s} {cond:16s} succ {r['success_rate']:.3f} coll {r['collision_episode_rate']:.3f} "
                  f"int {r['intervention_rate']:.3f} clear {r['min_clearance_mean']:.3f}", flush=True)
        results[name] = res
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
