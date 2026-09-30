"""PPO with the feasibility-consistency term (Eqs. 5-10) and the baselines.

Methods (all share the same network, PPO hyper-parameters and task reward):

* ``soft_penalty`` : no filter, collision penalty + termination (conventional RL).
* ``filter_only``  : hard safety filter, correction discarded.
* ``filter_rp``    : hard filter + reward penalty on the correction magnitude
                     (prior art, cf. Wabersich & Zeilinger 2021).
* ``frl``          : proposed method - hard filter + intervention quantity in the
                     reward (Eq. 12) + feasibility-consistency loss with the
                     correction-displacement target (Eq. 5).
* ablations of the Eq. (5) target: ``frl_direct`` (target = a_feas, Chen-style)
  and ``frl_meanproj`` (target = projection of the mean, alternative embodiment),
  and ``frl_sched`` (lambda_feas schedule + intervention-rate criterion, claims 16-17).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .env import DualPiperBoxLift, EnvConfig


# ---------------------------------------------------------------- networks
def mlp(inp, out, hidden=(256, 256, 128)):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Observation normaliser (bundled with the policy at deployment, Stage B)."""

    def __init__(self, dim, eps=1e-5, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps, self.clip = eps, clip

    @torch.no_grad()
    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


class ActorCritic(nn.Module):
    """Gaussian actor with a learned state-independent std, and a state-value critic."""

    def __init__(self, obs_dim, act_dim, init_std=0.5):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()

    def mean(self, obs_n):
        return self.actor(obs_n)

    def value(self, obs_n):
        return self.critic(obs_n).squeeze(-1)

    def dist(self, obs_n):
        mu = self.actor(obs_n)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu)), mu


# ---------------------------------------------------------------- config
@dataclass
class TrainConfig:
    method: str = "frl"
    seed: int = 0
    num_envs: int = 256
    rollout_steps: int = 24            # N-step rollout
    total_steps: int = 3_000_000
    epochs: int = 5
    minibatches: int = 4
    lr: float = 3e-4
    desired_kl: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    init_std: float = 0.5
    lambda_feas: float = 1.0
    # lambda schedule (claim 16) and intervention-rate criterion (claim 17)
    lambda_final: float = 0.1
    interv_threshold: float = 0.05
    w_interv: float = 0.5              # reward weight w2 for methods that use c_t
    eval_every: int = 10
    eval_envs: int = 256
    log_path: str | None = None
    ckpt_path: str | None = None
    env: EnvConfig = field(default_factory=EnvConfig)


METHODS = {
    # name: (filter on, collision penalty/termination, reward intervention penalty, feasibility target)
    "soft_penalty": dict(filter=False, collision_terminates=True, reward_penalty=False, target=None),
    "filter_only": dict(filter=True, collision_terminates=False, reward_penalty=False, target=None),
    "filter_rp": dict(filter=True, collision_terminates=False, reward_penalty=True, target=None),
    "frl": dict(filter=True, collision_terminates=False, reward_penalty=True, target="corr_disp"),
    "frl_noRP": dict(filter=True, collision_terminates=False, reward_penalty=False, target="corr_disp"),
    "frl_direct": dict(filter=True, collision_terminates=False, reward_penalty=True, target="direct"),
    "frl_meanproj": dict(filter=True, collision_terminates=False, reward_penalty=True, target="mean_proj"),
    "frl_sched": dict(filter=True, collision_terminates=False, reward_penalty=True, target="corr_disp", schedule=True),
}


def make_env_cfg(tc: TrainConfig, num_envs: int | None = None) -> EnvConfig:
    spec = METHODS[tc.method]
    import copy
    ec = copy.deepcopy(tc.env)
    ec.num_envs = num_envs or tc.num_envs
    ec.filt.enabled = spec["filter"]
    ec.collision_terminates = spec["collision_terminates"]
    ec.w_interv = tc.w_interv if spec["reward_penalty"] else 0.0
    return ec


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model: ActorCritic, env_cfg: EnvConfig, seed: int, filter_on: bool = True,
             episodes: int | None = None, stochastic: bool = False) -> dict:
    """Run exactly one episode per environment with the frozen (noise-free) policy."""
    import copy
    ec = copy.deepcopy(env_cfg)
    if episodes is not None:
        ec.num_envs = episodes
    ec.collision_terminates = False  # measure, do not terminate
    env = DualPiperBoxLift(ec, seed=seed)
    env.filter_on = filter_on and ec.filt.enabled
    N = ec.num_envs
    obs = env.observe()
    alive = torch.ones(N, dtype=torch.bool)
    stats = {k: torch.zeros(N) for k in ["collision_steps", "margin_steps", "interventions", "fallbacks", "steps"]}
    min_clear = torch.full((N,), 1e9)
    success = torch.zeros(N, dtype=torch.bool)
    first_collision = torch.zeros(N, dtype=torch.bool)
    for _ in range(ec.max_steps):
        obs_n = model.norm(obs)
        if stochastic:
            d, _ = model.dist(obs_n)
            a = d.sample()
        else:
            a = model.mean(obs_n)
        obs, r, done, info = env.step(a)
        al = alive.float()
        stats["collision_steps"] += info["collision"].float() * al
        stats["margin_steps"] += info["margin_violation"].float() * al
        stats["interventions"] += info["m"].float() * al
        stats["fallbacks"] += info["fallback"].float() * al
        stats["steps"] += al
        min_clear = torch.where(alive, torch.minimum(min_clear, info["d_path"]), min_clear)
        first_collision |= info["collision"] & alive
        success |= info["success"] & alive
        alive &= ~done
        if not bool(alive.any()):
            break
    steps = stats["steps"].clamp_min(1)
    return {
        "success_rate": success.float().mean().item(),
        "collision_episode_rate": first_collision.float().mean().item(),
        "collision_step_rate": (stats["collision_steps"].sum() / steps.sum()).item(),
        "margin_violation_rate": (stats["margin_steps"].sum() / steps.sum()).item(),
        "intervention_rate": (stats["interventions"].sum() / steps.sum()).item(),
        "fallback_rate": (stats["fallbacks"].sum() / steps.sum()).item(),
        "min_clearance_mean": min_clear.mean().item(),
        "min_clearance_p05": min_clear.quantile(0.05).item(),
    }


# ---------------------------------------------------------------- training
def train(tc: TrainConfig, verbose: bool = True) -> dict:
    torch.manual_seed(tc.seed)
    spec = METHODS[tc.method]
    ec = make_env_cfg(tc)
    env = DualPiperBoxLift(ec, seed=tc.seed)
    env.randomize_episode_phase()
    model = ActorCritic(env.obs_dim, env.act_dim, tc.init_std)
    actor_params = list(model.actor.parameters()) + [model.log_std]
    opt = torch.optim.Adam(list(actor_params) + list(model.critic.parameters()), lr=tc.lr)
    lr = tc.lr
    target_mode = spec["target"]
    schedule = spec.get("schedule", False)
    lam_feas = tc.lambda_feas

    N, T = tc.num_envs, tc.rollout_steps
    obs = env.observe()
    iters = tc.total_steps // (N * T)
    log = []
    cum = {"env_steps": 0, "collisions": 0, "collision_episodes": 0, "episodes": 0, "successes": 0,
           "margin_violations": 0, "interventions": 0}
    ep_collided = torch.zeros(N, dtype=torch.bool)
    t_start = time.time()
    interv_window = []

    for it in range(iters):
        buf = {k: [] for k in ["obs", "a_raw", "a_feas", "m", "logp", "val", "rew", "done", "mu_target", "m_mu", "mu_old"]}
        it_stats = {"m": 0.0, "coll": 0.0, "margin": 0.0, "succ": 0, "eps": 0, "corr": 0.0, "fallback": 0.0,
                    "grasps": 0, "drops": 0}
        model.eval()
        for t in range(T):
            with torch.no_grad():
                model.norm.update(obs)
                obs_n = model.norm(obs)
                d, mu = model.dist(obs_n)
                a = d.sample()
                logp = d.log_prob(a).sum(-1)
                v = model.value(obs_n)
                if target_mode == "mean_proj":
                    # Alternative embodiment: pass the mean itself through IK -> filter -> FK.
                    outm = env.pipe(env.q, mu, True, env.f)
                    buf["mu_target"].append(outm["a_feas"])
                    buf["m_mu"].append(outm["m"].float())
            obs_next, r, done, info = env.step(a)
            # Time-out bootstrapping (truncation is not a terminal state).
            if bool(info["timeout"].any()):
                with torch.no_grad():
                    # value of the pre-reset successor is approximated with the current state's value
                    r = r + tc.gamma * v * info["timeout"].float()
            buf["obs"].append(obs); buf["a_raw"].append(a); buf["a_feas"].append(info["a_feas"])
            buf["m"].append(info["m"].float()); buf["logp"].append(logp); buf["val"].append(v)
            buf["mu_old"].append(mu)
            buf["rew"].append(r); buf["done"].append(done.float())
            it_stats["m"] += info["m"].float().sum().item()
            it_stats["coll"] += info["collision"].float().sum().item()
            it_stats["margin"] += info["margin_violation"].float().sum().item()
            it_stats["corr"] += (info["corr"] * info["m"].float()).sum().item()
            it_stats["fallback"] += info["fallback"].float().sum().item()
            it_stats["grasps"] += info["grasp_event"].sum().item()
            it_stats["drops"] += info["dropped"].sum().item()
            ep_collided |= info["collision"]
            it_stats["succ"] += (info["success"] & done).sum().item()
            it_stats["eps"] += done.sum().item()
            cum["collision_episodes"] += (ep_collided & done).sum().item()
            ep_collided &= ~done
            obs = obs_next

        with torch.no_grad():
            last_v = model.value(model.norm(obs))
        rew = torch.stack(buf["rew"]); val = torch.stack(buf["val"]); dn = torch.stack(buf["done"])
        # ---- GAE, Eq. (9), truncated at the end of the N-step rollout
        adv = torch.zeros_like(rew)
        gae = torch.zeros(N)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val[t + 1]
            delta = rew[t] + tc.gamma * nv * (1 - dn[t]) - val[t]
            gae = delta + tc.gamma * tc.lam * (1 - dn[t]) * gae
            adv[t] = gae
        ret = adv + val  # value target R_t = A_t + V(s_t)

        flat = lambda x: torch.stack(x).reshape(N * T, -1).squeeze(-1)
        b_obs = torch.stack(buf["obs"]).reshape(N * T, -1)
        b_a = torch.stack(buf["a_raw"]).reshape(N * T, -1)
        b_af = torch.stack(buf["a_feas"]).reshape(N * T, -1)
        b_m = flat(buf["m"]); b_logp = flat(buf["logp"]); b_ret = ret.reshape(-1); b_adv = adv.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        if target_mode == "mean_proj":
            b_mt = torch.stack(buf["mu_target"]).reshape(N * T, -1)
            b_mm = flat(buf["m_mu"])
        b_a_clip = b_a.clamp(-1.0, 1.0)
        b_mu_old = torch.stack(buf["mu_old"]).reshape(N * T, -1)
        std_old = model.log_std.exp().detach().clone()

        # ---- lambda_feas schedule / intervention-rate criterion (claims 16-17)
        interv_rate = it_stats["m"] / (N * T)
        if schedule:
            interv_window = (interv_window + [interv_rate])[-5:]
            progress = min(1.0, (it + 1) / max(1, int(0.5 * iters)))
            lam_feas = tc.lambda_feas + (tc.lambda_final - tc.lambda_feas) * progress
            if len(interv_window) == 5 and sum(interv_window) / 5 < tc.interv_threshold:
                lam_feas = min(lam_feas, tc.lambda_final)

        # ---- update phase
        model.train()
        mb = N * T // tc.minibatches
        l_clip_acc = l_feas_acc = l_v_acc = kl_acc = 0.0
        n_up = 0
        for ep in range(tc.epochs):
            perm = torch.randperm(N * T)
            for k in range(tc.minibatches):
                ids = perm[k * mb:(k + 1) * mb]
                obs_n = model.norm(b_obs[ids])
                d, mu = model.dist(obs_n)
                logp = d.log_prob(b_a[ids]).sum(-1)       # ratio on the RAW action, Eq. (8)
                ratio = torch.exp(logp - b_logp[ids])
                A = b_adv[ids]
                l_clip = -torch.min(ratio * A, ratio.clamp(1 - tc.clip, 1 + tc.clip) * A).mean()  # Eq. (7)
                # ---- feasibility-consistency term, Eq. (5)
                m = b_m[ids]
                if target_mode == "corr_disp":
                    target = (mu + (b_af[ids] - b_a_clip[ids])).detach()
                    l_feas = (m * ((mu - target) ** 2).sum(-1)).mean()
                elif target_mode == "direct":
                    target = b_af[ids].detach()
                    l_feas = (m * ((mu - target) ** 2).sum(-1)).mean()
                elif target_mode == "mean_proj":
                    mm = b_mm[ids]
                    target = b_mt[ids].detach()
                    l_feas = (mm * ((mu - target) ** 2).sum(-1)).mean()
                else:
                    l_feas = torch.zeros(())
                l_actor = l_clip + lam_feas * l_feas - tc.entropy_coef * d.entropy().sum(-1).mean()  # Eq. (6)
                v = model.value(obs_n)
                l_v = ((v - b_ret[ids]) ** 2).mean()          # Eq. (10); critic only
                loss = l_actor + tc.value_coef * l_v
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    # exact KL(old || new) between diagonal Gaussians
                    std_new = model.log_std.exp()
                    kl = (torch.log(std_new / std_old) + (std_old ** 2 + (b_mu_old[ids] - mu) ** 2)
                          / (2 * std_new ** 2) - 0.5).sum(-1).mean().item()
                l_clip_acc += l_clip.item(); l_feas_acc += float(l_feas); l_v_acc += l_v.item(); kl_acc += kl
                n_up += 1
            # adaptive learning rate on the KL
        kl_mean = kl_acc / n_up
        if kl_mean > 2 * tc.desired_kl:
            lr = max(1e-5, lr / 1.5)
        elif kl_mean < tc.desired_kl / 2:
            lr = min(1e-3, lr * 1.5)
        for g in opt.param_groups:
            g["lr"] = lr

        cum["env_steps"] += N * T
        cum["collisions"] += int(it_stats["coll"])
        cum["margin_violations"] += int(it_stats["margin"])
        cum["interventions"] += int(it_stats["m"])
        cum["episodes"] += int(it_stats["eps"])
        cum["successes"] += int(it_stats["succ"])
        row = {
            "iter": it, "env_steps": cum["env_steps"], "time_s": time.time() - t_start,
            "train_success_rate": it_stats["succ"] / max(1, it_stats["eps"]),
            "train_grasps_per_episode": it_stats["grasps"] / max(1, it_stats["eps"]),
            "train_drops_per_episode": it_stats["drops"] / max(1, it_stats["eps"]),
            "intervention_rate": interv_rate,
            "mean_correction": it_stats["corr"] / max(1.0, it_stats["m"]),
            "fallback_rate": it_stats["fallback"] / (N * T),
            "collision_step_rate": it_stats["coll"] / (N * T),
            "margin_violation_rate": it_stats["margin"] / (N * T),
            "cum_collisions": cum["collisions"], "cum_collision_episodes": cum["collision_episodes"],
            "cum_episodes": cum["episodes"], "cum_margin_violations": cum["margin_violations"],
            "mean_reward": rew.mean().item(), "l_clip": l_clip_acc / n_up, "l_feas": l_feas_acc / n_up,
            "l_value": l_v_acc / n_up, "kl": kl_mean, "lr": lr, "lambda_feas": lam_feas if target_mode else 0.0,
            "std": model.log_std.exp().mean().item(),
        }
        if tc.eval_every and (it % tc.eval_every == 0 or it == iters - 1):
            ev_on = evaluate(model, ec, seed=10_000 + tc.seed, filter_on=True, episodes=tc.eval_envs)
            row.update({f"eval_on_{k}": v for k, v in ev_on.items()})
            if ec.filt.enabled:
                ev_off = evaluate(model, ec, seed=10_000 + tc.seed, filter_on=False, episodes=tc.eval_envs)
                row.update({f"eval_off_{k}": v for k, v in ev_off.items()})
        log.append(row)
        if verbose and (it % tc.eval_every == 0 or it == iters - 1):
            msg = (f"[{tc.method} s{tc.seed}] it {it}/{iters} steps {cum['env_steps']/1e6:.2f}M "
                   f"succ {row['train_success_rate']:.2f} grasp {row['train_grasps_per_episode']:.2f} drop {row['train_drops_per_episode']:.2f} interv {interv_rate:.3f} coll {cum['collisions']} "
                   f"R {row['mean_reward']:.2f} Lf {row['l_feas']:.4f} std {row['std']:.2f} t {row['time_s']:.0f}s")
            if "eval_on_success_rate" in row:
                msg += f" | eval_on succ {row['eval_on_success_rate']:.2f} int {row['eval_on_intervention_rate']:.3f}"
            if "eval_off_success_rate" in row:
                msg += f" | eval_off succ {row['eval_off_success_rate']:.2f} coll {row['eval_off_collision_episode_rate']:.2f}"
            print(msg, flush=True)
            if tc.log_path:
                import json
                with open(tc.log_path, "w") as fh:
                    json.dump({"config": _cfg_dict(tc), "log": log}, fh)
    if tc.log_path:
        import json
        with open(tc.log_path, "w") as fh:
            json.dump({"config": _cfg_dict(tc), "log": log}, fh)
    if tc.ckpt_path:
        torch.save({"model": model.state_dict(), "config": _cfg_dict(tc)}, tc.ckpt_path)
    return {"model": model, "log": log, "env_cfg": ec}


def _cfg_dict(tc: TrainConfig) -> dict:
    from dataclasses import asdict
    return asdict(tc)
