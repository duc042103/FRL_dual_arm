"""Result 3a - controlled study of the Eq. (5) target (Review Round 2, Section C).

Setting: a Gaussian policy a_raw = mu + eps, eps ~ N(0, sigma^2 I), and a safety
filter that projects onto a feasible set F. We compare the expected gradient
that the feasibility-consistency term applies to the distribution mean for
three targets:

  corr_disp  (claimed):   L = m || mu - sg(mu + a_feas - a_raw) ||^2   -> dL/dmu = -2 m (a_feas - a_raw)
  direct     (prior art): L = m || mu - sg(a_feas) ||^2                -> dL/dmu =  2 m (mu - a_feas) = -2 m (eps + da)
  mean_proj  (alt. emb.): L = m_mu || mu - sg(Proj(mu)) ||^2           -> needs one extra projection per step

Part 1 (1-D, F = {a <= 0}): expected gradient field as a function of the mean
position, and the conditional noise bias E[eps | m = 1].
Part 2 (1-D dynamics): the mean is pushed outward by a constant task gradient
g (the task optimum lies beyond the constraint, i.e. the constraint is binding)
and updated by SGD on  -g*mu + lambda * L_feas. We report the equilibrium mean
position and the deterministic (noise-free) intervention indicator.
Part 3 (closed form): equilibrium position of the mean as a function of the
task pressure g. Only the correction-displacement target keeps the mean itself
feasible, for every g below g* = 2 lambda sigma / sqrt(2 pi).
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.plot_style import COLORS, apply_style  # noqa: E402

rng = np.random.default_rng(0)
SIGMA = 0.5
MC = 400_000


def grad_1d(mu, target, sigma=SIGMA, b=0.0, n=MC):
    """Monte-Carlo expected dL/dmu for the 1-D half-line F = {a <= b}."""
    eps = rng.normal(0.0, sigma, n)
    a = mu + eps
    a_feas = np.minimum(a, b)
    m = (a > b).astype(float)
    if target == "corr_disp":
        g = -2.0 * m * (a_feas - a)
    elif target == "direct":
        g = 2.0 * m * (mu - a_feas)
    elif target == "mean_proj":
        g = np.full(n, 2.0 * max(mu - b, 0.0))
    else:
        raise ValueError(target)
    return g.mean(), g.std() / np.sqrt(n), (eps[m > 0].mean() if m.sum() > 0 else 0.0), m.mean()


def dynamics_1d(target, g_task=0.2, lam=1.0, lr=0.05, steps=4000, sigma=SIGMA, batch=256, mu0=-1.0):
    mu = mu0
    traj = []
    for _ in range(steps):
        eps = rng.normal(0.0, sigma, batch)
        a = mu + eps
        a_feas = np.minimum(a, 0.0)
        m = (a > 0).astype(float)
        if target == "corr_disp":
            gf = (-2.0 * m * (a_feas - a)).mean()
        elif target == "direct":
            gf = (2.0 * m * (mu - a_feas)).mean()
        elif target == "mean_proj":
            gf = 2.0 * max(mu, 0.0)
        else:  # filter_only: no feasibility term
            gf = 0.0
        # task gradient (reward keeps pushing outward) + feasibility term
        mu = mu - lr * (-g_task + lam * gf)
        traj.append(mu)
    traj = np.array(traj)
    mu_eq = traj[-1000:].mean()
    return traj, mu_eq


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _bisect(fun, lo, hi, it=200):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if fun(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def equilibrium(target, g, lam=1.0, sigma=SIGMA):
    """Closed-form equilibrium mean mu* where the feasibility force balances a task pressure g.

    corr_disp : g = 2 lam E[(mu + eps)^+]      = 2 lam sigma (phi(z) + z Phi(z)),  z = mu / sigma
    direct    : g = 2 lam E[m (mu - a_feas)]   = 2 lam mu Phi(z)                  (mu > 0)
    mean_proj : g = 2 lam max(mu, 0)
    filter_only: no restoring force -> mu* = +inf
    """
    if target == "corr_disp":
        return _bisect(lambda mu: 2 * lam * sigma * (_phi(mu / sigma) + mu / sigma * _Phi(mu / sigma)) - g, -10, 10)
    if target == "direct":
        return _bisect(lambda mu: 2 * lam * max(mu, 0.0) * _Phi(mu / sigma) - g, -10, 10)
    if target == "mean_proj":
        return g / (2 * lam)
    return float("inf")


def main(out_dir="results"):
    apply_style()
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    targets = ["corr_disp", "direct", "mean_proj"]
    labels = {"corr_disp": "Correction-displacement target (claimed, Eq. 5)",
              "direct": "Direct target a_feas (prior art)",
              "mean_proj": "Projected-mean target (alt. embodiment)",
              "filter_only": "Filter only (no feasibility term)"}
    mus = np.linspace(-2.0, 1.0, 61)
    field = {t: [] for t in targets}
    noise_bias, p_int = [], []
    for mu in mus:
        for t in targets:
            g, se, eb, pm = grad_1d(mu, t)
            field[t].append(g)
        _, _, eb, pm = grad_1d(mu, "direct")
        noise_bias.append(eb)
        p_int.append(pm)

    results = {"sigma": SIGMA}
    # Gradient at a feasible mean close to the boundary (mu = -0.25 sigma).
    for mu_probe in [-0.5, -0.25, 0.0, 0.25]:
        results[f"grad_at_mu={mu_probe}"] = {t: float(grad_1d(mu_probe, t)[0]) for t in targets}
        results[f"E[eps|m=1]_at_mu={mu_probe}"] = float(grad_1d(mu_probe, "direct")[2])

    # 1-D dynamics with a binding constraint
    dyn = {}
    for t in ["filter_only", "direct", "corr_disp", "mean_proj"]:
        traj, mu_eq = dynamics_1d(t)
        # deterministic (noise-free) intervention and violation of the mean
        dyn[t] = {"traj": traj, "mu_eq": float(mu_eq), "mean_needs_correction": bool(mu_eq > 0.0),
                  "stochastic_intervention_rate": float((rng.normal(mu_eq, SIGMA, 200000) > 0).mean())}
    results["dynamics_1d"] = {t: {k: v for k, v in d.items() if k != "traj"} for t, d in dyn.items()}

    # Closed-form equilibria vs task pressure g (lambda = 1)
    gs = np.linspace(0.0, 1.2, 121)
    eq = {t: np.array([equilibrium(t, g) for g in gs]) for t in ["direct", "corr_disp", "mean_proj"]}
    g_crit = 2 * SIGMA * _phi(0.0)
    results["equilibrium"] = {
        "critical_pressure_corr_disp (mean stays feasible below)": g_crit,
        "note": "for g > 0 the direct and projected-mean targets always settle with an infeasible mean",
        **{f"mu_star/sigma_at_g={g}": {t: equilibrium(t, g) / SIGMA for t in ["direct", "corr_disp", "mean_proj"]}
           for g in [0.1, 0.2, 0.4]},
    }

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for t in targets:
        ax.plot(mus / SIGMA, -np.array(field[t]), color=COLORS[t], lw=2.2, label=labels[t])
    ax.axvline(0, color="0.3", ls="--", lw=1)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.fill_betweenx([-3, 3], 0, 1.0 / SIGMA * 1.0, color="#d62728", alpha=0.07)
    ax.set_ylim(-1.6, 1.0)
    ax.set_xlabel(r"mean position $\mu/\sigma$  (boundary at 0, infeasible side shaded)")
    ax.set_ylabel(r"expected update direction $-\partial L_{feas}/\partial\mu$")
    ax.set_title("(a) Update applied to the mean (1-D)")
    ax.annotate("direct target pulls a FEASIBLE\nmean toward the boundary", xy=(-0.5 / SIGMA * 0.5, 0.18),
                xytext=(-3.9, 0.55), fontsize=9, arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.legend(loc="lower left", fontsize=8.5)

    ax = axes[1]
    for t in ["filter_only", "direct", "corr_disp", "mean_proj"]:
        ax.plot(np.arange(len(dyn[t]["traj"])), dyn[t]["traj"] / SIGMA, color=COLORS[t], lw=2, label=labels[t])
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.fill_between([0, len(dyn['direct']['traj'])], 0, 3, color="#d62728", alpha=0.07)
    ax.set_ylim(-2.2, 2.5)
    ax.set_xlabel("update step")
    ax.set_ylabel(r"mean position $\mu/\sigma$")
    ax.set_title("(b) Binding constraint: mean under task pressure")
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[2]
    for t in ["direct", "mean_proj", "corr_disp"]:
        ax.plot(gs / (SIGMA), eq[t] / SIGMA, color=COLORS[t], lw=2.2, label=labels[t])
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.axvline(g_crit / SIGMA, color=COLORS["corr_disp"], ls=":", lw=1.2)
    ax.fill_between(gs / SIGMA, 0, 3, color="#d62728", alpha=0.07)
    ax.text(g_crit / SIGMA + 0.03, -1.6, r"$g^*=2\lambda\sigma/\sqrt{2\pi}$", color=COLORS["corr_disp"], fontsize=9)
    ax.set_ylim(-2.0, 1.6)
    ax.set_xlabel(r"task pressure toward the constraint $g/(\lambda\sigma)$")
    ax.set_ylabel(r"equilibrium mean $\mu^*/\sigma$ (closed form)")
    ax.set_title("(c) Where the mean settles (filter only: +inf)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(out_dir, "figures", "fig3a_target_bias_toy.png")
    fig.savefig(path, dpi=160)
    with open(os.path.join(out_dir, "toy_bias.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
