"""Aggregate training logs and deployment evaluations into figures and tables.

Outputs (in results/):
  figures/fig1_training_benchmark.png   Result 1 - collisions, success, intervention rate vs. env steps
  figures/fig2_deployment.png           Result 2 - robustness to a removed / slow / approximate filter
  figures/fig3b_target_ablation.png     Result 3b - Eq. (5) target forms in the dual-arm task
  summary.json, tables.md               numbers used in README
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.plot_style import COLORS, LABELS, apply_style  # noqa: E402

MAIN = ["soft_penalty", "filter_only", "filter_rp", "frl", "frl_noRP"]
FACT = ["filter_only", "filter_rp", "frl_noRP", "frl"]
ABL = ["frl", "frl_direct", "frl_meanproj"]


def load_runs(run_dir):
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if not os.path.exists(f[:-5] + ".pt"):
            continue  # run still in progress
        with open(f) as fh:
            d = json.load(fh)
        runs[d["config"]["method"]].append(d)
    return runs


def series(run, key):
    xs = np.array([r["env_steps"] for r in run["log"] if key in r], dtype=float)
    ys = np.array([r[key] for r in run["log"] if key in r], dtype=float)
    return xs, ys


def smooth(y, k=5):
    if len(y) < k:
        return y
    ker = np.ones(k) / k
    pad = np.concatenate([np.full(k - 1, y[0]), y])
    return np.convolve(pad, ker, mode="valid")


def band(ax, runs, key, color, label, sm=1, scale=1.0, x_common=None, seeds=False, lo=0.0, hi=None):
    """Mean over seeds (bold) with a standard-error band; optionally thin per-seed curves."""
    curves = []
    for run in runs:
        x, y = series(run, key)
        if len(x) == 0:
            continue
        y = smooth(y, sm) * scale
        if x_common is None:
            x_common = x
        curves.append(np.interp(x_common, x, y))
    if not curves:
        return None
    Y = np.stack(curves)
    m = Y.mean(0)
    se = Y.std(0) / np.sqrt(len(curves))
    if seeds:
        for y in Y:
            ax.plot(x_common / 1e6, y, color=color, lw=0.7, alpha=0.35)
    ax.plot(x_common / 1e6, m, color=color, lw=2.2, label=label)
    if len(curves) > 1 and not seeds:
        ax.fill_between(x_common / 1e6, np.clip(m - se, lo, hi), np.clip(m + se, lo, hi), color=color, alpha=0.2, lw=0)
    return Y


def steps_to_threshold(run, key, thr):
    x, y = series(run, key)
    idx = np.nonzero(y >= thr)[0]
    return float(x[idx[0]]) if len(idx) else float("nan")


def ms(vals):
    v = np.array(vals, dtype=float)
    v = v[~np.isnan(v)] if np.any(~np.isnan(v)) else v
    return float(np.mean(v)) if len(v) else float("nan"), float(np.std(v)) if len(v) else float("nan")


def fmt(m, s, pct=False, digits=2):
    if np.isnan(m):
        return "n/a"
    if pct:
        return f"{100*m:.1f} ± {100*s:.1f}%"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def training_figure(runs, out_dir, summary, methods):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    for m in methods:
        if m not in runs:
            continue
        c, lab = COLORS[m], LABELS[m]
        band(axes[0], runs[m], "cum_collision_episodes", c, lab)
        band(axes[1], runs[m], "eval_on_success_rate", c, lab, scale=100, seeds=True)
        if m != "soft_penalty":
            band(axes[2], runs[m], "intervention_rate", c, lab, sm=5, scale=100)
            band(axes[3], runs[m], "eval_off_collision_episode_rate", c, lab, scale=100)
    axes[0].set_title("(a) Episodes with an arm-arm collision\n(cumulative, during training; band = s.e.m.)")
    axes[0].set_ylabel("collision episodes")
    axes[1].set_title("(b) Task success vs. environment steps\n(noise-free policy; bold = mean, thin = each seed)")
    axes[1].set_ylabel("success rate [%]")
    axes[2].set_title("(c) Filter intervention rate during training\n(fraction of control steps corrected)")
    axes[2].set_ylabel("intervention rate [%]")
    axes[3].set_title("(d) Filter dependence: collisions when the\nfilter is REMOVED at evaluation")
    axes[3].set_ylabel("episodes with collision [%]")
    for ax in axes:
        ax.set_xlabel("environment steps [M]")
    axes[1].legend(loc="upper left", fontsize=8.5)
    axes[3].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig1_training_benchmark.png"), dpi=160)
    plt.close(fig)

    # table 1
    rows = []
    for m in methods:
        if m not in runs:
            continue
        R = runs[m]
        last = [r["log"][-1] for r in R]
        tail = lambda key: [np.mean([row[key] for row in r["log"][-10:] if key in row]) for r in R]
        coll_eps = [l["cum_collision_episodes"] for l in last]
        coll_steps = [l["cum_collisions"] for l in last]
        eps = [l["cum_episodes"] for l in last]
        s80 = [steps_to_threshold(r, "eval_on_success_rate", 0.8) for r in R]
        row = {
            "method": m, "seeds": len(R),
            "collision_episodes_train": ms(coll_eps), "collision_steps_train": ms(coll_steps),
            "episodes_train": ms(eps),
            "steps_to_80pct_success": ms(s80), "n_seeds_reached_80": int(np.sum(~np.isnan(s80))),
            "final_success": ms(tail("eval_on_success_rate")),
            "final_interv_train": ms(tail("intervention_rate")),
            "mean_interv_whole_training": ms([np.mean([row["intervention_rate"] for row in r["log"]]) for r in R]),
            "final_interv_eval": ms(tail("eval_on_intervention_rate")),
            "final_filter_off_collision": ms(tail("eval_off_collision_episode_rate")) if m != "soft_penalty"
            else ms(tail("eval_on_collision_episode_rate")),
        }
        rows.append(row)
    summary["table1"] = rows
    return rows


def deployment_figure(dep, out_dir, summary, methods):
    conds = ["filter_on", "reduced_rate", "approx_model", "noisy_filter_on", "filter_off"]
    cond_lab = {"filter_on": "Nominal filter", "reduced_rate": "Filter every\n3rd cycle",
                "approx_model": "Approximate filter\n(capsules -2 cm)", "noisy_filter_on": "Nominal filter,\n5x noise",
                "filter_off": "Filter removed"}
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name, r in dep.items():
        for c in conds:
            for k, v in r[c].items():
                agg[r["method"]][c][k].append(v)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
    width = 0.8 / max(1, len([m for m in methods if m in agg]))
    x = np.arange(len(conds))
    keys = [("collision_episode_rate", "(a) Episodes with an arm-arm collision [%]"),
            ("success_rate", "(b) Task success [%] (dots = individual seeds)"),
            ("intervention_rate", "(c) Runtime filter intervention rate [%]")]
    ms_ = [m for m in methods if m in agg]
    for ai, (key, title) in enumerate(keys):
        ax = axes[ai]
        for i, m in enumerate(ms_):
            vals = [np.mean(agg[m][c][key]) * 100 for c in conds]
            xpos = x + (i - (len(ms_) - 1) / 2) * width
            ax.bar(xpos, vals, width, color=COLORS[m], label=LABELS[m], alpha=0.9)
            for j, c in enumerate(conds):
                pts = np.array(agg[m][c][key]) * 100
                jitter = np.linspace(-0.25, 0.25, len(pts)) * width if len(pts) > 1 else np.zeros(1)
                ax.scatter(xpos[j] + jitter, pts, s=7, color="black", zorder=3, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels([cond_lab[c] for c in conds], fontsize=8.5)
        ax.set_title(title)
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig2_deployment.png"), dpi=160)
    plt.close(fig)
    table = {}
    for m in ms_:
        table[m] = {c: {k: ms(agg[m][c][k]) for k in ["success_rate", "collision_episode_rate", "intervention_rate",
                                                      "margin_violation_rate", "min_clearance_mean"]} for c in conds}
        table[m]["seeds"] = len(agg[m]["filter_on"]["success_rate"])
    summary["table2"] = table
    # Filter dependence among policies that actually solve the task (a policy that never
    # approaches the other arm is trivially collision-free).
    solved_tab = {}
    for m in ms_:
        native = "filter_off" if m == "soft_penalty" else "filter_on"  # soft penalty is trained without a filter
        rows = [r for r in dep.values() if r["method"] == m and r[native]["success_rate"] >= 0.8]
        if not rows:
            solved_tab[m] = {"n": 0}
            continue
        solved_tab[m] = {"n": len(rows), "total": len([r for r in dep.values() if r["method"] == m])}
        for c in conds:
            solved_tab[m][c] = {k: ms([r[c][k] for r in rows]) for k in
                                ["success_rate", "collision_episode_rate", "intervention_rate", "margin_violation_rate"]}
            solved_tab[m][c]["collision_episodes"] = int(round(sum(r[c]["collision_episode_rate"] * 1000 for r in rows)))
    summary["table2_solved"] = solved_tab
    return table


def ablation_figure(runs, dep, out_dir, summary):
    ms_ = [m for m in ABL if m in runs]
    if len(ms_) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for m in ms_:
        band(axes[0], runs[m], "eval_on_success_rate", COLORS[m], LABELS[m], scale=100, seeds=True)
        band(axes[1], runs[m], "intervention_rate", COLORS[m], LABELS[m], sm=9, scale=100)
    axes[0].set_title("(a) Task success (noise-free policy)\nbold = mean, thin = each seed")
    axes[0].set_ylabel("success rate [%]")
    axes[1].set_title("(b) Filter intervention rate during training\n(stochastic policy; band = s.e.m.)")
    axes[1].set_ylabel("intervention rate [%]")
    for ax in axes[:2]:
        ax.set_xlabel("environment steps [M]")
    axes[0].legend(fontsize=8.5, loc="upper left")
    whole = {m: [np.mean([row["intervention_rate"] for row in r["log"]]) * 100 for r in runs[m]] for m in ms_}
    tpi = {m: [r["log"][-1]["time_s"] / r["log"][-1]["env_steps"] * 1e6 for r in runs[m]] for m in ms_}
    x = np.arange(len(ms_))
    ax = axes[2]
    vals = [np.mean(whole[m]) for m in ms_]
    ax.bar(x, vals, 0.6, color=[COLORS[m] for m in ms_], alpha=0.9)
    for i, m in enumerate(ms_):
        pts = np.array(whole[m])
        ax.scatter(i + np.linspace(-0.12, 0.12, len(pts)), pts, s=12, color="black", zorder=3, lw=0)
        ax.text(i, max(pts) + 0.4, f"{np.mean(tpi[m]):.0f} s / 1M steps", ha="center", fontsize=8.5)
    ax.set_ylim(0, max(max(v) for v in whole.values()) * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels({"frl": "correction-displacement\n(Eq. 5, claimed)", "frl_direct": "direct target a_feas",
                        "frl_meanproj": "projected mean"}[m] for m in ms_)
    ax.set_ylabel("intervention rate, whole training [%]")
    ax.set_title("(c) Filter interventions over the whole training\n(dots = seeds; label = training wall-clock)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig3b_target_ablation.png"), dpi=160)
    plt.close(fig)
    agg = defaultdict(lambda: defaultdict(list))
    for name, r in dep.items():
        if r["method"] in ms_:
            agg[r["method"]]["off_coll"].append(r["filter_off"]["collision_episode_rate"])
            agg[r["method"]]["on_int"].append(r["filter_on"]["intervention_rate"])
    summary["table3"] = {m: {"final_success": ms([np.mean([row["eval_on_success_rate"] for row in r["log"][-10:]
                                                           if "eval_on_success_rate" in row]) for r in runs[m]]),
                             "seeds": len(runs[m]),
                             "interv_whole_training": ms([v / 100 for v in whole[m]]),
                             "l_feas_mean": ms([np.mean([row["l_feas"] for row in r["log"]]) for r in runs[m]]),
                             "filter_on_intervention": ms(agg[m]["on_int"]),
                             "filter_off_collision": ms(agg[m]["off_coll"]),
                             "seconds_per_1M_steps": ms(tpi[m])} for m in ms_}


def factorial_table(runs, dep, summary):
    """2x2 factorial: {reward penalty on c_t} x {feasibility loss, Eq. 5}."""
    rows = {}
    for m in FACT:
        if m not in runs:
            continue
        R = runs[m]
        off = [d["filter_off"]["collision_episode_rate"] for d in dep.values() if d["method"] == m]
        rows[m] = {
            "seeds": len(R),
            "final_success": ms([np.mean([row["eval_on_success_rate"] for row in r["log"][-10:]
                                          if "eval_on_success_rate" in row]) for r in R]),
            "steps_to_80pct_success": ms([steps_to_threshold(r, "eval_on_success_rate", 0.8) for r in R]),
            "n80": int(np.sum(~np.isnan([steps_to_threshold(r, "eval_on_success_rate", 0.8) for r in R]))),
            "mean_interv_whole_training": ms([np.mean([row["intervention_rate"] for row in r["log"]]) for r in R]),
            "filter_off_collision": ms(off) if off else ms([np.mean([row["eval_off_collision_episode_rate"]
                                                                     for row in r["log"][-10:]
                                                                     if "eval_off_collision_episode_rate" in row])
                                                            for r in R]),
        }
    summary["table4"] = rows


def fisher_two_sided(a, n1, b, n2):
    """Two-sided Fisher exact test for a/n1 vs b/n2 successes."""
    from math import comb
    K, N = a + b, n1 + n2
    def pmf(k):
        return comb(n1, k) * comb(n2, K - k) / comb(N, K)
    p0 = pmf(a)
    return float(sum(pmf(k) for k in range(max(0, K - n2), min(K, n1) + 1) if pmf(k) <= p0 * (1 + 1e-9)))


def permutation_p(x, y, n=20000, seed=0):
    """Two-sided permutation test on the difference of means."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = abs(x.mean() - y.mean())
    z = np.concatenate([x, y])
    cnt = 0
    for _ in range(n):
        rng.shuffle(z)
        cnt += abs(z[:len(x)].mean() - z[len(x):].mean()) >= obs - 1e-12
    return float((cnt + 1) / (n + 1))


def stats_table(runs, dep, summary):
    """Each FRL variant vs. each baseline: seeds that solve the task and filter reliance."""
    def solved(r):
        return np.mean([row["eval_on_success_rate"] for row in r["log"][-10:] if "eval_on_success_rate" in row]) >= 0.8
    def whole(r):
        return np.mean([row["intervention_rate"] for row in r["log"]])
    out = {}
    for f in ["frl_noRP", "frl"]:
        if f not in runs:
            continue
        f_s = [solved(r) for r in runs[f]]
        f_i = [whole(r) for r in runs[f]]
        f_off = [d["filter_off"]["collision_episode_rate"] for d in dep.values() if d["method"] == f]
        for m in ["soft_penalty", "filter_only", "filter_rp"]:
            if m not in runs:
                continue
            b_s = [solved(r) for r in runs[m]]
            row = {"frl": f, "base": m, "frl_solved": f"{sum(f_s)}/{len(f_s)}", "base_solved": f"{sum(b_s)}/{len(b_s)}",
                   "fisher_p_solved": fisher_two_sided(sum(f_s), len(f_s), sum(b_s), len(b_s))}
            if m != "soft_penalty":
                b_i = [whole(r) for r in runs[m]]
                b_off = [d["filter_off"]["collision_episode_rate"] for d in dep.values() if d["method"] == m]
                row["perm_p_interventions"] = permutation_p(f_i, b_i)
                if b_off and f_off:
                    row["perm_p_filter_off_collisions"] = permutation_p(f_off, b_off)
            out[f"{f}_vs_{m}"] = row
    summary["stats"] = out


def write_tables(summary, out_dir):
    L = []
    if "table1" in summary:
        L.append("### Table 1 - Training benchmark (mean ± std over seeds)\n")
        L.append("| Method | Seeds | Collision episodes during training | Env steps to 80% success | Final success | "
                 "Filter interventions, whole training | Intervention rate (training, final) | "
                 "Intervention rate (noise-free policy, final) | Collision rate if filter removed (final) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in summary["table1"]:
            s80 = r["steps_to_80pct_success"]
            s80s = "n/a" if np.isnan(s80[0]) else f"{s80[0]/1e6:.2f} ± {s80[1]/1e6:.2f} M ({r['n_seeds_reached_80']}/{r['seeds']})"
            nf = r["method"] == "soft_penalty"
            L.append(f"| {LABELS[r['method']]} | {r['seeds']} | {fmt(*r['collision_episodes_train'], digits=0)} | {s80s} | "
                     f"{fmt(*r['final_success'], pct=True)} | {'-' if nf else fmt(*r['mean_interv_whole_training'], pct=True)} | "
                     f"{'-' if nf else fmt(*r['final_interv_train'], pct=True)} | "
                     f"{'-' if nf else fmt(*r['final_interv_eval'], pct=True)} | {fmt(*r['final_filter_off_collision'], pct=True)} |")
        L.append("")
    if "table2" in summary:
        L.append("### Table 2 - Deployment robustness (noise-free policy, 1000 episodes per seed and condition)\n")
        conds = ["filter_on", "reduced_rate", "approx_model", "noisy_filter_on", "filter_off"]
        L.append("| Method | " + " | ".join(f"{c}: success / collision" for c in conds) + " |")
        L.append("|---|" + "---|" * len(conds))
        for m, t in summary["table2"].items():
            cells = [f"{fmt(*t[c]['success_rate'], pct=True)} / {fmt(*t[c]['collision_episode_rate'], pct=True)}" for c in conds]
            L.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
        L.append("")
    if "table2_solved" in summary:
        L.append("### Table 2b - Filter dependence among the seeds that solve the task "
                 "(success >= 80% in the training configuration)\n")
        L.append("| Method | Solved seeds | Runtime intervention, nominal filter | Filter removed: success / collision episodes | "
                 "Filter removed: steps inside the 4 cm margin | "
                 "Approximate filter: success / collision episodes | Filter every 3rd cycle: collision episodes |")
        L.append("|---|---|---|---|---|---|---|")
        for m, t in summary["table2_solved"].items():
            if t["n"] == 0:
                L.append(f"| {LABELS[m]} | 0 | - | - | - | - | - |")
                continue
            ne = t["n"] * 1000
            L.append(f"| {LABELS[m]} | {t['n']}/{t['total']} | {fmt(*t['filter_on']['intervention_rate'], pct=True)} | "
                     f"{fmt(*t['filter_off']['success_rate'], pct=True)} / {t['filter_off']['collision_episodes']} of {ne} | "
                     f"{fmt(*t['filter_off']['margin_violation_rate'], pct=True)} | "
                     f"{fmt(*t['approx_model']['success_rate'], pct=True)} / {t['approx_model']['collision_episodes']} of {ne} | "
                     f"{t['reduced_rate']['collision_episodes']} of {ne} |")
        L.append("")
    if "table3" in summary:
        L.append("### Table 3 - Eq. (5) target form in the dual-arm task\n")
        L.append("| Target | Seeds | Filter interventions, whole training | Final success | "
                 "Runtime intervention of the frozen policy | Collision rate, filter removed | Wall-clock [s / 1M steps] |")
        L.append("|---|---|---|---|---|---|---|")
        for m, t in summary["table3"].items():
            L.append(f"| {LABELS[m]} | {t['seeds']} | {fmt(*t['interv_whole_training'], pct=True)} | {fmt(*t['final_success'], pct=True)} | "
                     f"{fmt(*t['filter_on_intervention'], pct=True)} | {fmt(*t['filter_off_collision'], pct=True)} | "
                     f"{fmt(*t['seconds_per_1M_steps'], digits=0)} |")
    if "stats" in summary:
        L.append("")
        L.append("### Statistical tests - FRL variants vs. each baseline (two-sided)\n")
        L.append("| Comparison | Seeds solving the task (FRL / baseline) | Fisher exact p | "
                 "Permutation p, interventions over training | Permutation p, collisions with filter removed |")
        L.append("|---|---|---|---|---|")
        for k, t in summary["stats"].items():
            pi = f"{t['perm_p_interventions']:.4f}" if "perm_p_interventions" in t else "-"
            pc = f"{t['perm_p_filter_off_collisions']:.4f}" if "perm_p_filter_off_collisions" in t else "-"
            L.append(f"| {LABELS[t['frl']]} vs. {LABELS[t['base']]} | {t['frl_solved']} / {t['base_solved']} | "
                     f"{t['fisher_p_solved']:.3f} | {pi} | {pc} |")
    if "table4" in summary and summary["table4"]:
        L.append("")
        L.append("### Table 4 - 2x2 factorial: what each ingredient contributes\n")
        L.append("| Method | Reward penalty on c_t | Feasibility loss (Eq. 5) | Final success | Env steps to 80% success | "
                 "Filter interventions, whole training | Collision rate, filter removed |")
        L.append("|---|---|---|---|---|---|---|")
        flags = {"filter_only": ("no", "no"), "filter_rp": ("yes", "no"), "frl_noRP": ("no", "yes"), "frl": ("yes", "yes")}
        for m, t in summary["table4"].items():
            s80 = t["steps_to_80pct_success"]
            s80s = "never" if np.isnan(s80[0]) else f"{s80[0]/1e6:.2f} ± {s80[1]/1e6:.2f} M ({t['n80']}/{t['seeds']})"
            L.append(f"| {LABELS[m]} | {flags[m][0]} | {flags[m][1]} | {fmt(*t['final_success'], pct=True)} | {s80s} | "
                     f"{fmt(*t['mean_interv_whole_training'], pct=True)} | {fmt(*t['filter_off_collision'], pct=True)} |")
    with open(os.path.join(out_dir, "tables.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    apply_style()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--deploy", default="results/deploy_eval.json")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    runs = load_runs(a.runs)
    summary = {}
    training_figure(runs, a.out, summary, MAIN)
    dep = {}
    if os.path.exists(a.deploy):
        with open(a.deploy) as fh:
            dep = json.load(fh)
        deployment_figure(dep, a.out, summary, MAIN)
    ablation_figure(runs, dep, a.out, summary)
    factorial_table(runs, dep, summary)
    stats_table(runs, dep, summary)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    write_tables(summary, a.out)


if __name__ == "__main__":
    main()
