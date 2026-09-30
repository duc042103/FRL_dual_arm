"""Shared plotting style for all figures."""

import matplotlib.pyplot as plt

COLORS = {
    "soft_penalty": "#d62728",
    "filter_only": "#7f7f7f",
    "filter_rp": "#ff7f0e",
    "frl": "#1f77b4",
    "frl_noRP": "#08306b",
    "frl_direct": "#9467bd",
    "frl_meanproj": "#2ca02c",
    "frl_sched": "#8c564b",
    # toy study
    "corr_disp": "#1f77b4",
    "direct": "#9467bd",
    "mean_proj": "#2ca02c",
}

LABELS = {
    "soft_penalty": "Soft penalty (no filter)",
    "filter_only": "Hard filter only",
    "filter_rp": "Hard filter + reward penalty",
    "frl": "FRL, full (L_feas + c_t reward penalty)",
    "frl_noRP": "FRL, L_feas only (w2 = 0)",
    "frl_direct": "FRL full, direct target a_feas",
    "frl_meanproj": "FRL full, projected-mean target",
    "frl_sched": "FRL + lambda schedule",
}


def apply_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.dpi": 110,
    })
