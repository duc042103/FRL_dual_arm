"""FIG. 0 - the dual-Piper capsule model and why the constraint is binding.

Left: preferred grasp orientation (approach axis pitched 30 deg forward) for a
7.5 cm box - the gripper housings violate the safety margin (red segment =
closest-point pair). Right: grippers tilted outward - feasible.
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frl.collision import ArmArmDistance  # noqa: E402
from frl.env import START_LEFT, MIRROR  # noqa: E402
from frl.kinematics import CAPSULES, DualPiperKinematics  # noqa: E402

kin = DualPiperKinematics(0.5)
dist = ArmArmDistance(kin)


def solve(q, targets, zt, iters=400, step=0.3):
    for _ in range(iters):
        f = kin.fk(q)
        e = torch.cat([targets - f["tcp_pos"], torch.cross(f["approach"], zt, dim=-1).clamp(-0.3, 0.3)], -1)
        J = kin.tcp_jacobian(f)
        dq = (J.transpose(-1, -2) @ torch.linalg.solve(J @ J.transpose(-1, -2) + 1e-3 * torch.eye(6), e.unsqueeze(-1)))
        q = kin.clamp_limits(q + step * dq.squeeze(-1).reshape(q.shape[0], 12))
    return q


def approach(pitch_deg, out_deg):
    p, t = math.radians(pitch_deg), math.radians(out_deg)
    zl = torch.tensor([math.sin(p) * math.cos(t), -math.sin(t), -math.cos(p) * math.cos(t)])
    return torch.stack([zl, zl * torch.tensor([1.0, -1.0, 1.0])])[None]


def capsule_mesh(p, q, r, n=14):
    v = q - p
    L = np.linalg.norm(v)
    v = v / max(L, 1e-9)
    a = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
    n1 = np.cross(v, a); n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    th = np.linspace(0, 2 * np.pi, n)
    s = np.linspace(0, L, 2)
    T, S = np.meshgrid(th, s)
    X = p[0] + v[0] * S + r * (np.cos(T) * n1[0] + np.sin(T) * n2[0])
    Y = p[1] + v[1] * S + r * (np.cos(T) * n1[1] + np.sin(T) * n2[1])
    Z = p[2] + v[2] * S + r * (np.cos(T) * n1[2] + np.sin(T) * n2[2])
    return X, Y, Z


def draw(ax, q, w, title):
    f = kin.fk(q)
    P, Q = kin.capsule_segments(f)
    for arm, col in [(0, "#1f77b4"), (1, "#ff7f0e")]:
        for c in range(len(CAPSULES)):
            X, Y, Z = capsule_mesh(P[0, arm, c].numpy(), Q[0, arm, c].numpy(), CAPSULES[c].radius)
            ax.plot_surface(X, Y, Z, color=col, alpha=0.35, linewidth=0)
            ax.plot(*zip(P[0, arm, c].numpy(), Q[0, arm, c].numpy()), color=col, lw=1.5)
    # box
    x0, y0, h = 0.29, 0.0, 0.08
    xs = [x0 - 0.04, x0 + 0.04]
    ys = [y0 - w / 2, y0 + w / 2]
    for zz in [0, h]:
        ax.plot([xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[0], ys[0], ys[1], ys[1], ys[0]], [zz] * 5, color="0.3", lw=1)
    for xx in xs:
        for yy in ys:
            ax.plot([xx, xx], [yy, yy], [0, h], color="0.3", lw=1)
    d, c1, c2 = dist.pairwise(f)
    k = int(d[0].argmin())
    a, b = c1[0, k].numpy(), c2[0, k].numpy()
    dm = float(d[0, k])
    ax.plot(*zip(a, b), color="#d62728" if dm < 0.04 else "#2ca02c", lw=3)
    ax.set_title(f"{title}\nd_min = {100*dm:.1f} cm  (d_safe = 4.0 cm)", fontsize=10)
    ax.set_xlim(-0.05, 0.45); ax.set_ylim(-0.3, 0.3); ax.set_zlim(0, 0.5)
    ax.set_box_aspect((0.5, 0.6, 0.5))
    ax.view_init(elev=22, azim=-150)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")


def main(out_dir="results"):
    ql = torch.tensor(START_LEFT)
    q0 = torch.cat([ql, ql * torch.tensor(MIRROR)])[None]
    w = 0.075
    tgt = torch.tensor([[[0.29, w / 2, 0.07], [0.29, -w / 2, 0.07]]])
    q_nom = solve(q0.clone(), tgt, approach(30, 0))
    q_tilt = solve(q0.clone(), tgt, approach(30, 28))
    fig = plt.figure(figsize=(13, 5.2))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw(ax, q0, w, "Start pose")
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    draw(ax, q_nom, w, "Preferred grasp orientation: INFEASIBLE")
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    draw(ax, q_tilt, w, "Grippers tilted outward 28 deg: feasible")
    fig.suptitle("Dual AgileX Piper capsule model (DH from piper_sdk 0.6.2), bimanual lift of a 7.5 cm box", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    path = os.path.join(out_dir, "figures", "fig0_setup.png")
    fig.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
