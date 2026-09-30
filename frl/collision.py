"""Closed-form capsule-to-capsule minimum distance between the two arms (Eq. 3).

Each link is bounded by a capsule (a line segment swept by a sphere). The
distance between two capsules is the segment-to-segment distance minus both
radii; it is evaluated in closed form and batched over environments and over
all left/right capsule pairs, as described in the specification.
"""

from __future__ import annotations

import torch

from .kinematics import NUM_CAPSULES, DualPiperKinematics


def segment_closest_points(p1, q1, p2, q2, eps: float = 1e-9):
    """Closest points between segments [p1,q1] and [p2,q2] (Ericson, RTCD 5.1.9).

    All inputs broadcast to (..., 3). Returns (c1, c2) closest points.
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    s = torch.where(denom > eps, ((b * f - c * e) / denom.clamp_min(eps)).clamp(0.0, 1.0), torch.zeros_like(denom))
    t = (b * s + f) / e.clamp_min(eps)
    s = torch.where(t < 0.0, (-c / a.clamp_min(eps)).clamp(0.0, 1.0), torch.where(t > 1.0, ((b - c) / a.clamp_min(eps)).clamp(0.0, 1.0), s))
    t = t.clamp(0.0, 1.0)
    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t[..., None]
    return c1, c2


class ArmArmDistance:
    """Minimum distance between any link of the left arm and any link of the right arm."""

    def __init__(self, kin: DualPiperKinematics, radius_scale: float = 1.0, radius_offset: float = 0.0):
        self.kin = kin
        # radius_scale / radius_offset let the evaluation build an *approximate*
        # (mis-modelled) filter while collisions are still judged with the true model.
        self.radii = (kin.radii * radius_scale + radius_offset).clamp_min(0.0)
        C = NUM_CAPSULES
        self.ii, self.jj = torch.meshgrid(torch.arange(C), torch.arange(C), indexing="ij")
        self.ii = self.ii.reshape(-1)
        self.jj = self.jj.reshape(-1)
        self.pair_radius = self.radii[self.ii] + self.radii[self.jj]

    def pairwise(self, f: dict):
        P, Q = self.kin.capsule_segments(f)  # (B, 2, C, 3)
        pL, qL = P[:, 0, self.ii], Q[:, 0, self.ii]
        pR, qR = P[:, 1, self.jj], Q[:, 1, self.jj]
        c1, c2 = segment_closest_points(pL, qL, pR, qR)
        dist = (c1 - c2).norm(dim=-1) - self.pair_radius
        return dist, c1, c2

    def dmin(self, q: torch.Tensor, f: dict | None = None) -> torch.Tensor:
        f = self.kin.fk(q) if f is None else f
        dist, _, _ = self.pairwise(f)
        return dist.min(-1).values

    def dmin_and_grad(self, q: torch.Tensor):
        """Minimum distance (B,) and its analytic gradient w.r.t. q (B, 12).

        The gradient uses the closest-point pair of the active capsule pair and
        the link Jacobians: dd/dq_L = n^T J_L(c1), dd/dq_R = -n^T J_R(c2).
        """
        f = self.kin.fk(q)
        dist, c1, c2 = self.pairwise(f)
        dmin, k = dist.min(-1)
        B = q.shape[0]
        idx = torch.arange(B, device=q.device)
        a = c1[idx, k]
        b = c2[idx, k]
        diff = a - b
        n = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        link_l = self.kin.capsule_link[self.ii[k]]
        link_r = self.kin.capsule_link[self.jj[k]]
        zeros = torch.zeros(B, dtype=torch.long, device=q.device)
        JL = self.kin.point_jacobian(f, zeros, link_l, a)  # (B, 3, 6)
        JR = self.kin.point_jacobian(f, zeros + 1, link_r, b)
        gL = torch.einsum("bi,bij->bj", n, JL)
        gR = -torch.einsum("bi,bij->bj", n, JR)
        return dmin, torch.cat([gL, gR], -1), f
