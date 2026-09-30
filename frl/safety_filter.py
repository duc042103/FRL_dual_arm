"""Cross-space action pipeline: IK (Eq. 2) -> feasibility test (Eq. 3) ->
projection (Eq. 4 / 4a) -> FK back-mapping (Eq. 4b).

A task-space raw action ``a_raw`` (12-D: per gripper a translational
displacement and a rotation vector, in normalised units) is mapped to a desired
joint configuration by damped-least-squares differential IK, tested against the
arm-arm minimum-distance constraint ``d_min(q) >= d_safe``, projected onto the
feasible set by the iterative linearised projection of Eq. (4a) when infeasible
(with the current configuration as a feasible fallback), and mapped back to a
task-space corrected action ``a_feas`` by forward kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .collision import ArmArmDistance
from .kinematics import DualPiperKinematics, pose_difference


@dataclass
class FilterConfig:
    enabled: bool = True
    d_safe: float = 0.04            # safety margin of Eq. (3) [m]
    iter_cap: int = 10              # iteration cap of Eq. (4a)
    overshoot: float = 0.002        # target d_safe + overshoot in each linearised step [m]
    max_proj_step: float = 0.10     # per-iteration joint step cap [rad]
    joint_weights: list = field(default_factory=lambda: [1.0] * 12)  # diag(W)
    # Model error of an *approximate* filter (used only in deployment tests).
    radius_scale: float = 1.0
    radius_offset: float = 0.0
    # Run the filter only every ``period`` control steps (reduced-rate filter test).
    period: int = 1


@dataclass
class PipelineConfig:
    trans_scale: float = 0.01       # metres per unit of normalised translational action
    rot_scale: float = 0.06         # radians per unit of normalised rotational action
    dls_lambda: float = 0.05        # damping of the DLS differential IK
    motion_bound: float = 0.015     # per-step bound on the motion of every link point [m]


class ActionPipeline:
    def __init__(self, kin: DualPiperKinematics, pcfg: PipelineConfig, fcfg: FilterConfig):
        self.kin = kin
        self.pcfg = pcfg
        self.fcfg = fcfg
        self.filter_dist = ArmArmDistance(kin, fcfg.radius_scale, fcfg.radius_offset)
        self.true_dist = ArmArmDistance(kin)
        w = torch.tensor(fcfg.joint_weights, dtype=kin.dtype, device=kin.device)
        self.w_inv = 1.0 / w
        s = [pcfg.trans_scale] * 3 + [pcfg.rot_scale] * 3
        self.scale = torch.tensor(s + s, dtype=kin.dtype, device=kin.device)

    # --------------------------------------------------------------- Eq. (2)
    def inverse_kinematics(self, q_cur: torch.Tensor, f_cur: dict, a: torch.Tensor) -> torch.Tensor:
        """DLS differential IK: q_des = q_cur + J^T (J J^T + lambda^2 I)^-1 [dp; dphi].

        Joint-limit aware: joints whose step would leave the hardware range are
        saturated at the limit, their contribution is removed from the target
        twist, and the remaining joints are re-solved (two passes).
        """
        B = q_cur.shape[0]
        e = (a.clamp(-1.0, 1.0) * self.scale).view(B, 2, 6)
        J = self.kin.tcp_jacobian(f_cur)  # (B, 2, 6, 6)
        lam2 = self.pcfg.dls_lambda ** 2
        eye = torch.eye(6, dtype=J.dtype, device=J.device)
        qc = q_cur.view(B, 2, 6)
        lo, hi = self.kin.q_lower.view(2, 6), self.kin.q_upper.view(2, 6)
        free = torch.ones(B, 2, 6, dtype=J.dtype, device=J.device)
        dq_fixed = torch.zeros(B, 2, 6, dtype=J.dtype, device=J.device)
        for _ in range(3):
            Jf = J * free[..., None, :]
            e_res = e - (J @ dq_fixed.unsqueeze(-1)).squeeze(-1)
            y = torch.linalg.solve(Jf @ Jf.transpose(-1, -2) + lam2 * eye, e_res.unsqueeze(-1))
            dq = (Jf.transpose(-1, -2) @ y).squeeze(-1) + dq_fixed
            q_new = qc + dq
            over = ((q_new > hi) | (q_new < lo)) & (free > 0)
            if not bool(over.any()):
                break
            sat = torch.where(q_new > hi, hi - qc, lo - qc)
            dq_fixed = torch.where(over, sat, dq_fixed)
            free = free * (~over).to(J.dtype)
        q_des = self.kin.clamp_limits(q_cur + dq.reshape(B, 12))
        # No IK solution (numerical failure): hold the current configuration (claim 19).
        bad = ~torch.isfinite(q_des).all(-1, keepdim=True)
        return torch.where(bad, q_cur, q_des)

    def _max_point_motion(self, f_a: dict, f_b: dict) -> torch.Tensor:
        """Max displacement over all capsule end points (exact max over each rigid segment)."""
        Pa, Qa = self.kin.capsule_segments(f_a)
        Pb, Qb = self.kin.capsule_segments(f_b)
        return torch.maximum((Pa - Pb).norm(dim=-1).amax((1, 2)), (Qa - Qb).norm(dim=-1).amax((1, 2)))

    def enforce_motion_bound(self, q_cur: torch.Tensor, f_cur: dict, q_des: torch.Tensor):
        """Scale the joint step so that no link point moves more than ``motion_bound``."""
        for _ in range(2):
            f_des = self.kin.fk(q_des)
            disp = self._max_point_motion(f_des, f_cur)
            k = (self.pcfg.motion_bound / disp.clamp_min(1e-9)).clamp(max=1.0)
            if bool((k >= 0.999).all()):
                break
            q_des = q_cur + (q_des - q_cur) * k[:, None]
        return q_des

    # ------------------------------------------------------ Eq. (3), (4), (4a)
    def project(self, q_cur: torch.Tensor, q_des: torch.Tensor):
        """Safety filter. Returns (q_cmd, m, n_iter, fallback)."""
        d_safe = self.fcfg.d_safe
        d_des = self.filter_dist.dmin(q_des)
        infeasible = d_des < d_safe
        q_cmd = q_des.clone()
        n_iter = torch.zeros(q_des.shape[0], dtype=torch.long, device=q_des.device)
        fallback = torch.zeros_like(infeasible)
        if not bool(infeasible.any()):
            return q_cmd, infeasible, n_iter, fallback
        idx = torch.nonzero(infeasible).squeeze(-1)
        q = q_des[idx].clone()
        active = torch.ones(idx.shape[0], dtype=torch.bool, device=q.device)
        for it in range(self.fcfg.iter_cap):
            d, g, _ = self.filter_dist.dmin_and_grad(q)
            active = d < d_safe
            if not bool(active.any()):
                break
            viol = (d_safe + self.fcfg.overshoot - d).clamp_min(0.0)
            wg = self.w_inv * g
            denom = (g * wg).sum(-1).clamp_min(1e-9)
            dq = (viol / denom)[:, None] * wg
            cap = (self.fcfg.max_proj_step / dq.abs().amax(-1).clamp_min(1e-9)).clamp(max=1.0)
            dq = dq * cap[:, None] * active[:, None]
            q = self.kin.clamp_limits(q + dq)
            n_iter[idx] += active.long()
        d_final = self.filter_dist.dmin(q)
        still_bad = d_final < d_safe
        # Feasible fallback: the current configuration (feasible by induction).
        q = torch.where(still_bad[:, None], q_cur[idx], q)
        q_cmd[idx] = q
        fallback[idx] = still_bad
        return q_cmd, infeasible, n_iter, fallback

    # ----------------------------------------------------------- Eq. (4b)
    def back_map(self, f_cur: dict, f_cmd: dict) -> torch.Tensor:
        """Corrected task-space action a_feas = x_cmd (-) x_cur, in normalised units."""
        B = f_cur["tcp_pos"].shape[0]
        diff = pose_difference(f_cmd["tcp_pos"], f_cmd["tcp_R"], f_cur["tcp_pos"], f_cur["tcp_R"])
        return diff.reshape(B, 12) / self.scale

    # ----------------------------------------------------------- full chain
    def __call__(self, q_cur: torch.Tensor, a_raw: torch.Tensor, filter_on: bool | torch.Tensor = True,
                 f_cur: dict | None = None) -> dict:
        f_cur = self.kin.fk(q_cur) if f_cur is None else f_cur
        q_des = self.inverse_kinematics(q_cur, f_cur, a_raw)
        q_des = self.enforce_motion_bound(q_cur, f_cur, q_des)
        B = q_cur.shape[0]
        if isinstance(filter_on, bool):
            filter_on = torch.full((B,), filter_on, dtype=torch.bool, device=q_cur.device)
        q_cmd = q_des.clone()
        m = torch.zeros(B, dtype=torch.bool, device=q_cur.device)
        n_iter = torch.zeros(B, dtype=torch.long, device=q_cur.device)
        fallback = torch.zeros(B, dtype=torch.bool, device=q_cur.device)
        if self.fcfg.enabled and bool(filter_on.any()):
            sub = torch.nonzero(filter_on).squeeze(-1)
            qc, mm, ni, fb = self.project(q_cur[sub], q_des[sub])
            q_cmd[sub], m[sub], n_iter[sub], fallback[sub] = qc, mm, ni, fb
        f_cmd = self.kin.fk(q_cmd)
        a_feas = self.back_map(f_cur, f_cmd)
        return {"q_des": q_des, "q_cmd": q_cmd, "a_feas": a_feas, "m": m, "n_iter": n_iter,
                "fallback": fallback, "f_cmd": f_cmd}
