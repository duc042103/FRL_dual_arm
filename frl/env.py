"""Vectorised dual-Piper cooperative box-lift environment (kinematic simulation).

Task (exemplary embodiment of the specification): a compact box whose mass
exceeds the rated payload of one Piper (1.5 kg) must be grasped by both
grippers and lifted. Each gripper must reach a grasp point on the top edge of
its side of the box. The preferred grasp orientation is the natural Piper
top-down pinch (approach axis pitched 30 deg forward from vertical). For narrow
boxes the two gripper housings in that orientation violate the arm-arm safety
margin (or collide), so the constrained optimum lies on the boundary of the
feasible set: the grippers must tilt outward. This makes the self-collision
constraint *binding*, which is the regime in which a filter-only policy becomes
dependent on its filter.

Everything is batched over ``num_envs`` environments in PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import torch

from .kinematics import DualPiperKinematics
from .safety_filter import ActionPipeline, FilterConfig, PipelineConfig

START_LEFT = [-0.10, 1.45, -1.30, 0.0, 1.05, 0.0]
MIRROR = [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0]


@dataclass
class EnvConfig:
    num_envs: int = 256
    base_separation: float = 0.50
    max_steps: int = 80
    start_noise: float = 0.05          # rad, uniform perturbation of the start pose
    # Box distribution
    box_x: tuple = (0.26, 0.32)
    box_y: tuple = (-0.03, 0.03)
    box_w: tuple = (0.07, 0.11)
    box_h: float = 0.08
    grasp_depth: float = 0.01          # grasp point below the top edge
    grasp_tol: float = 0.02
    nominal_pitch: float = 30.0        # deg, preferred approach axis pitched forward from vertical
    grasp_max_tilt: float = 0.785      # 45 deg away from the preferred approach axis
    drop_sep_tol: float = 0.03
    lift_height: float = 0.08
    # Domain randomisation (Stage A): sensor and actuation noise.
    obs_joint_noise: float = 0.002
    act_joint_noise: float = 0.002
    # Reward, Eq. (12) plus dense task shaping shared by all methods.
    w_progress: float = 20.0           # potential-based shaping per metre of mean grasp-point error
    w_reach: float = 0.5
    reach_scale: float = 0.03
    w_ori: float = 0.5
    w_hold: float = 0.5
    w_lift: float = 1.0
    w_lift_progress: float = 20.0      # potential-based shaping per metre of box height
    w_grasp: float = 1.0
    w_success: float = 5.0             # w1
    w_interv: float = 0.0              # w2 (filter-intervention quantity c_t)
    w_step: float = 0.01               # w3
    w_collision: float = 10.0          # soft-penalty baseline only
    w_drop: float = 2.0
    collision_terminates: bool = True
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)


class DualPiperBoxLift:
    obs_dim = 62
    act_dim = 12

    def __init__(self, cfg: EnvConfig, seed: int = 0, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.kin = DualPiperKinematics(cfg.base_separation, device=device)
        self.pipe = ActionPipeline(self.kin, cfg.pipeline, cfg.filt)
        self.true_dist = self.pipe.true_dist
        N = cfg.num_envs
        ql = torch.tensor(START_LEFT)
        self.q_start = torch.cat([ql, ql * torch.tensor(MIRROR)]).to(self.device)
        self.q = self.q_start.repeat(N, 1)
        self.q_prev = self.q.clone()
        self.prev_action = torch.zeros(N, 12, device=self.device)
        self.phase = torch.zeros(N, dtype=torch.long, device=self.device)
        self.box = torch.zeros(N, 3, device=self.device)
        self.box_w = torch.zeros(N, device=self.device)
        self.box_z0 = torch.zeros(N, device=self.device)
        self.grasp_off = torch.zeros(N, 3, device=self.device)
        self.grasp_sep = torch.zeros(N, device=self.device)
        self.t = torch.zeros(N, dtype=torch.long, device=self.device)
        self.lifted = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.err_prev = torch.zeros(N, 2, device=self.device)
        self.filter_on = True
        self.step_count = 0
        self.reset_idx(torch.arange(N, device=self.device))
        self.f = self.kin.fk(self.q)

    def randomize_episode_phase(self):
        """Stagger the first episodes so that every rollout window covers all episode phases."""
        self.t = torch.randint(0, self.cfg.max_steps, (self.cfg.num_envs,), generator=self.gen).to(self.device)

    # ------------------------------------------------------------------ utils
    def _rand(self, n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=self.gen).to(self.device)

    def grasp_points(self) -> torch.Tensor:
        z = self.box[:, 2] + self.cfg.box_h - self.cfg.grasp_depth
        left = torch.stack([self.box[:, 0], self.box[:, 1] + self.box_w / 2, z], -1)
        right = torch.stack([self.box[:, 0], self.box[:, 1] - self.box_w / 2, z], -1)
        return torch.stack([left, right], 1)

    def tilt(self, f: dict) -> torch.Tensor:
        """Angle between each gripper approach axis and the preferred approach axis (B, 2)."""
        p = math.radians(self.cfg.nominal_pitch)
        n = torch.tensor([math.sin(p), 0.0, -math.cos(p)], device=self.device)
        return torch.acos((f["approach"] * n).sum(-1).clamp(-1.0, 1.0))

    def reset_idx(self, ids: torch.Tensor):
        n = ids.numel()
        if n == 0:
            return
        c = self.cfg
        noise = (torch.rand(n, 12, generator=self.gen).to(self.device) * 2 - 1) * c.start_noise
        self.q[ids] = self.kin.clamp_limits(self.q_start + noise)
        self.q_prev[ids] = self.q[ids]
        self.prev_action[ids] = 0.0
        self.phase[ids] = 0
        self.box[ids, 0] = self._rand(n, *c.box_x)
        self.box[ids, 1] = self._rand(n, *c.box_y)
        self.box[ids, 2] = 0.0  # box bottom rests on the table (z of the box reference = bottom)
        self.box_z0[ids] = 0.0
        self.box_w[ids] = self._rand(n, *c.box_w)
        self.t[ids] = 0
        self.lifted[ids] = False
        f = self.kin.fk(self.q[ids])
        self.err_prev[ids] = (f["tcp_pos"] - self.grasp_points()[ids]).norm(dim=-1)

    # ------------------------------------------------------------ observation
    def observe(self) -> torch.Tensor:
        c = self.cfg
        f = self.f
        qn = self.q + c.obs_joint_noise * torch.randn(self.q.shape, generator=self.gen).to(self.device)
        dq = (self.q - self.q_prev) / 0.05
        tcp = f["tcp_pos"].reshape(-1, 6)
        app = f["approach"].reshape(-1, 6)
        rel = (self.grasp_points() - f["tcp_pos"]).reshape(-1, 6) / 0.1
        lift = ((self.box[:, 2] - self.box_z0) / c.lift_height)[:, None]
        sep = (f["tcp_pos"][:, 0] - f["tcp_pos"][:, 1]).norm(dim=-1)
        sep_dev = ((sep - self.grasp_sep) * (self.phase == 1).float() / c.drop_sep_tol)[:, None]
        obs = torch.cat([
            qn, dq, tcp, app, rel, self.box, self.box_w[:, None] * 10.0, lift, sep_dev,
            self.phase[:, None].float(), self.prev_action, (self.t.float() / c.max_steps)[:, None],
        ], -1)
        return obs

    # ------------------------------------------------------------------- step
    def step(self, a_raw: torch.Tensor) -> tuple:
        c = self.cfg
        N = c.num_envs
        f_cur = self.f
        # Reduced-rate filter (deployment test): filter active only every `period` steps.
        if c.filt.period > 1:
            filt_on = (self.t % c.filt.period) == 0
        else:
            filt_on = torch.full((N,), bool(self.filter_on), dtype=torch.bool, device=self.device)
        if not self.filter_on:
            filt_on = torch.zeros(N, dtype=torch.bool, device=self.device)
        out = self.pipe(self.q, a_raw, filt_on, f_cur)
        q_next = out["q_cmd"]
        if c.act_joint_noise > 0:
            q_next = self.kin.clamp_limits(
                q_next + c.act_joint_noise * torch.randn(q_next.shape, generator=self.gen).to(self.device))
        # --- collision check with the TRUE capsule model at the end point and mid-point of the step
        f_next = self.kin.fk(q_next)
        d_end = self.true_dist.dmin(q_next, f_next)
        d_mid = self.true_dist.dmin(0.5 * (self.q + q_next))
        d_path = torch.minimum(d_end, d_mid)
        collision = d_path < 0.0

        self.q_prev = self.q
        self.q = q_next
        self.f = f_next
        self.t += 1

        # --- task logic
        gp = self.grasp_points()
        err = (f_next["tcp_pos"] - gp).norm(dim=-1)  # (N, 2)
        tilt = self.tilt(f_next)
        grasped_now = (self.phase == 0) & (err < c.grasp_tol).all(-1) & (tilt < c.grasp_max_tilt).all(-1)
        mid = f_next["tcp_pos"].mean(1)
        sep = (f_next["tcp_pos"][:, 0] - f_next["tcp_pos"][:, 1]).norm(dim=-1)
        if bool(grasped_now.any()):
            self.phase[grasped_now] = 1
            self.grasp_off[grasped_now] = self.box[grasped_now] - mid[grasped_now]
            self.grasp_sep[grasped_now] = sep[grasped_now]
        holding = self.phase == 1
        box_z_prev = self.box[:, 2].clone()
        dropped = holding & (((sep - self.grasp_sep).abs() > c.drop_sep_tol) | (tilt > c.grasp_max_tilt + 0.26).any(-1))
        self.box = torch.where(holding[:, None], mid + self.grasp_off, self.box)
        lift_frac = ((self.box[:, 2] - self.box_z0) / c.lift_height).clamp(0.0, 1.0)
        new_success = holding & ~dropped & (lift_frac >= 1.0) & ~self.lifted
        self.lifted |= new_success

        # --- reward, Eq. (12) + shaping
        a_clip = a_raw.clamp(-1.0, 1.0)
        corr = (out["a_feas"] - a_clip).norm(dim=-1)
        c_t = out["m"].float() * corr.clamp(max=1.0)
        was_approaching = ~holding | grasped_now
        r_progress = c.w_progress * (self.err_prev - err).mean(-1) * was_approaching.float()
        self.err_prev = err
        r_reach = torch.where(holding, torch.full_like(corr, c.w_reach),
                              c.w_reach * torch.exp(-err / c.reach_scale).mean(-1)) + r_progress
        r_ori = c.w_ori * torch.cos(tilt).mean(-1)
        r_hold = c.w_hold * holding.float()
        r_lift = (c.w_lift * lift_frac + c.w_lift_progress * (self.box[:, 2] - box_z_prev)) * holding.float()
        reward = (r_reach + r_ori + r_hold + r_lift + c.w_grasp * grasped_now.float()
                  + c.w_success * new_success.float() - c.w_interv * c_t - c.w_step
                  - c.w_drop * dropped.float())
        terminal = dropped.clone()
        if c.collision_terminates:
            reward = reward - c.w_collision * collision.float()
            terminal |= collision
        timeout = (self.t >= c.max_steps) & ~terminal
        done = terminal | timeout

        info = {
            "m": out["m"], "a_feas": out["a_feas"], "q_des": out["q_des"], "n_iter": out["n_iter"],
            "fallback": out["fallback"], "d_path": d_path, "collision": collision,
            "margin_violation": d_path < c.filt.d_safe, "success": self.lifted.clone(),
            "grasped": holding.clone(), "grasp_event": grasped_now, "dropped": dropped, "terminal": terminal, "timeout": timeout,
            "done": done, "c_t": c_t, "corr": corr, "tilt": tilt.mean(-1), "filter_on": filt_on,
        }
        self.prev_action = a_clip
        # --- auto reset
        done_ids = torch.nonzero(done).squeeze(-1)
        if done_ids.numel() > 0:
            self.reset_idx(done_ids)
            self.f = self.kin.fk(self.q)
        self.step_count += 1
        return self.observe(), reward, done, info
