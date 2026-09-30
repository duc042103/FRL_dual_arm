"""Batched kinematics of a dual AgileX Piper system (PyTorch).

The Denavit-Hartenberg table is copied verbatim from the official AgileX SDK
(``piper_sdk==0.6.2``, ``piper_sdk/kinematics/piper_fk.py``, ``dh_is_offset=0x01``),
which is the SDK pinned by the reference dual-Piper towel-folding project.
The joint limits are the Piper hardware limits used by that project's safety
layer (``src/piper_towel_folding/safety.py``).

Convention: modified (Craig) DH, ``T_i = RotX(alpha_i) TransX(a_i) RotZ(q_i + theta_i) TransZ(d_i)``.
Frame ``M_i = T_0 ... T_i``; the z-axis of ``M_i`` is the axis of joint ``i``
(0-indexed) and its origin lies on that axis.

World frame: x forward, y to the left, z up. Both arms face +x and are mounted
on the table plane; the left base sits at +y, the right base at -y.
A dual-arm joint vector has 12 entries: ``[left j1..j6, right j1..j6]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Piper DH table (piper_sdk 0.6.2, dh_is_offset = 0x01). Units: metres, radians.
# ---------------------------------------------------------------------------
DH_A = [0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0]
DH_ALPHA = [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2]
DH_THETA = [0.0, -math.pi * 172.22 / 180, -102.78 / 180 * math.pi, 0.0, 0.0, 0.0]
DH_D = [0.123, 0.0, 0.0, 0.25075, 0.0, 0.091]

# Piper hardware joint limits (reference project safety.py, AgileX SDK 0.6.2).
JOINT_LOWER = [-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439]
JOINT_UPPER = [2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439]

# Flange-to-fingertip (TCP) distance of the Piper parallel-jaw gripper.
TCP_OFFSET = 0.135


@dataclass
class Capsule:
    name: str
    radius: float
    link: int  # index of the frame the capsule is rigidly attached to (-1 = static base)


# Capsule model of one Piper arm. Radii are conservative bounding radii of the
# Piper links (upper arm / forearm tubes, wrist housing, gripper housing, fingers).
FINGER_LENGTH = 0.06
CAPSULES = [
    Capsule("base", 0.060, -1),          # base column, (0,0,0) -> shoulder
    Capsule("upper_arm", 0.045, 1),      # shoulder (M1) -> elbow (M2)
    Capsule("forearm", 0.040, 2),        # elbow (M2) -> wrist centre (M3)
    Capsule("wrist", 0.040, 5),          # wrist centre (M4) -> flange (M5)
    Capsule("gripper_housing", 0.045, 5),  # flange -> finger root
    Capsule("fingers", 0.010, 5),        # finger root -> TCP
]
NUM_CAPSULES = len(CAPSULES)


def _dh_transform(alpha: float, a: float, theta: torch.Tensor, d: float) -> torch.Tensor:
    """Modified-DH link transform, batched over ``theta`` (any shape) -> (..., 4, 4)."""
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    rows = [
        torch.stack([ct, -st, zeros, zeros + a], -1),
        torch.stack([st * ca, ct * ca, zeros - sa, zeros - sa * d], -1),
        torch.stack([st * sa, ct * sa, zeros + ca, zeros + ca * d], -1),
        torch.stack([zeros, zeros, zeros, ones], -1),
    ]
    return torch.stack(rows, -2)


class DualPiperKinematics:
    """Forward kinematics, Jacobians and capsule geometry for two Piper arms."""

    def __init__(self, base_separation: float = 0.50, device: str = "cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        half = base_separation / 2.0
        # (2, 3) base positions: left at +y, right at -y.
        self.base_pos = torch.tensor([[0.0, half, 0.0], [0.0, -half, 0.0]], device=device, dtype=dtype)
        lower = torch.tensor(JOINT_LOWER, device=device, dtype=dtype)
        upper = torch.tensor(JOINT_UPPER, device=device, dtype=dtype)
        self.q_lower = torch.cat([lower, lower])
        self.q_upper = torch.cat([upper, upper])
        self.radii = torch.tensor([c.radius for c in CAPSULES], device=device, dtype=dtype)
        self.capsule_link = torch.tensor([c.link for c in CAPSULES], device=device)

    # ------------------------------------------------------------------ FK
    def fk(self, q: torch.Tensor) -> dict:
        """Forward kinematics.

        Args:
            q: (B, 12) joint positions.
        Returns dict with
            R: (B, 2, 6, 3, 3) frame rotations in world,
            o: (B, 2, 6, 3) frame origins in world,
            tcp_pos: (B, 2, 3), tcp_R: (B, 2, 3, 3), approach: (B, 2, 3) gripper z-axis.
        """
        B = q.shape[0]
        qa = q.view(B, 2, 6)
        M = torch.eye(4, device=q.device, dtype=q.dtype).expand(B, 2, 4, 4).clone()
        M[..., :3, 3] = self.base_pos
        Rs, os_ = [], []
        for i in range(6):
            T = _dh_transform(DH_ALPHA[i], DH_A[i], qa[..., i] + DH_THETA[i], DH_D[i])
            M = M @ T
            Rs.append(M[..., :3, :3])
            os_.append(M[..., :3, 3])
        R = torch.stack(Rs, 2)
        o = torch.stack(os_, 2)
        approach = R[:, :, 5, :, 2]
        tcp_pos = o[:, :, 5] + TCP_OFFSET * approach
        return {"R": R, "o": o, "tcp_pos": tcp_pos, "tcp_R": R[:, :, 5], "approach": approach}

    # ------------------------------------------------------------ capsules
    def capsule_segments(self, f: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Capsule axis end points. Returns (P, Q) each (B, 2, C, 3)."""
        o, z6 = f["o"], f["approach"]
        base = self.base_pos.expand(o.shape[0], 2, 3)
        flange = o[:, :, 5]
        finger_root = flange + (TCP_OFFSET - FINGER_LENGTH) * z6
        # o[:, :, 0] is the shoulder (top of the base column, on the joint-1 axis).
        P = torch.stack([base, o[:, :, 1], o[:, :, 2], o[:, :, 4], flange, finger_root], 2)
        Q = torch.stack([o[:, :, 0], o[:, :, 2], o[:, :, 3], flange, finger_root, f["tcp_pos"]], 2)
        return P, Q

    # ------------------------------------------------------------ Jacobians
    def point_jacobian(self, f: dict, arm: torch.Tensor, link: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Positional Jacobian (B, 3, 6) of world point ``p`` rigidly attached to ``link`` of ``arm``.

        Args:
            arm: (B,) arm index (0 left, 1 right); link: (B,) frame index (-1 static);
            p: (B, 3) world point.
        """
        B = p.shape[0]
        idx = torch.arange(B, device=p.device)
        z = f["R"][idx, arm, :, :, 2]  # (B, 6, 3)
        o = f["o"][idx, arm]  # (B, 6, 3)
        cols = torch.cross(z, p[:, None, :] - o, dim=-1)  # (B, 6, 3)
        mask = (torch.arange(6, device=p.device)[None, :] <= link[:, None]).to(p.dtype)
        return (cols * mask[..., None]).transpose(1, 2)

    def tcp_jacobian(self, f: dict) -> torch.Tensor:
        """Geometric Jacobian of both TCPs, (B, 2, 6, 6): rows = [linear; angular], cols = joints."""
        z = f["R"][..., :, :, 2]  # (B, 2, 6, 3)
        o = f["o"]
        p = f["tcp_pos"][:, :, None, :]
        lin = torch.cross(z, p - o, dim=-1)  # (B, 2, 6, 3)
        return torch.cat([lin, z], -1).transpose(-1, -2)

    def clamp_limits(self, q: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(q, self.q_upper), self.q_lower)


# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------
def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) -> rotation vector (..., 3). Robust for small angles."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cos)
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    sin = torch.sin(angle)
    scale = torch.where(sin.abs() > 1e-4, angle / (2.0 * sin), 0.5 + angle ** 2 / 12.0)
    return w * scale[..., None]


def pose_difference(p_new: torch.Tensor, R_new: torch.Tensor, p_cur: torch.Tensor, R_cur: torch.Tensor) -> torch.Tensor:
    """Task-space displacement x_new (-) x_cur = [p_new - p_cur, log(R_new R_cur^T)] (..., 6)."""
    dR = R_new @ R_cur.transpose(-1, -2)
    return torch.cat([p_new - p_cur, so3_log(dR)], -1)
