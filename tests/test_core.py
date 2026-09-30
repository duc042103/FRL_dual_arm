"""Unit tests for kinematics, capsule distance, safety filter and the Eq. (5) target."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frl.collision import ArmArmDistance, segment_closest_points  # noqa: E402
from frl.env import DualPiperBoxLift, EnvConfig  # noqa: E402
from frl.kinematics import DualPiperKinematics  # noqa: E402
from frl.safety_filter import ActionPipeline, FilterConfig, PipelineConfig  # noqa: E402

torch.manual_seed(0)


def test_fk_matches_piper_sdk():
    sdk = pytest.importorskip("piper_sdk.kinematics.piper_fk")
    fk = sdk.C_PiperForwardKinematics(0x01)
    kin = DualPiperKinematics(base_separation=0.0)
    for _ in range(20):
        q = (torch.rand(6) * (kin.q_upper[:6] - kin.q_lower[:6]) + kin.q_lower[:6]).double()
        ref = fk.CalFK(q.tolist())[-1]  # flange pose, mm
        f = kin.fk(torch.cat([q, q]).float()[None])
        flange = f["o"][0, 0, 5] * 1000.0
        assert torch.allclose(flange, torch.tensor(ref[:3], dtype=torch.float32), atol=0.05)


def test_segment_distance_against_sampling():
    for _ in range(50):
        p1, q1, p2, q2 = torch.randn(4, 3)
        c1, c2 = segment_closest_points(p1, q1, p2, q2)
        s = torch.linspace(0, 1, 201)
        a = p1 + (q1 - p1) * s[:, None]
        b = p2 + (q2 - p2) * s[:, None]
        brute = torch.cdist(a, b).min()
        assert (c1 - c2).norm() <= brute + 1e-4
        assert (c1 - c2).norm() >= brute - 2e-2


def test_distance_gradient_matches_finite_difference():
    kin = DualPiperKinematics()
    dist = ArmArmDistance(kin)
    env = DualPiperBoxLift(EnvConfig(num_envs=1))
    q = (env.q_start + 0.3 * torch.randn(8, 12)).double()
    kin64 = DualPiperKinematics(dtype=torch.float64)
    dist64 = ArmArmDistance(kin64)
    d, g, _ = dist64.dmin_and_grad(q)
    eps = 1e-6
    for i in range(12):
        dq = torch.zeros(12, dtype=torch.float64)
        dq[i] = eps
        fd = (dist64.dmin(q + dq) - dist64.dmin(q - dq)) / (2 * eps)
        assert torch.allclose(g[:, i], fd, atol=1e-4)
    assert dist is not None


def test_filter_returns_feasible_configuration():
    kin = DualPiperKinematics()
    pipe = ActionPipeline(kin, PipelineConfig(), FilterConfig())
    env = DualPiperBoxLift(EnvConfig(num_envs=1))
    B = 512
    q = env.q_start.repeat(B, 1)
    gen = torch.Generator().manual_seed(1)
    total_m = 0
    for _ in range(60):
        a = torch.randn(B, 12, generator=gen)
        a[:, 1] -= 1.0
        a[:, 7] += 1.0  # drive the grippers toward each other
        out = pipe(q, a)
        d = pipe.true_dist.dmin(out["q_cmd"])
        assert bool((d >= pipe.fcfg.d_safe - 1e-5).all())
        total_m += int(out["m"].sum())
        q = out["q_cmd"]
    assert total_m > 0  # the filter was exercised


def test_motion_bound():
    kin = DualPiperKinematics()
    pipe = ActionPipeline(kin, PipelineConfig(), FilterConfig(enabled=False))
    env = DualPiperBoxLift(EnvConfig(num_envs=1))
    q = env.q_start.repeat(256, 1)
    f = kin.fk(q)
    out = pipe(q, torch.randn(256, 12) * 3.0)
    disp = pipe._max_point_motion(kin.fk(out["q_cmd"]), f)
    assert bool((disp <= pipe.pcfg.motion_bound * 1.05).all())
    assert pipe.pcfg.motion_bound < FilterConfig().d_safe / 2


def test_correction_displacement_gradient_equals_minus_two_delta():
    """dL/dmu = -2 m (a_feas - a_raw): the mean moves by the correction, independent of the noise."""
    mu = torch.randn(16, 12, requires_grad=True)
    a_raw = mu.detach() + 0.5 * torch.randn(16, 12)
    a_feas = a_raw + 0.1 * torch.randn(16, 12)
    m = (torch.rand(16) > 0.5).float()
    target = (mu + (a_feas - a_raw)).detach()
    loss = (m * ((mu - target) ** 2).sum(-1)).sum()
    loss.backward()
    assert torch.allclose(mu.grad, -2 * m[:, None] * (a_feas - a_raw), atol=1e-6)
    assert math.isclose(loss.item(), (m * ((a_feas - a_raw) ** 2).sum(-1)).sum().item(), rel_tol=1e-5)
