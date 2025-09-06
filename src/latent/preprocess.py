"""
Data preprocessing utilities:

- Resample sequences to target FPS using linear time-warping
- Crop/pad sequences to fixed length T
- Skeleton normalization: unify bone lengths using a provided rest pose
- Convert global positions to (root velocities, local joint rotations 6D)
- Inverse: from (root velocities, local rotations) to global joint positions via FK

Acceptance test for step-1 relies on reconstructing positions with <1e-3 error.
"""
from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn.functional as F

from .skeleton import (
    Skeleton,
    ForwardKinematics,
    estimate_rest_offsets_from_positions,
    rot6d_to_rotmat,
    rotmat_to_rot6d,
    integrate_root_motion,
)


def linear_resample(sequence: torch.Tensor, src_fps: float, tgt_fps: float) -> torch.Tensor:
    """Resample per-frame data from src_fps to tgt_fps with linear interpolation.

    sequence: (T, ...)
    Returns: (T_new, ...)
    """
    if abs(src_fps - tgt_fps) < 1e-6:
        return sequence
    T = sequence.shape[0]
    duration = (T - 1) / src_fps
    T_new = int(round(duration * tgt_fps)) + 1
    t_src = torch.linspace(0.0, 1.0, T, device=sequence.device)
    t_tgt = torch.linspace(0.0, 1.0, T_new, device=sequence.device)
    seq_flat = sequence.reshape(T, -1).T.unsqueeze(0).unsqueeze(0)  # (1,1,C,T)
    t_src_grid = t_src.view(1, 1, 1, T)
    t_tgt_grid = t_tgt.view(1, 1, 1, T_new)
    # Use grid_sample by building a 1D grid along time
    grid = torch.stack([torch.zeros_like(t_tgt_grid), 2 * t_tgt_grid - 1], dim=-1)  # (1,1,1,T_new,2)
    grid = grid.squeeze(0)  # (1,1,T_new,2)
    # Pad to allow border sampling
    seq_img = F.pad(seq_flat, (0, 0, 1, 1), mode='replicate')  # (1,1,C,T+2)
    # Build normalized coordinates for T+2
    # But simpler: use linear interpolation per-channel
    out = torch.interp(t_tgt, t_src, sequence.reshape(T, -1)).reshape(T_new, *sequence.shape[1:])
    return out


def crop_or_pad(sequence: torch.Tensor, T: int) -> torch.Tensor:
    cur_T = sequence.shape[0]
    if cur_T == T:
        return sequence
    if cur_T > T:
        return sequence[:T]
    pad = T - cur_T
    last = sequence[-1:].expand(pad, *sequence.shape[1:])
    return torch.cat([sequence, last], dim=0)


def positions_to_local_rot6d(
    positions: torch.Tensor, skeleton: Skeleton
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert absolute joint positions to (root velocities, local rotations 6D).

    positions: (T, J, 3)
    Returns:
      root_vel: (T, 3) in local frame (Δx, Δz, Δyaw)
      root_trans: (T, 3)
      rot6d: (T, J, 6)
    """
    device = positions.device
    T, J, _ = positions.shape
    parents = skeleton.parents
    rest_offsets = skeleton.rest_offsets

    # Root translation and yaw from trajectory; define yaw from velocity heading atan2(z, x)
    root_pos = positions[:, 0]  # (T,3)
    # Approximate yaw from velocity
    vel = torch.zeros_like(root_pos)
    vel[1:] = root_pos[1:] - root_pos[:-1]
    yaw = torch.atan2(vel[:, 2], vel[:, 0])  # atan2(z, x)
    dyaw = torch.zeros_like(yaw)
    dyaw[1:] = yaw[1:] - yaw[:-1]
    # Local planar velocities
    cos_y = torch.cos(yaw[:-1])
    sin_y = torch.sin(yaw[:-1])
    vx = (vel[1:, 0] * cos_y + vel[1:, 2] * sin_y)
    vz = (-vel[1:, 0] * sin_y + vel[1:, 2] * cos_y)
    root_vel = torch.zeros(T, 3, device=device)
    root_vel[1:, 0] = vx
    root_vel[1:, 1] = vz
    root_vel[1:, 2] = dyaw[1:]

    # Build local rotations consistent with FK: for joint j>0,
    # local_R[j] should map rest_offset[j] (in parent frame) to target vector expressed in parent frame.
    local_R = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(T, J, 1, 1)

    # Precompute parent global rotations including root yaw, with root local_R = I
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    R_yaw = torch.zeros(T, 3, 3, device=device)
    R_yaw[:, 0, 0] = cos_y
    R_yaw[:, 0, 2] = sin_y
    R_yaw[:, 1, 1] = 1.0
    R_yaw[:, 2, 0] = -sin_y
    R_yaw[:, 2, 2] = cos_y

    global_R = torch.zeros(T, J, 3, 3, device=device)
    global_R[:, 0] = R_yaw

    def align_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Efficient rotation from vector a to b (Frisvad 2012-like), batch version.

        a, b: (T,3) -> R: (T,3,3) such that R @ a = b
        """
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        v = torch.cross(a_n, b_n, dim=-1)              # (T,3)
        c = (a_n * b_n).sum(dim=-1, keepdim=True)       # (T,1)
        Tn = a.shape[0]
        I = torch.eye(3, device=a.device).unsqueeze(0).repeat(Tn, 1, 1)
        # If vectors are almost identical
        same = (c.squeeze(-1) > 1.0 - 1e-5)
        R = I.clone()
        if same.any():
            R[same] = I[same]
        # If vectors are opposite
        opp = (c.squeeze(-1) < -1.0 + 1e-5)
        if opp.any():
            a_opp = a_n[opp]
            # Choose an arbitrary orthogonal axis
            helper = torch.tensor([1.0, 0.0, 0.0], device=a.device).expand_as(a_opp)
            use_y = (a_opp[:, :1].abs() > 0.9)
            helper = torch.where(use_y.expand_as(helper), torch.tensor([0.0, 1.0, 0.0], device=a.device).expand_as(helper), helper)
            u = F.normalize(torch.cross(a_opp, helper, dim=-1), dim=-1)  # (N,3)
            # 180-degree rotation: R = I + 2 * [u]_x^2
            K = torch.zeros(u.size(0), 3, 3, device=a.device)
            K[:, 0, 1] = -u[:, 2]
            K[:, 0, 2] = u[:, 1]
            K[:, 1, 0] = u[:, 2]
            K[:, 1, 2] = -u[:, 0]
            K[:, 2, 0] = -u[:, 1]
            K[:, 2, 1] = u[:, 0]
            R_opp = I[opp] + 2 * torch.bmm(K, K)
            R[opp] = R_opp
        # General case
        gen = ~(same | opp)
        if gen.any():
            v_g = v[gen]
            c_g = c[gen].view(-1, 1, 1)
            K = torch.zeros(v_g.size(0), 3, 3, device=a.device)
            K[:, 0, 1] = -v_g[:, 2]
            K[:, 0, 2] = v_g[:, 1]
            K[:, 1, 0] = v_g[:, 2]
            K[:, 1, 2] = -v_g[:, 0]
            K[:, 2, 0] = -v_g[:, 1]
            K[:, 2, 1] = v_g[:, 0]
            h = (1.0 / (1.0 + c_g)).clamp(max=1e8)
            R_g = I[gen] + K + torch.bmm(K, K) * h
            R[gen] = R_g
        return R

    for j in range(1, J):
        p = int(parents[j].item())
        # target vector in world
        tgt_world = positions[:, j] - positions[:, p]  # (T,3)
        # express target in parent frame
        tgt_parent = (global_R[:, p].transpose(1, 2) @ tgt_world.unsqueeze(-1)).squeeze(-1)
        # source vector is rest offset in parent frame
        src = rest_offsets[j].view(1, 3).expand(T, 3)
        R_local = align_vectors(src, tgt_parent)
        # If target equals source (within eps), force identity to avoid numerical issues
        close = (tgt_parent - src).norm(dim=-1) < 1e-9  # (T,)
        if close.any():
            I = torch.eye(3, device=device).unsqueeze(0).expand(T, 3, 3)
            R_local[close] = I[close]
        local_R[:, j] = R_local
        global_R[:, j] = global_R[:, p] @ R_local

    rot6d = rotmat_to_rot6d(local_R)  # (T,J,6)

    return root_vel, root_pos, rot6d


def reconstruct_positions_from_local(
    root_vel: torch.Tensor, rot6d: torch.Tensor, skeleton: Skeleton, initial_root: torch.Tensor | None = None
) -> torch.Tensor:
    """Reconstruct absolute positions via integrating root motion and FK.

    root_vel: (T,3)
    rot6d: (T,J,6)
    Returns positions: (T,J,3)
    """
    T = rot6d.shape[0]
    if initial_root is None:
        trans, yaw = integrate_root_motion(root_vel.unsqueeze(0))  # (1,T,3), (1,T)
    else:
        trans, yaw = integrate_root_motion(root_vel.unsqueeze(0), initial_root.unsqueeze(0))
    fk = ForwardKinematics(skeleton.parents, skeleton.rest_offsets).to(rot6d.device)
    pos = fk(
        rot6d=rot6d.unsqueeze(0),
        root_translation=trans,
        root_yaw=yaw,
    )
    return pos.squeeze(0)


def acceptance_check_positions(
    positions_gt: torch.Tensor, positions_rec: torch.Tensor, atol: float = 1e-3
) -> Dict[str, float]:
    """Compute L2 error between reconstructed and ground-truth positions."""
    diff = positions_gt - positions_rec
    mse = (diff.pow(2).sum(dim=-1).mean()).item()
    max_err = diff.norm(dim=-1).max().item()
    ok = float(max_err <= atol)
    return {"mse": mse, "max_err": max_err, "passed": ok}


