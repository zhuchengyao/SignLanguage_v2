"""
Skeleton definitions and forward kinematics utilities for 3D human pose.

This module provides:
- Joint hierarchy (parents array)
- Rest offsets (bone vectors in the rest pose)
- Conversions between 6D rotation representation and rotation matrices
- Differentiable Forward Kinematics (FK)

Notes
-----
- We keep the skeleton generic: provide a default 22-joint hierarchy similar to HumanML3D.
- For real datasets, you can estimate rest offsets from the first frame via
  `estimate_rest_offsets_from_positions`. Using those offsets consistently in
  both encoding and decoding ensures reconstruction error is near-zero.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Skeleton:
    parents: torch.Tensor  # shape: (J,), dtype: long, parent index for each joint; root has parent -1
    rest_offsets: torch.Tensor  # shape: (J, 3), rest bone vectors from parent -> joint in rest pose

    def num_joints(self) -> int:
        return int(self.parents.numel())


def create_default_skeleton(device: Optional[torch.device] = None) -> Skeleton:
    """Create a compact 22-joint skeleton similar to HumanML3D.

    The exact offsets are placeholders consistent with a T-pose-like layout and
    intended for synthetic tests. For real datasets, use
    `estimate_rest_offsets_from_positions` from a rest or first frame.
    """
    # Parent indices for 22 joints. -1 denotes root (pelvis)
    parents = torch.tensor([
        -1,  # 0: pelvis (root)
         0,  # 1: spine
         1,  # 2: spine1
         2,  # 3: neck
         3,  # 4: head
         2,  # 5: left_shoulder
         5,  # 6: left_elbow
         6,  # 7: left_wrist
         2,  # 8: right_shoulder
         8,  # 9: right_elbow
         9,  # 10: right_wrist
         0,  # 11: left_hip
        11,  # 12: left_knee
        12,  # 13: left_ankle
         0,  # 14: right_hip
        14,  # 15: right_knee
        15,  # 16: right_ankle
         4,  # 17: head_top (aux)
         7,  # 18: left_hand (aux)
        10,  # 19: right_hand (aux)
        13,  # 20: left_toe (aux)
        16,  # 21: right_toe (aux)
    ], dtype=torch.long, device=device)

    # Rest offsets (meters) in a rough T-pose-like shape
    rest_offsets = torch.tensor([
        [0.0, 0.0, 0.0],     # pelvis (root)
        [0.0, 0.10, 0.0],    # spine
        [0.0, 0.10, 0.0],    # spine1
        [0.0, 0.10, 0.0],    # neck
        [0.0, 0.12, 0.0],    # head
        [0.08, 0.02, 0.0],   # left_shoulder
        [0.14, 0.00, 0.0],   # left_elbow
        [0.14, 0.00, 0.0],   # left_wrist
        [-0.08, 0.02, 0.0],  # right_shoulder
        [-0.14, 0.00, 0.0],  # right_elbow
        [-0.14, 0.00, 0.0],  # right_wrist
        [0.08, -0.10, 0.0],  # left_hip
        [0.00, -0.20, 0.0],  # left_knee
        [0.00, -0.20, 0.0],  # left_ankle
        [-0.08, -0.10, 0.0], # right_hip
        [0.00, -0.20, 0.0],  # right_knee
        [0.00, -0.20, 0.0],  # right_ankle
        [0.00, 0.08, 0.0],   # head_top
        [0.04, 0.00, 0.0],   # left_hand aux
        [-0.04, 0.00, 0.0],  # right_hand aux
        [0.00, 0.00, 0.08],  # left_toe
        [0.00, 0.00, 0.08],  # right_toe
    ], dtype=torch.float32, device=device)

    return Skeleton(parents=parents, rest_offsets=rest_offsets)


def estimate_rest_offsets_from_positions(
    positions_rest_frame: torch.Tensor, parents: torch.Tensor
) -> torch.Tensor:
    """Estimate rest offsets (bone vectors) from a reference pose.

    Parameters
    ----------
    positions_rest_frame : (J, 3) torch.Tensor
        Joint positions of a rest-like pose. The pelvis/root should be at index 0.
    parents : (J,) torch.LongTensor
        Parent indices; root has -1.

    Returns
    -------
    (J, 3) torch.Tensor with bone vectors parent->child.
    """
    J = positions_rest_frame.size(0)
    rest_offsets = torch.zeros_like(positions_rest_frame)
    for j in range(J):
        p = int(parents[j].item())
        if p < 0:
            rest_offsets[j] = torch.zeros(3, device=positions_rest_frame.device)
        else:
            rest_offsets[j] = positions_rest_frame[j] - positions_rest_frame[p]
    return rest_offsets


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Implementation per "On the Continuity of Rotation Representations in Neural
    Networks" (Zhou et al., CVPR 2019).

    Parameters
    ----------
    x : (B, J, 6) or (N, 6)

    Returns
    -------
    rot_mats : (..., 3, 3)
    """
    orig_shape = x.shape
    x = x.view(-1, 6)
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    rot = torch.stack([b1, b2, b3], dim=-1)  # (N, 3, 3)
    return rot.view(*orig_shape[:-1], 3, 3)


def rotmat_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation by taking the first two columns.

    Parameters
    ----------
    R : (..., 3, 3)
    """
    first_two = R[..., :, :2]  # (..., 3, 2)
    return first_two.reshape(*R.shape[:-2], 6)


class ForwardKinematics(nn.Module):
    """Differentiable forward kinematics.

    Given local joint rotations and rest offsets, compute global joint positions.

    Inputs
    ------
    - rot6d: (B, T, J, 6)
    - root_translation: (B, T, 3) world-space root translations
    - root_yaw: (B, T) world-space root yaw angle in radians
    - parents: (J,) long
    - rest_offsets: (J,3)

    Outputs
    -------
    - positions: (B, T, J, 3)
    """

    def __init__(self, parents: torch.Tensor, rest_offsets: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("parents", parents.clone().long())
        self.register_buffer("rest_offsets", rest_offsets.clone().float())

    def forward(
        self,
        rot6d: torch.Tensor,
        root_translation: torch.Tensor,
        root_yaw: torch.Tensor,
    ) -> torch.Tensor:
        B, T, J, _ = rot6d.shape
        device = rot6d.device

        # Convert 6D local rotations to matrices
        local_R = rot6d_to_rotmat(rot6d)  # (B, T, J, 3, 3)

        # Precompute cumulative yaw rotation around Y-axis using previous yaw (heading)
        # Here root_yaw[t] is absolute yaw; the forward direction in world is [cos(yaw), 0, sin(yaw)]
        cy = torch.cos(root_yaw)
        sy = torch.sin(root_yaw)
        R_yaw = torch.zeros(B, T, 3, 3, device=device)
        R_yaw[..., 0, 0] = cy
        R_yaw[..., 0, 2] = sy
        R_yaw[..., 1, 1] = 1.0
        R_yaw[..., 2, 0] = -sy
        R_yaw[..., 2, 2] = cy

        # Buffers for outputs
        positions = torch.zeros(B, T, J, 3, device=device)
        global_R = torch.zeros(B, T, J, 3, 3, device=device)

        # Root joint (assumed index 0)
        root_idx = 0
        global_R[:, :, root_idx] = R_yaw @ local_R[:, :, root_idx]
        positions[:, :, root_idx] = root_translation

        # Other joints in hierarchical order
        # A simple topological order from parents array
        joint_indices = list(range(J))
        joint_indices.remove(root_idx)
        for j in joint_indices:
            p = int(self.parents[j].item())
            assert p >= 0, "Non-root joints must have a valid parent"
            # Global rotation is parent's global rotation times local rotation
            global_R[:, :, j] = global_R[:, :, p] @ local_R[:, :, j]
            # Position: parent position + parent_global_R @ (local_R[j] @ rest_offset[j])
        offset = self.rest_offsets[j].view(3, 1)                   # (3,1)
        local_rotated = (local_R[:, :, j] @ offset).squeeze(-1)    # (B,T,3)
        rotated = (global_R[:, :, p] @ local_rotated.unsqueeze(-1)).squeeze(-1)  # (B,T,3)
        positions[:, :, j] = positions[:, :, p] + rotated

        return positions


def integrate_root_motion(
    root_velocities: torch.Tensor, initial_root: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate root velocities (Δx, Δz, Δyaw) to absolute root translation and yaw.

    Parameters
    ----------
    root_velocities : (B, T, 3) torch.Tensor
        The per-frame velocity in the local root frame: Δx (forward), Δz (side), Δyaw (heading).
        Units are meters per frame and radians per frame.
    initial_root : (B, 3) or None
        Initial [x, y, z] world-space translation. If None, zeros.

    Returns
    -------
    root_translation : (B, T, 3)
    root_yaw : (B, T)
    """
    B, T, _ = root_velocities.shape
    device = root_velocities.device
    if initial_root is None:
        initial_root = torch.zeros(B, 3, device=device)

    # Unpack
    dx_local = root_velocities[..., 0]
    dz_local = root_velocities[..., 1]
    dyaw = root_velocities[..., 2]

    # Integrate yaw first
    yaw = torch.zeros(B, T, device=device)
    yaw[:, 0] = dyaw[:, 0]
    for t in range(1, T):
        yaw[:, t] = yaw[:, t - 1] + dyaw[:, t]

    # Rotate local planar velocities into world frame and integrate translation
    trans = torch.zeros(B, T, 3, device=device)
    trans[:, 0] = initial_root
    for t in range(1, T):
        # Current yaw determines heading
        cy = torch.cos(yaw[:, t - 1])
        sy = torch.sin(yaw[:, t - 1])
        # Local (x,z) to world (x,z)
        vx = dx_local[:, t] * cy - dz_local[:, t] * sy
        vz = dx_local[:, t] * sy + dz_local[:, t] * cy
        step = torch.stack([vx, torch.zeros_like(vx), vz], dim=-1)
        trans[:, t] = trans[:, t - 1] + step

    return trans, yaw


