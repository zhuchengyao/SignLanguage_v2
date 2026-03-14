"""KALS (Kinematic-Aware Loss Set) — physics-based constraints for generated poses.

Includes:
  1. Bone length constancy loss
  2. Joint angle legality loss
  3. Bilateral hand coordination loss
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Skeleton topology (same indexing as skeleton2d.py)
# body: 0..7, left hand: 8..28, right hand: 29..49
# ---------------------------------------------------------------------------

BODY_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
]

HAND_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Physiological angle limits (degrees) per joint type — approximate ranges.
# index → (min_deg, max_deg) for the interior angle at that joint.
ELBOW_RANGE = (0.0, 170.0)
FINGER_RANGE = (0.0, 90.0)
WRIST_RANGE = (-80.0, 80.0)

# Map body joint indices to angle limits (angle measured at the middle joint of a triplet)
BODY_ANGLE_TRIPLETS: List[Tuple[int, int, int, float, float]] = [
    # (parent, joint, child, min_deg, max_deg)
    (2, 3, 4, *ELBOW_RANGE),   # left elbow
    (5, 6, 7, *ELBOW_RANGE),   # right elbow
]

# Finger PIP / DIP joints in local hand space (0-indexed within 21-joint hand)
FINGER_ANGLE_TRIPLETS_LOCAL: List[Tuple[int, int, int]] = [
    (1, 2, 3), (2, 3, 4),       # thumb
    (5, 6, 7), (6, 7, 8),       # index
    (9, 10, 11), (10, 11, 12),  # middle
    (13, 14, 15), (14, 15, 16), # ring
    (17, 18, 19), (18, 19, 20), # pinky
]


# ---------------------------------------------------------------------------
# 1. Bone length constancy
# ---------------------------------------------------------------------------

def _bone_lengths(xy: torch.Tensor, bones: List[Tuple[int, int]]) -> torch.Tensor:
    """xy: (B, T, J, 2). Returns (B, T, num_bones)."""
    parts = []
    for a, b in bones:
        bl = (xy[:, :, a] - xy[:, :, b]).norm(dim=-1, keepdim=True)
        parts.append(bl)
    return torch.cat(parts, dim=-1)


def bone_length_constancy_loss(
    pred_xy: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalise temporal variation in bone lengths.

    pred_xy: (B, T, 50, 2) predicted joint positions.
    mask: (B, T) bool.
    Returns scalar loss.
    """
    body_xy = pred_xy[:, :, :8]
    left_xy = pred_xy[:, :, 8:29]
    right_xy = pred_xy[:, :, 29:50]

    body_bl = _bone_lengths(body_xy, BODY_BONES)
    left_bl = _bone_lengths(left_xy, HAND_BONES)
    right_bl = _bone_lengths(right_xy, HAND_BONES)
    all_bl = torch.cat([body_bl, left_bl, right_bl], dim=-1)  # (B, T, num)

    delta = (all_bl[:, 1:] - all_bl[:, :-1]).abs()  # (B, T-1, num)

    if mask is not None:
        pair_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).float()
        loss = (delta * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)
    else:
        loss = delta.mean()
    return loss


# ---------------------------------------------------------------------------
# 2. Joint angle legality
# ---------------------------------------------------------------------------

def _joint_angle_deg(
    p_a: torch.Tensor, p_b: torch.Tensor, p_c: torch.Tensor,
) -> torch.Tensor:
    """Compute interior angle at p_b formed by p_a-p_b-p_c in degrees."""
    v1 = p_a - p_b
    v2 = p_c - p_b
    cos_theta = F.cosine_similarity(v1, v2, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_theta) * (180.0 / 3.14159265)


def joint_angle_legality_loss(
    pred_xy: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalise joint angles outside physiological ranges.

    pred_xy: (B, T, 50, 2).
    """
    losses = []

    # Body angles
    for parent, joint, child, lo, hi in BODY_ANGLE_TRIPLETS:
        angle = _joint_angle_deg(pred_xy[:, :, parent], pred_xy[:, :, joint], pred_xy[:, :, child])
        violation = F.relu(angle - hi) + F.relu(lo - angle)
        losses.append(violation)

    # Finger angles (left hand: global 8..28, right hand: 29..49)
    for offset in [8, 29]:
        for a, b, c in FINGER_ANGLE_TRIPLETS_LOCAL:
            angle = _joint_angle_deg(
                pred_xy[:, :, offset + a],
                pred_xy[:, :, offset + b],
                pred_xy[:, :, offset + c],
            )
            lo, hi = FINGER_RANGE
            violation = F.relu(angle - hi) + F.relu(lo - angle)
            losses.append(violation)

    all_violations = torch.stack(losses, dim=-1)  # (B, T, num_triplets)

    if mask is not None:
        all_violations = all_violations * mask.unsqueeze(-1).float()
        return all_violations.sum() / mask.sum().clamp(min=1.0)
    return all_violations.mean()


# ---------------------------------------------------------------------------
# 3. Bilateral hand coordination
# ---------------------------------------------------------------------------

def bilateral_hand_coordination_loss(
    pred_xy: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft penalty on extreme velocity asymmetry between left and right hands.

    Encourages rhythmic coordination (not mirror symmetry).
    pred_xy: (B, T, 50, 2).
    """
    left_xy = pred_xy[:, :, 8:29]   # (B, T, 21, 2)
    right_xy = pred_xy[:, :, 29:50]

    left_vel = (left_xy[:, 1:] - left_xy[:, :-1]).norm(dim=-1).mean(dim=-1)   # (B, T-1)
    right_vel = (right_xy[:, 1:] - right_xy[:, :-1]).norm(dim=-1).mean(dim=-1)

    # Normalise each to [0, 1] range per sequence, then compute difference
    lv_max = left_vel.amax(dim=-1, keepdim=True).clamp(min=1e-6)
    rv_max = right_vel.amax(dim=-1, keepdim=True).clamp(min=1e-6)
    left_norm = left_vel / lv_max
    right_norm = right_vel / rv_max

    diff = (left_norm - right_norm).abs()  # (B, T-1)

    if mask is not None:
        pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
        return (diff * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)
    return diff.mean()


# ---------------------------------------------------------------------------
# Combined KALS loss
# ---------------------------------------------------------------------------

def kals_loss(
    pred_xy: torch.Tensor,
    mask: torch.Tensor | None = None,
    w_bone: float = 0.10,
    w_angle: float = 0.05,
    w_sym: float = 0.03,
) -> Tuple[torch.Tensor, dict]:
    """Compute all KALS components and return weighted sum + breakdown."""
    l_bone = bone_length_constancy_loss(pred_xy, mask)
    l_angle = joint_angle_legality_loss(pred_xy, mask)
    l_sym = bilateral_hand_coordination_loss(pred_xy, mask)

    total = w_bone * l_bone + w_angle * l_angle + w_sym * l_sym
    return total, {
        "bone_loss": l_bone.detach(),
        "angle_loss": l_angle.detach(),
        "symmetry_loss": l_sym.detach(),
    }
