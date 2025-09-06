from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_body_connections() -> List[Tuple[int, int]]:
    # Same as ASLVisualizer.body_connections
    return [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7)]


def get_hand_connections() -> List[Tuple[int, int]]:
    # Same as ASLVisualizer.hand_connections (21 joints per hand)
    return [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]


@dataclass
class Skeleton2D:
    # Global indexing for 50 joints: 0..7 body, 8..28 left hand, 29..49 right hand
    # We allow multiple roots: body_root=1 (neck-ish), left_wrist=8, right_wrist=29
    parents: torch.Tensor  # (50,) int64, -1 for roots
    rest_offsets: torch.Tensor  # (50,2), parent->child bone vector at rest frame
    roots: List[int]

    def num_joints(self) -> int:
        return int(self.parents.numel())


def build_parents_for_50() -> Tuple[torch.Tensor, List[int]]:
    # Build a tree structure consistent with ASLVisualizer connections and global indexing
    # Global mapping:
    # body: 0..7, left-hand: 8..28 (local 0..20), right-hand: 29..49 (local 0..20)
    body_edges = get_body_connections()
    hand_edges = get_hand_connections()

    parents = torch.full((50,), -1, dtype=torch.long)
    # Body: choose body root as 1; orient edges away from root
    body_root = 1
    # Make adjacency
    adj: Dict[int, List[int]] = {i: [] for i in range(8)}
    for a, b in body_edges:
        adj[a].append(b)
        adj[b].append(a)
    # BFS from root to set parents
    from collections import deque
    q = deque([body_root])
    visited = {body_root}
    parents[body_root] = -1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in visited:
                continue
            parents[v] = u
            visited.add(v)
            q.append(v)

    # Left hand: attach wrist (8) to body joint 4 (left wrist) for exact alignment
    left_root = 8
    parents[left_root] = 4
    for a, b in hand_edges:
        ga = left_root + a
        gb = left_root + b
        # orient from wrist outward using BFS: since list is already from proximal to distal
        parents[gb] = ga

    # Right hand: attach wrist (29) to body joint 7 (right wrist)
    right_root = 29
    parents[right_root] = 7
    for a, b in hand_edges:
        ga = right_root + a
        gb = right_root + b
        parents[gb] = ga

    roots = [body_root]
    return parents, roots


def split_pose150(pose150: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pose150: (..., 150) with [body(24), right(63), left(63)] each triple x,y,conf
    assert pose150.size(-1) == 150
    body = pose150[..., :24].reshape(*pose150.shape[:-1], 8, 3)
    right = pose150[..., 24:87].reshape(*pose150.shape[:-1], 21, 3)
    left = pose150[..., 87:150].reshape(*pose150.shape[:-1], 21, 3)
    # Replace zeros/invalid with nearest valid (simple forward-fill on last dim)
    # Here only drop positions with extremely small magnitude
    def clean(arr):
        # arr: (T, K, 3)
        xy = arr[..., :2]
        conf = arr[..., 2:3]
        mask = ((conf > 0.05) & (xy.abs().sum(dim=-1, keepdim=True) > 1e-6)).squeeze(-1)  # (T,K)
        Tdim = xy.size(-3)
        K = xy.size(-2)
        xy_ff = xy.clone()
        # Initialize 'last' as the first valid observation per joint if exists
        last = xy_ff[0].clone()  # (K,2)
        for k in range(K):
            first_valid = None
            for t0 in range(Tdim):
                if mask[t0, k]:
                    first_valid = xy_ff[t0, k]
                    break
            if first_valid is not None:
                last[k] = first_valid
        for t in range(Tdim):
            cur = xy_ff[t]
            cur_mask = mask[t].unsqueeze(-1).expand(K, 2)  # (K,2)
            xy_ff[t] = torch.where(cur_mask, cur, last)
            last = xy_ff[t]
        return torch.cat([xy_ff, conf], dim=-1)
    body = clean(body)
    right = clean(right)
    left = clean(left)
    return body, right, left


def adjust_shoulder_mid_root(xy: torch.Tensor, root_idx: int = 1, left_shoulder_idx: int = 2, right_shoulder_idx: int = 5) -> torch.Tensor:
    """Set root joint (index 1) to the midpoint of shoulders (2 and 5)."""
    xy_adj = xy.clone()
    mid = 0.5 * (xy_adj[:, left_shoulder_idx] + xy_adj[:, right_shoulder_idx])
    xy_adj[:, root_idx] = mid
    return xy_adj


def pose150_to_global50_xy(pose150: torch.Tensor) -> torch.Tensor:
    # Returns (..., 50, 2); ignore conf
    body, right, left = split_pose150(pose150)
    xy = torch.zeros(*pose150.shape[:-1], 50, 2, dtype=pose150.dtype, device=pose150.device)
    xy[..., 0:8, :] = body[..., :, :2]
    xy[..., 8:29, :] = left[..., :, :2]
    xy[..., 29:50, :] = right[..., :, :2]
    return xy


def global50_xy_to_pose150(xy: torch.Tensor, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
    # xy: (..., 50, 2)
    # conf: (..., 50) or None -> defaults to ones
    if conf is None:
        # conf shape should match leading dims (..., J)
        conf = torch.ones(*xy.shape[:-2], xy.size(-2), device=xy.device, dtype=xy.dtype)
    body_xy = xy[..., 0:8, :]
    left_xy = xy[..., 8:29, :]
    right_xy = xy[..., 29:50, :]
    body_conf = conf[..., 0:8]
    right_conf = conf[..., 29:50]
    left_conf = conf[..., 8:29]
    body = torch.cat([body_xy, body_conf.unsqueeze(-1)], dim=-1).reshape(*xy.shape[:-2], 24)
    right = torch.cat([right_xy, right_conf.unsqueeze(-1)], dim=-1).reshape(*xy.shape[:-2], 63)
    left = torch.cat([left_xy, left_conf.unsqueeze(-1)], dim=-1).reshape(*xy.shape[:-2], 63)
    return torch.cat([body, right, left], dim=-1)


def compute_rest_offsets_xy(positions_xy: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    # positions_xy: (J,2) at rest frame (e.g., first frame)
    J = positions_xy.size(-2)
    rest = torch.zeros(J, 2, device=positions_xy.device, dtype=positions_xy.dtype)
    for j in range(J):
        p = int(parents[j].item())
        if p >= 0:
            rest[j] = positions_xy[j] - positions_xy[p]
    return rest


def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: (...,2)
    # returns angle theta where R(theta) * a = b in least squares
    # theta = atan2(cross, dot)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = (a * b).sum(dim=-1)
    return torch.atan2(cross, dot)


def rot2d(theta: torch.Tensor) -> torch.Tensor:
    # theta: (...,)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    R = torch.zeros(*theta.shape, 2, 2, device=theta.device, dtype=theta.dtype)
    R[..., 0, 0] = cos
    R[..., 0, 1] = -sin
    R[..., 1, 0] = sin
    R[..., 1, 1] = cos
    return R


class FK2D(nn.Module):
    def __init__(self, parents: torch.Tensor, roots: List[int], rest_offsets: torch.Tensor):
        super().__init__()
        self.register_buffer('parents', parents.clone().long())
        self.roots = roots
        self.register_buffer('rest_offsets', rest_offsets.clone())

    def forward(self, local_thetas: torch.Tensor, root_positions: torch.Tensor, bone_scales: Optional[torch.Tensor] = None) -> torch.Tensor:
        # local_thetas: (T,J) radians, 0 for roots; root_positions: (T,J,2) only used for roots
        # bone_scales: optional (T,J), scales along each rest offset
        T, J = local_thetas.shape
        device = local_thetas.device
        positions = torch.zeros(T, J, 2, device=device)
        global_R = torch.eye(2, device=device).view(1, 1, 2, 2).expand(T, J, 2, 2).clone()

        # Initialize roots
        for r in self.roots:
            positions[:, r] = root_positions[:, r]
        # Smooth root trajectory aggressively (Savitzky-Golay like 5-point)
        # Revert to mild smoothing to avoid bias
        if T > 2:
            for r in self.roots:
                pr = positions[:, r]
                pr[1:-1] = (pr[0:-2] + 4*pr[1:-1] + pr[2:]) / 6.0
                positions[:, r] = pr
            global_R[:, r] = rot2d(torch.zeros(T, device=device))  # identity

        # Topological order from roots
        children = {i: [] for i in range(J)}
        for j in range(J):
            p = int(self.parents[j].item())
            if p >= 0:
                children[p].append(j)
        from collections import deque
        order = []
        dq = deque(self.roots)
        visited = set(self.roots)
        while dq:
            u = dq.popleft()
            for v in children[u]:
                if v in visited:
                    continue
                order.append(v)
                visited.add(v)
                dq.append(v)
        for j in order:
            p = int(self.parents[j].item())
            if p < 0:
                continue
            R_local = rot2d(local_thetas[:, j])
            global_R[:, j] = torch.matmul(global_R[:, p], R_local)
            off = self.rest_offsets[j].view(1, 2, 1).expand(T, 2, 1)
            if bone_scales is not None:
                scale = bone_scales[:, j].view(T, 1, 1)
                off = off * scale
            # Use child's global rotation to rotate the rest offset
            step = torch.matmul(global_R[:, j], off).squeeze(-1)
            positions[:, j] = positions[:, p] + step
        return positions


def positions_to_local2d(positions_xy: torch.Tensor, parents: torch.Tensor, roots: List[int], rest_offsets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # positions_xy: (T,J,2)
    T, J, _ = positions_xy.shape
    # Root positions
    root_positions = torch.zeros(T, J, 2, device=positions_xy.device, dtype=positions_xy.dtype)
    for r in roots:
        root_positions[:, r] = positions_xy[:, r]
    # Local angles per joint
    local_theta = torch.zeros(T, J, device=positions_xy.device, dtype=positions_xy.dtype)
    # Per-frame bone scale relative to rest length
    bone_scales = torch.ones(T, J, device=positions_xy.device, dtype=positions_xy.dtype)
    # Global rotations for parents (start as identity); used progressively
    global_theta = torch.zeros(T, J, device=positions_xy.device, dtype=positions_xy.dtype)
    for r in roots:
        global_theta[:, r] = 0.0
    # Build topological order from roots
    children = {i: [] for i in range(J)}
    for j in range(J):
        p = int(parents[j].item())
        if p >= 0:
            children[p].append(j)
    from collections import deque
    order = []
    dq = deque(roots)
    visited = set(roots)
    while dq:
        u = dq.popleft()
        for v in children[u]:
            if v in visited:
                continue
            order.append(v)
            visited.add(v)
            dq.append(v)

    for j in order:
        p = int(parents[j].item())
        if p < 0:
            continue
        # target vector in world
        tgt = positions_xy[:, j] - positions_xy[:, p]
        # if target is near zero (missing), fall back to parent's forward with rest length
        miss = (tgt.norm(dim=-1) < 1e-5)
        if miss.any():
            gp = global_theta[:, p]
            fwd = torch.stack([torch.cos(gp), torch.sin(gp)], dim=-1)
            tgt[miss] = fwd[miss] * rest_offsets[j].norm().clamp_min(1e-6)
        # express target in parent frame by rotating by -global_theta[parent]
        gp = global_theta[:, p]
        cos = torch.cos(-gp)
        sin = torch.sin(-gp)
        Rinv = torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)  # (T,2,2)
        tgt_parent = torch.matmul(Rinv, tgt.unsqueeze(-1)).squeeze(-1)
        src = rest_offsets[j].view(1, 2).expand(T, 2)
        src_len = src.norm(dim=-1).clamp_min(1e-6)
        tgt_len = tgt_parent.norm(dim=-1).clamp_min(1e-6)
        # freeze scales for hand finger chains to reduce jitter amplification
        is_hand = (j >= 8 and j <= 28) or (j >= 29 and j <= 49)
        if is_hand:
            bone_scales[:, j] = 1.0
        else:
            bone_scales[:, j] = tgt_len / src_len
        # Normalize to unit for angle computation
        src_n = src / src_len.unsqueeze(-1)
        tgt_n = tgt_parent / tgt_len.unsqueeze(-1)
        theta = angle_between(src_n, tgt_n)
        local_theta[:, j] = theta
        global_theta[:, j] = global_theta[:, p] + theta
    return root_positions, local_theta, bone_scales


def normalize_bone_lengths(xy: torch.Tensor, parents: torch.Tensor, rest_offsets: torch.Tensor) -> torch.Tensor:
    """Normalize each bone length to match rest_offsets length, preserving per-frame directions.

    xy: (T,J,2)
    parents: (J,)
    rest_offsets: (J,2)
    """
    T, J, _ = xy.shape
    out = xy.clone()
    rest_len = rest_offsets.norm(dim=-1)  # (J,)
    order = list(range(J))
    for t in range(T):
        for j in order:
            p = int(parents[j].item())
            if p < 0:
                continue
            vec = out[t, j] - out[t, p]
            cur_len = vec.norm().item()
            target_len = rest_len[j].item()
            if cur_len > 1e-8 and target_len > 0:
                out[t, j] = out[t, p] + vec * (target_len / cur_len)
            else:
                # Robust fallback: project along a plausible direction
                # 1) try parent's forward direction
                dir_vec = None
                pp = int(parents[p].item()) if p >= 0 else -1
                if pp >= 0:
                    pv = out[t, p] - out[t, pp]
                    if pv.norm().item() > 1e-8:
                        dir_vec = pv / pv.norm().clamp_min(1e-8)
                # 2) fallback to rest offset direction
                if dir_vec is None:
                    ro = rest_offsets[j]
                    if ro.norm().item() > 1e-8:
                        dir_vec = ro / ro.norm().clamp_min(1e-8)
                    else:
                        dir_vec = torch.tensor([1.0, 0.0], device=xy.device, dtype=xy.dtype)
                step = dir_vec * max(target_len, 0.0)
                out[t, j] = out[t, p] + step
    return out


