import numpy as np
from typing import Tuple


# Indices for body keypoints within the 50-joint layout used by this project
# Assumption based on asl_visualizer body connections:
# 0: pelvis/root, 1: chest/neck, 2: right_shoulder, 5: left_shoulder
PELVIS = 0
CHEST = 1
RIGHT_SHOULDER = 2
LEFT_SHOULDER = 5


def _safe_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = (b - a).astype(np.float32)
    if not np.isfinite(v).all():
        v = np.zeros_like(v)
    return v


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def normalize_frame_xy(points_50x2: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, dict]:
    """
    Normalize a single frame (50,2) by:
      1) translate to pelvis at origin
      2) rotate to align shoulder vector with +x axis
      3) scale by shoulder width (fallback to torso length)
    Returns normalized points and info dict with applied transform.
    """
    assert points_50x2.shape == (50, 2)
    p = points_50x2.astype(np.float32)

    pelvis = p[PELVIS]
    chest = p[CHEST]
    r_sh = p[RIGHT_SHOULDER]
    l_sh = p[LEFT_SHOULDER]

    # 1) translate
    p_t = p - pelvis

    # 2) rotate: align shoulder vector with x-axis
    sh_vec = _safe_vec(r_sh - pelvis, l_sh - pelvis)  # from right to left
    theta = float(np.arctan2(sh_vec[1], sh_vec[0]))
    R = _rotation_matrix(-theta)
    p_tr = (p_t @ R)

    # Ensure left shoulder is on +x relative to right after rotation; if not, mirror x
    r_sh_rot = (r_sh - pelvis) @ R
    l_sh_rot = (l_sh - pelvis) @ R
    if l_sh_rot[0] < r_sh_rot[0]:
        p_tr[:, 0] *= -1.0
        R = np.diag([-1.0, 1.0]).astype(np.float32) @ R

    # 3) scale
    sh_width = np.linalg.norm(l_sh_rot - r_sh_rot)
    torso_len = np.linalg.norm(((chest - pelvis) @ R))
    denom = float(max(sh_width, torso_len, eps))
    p_norm = p_tr / denom

    info = {
        'translation': pelvis.tolist(),
        'theta': theta,
        'scale_denom': denom,
    }
    return p_norm.astype(np.float32), info


def normalize_sequence_xy(seq_xy: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Apply per-frame normalization to a sequence of shape (T,50,2).
    Returns normalized sequence and summary info.
    """
    assert seq_xy.ndim == 3 and seq_xy.shape[1:] == (50, 2)
    T = seq_xy.shape[0]
    out = np.zeros_like(seq_xy, dtype=np.float32)
    infos = []
    for t in range(T):
        out[t], info = normalize_frame_xy(seq_xy[t])
        infos.append(info)
    return out, {'frames': infos}


