import os
import json
import argparse
from typing import Optional, List

import numpy as np

from src.asl_visualizer import ASLVisualizer

# ---- Helpers for canonical initial pose from template meta ----

def _pca_align_unit(p: np.ndarray) -> np.ndarray:
    """
    p: (50,2) -> centered, PCA-rotated to align first PC to x-axis, unit scale (RMS radius=1).
    """
    if p.ndim != 2 or p.shape != (50, 2):
        return p
    # Remove invalid rows (near zeros) softly by replacing with centroid
    mask = np.isfinite(p).all(axis=1)
    q = p.copy()
    if not mask.all():
        cen = np.nanmean(q[mask], axis=0) if mask.any() else np.zeros(2, dtype=np.float32)
        q[~mask] = cen
    # Center
    cen = q.mean(axis=0, keepdims=True)
    q = q - cen
    # PCA via SVD
    U, S, Vt = np.linalg.svd(q, full_matrices=False)
    R = Vt.T  # 2x2
    q = q @ R
    # Unit scale by RMS radius
    rms = np.sqrt((q ** 2).mean())
    if rms > 1e-8:
        q = q / rms
    return q.astype(np.float32)


def load_canonical_initial_pose_from_template(positions_npy_path: str) -> Optional[np.ndarray]:
    """
    Given a template positions npy path (e.g., *_positions.npy), try to read the sidecar
    template json (same basename .json), look up 'sources' (list of disp_*.npy paths),
    and build a canonical initial pose by averaging the first frame poses of those sources
    after normalization (_pca_align_unit).

    Returns (50,2) or None.
    """
    base = os.path.splitext(positions_npy_path)[0]
    json_path = base.replace("_positions", "") + ".json"
    if not os.path.exists(json_path):
        return None
    try:
        meta = json.load(open(json_path, 'r', encoding='utf-8'))
        sources = meta.get('sources', [])
        if not sources:
            return None
        poses = []
        for disp_npy in sources:
            js_sidecar = os.path.splitext(disp_npy)[0] + ".json"
            if not os.path.exists(js_sidecar):
                continue
            try:
                s_meta = json.load(open(js_sidecar, 'r', encoding='utf-8'))
                sample_dir = s_meta.get('sample_dir', '')
                pose_f = os.path.join(sample_dir, 'pose.json')
                if not os.path.exists(pose_f):
                    continue
                js = json.load(open(pose_f, 'r', encoding='utf-8'))
                frames = js.get('poses', [])
                if not frames:
                    continue
                first = frames[0]
                pose = first.get('pose_keypoints_2d', []) + first.get('hand_right_keypoints_2d', []) + first.get('hand_left_keypoints_2d', [])
                if len(pose) != 150:
                    continue
                arr = np.asarray(pose, dtype=np.float32).reshape(50, 3)[:, :2]
                poses.append(_pca_align_unit(arr))
            except Exception:
                continue
        if not poses:
            return None
        mean_pose = np.mean(np.stack(poses, axis=0), axis=0).astype(np.float32)
        return mean_pose
    except Exception:
        return None


def _procrustes_2d(src: np.ndarray, dst: np.ndarray, allow_scaling: bool = True):
    """
    Compute similarity transform that maps src -> dst in least-squares sense.
    src, dst: (N,2)
    Returns (scale s, rotation R 2x2, translation t 1x2)
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src/dst must be (N,2)")
    src_c = src - src.mean(axis=0, keepdims=True)
    dst_c = dst - dst.mean(axis=0, keepdims=True)
    # Compute optimal rotation with SVD
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    if allow_scaling:
        num = np.trace((R @ (src_c.T @ dst_c)))
        den = np.sum(src_c ** 2)
        s = float(num / (den + 1e-8))
    else:
        s = 1.0
    t = dst.mean(axis=0, keepdims=True) - s * (src.mean(axis=0, keepdims=True) @ R)
    return s, R.astype(np.float32), t.astype(np.float32)

def _find_first_nondegenerate_frame(positions: np.ndarray, min_spread: float = 1e-3) -> Optional[int]:
    """
    Return index of first frame whose point spread exceeds threshold; else None.
    positions: (T, 50, 2)
    """
    Tn = positions.shape[0]
    for i in range(Tn):
        frame = positions[i]
        spread = (frame.max(axis=0) - frame.min(axis=0))
        if float(np.max(spread)) > float(min_spread):
            return i
    return None

def load_initial_pose_from_meta(npy_path: str) -> Optional[np.ndarray]:
    """
    Try to locate sidecar JSON of the displacement npy and load the first frame
    of original pose.json from dataset to serve as initial pose (J=50,2).
    Returns (50,2) or None.
    """
    js_path = npy_path[:-4] + '.json'
    if not os.path.exists(js_path):
        return None
    try:
        meta = json.load(open(js_path, 'r', encoding='utf-8'))
        # Prefer explicit p0 paths if present
        for key in ('p0_norm_path', 'p0_path'):
            p0p = meta.get(key, '')
            if isinstance(p0p, str) and os.path.exists(p0p):
                arr = np.load(p0p)
                if arr.shape == (50, 2):
                    return arr.astype(np.float32)
        # Fallback: read from original dataset first frame
        sample_dir = meta.get('sample_dir', '')
        pose_f = os.path.join(sample_dir, 'pose.json')
        if not os.path.exists(pose_f):
            return None
        js = json.load(open(pose_f, 'r', encoding='utf-8'))
        frames = js.get('poses', [])
        if not frames:
            return None
        first = frames[0]
        pose = first.get('pose_keypoints_2d', []) + first.get('hand_right_keypoints_2d', []) + first.get('hand_left_keypoints_2d', [])
        if len(pose) != 150:
            return None
        arr = np.asarray(pose, dtype=np.float32).reshape(50, 3)
        return arr[:, :2].copy()
    except Exception:
        return None


def integrate_deltas(deltas: np.ndarray, p0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    deltas: (T, 50, 2) or (T, 50, 3) -> positions: (T+1, 50, 2)
    """
    if deltas.ndim != 3 or deltas.shape[1] != 50 or deltas.shape[2] < 2:
        raise ValueError(f"Unexpected deltas shape: {deltas.shape}")
    d_xy = deltas[..., :2].astype(np.float32)
    T = d_xy.shape[0]
    if p0 is None:
        p0 = np.zeros((50, 2), dtype=np.float32)
    pos = np.zeros((T + 1, 50, 2), dtype=np.float32)
    pos[0] = p0
    pos[1:] = p0 + np.cumsum(d_xy, axis=0)
    return pos


def positions_to_pose150(seq_pos: np.ndarray, conf_value: float = 1.0) -> np.ndarray:
    """
    seq_pos: (T, 50, 2) -> pose150 sequence (T, 150) with constant confidence.
    Order assumed: body(8), right(21), left(21), consistent with dataset encoding.
    """
    if seq_pos.ndim != 3 or seq_pos.shape[1] != 50 or seq_pos.shape[2] != 2:
        raise ValueError(f"Unexpected positions shape: {seq_pos.shape}")
    T = seq_pos.shape[0]
    out = np.zeros((T, 50, 3), dtype=np.float32)
    out[..., :2] = seq_pos
    out[..., 2] = float(conf_value)
    return out.reshape(T, 150)


def visualize_from_deltas(npy_paths: List[str], out_dir: str, title_suffix: str = "", use_meta_init: bool = True, zero_init: bool = False, invert_y: bool = True, invert_x: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    viz = ASLVisualizer(invert_y=invert_y, invert_x=invert_x)
    for npy_path in npy_paths:
        name = os.path.splitext(os.path.basename(npy_path))[0]
        deltas = np.load(npy_path)
        p0 = None
        if not zero_init and use_meta_init:
            p0 = load_initial_pose_from_meta(npy_path)
        positions = integrate_deltas(deltas, p0=p0)
        pose150_seq = positions_to_pose150(positions)
        out_path = os.path.join(out_dir, f"recon_{name}.gif")
        title = f"Reconstructed {name} {title_suffix}".strip()
        viz.create_animation(pose_sequence=pose150_seq, output_path=out_path, title=title)
        print(f"Saved {out_path}")


def visualize_from_positions(npy_paths: List[str], out_dir: str, title_suffix: str = "", invert_y: bool = True, invert_x: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    viz = ASLVisualizer(invert_y=invert_y, invert_x=invert_x)
    for npy_path in npy_paths:
        name = os.path.splitext(os.path.basename(npy_path))[0]
        positions = np.load(npy_path)  # (T,50,2) or (T+1,50,2)
        if positions.ndim != 3 or positions.shape[1] != 50 or positions.shape[2] != 2:
            raise ValueError(f"Unexpected positions shape: {positions.shape}")
        # Try to load canonical initial pose from template sidecar json (same base name)
        p0 = load_canonical_initial_pose_from_template(npy_path)
        if p0 is not None and positions.shape[0] >= 1:
            # Use first nondegenerate frame to estimate rigid alignment; fallback to pure translation
            ref_idx = _find_first_nondegenerate_frame(positions, min_spread=1e-3)
            if ref_idx is not None:
                src = positions[ref_idx]
                dst = p0
                s, R, t = _procrustes_2d(src, dst, allow_scaling=False)
                Tn = positions.shape[0]
                X = positions.reshape(Tn * 50, 2)
                Xp = (s * (X @ R)) + t
                positions = Xp.reshape(Tn, 50, 2).astype(np.float32)
            else:
                shift = (p0 - positions[0]).astype(np.float32)
                positions = positions + shift
        pose150_seq = positions_to_pose150(positions)
        out_path = os.path.join(out_dir, f"recon_{name}.gif")
        title = f"Reconstructed {name} {title_suffix}".strip()
        viz.create_animation(pose_sequence=pose150_seq, output_path=out_path, title=title)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Reconstruct and visualize motion from displacement or position sequences')
    parser.add_argument('--inputs', type=str, nargs='+', help='paths to *_deltas.npy or displacement .npy files')
    parser.add_argument('--positions', type=str, nargs='+', help='paths to *_positions.npy (already integrated)')
    parser.add_argument('--out_dir', type=str, default='./outputs/displacements/recon', help='output directory for GIFs')
    parser.add_argument('--zero_init', action='store_true', help='force zero initial pose for deltas mode')
    parser.add_argument('--no_invert_y', action='store_true', help='do not invert Y axis in visualization')
    parser.add_argument('--invert_x', action='store_true', help='horizontally mirror (invert X) in visualization')
    parser.add_argument('--no_meta_init', action='store_true', help='do not try to load initial pose from sidecar json')
    args = parser.parse_args()

    if not args.inputs and not args.positions:
        raise SystemExit('Please provide --inputs or --positions')

    if args.inputs:
        visualize_from_deltas(
            npy_paths=args.inputs,
            out_dir=args.out_dir,
            title_suffix="(deltas)",
            use_meta_init=(not args.no_meta_init),
            zero_init=args.zero_init,
            invert_y=(not args.no_invert_y),
            invert_x=args.invert_x,
        )
    if args.positions:
        visualize_from_positions(
            npy_paths=args.positions,
            out_dir=args.out_dir,
            title_suffix="(positions)",
            invert_y=(not args.no_invert_y),
            invert_x=args.invert_x,
        )


if __name__ == '__main__':
    main()

# ---- Helpers for canonical initial pose from template meta ----

def _pca_align_unit(p: np.ndarray) -> np.ndarray:
    """
    p: (50,2) -> centered, PCA-rotated to align first PC to x-axis, unit scale (RMS radius=1).
    """
    if p.ndim != 2 or p.shape != (50, 2):
        return p
    # Remove invalid rows (near zeros) softly by replacing with centroid
    mask = np.isfinite(p).all(axis=1)
    q = p.copy()
    if not mask.all():
        cen = np.nanmean(q[mask], axis=0) if mask.any() else np.zeros(2, dtype=np.float32)
        q[~mask] = cen
    # Center
    cen = q.mean(axis=0, keepdims=True)
    q = q - cen
    # PCA via SVD
    U, S, Vt = np.linalg.svd(q, full_matrices=False)
    R = Vt.T  # 2x2
    q = q @ R
    # Unit scale by RMS radius
    rms = np.sqrt((q ** 2).mean())
    if rms > 1e-8:
        q = q / rms
    return q.astype(np.float32)


def load_canonical_initial_pose_from_template(positions_npy_path: str) -> Optional[np.ndarray]:
    """
    Given a template positions npy path (e.g., *_positions.npy), try to read the sidecar
    template json (same basename .json), look up 'sources' (list of disp_*.npy paths),
    and build a canonical initial pose by averaging the first frame poses of those sources
    after normalization (_pca_align_unit).

    Returns (50,2) or None.
    """
    base = os.path.splitext(positions_npy_path)[0]
    json_path = base.replace("_positions", "") + ".json"
    if not os.path.exists(json_path):
        return None
    try:
        meta = json.load(open(json_path, 'r', encoding='utf-8'))
        sources = meta.get('sources', [])
        if not sources:
            return None
        poses = []
        for disp_npy in sources:
            js_sidecar = os.path.splitext(disp_npy)[0] + ".json"
            if not os.path.exists(js_sidecar):
                continue
            try:
                s_meta = json.load(open(js_sidecar, 'r', encoding='utf-8'))
                sample_dir = s_meta.get('sample_dir', '')
                pose_f = os.path.join(sample_dir, 'pose.json')
                if not os.path.exists(pose_f):
                    continue
                js = json.load(open(pose_f, 'r', encoding='utf-8'))
                frames = js.get('poses', [])
                if not frames:
                    continue
                first = frames[0]
                pose = first.get('pose_keypoints_2d', []) + first.get('hand_right_keypoints_2d', []) + first.get('hand_left_keypoints_2d', [])
                if len(pose) != 150:
                    continue
                arr = np.asarray(pose, dtype=np.float32).reshape(50, 3)[:, :2]
                poses.append(_pca_align_unit(arr))
            except Exception:
                continue
        if not poses:
            return None
        mean_pose = np.mean(np.stack(poses, axis=0), axis=0).astype(np.float32)
        return mean_pose
    except Exception:
        return None


