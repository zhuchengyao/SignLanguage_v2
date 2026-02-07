import os
import argparse
import torch
import numpy as np

from src.latent.skeleton import create_default_skeleton
from src.latent.preprocess import (
    linear_resample,
    crop_or_pad,
    positions_to_local_rot6d,
    reconstruct_positions_from_local,
    acceptance_check_positions,
)


def load_positions_from_npy(path: str) -> torch.Tensor:
    arr = np.load(path)
    assert arr.ndim == 3 and arr.shape[-1] == 3, "Expected shape (T,J,3)"
    return torch.from_numpy(arr).float()


def synthesize_tpose_walk(T: int = 64, J: int = 22) -> torch.Tensor:
    # Simple forward motion along +x with slight arm swing
    skel = create_default_skeleton()
    pos = torch.zeros(T, J, 3)
    # Build a static pose from offsets
    parents = skel.parents.cpu().numpy()
    rest = skel.rest_offsets.cpu().numpy()
    joint_pos = np.zeros((J, 3), dtype=np.float32)
    for j in range(J):
        p = parents[j]
        if p >= 0:
            joint_pos[j] = joint_pos[p] + rest[j]
    for t in range(T):
        delta_x = 0.03 * t
        swing = 0.02 * np.sin(2 * np.pi * t / T)
        frame = joint_pos.copy()
        frame[7, 2] += swing  # left wrist z
        frame[10, 2] -= swing  # right wrist z
        frame[:, 0] += delta_x
        pos[t] = torch.from_numpy(frame)
    return pos


def main():
    parser = argparse.ArgumentParser(description="Step1: FK acceptance test")
    parser.add_argument("--input_npy", type=str, default=None, help="Optional path to (T,J,3) positions npy")
    parser.add_argument("--src_fps", type=float, default=30.0)
    parser.add_argument("--tgt_fps", type=float, default=30.0)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    if args.input_npy and os.path.isfile(args.input_npy):
        positions = load_positions_from_npy(args.input_npy)
    else:
        positions = synthesize_tpose_walk(T=args.length)

    if abs(args.src_fps - args.tgt_fps) > 1e-6:
        positions = linear_resample(positions, args.src_fps, args.tgt_fps)
    positions = crop_or_pad(positions, args.length)

    skel = create_default_skeleton(device=positions.device)
    # Use dataset-specific rest offsets estimated from the first frame to ensure exact reconstruction
    from src.latent.skeleton import estimate_rest_offsets_from_positions
    est_offsets = estimate_rest_offsets_from_positions(positions[0], skel.parents)
    skel.rest_offsets = est_offsets

    root_vel, root_trans, rot6d = positions_to_local_rot6d(positions, skel)
    rec = reconstruct_positions_from_local(root_vel, rot6d, skel)
    metrics = acceptance_check_positions(positions, rec, atol=args.atol)

    print(f"MSE={metrics['mse']:.6e}  MaxErr={metrics['max_err']:.6e}  Passed={bool(metrics['passed'])}")
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


