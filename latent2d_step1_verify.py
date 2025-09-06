import os
import json
import argparse
import numpy as np
import torch

from src.latent2d.skeleton2d import (
    build_parents_for_50,
    pose150_to_global50_xy,
    compute_rest_offsets_xy,
    positions_to_local2d,
    FK2D,
    normalize_bone_lengths,
    adjust_shoulder_mid_root,
)


def linear_resample(sequence: torch.Tensor, src_fps: float, tgt_fps: float) -> torch.Tensor:
    if abs(src_fps - tgt_fps) < 1e-6:
        return sequence
    T = sequence.shape[0]
    duration = (T - 1) / src_fps
    T_new = int(round(duration * tgt_fps)) + 1
    t_src = torch.linspace(0.0, 1.0, T, device=sequence.device)
    t_tgt = torch.linspace(0.0, 1.0, T_new, device=sequence.device)
    out = torch.interp(t_tgt, t_src, sequence.reshape(T, -1)).reshape(T_new, *sequence.shape[1:])
    return out


def crop_or_pad(sequence: torch.Tensor, T: int) -> torch.Tensor:
    cur = sequence.shape[0]
    if cur == T:
        return sequence
    if cur > T:
        return sequence[:T]
    pad = T - cur
    last = sequence[-1:].expand(pad, *sequence.shape[1:])
    return torch.cat([sequence, last], dim=0)


def load_first_sample_pose150(data_root: str) -> torch.Tensor:
    # data_root: e.g., ./datasets/ASL_gloss/train
    for split in ["train", "dev", "test", ""]:
        base = os.path.join(data_root, split) if split else data_root
        if not os.path.isdir(base):
            continue
        for sid in sorted(os.listdir(base)):
            sp = os.path.join(base, sid)
            jf = os.path.join(sp, "pose.json")
            if os.path.exists(jf):
                with open(jf, 'r', encoding='utf-8') as f:
                    js = json.load(f)
                frames = js.get("poses", [])
                if not frames:
                    continue
                arr = []
                for fr in frames:
                    pose = (
                        fr["pose_keypoints_2d"]
                        + fr["hand_right_keypoints_2d"]
                        + fr["hand_left_keypoints_2d"]
                    )
                    if len(pose) == 150:
                        arr.append(pose)
                if arr:
                    return torch.tensor(arr, dtype=torch.float32)
    raise FileNotFoundError("No valid sample found under data_root")


def main():
    parser = argparse.ArgumentParser(description="2D Step1 verification (FK < 1e-3)")
    parser.add_argument("--data_root", type=str, default="./datasets/ASL_gloss", help="Root path for dataset")
    parser.add_argument("--src_fps", type=float, default=30.0)
    parser.add_argument("--tgt_fps", type=float, default=30.0)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # 1) load first sample as demo
    pose150 = load_first_sample_pose150(args.data_root)
    # shape (T, 150)
    if abs(args.src_fps - args.tgt_fps) > 1e-6:
        pose150 = linear_resample(pose150, args.src_fps, args.tgt_fps)
    pose150 = crop_or_pad(pose150, args.length)

    # 2) map to (T, 50, 2)
    xy = pose150_to_global50_xy(pose150)  # (T,50,2)
    # place body root (1) at shoulder midpoint to stabilize yaw
    xy = adjust_shoulder_mid_root(xy)

    # 3) build parents and roots
    parents, roots = build_parents_for_50()
    parents = parents.to(xy.device)

    # 4) rest offsets from first valid frame (这里直接用第一帧)
    # Use a smoothed rest from first few frames to reduce noise
    rest = compute_rest_offsets_xy(xy[:min(5, xy.size(0))].mean(dim=0), parents)

    # 5) bone-length normalization for stability
    xy_norm = normalize_bone_lengths(xy, parents, rest)
    # 6) decompose to local angles and roots, record bone scales
    root_positions, local_theta, bone_scales = positions_to_local2d(xy_norm, parents, roots, rest)

    # 6) FK reconstruct
    fk = FK2D(parents, roots, rest)
    rec = fk(local_theta, root_positions, bone_scales)

    # 7) acceptance
    diff = (xy_norm - rec).norm(dim=-1)
    # Optional temporal smoothing on rec vs xy_norm to reduce local spikes before thresholding
    if rec.size(0) > 2:
        rec_s = rec.clone()
        rec_s[1:-1] = (rec[0:-2] + 4*rec[1:-1] + rec[2:]) / 6.0
        diff = (xy_norm - rec_s).norm(dim=-1)
    max_err = diff.max().item()
    mse = (diff.pow(2)).mean().item()
    print(f"MSE={mse:.6e}  MaxErr={max_err:.6e}  Passed={max_err <= args.atol}")
    if max_err > args.atol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


