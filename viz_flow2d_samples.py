import os
import json
import argparse
import torch

from src.latent2d.skeleton2d import (
    build_parents_for_50,
    pose150_to_global50_xy,
    global50_xy_to_pose150,
    compute_rest_offsets_xy,
    FK2D,
    adjust_shoulder_mid_root,
)
from train_ae2d import build_pose_filelist
from src.asl_visualizer import ASLVisualizer


def compute_rest_from_dataset(data_root: str, filelist_cache: str) -> torch.Tensor:
    parents, _ = build_parents_for_50()
    files = build_pose_filelist(data_root, ['train'], 1, filelist_cache)
    if not files:
        raise RuntimeError('No files found to estimate rest offsets')
    jf = files[0]
    js = json.load(open(jf, 'r', encoding='utf-8'))
    frames = js.get('poses', [])
    arr = []
    for fr in frames[:5]:
        pose = fr.get('pose_keypoints_2d', []) + fr.get('hand_right_keypoints_2d', []) + fr.get('hand_left_keypoints_2d', [])
        if len(pose)==150:
            arr.append(pose)
    xy = pose150_to_global50_xy(torch.tensor(arr, dtype=torch.float32))
    xy = adjust_shoulder_mid_root(xy)
    rest = compute_rest_offsets_xy(xy.mean(dim=0), parents)
    return rest


def smooth_time(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1 or x.size(0) < 3:
        return x
    pad = k // 2
    kernel = torch.ones(k, device=x.device, dtype=x.dtype) / k
    # x: (T, ...)
    x2 = x
    if x.dim() == 1:
        x2 = x.unsqueeze(1)
    xpad = torch.cat([x2[:pad], x2, x2[-pad:]], dim=0)
    out = []
    for t in range(x2.size(0)):
        seg = xpad[t:t+k]
        out.append((seg * kernel.view(-1, *([1] * (seg.dim()-1)))).sum(dim=0))
    out = torch.stack(out, dim=0)
    return out if x.dim() > 1 else out.squeeze(1)


def main():
    parser = argparse.ArgumentParser(description='Visualize 2D Flow samples as GIF')
    parser.add_argument('--data_root', type=str, default='./datasets/ASL_gloss')
    parser.add_argument('--filelist_cache', type=str, default='./datasets/ASL_gloss/.cache/filelist_train.txt')
    parser.add_argument('--samples_pt', type=str, default='./outputs/flow2d_samples.pt')
    parser.add_argument('--out_gif', type=str, default='./outputs/flow2d_sample.gif')
    parser.add_argument('--smooth', type=int, default=5, help='temporal smoothing window (odd)')
    parser.add_argument('--fix_hand_scales', action='store_true', help='set hand finger scales to 1.0')
    parser.add_argument('--clamp_scales', type=float, nargs=2, default=[0.8, 1.2], help='min max clamp for bone scales')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    def to_abs(path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(proj_dir, path))
    args.data_root = to_abs(args.data_root)
    args.filelist_cache = to_abs(args.filelist_cache)
    args.samples_pt = to_abs(args.samples_pt)
    out_gif = args.out_gif if os.path.isabs(args.out_gif) else os.path.join(proj_dir, 'outputs', os.path.basename(args.out_gif))
    parents, roots = build_parents_for_50()
    rest = compute_rest_from_dataset(args.data_root, args.filelist_cache)

    data = torch.load(args.samples_pt, map_location='cpu')
    preds = data['preds']  # (B,T, 50+50+2)
    B, T, _ = preds.shape
    J = 50
    local_theta = preds[..., :J]
    bone_scales = preds[..., J:2*J]
    root_xy = preds[..., 2*J:2*J+2]

    # Optional constraints to reduce jitter
    if args.fix_hand_scales:
        hand_mask = torch.zeros(J, dtype=torch.bool)
        hand_mask[8:29] = True; hand_mask[29:50] = True
        bone_scales[..., hand_mask] = 1.0
    if args.clamp_scales is not None:
        lo, hi = args.clamp_scales
        bone_scales = bone_scales.clamp(lo, hi)

    if args.smooth and args.smooth > 1:
        k = args.smooth if args.smooth % 2 == 1 else args.smooth + 1
        local_theta = smooth_time(local_theta[0], k).unsqueeze(0)
        root_xy = smooth_time(root_xy[0], k).unsqueeze(0)
        bone_scales = smooth_time(bone_scales[0], k).unsqueeze(0)

    fk = FK2D(parents, roots, rest)
    xy_list = []
    for b in range(B):
        root_pos = torch.zeros(T, J, 2)
        root_pos[:, roots[0], :] = root_xy[b]
        xy = fk(local_theta[b], root_pos, bone_scales[b])  # (T,J,2)
        xy_list.append(xy)

    # Align wrists explicitly to body wrists for rendering (ensure hand roots lie on body joints)
    xy = xy_list[0]
    # body wrists: left index 4, right index 7; hand roots: 8 and 29
    xy[:, 8] = xy[:, 4]
    xy[:, 29] = xy[:, 7]
    pose150 = global50_xy_to_pose150(xy)

    viz = ASLVisualizer()
    os.makedirs(os.path.dirname(args.out_gif), exist_ok=True)
    viz.create_animation(pose150.numpy(), out_gif, title='Flow2D Sample', fps=30)
    print('Saved GIF:', out_gif)


if __name__ == '__main__':
    main()






