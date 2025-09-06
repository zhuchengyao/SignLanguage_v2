import os
import json
import argparse
import torch
import torch.nn.functional as F

from src.latent2d.model_ae2d import AE2DConfig
from src.latent2d.latent_head import LatentBottleneck2D, LatentConfig2D
from src.latent2d.flow_matching import FlowUNet1D, FlowConfig2D, sample_flow_cfg
from src.latent2d.text_encoder import TextEncoder, TextEncConfig
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


def main():
    parser = argparse.ArgumentParser(description='Text-conditioned 2D Flow inference with CFG')
    parser.add_argument('--data_root', type=str, default='./datasets/ASL_gloss')
    parser.add_argument('--filelist_cache', type=str, default='./datasets/ASL_gloss/.cache/filelist_train.txt')
    parser.add_argument('--guidance', type=float, default=3.0)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--text', type=str, default='hello')
    parser.add_argument('--out_gif', type=str, default='./outputs/flow2d_text.gif')
    # Stabilization options
    parser.add_argument('--smooth', type=int, default=0, help='temporal smoothing window (odd, 0 disables)')
    parser.add_argument('--fix_hand_scales', action='store_true', help='set hand finger scales to 1.0')
    parser.add_argument('--clamp_scales', type=float, nargs=2, default=None, help='min max clamp for bone scales, e.g., 0.8 1.2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Resolve project-relative paths
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    def to_abs(path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(proj_dir, path))
    args.data_root = to_abs(args.data_root)
    args.filelist_cache = to_abs(args.filelist_cache)
    checkpoints_dir = os.path.join(proj_dir, 'checkpoints')
    outputs_dir = os.path.join(proj_dir, 'outputs')

    # Load AE
    ae_best = os.path.join(checkpoints_dir, 'ae2d_best.pth')
    ae_last = os.path.join(checkpoints_dir, 'ae2d_last.pth')
    ae_path = ae_best if os.path.exists(ae_best) else (ae_last if os.path.exists(ae_last) else None)
    if ae_path is None:
        raise FileNotFoundError(f"AE2D checkpoint not found. Expected at {ae_best} or {ae_last}.")
    ae_ck = torch.load(ae_path, map_location='cpu', weights_only=False)
    ae_cfg = ae_ck.get('cfg', AE2DConfig())
    lat_cfg = LatentConfig2D(latent_dim=64)
    bottleneck = LatentBottleneck2D(ae_cfg, lat_cfg).to(device)
    bottleneck.ae.load_state_dict(ae_ck['model'])
    bottleneck.eval()

    # Load Flow
    flow_path = os.path.join(checkpoints_dir, 'flow2d_last.pth')
    flow_ck = torch.load(flow_path, map_location='cpu', weights_only=False)
    Fcfg = FlowConfig2D(latent_dim=flow_ck.get('latent_dim', 64), hidden=flow_ck.get('hidden', 512), layers=flow_ck.get('layers', 12))
    flow = FlowUNet1D(Fcfg).to(device)
    flow.load_state_dict(flow_ck['flow_state_dict'])
    flow.eval()

    # Text encoder
    txt_enc = TextEncoder(TextEncConfig()).to(device)
    cond = txt_enc.encode([args.text], device=device)  # (1,H)

    # Estimate rest offsets from one sample for FK
    parents, roots = build_parents_for_50()
    files = build_pose_filelist(args.data_root, ['train'], 1, args.filelist_cache)
    if not files:
        raise RuntimeError('No files to estimate rest offsets')
    js = json.load(open(files[0], 'r', encoding='utf-8'))
    frames = js.get('poses', [])
    arr = []
    for fr in frames[:5]:
        pose = fr.get('pose_keypoints_2d', []) + fr.get('hand_right_keypoints_2d', []) + fr.get('hand_left_keypoints_2d', [])
        if len(pose)==150: arr.append(pose)
    pose150 = torch.tensor(arr, dtype=torch.float32)           # (T,150)
    xy = pose150_to_global50_xy(pose150)                        # (T,50,2)
    xy = adjust_shoulder_mid_root(xy)
    rest = compute_rest_offsets_xy(xy.mean(dim=0), parents)     # (50,2)

    # Sample latent with CFG
    z = sample_flow_cfg(flow, steps=args.steps, shape=(1,64,lat_cfg.latent_dim), cond=cond, guidance=args.guidance)
    preds = bottleneck.decode_from_latent(z)
    J = 50
    local_theta = preds[0, :, :J]
    bone_scales = preds[0, :, J:2*J]
    root_xy = preds[0, :, 2*J:2*J+2]

    # Optional constraints and smoothing to reduce jitter and stiff fingers
    if args.fix_hand_scales:
        hand_mask = torch.zeros(J, dtype=torch.bool)
        hand_mask[8:29] = True
        hand_mask[29:50] = True
        bone_scales[hand_mask] = 1.0
    if args.clamp_scales is not None:
        lo, hi = args.clamp_scales
        bone_scales = bone_scales.clamp(lo, hi)
    if args.smooth and args.smooth > 1:
        k = args.smooth if args.smooth % 2 == 1 else args.smooth + 1
        # simple box smoothing
        def smooth_time(x: torch.Tensor, k: int) -> torch.Tensor:
            if k <= 1 or x.size(0) < 3:
                return x
            pad = k // 2
            xpad = torch.cat([x[:pad], x, x[-pad:]], dim=0)
            out = []
            for t in range(x.size(0)):
                seg = xpad[t:t+k]
                out.append(seg.mean(dim=0))
            return torch.stack(out, dim=0)
        local_theta = smooth_time(local_theta, k)
        root_xy = smooth_time(root_xy, k)
        bone_scales = smooth_time(bone_scales, k)

    # FK
    fk = FK2D(parents.to(device), roots, rest.to(device)).to(device)
    local_theta = local_theta.to(device)
    bone_scales = bone_scales.to(device)
    root_xy = root_xy.to(device)
    root_pos = torch.zeros(64, J, 2, device=device)
    root_pos[:, roots[0], :] = root_xy
    xy_out = fk(local_theta, root_pos, bone_scales).cpu()

    # Brace wrists to body wrists for visual tightness
    xy_out[:, 8] = xy_out[:, 4]
    xy_out[:, 29] = xy_out[:, 7]
    pose150 = global50_xy_to_pose150(xy_out).detach()

    viz = ASLVisualizer()
    # Resolve output path under project outputs by default if relative
    out_gif = args.out_gif if os.path.isabs(args.out_gif) else os.path.join(outputs_dir, os.path.basename(args.out_gif))
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    viz.create_animation(pose150.numpy(), out_gif, title=f'Text: {args.text}', fps=30)
    print('Saved:', out_gif)


if __name__ == '__main__':
    main()


