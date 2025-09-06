import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.latent2d.model_ae2d import AE2DConfig
from src.latent2d.latent_head import LatentBottleneck2D, LatentConfig2D
from src.latent2d.flow_matching import FlowUNet1D, FlowConfig2D, sample_flow
from src.latent2d.text_encoder import TextEncoder, TextEncConfig
from src.latent2d.skeleton2d import (
    build_parents_for_50,
    pose150_to_global50_xy,
    compute_rest_offsets_xy,
    normalize_bone_lengths,
    positions_to_local2d,
    FK2D,
    adjust_shoulder_mid_root,
)
from train_ae2d import build_pose_filelist


def main():
    parser = argparse.ArgumentParser(description='Train flow-matching on 2D latent')
    parser.add_argument('--data_root', type=str, default='./datasets/ASL_gloss')
    parser.add_argument('--filelist_cache', type=str, default='./datasets/ASL_gloss/.cache/filelist_train.txt')
    parser.add_argument('--max_files', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--steps', type=int, default=32, help='sampling steps for eval')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent dimension for bottleneck + flow')
    parser.add_argument('--flow_hidden', type=int, default=512, help='hidden width of flow network')
    parser.add_argument('--flow_layers', type=int, default=12, help='number of residual layers in flow network')
    parser.add_argument('--cond_drop_prob', type=float, default=0.2, help='CFG training: probability to drop text cond (use zeros)')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    # Resolve paths relative to this script's directory to avoid CWD issues
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    def to_abs(path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(proj_dir, path))
    args.data_root = to_abs(args.data_root)
    args.filelist_cache = to_abs(args.filelist_cache)
    checkpoints_dir = os.path.join(proj_dir, 'checkpoints')
    outputs_dir = os.path.join(proj_dir, 'outputs')
    parents, roots = build_parents_for_50()

    # Load AE checkpoint (best)
    # Load AE checkpoint with fallback
    ae_best = os.path.join(checkpoints_dir, 'ae2d_best.pth')
    ae_last = os.path.join(checkpoints_dir, 'ae2d_last.pth')
    ae_path = ae_best if os.path.exists(ae_best) else (ae_last if os.path.exists(ae_last) else None)
    if ae_path is None:
        raise FileNotFoundError(f"AE2D checkpoint not found. Expected at {ae_best} or {ae_last}. Please run train_ae2d.py first.")
    ckpt = torch.load(ae_path, map_location='cpu', weights_only=False)
    ae_cfg = ckpt.get('cfg', AE2DConfig())
    lat_cfg = LatentConfig2D(latent_dim=args.latent_dim)
    bottleneck = LatentBottleneck2D(ae_cfg, lat_cfg).to(device)
    bottleneck.ae.load_state_dict(ckpt['model'])
    for p in bottleneck.ae.parameters():
        p.requires_grad = False
    bottleneck.eval()

    # Flow model
    flow = FlowUNet1D(FlowConfig2D(latent_dim=lat_cfg.latent_dim, hidden=args.flow_hidden, layers=args.flow_layers)).to(device)
    opt = optim.AdamW(flow.parameters(), lr=args.lr)

    # Build file list and dataset in-memory (subset)
    files = build_pose_filelist(args.data_root, ['train'], args.max_files, args.filelist_cache)
    raw_xy_list = []  # store unnormalized xy (T,50,2)
    texts = []
    for jf in tqdm(files, desc='Load raw data'):
        try:
            js = json.load(open(jf,'r',encoding='utf-8'))
            frames = js.get('poses', [])
            if not frames:
                continue
            arr = []
            for fr in frames[:64]:
                pose = fr.get('pose_keypoints_2d', []) + fr.get('hand_right_keypoints_2d', []) + fr.get('hand_left_keypoints_2d', [])
                if len(pose)==150: arr.append(pose)
            if not arr: continue
            xy = pose150_to_global50_xy(torch.tensor(arr, dtype=torch.float32))
            xy = adjust_shoulder_mid_root(xy)
            raw_xy_list.append(xy)
            tf = jf.replace('pose.json', 'text.txt')
            if os.path.exists(tf):
                try:
                    texts.append(open(tf, 'r', encoding='utf-8').read().strip())
                except Exception:
                    texts.append("")
            else:
                texts.append("")
        except Exception:
            continue
    if not raw_xy_list:
        print('No data')
        return

    # Robust rest offsets: average over first K sequences
    with torch.no_grad():
        K = min(64, len(raw_xy_list))
        rest_sum = None
        for k in range(K):
            xy = raw_xy_list[k]
            mean_frame = xy[:min(5, xy.size(0))].mean(dim=0)
            r = compute_rest_offsets_xy(mean_frame, parents)
            rest_sum = r if rest_sum is None else (rest_sum + r)
        rest = (rest_sum / K)

    # Normalize bones using the averaged rest, then pad/truncate to T=64 and pack
    xy_batches = []
    for xy in raw_xy_list:
        xy_n = normalize_bone_lengths(xy, parents, rest)
        if xy_n.size(0) < 64:
            pad = xy_n[-1:].expand(64 - xy_n.size(0), 50, 2)
            xy_n = torch.cat([xy_n, pad], dim=0)
        else:
            xy_n = xy_n[:64]
        xy_batches.append(xy_n.view(1, 64, 100))
    X = torch.cat(xy_batches, dim=0).to(device)
    # Text cond encoder (frozen)
    txt_enc = TextEncoder(TextEncConfig()).to(device)
    C = txt_enc.encode(texts if texts else [""] * X.size(0), device=device)  # (N,512)

    # Training loop (flow matching with Gaussian base z0)
    N = X.size(0)
    B = max(1, args.batch_size)
    for epoch in range(args.epochs):
        flow.train(); total = 0.0
        perm = torch.randperm(N)
        for i in tqdm(range(0, perm.numel(), B), desc=f'Epoch {epoch+1}/{args.epochs}'):
            idxs = perm[i:i+B]
            x = X[idxs]
            with torch.no_grad():
                z1 = bottleneck.encode_to_latent(x)                          # (B,T,D)
                z0 = torch.randn_like(z1)                                    # Gaussian base
            t = torch.rand(z1.size(0), z1.size(1), 1, device=device)         # (B,T,1)
            zt = (1.0 - t) * z0 + t * z1                                     # linear path
            v_target = (z1 - z0)                                             # constant velocity
            # CFG training: random cond drop per sample
            c = C[idxs]
            drop_mask = (torch.rand(c.size(0), device=device) < args.cond_drop_prob).float().view(-1, 1)
            c = c * (1.0 - drop_mask)
            v_pred = flow(zt, t, cond=c)
            loss = F.mse_loss(v_pred, v_target)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0)
        print(f'Epoch {epoch+1}: flow_loss={total/max(1,N):.6f}')

    # Sampling sanity-check
    flow.eval()
    with torch.no_grad():
        z_samp = sample_flow(flow, steps=args.steps, shape=(1,64,bottleneck.latent_dim))
        preds = bottleneck.decode_from_latent(z_samp)
        os.makedirs(outputs_dir, exist_ok=True)
        out_pt = os.path.join(outputs_dir, 'flow2d_samples.pt')
        torch.save({'preds': preds.cpu()}, out_pt)
        print('Saved', out_pt)

    # Save flow checkpoint for later conditional inference
    os.makedirs(checkpoints_dir, exist_ok=True)
    ck = {
        'flow_state_dict': flow.state_dict(),
        'latent_dim': bottleneck.latent_dim,
        'hidden': args.flow_hidden,
        'layers': args.flow_layers,
    }
    flow_path = os.path.join(checkpoints_dir, 'flow2d_last.pth')
    torch.save(ck, flow_path)
    print('Saved', flow_path)


if __name__ == '__main__':
    main()



