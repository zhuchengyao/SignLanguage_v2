import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.latent2d.model_ae2d import AutoEncoder2D, AE2DConfig
from src.latent2d.skeleton2d import (
    build_parents_for_50,
    pose150_to_global50_xy,
    compute_rest_offsets_xy,
    normalize_bone_lengths,
    positions_to_local2d,
    FK2D,
    adjust_shoulder_mid_root,
)


def build_pose_filelist(root: str, splits: list[str], max_files: int, cache_path: str) -> list[str]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            files = [line.strip() for line in f if line.strip()]
        if max_files > 0:
            files = files[:max_files]
        if files:
            print(f"Loaded filelist cache: {len(files)} files from {cache_path}")
            return files
    files = []
    for sp in splits:
        base = os.path.join(root, sp)
        if not os.path.isdir(base):
            continue
        for sid in sorted(os.listdir(base)):
            jf = os.path.join(base, sid, 'pose.json')
            if os.path.exists(jf):
                files.append(jf)
                if max_files > 0 and len(files) >= max_files:
                    break
        if max_files > 0 and len(files) >= max_files:
            break
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(files))
    print(f"Saved filelist cache: {len(files)} files to {cache_path}")
    return files


def load_first_n_samples(root: str, max_samples: int = 64, max_T: int = 64, splits: list[str] | None = None, cache_path: str | None = None):
    from tqdm import tqdm as _tqdm
    if splits is None:
        splits = ['train']
    if cache_path is None:
        cache_path = os.path.join(root, '.cache', f'pose_filelist_{"_".join(splits)}.txt')
    filelist = build_pose_filelist(root, splits, max_samples, cache_path)
    samples = []
    for jf in _tqdm(filelist, desc='Loading JSONs'):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                js = json.load(f)
            frames = js.get('poses', [])
            if not frames:
                continue
            arr = []
            for fr in frames[:max_T]:
                pose = fr.get('pose_keypoints_2d', []) + fr.get('hand_right_keypoints_2d', []) + fr.get('hand_left_keypoints_2d', [])
                if len(pose) == 150:
                    arr.append(pose)
            if arr:
                samples.append(torch.tensor(arr, dtype=torch.float32))
        except KeyboardInterrupt:
            raise
        except Exception:
            continue
    return samples


def main():
    parser = argparse.ArgumentParser(description='Train simple 2D AE as step-2 baseline')
    parser.add_argument('--data_root', type=str, default='./datasets/ASL_gloss')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=-1, help='-1 means load all available samples')
    parser.add_argument('--splits', type=str, default='train', help='comma-separated: train,dev,test')
    parser.add_argument('--filelist_cache', type=str, default=None, help='path to cache file list; default under data_root/.cache')
    # Loss weights
    parser.add_argument('--w_theta', type=float, default=1.0)
    parser.add_argument('--w_scales', type=float, default=0.5)
    parser.add_argument('--w_root', type=float, default=1.0)
    parser.add_argument('--w_vel', type=float, default=0.2, help='temporal velocity loss weight')
    parser.add_argument('--w_acc', type=float, default=0.0, help='temporal acceleration loss weight')
    parser.add_argument('--hand_theta_weight', type=float, default=2.0, help='per-joint multiplier for hand joints in theta loss')
    parser.add_argument('--hand_scale_weight', type=float, default=1.5, help='per-joint multiplier for hand joints in scale loss')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    parents, roots = build_parents_for_50()
    parents = parents.to(device)

    # Load small subset
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    max_samples = (args.max_samples if args.max_samples != -1 else 10**9)
    seqs = load_first_n_samples(args.data_root, max_samples=max_samples, max_T=64, splits=splits, cache_path=(args.filelist_cache or None))
    if not seqs:
        print('No data found')
        return

    # Prepare training tensors
    xy_list = []
    rest = None
    for Tseq in seqs:
        xy = pose150_to_global50_xy(Tseq).to(device)
        xy = adjust_shoulder_mid_root(xy)
        if rest is None:
            rest = compute_rest_offsets_xy(xy[:min(5, xy.size(0))].mean(dim=0), parents)
        xy = normalize_bone_lengths(xy, parents, rest)
        xy_list.append(xy)

    # Pack into mini-batches (pad/truncate to 64)
    def to_xy_tensor(xy):
        # xy: (T,50,2) -> (1,64,100)
        T = xy.size(0)
        if T < 64:
            pad = xy[-1:].expand(64 - T, 50, 2)
            xy = torch.cat([xy, pad], dim=0)
        else:
            xy = xy[:64]
        return xy.view(1, 64, -1)

    X = torch.cat([to_xy_tensor(xy) for xy in xy_list], dim=0)  # (N,64,100)
    N = X.size(0)

    # Train/val split
    perm_all = torch.randperm(N)
    n_val = max(1, int(N * args.val_ratio))
    val_idx = perm_all[:n_val]
    train_idx = perm_all[n_val:]
    print(f"Loaded {N} sequences. Train: {train_idx.numel()}, Val: {val_idx.numel()} (batch_size={args.batch_size})")

    # Compute a stable rest from training sequences
    with torch.no_grad():
        rest_sum = None
        parents_cpu = parents.detach().cpu()
        for idx in train_idx.tolist():
            xy = X[idx].view(64, 50, 2).detach().cpu()
            mean_frame = xy[:5].mean(dim=0)
            r = compute_rest_offsets_xy(mean_frame, parents_cpu)
            rest_sum = r if rest_sum is None else (rest_sum + r)
        rest = (rest_sum / len(train_idx)).to(device)

    cfg = AE2DConfig(max_len=64)
    model = AutoEncoder2D(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5, verbose=True)

    # Loss: supervise on ground-truth local params (avoid backprop through FK to prevent inplace grad issues)
    def forward_and_loss(xy_in):
        B, T, _ = xy_in.shape
        preds = model(xy_in)  # (B,T,J+J+2)
        J = 50
        pred_theta = preds[..., :J]
        pred_scales = preds[..., J:2*J]
        pred_root = preds[..., 2*J:2*J+2]

        xy_gt = xy_in.view(B, T, J, 2)
        total = 0.0
        # Per-joint weights: body(0..7)=1.0, left(8..28)=hand_weight, right(29..49)=hand_weight
        jt_w_theta = torch.ones(J, device=xy_in.device) * 1.0
        jt_w_theta[8:29] = args.hand_theta_weight
        jt_w_theta[29:50] = args.hand_theta_weight
        jt_w_scales = torch.ones(J, device=xy_in.device) * 1.0
        jt_w_scales[8:29] = args.hand_scale_weight
        jt_w_scales[29:50] = args.hand_scale_weight

        for b in range(B):
            with torch.no_grad():
                rp_gt, th_gt, sc_gt = positions_to_local2d(xy_gt[b], parents, roots, rest)
                root_target = rp_gt[:, roots[0], :]
                sc_gt = sc_gt.clamp(0.8, 1.2)
            # Per-joint weighted L1 losses
            l_theta = (torch.abs(pred_theta[b] - th_gt) * jt_w_theta.view(1, J)).mean()
            l_sc = (torch.abs(pred_scales[b].clamp(0.8, 1.2) - sc_gt) * jt_w_scales.view(1, J)).mean()
            l_root = F.l1_loss(pred_root[b], root_target)
            # Temporal smoothness on theta/root
            l_vel = torch.tensor(0.0, device=xy_in.device)
            l_acc = torch.tensor(0.0, device=xy_in.device)
            if args.w_vel > 0:
                dtheta = pred_theta[b][1:] - pred_theta[b][:-1]
                droot = pred_root[b][1:] - pred_root[b][:-1]
                l_vel = (dtheta.abs().mean() + droot.abs().mean()) * 0.5
            if args.w_acc > 0 and T >= 3:
                ddtheta = pred_theta[b][2:] - 2*pred_theta[b][1:-1] + pred_theta[b][:-2]
                ddroot = pred_root[b][2:] - 2*pred_root[b][1:-1] + pred_root[b][:-2]
                l_acc = (ddtheta.abs().mean() + ddroot.abs().mean()) * 0.5
            total = total + (args.w_theta * l_theta + args.w_scales * l_sc + args.w_root * l_root + args.w_vel * l_vel + args.w_acc * l_acc)
        return total / B

    best_val = float('inf')
    epochs_no_improve = 0
    os.makedirs('./checkpoints', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        # Shuffle train indices and run mini-batches
        perm = train_idx[torch.randperm(train_idx.numel())]
        total = 0.0
        for i in tqdm(range(0, perm.numel(), args.batch_size), desc=f'Epoch {epoch+1}/{args.epochs}'):
            idxs = perm[i:i+args.batch_size]
            batch = X[idxs].to(device)
            loss = forward_and_loss(batch)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / (perm.numel() // max(1, args.batch_size))

        # Validation
        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for i in range(0, val_idx.numel(), args.batch_size):
                idxs = val_idx[i:i+args.batch_size]
                batch = X[idxs].to(device)
                vtotal += forward_and_loss(batch).item()
            val_loss = vtotal / max(1, (val_idx.numel() // args.batch_size))

        print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}')
        scheduler.step(val_loss)

        # Early stopping and checkpointing
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({'cfg': cfg, 'model': model.state_dict()}, './checkpoints/ae2d_best.pth')
            print('Saved ./checkpoints/ae2d_best.pth (best so far)')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f'Early stopping at epoch {epoch+1} (no improve {args.patience} epochs). Best val={best_val:.6f}')
                break

    # Save last
    torch.save({'cfg': cfg, 'model': model.state_dict()}, './checkpoints/ae2d_last.pth')
    print('Saved ./checkpoints/ae2d_last.pth')


if __name__ == '__main__':
    main()


