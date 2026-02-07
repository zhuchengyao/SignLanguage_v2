import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from src.latent2d.data_utils import load_first_n_pose_sequences
from src.latent2d.model_ae2d import AutoEncoder2D, AE2DConfig
from src.latent2d.skeleton2d import (
    build_parents_for_50,
    pose150_to_global50_xy,
    compute_rest_offsets_xy,
    normalize_bone_lengths,
    positions_to_local2d,
    adjust_shoulder_mid_root,
)


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
    # Diffusion settings
    parser.add_argument('--diffusion_steps', type=int, default=200)
    parser.add_argument('--diffusion_sample_steps', type=int, default=50)
    parser.add_argument('--beta_start', type=float, default=1e-4)
    parser.add_argument('--beta_end', type=float, default=2e-2)
    args = parser.parse_args()

    device = torch.device(args.device)
    parents, roots = build_parents_for_50()
    parents = parents.to(device)

    # Load small subset
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    max_samples = (args.max_samples if args.max_samples != -1 else 10**9)
    seqs = load_first_n_pose_sequences(
        data_root=args.data_root,
        splits=splits,
        max_samples=max_samples,
        max_frames=64,
        cache_path=(args.filelist_cache or None),
    )
    if not seqs:
        print('No data found')
        return
    print(f"Loaded raw sequences: {len(seqs)}")

    # Prepare training tensors
    xy_list = []
    rest = None
    for Tseq in seqs:
        if len(xy_list) % 64 == 0:
            print(f"[prep] processing seq {len(xy_list)}/{len(seqs)}")
        xy = pose150_to_global50_xy(Tseq).to(device)
        xy = adjust_shoulder_mid_root(xy)
        if rest is None:
            rest = compute_rest_offsets_xy(xy[:min(5, xy.size(0))].mean(dim=0), parents)
        xy = normalize_bone_lengths(xy, parents, rest)
        xy_list.append(xy)
    print("[prep] xy_list done")

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

    print("[pack] building X tensor")
    X = torch.cat([to_xy_tensor(xy) for xy in xy_list], dim=0)  # (N,64,100)
    print(f"[pack] X shape: {tuple(X.shape)}")
    N = X.size(0)

    # Train/val split
    perm_all = torch.randperm(N)
    n_val = max(1, int(N * args.val_ratio))
    val_idx = perm_all[:n_val]
    train_idx = perm_all[n_val:]
    print(f"Loaded {N} sequences. Train: {train_idx.numel()}, Val: {val_idx.numel()} (batch_size={args.batch_size})")

    # Compute a stable rest from training sequences
    with torch.no_grad():
        print("[rest] computing averaged rest from training split")
        rest_sum = None
        parents_cpu = parents.detach().cpu()
        for idx in train_idx.tolist():
            xy = X[idx].view(64, 50, 2).detach().cpu()
            mean_frame = xy[:5].mean(dim=0)
            r = compute_rest_offsets_xy(mean_frame, parents_cpu)
            rest_sum = r if rest_sum is None else (rest_sum + r)
        rest = (rest_sum / len(train_idx)).to(device)
        print("[rest] done")

    cfg = AE2DConfig(
        max_len=64,
        diffusion_steps=args.diffusion_steps,
        diffusion_sample_steps=args.diffusion_sample_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    model = AutoEncoder2D(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)

    # Loss: supervise on ground-truth local params (avoid backprop through FK to prevent inplace grad issues)
    def forward_and_loss(xy_in):
        B, T, _ = xy_in.shape
        J = 50
        xy_gt = xy_in.view(B, T, J, 2)
        # per-dim weights for diffusion loss
        # Per-joint weights: body(0..7)=1.0, left(8..28)=hand_weight, right(29..49)=hand_weight
        jt_w_theta = torch.ones(J, device=xy_in.device) * 1.0
        jt_w_theta[8:29] = args.hand_theta_weight
        jt_w_theta[29:50] = args.hand_theta_weight
        jt_w_scales = torch.ones(J, device=xy_in.device) * 1.0
        jt_w_scales[8:29] = args.hand_scale_weight
        jt_w_scales[29:50] = args.hand_scale_weight
        out_dim = J + J + 2
        w = torch.ones(out_dim, device=xy_in.device)
        w[:J] = args.w_theta * jt_w_theta
        w[J:2*J] = args.w_scales * jt_w_scales
        w[2*J:2*J+2] = args.w_root

        # Build target local parameters (y0) per sample
        target = torch.zeros(B, T, out_dim, device=xy_in.device)
        for b in range(B):
            with torch.no_grad():
                rp_gt, th_gt, sc_gt = positions_to_local2d(xy_gt[b], parents, roots, rest)
                root_target = rp_gt[:, roots[0], :]
                sc_gt = sc_gt.clamp(0.8, 1.2)
            target[b, :, :J] = th_gt
            target[b, :, J:2*J] = sc_gt
            target[b, :, 2*J:2*J+2] = root_target

        return model.diffusion_loss(
            xy_in,
            target,
            weights=w,
            vel_weight=args.w_vel,
            acc_weight=args.w_acc,
        )

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

