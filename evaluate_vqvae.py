import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from src.config import T2M_Config
from src.model_vqvae import VQ_VAE
from src.dataloader import ASLPoseDataset, collate_pose_batch
from src.asl_visualizer import ASLVisualizer


@torch.no_grad()
def evaluate(checkpoint_path: str, split: str = 'dev', output_dir: str = './outputs/eval_vqvae', num_samples: int = 8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint and config
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg: T2M_Config = ckpt['cfg']
    model = VQ_VAE(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # Dataset
    dataset = ASLPoseDataset(
        data_paths=[os.path.join(cfg.data_root, f"ASL_gloss/{split}")],
        split=split,
        pose_normalize=False,
        extern_mean=cfg.mean.cpu().numpy() if hasattr(cfg, 'mean') and cfg.mean is not None else None,
        extern_std=cfg.std.cpu().numpy() if hasattr(cfg, 'std') and cfg.std is not None else None,
        cache_in_memory=True,
        cache_max_items=getattr(cfg, 'dataset_cache_max_items', 1024),
    )

    # Simple sequential dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pose_batch, num_workers=0, pin_memory=False)

    os.makedirs(output_dir, exist_ok=True)
    vis = ASLVisualizer()

    picked = 0
    metrics = []
    for batch in tqdm(loader, total=min(len(loader), num_samples)):
        texts, poses, masks = batch
        if poses is None:
            continue
        poses = poses.to(device)
        masks = masks.to(device)

        out = model(poses, masks)
        recon = out['reconstructed']

        # Denormalize for visualization and metrics
        if hasattr(cfg, 'mean') and cfg.mean is not None:
            mean = cfg.mean.to(device)
            std = cfg.std.to(device)
            poses_dn = poses * std + mean
            recon_dn = recon * std + mean
        else:
            poses_dn = poses
            recon_dn = recon

        # Per-frame MSE on x,y only
        def xy_only(x):
            B, T, D = x.shape
            x = x.view(B, T, -1, 3)[..., :2]
            return x

        xy_gt = xy_only(poses_dn)
        xy_rc = xy_only(recon_dn)
        valid = masks.view(1, -1, 1, 1).float()
        mse_xy = ((xy_gt - xy_rc) ** 2 * valid).sum() / valid.sum().clamp(min=1.0)

        metrics.append({'mse_xy': mse_xy.item()})

        # Save side-by-side GIF: GT top, Recon bottom
        seq_gt = poses_dn[0].detach().cpu().numpy()
        seq_rc = recon_dn[0].detach().cpu().numpy()

        base = f"sample_{picked:03d}"
        vis.create_animation(seq_gt, os.path.join(output_dir, base + '_gt.gif'), title=f"GT: {texts[0]}")
        vis.create_animation(seq_rc, os.path.join(output_dir, base + '_recon.gif'), title=f"Recon: {texts[0]}")

        picked += 1
        if picked >= num_samples:
            break

    # Summary
    if metrics:
        mse_vals = [m['mse_xy'] for m in metrics]
        print(f"Eval {split}: MSE(xy) mean={np.mean(mse_vals):.4f} std={np.std(mse_vals):.4f} n={len(mse_vals)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/vqvae_model.pth')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--output_dir', type=str, default='./outputs/eval_vqvae')
    parser.add_argument('--num_samples', type=int, default=8)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.split, args.output_dir, args.num_samples)


if __name__ == '__main__':
    main()


