import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from .dataloader import ASLPoseDataset, BucketSampler, collate_pose_batch


def create_pose_dataloaders(cfg, sampler_cache_name: str) -> Tuple[DataLoader, DataLoader, ASLPoseDataset]:
    train_dataset = ASLPoseDataset(
        data_paths=[os.path.join(cfg.data_root, "ASL_gloss/train")],
        split="train",
        cache_in_memory=cfg.dataset_cache_in_memory,
        cache_max_items=cfg.dataset_cache_max_items,
    )
    val_dataset = ASLPoseDataset(
        data_paths=[os.path.join(cfg.data_root, "ASL_gloss/dev")],
        split="dev",
        extern_mean=train_dataset.pose_mean,
        extern_std=train_dataset.pose_std,
        cache_in_memory=cfg.dataset_cache_in_memory,
        cache_max_items=cfg.dataset_cache_max_items,
    )

    sampler_cache_path = os.path.join(cfg.data_root, "ASL_gloss/.cache", sampler_cache_name)
    train_sampler = BucketSampler(
        train_dataset,
        cfg.batch_size,
        boundaries=[50, 100, 150, 200, 300],
        shuffle=True,
        cache_path=sampler_cache_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_pose_batch,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        collate_fn=collate_pose_batch,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader, train_dataset


def _metric_or_zero(outputs: Dict[str, torch.Tensor], key: str, ref: torch.Tensor) -> torch.Tensor:
    value = outputs.get(key)
    if value is None:
        return ref.new_tensor(0.0)
    return value


def compute_vqvae_total_loss(outputs: Dict[str, torch.Tensor], cfg, epoch: int, use_warmup: bool = True) -> Dict[str, torch.Tensor]:
    recon_loss = outputs["recon_loss"]
    vq_loss = outputs["vq_loss"]
    coarse_vq_loss = _metric_or_zero(outputs, "coarse_vq_loss", recon_loss)

    weighted_recon = _metric_or_zero(outputs, "recon_weighted", recon_loss)
    bone_loss = _metric_or_zero(outputs, "bone_length_loss", recon_loss)
    velocity_loss = _metric_or_zero(outputs, "velocity_loss", recon_loss)
    accel_loss = _metric_or_zero(outputs, "accel_loss", recon_loss)

    if use_warmup:
        warmup_scale = min(1.0, (epoch + 1) / max(1, cfg.vq_weight_warmup_epochs))
    else:
        warmup_scale = 1.0
    vq_weight = cfg.vq_loss_weight * warmup_scale

    total_loss = (
        cfg.recon_loss_weight * recon_loss
        + vq_weight * (vq_loss + cfg.coarse_vq_loss_weight * coarse_vq_loss)
        + cfg.weighted_recon_loss_weight * weighted_recon
        + cfg.bone_length_loss_weight * bone_loss
        + cfg.temporal_velocity_loss_weight * velocity_loss
        + cfg.temporal_accel_loss_weight * accel_loss
    )

    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "vq_loss": vq_loss,
        "coarse_vq_loss": coarse_vq_loss,
        "recon_weighted": weighted_recon,
        "bone_length_loss": bone_loss,
        "velocity_loss": velocity_loss,
        "accel_loss": accel_loss,
        "vq_weight": recon_loss.new_tensor(vq_weight),
    }
