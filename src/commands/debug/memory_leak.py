import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import gc

from src.config import T2M_Config
from src.model_vqvae import VQ_VAE
# 我们仍然需要导入这些，但不会使用 BucketSampler
from src.dataloader import ASLPoseDataset, collate_pose_batch, BucketSampler 

def log_memory_usage(step, epoch):
    """打印当前GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[Step {step}, Epoch {epoch}] GPU Memory: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Debug VQ-VAE Memory Leak - Final Test')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')
    args = parser.parse_args()

    cfg = T2M_Config()
    device = cfg.get_device()
    
    # 1. 加载数据 (*** 这里是关键改动 ***)
    # ==================================
    print("--- Loading Data (Using Standard DataLoader) ---")
    train_dataset = ASLPoseDataset(
        data_paths=[os.path.join(cfg.data_root, getattr(cfg, 'dataset_name', 'ASL_gloss'), "train")],
        split="train", max_seq_len=cfg.max_seq_len
    )
    
    # *** 我们不再使用 BucketSampler ***
    # sampler_cache_path = os.path.join(cfg.data_root, "ASL_gloss/.cache", "train_sampler_cache.pt")
    # train_sampler = BucketSampler(train_dataset, cfg.batch_size, boundaries=[50, 100, 150, 200], shuffle=True, cache_path=sampler_cache_path)
    
    # *** 使用标准的 DataLoader，它内置了随机打乱和批处理功能 ***
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size, # 直接在这里设置batch_size
        shuffle=True,              # 使用标准的数据打乱方式
        collate_fn=collate_pose_batch,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True # 丢弃最后一个不完整的batch，保持batch size一致
    )
    print(f"DataLoader workers: {args.num_workers}. Custom BucketSampler DEACTIVATED.")

    # 2. 初始化模型和优化器
    # ==================================
    print("--- Initializing Model ---")
    model = VQ_VAE(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.vqvae_learning_rate)
    
    # 3. 最小化训练循环
    # ==================================
    print("--- Starting Minimal Training Loop ---")
    global_step = 0
    for epoch in range(cfg.vqvae_num_epochs):
        model.train()
        for batch in train_loader:
            if batch[0] is None:
                continue
            
            _, pose_sequences, masks = batch
            pose_sequences = pose_sequences.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pose_sequences, masks)
            total_loss = outputs['recon_loss'] + 0.25 * outputs['vq_loss']
            total_loss.backward()
            optimizer.step()
            
            if global_step % 50 == 0:
                log_memory_usage(global_step, epoch + 1)
            
            global_step += 1

        print(f"--- End of Epoch {epoch + 1} ---")
        del outputs, total_loss, pose_sequences, masks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory_usage(global_step, epoch + 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDebug script interrupted.")