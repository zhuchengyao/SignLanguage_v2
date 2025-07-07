# train_vae_gat.py (with Motion Loss)
import torch
import torch.optim as optim
from tqdm import tqdm
import os
from types import SimpleNamespace

from config import ModelConfig, TrainConfig
from data_loader import create_data_loaders
from vae_model_gat import GraphSequenceVAE, vae_loss_function

def get_kl_beta(epoch, total_epochs, start_epoch, peak_epoch, peak_beta):
    if epoch < start_epoch: return 0.0
    if epoch >= peak_epoch: return peak_beta
    return peak_beta * ((epoch - start_epoch) / (peak_epoch - start_epoch))

def main():
    # -- 配置 --
    m_cfg = ModelConfig()
    t_cfg = TrainConfig()
    
    # ✨ UNCOMMENTED: 使用你在config.py中定义的参数
    # 如果你想在这里覆盖config.py中的值，可以在这里取消注释并修改
    
    # ✨ ADDED: 明确加入速度损失的权重
    t_cfg.vae_velocity_weight = 1.0 # 这是一个关键超参数，建议从1.0开始

    device = torch.device(m_cfg.device)
    print(f"🚀 Starting GAT-VAE Training on {device} with Motion Loss (velo_w={t_cfg.vae_velocity_weight})")

    # -- 数据 --
    data_cfg = SimpleNamespace(**vars(m_cfg), **vars(t_cfg))
    train_loader, val_loader, _ = create_data_loaders(data_cfg)
    if m_cfg.pose_normalize:
        m_cfg.mean = torch.from_numpy(train_loader.dataset.pose_mean).float()
        m_cfg.std = torch.from_numpy(train_loader.dataset.pose_std).float()

    # -- 模型与优化器 --
    model = GraphSequenceVAE(m_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=t_cfg.vae_lr)
    
    best_val_loss = float('inf')
    
    # -- 训练循环 --
    for epoch in range(t_cfg.vae_epochs):
        if epoch == t_cfg.kl_start_epoch:
            print(f"✨ KL Annealing starts. Resetting best_val_loss from {best_val_loss:.4f} to infinity.")
            best_val_loss = float('inf')

        model.train()
        kl_beta = get_kl_beta(epoch, t_cfg.vae_epochs, t_cfg.kl_start_epoch, t_cfg.kl_peak_epoch, t_cfg.kl_peak_beta)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{t_cfg.vae_epochs} (β={kl_beta:.4f})")
        
        # ✨ CHANGED: 初始化新的损失累加器
        total_loss, total_recon, total_kld, total_velo = 0, 0, 0, 0
        
        for _, pose_seq, mask in pbar:
            pose_seq, mask = pose_seq.to(device), mask.to(device)
            recon_seq, mu, logvar = model(pose_seq, mask)
            
            # ✨ CHANGED: 调用新的损失函数
            loss, recon, kld, velo = vae_loss_function(
                recon_seq, pose_seq, mu, logvar, mask, kl_beta, t_cfg.vae_velocity_weight
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ✨ CHANGED: 累加所有损失项
            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            total_velo += velo.item()
            
            # ✨ CHANGED: 在进度条中显示速度损失
            pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon.item():.4f}", kld=f"{kld.item():.4f}", velo=f"{velo.item():.4f}")

        # -- 验证 --
        model.eval()
        avg_val_loss, avg_val_recon, avg_val_kld, avg_val_velo = 0, 0, 0, 0
        with torch.no_grad():
            for _, pose_seq, mask in val_loader:
                pose_seq, mask = pose_seq.to(device), mask.to(device)
                recon_seq, mu, logvar = model(pose_seq, mask)
                # ✨ CHANGED: 调用新的损失函数并累加
                loss, recon, kld, velo = vae_loss_function(
                    recon_seq, pose_seq, mu, logvar, mask, kl_beta, t_cfg.vae_velocity_weight
                )
                avg_val_loss += loss.item()
                avg_val_recon += recon.item()
                avg_val_kld += kld.item()
                avg_val_velo += velo.item()

        # ✨ CHANGED: 打印所有验证集损失指标
        num_val_batches = len(val_loader)
        avg_val_loss /= num_val_batches
        avg_val_recon /= num_val_batches
        avg_val_kld /= num_val_batches
        avg_val_velo /= num_val_batches
        print(f"Epoch {epoch+1} | Val_Loss: {avg_val_loss:.4f} [Recon: {avg_val_recon:.4f}, KLD: {avg_val_kld:.4f}, Velo: {avg_val_velo:.4f}]")

        if epoch >= t_cfg.kl_peak_epoch and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(t_cfg.checkpoint_dir, "vae_gat_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🎯 New Best GAT-VAE Saved! (Post-Annealing) Val Loss: {best_val_loss:.6f} → {save_path}")

if __name__ == '__main__':
    main()