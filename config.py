# config.py (Memory-Optimized "Slim" Version)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import os, torch

# ──────────────────────────
#  模型架构 & 推断相关配置
# ──────────────────────────
@dataclass
class ModelConfig:
    # --- 1. 数据与维度相关参数 ---
    pose_dim: int = 150
    text_embed_dim: int = 512        # 文本维度保持不变，因为它与预训练BERT相关
    pose_normalize: bool = True
    pose_clip_range: float = 8.0
    mean: torch.Tensor = field(default_factory=lambda: torch.zeros(150))
    std: torch.Tensor = field(default_factory=lambda: torch.ones(150))

    # --- 2. VAE 架构参数 (瘦身版) ---
    vae_checkpoint_path: str = "./checkpoints/vae_gat_best.pth"
    # GAT
    vae_pose_embed_dim: int = 256      # 原: 512
    vae_gat_hidden_dims: Tuple[int, ...] = (256, 256) # 原: (512, 512, 512)
    vae_gat_heads: Tuple[int, ...] = (4, 4)           # 原: (4, 4, 4)
    # GRU & Latent z
    vae_latent_dim: int = 256          # 原: 512
    vae_hidden_dim: int = 768          # 原: 1024
    vae_gru_layers: int = 2            # 原: 3
    vae_max_seq_len: int = 120         # 保持120的截断长度

    # --- 3. Latent Diffusion (LMD) 架构参数 (瘦身版) ---
    # Transformer 噪声预测器
    ldm_attn_layers: int = 6           # 原: 8
    ldm_attn_heads: int = 8            # 头的数量影响稍小，可保持不变
    
    # Diffusion 调度器
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 1e-2

    # Classifier-Free Guidance
    cfg_uncond_prob: float = 0.2  
    cfg_uncond_text: str = ""

    # ldm_hidden_dims 这个参数在切换到Transformer后已不再使用，可以安全忽略
    ldm_hidden_dims: Tuple[int, ...] = (2048, 2048, 2048)

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────
#   训练过程相关配置
# ──────────────────────────
@dataclass
class TrainConfig:
    # 路径
    data_root: str = "./datasets"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # 训练超参
    batch_size: int = 8                # 原: 2，可以稍微尝试调高一点点，如果还超显存就改回2
    num_epochs: int = 600
    warmup_epochs: int = 10
    learning_rate: float = 2e-4        # 原: 1e-4，模型变小，学习率可以适当提高一点点
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    mixed_precision: bool = True
    save_every: int = 25

    # VAE 训练参数 (这些参数在 train_vae_gat.py 中会被覆盖，这里的值仅作参考)
    vae_lr = 1e-4
    kl_start_epoch = 50
    kl_peak_epoch = 120
    kl_peak_beta = 0.002
    vae_epochs = 200

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)