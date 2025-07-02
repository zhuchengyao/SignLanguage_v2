# config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import os, torch


# ──────────────────────────
#  模型架构 & 推断相关配置
# ──────────────────────────
@dataclass
class ModelConfig:
    # 时序与姿态维度
    clip_len: int = 50                 # 每段动作帧数
    pose_dim: int = 150                # 50 joints × 3

    # 嵌入 / 隐层尺寸
    pose_embed_dim: int = 2048
    text_embed_dim: int = 512
    hidden_dim: int = 2048

    # GAT-Pose Encoder
    gat_hidden_dims: Tuple[int, ...] = (1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048)
    gat_heads: Tuple[int, ...] | None = (8, 8, 8, 8, 16, 16, 16, 16)

    # Transformer-based Pose Decoder
    decoder_layers: int = 8
    decoder_heads: int = 8

    # Diffusion
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    recon_loss_weight: float = 50.0    # L_recon 相对权重
    num_layers: int = 4                # NoisePredictor 里 STBlock 数量

    # 姿态归一化
    pose_normalize: bool = True
    pose_clip_range: float = 8.0
    mean: torch.Tensor = field(default_factory=lambda: torch.zeros(150))
    std: torch.Tensor = field(default_factory=lambda: torch.ones(150))

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────
#   训练过程相关配置
# ──────────────────────────
@dataclass
class TrainConfig:
    # 路径
    data_root: str = "./datasets/ASL_gloss"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # 训练超参
    batch_size: int = 16
    num_epochs: int = 600
    warmup_epochs: int = 10
    learning_rate: float = 5e-4
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    mixed_precision: bool = True
    save_every: int = 25               # 保存 checkpoint 的 epoch 间隔

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
