from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import os, torch


@dataclass
class ModelConfig:
    # ----- data -----
    pose_dim: int = 150
    pose_normalize: bool = True
    pose_clip_range: float = 8.0
    mean: torch.Tensor = field(default_factory=lambda: torch.zeros(150))
    std: torch.Tensor = field(default_factory=lambda: torch.ones(150))

    # ----- text -----
    text_embed_dim: int = 768
    text_encoder_model: str = "bert-base-uncased"

    # ----- VAE -----
    vae_latent_dim: int = 512
    vae_pose_embed_dim: int = 512
    vae_gat_hidden_dims: Tuple[int, ...] = (512, 512)
    vae_gat_heads: Tuple[int, ...] = (8, 8)
    vae_transformer_layers: int = 6
    vae_transformer_heads: int = 8
    vae_max_seq_len: int = 120
    vae_latent_chunk_size: int = 4  # <‑‑ average every k frames →  latent token

    # ----- DiT -----
    dit_latent_dim: int = 512
    dit_layers: int = 12
    dit_heads: int = 8

    # ----- Diffusion -----
    num_diffusion_steps: int = 1_000
    beta_schedule: str = "cosine"  # or "linear"
    cfg_uncond_prob: float = 0.15
    cfg_uncond_text: str = ""

    # ----- paths -----
    vae_recon_checkpoint_path: str = "./checkpoints/vae_recon_only_best.pth"
    vae_final_checkpoint_path: str = "./checkpoints/vae_final_best.pth"

    # ----- experimental flags -----
    use_rope: bool = True
    use_temporal_gate: bool = True

    # ----- derived -----
    num_latent_tokens: int = field(init=False)

    # utils
    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.num_latent_tokens = self.vae_max_seq_len // self.vae_latent_chunk_size


@dataclass
class TrainConfig:
    data_root: str = "./datasets"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # EMA & AMP
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0

    # workers
    num_workers: int = 4

    # ----- stage configs -----
    @dataclass
    class Diffusion:
        epochs: int = 800
        batch_size: int = 16
        lr: float = 1e-4
        warmup_steps: int = 2_000

    diffusion: "TrainConfig.Diffusion" = field(default_factory=Diffusion)

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs("outputs", exist_ok=True)


model_cfg = ModelConfig()
train_cfg = TrainConfig()