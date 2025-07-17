# motion_dit.py (Corrected)
import os
import torch
import torch.nn as nn
from typing import List
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from config import ModelConfig
from motion_vae import GraphTransformerVAE
from modules import TimeEmbedding, DiTBlock

class MotionDiT(nn.Module):
    # ... (This class does not need any changes)
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.latent_dim = cfg.dit_latent_dim
        self.text_embed_dim = cfg.text_embed_dim
        self.time_embed = TimeEmbedding(self.latent_dim)
        self.text_proj = nn.Linear(self.text_embed_dim, self.latent_dim)
        self.latent_proj = nn.Linear(cfg.vae_latent_dim, self.latent_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(self.latent_dim, cfg.dit_heads) for _ in range(cfg.dit_layers)
        ])
        self.out_proj = nn.Linear(self.latent_dim, cfg.vae_latent_dim)
    def forward(self, z, t, text_embeds, text_mask):
        # z: (B, num_latent_tokens, vae_latent_dim)
        z = self.latent_proj(z) # No .unsqueeze(1) needed. Shape: (B, num_tokens, dit_latent_dim)
        
        t_embed = self.time_embed(t).unsqueeze(1)
        text_embeds = self.text_proj(text_embeds)
        
        # 将 time embedding 和 text embedding 拼接作为 condition
        context = torch.cat([t_embed, text_embeds], 1)   # (B,1+L,D)

        x = z
        for block in self.blocks:
            x = block(x, context)
            
        return self.out_proj(x) # Output shape: (B, num_tokens, vae_latent_dim)

class MotionDiffusionModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # This is the line causing the error
        self.device = torch.device(cfg.get_device())

        # VAE
        self.vae = GraphTransformerVAE(cfg).to(self.device)
        print(f"📦 Preparing Grap-Transformer-VAE...")
        
        # ✨ FIX: Use the correct final VAE checkpoint path ✨
        final_vae_path = cfg.vae_final_checkpoint_path
        if os.path.exists(final_vae_path):
            print(f"   -> Loading weights from {final_vae_path}...")
            vae_state = torch.load(final_vae_path, map_location=self.device)
            self.vae.load_state_dict(vae_state, strict=False)   # ← 放这里！
        else:
            print(f"   -> ⚠️ VAE weights not found at {final_vae_path}.")
            
        self.vae.eval()
        for p in self.vae.parameters(): p.requires_grad_(False)

        # Text Encoder
        self.text_encoder_model = BertModel.from_pretrained(cfg.text_encoder_model).eval()
        self.tokenizer = BertTokenizer.from_pretrained(cfg.text_encoder_model)
        for p in self.text_encoder_model.parameters(): p.requires_grad_(False)

        # DiT Noise Predictor
        self.noise_predictor = MotionDiT(cfg)
        
        # ... (The rest of the file remains the same)
        # Diffusion Scheduler
        self.num_steps = cfg.num_diffusion_steps
        if cfg.beta_schedule == "cosine":
            t = torch.linspace(0, self.num_steps, self.num_steps + 1, dtype=torch.float64)
            alphas_cumprod = torch.cos(((t / self.num_steps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0., 0.999)
        else: # linear
            betas = torch.linspace(0.0001, 0.02, self.num_steps)
        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alphas_cumprod", torch.cumprod(1. - betas, dim=0).to(torch.float32))
        if cfg.pose_normalize:
            self.register_buffer("pose_mean", cfg.mean)
            self.register_buffer("pose_std", cfg.std)
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        return self.text_encoder_model(**tokens).last_hidden_state, tokens['attention_mask']

    def add_noise(self, x0, noise, t):
        # x0 shape: (B, T, D)
        # t shape: (B,)
        
        a_cumprod_t = self.alphas_cumprod[t]
        
        # ✨ THIS IS THE FIX ✨
        # Reshape to (B, 1, 1) to broadcast across token and latent dimensions
        a_cumprod_t = a_cumprod_t.view(-1, 1, 1) 
        
        return a_cumprod_t.sqrt() * x0 + (1 - a_cumprod_t).sqrt() * noise

    def forward(self, texts: List[str], pose_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = self.vae.encode(pose_seq, mask)
            z = mu
        t = torch.randint(0, self.num_steps, (z.size(0),), device=self.device)
        noise = torch.randn_like(z)
        noisy_z = self.add_noise(z, noise, t)
        text_embeds, text_mask = self.encode_text(texts)
        uncond_mask = torch.rand(z.size(0), device=self.device) < self.cfg.cfg_uncond_prob
        if uncond_mask.any():
            uncond_embeds, uncond_attn_mask = self.encode_text([self.cfg.cfg_uncond_text] * uncond_mask.sum())
            target_seq_len = text_embeds.size(1)
            current_seq_len = uncond_embeds.size(1)
            if current_seq_len < target_seq_len:
                pad_len = target_seq_len - current_seq_len
                uncond_embeds = torch.nn.functional.pad(uncond_embeds, (0, 0, 0, pad_len))
                uncond_attn_mask = torch.nn.functional.pad(uncond_attn_mask, (0, pad_len))
            text_embeds[uncond_mask] = uncond_embeds
            text_mask[uncond_mask] = uncond_attn_mask
        pred_noise = self.noise_predictor(noisy_z, t, text_embeds, text_mask)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, texts: List[str], T: int, num_inference_steps: int = 50, guidance_scale: float = 7.5) -> torch.Tensor:
        B = len(texts)
        # ✨ NEW: Initial noise is now a sequence ✨
        z = torch.randn(B, self.cfg.num_latent_tokens, self.cfg.vae_latent_dim, device=self.device)
        
        cond_embeds, cond_mask = self.encode_text(texts)
        uncond_embeds, uncond_mask = self.encode_text([self.cfg.cfg_uncond_text] * B)
        target_seq_len = cond_embeds.size(1)
        current_seq_len = uncond_embeds.size(1)
        if current_seq_len < target_seq_len:
            pad_len = target_seq_len - current_seq_len
            uncond_embeds = torch.nn.functional.pad(uncond_embeds, (0, 0, 0, pad_len))
            uncond_mask = torch.nn.functional.pad(uncond_mask, (0, pad_len))
        timesteps = torch.linspace(self.num_steps - 1, 0, num_inference_steps, device=self.device).long()
        for i, t in enumerate(tqdm(timesteps, desc=f"DiT Sampling (w={guidance_scale})")):
            t_batch = t.expand(B)
            pred_noise_cond = self.noise_predictor(z, t_batch, cond_embeds, cond_mask)
            pred_noise_uncond = self.noise_predictor(z, t_batch, uncond_embeds, uncond_mask)
            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=self.device)
            pred_x0 = (z - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            if i == len(timesteps) - 1:
                z = pred_x0
                break
            pred_dir_xt = (1 - alpha_t_prev).sqrt() * pred_noise
            z = alpha_t_prev.sqrt() * pred_x0 + pred_dir_xt
        recon_poses = self.vae.decode(z, T) # Pass the final denoised sequence z
        if self.cfg.pose_normalize and hasattr(self, 'pose_mean'):
            recon_poses = recon_poses * self.pose_std.to(self.device) + self.pose_mean.to(self.device)
        return recon_poses
