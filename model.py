# model.py (FINAL VERSION with Classifier-Free Guidance)
from __future__ import annotations
import math
from typing import List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from vae_model_gat import GraphSequenceVAE
from config import ModelConfig


class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").eval()
        for p in self.bert.parameters(): p.requires_grad_(False)
        self.proj = nn.Linear(768, cfg.text_embed_dim, bias=True)
    @torch.no_grad()
    def forward(self, texts: List[str] | str) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(texts, str): texts = [texts]
        device = self.proj.weight.device
        toks = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        attention_mask = toks['attention_mask'].to(device)
        toks = {k: v.to(device) for k, v in toks.items()}
        hidden_states = self.bert(**toks).last_hidden_state
        return self.proj(hidden_states.to(self.proj.weight.dtype)), attention_mask
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim, self.linear = dim, nn.Linear(dim, dim)
        nn.init.normal_(self.linear.weight, std=0.02)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1)))
        phase = t.to(torch.float32).unsqueeze(-1) * freqs
        pe = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return self.linear(pe)


class DiffusionScheduler(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # 使用余弦调度计算 alphas_cumprod
        steps = cfg.num_diffusion_steps
        s = 0.008 # 一个小的偏移量，防止 t=0 时出现问题
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((t / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # 从 alphas_cumprod 推导出 betas
        betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0., 0.999) # 确保 beta 值在合理范围内

        # 注册所有需要的 buffer
        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod[:-1].to(torch.float32))
        self.register_buffer("sqrt_1m_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod).to(torch.float32))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod).to(torch.float32))

    def add_noise(self, x0, noise, t):
        # add_noise 的逻辑保持不变
        a = self.sqrt_alphas_cumprod[t].view(-1, 1)
        s = self.sqrt_1m_alphas_cumprod[t].view(-1, 1)
        return a * x0 + s * noise



class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_attention_heads: int):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(query_dim, num_attention_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=query_dim, kdim=context_dim, vdim=context_dim,
            num_heads=num_attention_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4), nn.GELU(), nn.Linear(query_dim * 4, query_dim)
        )
        self.norm3 = nn.LayerNorm(query_dim)
    def forward(self, x, context=None, context_mask=None):
        x = x + self.attn1(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.attn2(self.norm2(x), context, context, key_padding_mask=context_mask)[0]
        x = x + self.ff(self.norm3(x))
        return x
class CrossAttentionNoisePredictor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.latent_dim = cfg.vae_latent_dim
        self.text_embed_dim = cfg.text_embed_dim
        self.time_proj = TimeEmbedding(self.latent_dim)
        self.latent_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.blocks = nn.ModuleList([
            BasicTransformerBlock(
                query_dim=self.latent_dim,
                context_dim=self.text_embed_dim,
                num_attention_heads=cfg.ldm_attn_heads
            ) for _ in range(cfg.ldm_attn_layers)
        ])
        self.out_proj = nn.Linear(self.latent_dim, self.latent_dim)
    def forward(self, z, text_e, t, text_mask):
        z = self.latent_proj(z)
        t_e = self.time_proj(t)
        x = z + t_e
        x = x.unsqueeze(1)
        inverted_text_mask = text_mask == 0
        for block in self.blocks:
            x = block(x, context=text_e, context_mask=inverted_text_mask)
        return self.out_proj(x).squeeze(1)


# ✨ REWRITTEN: 主模型现在完整支持 Classifier-Free Guidance
class LatentDiffusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # VAE
        self.vae = GraphSequenceVAE(cfg).to(self.device)
        print(f"📦 Loading GAT-VAE from {cfg.vae_checkpoint_path}...")
        self.vae.load_state_dict(torch.load(cfg.vae_checkpoint_path, map_location=self.device))
        self.vae.eval()
        for p in self.vae.parameters(): p.requires_grad_(False)
        print("✅ GAT-VAE loaded and frozen.")

        # Core Components
        self.txt_e = TextEncoder(cfg)
        self.noise_pred = CrossAttentionNoisePredictor(cfg)
        self.sched = DiffusionScheduler(cfg)
        
        if cfg.pose_normalize:
            self.register_buffer("pose_mean", cfg.mean)
            self.register_buffer("pose_std", cfg.std)

    # ✨ CHANGED: forward 方法现在包含随机丢弃文本条件的逻辑
    def forward(self, texts: List[str], pose_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = self.vae.encode(pose_seq, mask)
            z = mu 
        
        t = torch.randint(0, self.cfg.num_diffusion_steps, (z.size(0),), device=self.device)
        noise = torch.randn_like(z)
        noisy_z = self.sched.add_noise(z, noise, t)
        
        # Classifier-Free Guidance Training
        text_e, text_mask = self.txt_e(texts)
        # 以 p 的概率丢弃条件
        uncond_mask = torch.rand(text_e.size(0), device=self.device) < self.cfg.cfg_uncond_prob
        if uncond_mask.any():
            uncond_text_e, uncond_text_mask = self.txt_e([self.cfg.cfg_uncond_text] * uncond_mask.sum())
            # 确保无条件嵌入的序列长度与有条件的一致，用padding补齐
            if uncond_text_e.size(1) < text_e.size(1):
                pad_len = text_e.size(1) - uncond_text_e.size(1)
                uncond_text_e = nn.functional.pad(uncond_text_e, (0, 0, 0, pad_len))
                uncond_text_mask = nn.functional.pad(uncond_text_mask, (0, pad_len))
            
            text_e[uncond_mask] = uncond_text_e
            text_mask[uncond_mask] = uncond_text_mask

        pred_noise = self.noise_pred(noisy_z, text_e, t, text_mask)
        loss = nn.functional.mse_loss(pred_noise, noise)
        return loss

    # ✨ CHANGED: sample 方法现在实现了 CFG 的推理逻辑
    @torch.no_grad()
    def sample(self, texts: List[str], T: int, num_steps: int = 50, guidance_scale: float = 7.5) -> torch.Tensor:
        batch_size = len(texts)
        z = torch.randn(batch_size, self.cfg.vae_latent_dim, device=self.device)
        
        # 1. 获取有条件和无条件的文本嵌入
        cond_text_e, cond_text_mask = self.txt_e(texts)
        uncond_text_e, uncond_text_mask = self.txt_e([self.cfg.cfg_uncond_text] * batch_size)
        
        # 2. 确保序列长度一致
        max_len = cond_text_e.size(1)
        if uncond_text_e.size(1) < max_len:
            pad_len = max_len - uncond_text_e.size(1)
            uncond_text_e = nn.functional.pad(uncond_text_e, (0, 0, 0, pad_len))
            uncond_text_mask = nn.functional.pad(uncond_text_mask, (0, pad_len))

        ts = torch.linspace(self.cfg.num_diffusion_steps - 1, 0, num_steps, device=self.device).round().long()
        for t_step in tqdm(ts, desc=f"CFG Sampling (w={guidance_scale})"):
            t = t_step.expand(batch_size)
            
            # 3. 预测两个噪声
            noise_cond = self.noise_pred(z, cond_text_e, t, cond_text_mask)
            noise_uncond = self.noise_pred(z, uncond_text_e, t, uncond_text_mask)
            
            # 4. 应用指导
            eps = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            # 5. 执行 DDIM 步骤
            a_bar = self.sched.alphas_cumprod[t_step]
            sqrt_a, sqrt_1ma = a_bar.sqrt(), (1 - a_bar).sqrt()
            z0 = (z - sqrt_1ma * eps) / sqrt_a.clamp(min=1e-8)
            
            if t_step == 0: z = z0; break
            current_index = (ts == t_step).nonzero(as_tuple=True)[0]
            if current_index + 1 >= len(ts): z = z0; break
            t_prev = ts[current_index + 1]
            a_prev = self.sched.alphas_cumprod[t_prev]
            z = a_prev.sqrt() * z0 + (1 - a_prev).sqrt() * eps
        
        recon_poses = self.vae.decode(z, T)
        
        if self.cfg.pose_normalize and hasattr(self, 'pose_mean') and hasattr(self, 'pose_std'):
            recon_poses = recon_poses * self.pose_std + self.pose_mean
            
        return recon_poses