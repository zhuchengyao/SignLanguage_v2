import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import math
from .config import T2M_Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class MotionEncoder(nn.Module):
    def __init__(self, cfg: T2M_Config):
        super().__init__()
        self.input_projection = nn.Linear(cfg.pose_dim, cfg.motion_encoder_hidden_dim)
        self.pos_encoding = PositionalEncoding(cfg.motion_encoder_hidden_dim, cfg.model_max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.motion_encoder_hidden_dim, nhead=cfg.motion_encoder_heads,
            dim_feedforward=cfg.motion_encoder_hidden_dim * 4, dropout=cfg.motion_encoder_dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.motion_encoder_layers)
        self.final_norm = nn.LayerNorm(cfg.motion_encoder_hidden_dim)
        self.temporal_downsample = nn.Conv1d(
            cfg.motion_encoder_hidden_dim, cfg.embedding_dim,
            kernel_size=cfg.downsample_rate, stride=cfg.downsample_rate
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        x = x.transpose(1, 2)
        x = self.temporal_downsample(x)
        x = x.transpose(1, 2)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, cfg: T2M_Config):
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.codebook_size = cfg.codebook_size
        self.commitment_cost = cfg.commitment_cost
        self.use_ema = cfg.use_ema_update
        self.ema_decay = cfg.ema_decay
        self.epsilon = cfg.epsilon
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        if self.use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(self.codebook_size))
            self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.register_buffer('usage_count', torch.zeros(self.codebook_size))
        self.lookup_chunk_tokens = getattr(cfg, "vq_lookup_chunk_tokens", 4096)

    @torch.no_grad()
    def _argmin_euclid_chunked(self, flat_input: torch.Tensor) -> torch.Tensor:
        N = flat_input.size(0)
        chunk = self.lookup_chunk_tokens or N
        emb = self.embedding.weight
        out_list = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            x = flat_input[start:end].float()
            dist = (x.pow(2).sum(dim=1, keepdim=True) + emb.float().pow(2).sum(dim=1).unsqueeze(0) - 2 * (x @ emb.float().t()))
            out_list.append(torch.argmin(dist, dim=1))
        return torch.cat(out_list, dim=0)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, Tdown, D = inputs.shape
        flat_input = inputs.reshape(-1, D)
        
        if self.use_ema and self.training:
            encoding_indices = self._argmin_euclid_chunked(flat_input)
        else:
            distances = (flat_input.pow(2).sum(dim=1, keepdim=True) + self.embedding.weight.pow(2).sum(dim=1) - 2 * flat_input @ self.embedding.weight.t())
            encoding_indices = torch.argmin(distances, dim=1)

        quantized = self.embedding(encoding_indices).view(B, Tdown, D)
        if self.training:
            self.usage_count.index_add_(0, encoding_indices, torch.ones_like(encoding_indices, dtype=torch.float))
            if self.use_ema:
                encodings = F.one_hot(encoding_indices, self.codebook_size).type(flat_input.dtype)
                self.ema_cluster_size.mul_(self.ema_decay).add_(encodings.sum(0), alpha=1 - self.ema_decay)
                dw = encodings.t() @ flat_input
                self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
                embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(embed_normalized)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = self.commitment_cost * e_latent_loss
        codebook_loss = torch.tensor(0.0, device=inputs.device)

        quantized_st = inputs + (quantized - inputs).detach()
        return quantized_st, encoding_indices.view(B, Tdown), {'vq_loss': vq_loss, 'commitment_loss': e_latent_loss, 'codebook_loss': codebook_loss}

    @torch.no_grad()
    def reinit_dead_codes(self, inputs: torch.Tensor):
        if not self.training: return 0
        dead_codes_indices = torch.where(self.usage_count == 0)[0]
        if dead_codes_indices.numel() == 0: return 0
        flat_input = inputs.reshape(-1, self.embedding_dim)
        num_dead = dead_codes_indices.numel()
        rand_indices = torch.randint(0, flat_input.size(0), (num_dead,), device=inputs.device)
        new_vectors = flat_input[rand_indices]
        self.embedding.weight.data[dead_codes_indices] = new_vectors
        if self.use_ema: self.ema_w.data[dead_codes_indices] = new_vectors; self.ema_cluster_size[dead_codes_indices] = 1.0
        self.usage_count.zero_(); return num_dead

class MotionDecoder(nn.Module):
    def __init__(self, cfg: T2M_Config):
        super().__init__()
        self.input_projection = nn.Linear(cfg.embedding_dim, cfg.motion_decoder_hidden_dim)
        self.temporal_upsample = nn.ConvTranspose1d(cfg.motion_decoder_hidden_dim, cfg.motion_decoder_hidden_dim, kernel_size=cfg.downsample_rate, stride=cfg.downsample_rate)
        self.pos_encoding = PositionalEncoding(cfg.motion_decoder_hidden_dim, cfg.model_max_seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.motion_decoder_hidden_dim, nhead=cfg.motion_decoder_heads,
            dim_feedforward=cfg.motion_decoder_hidden_dim * 4, dropout=cfg.motion_decoder_dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.motion_decoder_layers)
        self.output_projection = nn.Linear(cfg.motion_decoder_hidden_dim, cfg.pose_dim)
        self.query_embedding = nn.Parameter(torch.randn(1, cfg.model_max_seq_len, cfg.motion_decoder_hidden_dim))

    def forward(self, memory: torch.Tensor, target_length: int) -> torch.Tensor:
        B = memory.size(0)
        memory = self.input_projection(memory)
        memory = memory.transpose(1, 2)
        memory = self.temporal_upsample(memory)
        memory = memory.transpose(1, 2)
        memory = memory[:, :target_length, :]
        memory = self.pos_encoding(memory)
        queries = self.query_embedding.expand(B, -1, -1)[:, :target_length, :]
        queries = self.pos_encoding(queries)
        decoded = self.transformer(tgt=queries, memory=memory)
        return self.output_projection(decoded)

class VQ_VAE(nn.Module):
    def __init__(self, cfg: T2M_Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = MotionEncoder(cfg)
        self.quantizer = VectorQuantizer(cfg)
        self.decoder = MotionDecoder(cfg)

    def encode(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        z_e = self.encoder(x, mask)
        z_q, indices, vq_losses = self.quantizer(z_e)
        return z_q, indices, vq_losses

    def decode(self, indices: torch.Tensor, target_length: int) -> torch.Tensor:
        z_q = self.quantizer.embedding(indices)
        return self.decoder(z_q, target_length)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
        B, T, D = x.shape
        z_e = self.encoder(x, mask)
        z_q, indices, vq_losses = self.quantizer(z_e)
        
        target_length = int(mask.sum(dim=1).max().item())
        reconstructed_short = self.decoder(z_q, target_length)
        
        if target_length < T:
            reconstructed = F.pad(reconstructed_short, (0, 0, 0, T - target_length))
        else:
            reconstructed = reconstructed_short

        valid_positions = mask.unsqueeze(-1).float()
        recon_loss = F.mse_loss(reconstructed * valid_positions, x * valid_positions, reduction='sum')
        recon_loss = recon_loss / valid_positions.sum().clamp(min=1.0)
        return {'reconstructed': reconstructed, 'indices': indices, 'recon_loss': recon_loss, **vq_losses, 'z_e': z_e}