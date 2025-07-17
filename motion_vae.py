# motion_vae.py
import torch
import torch.nn as nn
from typing import Tuple

# ✨ THIS LINE IS THE FIX ✨
from config import ModelConfig
from modules import PoseGatEncoder, SpatialMlpDecoder, PositionalEncoding

class GraphTransformerVAE(nn.Module):
    """
    Combines a GAT spatial encoder and a Transformer temporal encoder for pose sequences.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.pose_dim = cfg.pose_dim
        self.latent_dim = cfg.vae_latent_dim

        # 1. Spatial Encoder (GAT)
        self.spatial_encoder = PoseGatEncoder(
            num_joints=self.pose_dim // 3,
            pose_embed_dim=cfg.vae_pose_embed_dim,
            gat_hidden_dims=cfg.vae_gat_hidden_dims,
            gat_heads=cfg.vae_gat_heads
        )

        # 2. Temporal Encoder (Transformer)
        self.pos_encoder = PositionalEncoding(cfg.vae_pose_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.vae_pose_embed_dim, 
            nhead=cfg.vae_transformer_heads,
            batch_first=True,
            activation="gelu"
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.vae_transformer_layers)
        
        # Latent space projection
        self.fc_mu = nn.Linear(cfg.vae_pose_embed_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(cfg.vae_pose_embed_dim, self.latent_dim)
        
        # 3. Temporal Decoder (Transformer)
        self.decoder_latent_proj = nn.Linear(self.latent_dim, cfg.vae_pose_embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.vae_pose_embed_dim,
            nhead=cfg.vae_transformer_heads,
            batch_first=True,
            activation="gelu"
        )
        self.temporal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.vae_transformer_layers)
        
        # 4. Spatial Decoder (MLP)
        self.spatial_decoder = SpatialMlpDecoder(
            pose_embed_dim=cfg.vae_pose_embed_dim,
            pose_dim=self.pose_dim
        )
        self.sos_token = nn.Parameter(torch.randn(1, 1, cfg.vae_pose_embed_dim))

        # In motion_vae.py, class GraphTransformerVAE
    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, PoseDim)
        B, T, _ = x.shape
        
        pose_embeds = self.spatial_encoder(x) # (B, T, EmbDim)
        pose_embeds = self.pos_encoder(pose_embeds)
        
        padding_mask = ~mask # Transformer expects True for masked positions
        encoded_seq = self.temporal_encoder(pose_embeds, src_key_padding_mask=padding_mask)
        # encoded_seq: (B, T, EmbDim)
        
        # --- ✨ NEW: Multi-Token Pooling Logic ✨ ---
        chunk_size = self.cfg.vae_latent_chunk_size
        # 确保序列长度是 chunk_size 的整数倍，不足则填充
        pad_len = (chunk_size - T % chunk_size) % chunk_size
        if pad_len > 0:
            encoded_seq = torch.nn.functional.pad(encoded_seq, (0, 0, 0, pad_len))
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=False)

        T_padded = encoded_seq.shape[1]
        num_chunks = T_padded // chunk_size
        
        # 将 mask 变形以用于加权平均
        mask_chunks = mask.reshape(B, num_chunks, chunk_size).unsqueeze(-1) # (B, num_chunks, chunk_size, 1)
        
        # 将序列分块
        seq_chunks = encoded_seq.reshape(B, num_chunks, chunk_size, -1) # (B, num_chunks, chunk_size, EmbDim)
        
        # 对每个 chunk 内进行加权平均池化
        # 分子：(seq * mask).sum()
        # 分母：mask.sum()
        chunk_sum = (seq_chunks * mask_chunks).sum(dim=2) # (B, num_chunks, EmbDim)
        mask_sum = mask_chunks.sum(dim=2).clamp(min=1e-9) # (B, num_chunks, 1)
        
        pooled_output = chunk_sum / mask_sum # (B, num_chunks, EmbDim)
        # pooled_output is now a sequence of latents
        
        # --- ✨ END NEW LOGIC ✨ ---
        
        # mu 和 logvar 现在也是序列
        mu = self.fc_mu(pooled_output)       # (B, num_chunks, LatentDim)
        logvar = self.fc_logvar(pooled_output) # (B, num_chunks, LatentDim)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # mu, logvar: (B, num_chunks, LatentDim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std # Returns (B, num_chunks, LatentDim)

    def decode(self, z: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        # z: (B, num_chunks, LatentDim)
        B = z.size(0)
        
        # The latent sequence `z` acts as the memory for the decoder
        memory = self.decoder_latent_proj(z) # (B, num_chunks, EmbDim)
        
        # Create the decoder's input sequence, starting with a <SOS> token
        # The target sequence length is now the original full pose sequence length
        tgt_seq = self.pos_encoder(self.sos_token.expand(B, target_seq_len, -1))

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_seq_len).to(z.device)
        
        decoded_embeds = self.temporal_decoder(
            tgt=tgt_seq,
            memory=memory, # `memory` is now a sequence
            tgt_mask=tgt_mask
        )

        recon_x = self.spatial_decoder(decoded_embeds)
        return recon_x

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar) # z is now (B, num_chunks, LatentDim)
        recon_x = self.decode(z, x.size(1))
        return recon_x, mu, logvar