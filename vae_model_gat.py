# vae_model_gat.py (Fixed for Batching)
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

# 导入我们之前定义好的骨骼结构函数
try:
    # 假设 getSkeletalModelStructure 和 build_edge_index 在 model.py 中
    from model import getSkeletalModelStructure, build_edge_index
except (ImportError, ModuleNotFoundError):
    # 如果找不到，提供一个备用定义
    def getSkeletalModelStructure():
        BODY_JOINTS, HAND_JOINTS = 8, 21
        pose_conn = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
        hand_conn = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
        triplets = []
        for s, e in pose_conn: triplets.append((s, e, "body"))
        for s, e in hand_conn: triplets.append((s + BODY_JOINTS, e + BODY_JOINTS, "lh"))
        for s, e in hand_conn: triplets.append((s + BODY_JOINTS + HAND_JOINTS, e + BODY_JOINTS + HAND_JOINTS, "rh"))
        return triplets
    
    def build_edge_index(num_joints: int = 50):
        triplets = getSkeletalModelStructure()
        edges = {(s, e) for s, e, _ in triplets} | {(e, s) for s, e, _ in triplets}
        return torch.tensor(sorted(edges), dtype=torch.long).t()

try:
    import torch_geometric.nn as gnn
except ImportError:
    gnn = None

# 1. 空间编码器 (GAT-based)
class PoseGatEncoder(nn.Module):
    def __init__(self, num_joints, pose_embed_dim, gat_hidden_dims, gat_heads):
        super().__init__()
        if gnn is None:
            raise ImportError("torch_geometric must be installed to use PoseGatEncoder.")
        self.num_joints = num_joints
        
        dims = (3, *gat_hidden_dims)
        heads = list(gat_heads)

        self.register_buffer("edge_index", build_edge_index(num_joints))
        
        self.gconvs = nn.ModuleList()
        for i in range(len(dims) - 1):
            # 使用 GATv2Conv，性能通常更好
            self.gconvs.append(gnn.GATv2Conv(dims[i], dims[i+1] // heads[i], heads=heads[i], concat=True))

        # GAT输出的总维度
        self.gat_output_dim = self.num_joints * dims[-1]
        self.out_proj = nn.Linear(self.gat_output_dim, pose_embed_dim)
        self.act = nn.ELU()

    # ✨ FIXED: 修正了 forward 方法以正确处理批处理
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, J*3)
        B, T, _ = x_seq.shape
        device = x_seq.device
        
        # 1. 创建批处理版本的 edge_index
        # 单个骨骼的边数
        num_edges = self.edge_index.size(1)
        # 批次中的图总数
        num_graphs = B * T
        
        # 复制 edge_index 以匹配批次大小
        edge_index_batched = self.edge_index.repeat(1, num_graphs)
        
        # 创建偏移量，以便每个图的节点索引是独立的
        offset = torch.arange(num_graphs, device=device).repeat_interleave(num_edges) * self.num_joints
        edge_index_batched = edge_index_batched + offset.view(1, -1)
        
        # 2. 将输入展平为 GAT 期望的 2D 格式
        # (B, T, J*3) -> (B*T, J, 3) -> (B*T*J, 3)
        x = x_seq.view(B * T * self.num_joints, 3)
        
        # 3. 通过 GAT 层
        for conv in self.gconvs:
            # 现在 x 是 2D，edge_index_batched 也是适配的
            x = self.act(conv(x, edge_index_batched))
        
        # 4. 重新整形并投影
        # x 的形状是 (B*T*J, D_gat_out), D_gat_out 是 GAT 最后一层的输出维度
        # 我们需要把它变回 (B*T, J * D_gat_out)
        x = x.view(B * T, self.gat_output_dim)
        x = self.out_proj(x) # 投影到最终的姿态嵌入维度
        
        return x.view(B, T, -1) # (B, T, D_pose_embed)


# 2. 空间解码器 (MLP-based)
class SpatialMlpDecoder(nn.Module):
    def __init__(self, pose_embed_dim, pose_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pose_dim)
        )

    def forward(self, pose_embed_seq: torch.Tensor) -> torch.Tensor:
        # pose_embed_seq: (B, T, D_pose_embed)
        return self.net(pose_embed_seq)

# 3. GAT增强的序列VAE主模型
class GraphSequenceVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pose_dim = cfg.pose_dim
        self.latent_dim = cfg.vae_latent_dim
        self.max_seq_len = cfg.vae_max_seq_len

        # -- Spatial Encoder --
        self.spatial_encoder = PoseGatEncoder(
            num_joints=self.pose_dim // 3,
            pose_embed_dim=cfg.vae_pose_embed_dim,
            gat_hidden_dims=cfg.vae_gat_hidden_dims,
            gat_heads=cfg.vae_gat_heads
        )

        # -- Temporal Encoder (GRU) --
        self.temporal_encoder_gru = nn.GRU(
            input_size=cfg.vae_pose_embed_dim,
            hidden_size=cfg.vae_hidden_dim,
            num_layers=cfg.vae_gru_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc_mu = nn.Linear(cfg.vae_hidden_dim * 2, self.latent_dim)
        self.fc_logvar = nn.Linear(cfg.vae_hidden_dim * 2, self.latent_dim)
        
        # -- Temporal Decoder (GRU) --
        self.decoder_latent_proj = nn.Linear(self.latent_dim, cfg.vae_hidden_dim)
        self.temporal_decoder_gru = nn.GRU(
            input_size=cfg.vae_pose_embed_dim,
            hidden_size=cfg.vae_hidden_dim,
            num_layers=cfg.vae_gru_layers,
            batch_first=True
        )
        self.decoder_output_proj = nn.Linear(cfg.vae_hidden_dim, cfg.vae_pose_embed_dim)
        self.sos_token = nn.Parameter(torch.randn(1, 1, cfg.vae_pose_embed_dim))

        # -- Spatial Decoder --
        self.spatial_decoder = SpatialMlpDecoder(
            pose_embed_dim=cfg.vae_pose_embed_dim,
            pose_dim=self.pose_dim
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Spatial Encoding
        pose_embeds = self.spatial_encoder(x) # (B, T, D_pose_embed)

        # 2. Temporal Aggregation
        lengths = mask.sum(dim=1).cpu()
        if not all(l > 0 for l in lengths): # 安全检查
             # 返回一个零张量或者处理这个空序列的情况
             mu = torch.zeros(x.size(0), self.latent_dim, device=x.device)
             logvar = torch.zeros(x.size(0), self.latent_dim, device=x.device)
             return mu, logvar

        packed_x = nn.utils.rnn.pack_padded_sequence(pose_embeds, lengths, batch_first=True, enforce_sorted=False)
        
        _, hidden = self.temporal_encoder_gru(packed_x)
        last_hidden = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=-1)
        
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        # 1. Temporal Generation of pose embeddings
        batch_size = z.size(0)
        h_0 = self.decoder_latent_proj(z)
        h_0 = h_0.unsqueeze(0).repeat(self.temporal_decoder_gru.num_layers, 1, 1)
        
        decoder_input = self.sos_token.expand(batch_size, -1, -1)
        
        pose_embed_outputs = []
        hidden = h_0
        for _ in range(target_seq_len):
            output, hidden = self.temporal_decoder_gru(decoder_input, hidden)
            output = self.decoder_output_proj(output)
            pose_embed_outputs.append(output)
            decoder_input = output

        pose_embed_seq = torch.cat(pose_embed_outputs, dim=1) # (B, T, D_pose_embed)

        # 2. Spatial Decoding to coordinates
        recon_x = self.spatial_decoder(pose_embed_seq)
        return recon_x

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        target_seq_len = x.size(1) 
        recon_x = self.decode(z, target_seq_len)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, mask, kl_beta: float = 1.0, velocity_weight: float = 0.0):
    # 1. 重建损失 (Recon Loss)
    recon_loss = (recon_x - x).pow(2).mean(dim=-1)
    recon_loss = (recon_loss * mask).sum() / mask.sum().clamp(min=1)
    
    # 2. KL 散度损失
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # 3. ✨ 新增：运动速度损失 (Velocity Loss)
    velocity_loss = torch.tensor(0.0, device=x.device) # 确保在不计算时也是一个tensor
    if velocity_weight > 0 and x.size(1) > 1:
        # 计算速度: (B, T-1, D)
        pred_velocity = recon_x[:, 1:] - recon_x[:, :-1]
        true_velocity = x[:, 1:] - x[:, :-1]
        
        # 速度损失只在连续两帧都有效时计算
        velocity_mask = mask[:, 1:] & mask[:, :-1]
        
        # 计算速度的MSE损失
        mse_v = (pred_velocity - true_velocity).pow(2).mean(-1) # (B, T-1)
        # 应用掩码并求平均
        velocity_loss = (mse_v * velocity_mask).sum() / velocity_mask.sum().clamp(min=1)
    
    # 4. 合并总损失
    total_loss = recon_loss + kl_beta * kld_loss + velocity_weight * velocity_loss
    
    # 5. ✨ 返回所有损失项，便于监控
    return total_loss, recon_loss, kld_loss, velocity_loss