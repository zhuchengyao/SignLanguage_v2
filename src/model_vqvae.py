import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import math
from .config import T2M_Config
from .latent2d.skeleton2d import (
    FK2D,
    build_parents_for_50,
    global50_xy_to_pose150,
    get_body_connections,
    get_hand_connections,
    rot2d,
)

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
        # Prefer the consistent name `vq_lookup_token_chunk`; fall back to older `vq_lookup_chunk_tokens`
        self.lookup_chunk_tokens = getattr(cfg, "vq_lookup_token_chunk", getattr(cfg, "vq_lookup_chunk_tokens", 4096))

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
        self.cfg = cfg
        self.input_projection = nn.Linear(cfg.embedding_dim, cfg.motion_decoder_hidden_dim)
        self.temporal_upsample = nn.ConvTranspose1d(cfg.motion_decoder_hidden_dim, cfg.motion_decoder_hidden_dim, kernel_size=cfg.downsample_rate, stride=cfg.downsample_rate)
        self.pos_encoding = PositionalEncoding(cfg.motion_decoder_hidden_dim, cfg.model_max_seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.motion_decoder_hidden_dim, nhead=cfg.motion_decoder_heads,
            dim_feedforward=cfg.motion_decoder_hidden_dim * 4, dropout=cfg.motion_decoder_dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.motion_decoder_layers)
        # Output head: either direct pose or kinematic params
        if getattr(cfg, 'use_kinematic_decoder', False):
            J = getattr(cfg, 'num_joints_2d', 50)
            self.output_projection = nn.Linear(cfg.motion_decoder_hidden_dim, J + J + 2)
            # FK components prepared lazily on first forward
            self._fk_ready = False
            self.J = J
        else:
            self.output_projection = nn.Linear(cfg.motion_decoder_hidden_dim, cfg.pose_dim)
        self.query_embedding = nn.Parameter(torch.randn(1, cfg.model_max_seq_len, cfg.motion_decoder_hidden_dim))

    def _lazy_prepare_fk(self, device: torch.device, dtype: torch.dtype):
        if self._fk_ready:
            return
        parents, roots = build_parents_for_50()
        self.register_buffer('fk_parents', parents)
        self.fk_roots = roots
        # rest offsets: use unit bones; robust rest统计可另行估计，这里用单位长度保证可微分并避免外部依赖
        rest = torch.zeros(self.J, 2, device=device, dtype=dtype)
        # 简单设置：对每个非根关节给一个单位向量（x方向），避免零长度
        for j in range(self.J):
            if parents[j] >= 0:
                rest[j, 0] = 1.0
                rest[j, 1] = 0.0
        self.register_buffer('fk_rest', rest)
        self._fk_ready = True

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
        out = self.output_projection(decoded)
        if getattr(self, 'J', None) is None or not getattr(self.cfg, 'use_kinematic_decoder', False):
            return out
        # Interpret as [theta(J), scale(J), root(2)], run FK to get (T,J,2) then map to 150D pose
        device = out.device; dtype = out.dtype
        self._lazy_prepare_fk(device, dtype)
        J = self.J
        theta = out[..., :J]                  # (B,T,J)
        scales = out[..., J:2*J]              # (B,T,J)
        root_xy = out[..., 2*J:2*J+2]         # (B,T,2)
        poses = []
        parents = self.fk_parents
        rest = self.fk_rest
        root_index = self.fk_roots[0]
        for b in range(B):
            Tlen = target_length
            # Build topology (order from root)
            children = {int(i): [] for i in range(J)}
            for j in range(J):
                p = int(parents[j].item())
                if p >= 0:
                    children[p].append(j)
            from collections import deque
            order = []
            dq = deque([root_index])
            visited = {root_index}
            while dq:
                u = dq.popleft()
                for v in children[u]:
                    if v in visited:
                        continue
                    order.append(v)
                    visited.add(v)
                    dq.append(v)
            # Maps to avoid inplace writes on a single tensor
            gtheta_map = {root_index: torch.zeros(Tlen, device=device, dtype=dtype)}
            pos_map = {root_index: root_xy[b]}
            for j in order:
                p = int(parents[j].item())
                parent_theta = gtheta_map[p]
                cur_theta = parent_theta + theta[b, :, j]
                gtheta_map[j] = cur_theta
                R = rot2d(cur_theta)                                     # (T,2,2)
                off = rest[j].unsqueeze(0).expand(Tlen, 2)
                if scales is not None:
                    off = off * scales[b, :, j].unsqueeze(-1)
                step = torch.matmul(R, off.unsqueeze(-1)).squeeze(-1)    # (T,2)
                pos_map[j] = pos_map[p] + step
            pos_tensor = torch.stack([pos_map.get(j, torch.zeros(Tlen, 2, device=device, dtype=dtype)) for j in range(J)], dim=1)
            pose150 = global50_xy_to_pose150(pos_tensor)
            poses.append(pose150)
        poses = torch.stack(poses, dim=0)
        return poses

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

        # --- Sign-aware extras: weighted recon, bone-length, temporal smoothness ---
        def split_xy_conf(tensor_150: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # tensor_150: (...,150) -> body(8,2), right(21,2), left(21,2) and confidences (..., 8/21/21)
            body = tensor_150[..., :24].view(*tensor_150.shape[:-1], 8, 3)
            right = tensor_150[..., 24:87].view(*tensor_150.shape[:-1], 21, 3)
            left = tensor_150[..., 87:150].view(*tensor_150.shape[:-1], 21, 3)
            body_xy, body_c = body[..., :2], body[..., 2]
            right_xy, right_c = right[..., :2], right[..., 2]
            left_xy, left_c = left[..., :2], left[..., 2]
            return (body_xy, right_xy, left_xy), (body_c, right_c, left_c)

        # Denormalize to use confidences and true scale if stats are available
        x_denorm, rec_denorm = x, reconstructed
        if hasattr(self.cfg, 'mean') and hasattr(self.cfg, 'std') and self.cfg.mean is not None and self.cfg.std is not None:
            mean = self.cfg.mean.to(x.device)
            std = self.cfg.std.to(x.device)
            x_denorm = x * std + mean
            rec_denorm = reconstructed * std + mean

        # Weighted reconstruction on XY only
        (bx, rx, lx), (bc, rc, lc) = split_xy_conf(x_denorm)
        (br, rr, lr), _ = split_xy_conf(rec_denorm)
        # Build weights
        body_w = torch.full_like(bc, fill_value=getattr(self.cfg, 'body_point_weight', 1.0))
        hand_w = torch.full_like(rc, fill_value=getattr(self.cfg, 'hand_point_weight', 2.0))
        # confidence weighting
        if getattr(self.cfg, 'use_confidence_weight', True):
            gamma = getattr(self.cfg, 'conf_weight_gamma', 1.0)
            body_w = body_w * bc.clamp(min=0.0, max=1.0).pow(gamma)
            hand_w_r = hand_w * rc.clamp(min=0.0, max=1.0).pow(gamma)
            hand_w_l = hand_w * lc.clamp(min=0.0, max=1.0).pow(gamma)
        else:
            hand_w_r = hand_w
            hand_w_l = hand_w
        # Valid mask tiled to XY
        vmask_t = valid_positions.bool()
        # Compose weighted MSE on XY
        def mse_weighted(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, vm: torch.Tensor) -> torch.Tensor:
            # a,b: (B,T,K,2); w: (B,T,K); vm: (B,T) or (B,T,1)
            diff2 = (a - b).pow(2).sum(dim=-1)  # (B,T,K)
            vm2 = vm if vm.dim() == 3 else vm.unsqueeze(-1)     # (B,T,1)
            w2 = w * vm2.to(w.dtype)                            # (B,T,K)
            loss = (diff2 * w2).sum()
            denom = w2.sum().clamp(min=1.0)
            return loss / denom
        recon_weighted = mse_weighted(br, bx, body_w, vmask_t) \
            + mse_weighted(rr, rx, hand_w_r, vmask_t) \
            + mse_weighted(lr, lx, hand_w_l, vmask_t)

        # Bone-length consistency on XY
        def bone_length_loss(xy_pred: torch.Tensor, xy_gt: torch.Tensor, connections: Any, vm: torch.Tensor) -> torch.Tensor:
            # xy_*: (B,T,K,2); vm: (B,T,1)
            bl = 0.0
            denom = 0.0
            for a, b in connections:
                pa = xy_pred[..., a, :]
                pb = xy_pred[..., b, :]
                ga = xy_gt[..., a, :]
                gb = xy_gt[..., b, :]
                len_p = (pa - pb).norm(dim=-1)
                len_g = (ga - gb).norm(dim=-1).clamp(min=1e-6)
                bl += ((len_p - len_g).abs() * vm.squeeze(-1)).sum()
                denom += vm.sum().item()
            denom = max(denom, 1.0)
            return bl / denom
        bone_loss = bone_length_loss(br, bx, get_body_connections(), vmask_t)
        bone_loss += bone_length_loss(rr, rx, get_hand_connections(), vmask_t)
        bone_loss += bone_length_loss(lr, lx, get_hand_connections(), vmask_t)

        # Temporal smoothness (velocity/acceleration) on prediction
        def temporal_smooth_loss(xy_seq: torch.Tensor, vm: torch.Tensor, order: int = 1) -> torch.Tensor:
            # xy_seq: (B,T,K,2)
            if xy_seq.size(1) < (order + 1):
                return torch.tensor(0.0, device=xy_seq.device, dtype=xy_seq.dtype)
            if order == 1:
                d = xy_seq[:, 1:] - xy_seq[:, :-1]
                vmask = vm[:, 1:] * vm[:, :-1]                 # (B,T-1,1) or (B,T-1)
            else:
                d1 = xy_seq[:, 1:] - xy_seq[:, :-1]
                d = d1[:, 1:] - d1[:, :-1]
                vmask = vm[:, 2:] * vm[:, 1:-1] * vm[:, :-2]   # (B,T-2,1) or (B,T-2)
            loss = d.abs().sum(dim=-1)  # (B,T',K)
            vmask_k = vmask if vmask.dim() == 3 else vmask.unsqueeze(-1)  # (B,T',1)
            loss = (loss * vmask_k.to(loss.dtype)).sum()
            denom = vmask_k.sum().clamp(min=1.0)
            return loss / denom
        vel_loss = temporal_smooth_loss(torch.cat([br, rr, lr], dim=2), vmask_t, order=1)
        acc_loss = temporal_smooth_loss(torch.cat([br, rr, lr], dim=2), vmask_t, order=2)

        return {
            'reconstructed': reconstructed,
            'indices': indices,
            'recon_loss': recon_loss,
            'recon_weighted': recon_weighted.detach(),
            'bone_length_loss': bone_loss,
            'velocity_loss': vel_loss,
            'accel_loss': acc_loss,
            **vq_losses,
            'z_e': z_e,
        }