import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import math
from copy import deepcopy
from .config import T2M_Config
from .latent2d.skeleton2d import (
    build_parents_for_50,
    compute_rest_offsets_xy,
    global50_xy_to_pose150,
    get_body_connections,
    get_hand_connections,
    pose150_to_global50_xy,
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
                with torch.no_grad():
                    flat_input_detached = flat_input.detach()
                    encodings = F.one_hot(encoding_indices, self.codebook_size).type(flat_input_detached.dtype)
                    self.ema_cluster_size.mul_(self.ema_decay).add_(encodings.sum(0), alpha=1 - self.ema_decay)
                    dw = encodings.t() @ flat_input_detached
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
        self.use_hierarchy = getattr(cfg, 'use_hierarchical_codebook', False)
        self.coarse_factor = getattr(cfg, 'coarse_downsample_factor', 1)
        self.input_projection = nn.Linear(cfg.embedding_dim, cfg.motion_decoder_hidden_dim)
        self.temporal_upsample = nn.ConvTranspose1d(cfg.motion_decoder_hidden_dim, cfg.motion_decoder_hidden_dim, kernel_size=cfg.downsample_rate, stride=cfg.downsample_rate)
        self.enable_hand_residual = bool(
            getattr(cfg, 'enable_hand_residual', False) and getattr(cfg, 'use_kinematic_decoder', False)
        )
        if self.enable_hand_residual:
            self.hand_residual_scale = float(getattr(cfg, 'hand_residual_scale', 1.0))
            self.hand_residual_scale = min(
                self.hand_residual_scale,
                float(getattr(cfg, "hand_residual_max_scale", 0.2)),
            )
            self.hand_residual_head = nn.Linear(cfg.motion_decoder_hidden_dim, cfg.pose_dim)
            hand_mask = torch.zeros(cfg.pose_dim)
            for base in (24, 87):  # right hand, left hand
                for j in range(21):
                    hand_mask[base + 3 * j] = 1.0      # x
                    hand_mask[base + 3 * j + 1] = 1.0  # y
            self.register_buffer('hand_residual_mask', hand_mask.view(1, 1, -1))
        if self.use_hierarchy:
            self.coarse_input_projection = nn.Linear(cfg.coarse_embedding_dim, cfg.motion_decoder_hidden_dim)
            total_stride = cfg.downsample_rate * max(1, self.coarse_factor)
            self.coarse_temporal_upsample = nn.ConvTranspose1d(
                cfg.motion_decoder_hidden_dim,
                cfg.motion_decoder_hidden_dim,
                kernel_size=total_stride,
                stride=total_stride
            )
            self.coarse_residual_scale = nn.Parameter(torch.tensor(getattr(cfg, 'coarse_decoder_residual_scale', 1.0), dtype=torch.float32))
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
            self.kinematic_scale_center = float(getattr(cfg, "kinematic_scale_center", 1.0))
            self.kinematic_scale_range = float(getattr(cfg, "kinematic_scale_range", 0.25))
            self.kinematic_scale_min = float(getattr(cfg, "kinematic_scale_min", 0.6))
            self.kinematic_scale_max = float(getattr(cfg, "kinematic_scale_max", 1.4))
            # FK components prepared lazily on first forward
            self._fk_ready = False
            self.J = J
        else:
            self.output_projection = nn.Linear(cfg.motion_decoder_hidden_dim, cfg.pose_dim)
        self.query_embedding = nn.Parameter(torch.randn(1, cfg.model_max_seq_len, cfg.motion_decoder_hidden_dim))

    def _build_default_rest_offsets(self, parents: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rest = torch.zeros(self.J, 2, device=device, dtype=dtype)

        # Body canonical offsets (keeps torso/arm scale larger than fingers).
        body_dirs = {
            0: (0.0, -1.0),  # chest -> pelvis
            2: (1.0, 0.0),   # chest -> shoulder
            5: (-1.0, 0.0),  # chest -> shoulder
            3: (1.0, 0.0),   # shoulder -> elbow
            6: (-1.0, 0.0),  # shoulder -> elbow
            4: (1.0, 0.0),   # elbow -> wrist
            7: (-1.0, 0.0),  # elbow -> wrist
        }
        body_len = {0: 0.70, 2: 0.45, 5: 0.45, 3: 0.55, 6: 0.55, 4: 0.48, 7: 0.48}
        for j, (dx, dy) in body_dirs.items():
            rest[j, 0] = dx * body_len[j]
            rest[j, 1] = dy * body_len[j]

        # Hand roots coincide with body wrists.
        rest[8] = 0.0
        rest[29] = 0.0

        finger_chains = [
            [1, 2, 3, 4],      # thumb
            [5, 6, 7, 8],      # index
            [9, 10, 11, 12],   # middle
            [13, 14, 15, 16],  # ring
            [17, 18, 19, 20],  # pinky
        ]
        base_angles = [-0.70, -0.30, 0.00, 0.25, 0.50]
        seg_lengths = [0.22, 0.17, 0.13, 0.10]

        def fill_one_hand(root: int, mirrored: bool):
            sign = -1.0 if mirrored else 1.0
            for ci, chain in enumerate(finger_chains):
                angle = base_angles[ci] * sign
                dx = math.cos(angle)
                dy = math.sin(angle)
                for si, local_idx in enumerate(chain):
                    gidx = root + local_idx
                    seg_len = seg_lengths[min(si, len(seg_lengths) - 1)]
                    rest[gidx, 0] = dx * seg_len
                    rest[gidx, 1] = dy * seg_len

        fill_one_hand(8, mirrored=True)
        fill_one_hand(29, mirrored=False)

        # Keep roots at zero.
        for j in range(self.J):
            if int(parents[j].item()) < 0:
                rest[j] = 0.0
        return rest

    def _lazy_prepare_fk(self, device: torch.device, dtype: torch.dtype):
        if self._fk_ready:
            return
        parents, roots = build_parents_for_50()
        self.register_buffer('fk_parents', parents)
        self.fk_roots = roots

        # Start from canonical prior; if cfg.mean exists, replace with data-driven rest and keep ratio.
        rest = self._build_default_rest_offsets(parents, device=device, dtype=dtype)
        cfg_mean = getattr(self.cfg, "mean", None)
        if isinstance(cfg_mean, torch.Tensor) and cfg_mean.numel() == 150:
            try:
                mean_pose = cfg_mean.to(device=device, dtype=dtype).view(1, 150)
                mean_xy = pose150_to_global50_xy(mean_pose)[0]  # (50,2)
                rest_from_data = compute_rest_offsets_xy(mean_xy, parents).to(device=device, dtype=dtype)

                # Normalize by torso/shoulder scale so FK operates in a stable canonical space.
                torso = (mean_xy[1] - mean_xy[0]).norm()
                shoulder = (mean_xy[2] - mean_xy[5]).norm()
                base = torch.max(torch.stack([torso, shoulder, mean_xy.new_tensor(1e-4)]))
                rest_from_data = rest_from_data / base

                # Wrist hand roots should not introduce extra offsets.
                rest_from_data[8] = 0.0
                rest_from_data[29] = 0.0

                for j in range(self.J):
                    if int(parents[j].item()) < 0:
                        rest_from_data[j] = 0.0
                        continue
                    if j in (8, 29):
                        continue
                    cand = rest_from_data[j]
                    if torch.isfinite(cand).all() and cand.norm().item() > 1e-5:
                        rest[j] = cand
            except Exception:
                pass

        self.register_buffer('fk_rest', rest)
        self._fk_ready = True

    def forward(self, memory: torch.Tensor, target_length: int, coarse_memory: torch.Tensor = None) -> torch.Tensor:
        B = memory.size(0)
        memory = self.input_projection(memory)
        memory = memory.transpose(1, 2)
        memory = self.temporal_upsample(memory)
        memory = memory.transpose(1, 2)
        if memory.size(1) < target_length:
            pad = memory.new_zeros(B, target_length - memory.size(1), memory.size(2))
            memory = torch.cat([memory, pad], dim=1)
        memory = memory[:, :target_length, :]
        if self.use_hierarchy and coarse_memory is not None:
            coarse = self.coarse_input_projection(coarse_memory)
            coarse = coarse.transpose(1, 2)
            coarse = self.coarse_temporal_upsample(coarse)
            coarse = coarse.transpose(1, 2)
            if coarse.size(1) < target_length:
                pad = coarse.new_zeros(B, target_length - coarse.size(1), coarse.size(2))
                coarse = torch.cat([coarse, pad], dim=1)
            coarse = coarse[:, :target_length, :]
            memory = memory + self.coarse_residual_scale * coarse
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
        raw_scales = out[..., J:2*J]          # (B,T,J)
        scale_range = max(0.0, float(self.kinematic_scale_range))
        scales = float(self.kinematic_scale_center) + scale_range * torch.tanh(raw_scales)
        if self.kinematic_scale_min < self.kinematic_scale_max:
            scales = scales.clamp(min=float(self.kinematic_scale_min), max=float(self.kinematic_scale_max))
        else:
            scales = scales.clamp(min=1e-3)
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
        if self.enable_hand_residual:
            residual = self.hand_residual_head(decoded)
            mask = self.hand_residual_mask.to(residual.device)
            residual = residual * mask
            poses = poses + self.hand_residual_scale * residual
        return poses

class VQ_VAE(nn.Module):
    def __init__(self, cfg: T2M_Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = MotionEncoder(cfg)
        self.quantizer_fine = VectorQuantizer(cfg)
        self.quantizer = self.quantizer_fine  # backward compatibility
        self.decoder = MotionDecoder(cfg)
        self.use_hierarchy = getattr(cfg, 'use_hierarchical_codebook', False)
        self.coarse_factor = max(1, getattr(cfg, 'coarse_downsample_factor', 1))
        self.coarse_condition_weight = getattr(cfg, 'coarse_condition_weight', 1.0)
        if self.use_hierarchy:
            coarse_cfg = deepcopy(cfg)
            coarse_cfg.embedding_dim = getattr(cfg, 'coarse_embedding_dim', cfg.embedding_dim)
            coarse_cfg.codebook_size = getattr(cfg, 'coarse_codebook_size', cfg.codebook_size)
            coarse_cfg.commitment_cost = getattr(cfg, 'coarse_commitment_cost', cfg.commitment_cost)
            coarse_cfg.vq_lookup_token_chunk = getattr(cfg, 'coarse_lookup_token_chunk', cfg.vq_lookup_token_chunk)
            self.quantizer_coarse = VectorQuantizer(coarse_cfg)
            coarse_dim = coarse_cfg.embedding_dim
            self.coarse_projection = nn.Sequential(
                nn.LayerNorm(cfg.embedding_dim),
                nn.Linear(cfg.embedding_dim, coarse_dim)
            )
            self.coarse_condition_adapter = nn.Sequential(
                nn.LayerNorm(coarse_dim),
                nn.Linear(coarse_dim, cfg.embedding_dim)
            )
        else:
            self.quantizer_coarse = None

    def _pool_to_coarse(self, fine_latent: torch.Tensor) -> torch.Tensor:
        if not self.use_hierarchy:
            raise RuntimeError("Hierarchical pooling requested but hierarchy disabled.")
        B, T, D = fine_latent.shape
        factor = self.coarse_factor
        pad_len = (factor - (T % factor)) % factor
        if pad_len > 0:
            pad = fine_latent.new_zeros(B, pad_len, D)
            fine_latent = torch.cat([fine_latent, pad], dim=1)
        new_len = fine_latent.size(1) // factor
        coarse = fine_latent.view(B, new_len, factor, D).mean(dim=2)
        return coarse

    def _repeat_to_length(self, coarse_latent: torch.Tensor, repeat_factor: int, target_len: int) -> torch.Tensor:
        if coarse_latent is None:
            return None
        B, T, D = coarse_latent.shape
        expanded = coarse_latent.unsqueeze(2).repeat(1, 1, repeat_factor, 1).view(B, -1, D)
        if expanded.size(1) < target_len:
            pad = expanded.new_zeros(B, target_len - expanded.size(1), D)
            expanded = torch.cat([expanded, pad], dim=1)
        return expanded[:, :target_len, :]

    def encode(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        z_e = self.encoder(x, mask)
        coarse_outputs = {}
        coarse_memory = None
        if self.use_hierarchy:
            z_coarse = self._pool_to_coarse(z_e)
            z_coarse = self.coarse_projection(z_coarse)
            zq_coarse, idx_coarse, coarse_vq = self.quantizer_coarse(z_coarse)
            coarse_cond = self.coarse_condition_adapter(zq_coarse)
            coarse_cond_expanded = self._repeat_to_length(coarse_cond, self.coarse_factor, z_e.size(1))
            z_e = z_e + self.coarse_condition_weight * coarse_cond_expanded
            coarse_outputs = {
                'coarse_indices': idx_coarse,
                'coarse_z_q': zq_coarse,
                'coarse_vq_loss': coarse_vq['vq_loss'],
                'coarse_commitment_loss': coarse_vq['commitment_loss'],
                'coarse_codebook_loss': coarse_vq['codebook_loss'],
            }
            coarse_memory = zq_coarse
        z_q, indices, vq_losses = self.quantizer_fine(z_e)
        if coarse_outputs:
            vq_losses = {**vq_losses, **{k: v for k, v in coarse_outputs.items() if k != 'coarse_z_q'}}
        return z_q, indices, vq_losses

    def decode(self, indices: torch.Tensor, target_length: int, coarse_indices: torch.Tensor = None) -> torch.Tensor:
        z_q = self.quantizer_fine.embedding(indices)
        coarse_memory = None
        if self.use_hierarchy and coarse_indices is not None and self.quantizer_coarse is not None:
            coarse_memory = self.quantizer_coarse.embedding(coarse_indices)
        return self.decoder(z_q, target_length, coarse_memory=coarse_memory)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
        B, T, D = x.shape
        z_e = self.encoder(x, mask)
        coarse_indices = None
        coarse_memory = None
        coarse_losses = {}
        if self.use_hierarchy:
            z_coarse = self._pool_to_coarse(z_e)
            z_coarse = self.coarse_projection(z_coarse)
            zq_coarse, coarse_indices, coarse_vq = self.quantizer_coarse(z_coarse)
            coarse_cond = self.coarse_condition_adapter(zq_coarse)
            coarse_cond = self._repeat_to_length(coarse_cond, self.coarse_factor, z_e.size(1))
            z_e = z_e + self.coarse_condition_weight * coarse_cond
            coarse_memory = zq_coarse
            coarse_losses = {
                'coarse_indices': coarse_indices,
                'coarse_vq_loss': coarse_vq['vq_loss'],
                'coarse_commitment_loss': coarse_vq['commitment_loss'],
                'coarse_codebook_loss': coarse_vq['codebook_loss'],
            }
        z_q, indices, vq_losses = self.quantizer_fine(z_e)
        
        target_length = int(mask.sum(dim=1).max().item())
        reconstructed_short = self.decoder(z_q, target_length, coarse_memory=coarse_memory)
        
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
            'coarse_indices': coarse_indices,
            'recon_loss': recon_loss,
            'recon_weighted': recon_weighted,
            'bone_length_loss': bone_loss,
            'velocity_loss': vel_loss,
            'accel_loss': acc_loss,
            **vq_losses,
            **coarse_losses,
            'z_e': z_e,
        }
