import torch

class T2M_Config:
    def __init__(self):
        # --- General ---
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Data ---
        self.data_root = "./datasets"
        self.pose_dim = 150
        # Kinematic layout
        self.use_kinematic_decoder = True
        self.num_joints_2d = 50
        
        # --- VQ-VAE Architecture ---
        self.model_max_seq_len = 500
        self.motion_encoder_layers = 6
        self.motion_encoder_heads = 12
        self.motion_encoder_hidden_dim = 768
        self.motion_encoder_dropout = 0.1
        self.motion_decoder_layers = 6
        self.motion_decoder_heads = 12
        self.motion_decoder_hidden_dim = 768
        self.motion_decoder_dropout = 0.1
        self.embedding_dim = 384
        self.codebook_size = 2048
        self.downsample_rate = 4

        # --- VQ-VAE Quantizer ---
        self.commitment_cost = 0.25
        self.use_ema_update = True
        self.ema_decay = 0.99
        self.epsilon = 1e-5
        self.vq_lookup_token_chunk = 4096
        self.vq_lookup_code_chunk = 4096

        # --- VQ-VAE Training ---
        self.vqvae_checkpoint_path = f"./checkpoints/vqvae_model.pth"
        self.batch_size = 32
        self.val_batch_size = 16
        self.vqvae_num_epochs = 120
        self.vqvae_learning_rate = 3e-4
        
        self.recon_loss_weight = 1.0
        self.vq_loss_weight = 0.25
        self.vq_weight_warmup_epochs = 10

        # --- Sign-aware Reconstruction (新增) ---
        # 关键点组权重
        self.body_point_weight = 1.0
        self.hand_point_weight = 2.0
        # 是否使用置信度作为重建加权（基于反归一化后的置信度 0~1）
        self.use_confidence_weight = True
        self.conf_weight_gamma = 1.0  # 置信度幂次放大因子
        # 正则项权重
        self.bone_length_loss_weight = 0.10
        self.temporal_velocity_loss_weight = 0.05
        self.temporal_accel_loss_weight = 0.00

        # --- DataLoader & Dataset 性能参数（为避免Win内存抖动）---
        self.num_workers = 0            # Windows 建议 0，避免多进程开销
        self.pin_memory = False         # 关闭以减少内存搬运抖动
        self.persistent_workers = False # 仅当 num_workers>0 才生效
        self.prefetch_factor = 2        # 仅当 num_workers>0 才生效

        # 数据集样本缓存，减少频繁JSON解析与归一化（小缓存）
        self.dataset_cache_in_memory = True
        self.dataset_cache_max_items = 1024

        # --- GPT Architecture ---
        self.gpt_checkpoint_path = f"./checkpoints/t2m_gpt_model.pth"
        self.text_model_name = "bert-base-uncased"
        self.gpt_layers = 8
        self.gpt_heads = 12
        self.gpt_hidden_dim = 768
        self.gpt_dropout = 0.1
        self.vq_sequence_length = self.model_max_seq_len // self.downsample_rate

        # >>> ADDED THIS MISSING SECTION <<<
        # --- GPT Training ---
        self.gpt_num_epochs = 50
        self.gpt_learning_rate = 1e-4
        self.use_eos_token = True

    def get_device(self):
        return torch.device(self.device)