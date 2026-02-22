import torch


class T2M_Config:
    def __init__(self):
        # --- General ---
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Data ---
        self.data_root = "./datasets"
        self.dataset_name = "ASL_gloss"  # subdirectory under data_root (e.g. ASL_gloss, ASL_sentence)
        self.pose_dim = 150

        # --- Kinematic layout ---
        self.use_kinematic_decoder = False
        self.num_joints_2d = 50
        self.enable_hand_residual = False
        self.hand_residual_scale = 0.2
        self.hand_residual_max_scale = 0.2
        self.kinematic_scale_center = 1.0
        self.kinematic_scale_range = 0.25
        self.kinematic_scale_min = 0.6
        self.kinematic_scale_max = 1.4

        # --- VQ-VAE Architecture ---
        self.model_max_seq_len = 1500
        self.motion_encoder_layers = 4
        self.motion_encoder_heads = 4
        self.motion_encoder_hidden_dim = 256
        self.motion_encoder_dropout = 0.1
        self.motion_decoder_layers = 4
        self.motion_decoder_heads = 4
        self.motion_decoder_hidden_dim = 256
        self.motion_decoder_dropout = 0.1
        self.embedding_dim = 256
        self.codebook_size = 512
        self.downsample_rate = 4

        # --- Hierarchical VQ (coarse-to-fine) ---
        # Default off to avoid GPT/inference mismatch unless coarse tokens are explicitly modeled.
        self.use_hierarchical_codebook = False
        self.coarse_downsample_factor = 4
        self.coarse_codebook_size = 1024
        self.coarse_embedding_dim = self.embedding_dim
        self.coarse_commitment_cost = 0.25
        self.coarse_condition_weight = 0.5
        self.coarse_vq_loss_weight = 0.5
        self.coarse_decoder_residual_scale = 0.5
        self.coarse_lookup_token_chunk = 2048

        # --- VQ-VAE Quantizer ---
        self.commitment_cost = 1.0
        self.use_ema_update = True
        self.ema_decay = 0.95
        self.epsilon = 1e-5
        self.vq_lookup_token_chunk = 4096
        self.vq_lookup_code_chunk = 4096

        # --- VQ-VAE Training ---
        self.vqvae_checkpoint_path = "./checkpoints/vqvae_model.pth"
        self.batch_size = 32
        self.val_batch_size = 16
        self.vqvae_num_epochs = 200
        self.vqvae_learning_rate = 3e-4
        self.recon_loss_weight = 1.0
        self.vq_loss_weight = 1.0
        self.vq_weight_warmup_epochs = 5

        # --- Sign-aware reconstruction and regularization ---
        self.body_point_weight = 1.0
        self.hand_point_weight = 2.0
        self.use_confidence_weight = True
        self.conf_weight_gamma = 1.0
        self.weighted_recon_loss_weight = 0.5
        self.bone_length_loss_weight = 0.5
        self.temporal_velocity_loss_weight = 0.1
        self.temporal_accel_loss_weight = 0.02

        # --- DataLoader ---
        self.num_workers = 0
        self.pin_memory = False
        self.persistent_workers = False
        self.prefetch_factor = 2

        # --- Dataset cache ---
        self.dataset_cache_in_memory = True
        self.dataset_cache_max_items = 1024

        # --- Motion VAE Architecture ---
        self.vae_latent_dim = 128
        self.vae_checkpoint_path = "./checkpoints/vae_model.pth"
        self.vae_num_epochs = 200
        self.vae_learning_rate = 3e-4
        self.kl_weight = 0.001
        self.kl_warmup_epochs = 20

        # --- Latent Flow Matching ---
        self.flow_checkpoint_path = "./checkpoints/flow_model.pth"
        self.flow_hidden_dim = 512
        self.flow_layers = 12
        self.flow_attn_heads = 8
        self.flow_attn_every_n = 2   # temporal self-attention every N MLP blocks
        self.flow_cond_dim = 768
        self.flow_num_epochs = 300
        self.flow_learning_rate = 2e-4
        self.flow_sample_steps = 50
        self.flow_cfg_guidance = 3.0
        self.flow_cond_drop_prob = 0.1
        self.flow_text_model = "distilbert-base-uncased"

        # --- Dual-branch VAE (body/hand decoupled) ---
        self.dual_body_dim = 24     # 8 body joints × 3
        self.dual_hand_dim = 126    # (21+21) hand joints × 3
        self.dual_body_latent_dim = 64
        self.dual_hand_latent_dim = 96
        self.dual_body_hidden_dim = 192
        self.dual_hand_hidden_dim = 256
        self.dual_enc_layers = 3
        self.dual_dec_layers = 3
        self.dual_heads = 4
        self.dual_dropout = 0.1
        self.dual_body_recon_weight = 1.0
        self.dual_hand_recon_weight = 2.0   # hands need higher fidelity
        self.dual_sync_weight = 0.5         # wrist sync between body & hand
        self.dual_body_vel_weight = 0.2     # stronger body smoothness
        self.dual_body_acc_weight = 0.05
        self.dual_hand_vel_weight = 0.1
        self.dual_hand_acc_weight = 0.02
        self.dual_kl_weight = 0.001
        self.dual_kl_warmup_epochs = 20
        self.dual_vae_checkpoint_path = "./checkpoints/dual_vae_model.pth"
        self.dual_vae_num_epochs = 200
        self.dual_vae_learning_rate = 3e-4

        # --- Dual-branch Flow ---
        self.dual_flow_body_checkpoint = "./checkpoints/dual_flow_body.pth"
        self.dual_flow_hand_checkpoint = "./checkpoints/dual_flow_hand.pth"
        self.dual_flow_hidden = 384
        self.dual_flow_layers = 8
        self.dual_flow_attn_heads = 8
        self.dual_flow_attn_every_n = 2
        self.dual_flow_num_epochs = 300
        self.dual_flow_learning_rate = 2e-4

        # --- GPT Architecture ---
        self.gpt_checkpoint_path = "./checkpoints/t2m_gpt_model.pth"
        self.text_model_name = "bert-base-uncased"
        self.gpt_layers = 8
        self.gpt_heads = 12
        self.gpt_hidden_dim = 768
        self.gpt_dropout = 0.1

        # --- GPT Training ---
        self.gpt_num_epochs = 50
        self.gpt_learning_rate = 1e-4
        self.use_eos_token = True

        # Runtime-only normalization stats, populated during training.
        self.mean = None
        self.std = None

        self.refresh_dependent_values()

    def refresh_dependent_values(self):
        self.vq_sequence_length = max(1, self.model_max_seq_len // self.downsample_rate)
        self.coarse_embedding_dim = self.embedding_dim
        if not self.use_kinematic_decoder:
            self.enable_hand_residual = False

    def apply_overrides(self, overrides: dict):
        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(self, key):
                setattr(self, key, value)
        self.refresh_dependent_values()

    def get_device(self):
        return torch.device(self.device)
