import torch


class HLC_NAR_Config:
    def __init__(self):
        # --- General ---
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Data ---
        self.data_root = "./datasets"
        self.pose_dim = 150
        self.num_joints = 50
        self.body_joints = 8
        self.hand_joints = 42  # 21 left + 21 right
        self.body_dim = 16     # 8 * 2
        self.hand_dim = 84     # 42 * 2
        self.coord_dim = 2     # x, y
        self.max_seq_len = 200

        # --- Text Encoder ---
        self.text_model_name = "bert-base-uncased"
        self.text_hidden_dim = 768
        self.freeze_text_encoder = True

        # --- HLC Encoder Architecture ---
        self.hlc_d_model = 256
        self.hlc_nhead = 8
        self.hlc_num_layers = 4
        self.hlc_dim_ff = 512
        self.hlc_dropout = 0.1

        # --- HLC Codebook ---
        self.global_codebook_size = 512
        self.local_codebook_size = 1024
        self.vq_embedding_dim = 256
        self.vq_commitment_cost = 0.25
        self.vq_ema_decay = 0.99
        self.vq_epsilon = 1e-5

        # --- NAR Decoder ---
        self.nar_d_model = 512
        self.nar_nhead = 8
        self.nar_num_layers = 6
        self.nar_dim_ff = 1024
        self.nar_dropout = 0.1
        self.nar_input_dim = 1024  # 4 codebooks * 256

        # --- Token Predictor ---
        self.tok_pred_layers = 4
        self.tok_pred_nhead = 8
        self.tok_pred_dim_ff = 1024
        self.tok_pred_dropout = 0.1

        # --- Length Predictor ---
        self.len_pred_hidden = 256
        self.max_target_len = 200

        # --- RTP (Rhythm-Tempered Phase) ---
        self.rtp_hidden_dim = 256
        self.rtp_nhead = 4
        self.rtp_num_layers = 2
        self.rtp_alpha = 0.5

        # --- KALS Loss Weights ---
        self.kals_bone_weight = 0.10
        self.kals_angle_weight = 0.05
        self.kals_symmetry_weight = 0.03

        # --- Reconstruction Loss ---
        self.recon_loss_weight = 1.0
        self.body_recon_weight = 1.0
        self.hand_recon_weight = 2.0

        # --- VQ Loss ---
        self.vq_loss_weight = 0.25
        self.vq_warmup_epochs = 5

        # --- Token Prediction Loss ---
        self.token_pred_loss_weight = 1.0

        # --- RTP Loss ---
        self.rtp_loss_weight = 0.1

        # --- Length Prediction Loss ---
        self.length_pred_loss_weight = 0.01

        # --- Stage 1: Reconstruction Warm-up ---
        self.stage1_epochs = 80
        self.stage1_lr = 3e-4
        self.stage1_batch_size = 32
        self.stage1_val_batch_size = 16

        # --- Stage 2: RTP Alignment ---
        self.stage2_epochs = 30
        self.stage2_lr = 1e-4
        self.stage2_batch_size = 32
        self.stage2_val_batch_size = 16

        # --- Stage 3: Joint Fine-tuning ---
        self.stage3_epochs = 40
        self.stage3_lr = 5e-5
        self.stage3_batch_size = 24
        self.stage3_val_batch_size = 16

        # --- DataLoader ---
        self.num_workers = 0
        self.pin_memory = False

        # --- Checkpoints ---
        self.checkpoint_dir = "./checkpoints/hlc_nar"
        self.stage1_ckpt = "./checkpoints/hlc_nar/stage1_best.pth"
        self.stage2_ckpt = "./checkpoints/hlc_nar/stage2_best.pth"
        self.stage3_ckpt = "./checkpoints/hlc_nar/stage3_best.pth"

    def get_device(self):
        return torch.device(self.device)
