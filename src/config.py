import torch

class T2M_Config:
    def __init__(self):
        # --- General ---
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Data ---
        self.data_root = "./datasets"
        self.pose_dim = 150
        
        # --- VQ-VAE Architecture ---
        self.model_max_seq_len = 500
        self.motion_encoder_layers = 4
        self.motion_encoder_heads = 8
        self.motion_encoder_hidden_dim = 512
        self.motion_encoder_dropout = 0.1
        self.motion_decoder_layers = 4
        self.motion_decoder_heads = 8
        self.motion_decoder_hidden_dim = 512
        self.motion_decoder_dropout = 0.1
        self.embedding_dim = 256
        self.codebook_size = 1024
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
        self.vqvae_num_epochs = 205 # You can set this back to 200 if you wish
        self.vqvae_learning_rate = 3e-4
        
        self.recon_loss_weight = 1.0
        self.vq_loss_weight = 0.25
        self.vq_weight_warmup_epochs = 10

        # --- GPT Architecture ---
        self.gpt_checkpoint_path = f"./checkpoints/t2m_gpt_model.pth"
        self.text_model_name = "bert-base-uncased"
        self.gpt_layers = 6
        self.gpt_heads = 8
        self.gpt_hidden_dim = 512
        self.gpt_dropout = 0.1
        self.vq_sequence_length = self.model_max_seq_len // self.downsample_rate

        # >>> ADDED THIS MISSING SECTION <<<
        # --- GPT Training ---
        self.gpt_num_epochs = 100
        self.gpt_learning_rate = 1e-4

    def get_device(self):
        return torch.device(self.device)