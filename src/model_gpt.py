import torch
import torch.nn as nn
from transformers import BertModel

from .config import T2M_Config
from .model_vqvae import PositionalEncoding

class T2M_GPT(nn.Module):
    def __init__(self, cfg: T2M_Config):
        super().__init__()
        self.cfg = cfg
        
        # 1. Text Encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(cfg.text_model_name)
        # Freeze BERT parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # >>> ADDED: A projection layer to match dimensions <<<
        # BERT outputs 768, but our GPT model works with 512
        bert_hidden_size = self.text_encoder.config.hidden_size # This is 768
        self.text_feature_projection = nn.Linear(bert_hidden_size, cfg.gpt_hidden_dim)

        # 2. Motion Token Embedding (+SOS/+EOS)
        vocab_size = cfg.codebook_size + 1  # +1 for SOS at least
        if getattr(cfg, 'use_eos_token', True):
            vocab_size += 1  # EOS
        self.sos_id = cfg.codebook_size
        self.eos_id = cfg.codebook_size + 1 if getattr(cfg, 'use_eos_token', True) else None
        self.motion_embedding = nn.Embedding(vocab_size, cfg.gpt_hidden_dim)
        
        # 3. Positional Encoding
        self.pos_encoding = PositionalEncoding(cfg.gpt_hidden_dim, cfg.model_max_seq_len)
        
        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.gpt_hidden_dim, nhead=cfg.gpt_heads,
            dim_feedforward=cfg.gpt_hidden_dim * 4, dropout=cfg.gpt_dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.gpt_layers)
        
        # 5. Output Head
        self.output_head = nn.Linear(cfg.gpt_hidden_dim, self.motion_embedding.num_embeddings)
        
    def forward(self, tokenized_text, motion_tokens: torch.Tensor, motion_mask: torch.Tensor = None):
        # 1. Get text features from BERT (Dimension: 768)
        text_encoder_output = self.text_encoder(**tokenized_text)
        text_features = text_encoder_output.last_hidden_state
        
        # >>> ADDED: Project text features to match GPT's dimension <<<
        # Project from 768 -> 512
        text_features = self.text_feature_projection(text_features)
        
        # 2. Embed motion tokens and add positional encoding
        motion_embed = self.motion_embedding(motion_tokens)
        motion_embed = self.pos_encoding(motion_embed)
        
        # 3. Create masks
        causal_mask = self.generate_causal_mask(motion_tokens.size(1)).to(motion_tokens.device)
        text_padding_mask = (tokenized_text['attention_mask'] == 0)
        
        # 4. Pass through the Transformer Decoder
        output = self.transformer_decoder(
            tgt=motion_embed,
            memory=text_features, # Now has the correct dimension (512)
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~motion_mask if motion_mask is not None else None,
            memory_key_padding_mask=text_padding_mask
        )
        
        # 5. Project to vocabulary size
        logits = self.output_head(output)
        return logits

    @staticmethod
    def generate_causal_mask(size: int):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask