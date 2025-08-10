import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import datetime
import wandb
from transformers import BertTokenizer

from src.config import T2M_Config
from src.model_vqvae import VQ_VAE
from src.model_gpt import T2M_GPT
from src.dataloader import ASLPoseDataset, collate_pose_batch, BucketSampler

class GPTTrainer:
    def __init__(self, cfg: T2M_Config, use_wandb: bool):
        self.cfg = cfg
        self.device = cfg.get_device()
        self.use_wandb = use_wandb

        # 1. Load the pre-trained VQ-VAE
        print(f"📥 Loading pre-trained VQ-VAE from: {cfg.vqvae_checkpoint_path}")
        vqvae_checkpoint = torch.load(cfg.vqvae_checkpoint_path, map_location="cpu")
        self.vq_vae = VQ_VAE(vqvae_checkpoint['cfg']).to(self.device)
        self.vq_vae.load_state_dict(vqvae_checkpoint['model_state_dict'])
        self.vq_vae.eval()
        for param in self.vq_vae.parameters():
            param.requires_grad = False
        print("✅ VQ-VAE loaded and frozen.")
        
        # 2. Initialize the T2M-GPT model and tokenizer
        self.model = T2M_GPT(cfg).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)
        
        # 3. Optimizer and Scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.gpt_learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.gpt_num_epochs)
        
        # 4. Loss Function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.global_step = 0
        self.best_loss = float('inf')

        print(f"🤖 T2M-GPT Model Parameters (Trainable): {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def create_dataloaders(self):
        train_dataset = ASLPoseDataset(data_paths=[os.path.join(self.cfg.data_root, "ASL_gloss/train")], split="train")
        val_dataset = ASLPoseDataset(data_paths=[os.path.join(self.cfg.data_root, "ASL_gloss/dev")], split="dev", extern_mean=train_dataset.pose_mean, extern_std=train_dataset.pose_std)
        sampler_cache_path = os.path.join(self.cfg.data_root, "ASL_gloss/.cache", "gpt_train_sampler_cache.pt")
        train_sampler = BucketSampler(train_dataset, self.cfg.batch_size, boundaries=[50, 100, 150, 200, 300], shuffle=True, cache_path=sampler_cache_path)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_pose_batch, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.val_batch_size, shuffle=False, collate_fn=collate_pose_batch, num_workers=4, pin_memory=True)
        return train_loader, val_loader

    @torch.no_grad()
    def encode_motion_to_tokens(self, poses, masks):
        _, indices, _ = self.vq_vae.encode(poses, masks)
        return indices

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self.model.text_encoder.eval() # Keep BERT frozen
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg.gpt_num_epochs}")

        for texts, pose_sequences, pose_masks in pbar:
            if texts is None: continue
            
            pose_sequences = pose_sequences.to(self.device)
            pose_masks = pose_masks.to(self.device)
            
            gt_motion_tokens = self.encode_motion_to_tokens(pose_sequences, pose_masks)
            
            B, T_down = gt_motion_tokens.shape
            num_valid_poses = pose_masks.sum(dim=1)
            num_valid_tokens = torch.ceil(num_valid_poses / self.cfg.downsample_rate).long()
            motion_token_mask = torch.arange(T_down, device=self.device).expand(B, T_down) < num_valid_tokens.unsqueeze(1)
            
            sos_token = self.cfg.codebook_size
            sos_tensor = torch.full((B, 1), sos_token, device=self.device, dtype=torch.long)
            
            input_tokens = torch.cat([sos_tensor, gt_motion_tokens[:, :-1]], dim=1)
            target_tokens = gt_motion_tokens
            input_mask = torch.cat([torch.ones_like(sos_tensor, dtype=torch.bool), motion_token_mask[:, :-1]], dim=1)
            
            tokenized_text = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            
            predicted_logits = self.model(tokenized_text, input_tokens, input_mask)
            
            target_tokens[~motion_token_mask] = self.criterion.ignore_index
            loss = self.criterion(predicted_logits.reshape(-1, self.cfg.codebook_size), target_tokens.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            if self.use_wandb:
                wandb.log({'train/gpt_loss': loss.item()}, step=self.global_step)
            self.global_step += 1

    # >>> NEW: Validation method <<<
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")
        for texts, pose_sequences, pose_masks in pbar:
            if texts is None: continue
            pose_sequences, pose_masks = pose_sequences.to(self.device), pose_masks.to(self.device)
            gt_motion_tokens = self.encode_motion_to_tokens(pose_sequences, pose_masks)
            
            B, T_down = gt_motion_tokens.shape
            num_valid_poses = pose_masks.sum(dim=1)
            num_valid_tokens = torch.ceil(num_valid_poses / self.cfg.downsample_rate).long()
            motion_token_mask = torch.arange(T_down, device=self.device).expand(B, T_down) < num_valid_tokens.unsqueeze(1)
            
            sos_token = self.cfg.codebook_size
            sos_tensor = torch.full((B, 1), sos_token, device=self.device, dtype=torch.long)
            
            input_tokens = torch.cat([sos_tensor, gt_motion_tokens[:, :-1]], dim=1)
            target_tokens = gt_motion_tokens
            input_mask = torch.cat([torch.ones_like(sos_tensor, dtype=torch.bool), motion_token_mask[:, :-1]], dim=1)
            
            tokenized_text = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            
            predicted_logits = self.model(tokenized_text, input_tokens, input_mask)
            
            target_tokens[~motion_token_mask] = self.criterion.ignore_index
            loss = self.criterion(predicted_logits.reshape(-1, self.cfg.codebook_size), target_tokens.reshape(-1))
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        if self.use_wandb:
            wandb.log({'val/gpt_loss': avg_loss, 'epoch': epoch})
        return avg_loss

    # >>> NEW: Checkpoint saving method <<<
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        os.makedirs(os.path.dirname(self.cfg.gpt_checkpoint_path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'cfg': self.cfg,
        }
        latest_path = self.cfg.gpt_checkpoint_path.replace('.pth', '_latest.pth')
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, self.cfg.gpt_checkpoint_path)
            print(f"💾 Saved best GPT model to: {self.cfg.gpt_checkpoint_path}")

    def train(self):
        train_loader, val_loader = self.create_dataloaders()
        for epoch in range(self.cfg.gpt_num_epochs):
            self.train_epoch(train_loader, epoch)
            
            # >>> NEW: Call validation and checkpointing <<<
            val_loss = self.validate(val_loader, epoch)
            print(f"\nEpoch {epoch+1} Summary: Validation Loss = {val_loss:.4f}\n")
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best=is_best)
            self.scheduler.step()
        print(f"🎉 GPT Training complete! Best validation loss: {self.best_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train T2M-GPT (Phase 2)')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    args = parser.parse_args()
    cfg = T2M_Config()
    
    if args.wandb:
        wandb.init(project="t2m-gpt-sign-language", name=f"gpt_{datetime.now().strftime('%Y%m%d_%H%M')}", config=vars(cfg))
        
    trainer = GPTTrainer(cfg, args.wandb)
    trainer.train()

if __name__ == "__main__":
    main()