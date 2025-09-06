import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import datetime
import wandb
import time

from src.config import T2M_Config
from src.model_vqvae import VQ_VAE
from src.dataloader import ASLPoseDataset, collate_pose_batch, BucketSampler

def codebook_perplexity(indices, codebook_size):
    flat_indices = indices.view(-1); hist = torch.bincount(flat_indices, minlength=codebook_size).float(); probs = hist / hist.sum().clamp(min=1); ppl = torch.exp(-(probs * (probs + 1e-9).log()).sum()); return ppl.item()

def log_memory_usage(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2; reserv = torch.cuda.memory_reserved() / 1024**2
        print(f"[{tag}] GPU Memory: Allocated={alloc:.1f}MB, Reserved={reserv:.1f}MB")

class VQVAETrainer:
    def __init__(self, cfg: T2M_Config, use_wandb: bool, resume: bool):
        self.cfg = cfg; self.device = cfg.get_device(); self.use_wandb = use_wandb
        self.model = VQ_VAE(cfg).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.vqvae_learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.vqvae_num_epochs, eta_min=1e-6)
        self.global_step = 0; self.best_loss = float('inf'); self.start_epoch = 0
        print(f"VQ-VAE Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}"); print(f"Device: {self.device}")
        if resume:
            self.load_checkpoint()

    def create_dataloaders(self):
        train_dataset = ASLPoseDataset(data_paths=[os.path.join(self.cfg.data_root, "ASL_gloss/train")], split="train")
        val_dataset = ASLPoseDataset(data_paths=[os.path.join(self.cfg.data_root, "ASL_gloss/dev")], split="dev", extern_mean=train_dataset.pose_mean, extern_std=train_dataset.pose_std)
        self.cfg.mean = torch.from_numpy(train_dataset.pose_mean).float(); self.cfg.std = torch.from_numpy(train_dataset.pose_std).float()
        sampler_cache_path = os.path.join(self.cfg.data_root, "ASL_gloss/.cache", "train_sampler_cache.pt")
        train_sampler = BucketSampler(train_dataset, self.cfg.batch_size, boundaries=[50, 100, 150, 200, 300], shuffle=True, cache_path=sampler_cache_path)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_pose_batch,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            collate_fn=collate_pose_batch,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )
        print(f"Train Samples: {len(train_dataset)}, Val Samples: {len(val_dataset)}")
        return train_loader, val_loader

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        os.makedirs(os.path.dirname(self.cfg.vqvae_checkpoint_path), exist_ok=True); checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'val_loss': val_loss, 'global_step': self.global_step, 'cfg': self.cfg}; latest_path = self.cfg.vqvae_checkpoint_path.replace('.pth', '_latest.pth'); torch.save(checkpoint, latest_path)
        if is_best: torch.save(checkpoint, self.cfg.vqvae_checkpoint_path); print(f"Saved best model to: {self.cfg.vqvae_checkpoint_path}")

    def load_checkpoint(self):
        latest_path = self.cfg.vqvae_checkpoint_path.replace('.pth', '_latest.pth')
        if not os.path.exists(latest_path):
            print(f"No checkpoint found at {latest_path}. Starting from scratch."); return
        print(f"Resuming training from checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False); self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']); self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1; self.global_step = checkpoint['global_step']; self.best_loss = checkpoint['val_loss']
        print(f"Resumed from epoch {self.start_epoch}. Best loss so far: {self.best_loss:.4f}")

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg.vqvae_num_epochs}")
        for batch in pbar:
            if batch[0] is None: continue
            _, pose_sequences, masks = batch
            pose_sequences, masks = pose_sequences.to(self.device), masks.to(self.device)
            outputs = self.model(pose_sequences, masks)
            recon_loss, vq_loss = outputs['recon_loss'], outputs['vq_loss']
            cur_vq_w = self.cfg.vq_loss_weight * min(1.0, (epoch + 1) / self.cfg.vq_weight_warmup_epochs)
            total_loss = self.cfg.recon_loss_weight * recon_loss + cur_vq_w * vq_loss
            self.optimizer.zero_grad(set_to_none=True); total_loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0); self.optimizer.step()
            # report extra metrics
            # codebook perplexity (token usage diversity)
            try:
                ppl = codebook_perplexity(outputs['indices'], self.cfg.codebook_size)
            except Exception:
                ppl = 0.0
            postfix = {
                'Loss': f'{total_loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'wRecon': f"{outputs.get('recon_weighted', torch.tensor(0)).item():.4f}",
                'Bone': f"{outputs.get('bone_length_loss', torch.tensor(0)).item():.4f}",
                'Vel': f"{outputs.get('velocity_loss', torch.tensor(0)).item():.4f}",
                'PPL': f"{ppl:.1f}"
            }
            pbar.set_postfix(postfix)
            if self.use_wandb:
                wandb.log({
                    'train/total_loss': total_loss.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/vq_loss': vq_loss.item(),
                    'train/codebook_perplexity': ppl
                }, step=self.global_step)
            self.global_step += 1

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_recon_loss, total_vq_loss = 0, 0
        total_bone, total_vel, total_wrec = 0, 0, 0
        total_ppl, count_ppl = 0.0, 0
        for batch in tqdm(val_loader, desc="Validating"):
            if batch[0] is None: continue
            _, pose_sequences, masks = batch
            pose_sequences, masks = pose_sequences.to(self.device), masks.to(self.device)
            outputs = self.model(pose_sequences, masks)
            total_recon_loss += outputs['recon_loss'].item()
            total_vq_loss += outputs['vq_loss'].item()
            total_wrec += outputs.get('recon_weighted', torch.tensor(0)).item()
            total_bone += outputs.get('bone_length_loss', torch.tensor(0)).item()
            total_vel  += outputs.get('velocity_loss', torch.tensor(0)).item()
            try:
                total_ppl += codebook_perplexity(outputs['indices'], self.cfg.codebook_size)
                count_ppl += 1
            except Exception:
                pass
        avg_recon = total_recon_loss / len(val_loader); avg_vq = total_vq_loss / len(val_loader)
        avg_wrec = total_wrec / len(val_loader); avg_bone = total_bone / len(val_loader); avg_vel = total_vel / len(val_loader)
        total_loss = self.cfg.recon_loss_weight * avg_recon + self.cfg.vq_loss_weight * avg_vq
        avg_ppl = (total_ppl / max(count_ppl, 1))
        return total_loss, avg_recon, avg_vq, avg_wrec, avg_bone, avg_vel, avg_ppl

    def train_session(self, target_end_epoch: int):
        train_loader, val_loader = self.create_dataloaders()
        print(f"Starting VQ-VAE training session from epoch {self.start_epoch + 1}...")
        for epoch in range(self.start_epoch, target_end_epoch):
            self.train_epoch(train_loader, epoch)
            val_loss, val_recon, val_vq, val_wrec, val_bone, val_vel, val_ppl = self.validate(val_loader)
            self.scheduler.step()
            print(f"\nEpoch {epoch+1} Summary:"); log_memory_usage(f"End of Epoch {epoch+1}"); print(f"  Validation Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, VQ: {val_vq:.4f}, wRecon: {val_wrec:.4f}, Bone: {val_bone:.4f}, Vel: {val_vel:.4f}, PPL: {val_ppl:.1f})")
            if self.use_wandb:
                wandb.log({
                    'val/total_loss': val_loss,
                    'val/recon_loss': val_recon,
                    'val/recon_weighted': val_wrec,
                    'val/bone_length_loss': val_bone,
                    'val/velocity_loss': val_vel,
                    'val/vq_loss': val_vq,
                    'val/codebook_perplexity': val_ppl,
                    'epoch': epoch
                })
            is_best = val_loss < self.best_loss
            if is_best: self.best_loss = val_loss
            self.save_checkpoint(epoch, val_loss, is_best)
        print(f"\nSession complete up to epoch {target_end_epoch}!")

def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE for T2M-GPT in Sessions')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--session_epochs', type=int, default=3, help='Number of epochs per training session')
    args = parser.parse_args()
    
    cfg = T2M_Config()
    
    if args.wandb:
        wandb.init(project="t2m-gpt-sign-language", config=vars(cfg), name=f"vqvae_session_run_{datetime.now().strftime('%Y%m%d_%H%M')}", resume="allow", reinit=True)
    
    # >>> NEW LOGIC: Check for existing checkpoint before starting the main loop <<<
    start_epoch = 0
    latest_checkpoint_path = cfg.vqvae_checkpoint_path.replace('.pth', '_latest.pth')
    if os.path.exists(latest_checkpoint_path):
        print(f"Found existing checkpoint: {latest_checkpoint_path}")
        # We only load the checkpoint to find out which epoch to start from.
        # The trainer instance inside the loop will do the full loading.
        checkpoint = torch.load(latest_checkpoint_path, map_location='cpu', weights_only=False)
        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"Last completed epoch was {start_epoch - 1}. New sessions will start from here.")

    current_epoch = start_epoch
    while current_epoch < cfg.vqvae_num_epochs:
        # Determine if this session needs to load a checkpoint.
        # It needs to load if a checkpoint existed at the start OR if this is not the first session.
        is_resuming = (start_epoch > 0) or (current_epoch > 0)
        
        # Calculate the end epoch for this session
        session_target_epoch = min(current_epoch + args.session_epochs, cfg.vqvae_num_epochs)
        
        print(f"\n{'='*60}")
        print(f"Starting Training Session: To run from epoch {current_epoch + 1} -> {session_target_epoch}")
        print(f"{'='*60}\n")
        
        # Create a new trainer instance to simulate a script restart
        trainer = VQVAETrainer(cfg, args.wandb, resume=is_resuming)
        
        # The trainer will load its own start_epoch from the checkpoint, which is the source of truth
        current_epoch = trainer.start_epoch
        session_target_epoch = min(current_epoch + args.session_epochs, cfg.vqvae_num_epochs)

        # A safety check to prevent running an empty session
        if current_epoch >= session_target_epoch:
             print(f"All epochs up to {current_epoch} already trained. Nothing to do in this session.")
             current_epoch = session_target_epoch
             continue

        try:
            trainer.train_session(target_end_epoch=session_target_epoch)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            break
        
        current_epoch = session_target_epoch
        
        if current_epoch < cfg.vqvae_num_epochs:
            print(f"\nSimulating script restart... preparing next session.")
            time.sleep(2)

    print("\nAll training sessions complete!")
    if args.wandb and wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()