# debug_vae.py (Corrected)
import torch
import os
from config import model_cfg
from motion_vae import GraphTransformerVAE

def test_decoder_variance():
    print("🔬 Testing VAE Decoder for variance...")
    
    # 1. Load model config and weights
    device = torch.device(model_cfg.device)
    model = GraphTransformerVAE(model_cfg).to(device)
    
    vae_weights_path = model_cfg.vae_checkpoint_path
    if not os.path.exists(vae_weights_path):
        print(f"❌ VAE checkpoint not found at {vae_weights_path}. Aborting.")
        return
        
    print(f"   -> Loading weights from {vae_weights_path}")
    model.load_state_dict(torch.load(vae_weights_path, map_location=device))
    model.eval()

    # 2. Generate two different random latent vectors (z)
    z1 = torch.randn(1, model_cfg.vae_latent_dim, device=device)
    z2 = torch.randn(1, model_cfg.vae_latent_dim, device=device)

    # 3. Use the decoder to generate two action sequences
    with torch.no_grad():
        # ✨ FIX: Changed argument name from 'T' to 'target_seq_len'
        output1 = model.decode(z1, target_seq_len=80)
        output2 = model.decode(z2, target_seq_len=80)

    # 4. Calculate the mean absolute difference between the two outputs
    difference = torch.abs(output1 - output2).mean().item()
    print(f"\nMean absolute difference between two random generations: {difference}")

    # 5. Judge the result
    if difference < 1e-4:
        print("🔴 Problem: The decoder produced nearly identical outputs for different inputs. The VAE decoder might have collapsed.")
    else:
        print("🟢 Good: The decoder produced different outputs for different inputs. The VAE is likely working correctly.")

if __name__ == "__main__":
    test_decoder_variance()