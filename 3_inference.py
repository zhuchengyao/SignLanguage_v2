import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import BertTokenizer

from src.config import T2M_Config
from src.model_vqvae import VQ_VAE
from src.model_gpt import T2M_GPT
from src.asl_visualizer import ASLVisualizer

class T2M_Inference:
    def __init__(self, vqvae_path: str, gpt_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # --- Load Config from checkpoint ---
        print(f"Loading config from VQ-VAE checkpoint: {vqvae_path}")
        vqvae_checkpoint = torch.load(vqvae_path, map_location="cpu", weights_only=False)
        self.cfg = vqvae_checkpoint['cfg']
        
        # --- Load VQ-VAE Model ---
        self.vq_vae = VQ_VAE(self.cfg).to(self.device)
        # allow extra buffers/keys for compatibility
        self.vq_vae.load_state_dict(vqvae_checkpoint['model_state_dict'], strict=False)
        self.vq_vae.eval()
        print("VQ-VAE model loaded.")

        # --- Load T2M-GPT Model ---
        print(f"Loading T2M-GPT checkpoint: {gpt_path}")
        # Note: Assuming gpt_checkpoint also contains a state_dict. If it's just the model, adjust accordingly.
        gpt_checkpoint = torch.load(gpt_path, map_location="cpu", weights_only=False)
        self.t2m_gpt = T2M_GPT(self.cfg).to(self.device)
        # Check if the checkpoint is a dictionary with 'model_state_dict'
        if 'model_state_dict' in gpt_checkpoint:
            self.t2m_gpt.load_state_dict(gpt_checkpoint['model_state_dict'], strict=False)
        else:
            self.t2m_gpt.load_state_dict(gpt_checkpoint, strict=False)

        self.t2m_gpt.eval()
        print("T2M-GPT model loaded.")
        
        # --- Initialize Tokenizer and Visualizer ---
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.text_model_name)
        self.visualizer = ASLVisualizer()

    @torch.no_grad()
    def generate(self, text: str, temperature: float = 1.0, top_k: int = 20):
        print(f"Generating animation for text: '{text}'")
        
        # 1. Tokenize the input text
        tokenized_text = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        
        # 2. Autoregressive generation of motion tokens with EOS
        sos_token = self.cfg.codebook_size
        eos_token = self.cfg.codebook_size + 1 if getattr(self.cfg, 'use_eos_token', True) else None
        generated_tokens = torch.full((1, 1), sos_token, device=self.device, dtype=torch.long)
        
        # Max tokens to generate is defined in the config
        max_motion_tokens = self.cfg.vq_sequence_length
        
        print("Generating motion tokens...")
        for _ in tqdm(range(max_motion_tokens - 1)):
            input_tokens = generated_tokens
            input_mask = torch.ones_like(input_tokens, dtype=torch.bool)
            
            logits = self.t2m_gpt(tokenized_text, input_tokens, input_mask)
            next_token_logits = logits[:, -1, :]
            
            next_token_logits /= temperature
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            if eos_token is not None and int(next_token.item()) == eos_token:
                break
            
        # Remove SOS and optional EOS
        motion_indices = generated_tokens[:, 1:]
        if eos_token is not None:
            eos_pos = (motion_indices == eos_token).float().argmax(dim=1).item()
            if eos_pos > 0:
                motion_indices = motion_indices[:, :eos_pos]
        print(f"Generated {motion_indices.shape[1]} motion tokens.")
        
        # 3. Decode motion tokens into pose sequence using VQ-VAE
        print("Decoding tokens into pose sequence...")
        
        # >>> THE FIX: Calculate target length based on generated tokens <<<
        # The number of output poses is the number of tokens * the downsample rate.
        target_pose_len = motion_indices.shape[1] * self.cfg.downsample_rate
        pose_sequence = self.vq_vae.decode(motion_indices, target_length=target_pose_len)
        
        # 4. Denormalize the pose sequence
        if hasattr(self.cfg, 'mean') and self.cfg.mean is not None:
            mean = self.cfg.mean.to(self.device)
            std = self.cfg.std.to(self.device)
            pose_sequence = pose_sequence * std + mean
        
        return pose_sequence.cpu().numpy()[0]
def main():
    parser = argparse.ArgumentParser(description='T2M-GPT Inference')
    parser.add_argument('--vqvae_checkpoint', type=str, default='./checkpoints/vqvae_model.pth', help='Path to the trained VQ-VAE checkpoint.')
    parser.add_argument('--gpt_checkpoint', type=str, default='./checkpoints/t2m_gpt_model.pth', help='Path to the trained T2M-GPT checkpoint.')
    parser.add_argument('--text', type=str, help='Text prompt to generate motion for. If not provided, enters interactive mode.')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save the generated animations.')
    args = parser.parse_args()

    # --- Initialize the Inference Engine ---
    try:
        inference_engine = T2M_Inference(args.vqvae_checkpoint, args.gpt_checkpoint)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please ensure both checkpoint files exist.")
        return

    # --- Generation Loop ---
    def generate_and_save(text_prompt):
        pose_output = inference_engine.generate(text_prompt)
        
        # Sanitize text for filename
        safe_filename = "".join(c for c in text_prompt if c.isalnum() or c in (' ', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')
        output_path = os.path.join(args.output_dir, f"{safe_filename}.gif")
        
        print(f"Saving animation to {output_path}...")
        os.makedirs(args.output_dir, exist_ok=True)
        inference_engine.visualizer.create_animation(
            pose_sequence=pose_output,
            output_path=output_path,
            title=text_prompt
        )
        print(f"Animation saved successfully!")

    if args.text:
        generate_and_save(args.text)
    else:
        # Interactive mode
        print("\n\n--- T2M-GPT Interactive Mode ---")
        print("Enter text to generate a sign language animation. Type 'quit' or 'exit' to close.")
        while True:
            try:
                prompt = input("Enter prompt > ")
                if prompt.lower() in ['quit', 'exit']:
                    break
                if prompt:
                    generate_and_save(prompt)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"An error occurred: {e}")
        print("Exiting interactive mode. Goodbye!")

if __name__ == "__main__":
    main()