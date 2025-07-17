# inference.py (Corrected)
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
from datetime import datetime

# ✨ THIS LINE IS THE FIX ✨
from config import model_cfg
from motion_dit import MotionDiffusionModel
from modules import get_skeletal_model_structure

# Set up Chinese font display
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False





class PoseVisualizer:
    def __init__(self):
        self.skeletal_triplets = get_skeletal_model_structure()
        body_conns = [(s, e) for s, e, part in self.skeletal_triplets if part == 'body']
        lh_conns = [(s - 8, e - 8) for s, e, part in self.skeletal_triplets if part == 'lh']
        rh_conns = [(s - 29, e - 29) for s, e, part in self.skeletal_triplets if part == 'rh']
        self.conns = {"body": body_conns, "lh": lh_conns, "rh": rh_conns}
        self.colors = {"body": ("red", "blue"), "lh": ("green", "green"), "rh": ("orange", "orange")}

    def _draw_edges(self, ax, pts, conns, p_col, l_col, p_size, l_w):
        valid = np.abs(pts).sum(axis=1) > 1e-6
        if valid.any():
            ax.scatter(pts[valid, 0], -pts[valid, 1], c=p_col, s=p_size, alpha=0.9)
        for a, b in conns:
            if a < len(pts) and b < len(pts) and valid[a] and valid[b]:
                ax.plot([pts[a, 0], pts[b, 0]], [-pts[a, 1], -pts[b, 1]], color=l_col, linewidth=l_w, alpha=0.85)

    def create_animation(self, pose_seq, text, out_path):
        num_frames = pose_seq.shape[0]
        print(f"🎬 Creating animation with {num_frames} frames for text: '{text}'")
        
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Determine plotting range
        all_coords = pose_seq.reshape(num_frames, -1, 3)[:, :, :2]
        valid_coords = all_coords[np.abs(all_coords).sum(axis=2) > 1e-6]
        if len(valid_coords) == 0:
            print("❌ No valid coordinates to visualize."); return
        xmin, xmax = valid_coords[:, 0].min(), valid_coords[:, 0].max()
        ymin, ymax = valid_coords[:, 1].min(), valid_coords[:, 1].max()
        margin = 0.1 * max(xmax - xmin, ymax - ymin, 1)
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(-(ymax + margin), -(ymin - margin))

        body_end = 8 * 3
        lh_end = body_end + 21 * 3

        def _update(i):
            ax.clear()
            ax.set_xlim(xmin - margin, xmax + margin); ax.set_ylim(-(ymax + margin), -(ymin - margin))
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            
            frame_data = pose_seq[i]
            body = frame_data[:body_end].reshape(-1, 3)
            lh = frame_data[body_end:lh_end].reshape(-1, 3)
            rh = frame_data[lh_end:].reshape(-1, 3)

            self._draw_edges(ax, body, self.conns["body"], self.colors["body"][0], self.colors["body"][1], 30, 1.5)
            self._draw_edges(ax, lh, self.conns["lh"], self.colors["lh"][0], self.colors["lh"][1], 15, 1)
            self._draw_edges(ax, rh, self.conns["rh"], self.colors["rh"][0], self.colors["rh"][1], 15, 1)
            ax.set_title(f'"{text}" | Frame {i+1}/{num_frames}')

        ani = animation.FuncAnimation(fig, _update, frames=num_frames, interval=100, blit=False)
        ani.save(out_path, writer="pillow", fps=10)
        plt.close(fig)
        print(f"✅ Animation saved to {out_path}")


def main(args):
    print("📦 Loading model for inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    cfg_from_ckpt = ckpt.get('model_cfg', {})
    if not cfg_from_ckpt:
        print("⚠️ Warning: `model_cfg` not found in checkpoint. Using default config.")
        
    for key, value in cfg_from_ckpt.items():
        if hasattr(model_cfg, key):
            setattr(model_cfg, key, value)
            
    model = MotionDiffusionModel(model_cfg).to(device)
    
    if 'ema_state_dict' in ckpt:
        print("   -> Found EMA weights. Loading for inference.")
        ema_params = ckpt['ema_state_dict']
        model_params = model.noise_predictor.state_dict()
        
        ema_dict = {k: p for k, p in zip([k for k, v in model_params.items() if v.requires_grad], ema_params)}
        model.noise_predictor.load_state_dict(ema_dict, strict=False)
    else:
        print("   -> WARNING: EMA weights not found. Loading standard model weights.")
        model.noise_predictor.load_state_dict(ckpt['model_state_dict'])

    model.eval()

    print(f"\n💬 Generating pose for text: '{args.text}'")
    with torch.no_grad():
        output_seq = model.sample(
            texts=[args.text],
            T=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance
        )
    
    output_seq_np = output_seq.squeeze(0).cpu().numpy()

    visualizer = PoseVisualizer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = "".join(c for c in args.text if c.isalnum() or c in ("_", " ")).replace(" ", "_")[:30]
    out_name = f"output_{safe_text}_{timestamp}.gif"
    out_path = os.path.join("outputs", out_name)
    
    visualizer.create_animation(output_seq_np, args.text, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate sign language pose animation from text.")
    parser.add_argument("--text", type=str, required=True, help="The input text to generate sign language for.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/diffusion_dit_best.pth", help="Path to the trained diffusion model checkpoint.")
    parser.add_argument("--frames", type=int, default=80, help="Number of frames to generate for the animation.")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM inference steps.")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale for Classifier-Free Guidance.")
    
    args = parser.parse_args()
    main(args)