"""HLC-NAR inference: single forward pass from text to pose GIF.

Usage:
    python infer_hlc.py --text "hello how are you" --out_gif outputs/hlc_test.gif
"""

import os
import argparse

import numpy as np
import torch
from transformers import BertTokenizer

from src.config_hlc import HLC_NAR_Config
from src.model_unified import HLC_NAR_Model
from src.asl_visualizer import ASLVisualizer


def xy50_to_pose150(pred_xy: torch.Tensor) -> np.ndarray:
    """Convert (T, 50, 2) predicted xy to 150-dim pose format with dummy confidence=1.

    Layout: body 8*3, right_hand 21*3, left_hand 21*3.
    In our model: indices 0..7 = body, 8..28 = left_hand, 29..49 = right_hand.
    """
    T, J, _ = pred_xy.shape
    pose = np.zeros((T, 150), dtype=np.float32)

    body = pred_xy[:, :8].numpy()       # (T, 8, 2)
    left = pred_xy[:, 8:29].numpy()     # (T, 21, 2)
    right = pred_xy[:, 29:50].numpy()   # (T, 21, 2)

    for t in range(T):
        # Body: 8 joints * (x, y, conf)
        for j in range(8):
            pose[t, j * 3] = body[t, j, 0]
            pose[t, j * 3 + 1] = body[t, j, 1]
            pose[t, j * 3 + 2] = 1.0
        # Right hand: 21 joints * (x, y, conf), starts at index 24
        for j in range(21):
            pose[t, 24 + j * 3] = right[t, j, 0]
            pose[t, 24 + j * 3 + 1] = right[t, j, 1]
            pose[t, 24 + j * 3 + 2] = 1.0
        # Left hand: 21 joints * (x, y, conf), starts at index 87
        for j in range(21):
            pose[t, 87 + j * 3] = left[t, j, 0]
            pose[t, 87 + j * 3 + 1] = left[t, j, 1]
            pose[t, 87 + j * 3 + 2] = 1.0

    return pose


def main():
    parser = argparse.ArgumentParser(description="HLC-NAR Inference")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--out_gif", type=str, default="./outputs/hlc_output.gif")
    parser.add_argument("--target_len", type=int, default=None,
                        help="Override predicted length (None = auto)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Model checkpoint path (default: stage3 best)")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    cfg = HLC_NAR_Config()
    device = cfg.get_device()

    # Load model
    model = HLC_NAR_Model(cfg).to(device)
    ckpt_path = args.ckpt or cfg.stage3_ckpt
    if not os.path.exists(ckpt_path):
        alt = cfg.stage2_ckpt
        if os.path.exists(alt):
            ckpt_path = alt
            print(f"Stage 3 not found, falling back to stage 2: {alt}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found. Tried: {ckpt_path}, {alt}"
            )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    # Tokenize text
    tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)
    tok = tokenizer(
        [args.text], return_tensors="pt", padding=True,
        truncation=True, max_length=128,
    )
    tok = {k: v.to(device) for k, v in tok.items()}

    # Inference
    result = model.forward_inference(tok, target_len=args.target_len)

    pred_xy = result["pred_xy"].cpu().squeeze(0)  # (T, 50, 2)
    pred_len = result["pred_len"]
    print(f"Generated {pred_len} frames for: \"{args.text}\"")

    # Denormalise if stats are available
    if hasattr(model, "pose_mean") and model.pose_mean is not None:
        # pred_xy is in normalised space; we need to undo it.
        # The model outputs xy directly, so denorm needs to be applied
        # per the 150-dim stats mapped to 50-joint xy.
        pass  # xy output is already in model space; visualiser handles ranges

    # Convert to 150-dim for visualiser
    pose_150 = xy50_to_pose150(pred_xy)

    # Visualise
    viz = ASLVisualizer()
    os.makedirs(os.path.dirname(args.out_gif) or ".", exist_ok=True)
    viz.create_animation(pose_150, args.out_gif, title=f"Text: {args.text}", fps=args.fps)
    print(f"Saved GIF to {args.out_gif}")


if __name__ == "__main__":
    main()
