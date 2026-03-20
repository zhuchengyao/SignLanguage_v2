"""Generate GIFs for the 15 overfit cases using the trained checkpoint.

Critical: properly denormalizes model output (normalized space) back to
original coordinate space before visualization.
"""

import os
import json
import numpy as np
import torch
from transformers import BertTokenizer

from src.config_hlc import HLC_NAR_Config
from src.model_unified import HLC_NAR_Model
from src.asl_visualizer import ASLVisualizer
from train_overfit_15cases import CASE_IDS, DATA_ROOT, make_overfit_cfg, OverfitDataset


def build_denorm_map(pose_mean_150: np.ndarray, pose_std_150: np.ndarray):
    """Build (50, 2) mean/std arrays that match pred_xy joint ordering.

    pred_xy layout: [body(0-7), left_hand(8-28), right_hand(29-49)]  each (x,y)
    150-dim layout: body(0:24, 8*3), right_hand(24:87, 21*3), left_hand(87:150, 21*3)
    """
    mean_xy = np.zeros((50, 2), dtype=np.float32)
    std_xy  = np.zeros((50, 2), dtype=np.float32)

    # Body joints 0-7 → 150-dim indices j*3, j*3+1
    for j in range(8):
        mean_xy[j, 0] = pose_mean_150[j * 3]
        mean_xy[j, 1] = pose_mean_150[j * 3 + 1]
        std_xy[j, 0]  = pose_std_150[j * 3]
        std_xy[j, 1]  = pose_std_150[j * 3 + 1]

    # Left hand joints 8-28 → 150-dim indices 87+j*3, 87+j*3+1
    for j in range(21):
        mean_xy[8 + j, 0] = pose_mean_150[87 + j * 3]
        mean_xy[8 + j, 1] = pose_mean_150[87 + j * 3 + 1]
        std_xy[8 + j, 0]  = pose_std_150[87 + j * 3]
        std_xy[8 + j, 1]  = pose_std_150[87 + j * 3 + 1]

    # Right hand joints 29-49 → 150-dim indices 24+j*3, 24+j*3+1
    for j in range(21):
        mean_xy[29 + j, 0] = pose_mean_150[24 + j * 3]
        mean_xy[29 + j, 1] = pose_mean_150[24 + j * 3 + 1]
        std_xy[29 + j, 0]  = pose_std_150[24 + j * 3]
        std_xy[29 + j, 1]  = pose_std_150[24 + j * 3 + 1]

    return mean_xy, std_xy


def denorm_xy(pred_xy: np.ndarray, mean_xy: np.ndarray, std_xy: np.ndarray) -> np.ndarray:
    """Denormalize (T, 50, 2) from normalized space to original coordinates."""
    return pred_xy * std_xy[None, :, :] + mean_xy[None, :, :]


def xy50_to_pose150(xy: np.ndarray) -> np.ndarray:
    """Convert (T, 50, 2) denormalized xy to 150-dim for the visualizer."""
    T = xy.shape[0]
    pose = np.zeros((T, 150), dtype=np.float32)

    body  = xy[:, :8]       # (T, 8, 2)
    left  = xy[:, 8:29]     # (T, 21, 2)
    right = xy[:, 29:50]    # (T, 21, 2)

    for t in range(T):
        for j in range(8):
            pose[t, j * 3]     = body[t, j, 0]
            pose[t, j * 3 + 1] = body[t, j, 1]
            pose[t, j * 3 + 2] = 1.0
        for j in range(21):
            pose[t, 24 + j * 3]     = right[t, j, 0]
            pose[t, 24 + j * 3 + 1] = right[t, j, 1]
            pose[t, 24 + j * 3 + 2] = 1.0
        for j in range(21):
            pose[t, 87 + j * 3]     = left[t, j, 0]
            pose[t, 87 + j * 3 + 1] = left[t, j, 1]
            pose[t, 87 + j * 3 + 2] = 1.0

    return pose


def load_gt_pose150(sid: str) -> np.ndarray:
    """Load ground-truth raw poses for side-by-side comparison."""
    pose_f = os.path.join(DATA_ROOT, sid, "pose.json")
    with open(pose_f) as f:
        js = json.load(f)
    frames = js.get("poses", [])
    seqs = []
    for fr in frames:
        p = (fr.get("pose_keypoints_2d", [])
             + fr.get("hand_right_keypoints_2d", [])
             + fr.get("hand_left_keypoints_2d", []))
        if len(p) == 150:
            seqs.append(p)
    return np.array(seqs, dtype=np.float32)


def get_gt_len(sid: str) -> int:
    pose_f = os.path.join(DATA_ROOT, sid, "pose.json")
    with open(pose_f) as f:
        js = json.load(f)
    return len(js.get("poses", []))


def main():
    cfg = make_overfit_cfg()
    device = cfg.get_device()

    # Load model
    model = HLC_NAR_Model(cfg).to(device)
    ckpt_path = cfg.stage3_ckpt
    if not os.path.exists(ckpt_path):
        ckpt_path = cfg.stage2_ckpt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded: {ckpt_path}  (epoch={ckpt['epoch']}, loss={ckpt['best_loss']:.5f})")

    # Get normalization stats from dataset
    dataset = OverfitDataset(CASE_IDS, DATA_ROOT, max_seq_len=cfg.max_seq_len)
    mean_xy, std_xy = build_denorm_map(dataset.pose_mean, dataset.pose_std)

    tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)
    viz = ASLVisualizer()
    out_dir = "outputs/overfit_15cases"
    os.makedirs(out_dir, exist_ok=True)

    for i, sid in enumerate(CASE_IDS):
        text_f = os.path.join(DATA_ROOT, sid, "text.txt")
        with open(text_f, encoding="utf-8") as f:
            text = f.read().strip()

        tok = tokenizer([text], return_tensors="pt", padding=True,
                        truncation=True, max_length=128)
        tok = {k: v.to(device) for k, v in tok.items()}

        gt_len = get_gt_len(sid)
        result = model.forward_inference(tok, target_len=None)
        pred_len = result["pred_len"]

        if abs(pred_len - gt_len) > gt_len * 0.5:
            print(f"  [{i:2d}] len predictor off (pred={pred_len}, gt={gt_len}), using GT")
            result = model.forward_inference(tok, target_len=gt_len)
            pred_len = gt_len

        pred_xy = result["pred_xy"].cpu().squeeze(0).numpy()  # (T, 50, 2)

        # DENORMALIZE — the critical step
        pred_xy_denorm = denorm_xy(pred_xy, mean_xy, std_xy)
        pose150 = xy50_to_pose150(pred_xy_denorm)

        safe = sid.replace("/", "_").replace(" ", "_")
        gif_path = os.path.join(out_dir, f"{i:02d}_{safe}.gif")
        viz.create_animation(pose150, gif_path,
                             title=f"{text[:50]}", fps=15)

        # Also save GT for comparison
        gt_pose = load_gt_pose150(sid)
        gt_gif = os.path.join(out_dir, f"{i:02d}_{safe}_GT.gif")
        viz.create_animation(gt_pose, gt_gif,
                             title=f"[GT] {text[:45]}", fps=15)

        size_kb = os.path.getsize(gif_path) // 1024
        print(f"  [{i:2d}] pred={pred_len}f(gt={gt_len}f)  {size_kb}KB  {text[:60]}")

    print(f"\nDone. Pred + GT GIFs saved to {out_dir}/")


if __name__ == "__main__":
    main()
