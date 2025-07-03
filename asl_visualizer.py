#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASL 统一可视化工具（适配可变长 Text-to-Pose Diffusion）
功能 1️⃣  可视化真实数据集中的 ASL 样例
功能 2️⃣  使用 checkpoints/best.pth 推理生成 ASL 动画并可视化
"""

import os, json
from datetime import datetime
from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ───────── 模型 & 配置 ─────────
from model import TextToPoseDiffusion
from config import ModelConfig

# —— 中文字体 ——（若无 SimHei 可自行替换）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ASLVisualizer:
    """统一可视化工具（兼容可变长 Spatio-Temporal Diffusion 模型）"""

    BODY_JOINTS = 8
    HAND_JOINTS = 21                   # 左 / 右各 21
    POSE_DIM   = BODY_JOINTS * 3 + HAND_JOINTS * 3 * 2  # 150

    def __init__(self, checkpoint_path: str = "./checkpoints/best.pth"):
        self.ckpt_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model:      TextToPoseDiffusion | None = None
        self.model_cfg:  ModelConfig          | None = None

        # —— skeleton topology（与训练一致）——
        self.pose_conn = [
            (0, 1),
            (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
        ]
        self.hand_conn = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        print("🚀  ASL Visualizer ready  | device:", self.device)

    # ===============================================================
    #  功能 1️⃣  可视化真实数据
    # ===============================================================
    @staticmethod
    def _load_real_sample(folder: str):
        txt_f  = os.path.join(folder, "text.txt")
        pose_f = os.path.join(folder, "pose.json")
        if not (os.path.exists(txt_f) and os.path.exists(pose_f)):
            raise FileNotFoundError("❌ text.txt 或 pose.json 缺失")
        text  = open(txt_f,  "r", encoding="utf-8").read().strip()
        poses = json.load(open(pose_f, "r", encoding="utf-8"))["poses"]
        return text, poses

    def visualize_real_sample(self, folder: str,
                              max_frames: int = 64,
                              interval: int = 250):
        text, poses = self._load_real_sample(folder)
        poses = poses[:max_frames]
        self._create_animation(
            poses, text,
            out_prefix="real",
            interval=interval,
            tag="Real Data"
        )

    # ===============================================================
    #  功能 2️⃣  推理并可视化
    # ===============================================================
    def _load_model(self):
        if self.model is not None:
            return self.model

        print("📦  Loading model …")
        ckpt = None
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            print(f"🔑  weights loaded from {self.ckpt_path}")

        # ── ModelConfig ──
        if ckpt and "model_cfg" in ckpt:
            self.model_cfg = ModelConfig(**ckpt["model_cfg"])
            print("📝  ModelConfig from checkpoint")
        else:
            self.model_cfg = ModelConfig()
            print("📝  default ModelConfig()")

        # ── build & load weights ──
        self.model = TextToPoseDiffusion(self.model_cfg).to(self.device)
        if ckpt:
            state_dict = ckpt.get("model_state_dict", ckpt)
            miss, unexp = self.model.load_state_dict(state_dict, strict=False)
            if miss:   print("⚠️  missing:",   miss)
            if unexp:  print("⚠️  unexpected:", unexp)
        self.model.eval()
        print("✅  model ready")
        return self.model

    @torch.no_grad()
    def infer_and_visualize(self, text: str,
                            frames: int = 50,
                            steps: int  = 20,
                            interval: int = 100):
        """
        frames : 生成的帧长 T
        steps  : diffusion 采样步数 (<=50 合理)
        """
        model = self._load_model()
        print(f"🤖  Sampling '{text}' | frames={frames}, steps={steps} …")
        seq = model.sample([text], T=frames, num_steps=steps)     # (1,T,150)
        seq = seq.squeeze(0).cpu().numpy()                        # (T,150)

        if seq.shape[1] != self.POSE_DIM:
            raise ValueError(f"pose_dim {seq.shape[1]} != 150")

        body_end = self.BODY_JOINTS * 3
        lh_end   = body_end + self.HAND_JOINTS * 3
        body = seq[:, :body_end]
        lh   = seq[:, body_end:lh_end]
        rh   = seq[:, lh_end:]

        poses: List[dict] = []
        for b, l, r in zip(body, lh, rh):
            poses.append({
                "pose_keypoints_2d":       b.tolist(),
                "hand_left_keypoints_2d":  l.tolist(),
                "hand_right_keypoints_2d": r.tolist(),
            })

        self._create_animation(
            poses, text,
            out_prefix="infer",
            interval=interval,
            tag="Inference"
        )

    # ===============================================================
    #  绘制 / 动画
    # ===============================================================
    @staticmethod
    def _np_points(kpts):        # -> (N,3)
        return np.asarray(kpts).reshape(-1, 3)

    def _draw_edges(self, ax, pts, conns,
                    p_col, l_col, p_size, l_w):
        valid = (np.abs(pts[:, 0]) > 1e-6) | (np.abs(pts[:, 1]) > 1e-6)
        if valid.any():
            ax.scatter(pts[valid, 0], -pts[valid, 1],
                       c=p_col, s=p_size, alpha=0.85)
        for a, b in conns:
            if a < len(pts) and b < len(pts) and valid[a] and valid[b]:
                ax.plot([pts[a, 0], pts[b, 0]],
                        [-pts[a, 1], -pts[b, 1]],
                        color=l_col, linewidth=l_w, alpha=0.8)

    def _create_animation(self, poses, text,
                          out_prefix, interval, tag):
        print(f"🎬  {tag}  | {len(poses)} frames")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_txt  = "".join(c for c in text if c.isalnum() or c in ("_", " ")).replace(" ", "_")
        out_path  = f"{out_prefix}_{safe_txt}_{timestamp}.gif"

        # —— figure —— 
        fig, ax = plt.subplots(figsize=(7, 7))

        # —— view box —— 
        coords = []
        for fr in poses:
            for key in ("pose_keypoints_2d",
                        "hand_left_keypoints_2d",
                        "hand_right_keypoints_2d"):
                coords.append(self._np_points(fr[key])[:, :2])
        coords = np.concatenate(coords, 0)
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        margin = 0.1 * max(xmax - xmin, ymax - ymin, 1)
        xlim = (xmin - margin, xmax + margin)
        ylim = (-(ymax + margin), -(ymin - margin))

        def _update(i):
            ax.clear()
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            fr = poses[i]
            body = self._np_points(fr["pose_keypoints_2d"])
            lh   = self._np_points(fr["hand_left_keypoints_2d"])
            rh   = self._np_points(fr["hand_right_keypoints_2d"])

            self._draw_edges(ax, body, self.pose_conn,
                             "red",   "blue",   40, 2)
            self._draw_edges(ax, lh,   self.hand_conn,
                             "green", "green", 20, 1.5)
            self._draw_edges(ax, rh,   self.hand_conn,
                             "orange","orange",20, 1.5)
            ax.set_title(f'{tag}: "{text}"  |  Frame {i+1}/{len(poses)}')

        ani = animation.FuncAnimation(
            fig, _update,
            frames=len(poses),
            interval=interval, blit=False
        )
        ani.save(out_path, writer="pillow", fps=1000/interval)
        print(f"✅  saved → {out_path}")
        plt.show()

# ===============================================================
#  CLI
# ===============================================================
def main():
    vis = ASLVisualizer()
    while True:
        print("\n=== 选择功能 ===\n1) 可视化真实数据\n2) 推理生成\n3) 退出")
        choice = input("请输入 (1/2/3): ").strip()
        if choice == "1":
            folder = input("数据文件夹路径: ").strip()
            if os.path.isdir(folder):
                vis.visualize_real_sample(folder)
            else:
                print("❌  路径不存在")
        elif choice == "2":
            text = input("输入要生成的 ASL 词语: ").strip()
            if text:
                try:
                    T = int(input("生成帧长 (默认 50): ") or 50)
                except ValueError:
                    T = 50
                vis.infer_and_visualize(text, frames=T)
            else:
                print("❌  请输入有效文本")
        elif choice == "3":
            print("👋  Bye!"); break
        else:
            print("❌  无效选择")

if __name__ == "__main__":
    main()
